//! The stock EXL3 checkpoint repacked as an EAQ container: read the source shards, drop every
//! tensor the XPU reader cannot serve (the visual tower, the MTP draft), copy the rest
//! payload-for-payload into rewritten shards, and derive the manifest, the merged config.json
//! and the completion marker from the bytes just written — the marker last, so an interrupted
//! run never looks complete. Which names survive mirrors zig/src/xpu/exl3_eaq.zig's slot
//! table: a tensor with no place there refuses the repack here, rather than opening later as a
//! pack the Table would refuse. eaq_fixture.zig is the oracle for every manifest field shape;
//! bits live per part, so one pack mixes 1..4-bit tensors freely.
//!
//! Shards keep their source file names. Every member of one trellis bundle (the .trellis codes
//! with their suh/svh or su/sv scales, the .mul1 scalar and an optional .bias) moves into the
//! shard that holds the bundle's .trellis tensor: the container pins an entry's parts to its
//! entry's single shard, and the quantizer may leave a bundle split across two files.
//!
//! config.json is merged, not replaced — the runner still reads the pack's own
//! quantization_config from it. The eaq block goes wherever the container's configBlock reads
//! one: the top level when free, else text_config; it errors rather than clobbering a block
//! that names another method.
//!
//! A tensor is an entry when its name (after shedding the VLM's `language_model` segment and
//! a trailing `.weight`) lands on the reader's table; bundle suffixes group onto the base
//! entry. Tensors under `model.visual.` (no slot bucket) and `mtp.` (the draft, no Table
//! place) are dropped with their whole bundles; anything else with no place refuses.
const std = @import("std");
const Io = std.Io;
const Sha256 = std.crypto.hash.sha2.Sha256;
const eaq = @import("eaq_container.zig");
const exl3_format = @import("exl3_format.zig");
const safetensors = @import("safetensors.zig");

pub const family = "qwen3_5"; // the reader's family (zig/src/xpu/exl3_eaq.zig)
pub const pack_version = "eaq.pack/0";
pub const dictionary = "exl3.mul1.v1"; // the codebook the stock pack stores (config.json's "codebook")

/// A shard of the pack proper: an HF `model*.safetensors` file. A directory's other
/// .safetensors (a cal_trace.safetensors, say) are quantizer artifacts, not weights.
pub fn shardFile(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "model") and std.mem.endsWith(u8, name, ".safetensors");
}

pub const Error = error{
    UnmappedTensor, // a kept tensor the reader's slot table has no place for
    BadBundle, // an incomplete or inconsistent trellis bundle
    DuplicateTensor, // one tensor name in two source shards
    DuplicateEntry, // two kept tensors claiming one entry name
    BadConfig, // no room for the eaq block without clobbering another method's block
};

pub const Result = struct {
    shards: usize = 0,
    entries: usize = 0,
    n_trellis: usize = 0,
    n_dense: usize = 0,
    kept_tensors: usize = 0,
    dropped_tensors: usize = 0, // the visual tower and the MTP draft, gone from shard and manifest
    mul1_values: usize = 0, // distinct .mul1 payloads; one says the pack is a single mul1 dictionary
    manifest_sha256: [64]u8 = undefined,
};

// —— the reader's slot table, mirrored (zig/src/xpu/exl3_eaq.zig) —————————————————————
pub const layers = 64; // mirrors qwen27.zig LAYERS

pub fn isAttention(i: usize) bool {
    return (i + 1) % 4 == 0;
}

const Slot = enum { mixer_in, mixer_out, gu, down };
const Part = struct { layer: usize, slot: Slot, part: u8 };
const Place = union(enum) { head, embed, other, projection: Part };

fn slotParts(i: usize) struct { mixer_in: []const []const u8, mixer_out: []const []const u8 } {
    return if (isAttention(i))
        .{ .mixer_in = &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" }, .mixer_out = &.{"self_attn.o_proj"} }
    else
        .{ .mixer_in = &.{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b" }, .mixer_out = &.{"linear_attn.out_proj"} };
}

/// An entry name's place in the table; null outside it — the same rules as the reader's
/// slotFor, so what passes here is what map() later maps.
fn placeOf(name_in: []const u8) ?Place {
    const name = if (std.mem.endsWith(u8, name_in, ".weight")) name_in[0 .. name_in.len - ".weight".len] else name_in;
    if (std.mem.eql(u8, name, "lm_head")) return .head;
    if (std.mem.eql(u8, name, "model.embed_tokens")) return .embed;
    if (std.mem.eql(u8, name, "model.norm")) return .other;
    if (!std.mem.startsWith(u8, name, "model.layers.")) return null;
    const rest = name["model.layers.".len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const digits = rest[0..dot];
    if (digits.len == 0 or digits[0] < '0' or digits[0] > '9') return null; // no sign, no empty
    const layer = std.fmt.parseInt(usize, digits, 10) catch return null;
    if (layer >= layers) return null;
    const tail = rest[dot + 1 ..];
    const slots = slotParts(layer);
    for (slots.mixer_in, 0..) |p, pi| if (std.mem.eql(u8, tail, p)) return .{ .projection = .{ .layer = layer, .slot = .mixer_in, .part = @intCast(pi) } };
    for (slots.mixer_out, 0..) |p, pi| if (std.mem.eql(u8, tail, p)) return .{ .projection = .{ .layer = layer, .slot = .mixer_out, .part = @intCast(pi) } };
    if (std.mem.eql(u8, tail, "mlp.gate_proj")) return .{ .projection = .{ .layer = layer, .slot = .gu, .part = 0 } };
    if (std.mem.eql(u8, tail, "mlp.up_proj")) return .{ .projection = .{ .layer = layer, .slot = .gu, .part = 1 } };
    if (std.mem.eql(u8, tail, "mlp.down_proj")) return .{ .projection = .{ .layer = layer, .slot = .down, .part = 0 } };
    const known: []const []const u8 = if (isAttention(layer))
        &.{ "self_attn.q_norm", "self_attn.k_norm" }
    else
        &.{ "linear_attn.conv1d", "linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.norm" };
    for (known) |k| if (std.mem.eql(u8, tail, k)) return .other;
    if (std.mem.eql(u8, tail, "input_layernorm") or std.mem.eql(u8, tail, "post_attention_layernorm")) return .other;
    return null;
}

const Kind = enum { trellis, suh, svh, su, sv, mul1, bias };
const suffixes = [_]struct { []const u8, Kind }{
    .{ ".trellis", .trellis }, .{ ".suh", .suh }, .{ ".svh", .svh }, .{ ".su", .su },
    .{ ".sv", .sv }, .{ ".mul1", .mul1 }, .{ ".bias", .bias },
};

fn suffixOf(name: []const u8) ?struct { []const u8, Kind } {
    for (suffixes) |s| if (std.mem.endsWith(u8, name, s[0])) return s;
    return null;
}

/// The entry name a tensor's bytes serve: the VLM's `language_model` segment sheds (the
/// reader's table wears no wrapper) and a trailing `.weight` goes; the tensor keeps its own
/// name inside the part.
fn entryName(a: std.mem.Allocator, tensor: []const u8) ![]const u8 {
    var n: []const u8 = tensor;
    if (std.mem.startsWith(u8, n, "model.language_model."))
        n = try std.fmt.allocPrint(a, "model.{s}", .{n["model.language_model.".len..]});
    if (std.mem.endsWith(u8, n, ".weight")) n = n[0 .. n.len - ".weight".len];
    return n;
}

const Tensor = struct {
    name: []const u8,
    shard: usize,
    dtype: safetensors.DType,
    rank: u8,
    shape: [safetensors.max_rank]usize,
    begin: usize,
    end: usize,
    kept: bool = false,
    member: bool = false, // a trellis bundle's part; its entry is `bundle`
    bundle: usize = 0,
    kind: Kind = .trellis,
};

const Bundle = struct {
    base: []const u8,
    members: [7]?usize = @splat(null), // by @intFromEnum(Kind)
    owner: usize = 0, // the shard holding its .trellis; every member lives there in the repack
    k: usize = 0,
    n: usize = 0,
    halves: u8 = 0,
};

const Entry = struct {
    name: []const u8,
    trellis: bool,
    owner: usize, // the source shard that owns the entry's bytes
    tensor: usize = 0, // the dense entry's single tensor
    bundle: usize = 0, // the trellis entry's bundle
    shard_file: []const u8 = "", // filled after the shards are written
};

const Src = struct { file: Io.File, data: u64, meta: std.json.Value = .null };
const ShardRec = struct { src: usize, file: []const u8, bytes: u64, sha: [32]u8 };

pub fn repack(gpa: std.mem.Allocator, io: Io, src_dir: []const u8, dst_dir: []const u8, why: ?*eaq.Why) !Result {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    // An old marker would vouch for a pack this run has not finished publishing.
    Io.Dir.cwd().deleteFile(io, try std.fs.path.join(a, &.{ dst_dir, eaq.marker_name })) catch {};

    // 1. the source's shards, header by header; payloads stay unread until the copy
    var files: std.ArrayList([]const u8) = .empty;
    {
        var d = try Io.Dir.cwd().openDir(io, src_dir, .{ .iterate = true });
        defer d.close(io);
        var it = d.iterate();
        while (try it.next(io)) |e| if (shardFile(e.name)) try files.append(a, try a.dupe(u8, e.name));
        std.mem.sort([]const u8, files.items, {}, struct {
            fn less(_: void, x: []const u8, y: []const u8) bool {
                return std.mem.order(u8, x, y) == .lt;
            }
        }.less);
    }
    if (files.items.len == 0) return error.FileNotFound;

    const srcs = try a.alloc(Src, files.items.len);
    var opened: usize = 0;
    defer for (srcs[0..opened]) |*s| s.file.close(io);
    var tensors: std.ArrayList(Tensor) = .empty;
    var seen: std.StringHashMapUnmanaged(usize) = .empty;
    for (files.items, srcs, 0..) |name, *s, si| {
        s.* = .{ .file = try Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ src_dir, name }), .{}), .data = undefined };
        opened += 1;
        const len = try s.file.length(io);
        var prefix: [8]u8 = undefined;
        if (try s.file.readPositionalAll(io, &prefix, 0) != 8) return error.BadSource;
        const hl = try safetensors.headerSize(std.mem.readInt(u64, &prefix, .little), len);
        const json = try a.alloc(u8, @intCast(hl));
        if (try s.file.readPositionalAll(io, json, 8) != json.len) return error.BadSource;
        const header = try safetensors.parseHeader(a, json, std.math.cast(usize, len - 8 - hl) orelse return error.BadSource);
        s.data = 8 + hl;
        const root = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return error.BadSource;
        if (root == .object) s.meta = root.object.get("__metadata__") orelse .null;
        for (header.keys(), header.values()) |tname, e| {
            const gop = try seen.getOrPut(a, tname);
            if (gop.found_existing) {
                fail(why, "duplicate tensor {s}", .{tname});
                return error.DuplicateTensor;
            }
            gop.value_ptr.* = tensors.items.len;
            try tensors.append(a, .{ .name = tname, .shard = si, .dtype = e.dtype, .rank = e.rank, .shape = e.shape, .begin = e.begin, .end = e.end });
        }
    }

    // 2. classify: the dropped towers, the trellis bundles, the dense tensors
    var bundles: std.ArrayList(Bundle) = .empty;
    var bundle_ids: std.StringHashMapUnmanaged(usize) = .empty;
    var dropped: usize = 0;
    for (tensors.items, 0..) |*t, i| {
        if (std.mem.startsWith(u8, t.name, "model.visual.") or std.mem.startsWith(u8, t.name, "mtp.")) {
            dropped += 1;
            continue;
        }
        if (suffixOf(t.name)) |suf| {
            const base = t.name[0 .. t.name.len - suf[0].len];
            const at = try bundle_ids.getOrPut(a, base);
            if (!at.found_existing) {
                at.key_ptr.* = try a.dupe(u8, base);
                at.value_ptr.* = bundles.items.len;
                try bundles.append(a, .{ .base = at.key_ptr.* });
            }
            const b = &bundles.items[at.value_ptr.*];
            if (b.members[@intFromEnum(suf[1])]) |_| {
                fail(why, "{s}", .{t.name});
                return error.DuplicateTensor; // one bundle, two tensors of a kind
            }
            b.members[@intFromEnum(suf[1])] = i;
            t.* = .{ .name = t.name, .shard = t.shard, .dtype = t.dtype, .rank = t.rank, .shape = t.shape, .begin = t.begin, .end = t.end, .kept = true, .member = true, .bundle = at.value_ptr.*, .kind = suf[1] };
            continue;
        }
        if (placeOf(try entryName(a, t.name)) == null) {
            fail(why, "{s}", .{t.name});
            return error.UnmappedTensor;
        }
        t.kept = true;
    }

    // 3. entries: validated bundles first, then dense tensors, one name each
    var entries: std.ArrayList(Entry) = .empty;
    var names: std.StringHashMapUnmanaged(void) = .empty;
    for (bundles.items, 0..) |*b, bi| {
        const ename = try entryName(a, b.base);
        if (placeOf(ename) == null) {
            fail(why, "{s}", .{b.base});
            return error.UnmappedTensor;
        }
        const trellis = b.members[@intFromEnum(Kind.trellis)] orelse {
            fail(why, "{s}.trellis missing", .{b.base});
            return error.BadBundle;
        };
        const t = tensors.items[trellis];
        if (t.dtype != .i16 or t.rank != 3 or t.shape[0] == 0 or t.shape[1] == 0 or t.shape[2] % 8 != 0) {
            fail(why, "{s}.trellis is not I16 [K/16, N/16, halves*8]", .{b.base});
            return error.BadBundle;
        }
        b.k = t.shape[0] * 16;
        b.n = t.shape[1] * 16;
        b.halves = @intCast(t.shape[2] / 8);
        if (b.halves < 2 or b.halves > 16) { // the reader's Quant range; odd halves ride with mul1 alone
            fail(why, "{s}: halves {d} outside 2..16", .{ b.base, b.halves });
            return error.BadBundle;
        }
        b.owner = t.shard;
        const suh = b.members[@intFromEnum(Kind.suh)];
        const svh = b.members[@intFromEnum(Kind.svh)];
        const su = b.members[@intFromEnum(Kind.su)];
        const sv = b.members[@intFromEnum(Kind.sv)];
        if ((suh == null) != (svh == null) or (su == null) != (sv == null) or ((suh != null) and (su != null))) {
            fail(why, "{s}: needs both suh and svh (F16) or both su and sv (I16)", .{b.base});
            return error.BadBundle;
        }
        if (suh) |x| { // per-weight [K],[N] or per-16 [K/16],[N/16]; a mix refuses, as the reader's does
            const per_weight = rank1Of(tensors.items[x], .f16, b.k) and rank1Of(tensors.items[svh.?], .f16, b.n);
            const per_16 = rank1Of(tensors.items[x], .f16, b.k / 16) and rank1Of(tensors.items[svh.?], .f16, b.n / 16);
            if (!per_weight and !per_16) {
                fail(why, "{s}: scale shapes not [{d}] and [{d}] (or per-16)", .{ b.base, b.k, b.n });
                return error.BadBundle;
            }
        } else if (!(rank1Of(tensors.items[su.?], .i16, b.k / 16) and rank1Of(tensors.items[sv.?], .i16, b.n / 16))) {
            fail(why, "{s}: packed su/sv not I16 [{d}] and [{d}]", .{ b.base, b.k / 16, b.n / 16 });
            return error.BadBundle;
        }
        if (b.members[@intFromEnum(Kind.mul1)]) |x| {
            if (tensors.items[x].dtype != .i32 or tensors.items[x].rank != 0) {
                fail(why, "{s}.mul1 is not an I32 scalar", .{b.base});
                return error.BadBundle;
            }
        }
        if (b.members[@intFromEnum(Kind.bias)]) |x| {
            if (tensors.items[x].dtype != .f16) {
                fail(why, "{s}.bias is not F16", .{b.base});
                return error.BadBundle;
            }
        }
        if ((try names.getOrPut(a, ename)).found_existing) {
            fail(why, "{s}", .{b.base});
            return error.DuplicateEntry;
        }
        try entries.append(a, .{ .name = ename, .trellis = true, .owner = b.owner, .bundle = @intCast(bi) });
    }
    // a base may also exist as a plain tensor: that entry name is already taken
    for (tensors.items, 0..) |t, i| {
        if (!t.kept or t.member) continue;
        const ename = try entryName(a, t.name);
        if ((try names.getOrPut(a, ename)).found_existing) {
            fail(why, "{s}", .{t.name});
            return error.DuplicateEntry;
        }
        try entries.append(a, .{ .name = ename, .trellis = false, .owner = t.shard, .tensor = i });
    }
    var n_trellis: usize = 0;
    for (entries.items) |e| n_trellis += @intFromBool(e.trellis);

    // 4. output shard lists: tensors in source order; a split bundle's members move as one
    const out_lists = try a.alloc(std.ArrayList(usize), files.items.len);
    for (out_lists) |*l| l.* = .empty;
    const order = try a.alloc(usize, tensors.items.len);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, tensors.items, struct {
        fn lt(all: []const Tensor, x: usize, y: usize) bool {
            if (all[x].shard != all[y].shard) return all[x].shard < all[y].shard;
            return all[x].begin < all[y].begin;
        }
    }.lt);
    for (order) |i| {
        const t = tensors.items[i];
        if (!t.kept) continue;
        const owner = if (t.member) bundles.items[t.bundle].owner else t.shard;
        if (owner == t.shard) try out_lists[owner].append(a, i);
    }
    // the bundles keep their discovery order: entries and tensors index them by position
    for (bundles.items) |*b| for (b.members) |m| {
        const x = m orelse continue;
        if (tensors.items[x].shard != b.owner) try out_lists[b.owner].append(a, x);
    };

    // 5. write the shards, hashing every byte written; track the distinct .mul1 payloads
    const buf = try a.alloc(u8, 4 << 20);
    var recs: std.ArrayList(ShardRec) = .empty;
    var mul1: std.StringHashMapUnmanaged(void) = .empty;
    var kept: usize = 0;
    for (files.items, out_lists, 0..) |name, list, si| {
        if (list.items.len == 0) continue;
        try recs.append(a, try writeShard(a, io, dst_dir, si, name, srcs, list.items, tensors.items, buf, &mul1));
        kept += list.items.len;
    }
    if (mul1.count() > 1) {
        fail(why, "the pack holds {d} distinct .mul1 payloads; one mul1 dictionary cannot cover them", .{mul1.count()});
        return error.MultiMul1;
    }
    const file_of = try a.alloc(?usize, files.items.len);
    @memset(file_of, null);
    for (recs.items, 0..) |rec, ri| file_of[rec.src] = ri;
    for (entries.items) |*e| {
        const ri = file_of[e.owner] orelse {
            fail(why, "{s}: its shard held no kept tensor", .{e.name});
            return error.BadBundle;
        };
        e.shard_file = recs.items[ri].file;
    }

    // 6. the manifest, then config.json (which hashes it), then the marker last
    std.mem.sort(Entry, entries.items, {}, struct {
        fn less(_: void, x: Entry, y: Entry) bool {
            return std.mem.order(u8, x.name, y.name) == .lt;
        }
    }.less);
    const manifest = try manifestOf(a, entries.items, bundles.items, tensors.items, recs.items);
    const digest = sha(manifest);
    const hex = std.fmt.bytesToHex(digest, .lower);

    try mergeConfig(a, io, src_dir, dst_dir, &hex, why);
    try publish(a, io, dst_dir, eaq.manifest_name, manifest);
    try publish(a, io, dst_dir, eaq.marker_name, try std.fmt.allocPrint(
        a,
        \\{{"manifest_sha256":"{s}","eaq_layout":"{s}","n_shards":{d},"n_entries":{d}}}
        \\
    , .{ &hex, eaq.layout, recs.items.len, entries.items.len }));

    return .{
        .shards = recs.items.len,
        .entries = entries.items.len,
        .n_trellis = n_trellis,
        .n_dense = entries.items.len - n_trellis,
        .kept_tensors = kept,
        .dropped_tensors = dropped,
        .mul1_values = mul1.count(),
        .manifest_sha256 = hex,
    };
}

fn rank1Of(t: Tensor, dtype: safetensors.DType, len: usize) bool {
    return t.dtype == dtype and t.rank == 1 and t.shape[0] == len;
}

fn sha(data: []const u8) [32]u8 {
    var d: [32]u8 = undefined;
    Sha256.hash(data, &d, .{});
    return d;
}

/// The refusal note for whoever is holding the Why, mirroring the reader's fail().
fn fail(why: ?*eaq.Why, comptime fmt: []const u8, args: anytype) void {
    const w = why orelse return;
    w.len = if (std.fmt.bufPrint(&w.buf, fmt, args)) |t| t.len else |_| blk: {
        const note = "(refusal message too long)";
        @memcpy(w.buf[0..note.len], note);
        break :blk note.len;
    };
}

/// One rewritten shard: the source's kept tensors in order at contiguous offsets, its
/// __metadata__ kept as it was, moved bundle members last; hashed while writing.
fn writeShard(
    a: std.mem.Allocator,
    io: Io,
    dst_dir: []const u8,
    src_index: usize,
    file_name: []const u8,
    srcs: []const Src,
    list: []const usize,
    tensors: []const Tensor,
    buf: []u8,
    mul1: *std.StringHashMapUnmanaged(void),
) !ShardRec {
    var header: std.json.ObjectMap = .empty;
    if (srcs[src_index].meta == .object) try header.put(a, "__metadata__", srcs[src_index].meta);
    var offset: usize = 0;
    for (list) |i| {
        const t = tensors[i];
        var dims = std.json.Array.init(a);
        for (t.shape[0..t.rank]) |d| try dims.append(.{ .integer = @intCast(d) });
        var offs = std.json.Array.init(a);
        try offs.appendSlice(&.{ .{ .integer = @intCast(offset) }, .{ .integer = @intCast(offset + t.end - t.begin) } });
        var o: std.json.ObjectMap = .empty;
        try o.put(a, "dtype", .{ .string = dtypeName(t.dtype) });
        try o.put(a, "shape", .{ .array = dims });
        try o.put(a, "data_offsets", .{ .array = offs });
        try header.put(a, t.name, .{ .object = o });
        offset += t.end - t.begin;
    }
    const text = try std.json.Stringify.valueAlloc(a, std.json.Value{ .object = header }, .{});
    const padded = std.mem.alignForward(usize, text.len, 8);
    var head: [8]u8 = undefined;
    std.mem.writeInt(u64, &head, padded, .little);
    const spaces = try a.alloc(u8, padded - text.len);
    @memset(spaces, ' ');

    var atomic = try Io.Dir.cwd().createFileAtomic(io, try std.fs.path.join(a, &.{ dst_dir, file_name }), .{ .make_path = true, .replace = true });
    defer atomic.deinit(io);
    var h = Sha256.init(.{});
    var out: u64 = 0;
    for ([_][]const u8{ &head, text, spaces }) |piece| {
        try atomic.file.writePositionalAll(io, piece, out);
        h.update(piece);
        out += piece.len;
    }
    for (list) |i| {
        const t = tensors[i];
        const src = srcs[t.shard];
        const len = t.end - t.begin;
        var done: usize = 0;
        while (done < len) {
            const n: usize = @min(buf.len, len - done);
            if (try src.file.readPositionalAll(io, buf[0..n], src.data + t.begin + done) != n) return error.InputOutput;
            h.update(buf[0..n]);
            try atomic.file.writePositionalAll(io, buf[0..n], out + done);
            done += n;
        }
        if (std.mem.endsWith(u8, t.name, ".mul1")) {
            const put = try mul1.getOrPut(a, buf[0..4]);
            if (!put.found_existing) put.key_ptr.* = try a.dupe(u8, buf[0..4]);
        }
        out += len;
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    try atomic.replace(io);
    return .{ .src = src_index, .file = file_name, .bytes = out, .sha = d };
}

/// The container's manifest over the entries just written (the fixture's shape): inline
/// entries, the trellis params carrying each part's own bits, K and N.
fn manifestOf(
    a: std.mem.Allocator,
    entries: []const Entry,
    bundles: []const Bundle,
    tensors: []const Tensor,
    recs: []const ShardRec,
) ![]const u8 {
    var table: [32]u8 = undefined;
    eaq.codebookSha(.mul1, &table);
    var dict: std.json.ObjectMap = .empty;
    const table_hex = std.fmt.bytesToHex(table, .lower);
    try dict.put(a, "table_sha256", .{ .string = &table_hex });
    var dicts: std.json.ObjectMap = .empty;
    try dicts.put(a, dictionary, .{ .object = dict });

    var shards: std.json.Array = std.json.Array.init(a);
    for (recs) |r| {
        var s: std.json.ObjectMap = .empty;
        try s.put(a, "file", .{ .string = r.file });
        try s.put(a, "bytes", .{ .integer = @intCast(r.bytes) });
        const sha_hex: [64]u8 = std.fmt.bytesToHex(r.sha, .lower);
        try s.put(a, "sha256", .{ .string = try a.dupe(u8, &sha_hex) });
        try shards.append(.{ .object = s });
    }

    var list: std.json.Array = std.json.Array.init(a);
    for (entries) |e| {
        var entry: std.json.ObjectMap = .empty;
        try entry.put(a, "name", .{ .string = e.name });
        try entry.put(a, "format", .{ .string = if (e.trellis) "eaq.trellis/0" else "eaq.dense/0" });
        var params: std.json.ObjectMap = .empty;
        var parts: std.json.ObjectMap = .empty;
        if (e.trellis) {
            const b = bundles[e.bundle];
            try params.put(a, "bits", .{ .float = @as(f64, @floatFromInt(b.halves)) / 2.0 });
            try params.put(a, "K", .{ .integer = @intCast(b.k) });
            try params.put(a, "N", .{ .integer = @intCast(b.n) });
            try params.put(a, "dictionary", .{ .string = dictionary });
            const order = [_]Kind{ .trellis, .suh, .svh, .su, .sv, .mul1, .bias };
            for (order) |k| if (b.members[@intFromEnum(k)]) |ti| {
                const t = tensors[ti];
                try parts.put(a, @tagName(k), try partOf(a, t.name, t.dtype, t.shape[0..t.rank]));
            };
        } else {
            const t = tensors[e.tensor];
            try parts.put(a, "data", try partOf(a, t.name, t.dtype, t.shape[0..t.rank]));
        }
        try entry.put(a, "params", .{ .object = params });
        try entry.put(a, "shard", .{ .string = e.shard_file });
        try entry.put(a, "parts", .{ .object = parts });
        try list.append(.{ .object = entry });
    }

    var adapter: std.json.ObjectMap = .empty;
    try adapter.put(a, "family", .{ .string = family });
    var model: std.json.ObjectMap = .empty;
    try model.put(a, "adapter", .{ .object = adapter });

    var root: std.json.ObjectMap = .empty;
    try root.put(a, "eaq_layout", .{ .string = eaq.layout });
    try root.put(a, "eaq_pack", .{ .string = pack_version });
    try root.put(a, "model", .{ .object = model });
    const features: std.json.Array = std.json.Array.init(a);
    try root.put(a, "required_features", .{ .array = features });
    try root.put(a, "dictionaries", .{ .object = dicts });
    try root.put(a, "n_entries", .{ .integer = @intCast(entries.len) });
    var n_trellis: usize = 0;
    for (entries) |e| n_trellis += @intFromBool(e.trellis);
    try root.put(a, "n_trellis", .{ .integer = @intCast(n_trellis) });
    try root.put(a, "n_dense", .{ .integer = @intCast(entries.len - n_trellis) });
    try root.put(a, "shards", .{ .array = shards });
    try root.put(a, "entries_inline", .{ .bool = true });
    try root.put(a, "entries", .{ .array = list });
    return std.json.Stringify.valueAlloc(a, std.json.Value{ .object = root }, .{});
}

fn partOf(a: std.mem.Allocator, tensor: []const u8, dtype: safetensors.DType, shape: []const usize) !std.json.Value {
    var dims = std.json.Array.init(a);
    for (shape) |d| try dims.append(.{ .integer = @intCast(d) });
    var o: std.json.ObjectMap = .empty;
    try o.put(a, "tensor", .{ .string = tensor });
    try o.put(a, "dtype", .{ .string = dtypeName(dtype) });
    try o.put(a, "shape", .{ .array = dims });
    return .{ .object = o };
}

/// The eaq block into the pack's existing config.json: the top level when free (or already
/// eaq's), else text_config; it never touches a block that names another method, and refuses
/// when both blocks are taken. The exl3 block the runner reads survives verbatim.
fn mergeConfig(a: std.mem.Allocator, io: Io, src_dir: []const u8, dst_dir: []const u8, manifest_sha: []const u8, why: ?*eaq.Why) !void {
    var root: std.json.Value = .{ .object = .empty };
    const path = try std.fs.path.join(a, &.{ src_dir, "config.json" });
    if (Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 26))) |text| {
        root = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch {
            fail(why, "config.json is not JSON", .{});
            return error.BadConfig;
        };
    } else |e| if (e != error.FileNotFound) return e;
    if (root != .object) {
        fail(why, "config.json is not an object", .{});
        return error.BadConfig;
    }

    var block: std.json.ObjectMap = .empty;
    try block.put(a, "quant_method", .{ .string = eaq.quant_method });
    try block.put(a, "manifest", .{ .string = eaq.manifest_name });
    try block.put(a, "manifest_sha256", .{ .string = manifest_sha });

    const eaq_block: std.json.Value = .{ .object = block };
    const root_obj = &root.object;
    const methodIsEaq = struct {
        fn f(v: ?std.json.Value) bool {
            const x = v orelse return false;
            if (x != .object) return false;
            const m = x.object.get("quant_method") orelse return false;
            return m == .string and std.ascii.eqlIgnoreCase(m.string, eaq.quant_method);
        }
    }.f;

    if (methodIsEaq(root_obj.get("quantization_config")) or root_obj.get("quantization_config") == null) {
        try root_obj.put(a, "quantization_config", eaq_block); // the top level is free, or already ours
    } else if (root_obj.getPtr("text_config")) |tc| {
        if (tc.* != .object) {
            fail(why, "config.json text_config is not an object", .{});
            return error.BadConfig;
        }
        const q = tc.object.getPtr("quantization_config");
        if (q == null or methodIsEaq(if (q) |p| p.* else null)) {
            try tc.object.put(a, "quantization_config", eaq_block);
        } else {
            fail(why, "config.json quantization_config is taken at both levels", .{});
            return error.BadConfig;
        }
    } else {
        var tc: std.json.ObjectMap = .empty;
        try tc.put(a, "quantization_config", eaq_block);
        try root_obj.put(a, "text_config", .{ .object = tc });
    }
    const out = try std.json.Stringify.valueAlloc(a, root, .{ .whitespace = .indent_2 });
    try publish(a, io, dst_dir, "config.json", try std.mem.concat(a, u8, &.{ out, "\n" }));
}

fn publish(a: std.mem.Allocator, io: Io, dir: []const u8, name: []const u8, data: []const u8) !void {
    var atomic = try Io.Dir.cwd().createFileAtomic(io, try std.fs.path.join(a, &.{ dir, name }), .{ .make_path = true, .replace = true });
    defer atomic.deinit(io);
    try atomic.file.writePositionalAll(io, data, 0);
    try atomic.replace(io);
}

/// A dtype as a safetensors header names it: its tag in capitals (bf16 as BF16).
fn dtypeName(d: safetensors.DType) []const u8 {
    return switch (d) {
        inline else => |tag| {
            const name = comptime blk: {
                var up: [@tagName(tag).len]u8 = undefined;
                _ = std.ascii.upperString(&up, @tagName(tag));
                const out = up;
                break :blk &out;
            };
            return name;
        },
    };
}

/// A source tensor for the synthetic pack below.
pub const Spec = struct { name: []const u8, dtype: []const u8, shape: []const u8, bytes: usize };

/// A safetensors image over `specs`: payload bytes are each tensor's own i%251 count, so
/// every .mul1 scalar carries the same four bytes and a copied range can be checked for free.
pub fn image(a: std.mem.Allocator, specs: []const Spec) ![]u8 {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(a);
    var at: usize = 0;
    try header.append(a, '{');
    for (specs, 0..) |s, i| {
        try header.print(a, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":{s},\"data_offsets\":[{d},{d}]}}", .{ if (i > 0) "," else "", s.name, s.dtype, s.shape, at, at + s.bytes });
        at += s.bytes;
    }
    try header.append(a, '}');
    const out = try a.alloc(u8, 8 + header.items.len + at);
    std.mem.writeInt(u64, out[0..8], header.items.len, .little);
    @memcpy(out[8..][0..header.items.len], header.items);
    var off: usize = 0;
    for (specs) |s| {
        for (out[8 + header.items.len + off ..][0..s.bytes], 0..) |*b, i| b.* = @intCast(i % 251);
        off += s.bytes;
    }
    return out;
}

fn addT(a: std.mem.Allocator, l: *std.ArrayList(Spec), name: []const u8, dtype: []const u8, shape: []const u8, bytes: usize) !void {
    try l.append(a, .{ .name = name, .dtype = dtype, .shape = shape, .bytes = bytes });
}

/// A trellis bundle: the I16 codes [kt, nt, halves*8] with per-weight F16 scales and a uniform
/// I32 .mul1 scalar — the shape an EXL3 quantizer leaves behind, at 64-wide, 64-tall tiles.
fn addBundle(a: std.mem.Allocator, l: *std.ArrayList(Spec), base: []const u8, kt: usize, nt: usize, halves: usize) !void {
    try addT(a, l, try std.fmt.allocPrint(a, "{s}.trellis", .{base}), "I16", try std.fmt.allocPrint(a, "[{d},{d},{d}]", .{ kt, nt, halves * 8 }), 2 * kt * nt * halves * 8);
    try addT(a, l, try std.fmt.allocPrint(a, "{s}.suh", .{base}), "F16", try std.fmt.allocPrint(a, "[{d}]", .{kt * 16}), 2 * kt * 16);
    try addT(a, l, try std.fmt.allocPrint(a, "{s}.svh", .{base}), "F16", try std.fmt.allocPrint(a, "[{d}]", .{nt * 16}), 2 * nt * 16);
    try addT(a, l, try std.fmt.allocPrint(a, "{s}.mul1", .{base}), "I32", "[]", 4);
}

pub const SourceOpts = struct {
    /// write a model.visual.* tower (repack must drop it whole)
    visual: bool = true,
    /// write mtp.* draft tensors (repack must drop them whole)
    mtp: bool = true,
    /// write a cal_trace.safetensors (repack must not even read it)
    cal: bool = true,
    /// write a stray.weight no slot table knows (repack must refuse it)
    stray: bool = false,
    /// the halves of layer 3's v_proj — a fused mixer_in whose parts disagree on bits
    v_halves: ?u8 = null,
};

/// A complete Qwen3.8-27B EXL3 source directory at fixture scale: every one of the 64 layers,
/// the embedding, the head, the norms, GDN gates and convs, split into two shards with the
/// head's scales left behind in the first — the split the repacker must heal. Tensor names
/// wear the VLM's `language_model` segment exactly as the stock pack does.
pub fn writeSynthetic(gpa: std.mem.Allocator, io: Io, dir: Io.Dir, o: SourceOpts) !void {
    var s1: std.ArrayList(Spec) = .empty;
    var s2: std.ArrayList(Spec) = .empty;
    const L = "model.language_model.layers.";
    const attn = [_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj" };
    const gdn = [_][]const u8{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj" };
    const mlp = [_][]const u8{ "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" };
    for (0..layers) |l| {
        const t: *std.ArrayList(Spec) = if (l < 32) &s1 else &s2;
        if (isAttention(l)) {
            for (attn) |s| {
                const h: usize = if (o.v_halves != null and std.mem.eql(u8, s, "self_attn.v_proj") and l == 3) o.v_halves.? else 4;
                try addBundle(gpa, t, try std.fmt.allocPrint(gpa, "{s}{d}.{s}", .{ L, l, s }), 4, 4, h);
            }
            const anorms = [_][]const u8{ "self_attn.q_norm.weight", "self_attn.k_norm.weight" };
            for (anorms) |s|
                try addT(gpa, t, try std.fmt.allocPrint(gpa, "{s}{d}.{s}", .{ L, l, s }), "BF16", "[8]", 16);
        } else {
            for (gdn) |s| try addBundle(gpa, t, try std.fmt.allocPrint(gpa, "{s}{d}.{s}", .{ L, l, s }), 4, 4, 4);
            const ab = [_][]const u8{ "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight" };
            for (ab) |s|
                try addT(gpa, t, try std.fmt.allocPrint(gpa, "{s}{d}.{s}", .{ L, l, s }), "F16", "[2,8]", 32);
            try addT(gpa, t, try std.fmt.allocPrint(gpa, "{s}{d}.linear_attn.conv1d.weight", .{ L, l }), "BF16", "[8,1,4]", 64);
            try addT(gpa, t, try std.fmt.allocPrint(gpa, "{s}{d}.linear_attn.A_log", .{ L, l }), "BF16", "[2]", 4);
            try addT(gpa, t, try std.fmt.allocPrint(gpa, "{s}{d}.linear_attn.dt_bias", .{ L, l }), "BF16", "[2]", 4);
            try addT(gpa, t, try std.fmt.allocPrint(gpa, "{s}{d}.linear_attn.norm.weight", .{ L, l }), "BF16", "[8]", 16);
        }
        for (mlp) |s| try addBundle(gpa, t, try std.fmt.allocPrint(gpa, "{s}{d}.{s}", .{ L, l, s }), 4, 4, 4);
        const lnorms = [_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight" };
        for (lnorms) |s|
            try addT(gpa, t, try std.fmt.allocPrint(gpa, "{s}{d}.{s}", .{ L, l, s }), "BF16", "[8]", 16);
    }
    try addT(gpa, &s1, "model.language_model.embed_tokens.weight", "BF16", "[16,8]", 256);
    try addT(gpa, &s1, "model.language_model.norm.weight", "BF16", "[8]", 16);
    try addT(gpa, &s2, "lm_head.trellis", "I16", "[4,16,32]", 2 * 4 * 16 * 32); // K=64, N=256
    // the head's scales stay in the first shard: only a rewrite re-unites the bundle
    try addT(gpa, &s1, "lm_head.suh", "F16", "[64]", 128);
    try addT(gpa, &s1, "lm_head.svh", "F16", "[256]", 512);
    try addT(gpa, &s1, "lm_head.mul1", "I32", "[]", 4);
    if (o.visual) {
        try addBundle(gpa, &s1, "model.visual.blocks.0.attn.q_proj", 3, 3, 6);
        try addT(gpa, &s1, "model.visual.blocks.0.attn.q_proj.bias", "F16", "[1152]", 2304);
        try addT(gpa, &s1, "model.visual.patch_embed.proj.weight", "BF16", "[2,2]", 8);
        try addT(gpa, &s1, "model.visual.pos_embed.weight", "BF16", "[4]", 8);
    }
    if (o.mtp) {
        try addBundle(gpa, &s1, "mtp.fc", 4, 4, 4);
        try addT(gpa, &s1, "mtp.norm.weight", "BF16", "[8]", 16);
    }
    if (o.stray) try addT(gpa, &s2, "stray.weight", "BF16", "[2]", 4);

    try dir.writeFile(io, .{ .sub_path = "model-00001-of-00002.safetensors", .data = try image(gpa, s1.items) });
    try dir.writeFile(io, .{ .sub_path = "model-00002-of-00002.safetensors", .data = try image(gpa, s2.items) });
    if (o.cal) {
        const cal = [_]Spec{.{ .name = "cal.stats", .dtype = "F32", .shape = "[4]", .bytes = 16 }};
        try dir.writeFile(io, .{ .sub_path = "cal_trace.safetensors", .data = try image(gpa, &cal) });
    }
    try dir.writeFile(io, .{ .sub_path = "config.json", .data =
        \\{"model_type":"qwen3_5","quantization_config":{"quant_method":"exl3","bits":2.0,"codebook":"mul1"},
        \\"text_config":{"hidden_size":64,"vocab_size":248320}}
    });
}

fn tmpPath(gpa: std.mem.Allocator, sub: []const u8) ![]const u8 {
    return std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{sub});
}

const fixture = struct {
    pub const shards = 2;
    pub const entries = 851; // 401 trellis entries (392 layer slots + head + the two projections GDN serves dense), 450 dense
    pub const n_trellis = 401;
    pub const n_dense = 450;
    pub const kept = 2054; // 4 members on each trellis entry, one data part on each dense one
    pub const dropped = 12; // the 7-tensor visual tower and the 5-tensor MTP stub
};

test "the repacker publishes a pack the container opens, payload for payload" {
    var ar = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ar.deinit();
    const gpa = ar.allocator();
    const io = std.testing.io;
    var src = std.testing.tmpDir(.{});
    defer src.cleanup();
    var dst = std.testing.tmpDir(.{});
    defer dst.cleanup();
    try writeSynthetic(gpa, io, src.dir, .{});
    const sp = try tmpPath(gpa, std.mem.sliceTo(&src.sub_path, 0));
    const dp = try tmpPath(gpa, std.mem.sliceTo(&dst.sub_path, 0));

    var why: eaq.Why = .{};
    const got = try repack(gpa, io, sp, dp, &why);

    try std.testing.expectEqual(fixture.shards, got.shards);
    try std.testing.expectEqual(fixture.entries, got.entries);
    try std.testing.expectEqual(fixture.n_trellis, got.n_trellis);
    try std.testing.expectEqual(fixture.n_dense, got.n_dense);
    try std.testing.expectEqual(fixture.kept, got.kept_tensors);
    try std.testing.expectEqual(fixture.dropped, got.dropped_tensors);
    try std.testing.expectEqual(@as(usize, 1), got.mul1_values); // one multiplier across the pack: one dictionary

    var pack = try eaq.open(gpa, io, dp, .{ .family = family, .why = &why });
    defer pack.deinit();
    try std.testing.expectEqual(fixture.entries, pack.entries.len);
    try std.testing.expectEqual(fixture.kept, pack.n_tensors);
    try std.testing.expectEqual(eaq.Format.trellis, pack.entry("model.layers.3.self_attn.q_proj").?.format);
    try std.testing.expectEqual(eaq.Format.dense, pack.entry("model.layers.0.linear_attn.in_proj_a").?.format);
    try std.testing.expectEqual(eaq.Format.dense, pack.entry("model.embed_tokens").?.format);
    try std.testing.expectEqual(eaq.Format.trellis, pack.entry("lm_head").?.format);
    try std.testing.expectEqualStrings("model-00002-of-00002.safetensors", pack.entry("lm_head").?.shard);
    try std.testing.expect(pack.entry("model.visual.blocks.0.attn.q_proj") == null);
    try std.testing.expect(pack.entry("mtp.fc") == null);

    // the moved head bundle lives whole in the .trellis's shard, bytes as the source held them
    var f2 = try safetensors.File.open(gpa, io, try std.fs.path.join(gpa, &.{ dp, "model-00002-of-00002.safetensors" }));
    defer f2.close(io);
    for ([_][]const u8{ "lm_head.trellis", "lm_head.suh", "lm_head.svh", "lm_head.mul1" }) |n2| // the whole head bundle, healed from its split
        try std.testing.expect(f2.names.get(n2) != null);
    const head = f2.get("lm_head.trellis").?;
    try std.testing.expect(head.is(.i16, &.{ 4, 16, 32 }));
    for (head.bytes, 0..) |b, i| try std.testing.expectEqual(@as(u8, @intCast(i % 251)), b);
    var f1 = try safetensors.File.open(gpa, io, try std.fs.path.join(gpa, &.{ dp, "model-00001-of-00002.safetensors" }));
    defer f1.close(io);
    var s1 = try safetensors.File.open(gpa, io, try std.fs.path.join(gpa, &.{ sp, "model-00001-of-00002.safetensors" }));
    defer s1.close(io);
    try std.testing.expectEqualStrings(s1.get("model.language_model.embed_tokens.weight").?.bytes, f1.get("model.language_model.embed_tokens.weight").?.bytes);
    // the dropped towers left the shards themselves, not just the manifest
    try std.testing.expect(f1.names.get("model.visual.pos_embed.weight") == null and f1.names.get("mtp.norm.weight") == null and f1.names.get("lm_head.trellis") == null);
}

test "a tensor the slot table does not know refuses the repack and names it" {
    var ar = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ar.deinit();
    const gpa = ar.allocator();
    const io = std.testing.io;
    var src = std.testing.tmpDir(.{});
    defer src.cleanup();
    var dst = std.testing.tmpDir(.{});
    defer dst.cleanup();
    try writeSynthetic(gpa, io, src.dir, .{ .stray = true });
    var why: eaq.Why = .{};
    try std.testing.expectError(error.UnmappedTensor, repack(gpa, io, try tmpPath(gpa, std.mem.sliceTo(&src.sub_path, 0)), try tmpPath(gpa, std.mem.sliceTo(&dst.sub_path, 0)), &why));
    try std.testing.expectEqualStrings("stray.weight", why.text());
}

test "config.json is merged: the exl3 block stays, the eaq block lands in text_config" {
    var ar = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ar.deinit();
    const gpa = ar.allocator();
    const io = std.testing.io;
    var src = std.testing.tmpDir(.{});
    defer src.cleanup();
    var dst = std.testing.tmpDir(.{});
    defer dst.cleanup();
    try writeSynthetic(gpa, io, src.dir, .{});
    const sp = try tmpPath(gpa, std.mem.sliceTo(&src.sub_path, 0));
    const dp = try tmpPath(gpa, std.mem.sliceTo(&dst.sub_path, 0));
    const got = try repack(gpa, io, sp, dp, null);

    const cfg_text = try Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(gpa, &.{ dp, "config.json" }), gpa, .limited(1 << 20));
    const cfg = try std.json.parseFromSliceLeaky(std.json.Value, gpa, cfg_text, .{});
    const root_q = cfg.object.get("quantization_config").?;
    try std.testing.expectEqualStrings("exl3", root_q.object.get("quant_method").?.string); // the runner's block, verbatim
    const text_q = cfg.object.get("text_config").?.object.get("quantization_config").?;
    try std.testing.expectEqualStrings("eaq", text_q.object.get("quant_method").?.string);
    try std.testing.expectEqualStrings(eaq.manifest_name, text_q.object.get("manifest").?.string);
    try std.testing.expectEqualStrings(&got.manifest_sha256, text_q.object.get("manifest_sha256").?.string);
    try std.testing.expectEqual(@as(i64, 64), cfg.object.get("text_config").?.object.get("hidden_size").?.integer); // untouched fields survive
    const raw = try Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(gpa, &.{ dp, eaq.manifest_name }), gpa, .limited(1 << 26));
    try std.testing.expectEqualStrings(&got.manifest_sha256, &std.fmt.bytesToHex(sha(raw), .lower));
}

test "bits live per part: a mixer_in whose v_proj was quantized wider still repacks" {
    var ar = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ar.deinit();
    const gpa = ar.allocator();
    const io = std.testing.io;
    var src = std.testing.tmpDir(.{});
    defer src.cleanup();
    var dst = std.testing.tmpDir(.{});
    defer dst.cleanup();
    try writeSynthetic(gpa, io, src.dir, .{ .v_halves = 8 });
    const got = try repack(gpa, io, try tmpPath(gpa, std.mem.sliceTo(&src.sub_path, 0)), try tmpPath(gpa, std.mem.sliceTo(&dst.sub_path, 0)), null);
    try std.testing.expectEqual(fixture.entries, got.entries);
    var pack = try eaq.open(gpa, io, try tmpPath(gpa, std.mem.sliceTo(&dst.sub_path, 0)), .{ .family = family });
    defer pack.deinit();
    const bits = struct {
        fn f(v: ?std.json.Value) f64 {
            return switch (v.?) {
                .integer => |i| @floatFromInt(i),
                .float => |x| x,
                else => unreachable,
            };
        }
    }.f;
    const q = pack.entry("model.layers.3.self_attn.q_proj").?;
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), bits(q.params.get("bits")), 0.0000001);
    const v = pack.entry("model.layers.3.self_attn.v_proj").?;
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), bits(v.params.get("bits")), 0.0000001);
}

