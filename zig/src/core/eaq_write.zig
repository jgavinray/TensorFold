//! The fresh EAQ quant writer: turns the encoder's in-memory payloads — the source's bf16/f16
//! safetensors entries and the new trellis I16 parts, each carrying its own halves count — into
//! a complete EAQ pack: rewritten shards with recomputed offsets and hashes, a manifest whose
//! per-trellis-part params say bits (= halves/2, a float: odd halves are mul1's half steps), K
//! and N (from the trellis part's shape) and the dictionary (exl3.mul1.v1), a merged config.json
//! (the eaq block at the top level when free, else in text_config), and EAQ_COMPLETE.json last.
//!
//! The payloads are staged as source safetensors shards and the pack is published by
//! core/eaq_repack.zig itself: neither the repacker nor the container is modified or duplicated
//! here. Everything after the stage rides the repacker's path — its bundle checks (I16
//! [K/16, N/16, halves*8] codes, suh/svh or su/sv scales as a pair, the I32 scalar .mul1, one
//! mul1 payload across the whole pack), its slot table refusing names the reader cannot serve,
//! the hashed shard rewrite, the manifest, the config merge and the marker last, so an
//! interrupted run never looks complete.
//!
//! Bundle members (the .trellis with its scales, its .mul1 and an optional .bias) are staged
//! into one shard together, so the repacker never sees a split bundle — a bundle that cannot
//! be served never reaches staging either: the trellis tiles, the scale lengths against its
//! K and N, the mul1 codebook and the bias dtype are measured on the payloads first, and odd
//! halves refuse unless allow_odd_bits says the device loader admits the half-steps. Payload
//! bytes travel verbatim from `Input.payload` to the pack's shard — the tests below check
//! every one.
//! The driver's encoding proof binds into the chain the same way: Options.report is gated
//! (parse, contract, encoder, zero failed tiles, no identity fallback, size) and embedded in
//! every shard's __metadata__ beside the sha256 of its exact bytes — which the repacker
//! preserves, so the shard and manifest hashes cover the pack's own vouching.
//!
//! Like zig/src/xpu/exl3_eaq.zig this file is a peer of the core module, not a member: it
//! imports `core` and, for its tests only, the `exl3_eaq` reader module. A standalone test run
//! wires both:
//!
//!   zig test --name eaq-write -O ReleaseSafe -lc \
//!     --dep core --dep exl3_eaq -Mroot=zig/src/core/eaq_write.zig \
//!     --dep tokenizer -Mcore=zig/src/core/root.zig \
//!     --dep core -Mexl3_eaq=zig/src/xpu/exl3_eaq.zig \
//!     -Mtokenizer=zig/src/core/tokenizer/tokenizer.zig
//!
//! from the repo root (the tests live under .zig-cache/tmp).

const std = @import("std");
const Io = std.Io;
const core = @import("core");
const eaq = core.eaq_container;
const repack = core.eaq_repack;
const safetensors = core.safetensors;

pub const family = repack.family;
pub const pack_version = repack.pack_version;
pub const dictionary = repack.dictionary;

/// What the repacker reports: shard and entry counts and the manifest's sha256 in hex.
pub const Result = repack.Result;
pub const DType = safetensors.DType;

/// One source tensor as the quantizer hands it over: the name it wears inside the pack, the
/// dtype and shape the manifest will declare, and the payload bytes to write verbatim.
pub const Input = struct {
    name: []const u8,
    dtype: DType,
    shape: []const usize,
    payload: []const u8,

    /// The dtype's byte count times the shape; null on overflow.
    fn need(self: *const Input) ?usize {
        var n: usize = self.dtype.size();
        for (self.shape) |d| n = std.math.mul(usize, n, d) catch return null;
        return n;
    }
};

pub const Options = struct {
    /// The source's config.json, merged into the pack's: the eaq block (quant_method, manifest,
    /// manifest_sha256) lands at the top level when free, else in text_config, and every other
    /// field survives verbatim (core/eaq_repack.zig's mergeConfig). null: the pack's config.json
    /// holds only the eaq block.
    config: ?[]const u8 = null,
    /// The safetensors shards to spread the inputs over. Bundle members always stay in one shard.
    shards: usize = 2,
    /// The device loader serves only even weights (zig/src/xpu/exl3_model.zig:458-463 refuses
    /// odd ones with UnsupportedBits), so a bundle whose trellis carries an odd halves count
    /// writes a pack that opens but cannot run. Raise this once the loader admits the mul1
    /// half-steps (spec step 5); until then bundles with odd halves refuse at write time.
    allow_odd_bits: bool = false,
    /// The directory the source shards are staged in. null: `<dst_dir>-eaq-stage`, wiped before
    /// the run and removed after it. A named directory is wiped before staging and left alone
    /// after, for inspection of a failed run.
    staging: ?[]const u8 = null,
    /// The driver's quant_report.json bytes — the encoding proof (per-part tiles, optimal,
    /// closed, failed, pruned1, pruned2, clamped, the gap histogram, items) plus the contract
    /// string and the encoder id. The writer gates them before anything is staged — parseable
    /// JSON object, non-empty contract and encoder, per part a present non-negative failed
    /// count and identity_fallback bool, no failed tiles, no identity fallback, no status-2
    /// items, within the size budget — and then that the parts describe exactly this pack:
    /// one part per staged trellis by normalised prefix, k, n, halves, and the sha256 of the
    /// staged payload, all under the part's encoder equal to the report's. What ships is the
    /// report as one JSON string under the `__metadata__` key "eaq.quant_report" — the
    /// safetensors metadata map pins values to strings — beside the sha256 of these exact
    /// bytes under "eaq.quant_report_sha256", in every staged shard. The repacker copies that
    /// metadata verbatim into the published shards, so the shard hash covers it, the manifest
    /// hash covers that, and the digest in config.json pins the chain: the pack vouches from
    /// its own bytes that failed was zero and the named encoder produced it. The driver
    /// writes the same bytes to disk unchanged.
    report: ?[]const u8 = null,
};

pub const Error = error{
    Empty, // no inputs
    BadInput, // a nameless tensor, a rank past safetensors.max_rank, or payload bytes that do not fit the dtype and shape
    DuplicateTensor, // one tensor name used twice
    BadBundle, // a suffixed part is missing or measures wrong: the repacker's bundle rules (core/eaq_repack.zig:259-306) checked on the payloads before anything is staged
    OddHalves, // an odd halves count (bits x.5), which the device loader still refuses; write it only with allow_odd_bits
    BadReport, // Options.report is not a JSON object, or lacks a usable contract or encoder
    ReportFailed, // the report admits the encoding did not close: failed tiles or an identity-fallback part
    ReportTooLarge, // Options.report would push a staged shard header past the writer's budget
    ReportMismatch, // the report does not describe this pack: the part-to-trellis correspondence, k, n, halves, the trellis digest, or the part's encoder is off
};

/// What repack.repack can refuse, plus the input errors above.
pub const WriteError = Error || switch (@typeInfo(@typeInfo(@TypeOf(repack.repack)).@"fn".return_type.?)) {
    .error_union => |eu| eu.error_set,
    else => @compileError("repack.repack must return an error union"),
};

/// The bundle suffixes core/eaq_repack.zig groups tensors by: a tensor wearing one belongs to
/// the entry of its name-minus-suffix base, and must reach the shard that base's .trellis does.
const bundle_suffixes = [_][]const u8{ ".trellis", ".suh", ".svh", ".su", ".sv", ".mul1", ".bias" };

fn bundleBase(name: []const u8) []const u8 {
    for (bundle_suffixes) |s| if (std.mem.endsWith(u8, name, s)) return name[0 .. name.len - s.len];
    return name;
}

/// The bundle role a suffixed name plays, by the same table (the order mirrors the repacker's
/// Kind — trellis, suh, svh, su, sv, mul1, bias, core/eaq_repack.zig:108-112; keep both in step).
fn bundleKind(name: []const u8) ?u8 {
    for (bundle_suffixes, 0..) |s, k| if (std.mem.endsWith(u8, name, s)) return @intCast(k);
    return null;
}

/// A one-dimensional tensor of `dt` and exactly `len` elements.
fn rank1(t: Input, dt: DType, len: usize) bool {
    return t.dtype == dt and t.shape.len == 1 and t.shape[0] == len;
}

/// One refusal note for whoever holds the Why, mirroring the reader's fail().
fn note(why: ?*eaq.Why, comptime fmt: []const u8, args: anytype) void {
    const w = why orelse return;
    w.len = if (std.fmt.bufPrint(&w.buf, fmt, args)) |t| t.len else |_| blk: {
        const msg = "(writer refusal message too long)";
        @memcpy(w.buf[0..msg.len], msg);
        break :blk msg.len;
    };
}

/// A dtype as a safetensors header names it: its tag in capitals (bf16 as BF16).
fn dtypeName(d: DType) []const u8 {
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

/// A quant_report this large would push a staged header toward the container's cap
/// (max_header is 1 << 28, core/eaq_container.zig:14); a report of honest size — a few
/// hundred parts of counters — is orders of magnitude smaller.
const max_report_bytes = 16 * 1024 * 1024;

/// A non-empty string field of a parsed JSON object.
fn jsonString(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = obj.get(key) orelse return null;
    return if (v == .string and v.string.len > 0) v.string else null;
}

/// The entry name a staged tensor answers to — the repacker's own normalisation
/// (core/eaq_repack.zig:122-128): the VLM wrapper sheds, a trailing .weight goes.
fn normaliseEntry(a: std.mem.Allocator, name: []const u8) ![]const u8 {
    var n = name;
    if (std.mem.startsWith(u8, n, "model.language_model."))
        n = try std.fmt.allocPrint(a, "model.{s}", .{n["model.language_model.".len..]});
    if (std.mem.endsWith(u8, n, ".weight")) n = n[0 .. n.len - ".weight".len];
    return n;
}

/// The lowercase hex sha256 of `bytes`.
fn shaHex(a: std.mem.Allocator, bytes: []const u8) ![]const u8 {
    var dig: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &dig, .{});
    return a.dupe(u8, &std.fmt.bytesToHex(dig, .lower));
}

/// Writes `inputs[list]` as one staged shard: the header recomputes every data offset to
/// contiguous positions, the payload bytes go out untouched, and the driver's report, when
/// given, rides in `__metadata__` (the repacker preserves it verbatim, hashed with the file).
fn stageShard(
    a: std.mem.Allocator,
    io: Io,
    dir: []const u8,
    index: usize,
    total: usize,
    inputs: []const Input,
    list: []const usize,
    meta: ?std.json.Value,
) !void {
    var header: std.json.ObjectMap = .empty;
    if (meta) |m| try header.put(a, "__metadata__", m);
    var offset: usize = 0;
    for (list) |i| {
        const t = inputs[i];
        var dims = std.json.Array.init(a);
        for (t.shape) |d| try dims.append(.{ .integer = @intCast(d) });
        var offs = std.json.Array.init(a);
        try offs.appendSlice(&.{ .{ .integer = @intCast(offset) }, .{ .integer = @intCast(offset + t.payload.len) } });
        var o: std.json.ObjectMap = .empty;
        try o.put(a, "dtype", .{ .string = dtypeName(t.dtype) });
        try o.put(a, "shape", .{ .array = dims });
        try o.put(a, "data_offsets", .{ .array = offs });
        try header.put(a, t.name, .{ .object = o });
        offset += t.payload.len;
    }
    const text = try std.json.Stringify.valueAlloc(a, std.json.Value{ .object = header }, .{});
    const padded = std.mem.alignForward(usize, text.len, 8);
    var head: [8]u8 = undefined;
    std.mem.writeInt(u64, &head, padded, .little);
    const spaces = try a.alloc(u8, padded - text.len);
    @memset(spaces, ' ');

    const file = try std.fmt.allocPrint(a, "model-{d:0>5}-of-{d:0>5}.safetensors", .{ index + 1, total });
    var atomic = try Io.Dir.cwd().createFileAtomic(io, try std.fs.path.join(a, &.{ dir, file }), .{ .make_path = true, .replace = true });
    defer atomic.deinit(io);
    var out: u64 = 0;
    for ([_][]const u8{ &head, text, spaces }) |piece| {
        try atomic.file.writePositionalAll(io, piece, out);
        out += piece.len;
    }
    for (list) |i| {
        const t = inputs[i];
        try atomic.file.writePositionalAll(io, t.payload, out);
        out += t.payload.len;
    }
    try atomic.replace(io);
}

fn publish(a: std.mem.Allocator, io: Io, dir: []const u8, name: []const u8, data: []const u8) !void {
    var atomic = try Io.Dir.cwd().createFileAtomic(io, try std.fs.path.join(a, &.{ dir, name }), .{ .make_path = true, .replace = true });
    defer atomic.deinit(io);
    try atomic.file.writePositionalAll(io, data, 0);
    try atomic.replace(io);
}

/// Publishes a complete EAQ pack from in-memory payloads, the container's openPack accepting it
/// byte for byte: the payloads are staged as source safetensors shards (bundle members together,
/// names and payload lengths checked first) and handed to eaq_repack.repack, which rewrites the
/// shards with recomputed offsets while hashing them, derives the manifest from the bytes just
/// written, merges config.json over the caller's, and writes EAQ_COMPLETE.json last.
pub fn write(
    gpa: std.mem.Allocator,
    io: Io,
    dst_dir: []const u8,
    inputs: []const Input,
    o: Options,
    why: ?*eaq.Why,
) WriteError!Result {
    if (inputs.len == 0) return error.Empty;
    if (o.shards == 0) return error.BadInput;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    // Check every input once; each belongs to its bundle's group, so the group can be staged
    // into one shard — the container pins an entry's parts to that entry's single shard.
    var names: std.StringHashMapUnmanaged(void) = .empty;
    var groups: std.ArrayList([]const u8) = .empty;
    var group_ids: std.StringHashMapUnmanaged(usize) = .empty;
    var roles: std.ArrayList([7]?usize) = .empty; // each group's bundle parts, by bundleKind
    const group = try a.alloc(usize, inputs.len);
    for (inputs, 0..) |t, i| {
        const want = t.need() orelse {
            note(why, "{s}: shape overflows a payload size", .{t.name});
            return error.BadInput;
        };
        if (t.name.len == 0 or t.shape.len > safetensors.max_rank or want != t.payload.len) {
            note(why, "{s}: {s} payload is {d} bytes, the shape wants {d}", .{ t.name, dtypeName(t.dtype), t.payload.len, want });
            return error.BadInput;
        }
        if ((try names.getOrPut(a, t.name)).found_existing) {
            note(why, "{s}", .{t.name});
            return error.DuplicateTensor;
        }
        const base = bundleBase(t.name);
        const gop = try group_ids.getOrPut(a, base);
        if (!gop.found_existing) {
            gop.key_ptr.* = try a.dupe(u8, base);
            gop.value_ptr.* = groups.items.len;
            try groups.append(a, gop.key_ptr.*);
            try roles.append(a, @splat(@as(?usize, null)));
        }
        group[i] = gop.value_ptr.*;
        if (bundleKind(t.name)) |kind| roles.items[group[i]][kind] = i;
    }

    // A suffixed base must be a complete, servable bundle while the caller still holds the
    // payloads: a caller that swapped halves for rows, lost a modulo, or shipped a torn part
    // refuses here, not only when a reader opens the pack after a full quant run. The rules
    // mirror the repacker's (core/eaq_repack.zig:259-306); this additionally requires the
    // .mul1 the reader always demands (exl3_eaq.zig slotEntry) even for even halves, and
    // refuses missing scales outright (the repacker's else-arm would unwrap a null).
    for (groups.items, roles.items) |base, mem| {
        var bundled = false;
        for (mem) |m| {
            if (m != null) bundled = true;
        }
        if (!bundled) continue; // a plain tensor, not a bundle
        const ti = mem[0] orelse {
            note(why, "{s}.trellis missing", .{base});
            return error.BadBundle;
        };
        const t = inputs[ti];
        if (t.dtype != .i16 or t.shape.len != 3 or t.shape[0] == 0 or t.shape[1] == 0 or t.shape[2] % 8 != 0) {
            note(why, "{s}.trellis is not I16 [K/16, N/16, halves*8]", .{base});
            return error.BadBundle;
        }
        const k = t.shape[0] * 16;
        const n = t.shape[1] * 16;
        const halves = t.shape[2] / 8;
        if (halves < 2 or halves > 16) {
            note(why, "{s}: halves {d} outside 2..16", .{ base, halves });
            return error.BadBundle;
        }
        if (!o.allow_odd_bits and halves % 2 == 1) {
            note(why, "{s}: {d} halves is {d}.5 bits, which the device loader refuses (only even weights are servable) — pass allow_odd_bits once it admits the mul1 half-steps", .{ base, halves, halves / 2 });
            return error.OddHalves;
        }
        const suh = mem[1] != null;
        const svh = mem[2] != null;
        const su = mem[3] != null;
        const sv = mem[4] != null;
        if ((suh != svh) or (su != sv) or (suh and su)) {
            note(why, "{s}: needs both suh and svh (F16) or both su and sv (I16)", .{base});
            return error.BadBundle;
        }
        if (!suh and !su) {
            note(why, "{s}: no scale factors", .{base});
            return error.BadBundle;
        }
        if (suh) { // per-weight [K],[N] or per-16 [K/16],[N/16]; a mix refuses, as the reader's does
            const per_weight = rank1(inputs[mem[1].?], .f16, k) and rank1(inputs[mem[2].?], .f16, n);
            const per_16 = rank1(inputs[mem[1].?], .f16, k / 16) and rank1(inputs[mem[2].?], .f16, n / 16);
            if (!per_weight and !per_16) {
                note(why, "{s}: scale shapes not [{d}] and [{d}] (or per-16)", .{ base, k, n });
                return error.BadBundle;
            }
        } else if (!(rank1(inputs[mem[3].?], .i16, k / 16) and rank1(inputs[mem[4].?], .i16, n / 16))) {
            note(why, "{s}: packed su/sv not I16 [{d}] and [{d}]", .{ base, k / 16, n / 16 });
            return error.BadBundle;
        }
        const mi = mem[5] orelse {
            note(why, "{s}.mul1 missing: every trellis part carries the codebook", .{base});
            return error.BadBundle;
        };
        if (inputs[mi].dtype != .i32 or inputs[mi].shape.len != 0) {
            note(why, "{s}.mul1 is not an I32 scalar", .{base});
            return error.BadBundle;
        }
        if (mem[6]) |bi| {
            if (inputs[bi].dtype != .f16) {
                note(why, "{s}.bias is not F16", .{base});
                return error.BadBundle;
            }
        }
    }

    // The Q3 binding: the driver's encoding proof enters the pack's hash chain through the
    // staged shards' __metadata__, which the repacker copies verbatim into the published
    // shards (core/eaq_repack.zig:438-439) — the shard hash covers it, the manifest hash
    // covers that, and config.json's manifest_sha256 pins the chain. It is gated before any
    // byte is staged: a pack must not vouch for an encoder that reported failed tiles, fell
    // back to identity, or handed over something that is not the report the spec promises.
    var meta: ?std.json.Value = null;
    if (o.report) |rep| {
        if (rep.len > max_report_bytes) {
            note(why, "quant_report is {d} bytes, past the {d} MiB budget", .{ rep.len, max_report_bytes / (1024 * 1024) });
            return error.ReportTooLarge;
        }
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, rep, .{}) catch {
            note(why, "quant_report does not parse", .{});
            return error.BadReport;
        };
        if (parsed != .object) {
            note(why, "quant_report is not a JSON object", .{});
            return error.BadReport;
        }
        const top_contract = jsonString(parsed.object, "contract");
        const top_encoder = jsonString(parsed.object, "encoder");
        if (top_contract == null or top_encoder == null) {
            note(why, "quant_report lacks a contract or encoder string", .{});
            return error.BadReport;
        }
        const parts = switch (parsed.object.get("parts") orelse .null) {
            .array => |p| p.items,
            else => {
                note(why, "quant_report has no parts array", .{});
                return error.BadReport;
            },
        };
        // First the encoding's own verdict: failed tiles, identity fallbacks, and unresolved
        // (status 2) items refuse publication however well the report otherwise describes
        // the pack — the artifact's gate, not the run's.
        for (parts) |part| {
            if (part != .object) {
                note(why, "a quant_report part is not an object", .{});
                return error.BadReport;
            }
            const f = part.object.get("failed") orelse {
                note(why, "a quant_report part has no failed count: an uncounted part is not a proof", .{});
                return error.BadReport;
            };
            if (f != .integer or f.integer < 0) {
                note(why, "a quant_report failed count is missing or negative", .{});
                return error.BadReport;
            }
            if (f.integer > 0) {
                note(why, "quant_report says {d} failed tiles: nothing publishes until the encoder closes them", .{f.integer});
                return error.ReportFailed;
            }
            const ib = part.object.get("identity_fallback") orelse {
                note(why, "a quant_report part has no identity_fallback bool", .{});
                return error.BadReport;
            };
            if (ib != .bool) {
                note(why, "a quant_report identity_fallback is not a bool", .{});
                return error.BadReport;
            }
            if (ib.bool) {
                note(why, "quant_report has an identity-fallback part: the encoder did not produce that trellis", .{});
                return error.ReportFailed;
            }
            if (part.object.get("items")) |its| {
                if (its != .array) {
                    note(why, "a quant_report part has no items array", .{});
                    return error.BadReport;
                }
                for (its.array.items) |item| {
                    if (item != .object) {
                        note(why, "a quant_report item is not an object", .{});
                        return error.BadReport;
                    }
                    if (item.object.get("status")) |s| {
                        if (s != .integer) {
                            note(why, "a quant_report item status is not a number", .{});
                            return error.BadReport;
                        }
                        if (s.integer == 2) {
                            note(why, "quant_report has an unresolved (status 2) tile: the encoding did not close", .{});
                            return error.ReportFailed;
                        }
                    }
                }
            }
        }
        // Then the correspondence: parts and staged trellises match one-to-one by normalised
        // prefix, k, n, halves, and every part digests its trellis's exact payload; a part
        // from another encoder is refused. A resumed run whose per-layer records were lost
        // refuses here — after a crash the report no longer describes what is staged.
        var unstaged: std.StringHashMap(usize) = .init(a);
        defer unstaged.deinit();
        for (groups.items, roles.items) |base, mem| {
            if (mem[0]) |ti| try unstaged.put(try normaliseEntry(a, base), ti);
        }
        for (parts) |part| {
            const po = part.object;
            const raw_prefix = jsonString(po, "prefix") orelse {
                note(why, "a quant_report part has no prefix string", .{});
                return error.ReportMismatch;
            };
            // the part prefix arrives in whatever shape the driver recorded it — the raw
            // base name for a VLM-wrapped layer is not the pack name the staged tensor
            // wears — so it gets the same normalisation the staged side got above.
            const prefix = try normaliseEntry(a, raw_prefix);
            const pe = jsonString(po, "encoder") orelse {
                note(why, "quant_report part {s}: no encoder string", .{prefix});
                return error.ReportMismatch;
            };
            if (!std.mem.eql(u8, pe, top_encoder.?)) {
                note(why, "quant_report part {s} names encoder {s}, not the pack's {s}", .{ prefix, pe, top_encoder.? });
                return error.ReportMismatch;
            }
            const found = unstaged.fetchRemove(prefix) orelse {
                note(why, "quant_report part {s} names no staged trellis (a second part for one counts as none)", .{prefix});
                return error.ReportMismatch;
            };
            const t = inputs[found.value];
            inline for (.{ .k, .n, .halves }, [_]i64{
                @intCast(t.shape[0] * 16),
                @intCast(t.shape[1] * 16),
                @intCast(t.shape[2] / 8),
            }) |field, want| {
                switch (po.get(@tagName(field)) orelse .null) {
                    .integer => |x| if (x != want) {
                        note(why, "quant_report part {s}: {s} {d}, the staged trellis says {d}", .{ prefix, @tagName(field), x, want });
                        return error.ReportMismatch;
                    },
                    else => {
                        note(why, "quant_report part {s}: no {s}", .{ prefix, @tagName(field) });
                        return error.ReportMismatch;
                    },
                }
            }
            const got_sha = jsonString(po, "trellis_sha256") orelse {
                note(why, "quant_report part {s}: no trellis_sha256 to verify against the staged bytes", .{prefix});
                return error.ReportMismatch;
            };
            if (!std.mem.eql(u8, got_sha, try shaHex(a, t.payload))) {
                note(why, "quant_report part {s}: trellis_sha256 does not match the staged trellis bytes", .{prefix});
                return error.ReportMismatch;
            }
        }
        var left = unstaged.iterator();
        while (left.next()) |e| {
            note(why, "staged trellis {s} has no quant_report part", .{e.key_ptr.*});
            return error.ReportMismatch;
        }
        var m: std.json.ObjectMap = .empty;
        // safetensors pins __metadata__ to string-to-string: the report rides as one JSON
        // string of its exact bytes — HF-readable, and the digest key stays checkable from
        // the pack alone
        try m.put(a, "eaq.quant_report", .{ .string = rep });
        try m.put(a, "eaq.quant_report_sha256", .{ .string = try shaHex(a, rep) });
        meta = .{ .object = m };
    }

    // The repacker reads whatever model*.safetensors the staging dir holds, so stale files from
    // an interrupted run must never be among them.
    const stage = if (o.staging) |s| s else try std.fmt.allocPrint(a, "{s}-eaq-stage", .{dst_dir});
    Io.Dir.cwd().deleteTree(io, stage) catch {};
    defer if (o.staging == null) Io.Dir.cwd().deleteTree(io, stage) catch {};

    var staged: usize = 0;
    for (0..o.shards) |s| {
        var list: std.ArrayList(usize) = .empty;
        for (group, 0..) |g, i| if (g % o.shards == s) try list.append(a, i);
        if (list.items.len == 0) continue;
        try stageShard(a, io, stage, s, o.shards, inputs, list.items, meta);
        staged += 1;
    }
    if (staged == 0) return error.Empty;
    if (o.config) |cfg| try publish(a, io, stage, "config.json", cfg);

    // The pack proper is the repacker's work: validation, the hashed shard rewrite with
    // recomputed offsets, the per-part manifest, the config merge and the marker last.
    return repack.repack(gpa, io, stage, dst_dir, why);
}

// ——————————————————————————————————————————————— tests ———————————————————————————————————————————————
// Synthetic models at the fixture's scale (zig/src/core/eaq_fixture.zig, and the reader tests in
// zig/src/xpu/exl3_eaq_test.zig): 64-wide tiles, K a multiple of 16, mul1 dictionaries, bf16 and
// f16 payloads — no device and no real weights. Every tensor carries its own payload pattern, so
// a verbatim copy is checked byte for byte.

fn readerMod() type {
    return @import("exl3_eaq");
}

/// A source config.json with nothing quantizing the pack: the eaq block must take the top level.
const toy_config =
    \\{"model_type":"qwen3_5","text_config":{"hidden_size":64,"vocab_size":248320}}
;
/// A source config.json whose top level the runner's exl3 block owns: the eaq block must go to
/// text_config, and the exl3 block must survive verbatim.
const exl3_config =
    \\{"model_type":"qwen3_5","quantization_config":{"quant_method":"exl3","bits":2.0,"codebook":"mul1"},
    \\"text_config":{"hidden_size":64,"vocab_size":248320}}
;

fn elements(dtype: DType, shape: []const usize) usize {
    var n = dtype.size();
    for (shape) |d| n *= d;
    return n;
}

const Toy = struct {
    a: std.mem.Allocator,
    list: std.ArrayList(Input) = .empty,
    seeds: usize = 0,

    /// One payload for every .mul1 tensor: a single multiplier is what lets the whole pack share
    /// the one exl3.mul1.v1 dictionary.
    const mul1_bytes = [_]u8{ 0x00, 0x00, 0x80, 0x3f };

    fn put(self: *Toy, name: []const u8, dtype: DType, shape: []const usize, payload: []const u8) !void {
        try self.list.append(self.a, .{ .name = name, .dtype = dtype, .shape = try self.a.dupe(usize, shape), .payload = payload });
    }

    /// A payload no other tensor in the model carries: byte j of tensor s is (j*37 + s*101 + 13) mod 256.
    fn pattern(self: *Toy, len: usize) ![]const u8 {
        const buf = try self.a.alloc(u8, len);
        const seed = self.seeds;
        self.seeds += 1;
        for (buf, 0..) |*b, j| b.* = @truncate((j *% 37) +% (seed *% 101) +% 13);
        return buf;
    }

    fn gen(self: *Toy, name: []const u8, dtype: DType, shape: []const usize) !void {
        try self.put(name, dtype, shape, try self.pattern(elements(dtype, shape)));
    }

    /// A trellis bundle at fixture scale: I16 codes [kt, nt, halves*8], the scales as F16
    /// per-weight suh/svh or packed I16 su/sv, and the shared mul1 scalar.
    fn bundle(self: *Toy, base: []const u8, kt: usize, nt: usize, halves: usize, sign_packed: bool) !void {
        try self.gen(try std.fmt.allocPrint(self.a, "{s}.trellis", .{base}), .i16, &.{ kt, nt, halves * 8 });
        if (sign_packed) {
            try self.gen(try std.fmt.allocPrint(self.a, "{s}.su", .{base}), .i16, &.{kt});
            try self.gen(try std.fmt.allocPrint(self.a, "{s}.sv", .{base}), .i16, &.{nt});
        } else {
            try self.gen(try std.fmt.allocPrint(self.a, "{s}.suh", .{base}), .f16, &.{ kt * 16 });
            try self.gen(try std.fmt.allocPrint(self.a, "{s}.svh", .{base}), .f16, &.{ nt * 16 });
        }
        try self.put(try std.fmt.allocPrint(self.a, "{s}.mul1", .{base}), .i32, &.{}, &mul1_bytes);
    }
};

const L = "model.language_model.layers.";

/// One layer's entries at mixed bits: the fused mixer_in's parts disagree on halves on purpose
/// (q 2.0 bits, k 3.0, v the half step 1.5), o_proj rides the packed su/sv scale style with a
/// bias, and the mlp's gate and up differ too. Only K and the codebook must agree within a slot.
fn addLayer(t: *Toy, l: usize) !void {
    const p = try std.fmt.allocPrint(t.a, "{s}{d}.", .{ L, l });
    const mlp = [_]struct { []const u8, usize }{
        .{ "mlp.gate_proj", 4 }, .{ "mlp.up_proj", 6 }, .{ "mlp.down_proj", 4 },
    };
    if ((l + 1) % 4 == 0) { // an attention layer (the reader's isAttention rule)
        const attn = [_]struct { []const u8, usize }{
            .{ "self_attn.q_proj", 4 }, .{ "self_attn.k_proj", 6 }, .{ "self_attn.v_proj", 3 },
        };
        for (attn) |a| try t.bundle(try std.fmt.allocPrint(t.a, "{s}{s}", .{ p, a[0] }), 4, 4, a[1], false);
        try t.bundle(try std.fmt.allocPrint(t.a, "{s}self_attn.o_proj", .{p}), 4, 4, 8, true);
        try t.gen(try std.fmt.allocPrint(t.a, "{s}self_attn.o_proj.bias", .{p}), .f16, &.{64});
        for ([_][]const u8{ "self_attn.q_norm.weight", "self_attn.k_norm.weight" }) |n|
            try t.gen(try std.fmt.allocPrint(t.a, "{s}{s}", .{ p, n }), .bf16, &.{8});
    } else { // a GDN layer
        const gdn = [_]struct { []const u8, usize }{
            .{ "linear_attn.in_proj_qkv", 4 }, .{ "linear_attn.in_proj_z", 4 }, .{ "linear_attn.out_proj", 5 },
        };
        for (gdn) |g| try t.bundle(try std.fmt.allocPrint(t.a, "{s}{s}", .{ p, g[0] }), 4, 4, g[1], false);
        for ([_][]const u8{ "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight" }) |n|
            try t.gen(try std.fmt.allocPrint(t.a, "{s}{s}", .{ p, n }), .f16, &.{ 2, 8 });
        try t.gen(try std.fmt.allocPrint(t.a, "{s}linear_attn.conv1d.weight", .{p}), .bf16, &.{ 8, 1, 4 });
        try t.gen(try std.fmt.allocPrint(t.a, "{s}linear_attn.A_log", .{p}), .bf16, &.{2});
        try t.gen(try std.fmt.allocPrint(t.a, "{s}linear_attn.dt_bias", .{p}), .bf16, &.{2});
        try t.gen(try std.fmt.allocPrint(t.a, "{s}linear_attn.norm.weight", .{p}), .bf16, &.{8});
    }
    for (mlp) |m| try t.bundle(try std.fmt.allocPrint(t.a, "{s}{s}", .{ p, m[0] }), 4, 4, m[1], false);
}

/// The embedding, the final norm and the head — the head at 3.5 bits (odd halves) over K=64, N=256.
fn addRest(t: *Toy) !void {
    try t.gen("model.language_model.embed_tokens.weight", .bf16, &.{ 16, 8 });
    try t.gen("model.language_model.norm.weight", .bf16, &.{8});
    try t.bundle("lm_head", 4, 16, 7, false);
}

fn tmpPath(gpa: std.mem.Allocator, sub: []const u8) ![]const u8 {
    return std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{sub});
}

/// A synthetic model of `n_layers` layers plus embed, norm and head, written as a pack.
fn writeToyPack(gpa: std.mem.Allocator, io: Io, dst: []const u8, n_layers: usize, why: ?*eaq.Why) !Result {
    var t = Toy{ .a = gpa };
    for (0..n_layers) |l| try addLayer(&t, l);
    try addRest(&t);
    const report = try makeReport(gpa, t.list.items, "host-ref");
    return write(gpa, io, dst, t.list.items, .{ .shards = 2, .config = toy_config, .allow_odd_bits = true, .report = report }, why); // the toy's out_proj and head ride odd halves
}

fn projectionAt(table: anytype, layer: usize, slot: anytype, part: usize) ?@TypeOf(table.projections[0]) {
    for (table.projections) |m| if (m.layer == layer and m.slot == slot and m.part == part) return m;
    return null;
}

/// The manifest's bits param for an entry, admitting only a JSON float: the odd halves counts
/// (bits = halves/2 with halves odd) must be written as floats to survive the hash round trip.
fn bitsFloat(e: *const eaq.Entry) ?f64 {
    const v = e.params.get("bits") orelse return null;
    return switch (v) {
        .float => |f| f,
        else => null,
    };
}

/// Read config.json back and flip exactly one hex digit of its manifest_sha256 value: the
/// bytes stay valid JSON, so only the hash check can refuse the pack afterwards.
fn flipConfigHash(gpa: std.mem.Allocator, io: Io, path: []const u8) ![]const u8 {
    const text = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    const edit = try gpa.dupe(u8, text);
    const key = std.mem.indexOf(u8, edit, "\"manifest_sha256\"") orelse return error.TestUnexpectedResult;
    var j = key + "\"manifest_sha256\"".len;
    while (edit[j] != '"') j += 1; // the value's opening quote: the needle already ends at the key's closing quote
    edit[j + 1] = if (edit[j + 1] == '0') '1' else '0'; // the digest's first hex digit
    try std.testing.expect(!std.mem.eql(u8, edit, text));
    return edit;
}

/// The driver-shaped report that describes `inputs`: the top-level contract and encoder,
/// and one part per staged trellis — normalised prefix, k, n, halves, all closed, and the
/// digest of that trellis's exact payload, as the correspondence gate demands.
fn makeReport(gpa: std.mem.Allocator, inputs: []const Input, encoder: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(gpa, "{\"contract\":\"exl3-enc/v1: q12 u32-rebased inf=0xFFFFFFFF clamp=4.0\",\"encoder\":\"");
    try out.appendSlice(gpa, encoder);
    try out.appendSlice(gpa, "\",\"parts\":[");
    var first = true;
    for (inputs) |t| {
        if (!std.mem.endsWith(u8, t.name, ".trellis")) continue;
        if (!first) try out.appendSlice(gpa, ",");
        first = false;
        const prefix = try normaliseEntry(gpa, t.name[0 .. t.name.len - ".trellis".len]);
        try out.appendSlice(gpa, try std.fmt.allocPrint(gpa,
            \\{{"prefix":"{s}","encoder":"{s}","k":{d},"n":{d},"halves":{d},"failed":0,"identity_fallback":false,"items":[],"trellis_sha256":"{s}"}}
        , .{ prefix, encoder, t.shape[0] * 16, t.shape[1] * 16, t.shape[2] / 8, try shaHex(gpa, t.payload) }));
    }
    try out.appendSlice(gpa, "]}");
    return out.items;
}

/// The single-part reports the correspondence gate is checked with, each off in one way.
const PartCase = enum { ok, absent, k16, orphan, dup, no_sha, bad_sha, other_encoder, status2, no_failed, neg_failed };

fn reportWith(gpa: std.mem.Allocator, m: PartCase, sha: []const u8) ![]const u8 {
    if (m == .absent) return try std.fmt.allocPrint(gpa,
        \\{{"contract":"exl3-enc/v1","encoder":"host-ref","parts":[]}}
    , .{});
    const prefix: []const u8 = if (m == .orphan) "model.layers.3.self_attn.v_proj" else "model.layers.3.self_attn.q_proj";
    const k: usize = if (m == .k16) 80 else 64;
    const enc: []const u8 = if (m == .other_encoder) "device-kernel" else "host-ref";
    const sha_field: []const u8 = if (m == .no_sha)
        ""
    else if (m == .bad_sha)
        ",\"trellis_sha256\":\"0000000000000000000000000000000000000000000000000000000000000000\""
    else
        try std.fmt.allocPrint(gpa, ",\"trellis_sha256\":\"{s}\"", .{sha});
    const items: []const u8 = if (m == .status2) ",\"items\":[{\"prefix\":\"model.layers.3.self_attn.q_proj\",\"tk\":0,\"tn\":0,\"encoder\":\"none\",\"status\":2,\"flags\":0,\"cost\":0,\"open_cost\":0,\"pinned\":0,\"reason\":\"tile did not close\"}]" else ",\"items\":[]";
    const failed_field: []const u8 = if (m == .no_failed)
        ""
    else if (m == .neg_failed)
        "\"failed\":-2,"
    else
        "\"failed\":0,";
    const one = try std.fmt.allocPrint(gpa,
        \\{{"prefix":"{s}","encoder":"{s}","k":{d},"n":{d},"halves":4,{s}"identity_fallback":false{s}{s}}}
    , .{ prefix, enc, k, 64, failed_field, sha_field, items });
    if (m != .dup) return try std.fmt.allocPrint(gpa,
        \\{{"contract":"exl3-enc/v1","encoder":"host-ref","parts":[{s}]}}
    , .{one});
    return try std.fmt.allocPrint(gpa,
        \\{{"contract":"exl3-enc/v1","encoder":"host-ref","parts":[{s},{s}]}}
    , .{ one, one });
}

const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectEqualStrings = std.testing.expectEqualStrings;
const expectApproxEqAbs = std.testing.expectApproxEqAbs;

test "the writer publishes a pack that opens and completes the reader's layer table" {
    const reader = readerMod();
    var ar = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ar.deinit();
    const gpa = ar.allocator();
    const io = std.testing.io;
    var dst = std.testing.tmpDir(.{});
    defer dst.cleanup();
    const dp = try tmpPath(gpa, std.mem.sliceTo(&dst.sub_path, 0));

    var t = Toy{ .a = gpa };
    for (0..repack.layers) |l| try addLayer(&t, l);
    try addRest(&t);

    var why: eaq.Why = .{};
    const report_bytes = try makeReport(gpa, t.list.items, "host-ref");
    const got = try write(gpa, io, dp, t.list.items, .{ .shards = 2, .config = toy_config, .allow_odd_bits = true, .report = report_bytes }, &why);
    try expectEqual(@as(usize, 2), got.shards);
    try expectEqual(@as(usize, 401), got.n_trellis); // 7 trellis entries per attention layer, 6 per GDN, the head
    try expectEqual(@as(usize, 322), got.n_dense);
    try expectEqual(@as(usize, 723), got.entries);
    try expectEqual(@as(usize, 1), got.mul1_values); // one multiplier payload across the pack: one dictionary
    try expectEqual(@as(usize, 0), got.dropped_tensors);

    var pack = try reader.openPack(gpa, io, dp, &why);
    defer pack.deinit();
    var table = reader.map(gpa, &pack, &why) catch |e| {
        std.debug.print("map refused: {s} — why: '{s}'\n", .{ @errorName(e), why.text() });
        return e;
    };
    defer table.deinit(gpa);

    // every slot of every one of the 64 layers is filled: attention 7 projections, GDN 8
    try expect(table.complete);
    try expectEqual(@as(usize, 16 * 7 + 48 * 8), table.projections.len);
    const head = table.head orelse return error.TestUnexpectedResult;
    try expect(!head.dense);
    const hq = head.q orelse return error.TestUnexpectedResult;
    try expectEqual(@as(u8, 7), hq.halves); // bits 3.5: an odd count rides with mul1 alone
    try expectEqual(@as(usize, 64), hq.k);
    try expectEqual(@as(usize, 256), hq.n);
    try expect(hq.suh);
    try expect(hq.codebook == core.exl3_format.Codebook.mul1);
    try expect(table.embed != null and table.embed.?.dense);

    // bits live per part: layer 3's fused mixer_in carries three widths, and an odd
    // halves count is a float in the manifest, K and N read off the trellis part's shape
    const v = pack.entry("model.layers.3.self_attn.v_proj") orelse return error.TestUnexpectedResult;
    try expectApproxEqAbs(@as(f64, 1.5), bitsFloat(v) orelse return error.TestUnexpectedResult, 0.0000001);
    try expectEqual(@as(i64, 64), v.params.get("K").?.integer);
    try expectEqual(@as(i64, 64), v.params.get("N").?.integer);
    try expectEqualStrings("exl3.mul1.v1", v.params.get("dictionary").?.string);
    const k3 = projectionAt(table, 3, reader.Slot.mixer_in, 1) orelse return error.TestUnexpectedResult;
    try expectEqual(@as(u8, 6), k3.q.?.halves);
    const q3 = projectionAt(table, 3, reader.Slot.mixer_in, 0) orelse return error.TestUnexpectedResult;
    try expectEqual(@as(u8, 4), q3.q.?.halves);
    const o3 = projectionAt(table, 3, reader.Slot.mixer_out, 0) orelse return error.TestUnexpectedResult;
    try expectEqual(@as(u8, 8), o3.q.?.halves);
    try expect(!o3.q.?.suh); // the packed su/sv style, not suh/svh
    try expect(o3.q.?.bias); // the F16 bias part travelled with the bundle

    // the manifest's hash chain: config.json and the marker both vouch for the bytes written
    const raw = try Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(gpa, &.{ dp, eaq.manifest_name }), gpa, .limited(1 << 26));
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(raw, &d, .{});
    try expectEqualStrings(&std.fmt.bytesToHex(d, .lower), &got.manifest_sha256);
    const marker = try Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(gpa, &.{ dp, eaq.marker_name }), gpa, .limited(1 << 20));
    const mk = try std.json.parseFromSliceLeaky(std.json.Value, gpa, marker, .{});
    try expectEqualStrings(&got.manifest_sha256, mk.object.get("manifest_sha256").?.string);

    // the Q3 binding rode through the repacker: every published shard carries the report
    // as one JSON string of its exact bytes beside the digest key — string-to-string,
    // HF-readable, inside what openPack just verified by shard hash
    {
        for ([_][]const u8{ "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors" }) |fname| {
            const shard = try Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(gpa, &.{ dp, fname }), gpa, .limited(1 << 26));
            const hl = std.mem.readInt(u64, shard[0..8], .little);
            const hdr = try std.json.parseFromSliceLeaky(std.json.Value, gpa, shard[8..][0..@intCast(hl)], .{});
            const meta = hdr.object.get("__metadata__") orelse return error.TestUnexpectedResult;
            const bound = meta.object.get("eaq.quant_report") orelse return error.TestUnexpectedResult;
            try expect(bound == .string); // not an object: safetensors pins metadata to strings
            try expectEqualStrings(report_bytes, bound.string);
            const rec = meta.object.get("eaq.quant_report_sha256") orelse return error.TestUnexpectedResult;
            try expect(rec == .string);
            try expectEqualStrings(try shaHex(gpa, bound.string), rec.string); // digest of the decoded report
            try expectEqualStrings(try shaHex(gpa, report_bytes), rec.string); // of the bytes handed in
        }
    }

    // every payload byte that went in comes back out of the rewritten shards, verbatim
    var f1 = try safetensors.File.open(gpa, io, try std.fs.path.join(gpa, &.{ dp, "model-00001-of-00002.safetensors" }));
    defer f1.close(io);
    var f2 = try safetensors.File.open(gpa, io, try std.fs.path.join(gpa, &.{ dp, "model-00002-of-00002.safetensors" }));
    defer f2.close(io);
    for (t.list.items) |in| {
        const tg = f1.get(in.name) orelse f2.get(in.name) orelse {
            std.debug.print("tensor {s} missing from both shards\n", .{in.name});
            return error.TestUnexpectedResult;
        };
        try expect(tg.is(in.dtype, in.shape));
        try expect(std.mem.eql(u8, in.payload, tg.bytes));
    }

    // bundle members keep one shard: the manifest pins an entry's parts to a single file, so
    // the grouping that stages them together must also be the shard the entry names — checked
    // for the plain-scale bundle and for the packed one that carries a bias
    {
        const Part = struct { tensor: []const u8, entry: []const u8, parts: []const []const u8 };
        for ([_]Part{
            .{ .tensor = L ++ "3.self_attn.q_proj", .entry = "model.layers.3.self_attn.q_proj", .parts = &.{ ".trellis", ".suh", ".svh", ".mul1" } },
            .{ .tensor = L ++ "3.self_attn.o_proj", .entry = "model.layers.3.self_attn.o_proj", .parts = &.{ ".trellis", ".su", ".sv", ".mul1", ".bias" } },
        }) |bc| {
            var shard_of: []const u8 = "";
            for (bc.parts) |suf| {
                const name = try std.fmt.allocPrint(gpa, "{s}{s}", .{ bc.tensor, suf }); // the header wears the source's name
                const at: []const u8 = if (f1.get(name) != null)
                    "model-00001-of-00002.safetensors"
                else if (f2.get(name) != null)
                    "model-00002-of-00002.safetensors"
                else {
                    std.debug.print("bundle part {s} missing from both shards\n", .{name});
                    return error.TestUnexpectedResult;
                };
                if (shard_of.len == 0) shard_of = at else try expectEqualStrings(shard_of, at);
            }
            const ent = pack.entry(bc.entry) orelse return error.TestUnexpectedResult;
            try expectEqualStrings(shard_of, ent.shard); // the manifest names exactly that file
        }
    }

    // the default staging directory does not survive the publish
    try std.testing.expectError(error.FileNotFound, Io.Dir.cwd().access(io, try std.fmt.allocPrint(gpa, "{s}-eaq-stage", .{dp}), .{}));
}

test "a two-layer synthetic model at mixed bits per part writes, opens and maps" {
    const reader = readerMod();
    var ar = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ar.deinit();
    const gpa = ar.allocator();
    const io = std.testing.io;
    var dst = std.testing.tmpDir(.{});
    defer dst.cleanup();
    const dp = try tmpPath(gpa, std.mem.sliceTo(&dst.sub_path, 0));

    var why: eaq.Why = .{};
    const got = try writeToyPack(gpa, io, dp, 2, &why);
    try expectEqual(@as(usize, 2), got.shards);
    try expectEqual(@as(usize, 13), got.n_trellis); // qkv z out gate up down per layer, and the head
    try expectEqual(@as(usize, 14), got.n_dense); // a b conv1d A_log dt_bias norm per layer, embed and model.norm
    try expectEqual(@as(usize, 27), got.entries);

    var pack = try reader.openPack(gpa, io, dp, &why);
    defer pack.deinit();
    var table = reader.map(gpa, &pack, null) catch |e| {
        std.debug.print("map refused a two-layer pack: {s}\n", .{@errorName(e)});
        return e;
    };
    defer table.deinit(gpa);
    // both layers are GDN under the (layer+1)%4 rule, so each fills 8 slots; the table's
    // completion rule demands all 64 layers, so a two-layer pack maps clean but incomplete
    try expectEqual(@as(usize, 16), table.projections.len);
    try expect(!table.complete);
    try expect(table.head != null and table.head.?.q != null);
    try expect(table.embed != null and table.embed.?.dense);
    const o = pack.entry("model.layers.0.linear_attn.out_proj") orelse return error.TestUnexpectedResult;
    try expectApproxEqAbs(@as(f64, 2.5), bitsFloat(o) orelse return error.TestUnexpectedResult, 0.0000001); // the odd 5 halves
    const a0 = projectionAt(table, 0, reader.Slot.mixer_in, 2) orelse return error.TestUnexpectedResult;
    try expect(a0.dense and a0.q == null); // the GDN gates ride the dense route
}

test "a torn trellis bundle refuses at write time, before a byte is staged" {
    var ar = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ar.deinit();
    const gpa = ar.allocator();
    const io = std.testing.io;
    var dst = std.testing.tmpDir(.{});
    defer dst.cleanup();
    const dp = try tmpPath(gpa, std.mem.sliceTo(&dst.sub_path, 0));
    const base = "model.layers.3.self_attn.q_proj"; // a mapped mixer_in part, so refusals come from the tearing alone

    var t = Toy{ .a = gpa };
    try t.bundle(base, 4, 4, 4, false); // trellis, suh, svh, mul1 — the parts the cases below tear
    const good = try gpa.dupe(Input, t.list.items);

    const Case = struct { list: []const Input, want: []const u8 };
    var cases: std.ArrayList(Case) = .empty;

    { // the codes are a rank-3 tile grid whose third axis counts eights
        var x = try gpa.dupe(Input, good);
        x[0].shape = &.{ 16, 32 }; // the same bytes laid out as rows, not tiles
        try cases.append(gpa, .{ .list = x, .want = ".trellis is not I16" });
    }
    {
        var x = try gpa.dupe(Input, good);
        x[0].shape = &.{ 4, 4, 33 }; // a lost modulo
        x[0].payload = try t.pattern(4 * 4 * 33 * 2);
        try cases.append(gpa, .{ .list = x, .want = ".trellis is not I16" });
    }
    { // halves outside the reader's Quant range
        var x = try gpa.dupe(Input, good);
        x[0].shape = &.{ 4, 4, 8 };
        x[0].payload = try t.pattern(4 * 4 * 8 * 2);
        try cases.append(gpa, .{ .list = x, .want = "halves 1 outside" });
    }
    {
        var x = try gpa.dupe(Input, good);
        x[0].shape = &.{ 4, 4, 136 };
        x[0].payload = try t.pattern(4 * 4 * 136 * 2);
        try cases.append(gpa, .{ .list = x, .want = "halves 17 outside" });
    }
    { // every part of the bundle ships
        try cases.append(gpa, .{ .list = good[1..], .want = ".trellis missing" });
        var l: std.ArrayList(Input) = .empty;
        try l.append(gpa, good[0]);
        try l.append(gpa, good[3]);
        try cases.append(gpa, .{ .list = l.items, .want = "no scale factors" });
    }
    { // the scales measure the trellis: K and N, or their sixteenths, never anything else
        var x = try gpa.dupe(Input, good);
        x[1].shape = &.{4 * 16 + 16}; // a caller that read rows for columns
        x[1].payload = try t.pattern(2 * (4 * 16 + 16)); // as many F16s as the shape claims
        try cases.append(gpa, .{ .list = x, .want = "scale shapes not [64] and [64]" });
    }
    { // one scale style or the other, never a mix and never half a pair
        try t.gen(try std.fmt.allocPrint(gpa, "{s}.su", .{base}), .i16, &.{4});
        const mixed = try gpa.dupe(Input, t.list.items);
        try cases.append(gpa, .{ .list = mixed, .want = "needs both suh and svh" });
        try t.gen(try std.fmt.allocPrint(gpa, "{s}.sv", .{base}), .i16, &.{4});
        var l: std.ArrayList(Input) = .empty;
        try l.appendSlice(gpa, good);
        try l.append(gpa, t.list.items[t.list.items.len - 1]); // an sv with no su
        try cases.append(gpa, .{ .list = l.items, .want = "needs both suh and svh" });
    }
    { // packed su/sv measure K/16 and N/16
        var p = Toy{ .a = gpa };
        try p.bundle(base, 4, 4, 4, true);
        var x = try gpa.dupe(Input, p.list.items);
        x[1].shape = &.{6};
        x[1].payload = try p.pattern(12);
        try cases.append(gpa, .{ .list = x, .want = "packed su/sv not I16 [4] and [4]" });
    }
    { // the codebook is a scalar and the bias, when present, is F16
        var x = try gpa.dupe(Input, good);
        x[3].shape = &.{1};
        try cases.append(gpa, .{ .list = x, .want = ".mul1 is not an I32 scalar" });
        const bias = blk: { // a fresh Toy: the shared one already carries the su/sv cases
            var bt = Toy{ .a = gpa };
            try bt.gen(try std.fmt.allocPrint(gpa, "{s}.bias", .{base}), .bf16, &.{1});
            break :blk bt.list.items[0];
        };
        var l: std.ArrayList(Input) = .empty;
        try l.appendSlice(gpa, good);
        try l.append(gpa, bias);
        try cases.append(gpa, .{ .list = l.items, .want = ".bias is not F16" });
    }

    for (cases.items) |c| {
        var why: eaq.Why = .{};
        std.testing.expectError(error.BadBundle, write(gpa, io, dp, c.list, .{}, &why)) catch |e| {
            std.debug.print("case '{s}': ", .{c.want});
            return e;
        };
        if (std.mem.indexOf(u8, why.text(), c.want) == null) {
            std.debug.print("refusal said '{s}', wanted '{s}'\n", .{ why.text(), c.want });
            return error.TestUnexpectedResult;
        }
    }

    // a well-measured bundle still writes — and nothing was staged by the refusals
    _ = try write(gpa, io, dp, good, .{}, null);
    try std.testing.expectError(error.FileNotFound, Io.Dir.cwd().access(io, try std.fmt.allocPrint(gpa, "{s}-eaq-stage", .{dp}), .{}));
}

test "odd halves refuse unless allow_odd_bits is raised" {
    var ar = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ar.deinit();
    const gpa = ar.allocator();
    const io = std.testing.io;
    var dst = std.testing.tmpDir(.{});
    defer dst.cleanup();
    const dp = try tmpPath(gpa, std.mem.sliceTo(&dst.sub_path, 0));

    // 1.5 bits over K=64, N=64: a pack that would open and then never run, since the device
    // loader serves only even weights (zig/src/xpu/exl3_model.zig:458-463).
    var t = Toy{ .a = gpa };
    try t.bundle("model.layers.3.self_attn.v_proj", 4, 4, 3, false);
    var why: eaq.Why = .{};
    try std.testing.expectError(error.OddHalves, write(gpa, io, dp, t.list.items, .{}, &why));
    try expect(std.mem.indexOf(u8, why.text(), "allow_odd_bits") != null);

    // once the loader admits the mul1 half-steps (spec step 5), the very same call writes
    _ = try write(gpa, io, dp, t.list.items, .{ .allow_odd_bits = true }, null);
}

test "a report that does not describe the pack it ships refuses at publish" {
    const reader = readerMod();
    var ar = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ar.deinit();
    const gpa = ar.allocator();
    const io = std.testing.io;
    var dst = std.testing.tmpDir(.{});
    defer dst.cleanup();
    const dp = try tmpPath(gpa, std.mem.sliceTo(&dst.sub_path, 0));
    var dst2 = std.testing.tmpDir(.{}); // refusals publish nothing; the plain-pack case gets its own dir
    defer dst2.cleanup();
    const dp2 = try tmpPath(gpa, std.mem.sliceTo(&dst2.sub_path, 0));

    var t = Toy{ .a = gpa };
    try t.bundle("model.layers.3.self_attn.q_proj", 4, 4, 4, false);
    const good = try gpa.dupe(Input, t.list.items);
    const sha = try shaHex(gpa, good[0].payload); // the digest every honest part carries

    // an honest report publishes and opens — and so does a pack with no report at all
    const honest = try reportWith(gpa, .ok, sha);
    _ = try write(gpa, io, dp, good, .{ .config = toy_config, .report = honest }, null);
    var opened = try reader.openPack(gpa, io, dp, null);
    opened.deinit();
    _ = try write(gpa, io, dp2, good, .{ .config = toy_config }, null);
    var plain = try reader.openPack(gpa, io, dp2, null);
    plain.deinit();

    const Case = struct { m: PartCase, err: anyerror, want: []const u8 };
    for ([_]Case{
        .{ .m = .absent, .err = error.ReportMismatch, .want = "has no quant_report part" },
        .{ .m = .k16, .err = error.ReportMismatch, .want = "k 80, the staged trellis says 64" },
        .{ .m = .orphan, .err = error.ReportMismatch, .want = "names no staged trellis" },
        .{ .m = .dup, .err = error.ReportMismatch, .want = "a second part for one counts as none" },
        .{ .m = .no_sha, .err = error.ReportMismatch, .want = "no trellis_sha256" },
        .{ .m = .bad_sha, .err = error.ReportMismatch, .want = "does not match the staged trellis bytes" },
        .{ .m = .other_encoder, .err = error.ReportMismatch, .want = "not the pack's" },
        .{ .m = .status2, .err = error.ReportFailed, .want = "status 2" },
        .{ .m = .no_failed, .err = error.BadReport, .want = "has no failed count" },
        .{ .m = .neg_failed, .err = error.BadReport, .want = "missing or negative" },
    }) |c| {
        var why: eaq.Why = .{};
        const rep_c = try reportWith(gpa, c.m, sha);
        if (write(gpa, io, dp, good, .{ .report = rep_c }, &why)) |_| {
            std.debug.print("case {s}: published\n", .{@tagName(c.m)});
            return error.TestUnexpectedResult;
        } else |e| {
            if (e != c.err) {
                std.debug.print("case {s}: {s}\n", .{ @tagName(c.m), @errorName(e) });
                return error.TestUnexpectedResult;
            }
        }
        if (std.mem.indexOf(u8, why.text(), c.want) == null) {
            std.debug.print("case {s}: refusal said '{s}', wanted '{s}'\n", .{ @tagName(c.m), why.text(), c.want });
            return error.TestUnexpectedResult;
        }
    }
}

test "the bound quant_report is hash-covered and its refusals are loud" {
    const reader = readerMod();
    var ar = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ar.deinit();
    const gpa = ar.allocator();
    const io = std.testing.io;
    var dst = std.testing.tmpDir(.{});
    defer dst.cleanup();
    const dp = try tmpPath(gpa, std.mem.sliceTo(&dst.sub_path, 0));
    var dst2 = std.testing.tmpDir(.{}); // the refusal cases publish nothing, so they get a dir of their own
    defer dst2.cleanup();
    const dp2 = try tmpPath(gpa, std.mem.sliceTo(&dst2.sub_path, 0));

    var t = Toy{ .a = gpa };
    try t.bundle("model.layers.3.self_attn.q_proj", 4, 4, 4, false);
    const good = try gpa.dupe(Input, t.list.items);

    const rep = try makeReport(gpa, good, "host-ref");
    _ = try write(gpa, io, dp, good, .{ .config = toy_config, .report = rep }, null);
    var ok_pack = try reader.openPack(gpa, io, dp, null);
    ok_pack.deinit();

    // one letter flipped inside the bound report moves the shard hash: the proof is inside
    // the chain, so tampering with it refuses at open like tampering with a payload byte
    {
        const bytes = try Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(gpa, &.{ dp, "model-00001-of-00002.safetensors" }), gpa, .limited(1 << 26));
        const at = std.mem.indexOf(u8, bytes, "exl3-enc/v1") orelse return error.TestUnexpectedResult; // inside eaq.quant_report
        const edit = try gpa.dupe(u8, bytes);
        edit[at + 5] ^= 0x20; // 'e' → 'E' in "enc": valid JSON, different bytes
        try publish(gpa, io, dp, "model-00001-of-00002.safetensors", edit);
        try std.testing.expectError(eaq.Error.ShardHashMismatch, reader.openPack(gpa, io, dp, null));
    }

    // the report is the artifact's gate, not the run's: failed tiles and an identity
    // fallback refuse publication, and the unfinished pack shows no marker
    {
        const failed =
            \\{"contract":"exl3-enc/v1","encoder":"host-ref",
            \\"parts":[{"prefix":"model.layers.3.self_attn.q_proj","failed":1,"identity_fallback":false}]}
        ;
        var why: eaq.Why = .{};
        try std.testing.expectError(error.ReportFailed, write(gpa, io, dp2, good, .{ .report = failed }, &why));
        try expect(std.mem.indexOf(u8, why.text(), "failed tiles") != null);
        const fallback =
            \\{"contract":"exl3-enc/v1","encoder":"host-ref",
            \\"parts":[{"prefix":"model.layers.3.self_attn.q_proj","failed":0,"identity_fallback":true}]}
        ;
        try std.testing.expectError(error.ReportFailed, write(gpa, io, dp2, good, .{ .report = fallback }, null));
        try std.testing.expectError(error.FileNotFound, Io.Dir.cwd().access(io, try std.fs.path.join(gpa, &.{ dp2, eaq.marker_name }), .{}));
    }

    // and a report that is not the report the spec promises never reaches the hash chain
    {
        try std.testing.expectError(error.BadReport, write(gpa, io, dp2, good, .{ .report = "not json at all {{{" }, null));
        try std.testing.expectError(error.BadReport, write(gpa, io, dp2, good, .{ .report = "{\"parts\":[{\"failed\":0}]}" }, null)); // no contract, no encoder
        try std.testing.expectError(error.BadReport, write(gpa, io, dp2, good, .{ .report = "{\"contract\":\"c\",\"encoder\":\"e\"}" }, null)); // no parts
        const big = try gpa.alloc(u8, max_report_bytes + 1);
        @memset(big, 'x');
        try std.testing.expectError(error.ReportTooLarge, write(gpa, io, dp2, good, .{ .report = big }, null));
    }

    // the literal malformed emission the driver writes today (quant-driver.zig:1874:
    // identity_fallback with no closing quote) is BadReport, exactly as the ruling warned
    {
        const malformed =
            \\{"contract":"exl3-enc/v1","encoder":"host-ref","parts":[{"prefix":"model.layers.3.self_attn.q_proj","identity_fallback: false}]}
        ;
        try std.testing.expectError(error.BadReport, write(gpa, io, dp2, good, .{ .report = malformed }, null));
    }
}

test "a flipped byte inside a shard refuses at open" {
    const reader = readerMod();
    var ar = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ar.deinit();
    const gpa = ar.allocator();
    const io = std.testing.io;
    var dst = std.testing.tmpDir(.{});
    defer dst.cleanup();
    const dp = try tmpPath(gpa, std.mem.sliceTo(&dst.sub_path, 0));
    _ = try writeToyPack(gpa, io, dp, 2, null);

    const file = "model-00001-of-00002.safetensors";
    const raw = try Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(gpa, &.{ dp, file }), gpa, .limited(1 << 24));
    const flipped = try gpa.dupe(u8, raw);
    flipped[flipped.len - 1] ^= 1; // payload bytes, past every header
    try publish(gpa, io, dp, file, flipped);

    var why: eaq.Why = .{};
    try std.testing.expectError(eaq.Error.ShardHashMismatch, reader.openPack(gpa, io, dp, &why));
    try expectEqualStrings(file, why.text());
}

test "a manifest edited after config hashed it refuses at open" {
    const reader = readerMod();
    var ar = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ar.deinit();
    const gpa = ar.allocator();
    const io = std.testing.io;
    var dst = std.testing.tmpDir(.{});
    defer dst.cleanup();
    const dp = try tmpPath(gpa, std.mem.sliceTo(&dst.sub_path, 0));
    _ = try writeToyPack(gpa, io, dp, 2, null);

    const raw = try Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(gpa, &.{ dp, eaq.manifest_name }), gpa, .limited(1 << 26));
    try publish(gpa, io, dp, eaq.manifest_name, try std.mem.concat(gpa, u8, &.{ raw, " " }));
    try std.testing.expectError(eaq.Error.ManifestHashMismatch, reader.openPack(gpa, io, dp, null));
}

test "a config.json stating another manifest hash refuses at open" {
    const reader = readerMod();
    var ar = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ar.deinit();
    const gpa = ar.allocator();
    const io = std.testing.io;
    var dst = std.testing.tmpDir(.{});
    defer dst.cleanup();
    const dp = try tmpPath(gpa, std.mem.sliceTo(&dst.sub_path, 0));

    // The tamper cases rewrite config.json after the pack is complete, so they need a pack
    // whose config.json exists and whose manifest digest they can key on.
    const res = try writeToyPack(gpa, io, dp, 2, null);
    try expectEqual(@as(usize, 2), res.shards);
    const cfg_path = try std.fs.path.join(gpa, &.{ dp, "config.json" });

    // One hex digit of the digest flipped inside config.json: the bytes stay valid JSON and
    // the manifest and shards are untouched, so only the hash check can still refuse the pack.
    {
        const edit = try flipConfigHash(gpa, io, cfg_path);
        try publish(gpa, io, dp, "config.json", edit);
        try std.testing.expectError(eaq.Error.ManifestHashMismatch, reader.openPack(gpa, io, dp, null));
        try publish(gpa, io, dp, "config.json", try std.fmt.allocPrint(gpa,
            \\{{"quantization_config":{{"quant_method":"eaq","manifest":"{s}","manifest_sha256":"{s}"}}}}
        , .{ eaq.manifest_name, res.manifest_sha256 })); // the honest digest, for the next case
        var back = try reader.openPack(gpa, io, dp, null);
        back.deinit();
    }

    // and a config that simply states another digest entirely
    {
        const other = try std.fmt.allocPrint(gpa,
            \\{{"quantization_config":{{"quant_method":"eaq","manifest":"{s}","manifest_sha256":"{s}"}}}}
        , .{ eaq.manifest_name, "abcabc1234567890abcabc1234567890abcabc1234567890abcabc1234567890" });
        try publish(gpa, io, dp, "config.json", other);
        try std.testing.expectError(eaq.Error.ManifestHashMismatch, reader.openPack(gpa, io, dp, null));
    }
}

test "the eaq block takes the top level when config.json leaves it free" {
    var ar = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ar.deinit();
    const gpa = ar.allocator();
    const io = std.testing.io;
    var dst = std.testing.tmpDir(.{});
    defer dst.cleanup();
    const dp = try tmpPath(gpa, std.mem.sliceTo(&dst.sub_path, 0));
    var t = Toy{ .a = gpa };
    for (0..2) |l| try addLayer(&t, l);
    try addRest(&t);
    const got = try write(gpa, io, dp, t.list.items, .{ .shards = 2, .config = toy_config, .allow_odd_bits = true }, null);

    const text = try Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(gpa, &.{ dp, "config.json" }), gpa, .limited(1 << 24));
    const cfg = try std.json.parseFromSliceLeaky(std.json.Value, gpa, text, .{});
    const q = cfg.object.get("quantization_config") orelse return error.TestUnexpectedResult;
    try expectEqualStrings(eaq.quant_method, q.object.get("quant_method").?.string);
    try expectEqualStrings(eaq.manifest_name, q.object.get("manifest").?.string);
    try expectEqualStrings(&got.manifest_sha256, q.object.get("manifest_sha256").?.string);
    try expectEqual(@as(i64, 64), cfg.object.get("text_config").?.object.get("hidden_size").?.integer); // untouched fields survive
}

test "the eaq block lands in text_config when the exl3 block owns the top" {
    var ar = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ar.deinit();
    const gpa = ar.allocator();
    const io = std.testing.io;
    var dst = std.testing.tmpDir(.{});
    defer dst.cleanup();
    const dp = try tmpPath(gpa, std.mem.sliceTo(&dst.sub_path, 0));
    var t = Toy{ .a = gpa };
    for (0..2) |l| try addLayer(&t, l);
    try addRest(&t);
    const got = try write(gpa, io, dp, t.list.items, .{ .shards = 2, .config = exl3_config, .allow_odd_bits = true }, null);

    const text = try Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(gpa, &.{ dp, "config.json" }), gpa, .limited(1 << 24));
    const cfg = try std.json.parseFromSliceLeaky(std.json.Value, gpa, text, .{});
    const root_q = cfg.object.get("quantization_config") orelse return error.TestUnexpectedResult;
    try expectEqualStrings("exl3", root_q.object.get("quant_method").?.string); // the runner's block, verbatim
    try expectEqualStrings("mul1", root_q.object.get("codebook").?.string);
    const text_q = cfg.object.get("text_config").?.object.get("quantization_config") orelse return error.TestUnexpectedResult;
    try expectEqualStrings(eaq.quant_method, text_q.object.get("quant_method").?.string);
    try expectEqualStrings(&got.manifest_sha256, text_q.object.get("manifest_sha256").?.string);
}

test "payloads that do not fit their dtype and shape, a nameless tensor and a double name refuse" {
    var ar = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer ar.deinit();
    const gpa = ar.allocator();
    const io = std.testing.io;
    var dst = std.testing.tmpDir(.{});
    defer dst.cleanup();
    const dp = try tmpPath(gpa, std.mem.sliceTo(&dst.sub_path, 0));

    const short = [_]u8{ 1, 2, 3 };
    const in1 = [_]Input{.{ .name = "model.norm.weight", .dtype = .bf16, .shape = &.{8}, .payload = &short }};
    try std.testing.expectError(Error.BadInput, write(gpa, io, dp, &in1, .{}, null));

    const in2 = [_]Input{.{ .name = "", .dtype = .bf16, .shape = &.{2}, .payload = &.{ 1, 2 } }};
    try std.testing.expectError(Error.BadInput, write(gpa, io, dp, &in2, .{}, null));

    const ok = [_]u8{ 1, 2, 3, 4 };
    const in3 = [_]Input{
        .{ .name = "model.norm.weight", .dtype = .bf16, .shape = &.{2}, .payload = &ok },
        .{ .name = "model.norm.weight", .dtype = .bf16, .shape = &.{2}, .payload = &ok },
    };
    try std.testing.expectError(Error.DuplicateTensor, write(gpa, io, dp, &in3, .{}, null));

    try std.testing.expectError(Error.Empty, write(gpa, io, dp, &.{}, .{}, null));
    // refusing never leaves a half-published pack: the marker is the repacker's last word, and
    // an input that never staged means no marker at all
    try std.testing.expectError(error.FileNotFound, Io.Dir.cwd().access(io, try std.fs.path.join(gpa, &.{ dp, eaq.marker_name }), .{}));
}
