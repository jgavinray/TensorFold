//! The XPU reader for EAQ packs: verifies the container, refuses what this backend cannot
//! serve, and maps the entries onto the Qwen3.8-27B layer table (zig/src/xpu/load.zig Layer)
//! for the weight upload of PRs 2-4. The Pack must outlive the Table: names and part shapes
//! borrow its arena.
const std = @import("std");
const core = @import("core");
const eaq = core.eaq_container;
const exl3_format = core.exl3_format;

pub const family = "qwen3_5";

/// Mirrors qwen27.zig LAYERS; the load path asserts the two agree before it maps.
pub const LAYERS = 64;

/// Mirrors qwen27.zig isAttention: every fourth layer from the fourth is attention, the rest GDN.
pub fn isAttention(i: usize) bool {
    return (i + 1) % 4 == 0;
}

pub const Error = error{
    EaqFp8Entry, // the pack carries an eaq.fp8/0 entry: no fp8 operand exists on this card's XMX
    UnknownProjection, // an entry name outside the Qwen3.8-27B layer table
    DuplicateProjection, // two entries claim one layer's slot and part
    MismatchedFusedK, // fused parts of one slot disagree on K
    BadTrellisParams, // a trellis entry's params lack usable bits, K, N or a known dictionary
    BadTrellisPart, // a trellis part is missing, not I16, or shaped against the params
    BadScalePart, // the scale parts are missing, mixed, or shaped against K/N
    BadBiasPart, // the bias part is not F16
} || eaq.Error;

pub const Slot = enum(u2) { mixer_in, mixer_out, gu, down };

/// The trellis quantities of one entry; dense entries carry none of them (`q` null).
/// `halves` is 2*bits and may be odd — the odd counts are mul1's half steps; the
/// trellis word count per tile row is halves * 8.
pub const Quant = struct {
    halves: u8,
    k: usize,
    n: usize,
    codebook: exl3_format.Codebook,
    suh: bool, // fp16 scales suh/svh; false: the packed sign words su/sv (I16, [K/16]/[N/16])
    bias: bool,
};

/// One mapped projection; `dense` marks an eaq.dense/0 entry, which the load path may serve
/// through the fp16 route instead of the trellis kernels.
pub const Mapped = struct {
    name: []const u8,
    entry: *const eaq.Entry, // the part tensor names, dtypes and shapes live here
    layer: usize,
    slot: Slot,
    part: u8, // the part's index inside a fused projection (q|k|v, in_proj_qkv|z|a|b, gate|up)
    dense: bool = false,
    q: ?Quant = null,
};

/// The head: trellis in the packs served today, dense otherwise; served at stored width.
pub const Head = struct {
    name: []const u8,
    entry: *const eaq.Entry,
    dense: bool,
    q: ?Quant = null,
};

/// One entry that fills no projection slot: a norm, the embedding, the GDN tensors.
pub const Other = struct {
    name: []const u8,
    entry: *const eaq.Entry,
    dense: bool,
};

/// Every entry of the pack sorted onto the layer table. `complete` is false when any
/// layer's slots (or the head or the embedding) are missing; the load path refuses such
/// a pack before upload.
pub const Table = struct {
    projections: []const Mapped = &.{},
    head: ?Head = null,
    embed: ?Other = null,
    others: []const Other = &.{},
    complete: bool = false,

    pub fn deinit(t: *Table, gpa: std.mem.Allocator) void {
        gpa.free(t.projections);
        gpa.free(t.others);
        t.* = undefined;
    }
};

fn fail(why: ?*eaq.Why, comptime fmt: []const u8, args: anytype) void {
    const w = why orelse return;
    w.len = if (std.fmt.bufPrint(&w.buf, fmt, args)) |t| t.len else |_| blk: {
        const note = "(refusal message too long)";
        @memcpy(w.buf[0..note.len], note);
        break :blk note.len;
    };
}

/// Opens the pack as this backend serves it: family qwen3_5, no runtime features —
/// a manifest's required_features refuses here — and no fp8 entry anywhere, since the
/// card's XMX has no fp8 operand and one fp8 layer makes the pack unservable.
pub fn openPack(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, why: ?*eaq.Why) !eaq.Pack {
    var pack = try eaq.open(gpa, io, dir, .{ .family = family, .features = &.{}, .why = why });
    errdefer pack.deinit();
    for (pack.entries) |*e| if (e.format == .fp8) {
        fail(why, "entry {s}: eaq.fp8/0 has no operand on this card", .{e.name});
        return error.EaqFp8Entry;
    };
    return pack;
}

/// The projection parts of one layer's slots, in load order (zig/src/xpu/load.zig projection calls).
const Slots = struct { mixer_in: []const []const u8, mixer_out: []const []const u8 };
fn slotParts(i: usize) Slots {
    return if (isAttention(i))
        .{ .mixer_in = &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" }, .mixer_out = &.{"self_attn.o_proj"} }
    else
        .{ .mixer_in = &.{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b" }, .mixer_out = &.{"linear_attn.out_proj"} };
}

/// One entry name's place in the table; null for a name outside it. A trailing ".weight"
/// is stripped first, so dense entries named like HF tensors map the same as trellis ones.
pub fn slotFor(name_in: []const u8) ?Place {
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
    if (layer >= LAYERS) return null;
    const tail = rest[dot + 1 ..];
    const slots = slotParts(layer);
    for (slots.mixer_in, 0..) |p, pi| if (std.mem.eql(u8, tail, p)) return .{ .projection = .{ .layer = layer, .slot = .mixer_in, .part = @intCast(pi) } };
    for (slots.mixer_out, 0..) |p, pi| if (std.mem.eql(u8, tail, p)) return .{ .projection = .{ .layer = layer, .slot = .mixer_out, .part = @intCast(pi) } };
    if (std.mem.eql(u8, tail, "mlp.gate_proj")) return .{ .projection = .{ .layer = layer, .slot = .gu, .part = 0 } };
    if (std.mem.eql(u8, tail, "mlp.up_proj")) return .{ .projection = .{ .layer = layer, .slot = .gu, .part = 1 } };
    if (std.mem.eql(u8, tail, "mlp.down_proj")) return .{ .projection = .{ .layer = layer, .slot = .down, .part = 0 } };
    const attn_known = [_][]const u8{ "self_attn.q_norm", "self_attn.k_norm" };
    const gdn_known = [_][]const u8{ "linear_attn.conv1d", "linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.norm" };
    const known: []const []const u8 = if (isAttention(layer)) &attn_known else &gdn_known;
    for (known) |k| if (std.mem.eql(u8, tail, k)) return .other;
    if (std.mem.eql(u8, tail, "input_layernorm") or std.mem.eql(u8, tail, "post_attention_layernorm")) return .other;
    return null;
}

fn codebookOf(dictionary: []const u8) ?exl3_format.Codebook {
    // the container's exact dictionary ids (core/eaq_container.zig checkDictionaries)
    const ids = [_]struct { []const u8, exl3_format.Codebook }{
        .{ "exl3.3inst.v1", .inst3 },
        .{ "exl3.mcg.v1", .mcg },
        .{ "exl3.mul1.v1", .mul1 },
    };
    for (ids) |id| if (std.mem.eql(u8, dictionary, id[0])) return id[1];
    return null;
}

fn quantOf(entry: *const eaq.Entry, why: ?*eaq.Why) Error!Quant {
    var q: Quant = .{ .halves = 0, .k = 0, .n = 0, .codebook = .inst3, .suh = false, .bias = false };
    const dict = jsonString(entry.params, "dictionary") orelse {
        fail(why, "entry {s}: no dictionary", .{entry.name});
        return error.BadTrellisParams;
    };
    q.codebook = codebookOf(dict) orelse {
        fail(why, "entry {s}: dictionary {s} names no TensorFold table", .{ entry.name, dict });
        return error.BadTrellisParams;
    };
    // halves = 2 * bits may be odd: the odd counts are mul1's half steps (core/exl3_load.zig admit)
    const bits = jsonNum(entry.params, "bits") orelse {
        fail(why, "entry {s}: no bits", .{entry.name});
        return error.BadTrellisParams;
    };
    const halves_f = bits * 2;
    if (!(bits >= 1 and bits <= 8) or @rem(halves_f, 1) != 0) {
        fail(why, "entry {s}: bits {d} is not a half-step count in 1..8", .{ entry.name, bits });
        return error.BadTrellisParams;
    }
    q.halves = @intFromFloat(halves_f);
    if (q.halves < 2 or q.halves > 16) {
        fail(why, "entry {s}: halves {d} outside 2..16", .{ entry.name, q.halves });
        return error.BadTrellisParams;
    }
    if (@rem(q.halves, 2) != 0 and q.codebook != .mul1) {
        fail(why, "entry {s}: odd halves {d} need the mul1 codebook", .{ entry.name, q.halves });
        return error.BadTrellisParams;
    }
    q.k = jsonInt(entry.params, "K") orelse {
        fail(why, "entry {s}: no K", .{entry.name});
        return error.BadTrellisParams;
    };
    q.n = jsonInt(entry.params, "N") orelse {
        fail(why, "entry {s}: no N", .{entry.name});
        return error.BadTrellisParams;
    };
    if (q.k == 0 or q.n == 0 or q.k % 16 != 0 or q.n % 16 != 0) {
        fail(why, "entry {s}: K {d} and N {d} must be nonzero multiples of 16", .{ entry.name, q.k, q.n });
        return error.BadTrellisParams;
    }
    var trellis_at: ?usize = null;
    var suh_at: ?usize = null;
    var svh_at: ?usize = null;
    var su_at: ?usize = null;
    var sv_at: ?usize = null;
    for (entry.parts, 0..) |p, i| {
        if (std.mem.eql(u8, p.name, "bias")) {
            q.bias = true;
            if (p.dtype != .f16) {
                fail(why, "entry {s}: bias part is {s}, not F16", .{ entry.name, @tagName(p.dtype) });
                return error.BadBiasPart;
            }
        } else if (std.mem.eql(u8, p.name, "trellis")) {
            trellis_at = i;
            if (p.dtype != .i16) {
                fail(why, "entry {s}: trellis part is {s}, not I16", .{ entry.name, @tagName(p.dtype) });
                return error.BadTrellisPart;
            }
            // [K/16, N/16, halves*8] per core/exl3_load.zig parts
            const want = [_]usize{ q.k / 16, q.n / 16, @as(usize, q.halves) * 8 };
            if (p.shape.len != 3 or p.shape[0] != want[0] or p.shape[1] != want[1] or p.shape[2] != want[2]) {
                fail(why, "entry {s}: trellis shape is not [{d}, {d}, {d}]", .{ entry.name, want[0], want[1], want[2] });
                return error.BadTrellisPart;
            }
        } else if (std.mem.eql(u8, p.name, "suh") or std.mem.eql(u8, p.name, "svh")) {
            if (p.dtype != .f16) {
                fail(why, "entry {s}: scale part {s} is {s}, not F16", .{ entry.name, p.name, @tagName(p.dtype) });
                return error.BadScalePart;
            }
            if (std.mem.eql(u8, p.name, "suh")) suh_at = i else svh_at = i;
        } else if (std.mem.eql(u8, p.name, "su") or std.mem.eql(u8, p.name, "sv")) {
            if (std.mem.eql(u8, p.name, "su")) su_at = i else sv_at = i;
        }
    }
    if (trellis_at == null) {
        fail(why, "entry {s}: no trellis part", .{entry.name});
        return error.BadTrellisPart;
    }
    // exactly one scale style, both parts present and consistently shaped
    if ((suh_at == null) != (svh_at == null) or (su_at == null) != (sv_at == null) or
        (suh_at != null and su_at != null) or (svh_at != null and sv_at != null) or
        (suh_at == null and su_at == null))
    {
        fail(why, "entry {s}: needs both suh and svh (F16) or both su and sv (I16)", .{entry.name});
        return error.BadScalePart;
    }
    if (suh_at) |si| {
        const suh = entry.parts[si].shape;
        const svh = entry.parts[svh_at.?].shape;
        // suh [K] with svh [N] (per-weight), or [K/16] with [N/16] (per-16); mixed refuses
        const per_weight = suh.len == 1 and svh.len == 1 and suh[0] == q.k and svh[0] == q.n;
        const per16 = suh.len == 1 and svh.len == 1 and suh[0] == q.k / 16 and svh[0] == q.n / 16;
        if (!per_weight and !per16) {
            fail(why, "entry {s}: scale shapes are not [{d}] and [{d}] (or per-16)", .{ entry.name, q.k, q.n });
            return error.BadScalePart;
        }
        q.suh = true;
    } else {
        // the packed sign words su/sv: I16 [K/16] and [N/16] (core/exl3_load.zig parts)
        const su = entry.parts[su_at.?].shape;
        const sv = entry.parts[sv_at.?].shape;
        const dt_ok = entry.parts[su_at.?].dtype == .i16 and entry.parts[sv_at.?].dtype == .i16;
        if (!dt_ok or su.len != 1 or sv.len != 1 or su[0] != q.k / 16 or sv[0] != q.n / 16) {
            fail(why, "entry {s}: packed su/sv are not I16 [{d}] and [{d}]", .{ entry.name, q.k / 16, q.n / 16 });
            return error.BadScalePart;
        }
        q.suh = false;
    }
    return q;
}

fn jsonNum(o: std.json.ObjectMap, key: []const u8) ?f64 {
    return switch (o.get(key) orelse return null) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

fn jsonInt(o: std.json.ObjectMap, key: []const u8) ?usize {
    return switch (o.get(key) orelse return null) {
        .integer => |i| if (i >= 0) @intCast(i) else null,
        .float => |f| if (f >= 0 and f <= 9_007_199_254_740_992 and @rem(f, 1) == 0) @intFromFloat(f) else null,
        else => null,
    };
}

fn jsonString(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    return switch (o.get(key) orelse return null) {
        .string => |s| s,
        else => null,
    };
}

/// One entry name's place in the layer table.
pub const Part = struct { layer: usize, slot: Slot, part: u8 };
pub const Place = union(enum) {
    projection: Part,
    head: void, // lm_head: trellis in the packs served today, dense otherwise
    embed: void, // model.embed_tokens: fp16/bf16 as stored
    other: void, // norms, the GDN gates' dense tensors, the conv, A_log, dt_bias
};

/// Maps every entry onto the Qwen3.8-27B layer table, sorted by (layer, slot, part).
/// Any fp8 entry refuses the whole pack: it cannot be served, so the pack is unservable
/// however complete the rest is. Callers must run `map` before any upload.
pub fn map(gpa: std.mem.Allocator, pack: *const eaq.Pack, why: ?*eaq.Why) (Error || std.mem.Allocator.Error)!Table {
    var projections: std.ArrayList(Mapped) = .empty;
    var others: std.ArrayList(Other) = .empty;
    errdefer {
        projections.deinit(gpa);
        others.deinit(gpa);
    }
    var table: Table = .{};
    for (pack.entries) |*e| {
        if (e.format == .fp8) {
            fail(why, "entry {s}: eaq.fp8/0 has no operand on this card", .{e.name});
            return error.EaqFp8Entry;
        }
        const place = slotFor(e.name) orelse {
            fail(why, "entry {s}: not a Qwen3.8-27B name", .{e.name});
            return error.UnknownProjection;
        };
        switch (place) {
            .head => {
                if (table.head != null) {
                    fail(why, "entries {s} and {s} both fill the head", .{ table.head.?.name, e.name });
                    return error.DuplicateProjection;
                }
                table.head = .{ .name = e.name, .entry = e, .dense = e.format == .dense, .q = if (e.format == .dense) null else try quantOf(e, why) };
            },
            .embed => {
                if (table.embed != null) {
                    fail(why, "entries {s} and {s} both fill the embedding", .{ table.embed.?.name, e.name });
                    return error.DuplicateProjection;
                }
                table.embed = .{ .name = e.name, .entry = e, .dense = e.format == .dense };
            },
            .other => {
                for (others.items) |o| if (std.mem.eql(u8, o.name, e.name)) {
                    fail(why, "entries {s} appear twice", .{e.name});
                    return error.DuplicateProjection;
                };
                try others.append(gpa, .{ .name = e.name, .entry = e, .dense = e.format == .dense });
            },
            .projection => |m| {
                for (projections.items) |p| if (p.layer == m.layer and p.slot == m.slot and p.part == m.part) {
                    fail(why, "entries {s} and {s} both fill layer {d}'s {s} part {d}", .{ p.name, e.name, m.layer, @tagName(m.slot), m.part });
                    return error.DuplicateProjection;
                };
                try projections.append(gpa, .{ .name = e.name, .entry = e, .layer = m.layer, .slot = m.slot, .part = m.part, .dense = e.format == .dense, .q = if (e.format == .dense) null else try quantOf(e, why) });
            },
        }
    }
    std.mem.sort(Mapped, projections.items, {}, struct {
        fn less(_: void, a: Mapped, b: Mapped) bool {
            if (a.layer != b.layer) return a.layer < b.layer;
            if (@intFromEnum(a.slot) != @intFromEnum(b.slot)) return @intFromEnum(a.slot) < @intFromEnum(b.slot);
            return a.part < b.part;
        }
    }.less);
    table.projections = try projections.toOwnedSlice(gpa);
    errdefer gpa.free(table.projections);
    table.others = try others.toOwnedSlice(gpa);
    errdefer gpa.free(table.others);
    // fused parts of one slot must agree on K (same input width) and on the trellis
    // style (halves, codebook): they were one projection before packing
    for (table.projections, 0..) |p, i| {
        if (i > 0 and p.layer == table.projections[i - 1].layer and p.slot == table.projections[i - 1].slot and p.part > 0) {
            const prev = table.projections[i - 1];
            if (prev.q != null and p.q != null) {
                const a = prev.q.?;
                const b = p.q.?;
                if (a.k != b.k or a.halves != b.halves or a.codebook != b.codebook) {
                    fail(why, "layer {d} {s}: part {d} disagrees with part {d} on K, halves or codebook", .{ p.layer, @tagName(p.slot), p.part, prev.part });
                    return error.MismatchedFusedK;
                }
            }
        }
    }
    table.complete = table.head != null and table.embed != null;
    if (table.complete) for (0..LAYERS) |l| {
        const slots = slotParts(l);
        if (countSlot(table.projections, l, .mixer_in) != slots.mixer_in.len or
            countSlot(table.projections, l, .mixer_out) != slots.mixer_out.len or
            countSlot(table.projections, l, .gu) != 2 or
            countSlot(table.projections, l, .down) != 1) table.complete = false;
    };
    return table;
}

fn countSlot(projections: []const Mapped, layer: usize, slot: Slot) usize {
    var n: usize = 0;
    for (projections) |p| if (p.layer == layer and p.slot == slot) {
        n += 1;
    };
    return n;
}
