//! tf-eaq-mixed-plan: emit an explicit-width allocator plan for a mixed half-step pack
//! (T-HS-10: halves 5 on every body part of layers 0..3 and on the head, halves 6
//! elsewhere). `tf-eaq-mixed-plan <base-plan.json> <out-plan.json> --odd-layers a..b
//! --odd-halves N --body-halves M --head-halves H [--method NAME]`. The base plan is an
//! allocator emission (core/eaq_alloc... core/eaq-alloc's plan-2.9.json shape: target_bpw,
//! head_bits, method, parts[{base, halves, bits}], plan_sha); its part set must be exactly
//! the 401 trellis members, so the widths are the only thing this tool rewrites.
//!
//! The bases leave normalized to the driver's lookup key: PartPlan.base as layerSlots
//! builds it (zig/src/xpu/quant.zig:284-321) — `model.language_model.layers.<i>.<tail>`,
//! `lm_head` — which is the uniform-plan key the driver's halvesOf consults
//! (zig/src/xpu/quant.zig:1383-1386); a `.weight` suffix there never hits, so the
//! allocator's emitted spelling (it carries the suffix) is shed here. The head stays
//! "lm_head" (headSlotPlan, zig/src/xpu/quant.zig:324-329).
//!
//! target_bpw is the K*N-weighted mean trellis bits over the body parts (spec 3.1's
//! body-only target), the geometry table mirrored from layerSlots' fused-slot table —
//! the same numbers the driver encodes against. plan_sha is core.exl3_alloc.planShaWith
//! over the canonical serialization (the driver's own intake recomputes it
//! identically, zig/src/xpu/quant.zig:1135-1140), and the file is re-read before the
//! tool prints its verdict, so a float that failed to round-trip the JSON would stop
//! here rather than at the driver's gate. head_bits/bits ride as decoration for
//! core.exl3_alloc.planFromJson; the driver's parser ignores unknown fields.
const std = @import("std");
const core = @import("core");
const alloc_mod = core.exl3_alloc;
const Io = std.Io;

const tail_spec = struct { tail: []const u8, k: u32, n: u32 };
const attn_tails = [_]tail_spec{
    .{ .tail = "self_attn.q_proj", .k = 5120, .n = 12288 },
    .{ .tail = "self_attn.k_proj", .k = 5120, .n = 1024 },
    .{ .tail = "self_attn.v_proj", .k = 5120, .n = 1024 },
    .{ .tail = "self_attn.o_proj", .k = 6144, .n = 5120 },
    .{ .tail = "mlp.gate_proj", .k = 5120, .n = 17408 },
    .{ .tail = "mlp.up_proj", .k = 5120, .n = 17408 },
    .{ .tail = "mlp.down_proj", .k = 17408, .n = 5120 },
};
const gdn_tails = [_]tail_spec{
    .{ .tail = "linear_attn.in_proj_qkv", .k = 5120, .n = 10240 },
    .{ .tail = "linear_attn.in_proj_z", .k = 5120, .n = 6144 },
    .{ .tail = "linear_attn.out_proj", .k = 6144, .n = 5120 },
    .{ .tail = "mlp.gate_proj", .k = 5120, .n = 17408 },
    .{ .tail = "mlp.up_proj", .k = 5120, .n = 17408 },
    .{ .tail = "mlp.down_proj", .k = 17408, .n = 5120 },
};

fn isAttention(i: usize) bool { // q.isAttention / repack.isAttention, the same rule
    return (i + 1) % 4 == 0;
}

fn bitsText(h: u8, buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "{d}.{d}", .{ @as(u8, h / 2), @as(u8, (h % 2) * 5) }) catch unreachable;
}

const Ctx = struct {
    a: std.mem.Allocator,
    // the set of distinct body bases seen: a name twice, or a count other than 400, refuses
    seen: std.StringHashMapUnmanaged(void),
    head_seen: bool = false,
    layer_a: usize,
    layer_b: usize,
    odd_halves: u8,
    body_halves: u8,
    head_halves: u8,
};

/// The width rule for one normalized base, plus its geometry through the map key.
fn classify(a: std.mem.Allocator, ctx: *Ctx, base_in: []const u8) !struct { base: []const u8, halves: u8, k: u32, n: u32, head: bool } {
    if (std.mem.eql(u8, base_in, "lm_head")) {
        if (ctx.head_seen) return error.DuplicatePart;
        ctx.head_seen = true;
        return .{ .base = try a.dupe(u8, "lm_head"), .halves = ctx.head_halves, .k = 5120, .n = 248320, .head = true };
    }
    const pre = "model.language_model.layers.";
    if (!std.mem.startsWith(u8, base_in, pre)) return error.BadBase;
    const rest = base_in[pre.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return error.BadBase;
    const layer = std.fmt.parseInt(usize, rest[0..dot], 10) catch return error.BadBase;
    if (layer >= 64) return error.BadBase;
    const tail = rest[dot + 1 ..];
    const tails: []const tail_spec = if (isAttention(layer)) &attn_tails else &gdn_tails;
    for (tails) |ts| {
        if (!std.mem.eql(u8, ts.tail, tail)) continue;
        const key = try std.fmt.allocPrint(a, "{s}{s}", .{ pre, rest }); // the base as the driver keys it
        const gop = try ctx.seen.getOrPut(a, key);
        if (gop.found_existing) return error.DuplicatePart;
        gop.key_ptr.* = try a.dupe(u8, key);
        gop.value_ptr.* = {};
        const odd = layer >= ctx.layer_a and layer <= ctx.layer_b;
        return .{ .base = key, .halves = if (odd) ctx.odd_halves else ctx.body_halves, .k = ts.k, .n = ts.n, .head = false };
    }
    return error.BadBase; // a tail this layer class does not have (q/k/v vs qkv/z are caught here)
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 7) return error.Usage;
    const a = init.arena.allocator();
    var maybe_base: ?[]const u8 = null;
    var maybe_out: ?[]const u8 = null;
    var la: usize = 0;
    var lb: usize = 0;
    var have_layers = false;
    var odd_h: u8 = 0;
    var body_h: u8 = 0;
    var head_h: u8 = 0;
    var method: []const u8 = "fragment-pack/v1";
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--odd-layers")) {
            i += 1;
            if (i >= args.len) return error.Usage;
            const v = args[i];
            const dot2 = std.mem.indexOf(u8, v, "..") orelse return error.Usage;
            la = try std.fmt.parseInt(usize, v[0..dot2], 10);
            lb = try std.fmt.parseInt(usize, v[dot2 + 2 ..], 10);
            if (la > lb or lb >= 64) return error.Usage;
            have_layers = true;
        } else if (std.mem.eql(u8, arg, "--odd-halves")) {
            i += 1;
            odd_h = try std.fmt.parseInt(u8, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--body-halves")) {
            i += 1;
            body_h = try std.fmt.parseInt(u8, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--head-halves")) {
            i += 1;
            head_h = try std.fmt.parseInt(u8, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--method")) {
            i += 1;
            method = args[i];
        } else if (maybe_base == null) {
            maybe_base = arg;
        } else if (maybe_out == null) {
            maybe_out = arg;
        } else return error.Usage;
    }
    const base_path = maybe_base orelse return error.Usage;
    const out_path = maybe_out orelse return error.Usage;
    if (!have_layers or odd_h < 2 or odd_h > 16 or body_h < 2 or body_h > 16 or head_h < 2 or head_h > 16) return error.Usage;

    const raw = try Io.Dir.cwd().readFileAlloc(io, base_path, a, .limited(1 << 24));
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, raw, .{ .ignore_unknown_fields = true });
    if (parsed != .object) return error.BadBasePlan;
    const obj = parsed.object;
    const pv = obj.get("parts") orelse return error.BadBasePlan;
    if (pv != .array) return error.BadBasePlan;
    // the base plan must be the allocator's canonical emission: re-hash what it declares
    {
        const declared = obj.get("plan_sha") orelse return error.BadBasePlan;
        if (declared != .string or declared.string.len != 64) return error.BadBasePlan;
        var tb: f64 = 0;
        if (obj.get("target_bpw")) |t| switch (t) {
            .float => |f| tb = f,
            .integer => |ii| tb = @floatFromInt(ii),
            else => return error.BadBasePlan,
        };
        var hb: u8 = 0;
        if (obj.get("head_bits")) |hbv| { // the allocator emits "4.0"-style strings; derive halves
            if (hbv != .string) return error.BadBasePlan;
            const dot = std.mem.indexOfScalar(u8, hbv.string, '.') orelse return error.BadBasePlan;
            const whole = try std.fmt.parseInt(u8, hbv.string[0..dot], 10);
            const frac = hbv.string[dot + 1 ..];
            if (!std.mem.eql(u8, frac, "0") and !std.mem.eql(u8, frac, "5")) return error.BadBasePlan;
            hb = whole * 2 + @intFromBool(frac[0] == '5');
        } else if (obj.get("head_halves")) |hh| {
            if (hh != .integer) return error.BadBasePlan;
            hb = @intCast(hh.integer);
        } else return error.BadBasePlan;
        const mv = obj.get("method") orelse return error.BadBasePlan;
        if (mv != .string) return error.BadBasePlan;
        var pw: std.ArrayList(alloc_mod.PartWidth) = .empty;
        for (pv.array.items) |item| {
            if (item != .object) return error.BadBasePlan;
            const b = item.object.get("base") orelse return error.BadBasePlan;
            const h = item.object.get("halves") orelse return error.BadBasePlan;
            if (b != .string or h != .integer) return error.BadBasePlan;
            try pw.append(a, .{ .base = b.string, .halves = @intCast(h.integer) });
        }
        const p = alloc_mod.Plan{ .target_bpw = tb, .head_halves = hb, .method = mv.string, .parts = pw.items, .plan_sha = undefined };
        const got = try alloc_mod.planShaWith(a, p);
        const hex = std.fmt.bytesToHex(got, .lower);
        if (!std.ascii.eqlIgnoreCase(hex[0..], declared.string)) {
            std.debug.print("mixed-plan: {s}: declared plan_sha {s}, canonical {s} — the base plan is not an intact allocator emission\n", .{ base_path, declared.string, &hex });
            return error.BadBasePlan;
        }
    }

    var ctx = Ctx{ .a = a, .seen = .empty, .layer_a = la, .layer_b = lb, .odd_halves = odd_h, .body_halves = body_h, .head_halves = head_h };
    var parts: std.ArrayList(alloc_mod.PartWidth) = .empty; // order preserved: the file's own order is the canonical order
    var texts: std.ArrayList(struct { base: []const u8, halves: u8 }) = .empty;
    var wsum_kn: f64 = 0;
    var wsum_bits: f64 = 0;
    for (pv.array.items) |item| {
        const b = item.object.get("base") orelse return error.BadBasePlan;
        if (b != .string) return error.BadBasePlan;
        var base = b.string;
        if (std.mem.endsWith(u8, base, ".weight")) base = base[0 .. base.len - ".weight".len]; // the driver's key carries no suffix
        const c = classify(a, &ctx, base) catch |e| {
            std.debug.print("mixed-plan: part \"{s}\": {t} — the base plan's part set is not the 401 trellis members\n", .{ b.string, e });
            return e;
        };
        try parts.append(a, .{ .base = c.base, .halves = c.halves });
        try texts.append(a, .{ .base = c.base, .halves = c.halves });
        if (!c.head) {
            const kn: f64 = @floatFromInt(@as(u64, c.k) * c.n);
            wsum_kn += kn;
            wsum_bits += kn * @as(f64, @floatFromInt(c.halves)) / 2.0;
        }
    }
    if (ctx.seen.count() != 400 or !ctx.head_seen) {
        std.debug.print("mixed-plan: base plan covers {d} body parts + head {} — the 401 (400 + lm_head) are required\n", .{ ctx.seen.count(), ctx.head_seen });
        return error.BadBasePlan;
    }
    const target: f64 = wsum_bits / wsum_kn;
    const plan = alloc_mod.Plan{ .target_bpw = target, .head_halves = head_h, .method = method, .parts = parts.items, .plan_sha = undefined };
    const sha = try alloc_mod.planShaWith(a, plan);

    // emit: driver fields first (quant.zig loadPlan reads target_bpw/head_halves/method/parts/plan_sha),
    // head_bits + per-part bits as decoration for core.exl3_alloc.planFromJson
    var aw: std.Io.Writer.Allocating = .init(a);
    var w = &aw.writer;
    try w.print("{{\n  \"target_bpw\": {d},\n  \"head_halves\": {d},\n  \"head_bits\": \"{s}\",\n  \"method\": \"{s}\",\n  \"parts\": [\n", .{
        target, head_h, bitsText(head_h, try a.alloc(u8, 8)), method,
    });
    for (texts.items, 0..) |t, ti| {
        if (ti != 0) try w.writeAll(",\n");
        try w.print("    {{\"base\": \"{s}\", \"halves\": {d}, \"bits\": \"{s}\"}}", .{ t.base, t.halves, bitsText(t.halves, try a.alloc(u8, 8)) });
    }
    const sha_hex = std.fmt.bytesToHex(sha, .lower);
    try w.print("\n  ],\n  \"plan_sha\": \"{s}\"\n}}\n", .{sha_hex[0..]});
    const text = try aw.toOwnedSlice();
    var atomic = try Io.Dir.cwd().createFileAtomic(io, out_path, .{ .make_path = true, .replace = true });
    defer atomic.deinit(io);
    try atomic.file.writePositionalAll(io, text, 0);
    try atomic.replace(io);

    // self-verify the written bytes: re-parse, re-hash; a target that lost the JSON
    // round trip stops here, not at the driver's plan gate
    const back_raw = try Io.Dir.cwd().readFileAlloc(io, out_path, a, .limited(1 << 24));
    const back = try std.json.parseFromSliceLeaky(std.json.Value, a, back_raw, .{ .ignore_unknown_fields = true });
    if (back != .object) return error.RoundTrip;
    var bp: std.ArrayList(alloc_mod.PartWidth) = .empty;
    for (back.object.get("parts").?.array.items) |item| {
        try bp.append(a, .{ .base = item.object.get("base").?.string, .halves = @intCast(item.object.get("halves").?.integer) });
    }
    const p2 = alloc_mod.Plan{
        .target_bpw = switch (back.object.get("target_bpw").?) {
            .float => |f| f,
            .integer => |ii| @floatFromInt(ii),
            else => return error.RoundTrip,
        },
        .head_halves = @intCast(back.object.get("head_halves").?.integer),
        .method = back.object.get("method").?.string,
        .parts = bp.items,
        .plan_sha = undefined,
    };
    const sha2 = try alloc_mod.planShaWith(a, p2);
    if (!std.mem.eql(u8, &sha, &sha2)) {
        std.debug.print("mixed-plan: written file re-hashes to {s}, tool hashed {s} — the JSON did not round trip\n", .{ std.fmt.bytesToHex(sha2, .lower), std.fmt.bytesToHex(sha, .lower) });
        return error.RoundTrip;
    }
    // and the allocator's own reader must take it too (head_bits string, parts, sha)
    const rp = try alloc_mod.planFromJson(a, back_raw);
    if (rp.parts.len != parts.items.len) return error.PartsMismatch;
    for (rp.parts, parts.items) |x, y| {
        if (!std.mem.eql(u8, x.base, y.base) or x.halves != y.halves) return error.PartsMismatch;
    }
    const sha_rp = try alloc_mod.planShaWith(a, rp);
    if (!std.mem.eql(u8, &sha, &sha_rp)) return error.PartsMismatch;
    std.debug.print("mixed-plan: {s}: {d} parts, layers {d}..{d} at {d} halves, rest at {d}, head {d}; target {d} bpw, sha {s}\n", .{
        out_path, parts.items.len, la, lb, odd_h, body_h, head_h, target, sha_hex[0..],
    });
}
