//! Qwen3.8-27B weights onto the device: int4 projections (MLX affine 4/64 or GPTQ 4/128) and the layer table.
const std = @import("std");
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
const fwd = @import("forward.zig");
const exl3_model = @import("exl3_model.zig");
const Checkpoint = dev.Checkpoint;
const D = q.D;
const Dev = dev.Dev;
const GDN_IN = q.GDN_IN;
const LAYERS = q.LAYERS;
const STASH = q.STASH;
const alloc = q.alloc;
const isAttention = q.isAttention;
const kvBytes = fwd.kvBytes;
const off = dev.off;

/// An int4 projection in the GEMM layout (words, fp16 scales, biases); `wide` picks the kernel family, `ilv` gate|up.
/// `dense16` (the calibration driver, eaq-ldlq-spec-fable.md 1.4): the Proj is a placeholder whose
/// weights are the fp16 [K, n] slab itself (already rotation- and scale-baked, `n` the NT pitch);
/// gemmRows runs the prompt GEMM over the slab directly. null on every serving path.
pub const Proj = struct { w: *anyopaque, s: *anyopaque, bg: *anyopaque, n: u32, k: u32, wide: bool, gs: u32 = 64, ilv: bool = false, exl3: ?*const exl3_model.Exl3 = null, dense16: ?*anyopaque = null };

pub const Layer = struct {
    in_norm: *anyopaque,
    post_norm: *anyopaque,
    attn: bool,
    mixer_in: Proj, // gdn.in or attn.qkv
    mixer_out: Proj, // out_proj or o_proj
    gu: Proj,
    down: Proj,
    // GDN
    ab: ?*anyopaque = null, // GPTQ: the a and b gates, fp16 [96, 5120] (dense, as stored), into P columns 16 384 …
    conv_w: ?*anyopaque = null,
    alog: ?*anyopaque = null,
    dt: ?*anyopaque = null,
    gnorm: ?*anyopaque = null,
    rec: ?*anyopaque = null, // fp32 [48, 128, 128]
    conv: [2]?*anyopaque = .{ null, null }, // fp16 [3, 10240], ping-pong
    conv_cur: u1 = 0,
    // a verified block's GDN inputs and its entry state, to replay the kept rows (Runner.commit)
    rec_save: ?*anyopaque = null,
    stash: ?*anyopaque = null, // per-row stash: q, k, v, g, β and the qkv rows
    // attention
    qn: ?*anyopaque = null,
    kn: ?*anyopaque = null,
    kc: ?*anyopaque = null, // bf16 [MAX_CTX, 4, 256]
    vc: ?*anyopaque = null,
    // rec, conv, conv_cur, kc and vc above are the selected slot's (Runner.select); each slot's own are kept here
    slot_rec: [MAX_SLOTS]?*anyopaque = @splat(null),
    slot_save: [MAX_SLOTS]?*anyopaque = @splat(null),
    slot_conv: [MAX_SLOTS][2]?*anyopaque = @splat(.{ null, null }),
    slot_cur: [MAX_SLOTS]u1 = @splat(0),
    slot_kc: [MAX_SLOTS]?*anyopaque = @splat(null),
    slot_vc: [MAX_SLOTS]?*anyopaque = @splat(null),
};

/// Sequences a runner holds at most (one a stream of the lane host).
pub const MAX_SLOTS = 16;

pub const Model = struct {
    embed_w: *anyopaque, // MLX: words [V, D/8] ; GPTQ: the fp16 table [V, D]
    embed_s: ?*anyopaque = null, // MLX only
    embed_b: ?*anyopaque = null,
    layers: [LAYERS]Layer,
    norm: *anyopaque,
    head: Proj,
    invf: *anyopaque,
};

pub fn bf16Tensor(d: *Dev, c: *const Checkpoint, name: []const u8) !*anyopaque {
    const t = try c.get(name);
    if (!std.mem.eql(u8, t.dtype, "BF16")) return error.UnexpectedDtype;
    return d.upload(t.bytes);
}

/// Fused projection from MLX parts stacked on N, repacked on the GPU through a raw staging area.
pub fn projection(d: *Dev, c: *const Checkpoint, prefix: []const u8, parts: []const []const u8, wide: bool, ilv: bool, stage: *anyopaque) !Proj {
    var n_total: u32 = 0;
    var k: u32 = 0;
    for (parts) |p| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.weight", .{ prefix, p }));
        n_total += @intCast(w.shape[0]);
        k = @intCast(w.shape[1] * 8);
    }
    // N padded to the GEMM tile: the padding columns have zero scales and biases, so they come out as zeros.
    const n_real = n_total;
    n_total = (n_total + 255) / 256 * 256;
    const out: Proj = .{ .w = try d.alloc_(@as(usize, k / 8) * n_total * 4), .s = try d.alloc_(@as(usize, k / 64) * n_total * 2), .bg = try d.alloc_(@as(usize, k / 64) * n_total * 2), .n = n_total, .k = k, .wide = wide, .ilv = ilv };
    if (ilv and (parts.len != 2 or n_real != n_total)) return error.BadInterleave;
    if (n_real != n_total) {
        try d.fill(out.s, @as(u16, 0), @as(usize, k / 64) * n_total * 2);
        try d.fill(out.bg, @as(u16, 0), @as(usize, k / 64) * n_total * 2);
    }
    var n0: u32 = 0;
    for (parts, 0..) |p, pi| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.weight", .{ prefix, p }));
        const s = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.scales", .{ prefix, p }));
        const b = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.biases", .{ prefix, p }));
        const n: u32 = @intCast(w.shape[0]);
        const sw = stage;
        const ss = off(stage, w.bytes.len);
        const sb = off(ss, s.bytes.len);
        try d.copyIn(sw, w.bytes);
        try d.copyIn(ss, s.bytes);
        try d.copyIn(sb, b.bytes);
        try d.launch(d.k.repack, .{ .x = (n + 15) / 16, .y = k / 8 }, .{ sw, ss, sb, out.w, out.s, out.bg, @as(i32, @intCast(n)), @as(i32, @intCast(k)), @as(i32, @intCast(n_total)), @as(i32, @intCast(n0)), @as(i32, if (ilv) @intCast(pi + 1) else 0) });
        try d.sync();
        n0 += n;
    }
    return out;
}

/// Fused projection from GPTQ symmetric 4-bit parts stacked on N, at least `min_n` columns (the extra ones zero).
pub fn projectionGptq(d: *Dev, c: *const Checkpoint, prefix: []const u8, parts: []const []const u8, wide: bool, min_n: u32, ilv: bool, stage: *anyopaque) !Proj {
    var n_total: u32 = 0;
    var k: u32 = 0;
    for (parts) |p| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.qweight", .{ prefix, p }));
        n_total += @intCast(w.shape[1]);
        k = @intCast(w.shape[0] * 8);
    }
    const n_real = n_total;
    n_total = (@max(n_total, min_n) + 255) / 256 * 256;
    const g = q.GS;
    const out: Proj = .{ .w = try d.alloc_(@as(usize, k / 8) * n_total * 4), .s = try d.alloc_(@as(usize, k / g) * n_total * 2), .bg = try d.alloc_(@as(usize, k / g) * n_total * 2), .n = n_total, .k = k, .wide = wide, .gs = g, .ilv = ilv };
    if (ilv and (parts.len != 2 or n_real != n_total)) return error.BadInterleave;
    if (n_real != n_total) {
        try d.fill(out.w, @as(u32, 0), @as(usize, k / 8) * n_total * 4);
        try d.fill(out.s, @as(u16, 0), @as(usize, k / g) * n_total * 2);
        try d.fill(out.bg, @as(u16, 0), @as(usize, k / g) * n_total * 2);
    }
    var n0: u32 = 0;
    for (parts, 0..) |p, pi| {
        const w = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.qweight", .{ prefix, p }));
        const sc = try c.get(try std.fmt.allocPrint(alloc, "{s}{s}.scales", .{ prefix, p }));
        if (!std.mem.eql(u8, sc.dtype, "F16") or sc.shape[0] * g != k) return error.UnexpectedGptqLayout;
        const n: u32 = @intCast(w.shape[1]);
        const ss = off(stage, w.bytes.len);
        try d.copyIn(stage, w.bytes);
        try d.copyIn(ss, sc.bytes);
        try d.launch(d.k.repack_gptq, .{ .x = (n + 15) / 16, .y = k / 8 }, .{ stage, ss, out.w, out.s, out.bg, @as(i32, @intCast(n)), @as(i32, @intCast(k)), @as(i32, @intCast(n_total)), @as(i32, @intCast(n0)), @as(i32, @intCast(g)), @as(i32, if (ilv) @intCast(pi + 1) else 0) });
        try d.sync();
        n0 += n;
    }
    return out;
}

/// An fp16 [N, K] weight quantized at load to MLX affine 4/64 in the GEMM layout, in row blocks via staging.
pub fn projectionQuantized(d: *Dev, c: *const Checkpoint, name: []const u8, wide: bool, stage: *anyopaque) !Proj {
    const t = try c.get(name);
    if (!std.mem.eql(u8, t.dtype, "F16")) return error.UnexpectedDtype;
    const n: u32 = @intCast(t.shape[0]);
    const k: u32 = @intCast(t.shape[1]);
    if (n % 256 != 0) return error.UnexpectedShape;
    const out: Proj = .{ .w = try d.alloc_(@as(usize, k / 8) * n * 4), .s = try d.alloc_(@as(usize, k / 64) * n * 2), .bg = try d.alloc_(@as(usize, k / 64) * n * 2), .n = n, .k = k, .wide = wide };
    const rows: u32 = 32768;
    var n0: u32 = 0;
    while (n0 < n) : (n0 += rows) {
        const r = @min(rows, n - n0);
        const raw = stage;
        const mw = off(raw, @as(usize, r) * k * 2);
        const ms = off(mw, @as(usize, r) * (k / 8) * 4);
        const mb = off(ms, @as(usize, r) * (k / 64) * 2);
        try d.copyIn(raw, t.bytes[@as(usize, n0) * k * 2 ..][0 .. @as(usize, r) * k * 2]);
        try d.launch(d.k.quantize_mlx, .{ .x = (r + 15) / 16, .y = k / 64 }, .{ raw, @as(i32, @intCast(r)), @as(i32, @intCast(k)), mw, ms, mb });
        try d.launch(d.k.repack, .{ .x = (r + 15) / 16, .y = k / 8 }, .{ mw, ms, mb, out.w, out.s, out.bg, @as(i32, @intCast(r)), @as(i32, @intCast(k)), @as(i32, @intCast(n)), @as(i32, @intCast(n0)), @as(i32, 0) });
        try d.sync();
    }
    return out;
}

/// A small weight to bf16 on the device; `centred`: a GPTQ RMSNorm weight stored as gamma − 1 gets its 1 back.
pub fn vecTensor(d: *Dev, c: *const Checkpoint, name: []const u8, centred: bool) !*anyopaque {
    const t = try c.get(name);
    if (std.mem.eql(u8, t.dtype, "BF16")) return d.upload(t.bytes);
    if (!std.mem.eql(u8, t.dtype, "F16")) return error.UnexpectedDtype;
    const n = t.bytes.len / 2;
    const out = try alloc.alloc(u16, n);
    defer alloc.free(out);
    for (out, 0..) |*o, i| {
        const h: f16 = @bitCast(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little));
        var v: f32 = h;
        if (centred) v += 1.0;
        const u: u32 = @bitCast(v);
        o.* = @intCast((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
    }
    return d.upload(std.mem.sliceAsBytes(out));
}

/// Qwen3.8-27B from a symmetric 4-bit GPTQ checkpoint (g128): int4 projections, fp16 embedding, head quantized to 4/64.
pub fn loadGptq(d: *Dev, c: *const Checkpoint) !Model {
    var m: Model = undefined;
    const stage = try d.alloc_(800 << 20);
    const e = try c.get("model.embed_tokens.weight");
    if (!std.mem.eql(u8, e.dtype, "F16")) return error.UnexpectedDtype;
    m.embed_w = try d.alloc_(e.bytes.len);
    try d.copyIn(m.embed_w, e.bytes);
    m.embed_s = null;
    m.embed_b = null;
    m.norm = try vecTensor(d, c, "model.norm.weight", true);
    m.head = try projectionQuantized(d, c, "lm_head.weight", true, stage);
    var invf: [32]f32 = undefined;
    for (&invf, 0..) |*f, i| f.* = @floatCast(std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(i)) / 32.0));
    m.invf = try d.upload(std.mem.sliceAsBytes(&invf));
    for (&m.layers, 0..) |*l, i| {
        const p = try std.fmt.allocPrint(alloc, "model.layers.{d}.", .{i});
        const f = struct {
            fn n(pp: []const u8, sx: []const u8) []const u8 {
                return std.fmt.allocPrint(alloc, "{s}{s}", .{ pp, sx }) catch unreachable;
            }
        }.n;
        l.* = .{ .in_norm = try vecTensor(d, c, f(p, "input_layernorm.weight"), true), .post_norm = try vecTensor(d, c, f(p, "post_attention_layernorm.weight"), true), .attn = isAttention(i), .mixer_in = undefined, .mixer_out = undefined, .gu = try projectionGptq(d, c, f(p, "mlp."), &.{ "gate_proj", "up_proj" }, true, 0, true, stage), .down = try projectionGptq(d, c, f(p, "mlp."), &.{"down_proj"}, false, 0, false, stage) };
        if (l.attn) {
            l.mixer_in = try projectionGptq(d, c, f(p, "self_attn."), &.{ "q_proj", "k_proj", "v_proj" }, true, 0, false, stage);
            l.mixer_out = try projectionGptq(d, c, f(p, "self_attn."), &.{"o_proj"}, false, 0, false, stage);
            l.qn = try vecTensor(d, c, f(p, "self_attn.q_norm.weight"), true);
            l.kn = try vecTensor(d, c, f(p, "self_attn.k_norm.weight"), true);
            l.kc = try d.alloc_(kvBytes());
            l.vc = try d.alloc_(kvBytes());
        } else {
            l.mixer_in = try projectionGptq(d, c, f(p, "linear_attn."), &.{ "in_proj_qkv", "in_proj_z" }, true, GDN_IN, false, stage);
            l.mixer_out = try projectionGptq(d, c, f(p, "linear_attn."), &.{"out_proj"}, false, 0, false, stage);
            const ta = try c.get(f(p, "linear_attn.in_proj_a.weight"));
            const tb = try c.get(f(p, "linear_attn.in_proj_b.weight"));
            if (!std.mem.eql(u8, ta.dtype, "F16") or ta.bytes.len != 48 * D * 2 or tb.bytes.len != 48 * D * 2) return error.UnexpectedShape;
            l.ab = try d.alloc_(96 * D * 2);
            try d.copyIn(l.ab.?, ta.bytes);
            try d.copyIn(off(l.ab.?, 48 * D * 2), tb.bytes);
            l.conv_w = try vecTensor(d, c, f(p, "linear_attn.conv1d.weight"), false);
            l.alog = try vecTensor(d, c, f(p, "linear_attn.A_log"), false);
            l.dt = try vecTensor(d, c, f(p, "linear_attn.dt_bias"), false);
            l.gnorm = try vecTensor(d, c, f(p, "linear_attn.norm.weight"), false);
            l.rec = try d.alloc_(48 * 128 * 128 * 4);
            l.rec_save = try d.alloc_(48 * 128 * 128 * 4);
            l.stash = try d.alloc_(STASH);
            try d.fill(l.rec.?, @as(u32, 0), 48 * 128 * 128 * 4);
            for (&l.conv) |*cv| {
                cv.* = try d.alloc_(3 * 10240 * 2);
                try d.fill(cv.*.?, @as(u16, 0), 3 * 10240 * 2);
            }
        }
        if (i % 8 == 7) std.debug.print("  layers loaded: {d}/{d}\n", .{ i + 1, LAYERS });
    }
    try d.sync();
    d.g.free(stage);
    d.allocated -= 800 << 20; // the staging area is gone
    return m;
}

/// A stock EXL3 pack: an admitted trellis group and the mul1 codebook marker, by name.
fn isExl3(c: *const Checkpoint) bool {
    return c.has("gate_proj.trellis") and c.has("mul1");
}

/// The fused projection parts of one layer, in load order — the twin of exl3_eaq.zig:112-117
/// without the GDN a and b gates: PR 4 serves those dense F16 and refuses them as trellis.
fn exl3SlotParts(i: usize) struct { mixer_in: []const []const u8, mixer_out: []const []const u8 } {
    return if (isAttention(i))
        .{ .mixer_in = &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" }, .mixer_out = &.{"self_attn.o_proj"} }
    else
        .{ .mixer_in = &.{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z" }, .mixer_out = &.{"linear_attn.out_proj"} };
}

/// A stored tensor name without the checkpoint's HF wrapper, so it compares against the
/// loader's own "model.layers."/"lm_head" prefixes (Checkpoint.get admits the same two).
fn canonExl3(name: []const u8) []const u8 {
    if (std.mem.startsWith(u8, name, "model.language_model.")) return name["model.language_".len..];
    if (std.mem.startsWith(u8, name, "language_model.")) return name["language_".len..];
    return name;
}

/// One admitted fused part: the stored prefix and its admitted group (loadExl3's admission table).
pub const Entry = struct { prefix: []const u8, g: exl3_model.Group, seen: usize = 0 };

/// The verdict of loadExl3's admission: the admitted parts, the census summary and whether
/// the pack ships lm_head.trellis.
pub const Admission = struct { entries: []Entry, summary: exl3_model.Summary, head_g: ?exl3_model.Group };

/// The host-side admission loadExl3 runs before its first device allocation: the
/// Table.complete mirror, the dense F16 gate rule, the head and the full .trellis census.
/// publish() re-runs it on the finished pack — writer and reader share one source of truth.
pub fn admitExl3(c: *const Checkpoint) !Admission {
    // ---- admission, no allocation: every layer's four slots, the head and the embedding
    // present before upload (the Table.complete mirror, exl3_eaq.zig:381-388 with the
    // countSlot rule); the first missing or unservable entry names itself and refuses ----
    var entries: std.ArrayList(Entry) = .empty;
    var summary: exl3_model.Summary = .{};
    var head_g: ?exl3_model.Group = null;
    for (0..LAYERS) |i| {
        const p = try std.fmt.allocPrint(alloc, "model.layers.{d}.", .{i});
        const f = struct {
            fn n(pp: []const u8, sx: []const u8) []const u8 {
                return std.fmt.allocPrint(alloc, "{s}{s}", .{ pp, sx }) catch unreachable;
            }
        }.n;
        const slots = exl3SlotParts(i);
        const fused = [_]struct { slot: []const []const u8 }{
            .{ .slot = slots.mixer_in },
            .{ .slot = slots.mixer_out },
            .{ .slot = &.{ "mlp.gate_proj", "mlp.up_proj" } },
            .{ .slot = &.{"mlp.down_proj"} },
        };
        for (fused, 0..) |sd, si| for (sd.slot, 0..) |part_name, pi| {
            const prefix = f(p, part_name);
            const g = exl3_model.admitGroup(c, prefix) catch |err| {
                exl3_model.refuse("layer {d} slot {d} part {d}: {s}", .{ i, si, pi, @errorName(err) });
                return err;
            };
            try entries.append(alloc, .{ .prefix = prefix, .g = g });
            summary.groups += 1;
            summary.mul1 += 1; // admitGroup refused every group that was not mul1
            switch (g.style) { .f16, .f16_16 => summary.suh_svh += 1, .words => summary.su_sv += 1 }
            summary.bits[g.halves] += 1; // bias stays 0: a group carrying one refused
        };
        if (!isAttention(i)) {
            // The GDN gates dense fp16 only: N = 48 straddles a 128 dequant block
            // (exl3.cl:259-262), and no gate pack stores them trellis.
            for ([_][]const u8{ "linear_attn.in_proj_a", "linear_attn.in_proj_b" }) |gate| {
                if (c.has(try std.fmt.allocPrint(alloc, "{s}.trellis", .{f(p, gate)}))) {
                    exl3_model.refuse("layer {d} {s}: the gate must be dense F16 48 x {d}", .{ i, gate, D });
                    return error.Exl3NotDense;
                }
            }
            const ta = try c.get(f(p, "linear_attn.in_proj_a.weight"));
            const tb = try c.get(f(p, "linear_attn.in_proj_b.weight"));
            if (!std.mem.eql(u8, ta.dtype, "F16") or ta.bytes.len != 48 * D * 2 or tb.bytes.len != 48 * D * 2) {
                exl3_model.refuse("layer {d}: the a/b gates are not dense F16 48 x {d}", .{ i, D });
                return error.Exl3NotDense;
            }
        }
    }
    // The embedding arrives F16 as stored, served like the seed GPTQ path (contract 6.6).
    if (c.has("embed_tokens.trellis")) {
        exl3_model.refuse("model.embed_tokens is trellis; PR 4 serves the embedding dense F16 only", .{});
        return error.Exl3NotDense;
    }
    if (!c.has("model.embed_tokens.weight")) {
        exl3_model.refuse("no model.embed_tokens.weight: the pack is incomplete", .{});
        return error.Exl3Incomplete;
    }
    // The head: trellis at stored width (the fp16 GEMV path, contract section 3), or a pack
    // that ships F16 lm_head.weight gets the seed head's own route below.
    // (the embedding uploads after the census below: every refusal lands before the first
    // device allocation, so a refused pack leaves nothing to free)
    if (c.has("lm_head.trellis")) {
        const g = exl3_model.admitGroup(c, "lm_head") catch |err| {
            exl3_model.refuse("lm_head: {s}", .{@errorName(err)});
            return err;
        };
        if (g.k != D or g.n != q.VOCAB) {
            exl3_model.refuse("lm_head: {d} x {d}, expected {d} x {d}", .{ g.k, g.n, D, q.VOCAB });
            return error.UnexpectedShape;
        }
        head_g = g;
        summary.groups += 1;
        summary.mul1 += 1;
        switch (g.style) { .f16, .f16_16 => summary.suh_svh += 1, .words => summary.su_sv += 1 }
        summary.bits[g.halves] += 1;
    } else if (c.has("lm_head.weight")) {
        const t = try c.get("lm_head.weight");
        if (!std.mem.eql(u8, t.dtype, "F16")) {
            exl3_model.refuse("lm_head.weight is {s}, not F16", .{t.dtype});
            return error.UnexpectedDtype;
        }
    } else {
        exl3_model.refuse("no lm_head: the pack is incomplete", .{});
        return error.Exl3Incomplete;
    }
    // The census: loadExl3 serves every .trellis tensor of the shards or refuses the pack
    // rather than half-serving it (the stock reader counts these as bad, exl3_load.zig:66).
    var head_seen: usize = 0;
    for (c.tensors.items) |t| {
        if (!std.mem.endsWith(u8, t.name, ".trellis")) continue;
        const prefix = canonExl3(t.name[0 .. t.name.len - ".trellis".len]);
        // other trees in a multimodal pack (the vision tower) are not served by this loader
        if (!std.mem.startsWith(u8, prefix, "model.layers.") and !std.mem.eql(u8, prefix, "lm_head")) continue;
        var hit = false;
        for (entries.items) |*en| if (std.mem.eql(u8, en.prefix, prefix)) {
            en.seen += 1;
            hit = true;
            break;
        };
        if (!hit and head_g != null and std.mem.eql(u8, prefix, "lm_head")) {
            head_seen += 1;
            hit = true;
        }
        if (!hit) {
            exl3_model.refuse("{s}: a trellis group the loader does not admit", .{t.name});
            return error.UnknownProjection;
        }
    }
    for (entries.items) |en| if (en.seen != 1) {
        exl3_model.refuse("{s}: {d} trellis tensors claim it", .{ en.prefix, en.seen });
        return error.DuplicateProjection;
    };
    if (head_seen > 1) {
        exl3_model.refuse("lm_head claimed by {d} trellis tensors", .{head_seen});
        return error.DuplicateProjection;
    }
    return .{ .entries = entries.items, .summary = summary, .head_g = head_g };
}

/// Qwen3.8-27B from a stock EXL3 checkpoint (mul1 codebook, whole bits): every projection a
/// trellis handle (Proj.exl3, buffers built by exl3_model.projectionExl3), the embedding and
/// the GDN a/b gates dense F16, and the head on its own path — trellis at stored width when
/// the pack ships lm_head.trellis, else the seed head's own 4/64 quantization. Every refusal
/// (the Table.complete mirror below and the exl3_model.admitGroup checks) lands before the
/// first device allocation; only after them does the GPTQ path's alloc-and-upload pattern run.
pub fn loadExl3(d: *Dev, c: *const Checkpoint) !Model {
    var m: Model = undefined;
    const adm = try admitExl3(c);
    const summary = adm.summary;
    const head_g = adm.head_g;
    // ---- upload: the admitted shapes become device buffers (contract section 2) ----
    // The embedding: F16 as stored, or widened host-side from BF16 (the EXL3 packs store BF16;
    // embed_dense reads F16). An oddly placed shard is copied to an aligned host buffer first.
    const e = try c.get("model.embed_tokens.weight");
    m.embed_w = try d.alloc_(e.bytes.len);
    if (std.mem.eql(u8, e.dtype, "F16")) {
        try d.copyIn(m.embed_w, e.bytes);
    } else if (std.mem.eql(u8, e.dtype, "BF16")) {
        const n = e.bytes.len / 2;
        var odd: ?[]u8 = null;
        defer if (odd) |b| alloc.free(b);
        if (@intFromPtr(e.bytes.ptr) % @alignOf(u16) != 0) {
            const tmp = try alloc.alloc(u8, e.bytes.len);
            @memcpy(tmp, e.bytes);
            odd = tmp;
        }
        const base: [*]const u8 = if (odd) |b| b.ptr else e.bytes.ptr;
        const src = @as([*]const u16, @ptrCast(@alignCast(base)))[0..n];
        const out = try alloc.alloc(u16, n);
        defer alloc.free(out);
        for (src, 0..) |b, i| out[i] = @bitCast(@as(f16, @floatCast(@as(f32, @bitCast(@as(u32, b) << 16)))));
        try d.copyIn(m.embed_w, std.mem.sliceAsBytes(out));
    } else {
        exl3_model.refuse("model.embed_tokens.weight is {s}, not F16 or BF16", .{e.dtype});
        return error.UnexpectedDtype;
    }
    m.embed_s = null;
    m.embed_b = null;
    m.norm = try vecTensor(d, c, "model.norm.weight", false);
    if (head_g != null) {
        m.head = try exl3_model.projectionExl3(d, c, "lm_head", &.{""}, true, 0, true);
    } else {
        // a dense F16 lm_head: the seed head's own route, staging and all (load.zig:198)
        const stage = try d.alloc_(800 << 20);
        m.head = try projectionQuantized(d, c, "lm_head.weight", true, stage);
        d.g.free(stage);
        d.allocated -= 800 << 20; // the staging area is gone
    }
    var invf: [32]f32 = undefined;
    for (&invf, 0..) |*f, i| f.* = @floatCast(std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(i)) / 32.0));
    m.invf = try d.upload(std.mem.sliceAsBytes(&invf));
    for (&m.layers, 0..) |*l, i| {
        const p = try std.fmt.allocPrint(alloc, "model.layers.{d}.", .{i});
        const f = struct {
            fn n(pp: []const u8, sx: []const u8) []const u8 {
                return std.fmt.allocPrint(alloc, "{s}{s}", .{ pp, sx }) catch unreachable;
            }
        }.n;
        l.* = .{ .in_norm = try vecTensor(d, c, f(p, "input_layernorm.weight"), false), .post_norm = try vecTensor(d, c, f(p, "post_attention_layernorm.weight"), false), .attn = isAttention(i), .mixer_in = undefined, .mixer_out = undefined, .gu = try exl3_model.projectionExl3(d, c, f(p, "mlp."), &.{ "gate_proj", "up_proj" }, true, 0, false), .down = try exl3_model.projectionExl3(d, c, f(p, "mlp."), &.{"down_proj"}, false, 0, false) };
        if (l.attn) {
            l.mixer_in = try exl3_model.projectionExl3(d, c, f(p, "self_attn."), &.{ "q_proj", "k_proj", "v_proj" }, true, 0, false);
            l.mixer_out = try exl3_model.projectionExl3(d, c, f(p, "self_attn."), &.{"o_proj"}, false, 0, false);
            l.qn = try vecTensor(d, c, f(p, "self_attn.q_norm.weight"), false);
            l.kn = try vecTensor(d, c, f(p, "self_attn.k_norm.weight"), false);
            l.kc = try d.alloc_(kvBytes());
            l.vc = try d.alloc_(kvBytes());
        } else {
            l.mixer_in = try exl3_model.projectionExl3(d, c, f(p, "linear_attn."), &.{ "in_proj_qkv", "in_proj_z" }, true, GDN_IN, false);
            l.mixer_out = try exl3_model.projectionExl3(d, c, f(p, "linear_attn."), &.{"out_proj"}, false, 0, false);
            const ta = try c.get(f(p, "linear_attn.in_proj_a.weight"));
            const tb = try c.get(f(p, "linear_attn.in_proj_b.weight"));
            if (!std.mem.eql(u8, ta.dtype, "F16") or ta.bytes.len != 48 * D * 2 or tb.bytes.len != 48 * D * 2) return error.UnexpectedShape;
            l.ab = try d.alloc_(96 * D * 2);
            try d.copyIn(l.ab.?, ta.bytes);
            try d.copyIn(off(l.ab.?, 48 * D * 2), tb.bytes);
            l.conv_w = try vecTensor(d, c, f(p, "linear_attn.conv1d.weight"), false);
            l.alog = try vecTensor(d, c, f(p, "linear_attn.A_log"), false);
            l.dt = try vecTensor(d, c, f(p, "linear_attn.dt_bias"), false);
            l.gnorm = try vecTensor(d, c, f(p, "linear_attn.norm.weight"), false);
            l.rec = try d.alloc_(48 * 128 * 128 * 4);
            l.rec_save = try d.alloc_(48 * 128 * 128 * 4);
            l.stash = try d.alloc_(STASH);
            try d.fill(l.rec.?, @as(u32, 0), 48 * 128 * 128 * 4);
            for (&l.conv) |*cv| {
                cv.* = try d.alloc_(3 * 10240 * 2);
                try d.fill(cv.*.?, @as(u16, 0), 3 * 10240 * 2);
            }
        }
        if (i % 8 == 7) std.debug.print("  layers loaded: {d}/{d}\n", .{ i + 1, LAYERS });
    }
    try d.sync();
    // the admission table (the Summary shape of core/exl3_load.zig:27-49): every figure from
    // admitted shapes, not from model output.
    var tr_bytes: usize = 0;
    var sc_bytes: usize = 0;
    for (adm.entries) |en| {
        const bytes = exl3_model.partBytes(en.g);
        tr_bytes += bytes.tr;
        sc_bytes += bytes.scales;
    }
    if (head_g) |hg| {
        const bytes = exl3_model.partBytes(hg);
        tr_bytes += bytes.tr;
        sc_bytes += bytes.scales;
    }
    std.debug.print("  exl3: {d} groups admitted (mul1 {d}, inst3 {d}, mcg {d}): scales suh/svh {d}, su/sv {d}, bias {d}\n", .{ summary.groups, summary.mul1, summary.inst3, summary.mcg, summary.suh_svh, summary.su_sv, summary.bias });
    std.debug.print("  bits:", .{}); // the histogram keys halves; print the bits each index means (2.5b style, one decimal)
    for (summary.bits, 0..) |n, h| if (n != 0) std.debug.print(" {d}.{d}b:{d}", .{ h / 2, (h % 2) * 5, n });
    std.debug.print("\n        dense: model.embed_tokens F16, linear_attn.in_proj_a/b F16 48x{d}, lm_head {s}\n", .{ D, if (head_g != null) "trellis (fp16 GEMV path)" else "F16 (quantized to 4/64 at load)" });
    std.debug.print("        device bytes: trellis {d}, scales {d}\n", .{ tr_bytes, sc_bytes });
    return m;
}

pub fn load(d: *Dev, c: *const Checkpoint) !Model {
    if (c.has("in_proj_qkv.qweight")) {
        q.GS = 128;
        return loadGptq(d, c);
    }
    if (isExl3(c)) {
        q.GS = 64;
        return loadExl3(d, c); // stock EXL3 pack: the dispatch's home, uploading via exl3_model
    }
    q.GS = 64;
    var m: Model = undefined;
    const stage = try d.alloc_(800 << 20);
    const e_w = try c.get("model.embed_tokens.weight");
    m.embed_w = try d.upload(e_w.bytes);
    m.embed_s = try bf16Tensor(d, c, "model.embed_tokens.scales");
    m.embed_b = try bf16Tensor(d, c, "model.embed_tokens.biases");
    m.norm = try bf16Tensor(d, c, "model.norm.weight");
    m.head = try projection(d, c, "lm_head", &.{""}, true, false, stage);
    var invf: [32]f32 = undefined;
    for (&invf, 0..) |*f, i| f.* = @floatCast(std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(i)) / 32.0));
    m.invf = try d.upload(std.mem.sliceAsBytes(&invf));
    for (&m.layers, 0..) |*l, i| {
        const p = try std.fmt.allocPrint(alloc, "model.layers.{d}.", .{i});
        const f = struct {
            fn n(pp: []const u8, s: []const u8) []const u8 {
                return std.fmt.allocPrint(alloc, "{s}{s}", .{ pp, s }) catch unreachable;
            }
        }.n;
        l.* = .{ .in_norm = try bf16Tensor(d, c, f(p, "input_layernorm.weight")), .post_norm = try bf16Tensor(d, c, f(p, "post_attention_layernorm.weight")), .attn = isAttention(i), .mixer_in = undefined, .mixer_out = undefined, .gu = try projection(d, c, f(p, "mlp."), &.{ "gate_proj", "up_proj" }, true, true, stage), .down = try projection(d, c, f(p, "mlp."), &.{"down_proj"}, false, false, stage) };
        if (l.attn) {
            l.mixer_in = try projection(d, c, f(p, "self_attn."), &.{ "q_proj", "k_proj", "v_proj" }, true, false, stage);
            l.mixer_out = try projection(d, c, f(p, "self_attn."), &.{"o_proj"}, false, false, stage);
            l.qn = try bf16Tensor(d, c, f(p, "self_attn.q_norm.weight"));
            l.kn = try bf16Tensor(d, c, f(p, "self_attn.k_norm.weight"));
            l.kc = try d.alloc_(kvBytes());
            l.vc = try d.alloc_(kvBytes());
        } else {
            l.mixer_in = try projection(d, c, f(p, "linear_attn."), &.{ "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b" }, true, false, stage);
            l.mixer_out = try projection(d, c, f(p, "linear_attn."), &.{"out_proj"}, false, false, stage);
            l.conv_w = try bf16Tensor(d, c, f(p, "linear_attn.conv1d.weight"));
            l.alog = try bf16Tensor(d, c, f(p, "linear_attn.A_log"));
            l.dt = try bf16Tensor(d, c, f(p, "linear_attn.dt_bias"));
            l.gnorm = try bf16Tensor(d, c, f(p, "linear_attn.norm.weight"));
            l.rec = try d.alloc_(48 * 128 * 128 * 4);
            l.rec_save = try d.alloc_(48 * 128 * 128 * 4);
            l.stash = try d.alloc_(STASH);
            try d.fill(l.rec.?, @as(u32, 0), 48 * 128 * 128 * 4);
            for (&l.conv) |*cv| {
                cv.* = try d.alloc_(3 * 10240 * 2);
                try d.fill(cv.*.?, @as(u16, 0), 3 * 10240 * 2);
            }
        }
        if (i % 8 == 7) std.debug.print("  layers loaded: {d}/{d}\n", .{ i + 1, LAYERS });
    }
    try d.sync();
    d.g.free(stage);
    d.allocated -= 800 << 20; // the staging area is gone
    return m;
}
