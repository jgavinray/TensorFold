//! The fresh-quant calibration entries (eaq-ldlq-spec-fable.md section 1.4): what the quantizer
//! drives the card through. One fp32 residual stream per calibration batch, layer-major: the
//! embedding (`embedStream`), 64 × `layerForward` over resident fp16 weight slabs (dense bf16
//! source widened for the capture pass, dequantized trellis for the apply pass), and `finalNorm`
//! (`Site.head`'s Hessian included). Each call is one layer over one row block: `forward`'s layer
//! loop body is `forward.layerBody` and so is this — one body, no drift (H1-H8 live here: GDN
//! `rec` / conv reset per row, the wide path only, the real T to the syrk never the pad `tp`,
//! and the H6 residual pairing that keeps the stream bit-identical to the serving forward —
//! T-CAP-1 is its proof).
//!
//! `Capture` itself is declared in `forward.zig` (its `Batch` consumes it; `calib.Capture` is the
//! driver-facing name) and re-exported below; its device launch (`forward.capture`, the
//! transpose16 + exl3_syrk32 queue) lives there beside the site hooks. Host finalize and damping
//! (section 2.4) and the quantizer driver are the next slices.
const std = @import("std");
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
const fwd = @import("forward.zig");
const load = @import("load.zig");
const exl3_model = @import("exl3_model.zig");
const Dev = dev.Dev;
const Model = load.Model;
const Proj = load.Proj;
const D = q.D;
const FF = q.FF;
const EPS = q.EPS;
const GDN_IN = q.GDN_IN;
const LAYERS = q.LAYERS;
const copy = dev.copy;
const off = dev.off;
const alloc = q.alloc;

/// The linear-input sites of the capture (`gemm_probe`'s five, spec 1.1): K is the site's input
/// width — mixer_in D, mixer_out 6144, gu D, down FF, head D — and the Hessian's side.
pub const Site = enum(u3) { mixer_in = 0, mixer_out = 1, gu = 2, down = 3, head = 4 };

/// fp16 [K, NT] resident weight slabs for one layer's four fused slots, NT-pitched like `wscr`:
/// dense weights for the capture pass, dequantized trellis for the apply pass. The buffers are
/// sized for the widest shapes (the GDN mixer_in pitch), reused layer over layer; a layer's
/// slots are filled by `dequantSlot` (trellis) or `uploadDense` (bf16 source) right before
/// `layerForward` runs that layer, and the GEMM is exl3_prompt16 over the slab directly.
pub const Slabs = struct {
    mixer_in: *anyopaque,
    mixer_out: *anyopaque,
    gu: *anyopaque,
    down: *anyopaque,

    /// Allocates and zeroes the four slabs (the zero covers every pad column the fused parts
    /// leave open — the mixer_in tile pads NT past the parts' end, and a recycled slab row can
    /// otherwise carry another slot's weights into the unread columns; the consumer ignores
    /// them, but zeros keep the reads defined).
    pub fn init(d: *Dev) !Slabs {
        return .{
            .mixer_in = try slabAlloc(d, D, GDN_IN), // 5120 x 16640 (the attention pitch 14336 fits under it)
            .mixer_out = try slabAlloc(d, 6144, D), // 6144 x 5120
            .gu = try slabAlloc(d, D, 2 * FF), // 5120 x 34816
            .down = try slabAlloc(d, FF, D), // 17408 x 5120
        };
    }

    pub fn deinit(s: *Slabs, d: *Dev) void {
        d.g.free(s.mixer_in);
        d.g.free(s.mixer_out);
        d.g.free(s.gu);
        d.g.free(s.down);
    }
};

fn slabAlloc(d: *Dev, K: usize, NT: usize) !*anyopaque {
    const p = try d.alloc_(K * NT * 2);
    try d.fill(p, @as(u16, 0), K * NT * 2);
    return p;
}

pub const Capture = fwd.Capture;

/// A capture for batches up to `tp_max` rows (rounded to 64; one row block at a time — the
/// driver sweeps layer-major and the syrk stages one batch's rows at a time). `h` starts null:
/// the driver points the sites it wants at its own zeroed fp32 [K_s, K_s] buffers. The staging
/// is zeroed once here: the syrk reads xpad rows to Tp directly (the per-batch hook zero-fills
/// T..Tp, but the first short batch must not read uninitialized memory), and xt gets its
/// T..Tp zeros from transpose16. The assert covers layerForward's fresh reset: it copies the
/// whole GDN state (3,145,728 B) from `acts.zeros`, which holds maxRows()*12288 bytes.
pub fn initCapture(d: *Dev, tp_max: u32) !Capture {
    if (tp_max < 64 or tp_max % 64 != 0) return error.CalibCaptureRows;
    if (@as(usize, q.maxRows()) * 48 * 128 * 2 < @as(usize, 48) * 128 * 128 * 4) return error.CalibZerosSmall; // XPU_CHUNK >= 128
    const c = Capture{
        .xpad = try d.alloc_(@as(usize, tp_max) * FF * 2),
        .xt = try d.alloc_(@as(usize, FF) * tp_max * 2),
        .tp_max = tp_max,
    };
    try d.fill(c.xpad, @as(u16, 0), @as(usize, tp_max) * FF * 2);
    try d.fill(c.xt, @as(u16, 0), @as(usize, FF) * tp_max * 2);
    return c;
}

pub fn deinitCapture(c: *Capture, d: *Dev) void {
    d.g.free(c.xpad);
    d.g.free(c.xt);
    c.* = .{ .xpad = undefined, .xt = undefined, .tp_max = 0 };
}

/// H7 (and the fidelity hooks): the layer math must be the plain serving math — the study
/// options quantize the captured activations in place or move the K/V and GDN paths, and an
/// armed fp64 reference replaces GEMMs and states host-side. The quantizer asserts them off
/// and refuses otherwise.
fn studyOff() !void {
    if (q.prefill_int8 or q.prefill_int4 or q.a8_group != 0 or q.kv8 != 0 or q.kv_i8 or q.rot_on != 0)
        return error.CalibStudyOn;
    if (fwd.gemm_hook != null or fwd.gemm_probe != null or fwd.gdn_probe != null)
        return error.CalibProbeArmed;
}

/// The stream's start: the embedding of one row block (embedRows, forward.zig), into the
/// caller's device fp32 stream at `x_out` (the runner stages `ids` into `acts.ids` first:
/// `Runner.calibEmbed`). T <= q.maxRows().
pub fn embedStream(d: *Dev, m: *Model, a: *fwd.Acts, ids: *anyopaque, x_out: *anyopaque, T: u32) !void {
    if (T == 0 or T > q.maxRows()) return error.TooManyRows;
    _ = a; // the stream needs no scratch from the shared acts; embedRows writes x_out direct
    try fwd.embedRows(d, m, ids, x_out, T);
}

/// Layer `i` over one row block: `x_in` fp32 [T, D] = the stream (x + r already added), the
/// state of layer i zeroed when `fresh` (every row starts from zero, capture and apply passes
/// alike — H1, H2, H3), `x_out` fp32 [T, D] = the stream after the layer (the driver may pass
/// the same buffer twice: the layer rewrites its rows in place). T <= q.maxRows(), `p0` the
/// row's position base (rows stay under the runner's context; chunks run in position order —
/// H5). Queues the capture syrk at every site with a non-null `cap.h` (§1.1's hooks, the real
/// T never the pad `tp` — H4). The four slot GEMMs run exl3_prompt16 over `slabs` (the dense
/// marker of spec 1.4: `Proj.dense16`, dispatched in gemmRows); the layer's small tensors
/// (norms, conv, a/b, qn/kn, KV) are the model's own, shared with serving — the whole-model
/// invariant of T-CAP-1. Wide always (H8: no stash, rec_save, multi or tree).
pub fn layerForward(d: *Dev, m: *Model, a: *fwd.Acts, i: usize, slabs: Slabs, x_in: *anyopaque, x_out: *anyopaque,
                    T: u32, p0: u32, fresh: bool, cap: ?*Capture) !void {
    try studyOff();
    if (i >= LAYERS) return error.CalibNoSuchLayer;
    if (T == 0 or T > q.maxRows() or T <= 16) return error.CalibRows; // wide only (H8)
    if (p0 + T > q.MAX_CTX) return error.CalibContext; // the KV rows the attention reads (H5)
    const l = &m.layers[i];
    if (fresh) {
        // H1: the GDN state enters each row at zero by a device-to-device copy from the acts
        // zeros (never d.fill: that stages on the host and syncs per call). H3: both conv
        // buffers zero and conv_cur back to 0, what Runner.reset does model-wide, per layer.
        const bytes = @as(usize, 48) * 128 * 128 * 4;
        if (l.rec) |rec| {
            try copy(d, rec, a.zeros, bytes);
            for (l.conv) |cv| try copy(d, cv.?, a.zeros, 3 * 10240 * 2);
            l.conv_cur = 0;
        }
        // an attention layer has nothing to zero: the KV cache is positional and each row
        // overwrites the positions it reads (H5).
    }
    const xbytes = @as(usize, T) * D * 4;
    try copy(d, a.x, x_in, xbytes);
    // the slab views (spec 1.4): placeholders with the dense marker, the pack's own n kept as
    // the NT pitch (exl3 Projs carry n = NT; ilv off so mlpIn takes the gemmRows path + swiglu,
    // the very path the trellis GEMMs take — serving parity, and T-CAP-1's).
    const mi = l.mixer_in;
    const mo = l.mixer_out;
    const gu = l.gu;
    const dn = l.down;
    l.mixer_in = slabProj(mi, slabs.mixer_in);
    l.mixer_out = slabProj(mo, slabs.mixer_out);
    l.gu = slabProj(gu, slabs.gu);
    l.down = slabProj(dn, slabs.down);
    const r = fwd.layerBody(d, m, a, l, i, .{ .T = T, .p0 = p0, .mode = .{ .prompt = true }, .first = true, .capture = cap });
    l.mixer_in = mi;
    l.mixer_out = mo;
    l.gu = gu;
    l.down = dn;
    try r;
    // H6, the residual pairing: the stream leaves this call fully summed. One add_rmsnorm
    // with flag 1 (the same kernel the next layer's first norm would have run, the same fused
    // x += r, the same fp32 add of the fp16 r) whose normalized output is discarded; then the
    // rows go back to the stream. Bit-identical to the forward's by construction — T-CAP-1.
    const gs: i32 = @intCast(q.GS);
    try d.launch(d.k.add_rmsnorm, .{ .x = T }, .{ a.x, a.r, l.in_norm, a.h, a.ahi, a.alo, @as(i32, D), EPS, @as(i32, 1), gs, fwd.no_q8, fwd.no_q8, @as(i32, @intCast(q.maxRows() + 32)), @as(i32, 0) });
    try copy(d, x_out, a.x, xbytes);
}

fn slabProj(p: Proj, slab: *anyopaque) Proj {
    var v = p;
    v.exl3 = null;
    v.ilv = false;
    v.dense16 = slab;
    return v;
}

/// The end: the final norm of the stream into fp16 hidden [T, D]; captures site 4 when
/// `cap.h[4]` is set. `x_in` is S_LAYERS — fully summed by layer LAYERS - 1's H6 tail — so the
/// serving head norm's add is already in: flag 0, the same kernel (the T-CAP-1 head-pass twin
/// of forward's norm before the head GEMM).
pub fn finalNorm(d: *Dev, m: *Model, a: *fwd.Acts, x_in: *anyopaque, hidden: *anyopaque, T: u32, cap: ?*Capture) !void {
    try studyOff();
    if (T == 0 or T > q.maxRows()) return error.TooManyRows;
    try copy(d, a.x, x_in, @as(usize, T) * D * 4);
    const hgs: i32 = @intCast(m.head.gs);
    try d.launch(d.k.add_rmsnorm, .{ .x = T }, .{ a.x, a.r, m.norm, a.h, a.ahi, a.alo, @as(i32, D), EPS, @as(i32, 0), hgs, fwd.no_q8, fwd.no_q8, @as(i32, 0), @as(i32, 0) });
    try copy(d, hidden, a.h, @as(usize, T) * D * 2);
    try fwd.capture(d, cap, 4, D, a.h, T); // the captured rows are the hidden rows
}

/// bf16 [N, K] as stored (HF out-major) -> fp16 [K, NT] slab columns [n0, n0 + N), pad columns
/// zero (bf16_to_slab). The count of source weights under 6.1e-5 (fp16's exactness floor for
/// the bf16 widening, spec 1.4) is logged per tensor.
pub fn uploadDense(d: *Dev, src_bf16: []const u8, N: u32, K: u32, NT: u32, n0: u32, slab: *anyopaque) !void {
    if (N == 0 or K == 0 or K % 16 != 0 or src_bf16.len != @as(usize, N) * K * 2) return error.CalibShape;
    if (@as(usize, n0) + N > NT or NT % 128 != 0 or n0 % 16 != 0) return error.CalibShape; // n0 % 16: an unaligned base makes Npad - n0 the kernel's own trap (it would return unwritten, silently leaving stale slab columns)
    var tiny: usize = 0;
    for (0 .. @as(usize, N) * K) |e| {
        const bits = @as(u16, src_bf16[2 * e]) | (@as(u16, src_bf16[2 * e + 1]) << 8);
        if (@abs(@as(f32, @bitCast(@as(u32, bits) << 16))) < 6.1e-5) tiny += 1;
    }
    if (tiny != 0) std.debug.print("calib: bf16_to_slab n0={d} N={d} K={d}: {d} source weights under 6.1e-5 (fp16 loses them)\n", .{ n0, N, K, tiny });
    const src = try stage(d, src_bf16.len); // the part's stored tensor, host -> device
    try d.copyIn(src, src_bf16);
    const npad: u32 = (@as(u32, n0) + N + 15) / 16 * 16 - n0; // the slab columns this call owns
    try d.launch(d.k.bf16_to_slab, .{ .x = @intCast(K / 16), .y = @intCast(npad / 16) }, .{
        src, slab, @as(i32, @intCast(N)), @as(i32, @intCast(K)), @as(i32, @intCast(NT)), @as(i32, @intCast(n0)), @as(i32, @intCast(npad)),
    });
}

/// A fused slot's trellis parts into its slab with the serving dequant (exl3_dequant128_c per
/// part, the rotation and the entry scales baked in, one launch per part at the part's column
/// base): the apply pass's slab, bit-identical to what the serving prompt GEMM reads from
/// `wscr`. `Slabs.init` zeroes the slab once, but the four slots differ in pitch (NT) and are
/// written independently, so a slot's columns beyond its own NT hold whatever the last writer
/// left, not guaranteed zeros from layer 0 on — the GEMM never reads columns past the slot's
/// own NT, so the stale values are inert. NT is the slot's Proj.n (the 256-padded N), shared
/// with the GEMM pitch.
pub fn dequantSlot(d: *Dev, hx: *const exl3_model.Exl3, K: u32, NT: u32, slab: *anyopaque) !void {
    if (hx.head or K % 128 != 0 or NT % 128 != 0) return error.CalibShape;
    for (hx.parts) |p| {
        if (p.dense) return error.CalibDensePart; // a dense (a/b-style) part belongs to uploadDense
        try d.launch(d.k.exl3_dequant128_c, .{ .x = @intCast(K / 128), .y = @intCast((p.n + 127) / 128) }, .{
            p.tr, p.suh, p.svh, slab,
            @as(i32, @intCast(K)), @as(i32, @intCast(p.n)), @as(i32, @intCast(NT)), @as(i32, p.halves), @as(i32, @intCast(p.n0)),
        });
    }
}

/// The dense upload's grow-only device staging (the same idiom as forward_exl3.xscratch: one
/// buffer for the process, one Runner owns the card at a time).
var stage_buf: ?*anyopaque = null;
var stage_bytes: usize = 0;

fn stage(d: *Dev, bytes: usize) !*anyopaque {
    if (bytes > stage_bytes) {
        if (stage_buf) |old| d.g.free(old);
        stage_buf = try d.alloc_(bytes);
        stage_bytes = bytes;
    }
    return stage_buf.?;
}

/// The layer-major sweep the quantizer drives, one batch of `ids` (host, T <= maxRows): the
/// embedding into `stream`, all layers with capture `cap` (slab fills are the driver's: see
/// the note on `Slabs`), and the final norm into `hidden`. Returns with `rows` accumulated per
/// site. `fresh_rows` must be true when the batch begins a row (it always does here — this is
/// the per-row entry of spec 1.2; a chunked row continues with layerForward's fresh = false).
pub fn captureRow(d: *Dev, m: *Model, a: *fwd.Acts, slabs: Slabs, ids: *anyopaque, stream: *anyopaque,
                  hidden: *anyopaque, T: u32, cap: ?*Capture) !void {
    try embedStream(d, m, a, ids, stream, T);
    for (0..LAYERS) |i| try layerForward(d, m, a, i, slabs, stream, stream, T, 0, true, cap);
    try finalNorm(d, m, a, stream, hidden, T, cap);
}

test "bf16_to_slab stores satisfy DST[k * NT + n0 + n] = (half) SRC[n * K + k]" {
    // The host mirror of calib.cl's substitution, so the fix fable's review found cannot regress
    // without a red test on any machine (the device round-trip T-SLAB in --calib-probe is the same
    // contract against the real kernel). N = 20 straddles a tile edge (one partial column tile),
    // n0 = 16 puts the part's columns off the slab base.
    const N: usize = 20;
    const K: usize = 32;
    const NT: usize = 128;
    const n0: usize = 16;
    const npad: usize = (n0 + N + 15) / 16 * 16 - n0; // the columns this upload owns, inside [n0, n0 + npad)
    const tileLoad = struct {
        fn one(k0: usize, nb0: usize) [16][16]u16 { // the kernel's load rule: tile[ty][tx] = SRC[(nb0+ty)*K + k0+tx], zero past N
            var t: [16][16]u16 = std.mem.zeroes([16][16]u16);
            for (0 .. t.len) |ty| for (0 .. t[ty].len) |tx| {
                const n = nb0 + ty;
                t[ty][tx] = if (n < N) @intCast(n * K + (k0 + tx) + 1) else 0; // unique per (n, k); 0 is the pad zero
            };
            return t;
        }
    }.one;
    const want = struct {
        fn at(n: usize, k: usize) u16 {
            return @intCast(n * K + k + 1);
        }
    }.at;
    var slab: [K * NT]u16 = std.mem.zeroes([K * NT]u16);
    var k0: usize = 0;
    while (k0 < K) : (k0 += 16) { // grid: x = K/16, y = npad/16
        var g: usize = 0;
        while (g < npad / 16) : (g += 1) {
            const nb0 = g * 16;
            const t = tileLoad(k0, nb0);
            for (0 .. 16) |ty| {
                for (0 .. 16) |tx| {
                    slab[(k0 + ty) * NT + n0 + nb0 + tx] = t[tx][ty]; // the fixed store
                }
            }
        }
    }
    const expect = std.testing.expect;
    for (0 .. K) |k| {
        for (0 .. N) |n| try expect(slab[k * NT + n0 + n] == want(n, k)); // the contract, every owned cell
        for (N .. npad) |np| try expect(slab[k * NT + n0 + np] == 0); // the pad columns hold zero
    }
    // The pre-fix store (DST[(k0 + tx)*NT + n0 + nb0 + ty] = tile[tx][ty]) leaves the tile
    // transposed: the test must be able to see that bug, not just bless the current code.
    var bad: [K * NT]u16 = std.mem.zeroes([K * NT]u16);
    k0 = 0;
    while (k0 < K) : (k0 += 16) {
        var g: usize = 0;
        while (g < npad / 16) : (g += 1) {
            const nb0 = g * 16;
            const t = tileLoad(k0, nb0);
            for (0 .. 16) |ty| {
                for (0 .. 16) |tx| {
                    bad[(k0 + tx) * NT + n0 + nb0 + ty] = t[tx][ty];
                }
            }
        }
    }
    var caught = false;
    for (0 .. K) |k| {
        for (0 .. N) |n| {
            caught = caught or bad[k * NT + n0 + n] != want(n, k);
        }
    }
    try expect(caught);
}
