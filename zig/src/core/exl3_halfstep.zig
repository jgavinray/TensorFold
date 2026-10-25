//! Step 5 (eaq-halfstep-spec-fable.md): the odd-halves host gate for the three fp16
//! decode kernels of kernels/xpu/qwen27/exl3_h.cl. The kernel arithmetic — the generalized
//! window start `a_v = 8*lane*halves - 16 + (((v+1)*halves)>>1)`, the floor and t1 ring
//! merges, the exact-integer half-even weight, the placement closed form and the decode
//! GEMV's four-accumulator order — is transcribed here verbatim from the kernels (the
//! offsets stay INLINE, exactly as in the kernels: no offset table on either side), so the
//! device probe (`xpu-qwen27 --exl3h-probe`, tests T-HS-4..7) compares the device runs of
//! exl3_dequant128_c / exl3_decode2_h / exl3_decode2_t1_h against these functions, and
//! these tests (T-HS-0..3) compare these functions against the oracle. exl3_format.zig is
//! the oracle for everything it defines (streamEnds, tileStates, unpack, codebookValue);
//! exl3_encode.zig supplies the producer-side tile builders the spec names (packStates of
//! closed sequences, encodeTile of Gaussian tiles).
//!
//! The kernel's `int bits` becomes `int halves` (2..16, odd allowed): nw = 4*halves ring
//! words, R = 128*halves ring bits, tile stride 8*halves int16 words. At even halves every
//! expression here reduces bit for bit to the pre-step-5 kernels' (the spec's 1.2 mapping
//! and T-HS-6 pin that); at odd halves the only generalization is the per-position offset
//! `(v + 1) * bits` -> `((v + 1) * halves) >> 1`.

const std = @import("std");
const exl3_format = @import("exl3_format.zig");
const exl3_encode = @import("exl3_encode.zig");

pub const Width = exl3_format.Width;

pub fn halvesOk(halves: u32) bool {
    return halves >= 2 and halves <= 16;
}

// ---------------------------------------------------------------------------
// 1.2 / 1.3: the window. ring words nw = 4*halves, ring bits R = 128*halves.

pub fn ringWords(halves: u32) u32 {
    return 4 * halves;
}
pub fn ringBits(halves: u32) u32 {
    return 128 * halves;
}

/// The tile's int16 words as the kernels view them: nw little-endian uint32s (the
/// exl3_format.streamBit pairing, word = lo | hi << 16).
pub fn ringOf(tile: []const i16, out: []u32) void {
    std.debug.assert(tile.len % 2 == 0 and out.len == tile.len / 2);
    for (0..out.len) |w| {
        const lo: u16 = @bitCast(tile[2 * w]);
        const hi: u16 = @bitCast(tile[2 * w + 1]);
        out[w] = @as(u32, lo) | (@as(u32, hi) << 16);
    }
}

/// The lane's first needed bit, lb = 8*lane*halves - 16 (may be negative; the kernel's
/// old `lo` was this with halves = 2*bits).
pub fn laneBase(halves: u32, lane: u32) i32 {
    return @as(i32, @intCast(8 * lane * halves)) - 16;
}

/// The window's start bit, unwrapped: a_v = lb + (((v + 1) * halves) >> 1). Equals
/// E(16*lane + v) - 16 with the unified E(p) = ((p + 1) * halves) >> 1, which is what
/// T-HS-0 pins against exl3_format.streamEnds.
pub fn windowStart(halves: u32, lane: u32, v: u32) i32 {
    return laneBase(halves, lane) + @as(i32, @intCast(((v + 1) * halves) >> 1));
}

/// E(p) for the unified closed form (kept for the tests' benefit).
pub fn streamEnd(halves: u32, p: u32) u32 {
    return ((p + 1) * halves) >> 1;
}

fn merge(ring: []const u32, nw: u32, uu: u32, s: u32) u16 {
    std.debug.assert(s < 32 and uu < nw);
    const w0 = ring[uu];
    const w1 = ring[if (uu + 1 == nw) 0 else uu + 1];
    const w32 = if (s == 0) w0 else
        (w0 << @as(u5, @intCast(s))) | (w1 >> @as(u5, @intCast(32 - s))); // the kernel's `s ? w0 << s | w1 >> (32 - s) : w0`
    return @truncate(w32 >> 16);
}

/// exl3_decode2_h / exl3_dequant128_c phase A's extraction (floor-division form): the
/// 16-bit state of position p = 16*lane + v. a_v's minimum is (halves >> 1) - 16 >= -15,
/// so the floor word index is -1 at most and uu = u + nw is the right ring word; the
/// maximum is R - 16, so only the second word ever wraps.
pub fn stateFloor(ring: []const u32, halves: u32, lane: u32, v: u32) u16 {
    const nw = ringWords(halves);
    std.debug.assert(ring.len == nw);
    const a_v = windowStart(halves, lane, v);
    const u = @divTrunc(if (a_v >= 0) a_v else a_v - 31, 32); // the kernel's C `/`: truncation on the pre-adjusted operand
    const uu: u32 = if (u < 0) @intCast(u + @as(i32, @intCast(nw))) else @intCast(u);
    return merge(ring, nw, uu, @intCast(a_v - u * 32));
}

/// exl3_decode2_t1_h's extraction: the av-shifted form (av = a_v + R, word av >> 5, shift
/// av & 31). av lies in [R - 16, 2R) and the word index in [0, 2*nw), both handled.
pub fn stateT1(ring: []const u32, halves: u32, lane: u32, v: u32) u16 {
    const nw = ringWords(halves);
    std.debug.assert(ring.len == nw);
    const av = windowStart(halves, lane, v) + @as(i32, @intCast(ringBits(halves)));
    std.debug.assert(av > 0); // av >= R - 16 >= 240 at every halves
    const w0i: u32 = @intCast(av >> 5); // positive: the arithmetic shift is the kernel's w0i
    const uu = if (w0i >= nw) w0i - nw else w0i;
    return merge(ring, nw, uu, @intCast(av & 31));
}

// ---------------------------------------------------------------------------
// The weight forms: the byte sum, the kernel's fast fp32 value, the exact integer
// half-even form of the dequant's phase A, and the exact real value for references.

/// sum = 1024 + (bs & 0xFFFF) + (bs >> 16), the kernel's byte-sum of state * 0x83DCD12D.
pub fn mul1Sum(state: u16) u32 {
    const xx = @as(u32, state) *% 0x83DCD12D;
    const bs = (xx & 0x00FF00FF) + ((xx >> 8) & 0x00FF00FF);
    return 1024 + (bs & 0xFFFF) + (bs >> 16);
}

const scale_bits: u16 = 0x1EEE;
const bias_bits: u16 = 0xC931;
const scale_f: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));
const bias_f: f32 = @floatCast(@as(f16, @bitCast(bias_bits)));

/// The decode GEMVs' fast fp32 weight form (exact per weight: the product fits 23 bits
/// and the sum 24, the exl3_h.cl:45-53 note); NOT the format's rounded weight.
pub fn fastWeight(sum: u32) f32 {
    return @as(f32, @floatFromInt(sum)) * scale_f + bias_f;
}

/// The dequant kernel's exact-integer half-even fp16 weight (exl3_h.cl's phase A; the
/// oracle proves it equals codebookValue for every state, exl3_format's integerFp16Weight
/// test): n2 = sum * 1774 - 2721792, rounded to 11 bits half-to-even, value r * 2^-18.
pub fn integerWeightF16(sum: u32) f16 {
    const n: i32 = @as(i32, @intCast(sum)) * 1774 - 2721792;
    const mag: u32 = @abs(n);
    std.debug.assert(mag != 0); // (1024+S)*1774 never equals 2721792 for S in [0, 1020]
    const b: u32 = 31 - @clz(mag);
    const q: u32 = if (b > 10) @as(u32, 1) << @intCast(b - 10) else 1;
    const rem: u32 = mag & (q - 1);
    var r: u32 = mag - rem;
    if (rem > (q >> 1) or (rem == (q >> 1) and (r & q) != 0)) r += q;
    const signed: i32 = if (n < 0) -@as(i32, @intCast(r)) else @as(i32, @intCast(r));
    return @floatCast(@as(f32, @floatFromInt(signed)) * 0x1.0p-18); // exact into f16: 11 bits
}

/// The same weight as the exact real value (the fast form before rounding and the
/// integer form's base alike): (sum * 1774 - 2721792) * 2^-18.
pub fn exactWeight(sum: u32) f64 {
    const n: i64 = @as(i64, @intCast(sum)) * 1774 - 2721792;
    return @as(f64, @floatFromInt(n)) * 0x1.0p-18;
}

/// The tile placement closed form of position p = 16*lane + v (exl3_format.tilePlacement's
/// row and col as the kernels' closed forms, l = p >> 3, j = p & 7).
pub fn place(lane: u32, v: u32) struct { row: u32, col: u32 } {
    const p = 16 * lane + v;
    const l = p >> 3;
    const j = p & 7;
    return .{
        .row = 2 * (l & 3) + (j & 1) + 8 * ((j >> 1) & 1),
        .col = (l >> 2) + 8 * (j >> 2),
    };
}

// ---------------------------------------------------------------------------
// 2. The dequant transcription (exl3_dequant128_c phase A only: the extraction plus the
// exact integer weight, stored where the kernel's blk lands it, which is the pre-rotation
// W_q the oracle's unpack writes). Phase B/C (the Hadamards and the scales) are not
// transcribed here: T-HS-4 checks the full kernel against exl3_format.dequantize inside
// the fp16 envelope, and T-HS-7's bit-identity ride is the exact-rotation replica the
// step-1 probe already keeps (qwen27_main.rotExact128).

/// One part's [K][N] pre-rotation weights, each tile's words decoded by the phase-A
/// transcription: `tr` is the k-major trellis ((K/16)*(N/16) tiles of 8*halves int16),
/// `wq` is [K][N] row-major fp16.
pub fn dequantTranscribe(tr: []const i16, halves: u32, tiles_k: u32, tiles_n: u32, wq: []f16) void {
    std.debug.assert(halvesOk(halves));
    const words = @as(usize, halves) * 8;
    const nw = ringWords(halves);
    std.debug.assert(tr.len == @as(usize, tiles_k) * tiles_n * words and wq.len == @as(usize, tiles_k) * 16 * tiles_n * 16);
    var ring: [64]u32 = undefined;
    const n = @as(usize, tiles_n) * 16;
    for (0..tiles_k) |tk| {
        for (0..tiles_n) |tn| {
            const tile = tr[(tk * tiles_n + tn) * words ..][0..words];
            ringOf(tile, ring[0..nw]);
            for (0..16) |lane| {
                for (0..16) |v| {
                    const st = stateFloor(ring[0..nw], halves, @intCast(lane), @intCast(v));
                    const pl = place(@intCast(lane), @intCast(v));
                    wq[(tk * 16 + pl.row) * n + tn * 16 + pl.col] = integerWeightF16(mul1Sum(st));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. The decode GEMV transcriptions: extraction, placement, the fast fp32 weight, the
// four accumulators per token row (acc0/acc1 for even v, acc0b/acc1b for odd v — the
// kernel's split by (j >> 2, v & 1)), accumulated kt-major then v-ascending, and the
// lane-pair combine (acc0 + acc0b) + (partner acc0 + partner acc0b). With `want_states`
// the extraction also lands there, tile order (tk * tiles_n + tn) * 256 + p — the exact
// index order exl3_format.tileStates is called in per tile (T-HS-3's state equality).

/// exl3_decode2_h (T = rows, T <= 8, kslices = 1, whole K): y is [rows][NT] fp16 at the
/// part's column base (NT is the kernel's pitch argument), x is [rows][K].
pub fn gemvTranscribe(tr: []const i16, halves: u32, k: u32, n: u32, nt: u32, x: []const f16, rows: u32, y: []f16, want_states: ?[]u32) void {
    std.debug.assert(halvesOk(halves) and rows >= 1 and rows <= 8);
    std.debug.assert(k % 16 == 0 and n % 16 == 0 and nt >= n);
    const words = @as(usize, halves) * 8;
    const nw = ringWords(halves);
    const tiles_k = k / 16;
    const tiles_n = n / 16;
    std.debug.assert(tr.len == @as(usize, tiles_k) * tiles_n * words);
    std.debug.assert(x.len == @as(usize, rows) * k and y.len == @as(usize, rows) * nt);
    if (want_states) |ws| std.debug.assert(ws.len == @as(usize, tiles_k) * tiles_n * 256);
    var ring: [64]u32 = undefined;
    for (0..tiles_n) |tn| {
        var acc: [16][4][8]f32 = undefined; // [lane][acc0,acc1,acc0b,acc1b][token]
        for (&acc) |*lane_acc| for (lane_acc) |*t4| {
            t4.* = @splat(0.0);
        };
        for (0..tiles_k) |tk| {
            const tile = tr[(tk * tiles_n + tn) * words ..][0..words];
            ringOf(tile, ring[0..nw]);
            for (0..16) |lane| {
                for (0..16) |v| {
                    const st = stateFloor(ring[0..nw], halves, @intCast(lane), @intCast(v));
                    if (want_states) |ws| ws[(tk * tiles_n + tn) * 256 + 16 * lane + v] = st;
                    const wgt = fastWeight(mul1Sum(st));
                    const pl = place(@intCast(lane), @intCast(v));
                    const sel: usize = (if ((v & 4) != 0) @as(usize, 1) else 0) + (if ((v & 1) != 0) @as(usize, 2) else 0);
                    for (0..rows) |t| {
                        const xv = @as(f32, @floatCast(x[t * k + tk * 16 + pl.row]));
                        acc[lane][sel][t] += xv * wgt;
                    }
                }
            }
        }
        for (0..8) |lane0| {
            const lane = lane0 * 2; // the kernel stores from even lanes; the pair sums both
            const l0 = &acc[lane];
            const l1 = &acc[lane + 1];
            for (0..rows) |t| {
                const a0 = (l0[0][t] + l0[2][t]) + (l1[0][t] + l1[2][t]);
                const a1 = (l0[1][t] + l0[3][t]) + (l1[1][t] + l1[3][t]);
                y[t * nt + tn * 16 + lane / 2] = @floatCast(a0);
                y[t * nt + tn * 16 + 8 + lane / 2] = @floatCast(a1);
            }
        }
    }
}

/// exl3_decode2_t1_h (the T == 1 fast path): the same math and output as
/// gemvTranscribe with rows = 1, transcribed as the kernel is written — scalar
/// accumulators, one activation load per weight, no token loop.
pub fn gemvT1Transcribe(tr: []const i16, halves: u32, k: u32, n: u32, nt: u32, x: []const f16, y: []f16) void {
    std.debug.assert(halvesOk(halves));
    const words = @as(usize, halves) * 8;
    const nw = ringWords(halves);
    const tiles_k = k / 16;
    const tiles_n = n / 16;
    std.debug.assert(tr.len == @as(usize, tiles_k) * tiles_n * words);
    std.debug.assert(x.len == k and y.len == nt);
    for (0..tiles_n) |tn| {
        var acc0: [16]f32 = @splat(0.0);
        var acc1: [16]f32 = @splat(0.0);
        var b0: [16]f32 = @splat(0.0);
        var b1: [16]f32 = @splat(0.0);
        var ring: [64]u32 = undefined;
        for (0..tiles_k) |tk| {
            const tile = tr[(tk * tiles_n + tn) * words ..][0..words];
            ringOf(tile, ring[0..nw]);
            for (0..16) |lane| {
                for (0..16) |v| {
                    const st = stateT1(ring[0..nw], halves, @intCast(lane), @intCast(v));
                    const wgt = fastWeight(mul1Sum(st));
                    const pl = place(@intCast(lane), @intCast(v));
                    const xv = @as(f32, @floatCast(x[tk * 16 + pl.row]));
                    if ((v & 4) != 0) { // j >> 2
                        if ((v & 1) != 0) b1[lane] += xv * wgt else acc1[lane] += xv * wgt;
                    } else {
                        if ((v & 1) != 0) b0[lane] += xv * wgt else acc0[lane] += xv * wgt;
                    }
                }
            }
        }
        for (0..8) |lane0| {
            const lane = lane0 * 2;
            const a0 = (acc0[lane] + b0[lane]) + (acc0[lane + 1] + b0[lane + 1]);
            const a1 = (acc1[lane] + b1[lane]) + (acc1[lane + 1] + b1[lane + 1]);
            y[tn * 16 + lane / 2] = @floatCast(a0);
            y[tn * 16 + 8 + lane / 2] = @floatCast(a1);
        }
    }
}

// ---------------------------------------------------------------------------
// 4. The independent GEMV reference: the exact fast-form weight matrix scattered by the
// placement, and the fp64 row product. The transcription and the device must both sit
// inside one measured fp16-ulp envelope of it (T-HS-3; the device half is T-HS-5).

/// The part's [K][N] exact weights (the fast form's real values) from the same trellis.
pub fn fastWeightMatrix(tr: []const i16, halves: u32, tiles_k: u32, tiles_n: u32, wf: []f64) void {
    const words = @as(usize, halves) * 8;
    const nw = ringWords(halves);
    std.debug.assert(tr.len == @as(usize, tiles_k) * tiles_n * words and wf.len == @as(usize, tiles_k) * 16 * tiles_n * 16);
    var ring: [64]u32 = undefined;
    const n = @as(usize, tiles_n) * 16;
    for (0..tiles_k) |tk| {
        for (0..tiles_n) |tn| {
            const tile = tr[(tk * tiles_n + tn) * words ..][0..words];
            ringOf(tile, ring[0..nw]);
            for (0..16) |lane| {
                for (0..16) |v| {
                    const st = stateFloor(ring[0..nw], halves, @intCast(lane), @intCast(v));
                    const pl = place(@intCast(lane), @intCast(v));
                    wf[(tk * 16 + pl.row) * n + tn * 16 + pl.col] = exactWeight(mul1Sum(st));
                }
            }
        }
    }
}

/// y (host fp64, [rows][n]) = x [rows][k] . wf [k][n], summed in fp64: the reference
/// whose fp16 form both transcriptions and the device kernels are bounded against.
pub fn gemvReference(x: []const f16, rows: u32, wf: []const f64, k: u32, n: u32, y: []f64) void {
    std.debug.assert(x.len == @as(usize, rows) * k and wf.len == @as(usize, k) * n and y.len == @as(usize, rows) * n);
    for (0..rows) |t| {
        for (0..n) |j| {
            var s: f64 = 0;
            for (0..k) |i| s += @as(f64, @floatCast(x[t * k + i])) * wf[i * n + j];
            y[t * n + j] = s;
        }
    }
}

/// The fp16 ulp distance: the count of representable f16 values between the two
/// (sign-magnitude monotone key; -0 and +0 are adjacent-zero equal).
pub fn f16UlpDist(a: f16, b: f16) u32 {
    const ua: u16 = @bitCast(a);
    const ub: u16 = @bitCast(b);
    const ka: i32 = if (ua & 0x8000 != 0) -@as(i32, @intCast(ua & 0x7FFF)) else @intCast(ua);
    const kb: i32 = if (ub & 0x8000 != 0) -@as(i32, @intCast(ub & 0x7FFF)) else @intCast(ub);
    return @abs(ka - kb);
}

/// The max fp16-ulp distance between a transcription's y (row-major [rows][nt], the part's
/// columns are [0, n) — the caller passes the part's slice) and the fp64 reference. The
/// distance is measured at the reference's own ulp, floored at the ulp of the batch scale
/// (the max |ref|) nine bits down: a K=256 fp32 accumulation carries an absolute error
/// floor (~2^-17 of the sum scale), so a row that cancels to near zero cannot be judged
/// in its own subnormal ulps — it compares at the scale the sum was computed at, which is
/// what the envelope in T-HS-3 is actually pinning.
pub fn gemvUlpBound(y: []const f16, nt: u32, yref: []const f64, rows: u32, n: u32) u32 {
    var scale: f64 = 6.103515625e-5; // smallest normal f16, so the floor never underflows
    for (yref) |r| scale = @max(scale, @abs(r));
    const floor = fp16UlpAt(scale * 0x1p-9);
    var worst: u32 = 0;
    for (0..rows) |t| {
        for (0..n) |j| {
            const g: f64 = @floatCast(y[t * nt + j]);
            const r = yref[t * n + j];
            const rf: f64 = @abs(@as(f16, @floatCast(r)));
            const denom = @max(fp16UlpAt(rf), floor);
            const d: u32 = @intFromFloat(@ceil(@abs(g - r) / denom));
            worst = @max(worst, d);
        }
    }
    return worst;
}

/// The fp16 ulp at magnitude v: 2^(floor(log2 v) - 10) for a normal v (v below the
/// smallest normal is clamped to it — callers use this as a distance denominator).
fn fp16UlpAt(v: f64) f64 {
    const e: i32 = @intFromFloat(@floor(std.math.log2(@max(v, 6.103515625e-5))));
    return @bitCast(@as(u64, @as(u64, @intCast(e + 1013))) << 52); // 2^(e-10) as an exact f64
}

// ---------------------------------------------------------------------------
// 5. Test tile sources: the producer's own builders (the spec's T-HS-1 sentence —
// packStates of random closed sequences, encodeTile of Gaussian tiles).

fn gaussPairSeed(r: std.Random) [2]f64 {
    const za = @max(r.float(f64), 1e-300);
    const zb = r.float(f64);
    const rr = @sqrt(-2.0 * @log(za));
    return .{ rr * @cos(2 * std.math.pi * zb), rr * @sin(2 * std.math.pi * zb) };
}

/// `tiles` tiles of one part filled with packStates(closedFromRing()) words; the states
/// behind them are reproducible from the same seeded Random in the same order.
pub fn fillClosedTiles(r: std.Random, halves: u32, tiles: usize, tr: []i16) void {
    std.debug.assert(halvesOk(halves));
    const words = @as(usize, halves) * 8;
    std.debug.assert(tr.len == tiles * words);
    const w: Width = .{ .halves = @intCast(halves) };
    for (0..tiles) |ti| {
        var states: [256]u16 = undefined;
        exl3_encode.closedFromRing(r, @intCast(halves), &states);
        exl3_encode.packStates(w, &states, tr[ti * words ..][0..words]);
    }
}

/// `tiles` tiles encoded from independent Gaussian [16][16] targets through the step-1
/// host encoder (the producer's real output at this width).
pub fn fillEncodedTiles(r: std.Random, gpa: std.mem.Allocator, halves: u32, tiles: usize, tr: []i16) !void {
    std.debug.assert(halvesOk(halves));
    const words = @as(usize, halves) * 8;
    std.debug.assert(tr.len == tiles * words);
    const w: Width = .{ .halves = @intCast(halves) };
    const m = exl3_encode.Metric.init();
    var scratch = try exl3_encode.Scratch.init(gpa, w);
    defer scratch.deinit(gpa);
    for (0..tiles) |ti| {
        var targets: [256]f16 = undefined;
        for (&targets) |*t| {
            const g = gaussPairSeed(r);
            t.* = @floatCast(g[0]);
        }
        const res = exl3_encode.encodeTile(&m, w, &targets, &scratch);
        if (res.status == .failed) return error.EncodeTileFailed;
        exl3_encode.packStates(w, &res.states, tr[ti * words ..][0..words]);
    }
}

/// Random fp16 activations in [-1, 1] (the probes use the same draw for X).
pub fn fillRandomX(r: std.Random, x: []f16) void {
    for (x) |*v| v.* = @floatCast(r.float(f32) * 2.0 - 1.0);
}
