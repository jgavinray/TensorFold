//! The host fp64 reference for block LDLQ (fable LDLQ spec section 4.3): the same loop
//! xpu/ldlq.zig runs on the device - strip to fp16, encodeTile per tile through the
//! step-1 reference encoder, error against the ORIGINAL weight, rank-16 feedback of the
//! strictly-lower L blocks - executed in fp64 on the host at test sizes. exl3_format.zig
//! and exl3_encode.zig are the oracle and the encoder it shares with the device: L = I
//! must reproduce core/exl3_encode.pack bit for bit (T-LDLQ-1), and the device driver is
//! tested against this file (T-LDLQ-3). The feedback sum accumulates in fp64 over the
//! device's fp32 operand values (l32, e32 in their upload form), i ascending, so the
//! device agrees with this reference to fp32 rounding; the fp64-ness is what makes it the
//! arbiter. D never enters the loop - it weights the reported proxy loss only.

const std = @import("std");
const exl3_encode = @import("exl3_encode.zig");
const exl3_format = @import("exl3_format.zig");

const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectEqualSlices = std.testing.expectEqualSlices;

pub const RefError = error{ BadDims, OutOfMemory, TileFailed, ProxyTooLarge, DumpOpenFailed };

/// The log2 buckets of cost - open_cost for PartReport.gap_hist: bucket 0 is 0, bucket k
/// (1..14) is [2^(k-1), 2^k - 1], bucket 15 is >= 2^14. The driver shares this definition.
pub fn gapBucket(diff: u64) u8 { // TileStat costs are u64 since the rebasing ruling
    if (diff == 0) return 0;
    const l: u8 = @intCast(63 - @clz(diff)); // floor(log2)
    return @min(15, l + 1);
}

pub const RefReport = struct {
    tiles: u32,
    optimal: u32,
    closed: u32,
    failed: u32,
    gap_hist: [16]u32,
    clamped: u64, // |T| > 4 count over every block (the encoder clamps to +/-4)
    eta_sq: f64, // sum ||eta_k||^2, eta = What - the block's fp16 target
};

/// The block-LDLQ loop of section 4.3 in fp64, one part.
/// wq64: the part's W_q [k, n] (rotated, scaled, section 4.2 step 3) - the ORIGINAL the
/// error and the proxy are measured against. l32: the slot's unit-block-lower L in the
/// device upload form [k, k]; identity means the calibration-free encoder.
/// trellis/results/eta receive the part's trellis words [kb*nt*tileWords], one Result per
/// tile in block order (j descending, tile ascending inside the block), and the optional
/// eta strips [k, n] (the fp32 rounding of the device's ldlq_eta output).
pub fn ldlqRef(
    gpa: std.mem.Allocator,
    m: *const exl3_encode.Metric,
    w: exl3_format.Width,
    wq64: []const f64,
    k: usize,
    n: usize,
    l32: []const f32,
    trellis: []i16,
    results: []exl3_encode.Result,
    eta: ?[]f32,
    dbg_path: ?[]const u8,
) RefError!RefReport {
    if (k == 0 or n == 0 or k % 16 != 0 or n % 16 != 0) return error.BadDims;
    const kb = k / 16;
    const nt = n / 16;
    const words = w.tileWords();
    if (wq64.len != k * n or l32.len != k * k) return error.BadDims;
    if (trellis.len != kb * nt * words or results.len != kb * nt) return error.BadDims;
    if (eta) |e| {
        if (e.len != k * n) return error.BadDims;
    }

    const w32 = try gpa.alloc(f32, k * n); // the uploaded W32 form: one f64 -> f32 rounding
    defer gpa.free(w32);
    for (wq64, w32) |q, *p| {
        if (!std.math.isFinite(q)) return error.BadDims;
        p.* = @floatCast(q);
    }
    const t32 = try gpa.alloc(f32, k * n); // the running targets: the device's T32, bit-faithfully
    defer gpa.free(t32);
    for (w32, t32) |p, *t| t.* = p; // T32 := W32

    const s16 = try gpa.alloc(f16, 16 * n); // the block's strip, the encoder's WQ
    defer gpa.free(s16);
    const q16 = try gpa.alloc(f16, 16 * n); // decoded codebook values of the chosen states
    defer gpa.free(q16);
    const e32 = try gpa.alloc(f32, 16 * n); // delta W against the original, fp32 like the device
    defer gpa.free(e32);
    var scratch = try exl3_encode.Scratch.init(gpa, w);
    defer scratch.deinit(gpa);

    var placement: [256]exl3_format.Placement = undefined;
    exl3_format.tilePlacement(&placement);

    var rep = RefReport{
        .tiles = @intCast(kb * nt),
        .optimal = 0,
        .closed = 0,
        .failed = 0,
        .gap_hist = @splat(0),
        .clamped = 0,
        .eta_sq = 0,
    };

    var dbg_fd: c_int = -1;
    var dbg_buf: std.ArrayList(u8) = .empty;
    defer dbg_buf.deinit(gpa);
    if (dbg_path) |path| {
        const sent = try gpa.dupeSentinel(u8, path, 0);
        defer gpa.free(sent);
        dbg_fd = ccreat(sent.ptr, 0o644);
        if (dbg_fd < 0) return error.DumpOpenFailed;
    }
    defer {
        if (dbg_fd >= 0) _ = cclose(dbg_fd);
    }
    var j: usize = kb;
    while (j > 0) {
        j -= 1;
        // strip_f16: S16 = fp16(T32[16j .. 16j+16]), counting |T| > 4 — the same f32->f16
        // rounding and the same compare the kernel performs on its fp32 strip
        for (0..16 * n) |i| {
            const t32v = t32[16 * j * n + i];
            s16[i] = @floatCast(t32v);
            if (@abs(t32v) > 4.0) rep.clamped += 1;
        }
        // the encoder launch: nt tiles of 16 x 16 through the step-1 reference encodeTile
        for (0..nt) |tn| {
            var tile: [256]f16 = undefined;
            for (0..256) |p| {
                const pl = placement[p];
                // matrix order, exactly pack's targets layout: encodeTile gathers placement
                tile[@as(usize, pl.row) * 16 + @as(usize, pl.col)] = s16[@as(usize, pl.row) * n + tn * 16 + @as(usize, pl.col)];
            }
            const res = exl3_encode.encodeTile(m, w, &tile, &scratch);
            if (res.status == .failed) return error.TileFailed; // the host aborts the quantization
            results[j * nt + tn] = res;
            exl3_encode.packStates(w, &res.states, trellis[(j * nt + tn) * words ..][0..words]);
            for (0..256) |p| {
                const pl = placement[p];
                q16[@as(usize, pl.row) * n + tn * 16 + @as(usize, pl.col)] =
                    exl3_format.codebookValue(.mul1, @intCast(res.states[p]));
            }
        }
        // strip_diff + the eta reduction over the block's contiguous rows
        for (0..16 * n) |i| {
            const qf: f32 = @floatCast(q16[i]);
            e32[i] = qf - w32[16 * j * n + i];
            const h: f32 = qf - @as(f32, @floatCast(s16[i]));
            rep.eta_sq += @as(f64, h) * @as(f64, h);
            if (eta) |e| e[16 * j * n + i] = h;
        }
        // ldlq_update, bit-faithful to the kernel (spec T-LDLQ-3 "trellis words identical,
        // eta strips equal to fp32 rounding"): fp32 fma chain, i ascending, from 0.0f,
        // and the one-rounding fp32 subtraction the kernel applies to T32. The
        // strictly-lower entries of an identity L are zero, so L = I leaves T untouched.
        if (j > 0) {
            for (0..16 * j) |r| {
                for (0..n) |c| {
                    var acc: f32 = 0;
                    for (0..16) |i| acc = @mulAdd(f32, l32[(16 * j + i) * k + r], e32[i * n + c], acc);
                    t32[r * n + c] -= acc;
                }
            }
        }
        if (dbg_fd >= 0) try dumpRefBlock(&dbg_buf, dbg_fd, gpa, j, s16, q16, t32, results[j * nt ..][0..nt], n, k);
    }

    for (results) |res| {
        switch (res.status) {
            .optimal => rep.optimal += 1,
            .closed => rep.closed += 1,
            .failed => rep.failed += 1,
        }
        rep.gap_hist[gapBucket(res.cost -% res.open_cost)] += 1;
    }
    return rep;
}

/// tr(deltaW^T H' deltaW) for a finished part (section 4.6): deltaW = unpack(trellis) -
/// wq64, with H' = L D L^T reconstructed from the BlockLdl forms (l32 the full [k, k]
/// upload form, d the [k/16][16][16] blocks), fp64 on the host. Guarded by
/// ProxyTooLarge beyond test sizes: the full product is the device job of section 4.6
/// (raw strip decoder + exl3_syrk32-style fp32 GEMMs); this host form is the always-on
/// number of the single-layer test and of T-LDLQ-1.
pub fn proxyLoss(
    gpa: std.mem.Allocator,
    w: exl3_format.Width,
    l32: []const f32,
    d: []const f64,
    trellis: []const i16,
    k: usize,
    n: usize,
    wq64: []const f64,
) RefError!f64 {
    if (k == 0 or n == 0 or k % 16 != 0 or n % 16 != 0) return error.BadDims;
    if (wq64.len != k * n or l32.len != k * k or d.len != (k / 16) * 256) return error.BadDims;
    if (@as(u64, k) * k * n > (1 << 28)) return error.ProxyTooLarge; // device path, section 4.6
    const words = w.tileWords();
    if (trellis.len != (k / 16) * (n / 16) * words) return error.BadDims;

    const got = try gpa.alloc(f16, k * n);
    defer gpa.free(got);
    exl3_format.unpack(w, .mul1, trellis, k / 16, n / 16, got);
    const v = try gpa.alloc(f64, k); // one column of deltaW
    defer gpa.free(v);
    const u = try gpa.alloc(f64, k); // L^T v
    defer gpa.free(u);
    const du = try gpa.alloc(f64, k); // D (L^T v)
    defer gpa.free(du);

    var loss: f64 = 0;
    for (0..n) |c| {
        for (0..k) |r| v[r] = @as(f64, @floatCast(got[r * n + c])) - wq64[r * n + c];
        for (0..k) |p| {
            var acc: f64 = 0;
            for (0..k) |r| acc += @as(f64, l32[r * k + p]) * v[r]; // (L^T v)[p]
            u[p] = acc;
        }
        for (0..k / 16) |bj| for (0..16) |a| {
            var acc: f64 = 0;
            for (0..16) |b| acc += d[(bj * 16 + a) * 16 + b] * u[bj * 16 + b];
            du[bj * 16 + a] = acc;
        };
        for (0..k) |r| {
            var acc: f64 = 0;
            for (0..k) |p| acc += @as(f64, l32[r * k + p]) * du[p]; // (L du)[r]
            loss += v[r] * acc;
        }
    }
    return loss;
}

/// tr(eta^T D eta) from the eta strips of ldlqRef: the D-weighted error of section 4.1,
/// the numerator of the D-ratio diagnostic (denominator: mean diag D times rep.eta_sq).
pub fn etaWeighted(d: []const f64, eta: []const f32, k: usize, n: usize) f64 {
    var tot: f64 = 0;
    for (0..k / 16) |bj| {
        for (0..n) |c| {
            for (0..16) |a| {
                var acc: f64 = 0;
                for (0..16) |b| {
                    acc += d[(bj * 16 + a) * 16 + b] * @as(f64, eta[(bj * 16 + b) * n + c]);
                }
                tot += @as(f64, eta[(bj * 16 + a) * n + c]) * acc;
            }
        }
    }
    return tot;
}

// ---------------------------------------------------------------------------
// test helpers

fn uniform01(r: std.Random) f32 {
    return (@as(f32, @floatFromInt(r.int(u32))) + 1.0) / 4294967296.0; // (0, 1]
}

/// Box-Muller standard normal, no dependency on a Gaussian distribution API.
fn gauss(r: std.Random) f32 {
    const za = uniform01(r);
    const zb = uniform01(r) - 0.5;
    return @sqrt(-2.0 * @log(za)) * @cos(2.0 * std.math.pi * zb);
}

/// A random unit-block-lower L [k, k] (diagonal 16x16 blocks exactly I, whole strictly-
/// lower 16x16 blocks random gaussian * scale, every other entry zero) and random
/// block-diagonal SPD D = A^T A + 16 I per block. H' = L D L^T is the random SPD H' of
/// T-LDLQ-1, and (L, D) is exactly what blockLdl would return for it.
fn randBlockLdl(r: std.Random, k: usize, l64: []f64, d: []f64) void {
    const kb = k / 16;
    for (0..k) |a| for (0..k) |b| {
        l64[a * k + b] = 0;
    };
    for (0..kb) |bc| for (0..kb) |br| {
        if (br <= bc) continue; // strictly-lower blocks only: row block below column block
        for (0..16) |a| for (0..16) |b| {
            l64[(br * 16 + a) * k + bc * 16 + b] = gauss(r) * 0.25;
        };
    };
    for (0..kb) |bc| for (0..16) |a| {
        l64[(bc * 16 + a) * k + bc * 16 + a] = 1; // diagonal blocks = I
    };
    for (0..kb) |bj| { // D blocks SPD: eigenvalues >= 16
        var a: [16 * 16]f64 = undefined;
        for (&a) |*x| x.* = gauss(r) * 0.4;
        for (0..16) |x| for (0..16) |y| {
            var acc: f64 = 0;
            for (0..16) |z| acc += a[z * 16 + x] * a[z * 16 + y];
            d[(bj * 16 + x) * 16 + y] = acc + @as(f64, if (x == y) 16 else 0);
        };
    }
}

/// tr(deltaW^T H deltaW) against an explicit fp64 H (T-LDLQ-1 scores both runs under the
/// real H' = L D L^T, not under each run's own factorization).
fn lossWithH(
    gpa: std.mem.Allocator,
    h: []const f64,
    w: exl3_format.Width,
    trellis: []const i16,
    k: usize,
    n: usize,
    wq64: []const f64,
) !f64 {
    const got = try gpa.alloc(f16, k * n);
    defer gpa.free(got);
    exl3_format.unpack(w, .mul1, trellis, k / 16, n / 16, got);
    const v = try gpa.alloc(f64, k);
    defer gpa.free(v);
    var loss: f64 = 0;
    for (0..n) |c| {
        for (0..k) |r| v[r] = @as(f64, @floatCast(got[r * n + c])) - wq64[r * n + c];
        for (0..k) |r| {
            var acc: f64 = 0;
            for (0..k) |q| acc += h[r * k + q] * v[q];
            loss += v[r] * acc;
        }
    }
    return loss;
}

// ---------------------------------------------------------------------------
// T-LDLQ-1: the host reference LDLQ on K = 64, N = 32, random Gaussian W_q, random SPD H'.
// L = I reproduces the step-1 pack bit for bit; with the real L the proxy loss is lower
// on at least 9 of 10 seeds (better in expectation, not per instance, spec section 6).

test "T-LDLQ-1 host reference: L = I reproduces pack bit for bit; real L lowers the proxy on >= 9/10 seeds" {
    const gpa = std.testing.allocator;
    const m = exl3_encode.Metric.init();
    const w = exl3_format.Width{ .halves = 6 }; // the 3-bit target of the step-4 packs
    const k = 64;
    const n = 32;
    const kb = k / 16;
    const nt = n / 16;
    const words = w.tileWords();

    var lower: u8 = 0;
    for (0..10) |seed| {
        var prng = std.Random.DefaultPrng.init(0x7CD1_5D10 ^ (@as(u64, seed) *% 0x9E37_79B9_7F4A_7C15));
        const r = prng.random();

        const wq64 = try gpa.alloc(f64, k * n);
        defer gpa.free(wq64);
        for (wq64) |*v| {
            v.* = @as(f64, gauss(r)); // unit-variance rotated target, the T5 shape
        }

        const l64 = try gpa.alloc(f64, k * k);
        defer gpa.free(l64);
        const d = try gpa.alloc(f64, (k / 16) * 256);
        defer gpa.free(d);
        randBlockLdl(r, k, l64, d);

        // H' = L D L^T, fp64, the SPD matrix both runs are scored under
        const h = try gpa.alloc(f64, k * k);
        defer gpa.free(h);
        for (0..k) |a| for (0..k) |b| {
            var acc: f64 = 0;
            for (0..k) |p| {
                const la = l64[a * k + p];
                if (la == 0) continue;
                var dbs: f64 = 0;
                const pb = p / 16; // D is block diagonal: only p's own block columns matter
                for (0..16) |q| dbs += l64[b * k + pb * 16 + @as(usize, q)] * d[(pb * 16 + @as(usize, p % 16)) * 16 + @as(usize, q)]; // (D L^T)[p][b]
                acc += la * dbs;
            }
            h[a * k + b] = acc;
        };
        // mirror-symmetry sanity of the constructed H'
        for (0..k) |a| for (0..k) |b| {
            try expect(@abs(h[a * k + b] - h[b * k + a]) <= 1e-12 * @max(1.0, @abs(h[b * k + a])));
        };

        const l32 = try gpa.alloc(f32, k * k);
        defer gpa.free(l32);
        for (l32, 0..) |*p, i| p.* = @floatCast(l64[i]);
        const l32i = try gpa.alloc(f32, k * k);
        defer gpa.free(l32i);
        for (l32i, 0..) |*p, i| p.* = @floatCast(if (i / k == i % k) @as(f32, 1) else @as(f32, 0));

        // run I: the identity L must equal the calibration-free packer bit for bit
        const trI = try gpa.alloc(i16, kb * nt * words);
        defer gpa.free(trI);
        const resI = try gpa.alloc(exl3_encode.Result, kb * nt);
        defer gpa.free(resI);
        const eta = try gpa.alloc(f32, k * n);
        defer gpa.free(eta);
        const repI = try ldlqRef(gpa, &m, w, wq64, k, n, l32i, trI, resI, eta, null);
        try expectEqual(@as(u64, 0), repI.failed);

        // the step-1 packer on the same fp16 strip the device would see (f64 -> f32 -> f16)
        const wq16 = try gpa.alloc(f16, k * n);
        defer gpa.free(wq16);
        for (wq64, wq16) |q, *p| p.* = @floatCast(@as(f32, @floatCast(q)));
        const trP = try gpa.alloc(i16, kb * nt * words);
        defer gpa.free(trP);
        var pack_scratch = try exl3_encode.Scratch.init(gpa, w);
        defer pack_scratch.deinit(gpa);
        try exl3_encode.pack(&m, w, wq16, kb, nt, trP, &pack_scratch, null, null);
        try expectEqualSlices(i16, trP, trI); // bit for bit, the calibration-free identity

        // eta of the identity run is the fp32 rounding of codebookValue - fp16(wq),
        // re-derived against the trellis-decoded weight and tied to the reported eta_sq
        var esq: f64 = 0;
        const dec = try gpa.alloc(f16, k * n);
        defer gpa.free(dec);
        exl3_format.unpack(w, .mul1, trI, kb, nt, dec);
        for (dec, wq16, eta) |wh, t, e| {
            const want: f32 = @as(f32, @floatCast(wh)) - @as(f32, @floatCast(t));
            try expectEqual(@as(u32, @bitCast(e)), @as(u32, @bitCast(want)));
            const ef: f64 = @floatCast(e);
            esq += ef * ef;
        }
        try expect(esq > 0);
        try expect(@abs(esq - repI.eta_sq) < 1e-9 * (1.0 + esq));

        // run L: the real factorization must lower the same part's proxy under the real H'
        const trL = try gpa.alloc(i16, kb * nt * words);
        defer gpa.free(trL);
        const resL = try gpa.alloc(exl3_encode.Result, kb * nt);
        defer gpa.free(resL);
        const repL = try ldlqRef(gpa, &m, w, wq64, k, n, l32, trL, resL, null, null);
        try expectEqual(@as(u64, 0), repL.failed);
        const lossI = try lossWithH(gpa, h, w, trI, k, n, wq64);
        const lossL = try lossWithH(gpa, h, w, trL, k, n, wq64);
        if (lossL < lossI) lower += 1;
    }
    try expect(lower >= 9);
}

/// The twin of ldlq.zig's dumpBlock: the same lines from the reference's own buffers,
/// for the offline diff. Status maps to the TileStat encoding (optimal 0, closed 1,
/// failed 2) so the parse is shared.
fn dumpRefBlock(
    buf: *std.ArrayList(u8),
    fd: c_int,
    a: std.mem.Allocator,
    j: usize,
    s16: []const f16,
    q16: []const f16,
    t32: []const f32,
    results: []const exl3_encode.Result,
    n: usize,
    k: usize,
) !void {
    for (results, 0..) |res, tn| {
        const status: u32 = switch (res.status) {
            .optimal => 0,
            .closed => 1,
            .failed => 2,
        };
        try buf.print(a, "REF ST {d} {d} {x} {x} {d} {d}\n", .{ j, tn, res.cost, res.open_cost, res.pinned, status });
    }
    try buf.print(a, "REF S16 {d}\n", .{j});
    for (s16) |v| try buf.print(a, "{x:0>4} ", .{@as(u16, @bitCast(v))});
    try buf.append(a, '\n');
    try buf.print(a, "REF Q16 {d}\n", .{j});
    for (q16) |v| try buf.print(a, "{x:0>4} ", .{@as(u16, @bitCast(v))});
    try buf.append(a, '\n');
    if (j > 0) {
        try buf.print(a, "REF T32 {d}\n", .{j});
        for (t32[0 .. @as(usize, 16 * j) * n]) |v| try buf.print(a, "{x:0>8} ", .{@as(u32, @bitCast(v))});
        try buf.append(a, '\n');
    }
    _ = k;
    if (buf.items.len > 0) {
        _ = cwrite(fd, buf.items.ptr, buf.items.len);
        buf.clearRetainingCapacity();
    }
}

const ccreat = @extern(*const fn (path: [*:0]const u8, mode: c_uint) callconv(.c) c_int, .{ .name = "creat" });
const cwrite = @extern(*const fn (fd: c_int, buf: [*]const u8, n: usize) callconv(.c) isize, .{ .name = "write" });
const cclose = @extern(*const fn (fd: c_int) callconv(.c) c_int, .{ .name = "close" });
