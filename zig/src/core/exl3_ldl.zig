//! Host block LDL for the EAQ fresh-quant LDLQ path (spec eaq-ldlq-spec-fable.md
//! section 3): fp64, host only, no device types.
//!
//! * `blockLdl` - blocked right-looking Cholesky (128-wide panels, threaded
//!   fp64 trailing update in packed 64x64 rank-K tasks) followed by the
//!   QTIP-style block LDL conversion in 16-row blocks: H' = L D L^T with L
//!   unit-block-lower (every diagonal 16x16 block exactly I) and D
//!   block-diagonal. A naive triple loop is explicitly not allowed by the
//!   spec (section 3.3); this is the blocked factorization it asks for.
//! * `rotateBoth` - fp64 FWHT along both axes of a row-major matrix in
//!   128-blocks with the oracle's sign and scale; exl3_format.rotate
//!   (h[i][j] = (-1)^popcount(i & j) / sqrt(128)) is the oracle, matched to
//!   1e-12 (T-LDL-3).
//! * `flipSigns` - congruence by a +-1 diagonal: x[i][j] *= s[i] * s[j].
//!
//! Determinism (spec section 2.3 mirrored): each H' element is updated by
//! exactly one worker per panel and every dot product accumulates in ascending
//! column order, so the factorization is invariant to the thread count.
//!
//! Environment knobs (host experiments only, sane defaults): EAQ_LDL_THREADS
//! overrides the worker count, EAQ_LDL_PANEL the Cholesky panel width (a
//! multiple of 16), EAQ_LDL_PHASES=1 prints a per-call phase breakdown
//! (checks / panel / normalize / trailing / finalize) for long-pole hunting.
//! EAQ_LDL_TIMING=1 enables the T-LDL-2 timing test; run it
//! `-Doptimize=ReleaseFast` (Debug mode is ~60x slower and is not the shipped
//! build). Standalone: `zig test zig/src/core/exl3_ldl.zig -lc`.

const std = @import("std");
const exl3_format = @import("exl3_format.zig");

const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectError = std.testing.expectError;
const expectApproxEqAbs = std.testing.expectApproxEqAbs;

/// The LDL block height: one row of trellis blocks. Fixed by the format.
pub const block_size: usize = 16;
/// Right-looking Cholesky panel width (spec section 3.3: 128-wide panels).
pub const default_panel: usize = 128;
/// Upper bound on factorization worker threads.
pub const max_threads: usize = 128;

pub const BlockLdl = struct {
    /// K, a multiple of 16.
    k: usize,
    /// [K, K] row-major, unit-block-lower L in fp32 for the device: strictly
    /// lower blocks live, diagonal 16x16 blocks are exactly I, blocks above
    /// the block diagonal are exactly 0.
    l: []f32,
    /// [K/16][16][16] fp64 block diagonal D (d[b*256 + a*16 + c] = (DL DL^T)[a][c]),
    /// for the proxy-loss report; never uploaded to the device.
    d: []f64,
    lambda: f64,
    mu: f64,

    pub fn deinit(b: *BlockLdl, gpa: std.mem.Allocator) void {
        gpa.free(b.l);
        gpa.free(b.d);
        b.* = .{ .k = 0, .l = &.{}, .d = &.{}, .lambda = 0, .mu = 0 };
    }
};

pub const LdlError = error{ NonFinite, NotPositive, OutOfMemory };

/// H' [K, K] fp64 symmetric (spec 2.4 steps 1-4 applied). On NotPositive
/// `pivot` is filled with the global row index p of the first non-positive
/// pivot (its 16-row block is p / 16); the dump of spec 3.2 is the caller's.
pub fn blockLdl(
    gpa: std.mem.Allocator,
    h: []const f64,
    k: usize,
    lambda: f64,
    mu: f64,
    pivot: *usize,
) LdlError!BlockLdl {
    return ldl(gpa, h, k, lambda, mu, pivot, null);
}

/// Verification helper for T-LDL-1: identical to `blockLdl` and additionally
/// stores the fp64 L (unit-block-lower, above the block diagonal exactly 0)
/// into `*l64_out`, which the caller owns and frees. The fp32 `BlockLdl.l`
/// alone cannot support a 1e-12 reconstruction check.
pub fn blockLdlL64(
    gpa: std.mem.Allocator,
    h: []const f64,
    k: usize,
    lambda: f64,
    mu: f64,
    pivot: *usize,
    l64_out: *[]f64,
) LdlError!BlockLdl {
    return ldl(gpa, h, k, lambda, mu, pivot, l64_out);
}

fn ldl(
    gpa: std.mem.Allocator,
    h: []const f64,
    k: usize,
    lambda: f64,
    mu: f64,
    pivot: *usize,
    l64_out: ?*[]f64,
) LdlError!BlockLdl {
    std.debug.assert(k > 0 and k % block_size == 0);
    std.debug.assert(h.len == k * k);
    const profile = envFlag("EAQ_LDL_PHASES");
    var ph = PhaseTimes{};
    var t0: u64 = 0;
    if (profile) t0 = monoNs();
    // Spec 3.2 policy: finiteness first (a NaN mirror must not trip the
    // symmetry assert before the finiteness verdict), then symmetry.
    for (h) |v| {
        if (!std.math.isFinite(v)) return error.NonFinite;
    }
    assertSymmetric(h, k);

    const A = gpa.alloc(f64, k * k) catch return error.OutOfMemory; // Schur complement / Lc lower
    errdefer gpa.free(A);
    @memcpy(A, h);
    if (profile) {
        ph.check_ns = monoNs() - t0;
        t0 = monoNs();
    }

    try cholBlocked(A, k, pivot, threadTarget(), if (profile) &ph else null);
    if (profile) {
        ph.chol_ns = monoNs() - t0;
        t0 = monoNs();
    }

    const m = k / block_size;
    const d = gpa.alloc(f64, m * 256) catch return error.OutOfMemory;
    errdefer gpa.free(d);
    const l = gpa.alloc(f32, k * k) catch return error.OutOfMemory;
    errdefer gpa.free(l);

    var l64: []f64 = &.{};
    if (l64_out) |o| {
        l64 = gpa.alloc(f64, k * k) catch return error.OutOfMemory;
        o.* = &.{};
        errdefer gpa.free(l64);
    }

    finalizeBlocks(A, k, l, d, l64);
    if (profile) {
        ph.fin_ns = monoNs() - t0;
        const ms: f64 = 1e6;
        std.debug.print(
            "phases K={d} threads={d} panel={d} checks={d:.1} ms chol={d:.1} ms (panel={d:.1} norm={d:.1} trail={d:.1}) fin={d:.1} ms\n",
            .{ k, threadTarget(), panelTarget(), @as(f64, @floatFromInt(ph.check_ns)) / ms,
                @as(f64, @floatFromInt(ph.chol_ns)) / ms,
                @as(f64, @floatFromInt(ph.panel_ns)) / ms,
                @as(f64, @floatFromInt(ph.norm_ns)) / ms,
                @as(f64, @floatFromInt(ph.trail_ns)) / ms,
                @as(f64, @floatFromInt(ph.fin_ns)) / ms },
        );
    }
    if (l64_out) |o| o.* = l64;
    gpa.free(A);
    return .{ .k = k, .l = l, .d = d, .lambda = lambda, .mu = mu };
}

const PhaseTimes = struct {
    check_ns: u64 = 0,
    chol_ns: u64 = 0,
    panel_ns: u64 = 0,
    norm_ns: u64 = 0,
    trail_ns: u64 = 0,
    fin_ns: u64 = 0,
};

fn monoNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn assertSymmetric(h: []const f64, k: usize) void {
    for (0..k) |i| {
        for (i + 1..k) |j| {
            std.debug.assert(h[i * k + j] == h[j * k + i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Cholesky: blocked right-looking, fp64, lower triangle only.
// ---------------------------------------------------------------------------

/// Accumulate a[]*b[] in ascending index order (the determinism contract).
fn dot(a: []const f64, b: []const f64) f64 {
    var s: f64 = 0;
    var i: usize = 0;
    while (i < a.len) : (i += 1) s += a[i] * b[i];
    return s;
}

fn cholBlocked(A: []f64, k: usize, pivot: *usize, nthreads: usize, prof: ?*PhaseTimes) LdlError!void {
    const nb = panelTarget();
    var p: usize = 0;
    while (p < k) : (p += nb) {
        const pe = @min(p + nb, k);
        const t_panel = if (prof) |_| monoNs() else 0;

        // Diagonal block: unblocked ascending-column Cholesky on rows p..pe.
        for (p..pe) |j| {
            const ja = A[j * k ..][0..k];
            const dsum = dot(ja[p..j], ja[p..j]);
            const piv = ja[j] - dsum;
            if (!(piv > 0)) { // catches <= 0 and any NaN from overflow
                pivot.* = j;
                return error.NotPositive;
            }
            const ljj = @sqrt(piv);
            ja[j] = ljj;
            for (j + 1..pe) |i| {
                const ia = A[i * k ..][0..k];
                ia[j] = (ia[j] - dot(ia[p..j], ja[p..j])) / ljj;
            }
        }
        if (prof) |pp| pp.panel_ns += monoNs() - t_panel;

        // L21: normalize the panel body (rows pe..k) against the diagonal
        // block. Row i only depends on the finished diagonal block, so whole
        // row chunks go to workers; heavy rows first.
        if (pe < k) {
            const t_norm = if (prof) |_| monoNs() else 0;
            var lctx = PanelRows{ .A = A, .k = k, .p = p, .pe = pe };
            const items = chunkCount(k - pe, nthreads);
            lctx.chunk = chunkSize(k - pe, items);
            parallelFor(PanelRows, &lctx, items, nthreads, PanelRows.normalize);
            if (prof) |pp| pp.norm_ns += monoNs() - t_norm;
        }

        // Trailing update: A[pe.., pe..] -= L21 L21^T (lower triangle only),
        // tiled 64x64 tasks: each element has one owner, t ascends inside the
        // 128-column packed chunk loop, and the packed tiles keep L21
        // streaming at ~32 B/element instead of the 1 KB/element of an
        // untiled dot form.
        if (pe < k) {
            const t_trail = if (prof) |_| monoNs() else 0;
            const tiles = (k - pe + syrk_tile - 1) / syrk_tile;
            var tctx = Trailing{ .A = A, .k = k, .p = p, .pe = pe, .tiles = tiles };
            parallelFor(Trailing, &tctx, tiles * tiles, nthreads, Trailing.syrkTask);
            if (prof) |pp| pp.trail_ns += monoNs() - t_trail;
        }
    }
}

const PanelRows = struct {
    A: []f64,
    k: usize,
    p: usize,
    pe: usize,
    chunk: usize = 1,

    fn normalize(self: *PanelRows, item: usize) void {
        // items * chunk can exceed the span: clamp the tail window to empty.
        const top = @max(self.k -| item * self.chunk, self.pe);
        const lo = @max(self.pe, top -| self.chunk);
        for (lo..top) |i| {
            const ia = self.A[i * self.k ..][0..self.k];
            for (self.p..self.pe) |j| {
                const ja = self.A[j * self.k ..][0..self.k];
                ia[j] = (ia[j] - dot(ia[self.p..j], ja[self.p..j])) / ja[j];
            }
        }
    }
};
/// Cholesky row/col task tile (64x64); packing chunk width is 128 columns.
const syrk_tile: usize = 64;
const syrk_pack: usize = 128;

const Trailing = struct {
    A: []f64,
    k: usize,
    p: usize,
    pe: usize,
    tiles: usize,

    fn syrkTask(self: *Trailing, item: usize) void {
        // Heaviest tasks first (bottom rows, widest columns) for load balance.
        const ti = self.tiles - 1 - item / self.tiles;
        const tj = self.tiles - 1 - item % self.tiles;
        if (tj > ti) return; // upper-triangle task: no owner, no writes
        const rb = self.pe + ti * syrk_tile;
        const re = @min(rb + syrk_tile, self.k);
        const cb = self.pe + tj * syrk_tile;
        const ce = @min(cb + syrk_tile, self.k);

        var xi: [syrk_pack * syrk_tile]f64 = undefined;
        var xj: [syrk_pack * syrk_tile]f64 = undefined;
        // Accumulate into a stack tile so every FMA operand is provably
        // disjoint from the A destination and the inner loop vectorizes.
        var acc: [syrk_tile * syrk_tile]f64 = undefined;
        for (&acc) |*v| v.* = 0;
        var tb = self.p;
        while (tb < self.pe) : (tb += syrk_pack) {
            const tc = @min(syrk_pack, self.pe - tb);
            for (0..tc) |t| {
                for (rb..re, 0..) |i, ii| xi[t * syrk_tile + ii] = self.A[i * self.k + tb + t];
                for (cb..ce, 0..) |j, jj| xj[t * syrk_tile + jj] = self.A[j * self.k + tb + t];
            }
            for (0..tc) |t| {
                const xit = xi[t * syrk_tile ..][0..syrk_tile];
                const xjt = xj[t * syrk_tile ..][0..syrk_tile];
                for (0..syrk_tile) |ii| {
                    const vit = xit[ii];
                    const row = acc[ii * syrk_tile ..][0..syrk_tile];
                    for (0..syrk_tile) |jj| row[jj] -= vit * xjt[jj];
                }
            }
        }
        for (rb..re, 0..) |i, ii| {
            const aii = self.A[i * self.k ..][0..self.k];
            const row = acc[ii * syrk_tile ..][0..syrk_tile];
            const jend = @min(ce, i + 1); // j <= i: strictly the lower triangle
            // acc holds -sum_t L21[ii][t] * L21[jj][t]: add it back.
            for (cb..jend, 0..) |j, jj| aii[j] += row[jj];
        }
    }
};

fn chunkCount(span: usize, nthreads: usize) usize {
    return @min(span, nthreads * 8);
}

fn chunkSize(span: usize, items: usize) usize {
    return (span + items - 1) / items;
}

// ---------------------------------------------------------------------------
// Small parallel-for: ticketed work items, spawned per dispatch. Dispatches
// happen O(K/panel) times, so per-dispatch spawn cost is noise; it keeps the
// factorization free of pool state and race surface.
// ---------------------------------------------------------------------------

const Job = struct {
    next: std.atomic.Value(usize),
    items: usize,
    ctx: *anyopaque,
    exec: *const fn (ctx: *anyopaque, item: usize) void,
};

fn drainJob(job: *Job) void {
    while (true) {
        const i = job.next.fetchAdd(1, .monotonic);
        if (i >= job.items) break;
        job.exec(@ptrCast(@alignCast(job.ctx)), i);
    }
}

fn parallelFor(
    comptime T: type,
    ctx: *T,
    items: usize,
    nthreads: usize,
    comptime exec: fn (*T, usize) void,
) void {
    if (items == 0) return;
    if (items <= 1 or nthreads <= 1) {
        for (0..items) |i| exec(ctx, i);
        return;
    }
    const Wrapper = struct {
        fn call(ec: *anyopaque, item: usize) void {
            exec(@ptrCast(@alignCast(ec)), item);
        }
    };
    var job = Job{
        .next = std.atomic.Value(usize).init(0),
        .items = items,
        .ctx = ctx,
        .exec = Wrapper.call,
    };
    var threads: [max_threads]std.Thread = undefined;
    const want = @min(nthreads - 1, max_threads - 1, items - 1);
    var spawned: usize = 0;
    while (spawned < want) : (spawned += 1) {
        threads[spawned] = std.Thread.spawn(.{}, drainJob, .{&job}) catch {
            break; // out of resources: run with fewer helpers, items stay covered
        };
    }
    drainJob(&job);
    for (threads[0..spawned]) |t| t.join();
}

// ---------------------------------------------------------------------------
// Block LDL conversion: D_i = DL_i DL_i^T and L[:, i] = Lc[:, i] inv(DL_i).
// The diagonal-block solve runs columns descending, so the diagonal 16x16
// block lands on exactly I in fp64 (x - x and x / x only) and stays I after
// the fp32 round.
// ---------------------------------------------------------------------------

fn finalizeBlocks(A: []f64, k: usize, l: []f32, d: []f64, l64: []f64) void {
    const m = k / block_size;
    @memset(l[0 .. k * k], 0);
    if (l64.len != 0) @memset(l64, 0);

    for (0..m) |bi| {
        const R = bi * block_size;

        // Snapshot the Cholesky diagonal block (strict lower; upper masked).
        var dl: [block_size * block_size]f64 = undefined;
        for (0..block_size) |a| {
            for (0..block_size) |c| {
                dl[a * block_size + c] = if (a >= c) A[(R + a) * k + R + c] else 0;
            }
        }

        // D_i = DL_i DL_i^T, ascending t; the mirrored pair computes the same
        // dot in the same order, so the block is exactly symmetric.
        for (0..block_size) |a| {
            for (0..block_size) |c| {
                var s: f64 = 0;
                for (0..@min(a, c) + 1) |t| s += dl[a * block_size + t] * dl[c * block_size + t];
                d[bi * 256 + a * block_size + c] = s;
            }
        }

        // Solve X DL_i = Lc[:, block i] for rows R..K by backward substitution
        // in the block column index (DL is lower, so columns descending).
        var c: usize = block_size;
        while (c > 0) {
            c -= 1;
            const dcc = dl[c * block_size + c];
            for (R..k) |r| {
                const ra = A[r * k ..][0..k];
                var s: f64 = if (r >= R + c) ra[R + c] else 0; // Lc is upper-zero; stored A holds a mirror there
                var kk: usize = c + 1;
                while (kk < block_size) : (kk += 1) s -= ra[R + kk] * dl[kk * block_size + c];
                ra[R + c] = s / dcc;
            }
        }

        // Publish the solved block column: strictly-lower blocks from A, the
        // diagonal block as literals so it is exactly I in fp32 and fp64.
        for (bi..m) |bj| {
            const base = bj * block_size * k + R;
            for (0..block_size) |a| {
                for (0..block_size) |cc| {
                    if (bj == bi and a <= cc) {
                        if (a == cc) {
                            l[base + a * k + cc] = 1;
                            if (l64.len != 0) l64[base + a * k + cc] = 1;
                        }
                        continue;
                    }
                    const v = A[(bj * block_size + a) * k + R + cc];
                    l[base + a * k + cc] = @floatCast(v);
                    if (l64.len != 0) l64[base + a * k + cc] = v;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Rotations and sign flips (spec 3.4; oracle exl3_format.rotate).
// ---------------------------------------------------------------------------

/// FWHT along axis 0 and axis 1 in 128-blocks with the oracle's sign and
/// scale (exl3_format.rotate), matching it to 1e-12 (T-LDL-3).
pub fn rotateBoth(x: []f64, rows: usize, cols: usize) void {
    const hd = exl3_format.hadamard_dim;
    std.debug.assert(rows % hd == 0 and cols % hd == 0);
    std.debug.assert(x.len == rows * cols);
    rotateAxis0(x, rows, cols);
    rotateAxis1(x, rows, cols);
}

fn rotateAxis0(x: []f64, rows: usize, cols: usize) void {
    const hd = exl3_format.hadamard_dim;
    const inv = 1.0 / @sqrt(@as(f64, @floatFromInt(hd)));
    var rb: usize = 0;
    while (rb < rows) : (rb += hd) {
        var cb: usize = 0;
        while (cb < cols) : (cb += hd) {
            var h: usize = 1;
            while (h < hd) : (h *= 2) {
                var g: usize = 0;
                while (g < hd) : (g += 2 * h) {
                    for (0..h) |j| {
                        const r1 = (rb + g + j) * cols + cb;
                        const r2 = r1 + h * cols;
                        for (0..hd) |c| {
                            const a = x[r1 + c];
                            const b = x[r2 + c];
                            x[r1 + c] = a + b;
                            x[r2 + c] = a - b;
                        }
                    }
                }
            }
            for (0..hd) |i| {
                const base = (rb + i) * cols + cb;
                for (0..hd) |c| x[base + c] *= inv;
            }
        }
    }
}

fn rotateAxis1(x: []f64, rows: usize, cols: usize) void {
    const hd = exl3_format.hadamard_dim;
    const inv = 1.0 / @sqrt(@as(f64, @floatFromInt(hd)));
    for (0..rows) |r| {
        const row = x[r * cols ..][0..cols];
        var b: usize = 0;
        while (b < cols) : (b += hd) {
            fwht128(row[b .. b + hd]);
            for (row[b .. b + hd]) |*v| v.* *= inv;
        }
    }
}

/// In-place fast Walsh-Hadamard transform, Sylvester ordering (the oracle's
/// (-1)^popcount(i & j)); scale applied by the caller.
fn fwht128(v: []f64) void {
    const hd = exl3_format.hadamard_dim;
    std.debug.assert(v.len == hd);
    var h: usize = 1;
    while (h < hd) : (h *= 2) {
        var g: usize = 0;
        while (g < hd) : (g += 2 * h) {
            for (0..h) |j| {
                const a = v[g + j];
                const b = v[g + j + h];
                v[g + j] = a + b;
                v[g + j + h] = a - b;
            }
        }
    }
}

/// x[i][j] *= s[i] * s[j], x [k, k] row-major, s[i] in {-1, +1}.
pub fn flipSigns(x: []f64, k: usize, s: []const i8) void {
    std.debug.assert(x.len == k * k and s.len == k);
    for (0..k) |i| {
        const si: f64 = @floatFromInt(s[i]);
        for (0..k) |j| {
            const sj: f64 = @floatFromInt(s[j]);
            x[i * k + j] *= si * sj;
        }
    }
}

// ---------------------------------------------------------------------------
// Tuning knobs.
// ---------------------------------------------------------------------------
fn envUsize(name: [*:0]const u8) ?usize {
    const v = std.c.getenv(name) orelse return null;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch null;
}

fn envFlag(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

fn threadTarget() usize {
    const n = envUsize("EAQ_LDL_THREADS") orelse (std.Thread.getCpuCount() catch 1);
    return std.math.clamp(n, 1, max_threads);
}

fn panelTarget() usize {
    var nb = envUsize("EAQ_LDL_PANEL") orelse default_panel;
    nb = std.math.clamp(nb, block_size, 4096);
    nb -= nb % block_size;
    return @max(nb, block_size);
}

// ---------------------------------------------------------------------------
// Tests (T-LDL-1..4, spec section 6).
// ---------------------------------------------------------------------------

fn fillRand(r: std.Random, m: []f64) void {
    for (m) |*v| v.* = r.float(f64) * 2 - 1;
}

const SpdFill = struct {
    G: []f64,
    k: usize,
    seed: u64,

    fn fillRow(self: *SpdFill, item: usize) void {
        var prng = std.Random.DefaultPrng.init(self.seed ^ (item *% 0x9E37_79B9_7F4A_7C15));
        fillRand(prng.random(), self.G[item * self.k ..][0..self.k]);
    }
};

const SpdMake = struct {
    A: []f64,
    G: []f64,
    k: usize,
    tiles: usize,

    fn makeTile(self: *SpdMake, item: usize) void {
        const ti = self.tiles - 1 - item / self.tiles;
        const tj = self.tiles - 1 - item % self.tiles;
        if (tj > ti) return;
        const rb = ti * syrk_tile;
        const re = @min(rb + syrk_tile, self.k);
        const cb = tj * syrk_tile;
        const ce = @min(cb + syrk_tile, self.k);

        var gi: [syrk_pack * syrk_tile]f64 = undefined;
        var gj: [syrk_pack * syrk_tile]f64 = undefined;
        var tb: usize = 0;
        while (tb < self.k) : (tb += syrk_pack) {
            const tc = @min(syrk_pack, self.k - tb);
            for (0..tc) |t| {
                for (rb..re, 0..) |i, ii| gi[t * syrk_tile + ii] = self.G[i * self.k + tb + t];
                for (cb..ce, 0..) |j, jj| gj[t * syrk_tile + jj] = self.G[j * self.k + tb + t];
            }
            for (0..tc) |t| {
                const git = gi[t * syrk_tile ..][0..syrk_tile];
                const gjt = gj[t * syrk_tile ..][0..syrk_tile];
                const first = tb + t == 0;
                for (rb..re) |i| {
                    const vit = git[i - rb];
                    const jend = @min(ce, i + 1);
                    for (cb..jend) |j| {
                        const p = vit * gjt[j - cb];
                        if (first) self.A[i * self.k + j] = p else self.A[i * self.k + j] += p;
                    }
                }
            }
        }
        const s: f64 = @floatFromInt(self.k);
        for (rb..re) |i| {
            for (cb..@min(ce, i + 1)) |j| {
                const v = self.A[i * self.k + j] / s + (if (i == j) @as(f64, 0.05) else 0);
                self.A[i * self.k + j] = v;
                self.A[j * self.k + i] = v; // same value: exact mirror
            }
        }
    }
};

/// SPD matrix A = G G^T / K + 0.05 I: entries O(1), eigenvalues in [0.05, ~1].
/// Task (tile row, tile col) with tile row >= tile col owns each (i, j) once
/// and mirrors with the same value, so the result is exactly symmetric for
/// any thread count.
fn fillSpd(k: usize, seed: u64) ![]f64 {
    const gpa = std.testing.allocator;
    const G = try gpa.alloc(f64, k * k);
    defer gpa.free(G);
    var gf = SpdFill{ .G = G, .k = k, .seed = seed };
    parallelFor(SpdFill, &gf, k, threadTarget(), SpdFill.fillRow);
    const A = try gpa.alloc(f64, k * k);
    errdefer gpa.free(A);
    const tiles = (k + syrk_tile - 1) / syrk_tile;
    var mk = SpdMake{ .A = A, .G = G, .k = k, .tiles = tiles };
    parallelFor(SpdMake, &mk, tiles * tiles, threadTarget(), SpdMake.makeTile);
    return A;
}

/// Precompute (L D) row by row: (LD)[i][16b+u] = sum_t L[i][16b+t] D_b[t][u].
const LdProduct = struct {
    l64: []const f64,
    d: []const f64,
    LD: []f64,
    k: usize,

    fn ld(self: *LdProduct, item: usize) void {
        const i = item;
        for (0..self.k) |y| {
            const b = y / block_size;
            const u = y % block_size;
            var s: f64 = 0;
            for (0..block_size) |t| {
                s += self.l64[i * self.k + b * block_size + t] * self.d[b * 256 + t * block_size + u];
            }
            self.LD[i * self.k + y] = s;
        }
    }
};

const RecCheck = struct {
    LD: []const f64,
    l64: []const f64,
    expected: []const f64,
    k: usize,
    tiles: usize,
    maxdiff: std.atomic.Value(u64) = std.atomic.Value(u64).init(0), // nonneg f64 bits, order-preserving

    fn rec(self: *RecCheck, item: usize) void {
        const ti = item / self.tiles;
        const tj = item % self.tiles;
        if (tj < ti) return; // upper-triangle task (i <= j)
        const rb = ti * syrk_tile;
        const re = @min(rb + syrk_tile, self.k);
        const cb = tj * syrk_tile;
        const ce = @min(cb + syrk_tile, self.k);

        var ab: [syrk_pack * syrk_tile]f64 = undefined;
        var bb: [syrk_pack * syrk_tile]f64 = undefined;
        var acc: [syrk_tile * syrk_tile]f64 = undefined;
        var tb: usize = 0;
        while (tb < self.k) : (tb += syrk_pack) {
            const tc = @min(syrk_pack, self.k - tb);
            for (0..tc) |t| {
                for (rb..re, 0..) |i, ii| ab[t * syrk_tile + ii] = self.LD[i * self.k + tb + t];
                for (cb..ce, 0..) |j, jj| bb[t * syrk_tile + jj] = self.l64[j * self.k + tb + t];
            }
            for (0..tc) |t| {
                const at = ab[t * syrk_tile ..][0..syrk_tile];
                const bt = bb[t * syrk_tile ..][0..syrk_tile];
                const first = tb + t == 0;
                for (rb..re, 0..) |_, ii| {
                    const vit = at[ii];
                    const row = acc[ii * syrk_tile ..][0..syrk_tile];
                    for (cb..ce, 0..) |_, jj| {
                        const p = vit * bt[jj];
                        if (first) row[jj] = p else row[jj] += p;
                    }
                }
            }
        }
        var local: f64 = 0;
        for (rb..re, 0..) |i, ii| {
            for (cb..ce, 0..) |j, jj| {
                const diff = @abs(acc[ii * syrk_tile + jj] - self.expected[i * self.k + j]);
                if (diff > local) local = diff;
            }
        }
        const want: u64 = @bitCast(local); // local is a nonneg magnitude: bits order like the f64
        var cur = self.maxdiff.load(.monotonic);
        while (want > cur) {
            const prev = self.maxdiff.cmpxchgStrong(cur, want, .monotonic, .monotonic) orelse break;
            cur = prev;
        }
    }
};

fn reconstructMaxDiff(l64: []const f64, d: []const f64, expected: []const f64, k: usize) !f64 {
    const gpa = std.testing.allocator;
    const LD = try gpa.alloc(f64, k * k);
    defer gpa.free(LD);
    var ldc = LdProduct{ .l64 = l64, .d = d, .LD = LD, .k = k };
    parallelFor(LdProduct, &ldc, k, threadTarget(), LdProduct.ld);
    const tiles = (k + syrk_tile - 1) / syrk_tile;
    var rc = RecCheck{ .LD = LD, .l64 = l64, .expected = expected, .k = k, .tiles = tiles };
    parallelFor(RecCheck, &rc, tiles * tiles, threadTarget(), RecCheck.rec);
    const bits = rc.maxdiff.load(.monotonic);
    return @bitCast(bits);
}

test "T-LDL-1 random SPD 256: L D L^T = H' to 1e-12, diagonal blocks exactly I, D blocks SPD" {
    const gpa = std.testing.allocator;
    const k = 256;
    const A = try fillSpd(k, 0x7D1_1);
    defer gpa.free(A);

    var pivot: usize = undefined;
    var l64: []f64 = undefined;
    var bld = try blockLdlL64(gpa, A, k, 0.025, 1.0, &pivot, &l64);
    defer bld.deinit(gpa);
    defer gpa.free(l64);

    try expectEqual(bld.k, k);
    try expectEqual(bld.lambda, 0.025);
    try expectEqual(bld.mu, 1.0);

    const maxdiff = try reconstructMaxDiff(l64, bld.d, A, k);
    try expect(maxdiff < 1e-12);

    // Every diagonal 16x16 block of l is exactly the identity.
    for (0..k / block_size) |b| {
        for (0..block_size) |a| {
            for (0..block_size) |cc| {
                const v = bld.l[(b * block_size + a) * k + b * block_size + cc];
                const want: f32 = if (a == cc) @as(f32, 1) else @as(f32, 0);
                try expect(v == want);
            }
        }
    }

    // D blocks are SPD (each admits an fp64 Cholesky) and exactly symmetric.
    for (0..k / block_size) |b| {
        const db = bld.d[b * 256 ..][0..256];
        for (0..block_size) |a| {
            for (0..block_size) |cc| try expect(db[a * block_size + cc] == db[cc * block_size + a]);
        }
        var p2: usize = undefined;
        var small = try blockLdl(gpa, db, block_size, 0.025, 1.0, &p2);
        small.deinit(gpa);
    }
}

test "T-LDL-2 blockLdl timings at K = 5120, 6144, 17408 (run on hyper03 with EAQ_LDL_TIMING=1)" {
    if (!envFlag("EAQ_LDL_TIMING")) return;
    const gpa = std.testing.allocator;
    for ([_]usize{ 5120, 6144, 17408 }) |k| {
        const A = try fillSpd(k, 0x5EED + k);
        defer gpa.free(A);
        var pivot: usize = undefined;
        var l64: []f64 = undefined;

        const t0 = monoNs();
        var bld = try blockLdlL64(gpa, A, k, 0.025, 1.0, &pivot, &l64);
        const ns: f64 = @floatFromInt(monoNs() - t0);

        const kof: f64 = @floatFromInt(k);
        const gflops = kof * kof * kof / 3.0 / ns; // (K^3/3) flops per ns = GFLOP/s
        // The 60 s budget at K = 17408 is a spec section 3.3 report trigger,
        // not a pass/fail gate: hyper03 is shared, so wall-clock here moves
        // with co-tenant load (observed 43 s quiet, 197 s contended). The
        // verdict is printed; correctness below still hard-fails.
        const over = (k == 17408) and (ns >= 60 * @as(f64, std.time.ns_per_s));
        std.debug.print(
            "T-LDL-2 K={d} threads={d} panel={d} blockLdl {d:.1} ms ({d:.1} GFLOP/s fp64){s}\n",
            .{ k, threadTarget(), panelTarget(), ns / std.time.ns_per_ms, gflops,
                if (over) " [OVER 60s BUDGET: spec 3.3 note applies]" else "" },
        );

        // Verify the threaded factorization at scale before releasing memory.
        const maxdiff = try reconstructMaxDiff(l64, bld.d, A, k);
        try expect(maxdiff < 1e-12);
        bld.deinit(gpa);
        gpa.free(l64);
    }
}

test "T-LDL-3 rotateBoth equals exl3_format.rotate axis 0 then 1 to 1e-12; trace preserved through flipSigns+rotateBoth" {
    const gpa = std.testing.allocator;
    const rows = 256;
    const cols = 384;
    var prng = std.Random.DefaultPrng.init(0xA11_CE);
    var r = prng.random();

    const x = try gpa.alloc(f64, rows * cols);
    defer gpa.free(x);
    fillRand(r, x);
    const a = try gpa.dupe(f64, x);
    defer gpa.free(a);
    const b = try gpa.dupe(f64, x);
    defer gpa.free(b);

    exl3_format.rotate(a, rows, cols, 0);
    exl3_format.rotate(a, rows, cols, 1);
    rotateBoth(b, rows, cols);

    var maxdiff: f64 = 0;
    for (a, b) |av, bv| {
        const diff = @abs(av - bv);
        if (diff > maxdiff) maxdiff = diff;
    }
    try expect(maxdiff < 1e-12);

    // flipSigns then rotateBoth is an orthogonal congruence: trace preserved.
    const k = 256;
    const y = try gpa.alloc(f64, k * k);
    defer gpa.free(y);
    fillRand(r, y);
    const s = try gpa.alloc(i8, k);
    defer gpa.free(s);
    for (s) |*sv| sv.* = if (r.boolean()) @as(i8, -1) else @as(i8, 1);

    var trace0: f64 = 0;
    for (0..k) |i| trace0 += y[i * k + i];
    flipSigns(y, k, s);
    rotateBoth(y, k, k);
    var trace1: f64 = 0;
    for (0..k) |i| trace1 += y[i * k + i];

    try expectApproxEqAbs(trace1, trace0, 1e-9);
}

test "T-LDL-4 one NaN returns NonFinite; planted negative eigenvalue returns NotPositive with its pivot block" {
    const gpa = std.testing.allocator;
    const k = 256;
    {
        const A = try fillSpd(k, 0xC0FFEE);
        defer gpa.free(A);
        A[3 * k + 5] = std.math.nan(f64);
        A[5 * k + 3] = std.math.nan(f64); // keep the matrix symmetric around the poison
        var pivot: usize = 0xDEAD_BEEF;
        try expectError(error.NonFinite, blockLdl(gpa, A, k, 0.025, 1.0, &pivot));
    }
    {
        const B = try fillSpd(k, 0xC0FFEE); // same seed: identical matrix before the plant
        defer gpa.free(B);
        const d = 16 * 5 + 3; // pivot must surface at this row, block 5
        B[d * k + d] = -(B[d * k + d] + 50.0); // diag < 0 forces a negative pivot at d
        var pivot: usize = 0xDEAD_BEEF;
        try expectError(error.NotPositive, blockLdl(gpa, B, k, 0.025, 1.0, &pivot));
        try expectEqual(pivot, d);
        try expectEqual(pivot / 16, 5);
    }
}
