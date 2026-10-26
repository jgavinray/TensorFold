//! The KL gate's statistic (eaq-ldlq-spec-fable.md §5.3 as corrected by
//! kl-order-ruling.md), pure host fp64, no device types: per position, from fp32
//! logits, `p = softmax(z_ref)`, `q = softmax(z_cand)` with max-subtracted
//! log-sum-exp over all vocab entries (both sides), `KL_t = Σ_v p_v (log p_v − log q_v)`
//! — forward KL with the reference weighting the sum, temperature 1, no clamp, no
//! floor on log q, no division by entropy or log V. The reverse KL is carried only as
//! the asymmetry column; the headline statistic is the forward direction, reported as
//! KL(ref||cand). Rows are independent, so the caller threads them one a worker; every
//! aggregate reduces in row order, so the numbers never depend on the thread schedule.
//!
//! A row's ref and cand slices may be the same memory (the `--self` pipeline check and
//! the ref-vs-ref identity run): every term is then the same fp64 expression of the
//! same input, so KL is exactly 0, the shift exactly 0 and top-1 agrees by
//! construction — the properties the self-check asserts.
//!
//! The ruling's invariants live here too: `Paired` is the per-position difference
//! block for two candidates against the same reference logits (mean and se of the
//! paired differences, top-1 discordance in McNemar form), `histCounts` the
//! per-position KL bucket counts that separate a two-valued statistic from a
//! heavy-tailed one.
const std = @import("std");

pub const Row = struct {
    kl: f64, // KL(p || q), nats
    kl_rev: f64, // KL(q || p)
    shift: f64, // max_v log q_v − log q at the reference's argmax ≥ 0: what greedy decoding sees
    top1: bool, // the two argmaxes agree (first index on a tie, both sides)
    a_ref: u32 = 0, // the reference's argmax (first index on a tie): the canary's and pairing's raw material
    a_cand: u32 = 0, // the candidate's argmax (first index on a tie)
};

/// One position's statistics over `n` vocab entries. Three passes: the two maxima (first
/// index wins a tie), the two sum-of-exponentials, then both KLs; fp64 throughout, so the
/// `--self` case cancels to exactly zero in every term.
pub fn rowKl(z_ref: []const f32, z_cand: []const f32) Row {
    std.debug.assert(z_ref.len == z_cand.len);
    const n: usize = z_ref.len;
    var m_p: f32 = -std.math.inf(f32);
    var m_q: f32 = -std.math.inf(f32);
    var a_p: usize = 0;
    var a_q: usize = 0;
    for (z_ref, 0..) |v, i| {
        if (v > m_p) {
            m_p = v;
            a_p = i;
        }
    }
    for (z_cand, 0..) |v, i| {
        if (v > m_q) {
            m_q = v;
            a_q = i;
        }
    }
    const mp: f64 = m_p;
    const mq: f64 = m_q;
    var s_p: f64 = 0;
    var s_q: f64 = 0;
    var i: usize = 0;
    while (i + 1 < n) : (i += 2) { // hand-unrolled by two: the hot loop over 248320 entries
        s_p += std.math.exp(@as(f64, z_ref[i]) - mp);
        s_q += std.math.exp(@as(f64, z_cand[i]) - mq);
        s_p += std.math.exp(@as(f64, z_ref[i + 1]) - mp);
        s_q += std.math.exp(@as(f64, z_cand[i + 1]) - mq);
    }
    if (i < n) {
        s_p += std.math.exp(@as(f64, z_ref[i]) - mp);
        s_q += std.math.exp(@as(f64, z_cand[i]) - mq);
    }
    const ls_p = @log(s_p);
    const ls_q = @log(s_q);
    const inv_p = 1.0 / s_p;
    const inv_q = 1.0 / s_q;
    var kl: f64 = 0;
    var rev: f64 = 0;
    for (z_ref, z_cand) |rp, rq| {
        const lp = @as(f64, rp) - mp - ls_p;
        const lq = @as(f64, rq) - mq - ls_q;
        const p = std.math.exp(@as(f64, rp) - mp) * inv_p;
        const qq = std.math.exp(@as(f64, rq) - mq) * inv_q;
        kl += p * (lp - lq);
        rev += qq * (lq - lp);
    }
    return .{
        .kl = kl,
        .kl_rev = rev,
        .shift = mq - @as(f64, z_cand[a_p]), // max log q − log q at argmax p (the −ls_q cancels)
        .top1 = a_p == a_q,
        .a_ref = @intCast(a_p),
        .a_cand = @intCast(a_q),
    };
}

/// The §5.3 table for a set of positions: mean (the gate statistic), standard error,
/// median / p99 / max (the tail), top-1 agreement and mean shift (greedy decoding),
/// reverse mean (asymmetry). n = 0 gives all-null; medians of an even count average the
/// two middle order statistics, p99 is the nearest-rank percentile (ceil(0.99 n)-th).
pub const Summary = struct {
    n: usize,
    mean: f64,
    se: f64, // standard error of the mean (sample SD / sqrt n); 0 when n < 2
    median: f64,
    p99: f64,
    max: f64,
    reverse_mean: f64,
    top1_frac: f64,
    shift_mean: f64,

    pub fn init(rows: []const Row) !Summary {
        if (rows.len == 0) return .{ .n = 0, .mean = 0, .se = 0, .median = 0, .p99 = 0, .max = 0, .reverse_mean = 0, .top1_frac = 0, .shift_mean = 0 };
        var sum: f64 = 0;
        var sum_rev: f64 = 0;
        var sum_shift: f64 = 0;
        var top1: usize = 0;
        var hi: f64 = -std.math.inf(f64);
        for (rows) |r| {
            sum += r.kl;
            sum_rev += r.kl_rev;
            sum_shift += r.shift;
            top1 += @intFromBool(r.top1);
            hi = @max(hi, r.kl);
        }
        const mean = sum / @as(f64, @floatFromInt(rows.len));
        var var_sum: f64 = 0;
        for (rows) |r| var_sum += (r.kl - mean) * (r.kl - mean);
        const se = if (rows.len >= 2) @sqrt(var_sum / @as(f64, @floatFromInt(rows.len - 1)) / @as(f64, @floatFromInt(rows.len))) else 0;
        const sorted = try std.heap.c_allocator.alloc(f64, rows.len);
        defer std.heap.c_allocator.free(sorted);
        for (rows, sorted) |r, *s| s.* = r.kl;
        std.mem.sort(f64, sorted, {}, std.sort.asc(f64));
        const n = sorted.len;
        const median = if (n % 2 == 1) sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
        const p99 = sorted[(99 * n + 99) / 100 - 1]; // nearest rank: ceil(0.99 n)-th order statistic
        return .{
            .n = n,
            .mean = mean,
            .se = se,
            .median = median,
            .p99 = p99,
            .max = hi,
            .reverse_mean = sum_rev / @as(f64, @floatFromInt(n)),
            .top1_frac = @as(f64, @floatFromInt(top1)) / @as(f64, @floatFromInt(n)),
            .shift_mean = sum_shift / @as(f64, @floatFromInt(n)),
        };
    }
};

/// The paired-difference block (kl-order-ruling.md invariant 3) for two candidates on
/// the same positions against the same reference logits: the mean and standard error
/// of the per-position difference `a − b`, and the top-1 discordance counts in
/// McNemar form (n01: a top-1 and b not, n10: the reverse). Ordering claims cite the
/// paired se and these counts — never quadrature of the marginal se, which throws
/// away the covariance two candidates share through the common reference. The
/// candidate the difference favours is the one with the smaller KL: mean_diff
/// negative by ≥ 3 se_diff favours a on the KL axis, and the sign of n10 − n01 names
/// it on the top-1 axis.
pub const Paired = struct {
    n: usize,
    mean_a: f64,
    mean_b: f64,
    mean_diff: f64, // mean_t (kl_a,t − kl_b,t)
    se_diff: f64, // standard error of the paired differences (sample SD / sqrt n)
    top1_a: usize,
    top1_b: usize,
    n01: usize,
    n10: usize,
    mcnemar_chi2: f64, // (n01−n10)² / (n01+n10); 0 when no position disagrees

    pub fn init(a: []const Row, b: []const Row) Paired {
        std.debug.assert(a.len == b.len);
        const n = a.len;
        if (n == 0) return .{ .n = 0, .mean_a = 0, .mean_b = 0, .mean_diff = 0, .se_diff = 0,
            .top1_a = 0, .top1_b = 0, .n01 = 0, .n10 = 0, .mcnemar_chi2 = 0 };
        var sa: f64 = 0;
        var sb: f64 = 0;
        var sd: f64 = 0;
        var ta: usize = 0;
        var tb: usize = 0;
        var n01: usize = 0;
        var n10: usize = 0;
        for (a, b) |ra, rb| {
            sa += ra.kl;
            sb += rb.kl;
            sd += ra.kl - rb.kl;
            ta += @intFromBool(ra.top1);
            tb += @intFromBool(rb.top1);
            n01 += @intFromBool(ra.top1 and !rb.top1);
            n10 += @intFromBool(rb.top1 and !ra.top1);
        }
        const nf: f64 = @floatFromInt(n);
        const mean_diff = sd / nf;
        var vs: f64 = 0;
        for (a, b) |ra, rb| {
            const d = ra.kl - rb.kl - mean_diff;
            vs += d * d;
        }
        const se_diff = if (n >= 2) @sqrt(vs / @as(f64, @floatFromInt(n - 1)) / nf) else 0;
        const disc = n01 + n10;
        const dd: i64 = @as(i64, @intCast(n01)) - @as(i64, @intCast(n10));
        const chi: f64 = if (disc > 0) @as(f64, @floatFromInt(@as(u64, @abs(dd)) * @as(u64, @abs(dd)))) / @as(f64, @floatFromInt(disc)) else 0;
        return .{
            .n = n, .mean_a = sa / nf, .mean_b = sb / nf, .mean_diff = mean_diff,
            .se_diff = se_diff, .top1_a = ta, .top1_b = tb, .n01 = n01, .n10 = n10,
            .mcnemar_chi2 = chi,
        };
    }
};

// ---- per-position KL histogram (kl-order-ruling.md decisive check 1) ----

/// Right-hand bin edges; bin 0 counts the exact zeros (the identity bin), bin i counts
/// (edges[i-1], edges[i]], the last bin counts everything above the final edge, and a
/// NaN falls into the last bin so it can never hide in the zero bin. A two-valued
/// histogram (mass only in bin 0 plus one interior bin, as the void table showed)
/// localizes the defect to the statistic kernel; a right-heavy-tailed shape is what a
/// real pair of softmax distributions has.
pub const hist_edges = [_]f64{ 1e-12, 1e-9, 1e-6, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0 };

/// The same edges as literals: the histogram dump prints its bin labels verbatim, so
/// they do not depend on how a float formatter happens to render 1e-12 today.
pub const hist_edge_names = [_][]const u8{
    "1e-12", "1e-9", "1e-6", "1e-3", "3e-3", "1e-2", "3e-2", "0.1", "0.3", "1.0", "3.0", "10.0",
};

pub const hist_bins = hist_edges.len + 2;

pub fn histCounts(rows: []const Row, counts: []u64) void {
    std.debug.assert(counts.len == hist_bins);
    @memset(counts, 0);
    for (rows) |r| {
        const v = r.kl;
        if (std.math.isNan(v)) { counts[hist_bins - 1] += 1; continue; }
        if (!(v > 0)) { counts[0] += 1; continue; } // exact zero (and a defensive home for −0)
        var b: usize = 1;
        while (b <= hist_edges.len and v > hist_edges[b - 1]) : (b += 1) {}
        counts[b] += 1;
    }
}

// ---- self-checks: runnable on any host (`xpu-eaq-kl --stats-selftest`), no device ----

/// A naive independent fp64 reference for the forward KL: materialize both softmaxes and
/// sum the definition. Quadratically slower per row by design — the test sizes are small.
fn naiveKl(z_ref: []const f32, z_cand: []const f32) !f64 {
    var p = try std.heap.page_allocator.alloc(f64, z_ref.len);
    defer std.heap.page_allocator.free(p);
    var q = try std.heap.page_allocator.alloc(f64, z_cand.len);
    defer std.heap.page_allocator.free(q);
    var mp: f64 = -std.math.inf(f64);
    var mq: f64 = -std.math.inf(f64);
    for (z_ref) |v| mp = @max(mp, @as(f64, v));
    for (z_cand) |v| mq = @max(mq, @as(f64, v));
    var sp: f64 = 0;
    var sq: f64 = 0;
    for (z_ref, 0..) |v, i| {
        p[i] = std.math.exp(@as(f64, v) - mp);
        sp += p[i];
    }
    for (z_cand, 0..) |v, i| {
        q[i] = std.math.exp(@as(f64, v) - mq);
        sq += q[i];
    }
    var t: f64 = 0;
    for (0 .. p.len) |i| {
        const pv = p[i] / sp;
        if (pv == 0) continue;
        t += pv * @log(pv * sq / q[i]);
    }
    return t;
}

test "rowKl: self rows are exactly zero, ties pick the first index both sides" {
    const gpa = std.testing.allocator;
    const z = try gpa.alloc(f32, 1000);
    defer gpa.free(z);
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const rnd = prng.random();
    for (z) |*v| v.* = @floatCast(rnd.float(f32) * 20 - 10);
    const r = rowKl(z, z);
    try std.testing.expectEqual(@as(f64, 0), r.kl);
    try std.testing.expectEqual(@as(f64, 0), r.kl_rev);
    try std.testing.expectEqual(@as(f64, 0), r.shift);
    try std.testing.expect(r.top1);
    try std.testing.expectEqual(r.a_ref, r.a_cand); // identity names the same token on both sides
    // an exact tie for the maximum: the first index wins, so shifting the duplicate stays top-1-equal
    var w = try gpa.alloc(f32, z.len);
    defer gpa.free(w);
    @memcpy(w, z);
    var am: usize = 0;
    for (z, 0..) |v, i| {
        if (v > z[am]) am = i;
    }
    w[0] = z[am]; // cand's max duplicated at an earlier index: the first-index rule splits the argmaxes
    if (am != 0) { // max at am; cand ties at 0 (earlier index wins in cand) → different argmax
        const r2 = rowKl(z, w);
        try std.testing.expect(!r2.top1);
        try std.testing.expectEqual(@as(u32, 0), r2.a_cand);
        try std.testing.expectApproxEqAbs(@as(f64, 0), r2.shift, 1e-12); // q still peaks at that value
    }
}

test "rowKl: argmaxes are the first winning index on each side" {
    const gpa = std.testing.allocator;
    const z = try gpa.alloc(f32, 500);
    const w = try gpa.alloc(f32, 500);
    defer {
        gpa.free(z);
        gpa.free(w);
    }
    var prng = std.Random.DefaultPrng.init(0xA11CE);
    const rnd = prng.random();
    for (z) |*v| v.* = @floatCast(rnd.float(f32) * 20 - 10);
    @memcpy(w, z);
    // plant the maxima so the expected argmaxes are known, first-index rule both sides
    z[7] = 100;
    z[201] = 100; // tie at 7: the first index wins
    w[33] = 99;
    const r = rowKl(z, w);
    try std.testing.expectEqual(@as(u32, 7), r.a_ref);
    try std.testing.expectEqual(@as(u32, 33), r.a_cand);
    try std.testing.expect(!r.top1);
}

test "rowKl: closed forms — shift invariance, two-category KL, reverse swap" {
    // uniform ref vs shifted-uniform cand: KL 0 (max-subtracted LSE ignores the shift)
    var z = [_]f32{ 0, 0, 0, 0, 0, 0 };
    var w = [_]f32{ 3.25, 3.25, 3.25, 3.25, 3.25, 3.25 };
    var r = rowKl(&z, &w);
    try std.testing.expectApproxEqAbs(@as(f64, 0), r.kl, 1e-12);
    // binary category: p uniform on {0,1}, q = softmax([0, b]) → KL = ln(1+e^b) − ln 2 − b/2
    const refs = [_]f32{ 0, 0 };
    const b: f64 = 0.7;
    const cands = [_]f32{ 0, @floatCast(b) };
    r = rowKl(&refs, &cands);
    const want = @log(1 + std.math.exp(@as(f64, cands[1]))) - @log(@as(f64, 2)) - @as(f64, cands[1]) / 2;
    try std.testing.expectApproxEqAbs(want, r.kl, 1e-12);
    // reverse swaps the arguments
    const rr = rowKl(&cands, &refs);
    try std.testing.expectApproxEqAbs(rr.kl, r.kl_rev, 1e-12);
}

test "rowKl: 4000-entry rows match the naive fp64 reference" {
    const gpa = std.testing.allocator;
    const z = try gpa.alloc(f32, 4000);
    const w = try gpa.alloc(f32, 4000);
    defer {
        gpa.free(z);
        gpa.free(w);
    }
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rnd = prng.random();
    for (z) |*v| v.* = @floatCast(rnd.float(f32) * 30 - 15);
    for (w) |*v| v.* = @floatCast(rnd.float(f32) * 30 - 15);
    const r = rowKl(z, w);
    const t = try naiveKl(z, w);
    const t_rev = try naiveKl(w, z);
    try std.testing.expectApproxEqRel(t, r.kl, 1e-9);
    try std.testing.expectApproxEqRel(t_rev, r.kl_rev, 1e-9);
    // Gibbs: KL of two softmaxes equals cross-entropy minus entropy of p
    var ce: f64 = 0;
    var ent: f64 = 0;
    const p = try gpa.alloc(f64, z.len);
    defer gpa.free(p);
    var m: f32 = -std.math.inf(f32);
    for (z) |v| m = @max(m, v);
    var s: f64 = 0;
    for (z, 0..) |v, i| {
        p[i] = std.math.exp(@as(f64, v) - @as(f64, m));
        s += p[i];
    }
    var mq: f32 = -std.math.inf(f32);
    for (w) |v| mq = @max(mq, v);
    var sq: f64 = 0;
    for (w) |v| sq += std.math.exp(@as(f64, v) - @as(f64, mq));
    for (0 .. p.len) |i| {
        const pv = p[i] / s;
        ce += -pv * (@as(f64, w[i]) - @as(f64, mq) - @log(sq)); // −Σ p·log q
        if (pv > 0) ent -= pv * @log(pv); // H(p)
    }
    try std.testing.expectApproxEqAbs(ce - ent, r.kl, 1e-9);
}

test "Summary: mean, SE, median, p99 against hand values" {
    const rows = [_]Row{
        .{ .kl = 0.01, .kl_rev = 0.02, .shift = 0.1, .top1 = true },
        .{ .kl = 0.03, .kl_rev = 0.03, .shift = 0.3, .top1 = false },
        .{ .kl = 0.02, .kl_rev = 0.01, .shift = 0.2, .top1 = true },
        .{ .kl = 0.99, .kl_rev = 1.01, .shift = 4.0, .top1 = true },
    };
    const s = try Summary.init(&rows);
    try std.testing.expectEqual(@as(usize, 4), s.n);
    const want_mean: f64 = (0.01 + 0.03 + 0.02 + 0.99) / 4.0;
    try std.testing.expectApproxEqAbs(want_mean, s.mean, 1e-12);
    // median of {0.01,0.02,0.03,0.99} = (0.02+0.03)/2; p99 nearest-rank = ceil(3.96)=4th = 0.99
    try std.testing.expectApproxEqAbs(0.025, s.median, 1e-12);
    try std.testing.expectApproxEqAbs(0.99, s.p99, 1e-12);
    try std.testing.expectApproxEqAbs(0.99, s.max, 1e-12);
    // sample SD of the four / 2: hand-check against the definition
    const mean: f64 = (0.01 + 0.03 + 0.02 + 0.99) / 4.0;
    var v: f64 = 0;
    for ([_]f64{ 0.01, 0.03, 0.02, 0.99 }) |k| v += (k - mean) * (k - mean);
    try std.testing.expectApproxEqAbs(@sqrt(v / 3 / 4), s.se, 1e-12);
    try std.testing.expectApproxEqAbs(0.75, s.top1_frac, 1e-12);
    const empty = try Summary.init(&[_]Row{});
    try std.testing.expectEqual(@as(usize, 0), empty.n);
}

test "Paired: hand differences, paired se, McNemar discordance" {
    const A = [_]Row{
        .{ .kl = 0.01, .kl_rev = 0.02, .shift = 0, .top1 = true },
        .{ .kl = 0.03, .kl_rev = 0.03, .shift = 0, .top1 = false },
        .{ .kl = 0.07, .kl_rev = 0.01, .shift = 0, .top1 = true },
        .{ .kl = 0.11, .kl_rev = 1.01, .shift = 0, .top1 = false },
    };
    const B = [_]Row{
        .{ .kl = 0.02, .kl_rev = 0.02, .shift = 0, .top1 = true },
        .{ .kl = 0.02, .kl_rev = 0.03, .shift = 0, .top1 = true },
        .{ .kl = 0.05, .kl_rev = 0.01, .shift = 0, .top1 = false },
        .{ .kl = 0.11, .kl_rev = 1.01, .shift = 0, .top1 = false },
    };
    const p = Paired.init(&A, &B);
    try std.testing.expectEqual(@as(usize, 4), p.n);
    // diffs −0.01, +0.01, +0.02, +0.00: mean +0.005 (B worse on the mean)
    try std.testing.expectApproxEqAbs(@as(f64, 0.005), p.mean_diff, 1e-12);
    const mean: f64 = 0.005;
    var v: f64 = 0;
    for ([_]f64{ -0.01, 0.01, 0.02, 0.00 }) |d| v += (d - mean) * (d - mean);
    try std.testing.expectApproxEqAbs(@sqrt(v / 3 / 4), p.se_diff, 1e-12);
    // row 1: A-not/B-yes → n10; row 2: A-yes/B-not → n01; rows 0 and 3 agree
    try std.testing.expectEqual(@as(usize, 1), p.n01);
    try std.testing.expectEqual(@as(usize, 1), p.n10);
    try std.testing.expectApproxEqAbs(@as(f64, 0), p.mcnemar_chi2, 1e-12); // balanced discordance
    // a candidate against itself: exact zero differences
    const s = Paired.init(&A, &A);
    try std.testing.expectEqual(@as(f64, 0), s.mean_diff);
    try std.testing.expectEqual(@as(f64, 0), s.se_diff);
    try std.testing.expectEqual(@as(usize, 0), s.n01);
    try std.testing.expectEqual(@as(usize, 0), s.n10);
    try std.testing.expectEqual(@as(f64, 0), s.mcnemar_chi2);
    try std.testing.expectEqual(s.mean_a, p.mean_a);
}

test "histCounts: identity all-zero bin, bucket placement, NaN cannot hide" {
    const rows = [_]Row{
        .{ .kl = 0, .kl_rev = 0, .shift = 0, .top1 = true },
        .{ .kl = 1e-13, .kl_rev = 0, .shift = 0, .top1 = true }, // bin 1: (0, 1e-12]
        .{ .kl = 0.05, .kl_rev = 0, .shift = 0, .top1 = false }, // (3e-2, 1e-1]: bin 8
        .{ .kl = 0.5, .kl_rev = 0, .shift = 0, .top1 = false }, // (0.3, 1.0]: bin 10
        .{ .kl = 1e9, .kl_rev = 0, .shift = 0, .top1 = false }, // last bin
        .{ .kl = std.math.nan(f64), .kl_rev = 0, .shift = 0, .top1 = false }, // last bin, loudly
    };
    var counts: [hist_bins]u64 = undefined;
    histCounts(&rows, &counts);
    var sum: u64 = 0;
    for (counts) |c| sum += c;
    try std.testing.expectEqual(@as(u64, rows.len), sum);
    try std.testing.expectEqual(@as(u64, 1), counts[0]);
    try std.testing.expectEqual(@as(u64, 1), counts[1]);
    try std.testing.expectEqual(@as(u64, 1), counts[8]);
    try std.testing.expectEqual(@as(u64, 1), counts[10]);
    try std.testing.expectEqual(@as(u64, 2), counts[hist_bins - 1]); // 1e9 and NaN together
    // an identity set is all bin 0
    const id = [_]Row{
        .{ .kl = 0, .kl_rev = 0, .shift = 0, .top1 = true },
        .{ .kl = 0, .kl_rev = 0, .shift = 0, .top1 = true },
        .{ .kl = 0, .kl_rev = 0, .shift = 0, .top1 = true },
    };
    histCounts(&id, &counts);
    try std.testing.expectEqual(@as(u64, 3), counts[0]);
    for (counts[1 ..]) |c| try std.testing.expectEqual(@as(u64, 0), c);
}

/// The same checks the unit tests make, callable from the binary without a test runner:
/// `xpu-eaq-kl --stats-selftest` proves the fp64 statistic against closed forms on any
/// host, no device. Any failed check returns an error.
pub fn selfCheck() !void {
    // identity: a row against itself is exactly zero everywhere
    const n: usize = 4096;
    const z = try std.heap.c_allocator.alloc(f32, n);
    defer std.heap.c_allocator.free(z);
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const rnd = prng.random();
    for (z) |*v| v.* = @floatCast(rnd.float(f32) * 40 - 20);
    const r = rowKl(z, z);
    try std.testing.expectEqual(@as(f64, 0), r.kl);
    try std.testing.expectEqual(@as(f64, 0), r.kl_rev);
    try std.testing.expectEqual(@as(f64, 0), r.shift);
    try std.testing.expect(r.top1);
    try std.testing.expectEqual(r.a_ref, r.a_cand);
    // shift invariance and the two-category closed form: KL(uniform2 || softmax[0,b]) = ln(1+e^b) − ln2 − b/2
    const refs = [_]f32{ 0, 0 };
    const b: f64 = 0.7;
    const cands = [_]f32{ 0, @floatCast(b) };
    const r2 = rowKl(&refs, &cands);
    const bf: f64 = @as(f64, cands[1]); // the f32-rounded b, as rowKl actually sees it
    try std.testing.expectApproxEqAbs(@log(1 + std.math.exp(bf)) - @log(@as(f64, 2)) - bf / 2, r2.kl, 1e-12);
    const r3 = rowKl(&cands, &refs);
    try std.testing.expectApproxEqAbs(r3.kl, r2.kl_rev, 1e-12);
    // against the naive reference on random rows
    const w = try std.heap.c_allocator.alloc(f32, n);
    defer std.heap.c_allocator.free(w);
    for (w) |*v| v.* = @floatCast(rnd.float(f32) * 40 - 20);
    const t = try naiveKl(z, w);
    const t_rev = try naiveKl(w, z);
    const r4 = rowKl(z, w);
    try std.testing.expectApproxEqRel(t, r4.kl, 1e-9);
    try std.testing.expectApproxEqRel(t_rev, r4.kl_rev, 1e-9);
    // aggregates against hand values
    const rows = [_]Row{
        .{ .kl = 0.01, .kl_rev = 0.02, .shift = 0.1, .top1 = true },
        .{ .kl = 0.03, .kl_rev = 0.03, .shift = 0.3, .top1 = false },
        .{ .kl = 0.02, .kl_rev = 0.01, .shift = 0.2, .top1 = true },
        .{ .kl = 0.99, .kl_rev = 1.01, .shift = 4.0, .top1 = true },
    };
    const s = try Summary.init(&rows);
    const want_mean: f64 = (0.01 + 0.03 + 0.02 + 0.99) / 4.0;
    try std.testing.expectApproxEqAbs(want_mean, s.mean, 1e-12);
    try std.testing.expectApproxEqAbs(0.025, s.median, 1e-12);
    try std.testing.expectApproxEqAbs(0.99, s.p99, 1e-12);
    try std.testing.expectApproxEqAbs(0.75, s.top1_frac, 1e-12);
    // the shape invariant of a real KL sample: mean at or above the median of the same rows
    try std.testing.expect(s.mean >= s.median);
    // the paired block against itself is exact zeros; against a shifted twin it is the hand difference
    const other = [_]Row{
        .{ .kl = 0.02, .kl_rev = 0.02, .shift = 0.1, .top1 = true },
        .{ .kl = 0.02, .kl_rev = 0.03, .shift = 0.3, .top1 = true },
        .{ .kl = 0.02, .kl_rev = 0.01, .shift = 0.2, .top1 = true },
        .{ .kl = 1.00, .kl_rev = 1.01, .shift = 4.0, .top1 = true },
    };
    const same = Paired.init(&rows, &rows);
    try std.testing.expectEqual(@as(f64, 0), same.mean_diff);
    try std.testing.expectEqual(@as(f64, 0), same.se_diff);
    try std.testing.expectEqual(@as(usize, 0), same.n01 + same.n10);
    const pr = Paired.init(&rows, &other);
    try std.testing.expectEqual(@as(usize, 4), pr.n);
    // diffs −0.01, +0.01, 0, −0.01: mean −0.0025
    try std.testing.expectApproxEqAbs(@as(f64, -0.0025), pr.mean_diff, 1e-12);
    try std.testing.expect(pr.se_diff > 0);
    try std.testing.expectEqual(@as(usize, 0), pr.n01); // other's top-1 is a superset
    try std.testing.expectEqual(@as(usize, 1), pr.n10); // rows[1] (top1=false) vs other (top1=true)
    // the histogram: an identity set is all bin 0, a mixed set sums to its size
    var counts: [hist_bins]u64 = undefined;
    histCounts(&rows, &counts);
    var sum: u64 = 0;
    for (counts) |c| sum += c;
    try std.testing.expectEqual(@as(u64, rows.len), sum);
    histCounts(&rows, &counts);
    try std.testing.expectEqual(@as(u64, 0), counts[0]); // no zeros in `rows`
    const zero = [_]Row{.{ .kl = 0, .kl_rev = 0, .shift = 0, .top1 = true }};
    histCounts(&zero, &counts);
    try std.testing.expectEqual(@as(u64, 1), counts[0]);
}
