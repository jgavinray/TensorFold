//! The KL gate's statistic (eaq-ldlq-spec-fable.md §5.3), pure host fp64, no device types:
//! per position, from fp32 logits, `p = softmax(z_ref)`, `q = softmax(z_cand)` with
//! max-subtracted log-sum-exp over all vocab entries, `KL_t = Σ_v p_v (log p_v − log q_v)`,
//! plus the reverse KL, the top-1 agreement and the shift of `log q` at the reference's
//! argmax. Rows are independent, so the caller threads them one a worker; every aggregate
//! reduces in row order, so the numbers never depend on the thread schedule.
//!
//! A row's ref and cand slices may be the same memory (the `--self` pipeline check): every
//! term is then the same fp64 expression of the same input, so KL is exactly 0, the shift
//! exactly 0 and top-1 agrees by construction — the properties the self-check asserts.
const std = @import("std");

pub const Row = struct {
    kl: f64, // KL(p || q), nats
    kl_rev: f64, // KL(q || p)
    shift: f64, // max_v log q_v − log q at the reference's argmax ≥ 0: what greedy decoding sees
    top1: bool, // the two argmaxes agree (first index on a tie, both sides)
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
        try std.testing.expectApproxEqAbs(@as(f64, 0), r2.shift, 1e-12); // q still peaks at that value
    }
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
}
