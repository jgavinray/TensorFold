//! EXL3 trellis encoder: the host scalar reference for the mul1 tail-biting
//! Viterbi encoder of one 16x16 tile (spec eaq-viterbi-spec-fable.md sections 1-3, 5, 6,
//! with the cost arithmetic of the u32-rebasing amendment / eaq-rebase-ruling-fable.md Q1).
//! exl3_format.zig is the oracle: streamEnds, tileStates, codebookValue,
//! integerFp16Weight (mirrored in test T2), tilePlacement, unpack, Width.

const std = @import("std");
const exl3_format = @import("exl3_format.zig");

pub const Width = exl3_format.Width;
pub const Codebook = exl3_format.Codebook;

pub const inf: u32 = 0xFFFF_FFFF;
pub const q_bits: u5 = 12; // Q12 metric domain
pub const target_clamp: f32 = 4.0; // targets clamp to [-4, 4] before quantization

const q_scale: f32 = @floatFromInt(@as(u32, 1) << q_bits);

const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

/// b[p] = E(p) - E(p-1), E(-1) = 0, computed from exl3_format.streamEnds.
pub fn freshBits(w: Width, b: *[256]u8) void {
    std.debug.assert(w.halves >= 2 and w.halves <= 16);
    var ends: [256]u32 = undefined;
    exl3_format.streamEnds(w, &ends);
    var prev: u32 = 0;
    for (0..256) |p| {
        b[p] = @intCast(ends[p] - prev);
        prev = ends[p];
    }
}

/// floor(clamp(t, -4, 4) * 4096). Asserts t is finite.
pub fn quantTarget(t: f16) i16 {
    const tf: f32 = @floatCast(t);
    std.debug.assert(std.math.isFinite(tf));
    const c = @min(@max(tf, -target_clamp), target_clamp);
    return @intFromFloat(@floor(c * q_scale));
}

/// q[s] = floor(f32(codebookValue(.mul1, s)) * 4096) for every state.
pub const Metric = struct {
    q: [65536]i16,

    pub fn init() Metric {
        var m: Metric = .{ .q = undefined };
        for (&m.q, 0..) |*v, s| {
            const f: f32 = @floatCast(exl3_format.codebookValue(.mul1, @intCast(s)));
            v.* = @intFromFloat(@floor(f * q_scale));
        }
        return m;
    }
};

pub const Status = enum(u8) { optimal = 0, closed = 1, failed = 2 };

pub const Result = struct {
    states: [256]u16, // state[p] in stream order; tileStates of the packed words must equal it
    cost: u64, // the chosen closed path's branch-metric sum, Q24, recovered as A2 + R2[255][e*]; maxInt when failed
    open_cost: u64, // pass 1's unconstrained minimum, A1 + R1[255][e*]; equals cost when optimal
    pinned: u16, // e*
    status: Status,
    flags: u8, // bit0 pruned in pass 1, bit1 pruned in pass 2 (amendment); optimality uncertified while bit0 is set
};

pub const Scratch = struct {
    cost: [2][]u32, // 65536 each
    bp: []u8, // 256 * 2^(16 - min b)

    pub fn init(gpa: std.mem.Allocator, w: Width) !Scratch {
        var b: [256]u8 = undefined;
        freshBits(w, &b);
        var minb: u8 = 8;
        for (b) |v| minb = @min(minb, v);
        const gmax: usize = @as(usize, 1) << @intCast(@as(u8, 16) - minb);
        const c0 = try gpa.alloc(u32, 65536);
        errdefer gpa.free(c0);
        const c1 = try gpa.alloc(u32, 65536);
        errdefer gpa.free(c1);
        const bp = try gpa.alloc(u8, 256 * gmax);
        errdefer gpa.free(bp);
        return .{ .cost = .{ c0, c1 }, .bp = bp };
    }

    pub fn deinit(s: *Scratch, gpa: std.mem.Allocator) void {
        gpa.free(s.cost[0]);
        gpa.free(s.cost[1]);
        gpa.free(s.bp);
    }
};

pub const Sweep = struct { sum: u64, pruned: bool };

/// One Viterbi sweep over the 256 positions under the amendment's rebased u32
/// arithmetic (eaq-rebase-ruling-fable.md Q1, both sides verbatim). Costs stay
/// u32, INF = 0xFFFFFFFF. R[p] is the stored vector, R[-1] the unchanged initial
/// vector of each pass. Per step the group predecessor minimum m is reduced by
/// B(p-1), this pass's previous step minimum (B(-1) = 0), before the saturating
/// add; B(p) = min over t of R[p] for p < 255 is accumulated into a u64 sum A so
/// the caller recovers absolute costs as A + R[255][e]. The leader's group
/// carries rel = 0, so B(p) < 2^30 and the sum can never itself saturate. A
/// finite predecessor that saturates sets pruned: R[p][t] = C[p][t] - A(p) holds
/// for every unpruned state (one uniform offset preserves comparisons and ties,
/// INF is untouched), but optimality is then uncertified. Initial costs must sit
/// in cost[0]; 256 alternating steps land the final vector back in cost[0].
/// Groups per step are g = t >> b(p); with back-pointers given (pass 2) each
/// group records its winning h, one byte at bp[p * gmax + g].
fn viterbiSteps(scratch: *Scratch, m: *const Metric, b: *const [256]u8, qt: *const [256]i16, bp: ?[]u8, gmax: usize) Sweep {
    std.debug.assert(scratch.cost[0].len == 65536 and scratch.cost[1].len == 65536);
    var sum: u64 = 0;
    var pruned = false;
    var base: u32 = 0; // B(-1) = 0 in both passes
    var cur: usize = 0;
    for (0..256) |p| {
        const bb: u5 = @intCast(b[p]);
        const n_fresh: u32 = @as(u32, 1) << bb;
        const hshift: u5 = 16 - bb;
        const gcount: usize = @as(usize, 1) << @intCast(@as(u8, 16) - @as(u8, bb));
        const read = scratch.cost[cur];
        const write = scratch.cost[cur ^ 1];
        const qtp: i32 = qt[p];
        var g: usize = 0;
        while (g < gcount) : (g += 1) {
            // predecessor min, lowest h wins on ties (strictly-less update from h = 0)
            var best_h: u32 = 0;
            var best: u32 = read[g];
            var h: u32 = 1;
            while (h < n_fresh) : (h += 1) {
                const c = read[g | (h << hshift)];
                if (c < best) {
                    best = c;
                    best_h = h;
                }
            }
            if (bp) |bpbuf| bpbuf[p * gmax + g] = @intCast(best_h);
            const rel: u32 = if (best == inf) inf else best - base; // B(p-1) <= every finite m, no underflow
            const group_base: usize = g << @intCast(bb);
            var n: u32 = 0;
            while (n < n_fresh) : (n += 1) {
                const t: usize = group_base | @as(usize, @intCast(n));
                const e: i32 = @as(i32, m.q[t]) - qtp;
                const d: u32 = @intCast(@as(i32, e * e)); // < 2^30 (max |e| = 30528 with the clamp)
                const v: u32 = if (rel == inf) inf else rel +| d;
                pruned = pruned or (rel != inf and v == inf);
                write[t] = v;
            }
        }
        if (p < 255) { // B(255) is never applied
            var step_min: u32 = inf;
            for (write) |v| step_min = @min(step_min, v);
            std.debug.assert(step_min != inf); // the leader's group has rel = 0, so B(p) < 2^30
            sum += step_min;
            base = step_min;
        }
        cur ^= 1;
    }
    return .{ .sum = sum, .pruned = pruned };
}

/// One tile in matrix order [16][16] row-major (row = k, col = n), the layout unpack writes.
/// Two passes over the tail-biting trellis: free start, then pinned at the pass-1
/// argmin e*. Traceback through the pass-2 back-pointers yields a closed sequence.
/// Costs follow the amendment: each pass rebases by its own step minima, and the
/// absolute costs are recovered as sum + R[255][e*]; failed is a bug signal (Q2).
pub fn encodeTile(m: *const Metric, w: Width, tile: *const [256]f16, scratch: *Scratch) Result {
    std.debug.assert(w.halves >= 2 and w.halves <= 16);
    var b: [256]u8 = undefined;
    freshBits(w, &b);
    var minb: u8 = 8;
    for (b) |v| minb = @min(minb, v);
    const gmax: usize = @as(usize, 1) << @intCast(@as(u8, 16) - minb);
    std.debug.assert(scratch.bp.len >= 256 * gmax);

    // targets gathered through the placement closed form; quantTarget asserts finiteness
    var placement: [256]exl3_format.Placement = undefined;
    exl3_format.tilePlacement(&placement);
    var qt: [256]i16 = undefined;
    for (0..256) |p| qt[p] = quantTarget(tile[placement[p].row * 16 + placement[p].col]);

    // pass 1, free start: R[-1][s] = 0 for all s; the sweep rebases by its own
    // step minima and returns its u64 sum A1 and pass-1 pruned flag
    @memset(scratch.cost[0], 0);
    const p1 = viterbiSteps(scratch, m, &b, &qt, null, gmax);

    // e* = argmin R1[255]: lowest cost, then lowest state index. The rebase is
    // one uniform offset within the pass, so the argmin is the true one; the
    // leader argument bounds the step minima below 2^30, so the minimum is finite.
    var e_star: u16 = 0;
    var r1_end: u32 = scratch.cost[0][0];
    for (1..65536) |s| {
        const c = scratch.cost[0][s];
        if (c < r1_end) {
            r1_end = c;
            e_star = @intCast(s);
        }
    }
    const open_cost: u64 = p1.sum + r1_end;

    // pass 2, pinned at e*: R[-1][s] = 0 iff s == e*, else INF; store back-pointers
    for (scratch.cost[0], 0..) |*v, s| v.* = if (s == e_star) 0 else inf;
    const p2 = viterbiSteps(scratch, m, &b, &qt, scratch.bp, gmax);
    const end_cost: u32 = scratch.cost[0][e_star];

    // traceback from e* back to position 0; the pinned start forces e == e* on closure
    var states: [256]u16 = undefined;
    var e: u32 = e_star;
    var p: usize = 256;
    while (p > 0) {
        p -= 1;
        states[p] = @intCast(e);
        const bb: u5 = @intCast(b[p]);
        const g: usize = e >> bb;
        const h: u32 = scratch.bp[p * gmax + g];
        e = @as(u32, @intCast(g)) | (@as(u32, h) << (16 - bb));
    }

    const failed = end_cost == inf or e != e_star; // amendment: failed is a bug signal, abort naming the tile
    const cost: u64 = if (failed) std.math.maxInt(u64) else p2.sum + end_cost;
    const flags: u8 = @intFromBool(p1.pruned) | (@as(u8, @intFromBool(p2.pruned)) << 1);
    const status: Status = if (failed)
        .failed
    else if (cost == open_cost and flags & 1 == 0) .optimal else .closed;
    std.debug.assert(failed or flags & 1 != 0 or open_cost <= cost); // section 2's inequality, uncertified under a pass-1 flag
    return .{
        .states = states,
        .cost = cost,
        .open_cost = open_cost,
        .pinned = e_star,
        .status = status,
        .flags = flags,
    };
}

/// Fresh bits of a closed state sequence into the tile's words; tile.len == w.tileWords().
pub fn packStates(w: Width, states: *const [256]u16, tile: []i16) void {
    std.debug.assert(tile.len == w.tileWords());
    var ends: [256]u32 = undefined;
    exl3_format.streamEnds(w, &ends);
    var b: [256]u8 = undefined;
    freshBits(w, &b);
    const nwords32: usize = @as(usize, w.halves) * 4;
    var ring: [64]u32 = @splat(0);
    for (0..256) |p| {
        const bb: u4 = @intCast(b[p]);
        const n: u16 = states[p] & ((@as(u16, 1) << bb) - 1);
        const base: u32 = ends[p] - bb;
        for (0..bb) |k| {
            const bit = (n >> @intCast(bb - 1 - @as(u4, @intCast(k)))) & 1;
            const i = base + @as(u32, @intCast(k));
            ring[i >> 5] |= @as(u32, bit) << @intCast(31 - (i & 31));
        }
    }
    // the int16 pairing is streamBit's little-endian uint32 view (exl3_format.zig:96-98)
    for (0..nwords32) |wi| {
        tile[2 * wi] = @bitCast(@as(u16, @truncate(ring[wi])));
        tile[2 * wi + 1] = @bitCast(@as(u16, @truncate(ring[wi] >> 16)));
    }
}

/// W_q [K, N] fp16 -> trellis int16 [K/16, N/16, 8*halves], k-major, the inverse of
/// exl3_format.unpack. mul1 only. results, when given, receives one Result per tile
/// in the same order. A tile whose encode is `failed` aborts the quantization (spec
/// section 2): the tile index lands in failed_tile, when given, and the error is printed
/// naming the tile; error.TileFailed is returned.
pub fn pack(m: *const Metric, w: Width, wq: []const f16, tiles_k: usize, tiles_n: usize, trellis: []i16, scratch: *Scratch, results: ?[]Result, failed_tile: ?*usize) !void {
    std.debug.assert(tiles_k > 0 and tiles_n > 0);
    const k = tiles_k * 16;
    const n = tiles_n * 16;
    std.debug.assert(wq.len == k * n);
    const words = w.tileWords();
    std.debug.assert(trellis.len == tiles_k * tiles_n * words);
    if (results) |r| std.debug.assert(r.len == tiles_k * tiles_n);
    var placement: [256]exl3_format.Placement = undefined;
    exl3_format.tilePlacement(&placement);
    var targets: [256]f16 = undefined;
    for (0..tiles_k) |tk| {
        for (0..tiles_n) |tn| {
            const ti = tk * tiles_n + tn;
            // matrix order, the layout unpack writes: targets[row*16 + col] = the wq element
            for (0..256) |p| targets[placement[p].row * 16 + placement[p].col] = wq[(tk * 16 + @as(usize, placement[p].row)) * n + tn * 16 + @as(usize, placement[p].col)];
            const res = encodeTile(m, w, &targets, scratch);
            if (res.status == .failed) { // spec section 2: the host aborts the quantization naming the tile
                if (failed_tile) |ft| ft.* = ti;
                std.debug.print("exl3_encode: halves {d}: tile {d} (tk {d}, tn {d}) failed: cost={x} open={x} pinned={x}\n", .{ w.halves, ti, tk, tn, res.cost, res.open_cost, res.pinned });
                return error.TileFailed;
            }
            packStates(w, &res.states, trellis[ti * words ..][0..words]);
            if (results) |r| r[ti] = res;
        }
    }
}

// ---------------------------------------------------------------------------
// test helpers

/// A closed state sequence read twice around a random ring: state[p] is the 16 ring
/// bits ending at E(p), MSB first, exactly the tileStates window (mirrors exl3_format.zig:108-112).
/// Public: the device probe plants exact-codebook tie tiles with it (T7, T-LDLQ-6).
pub fn closedFromRing(r: std.Random, halves: u16, states: *[256]u16) void {
    std.debug.assert(halves >= 2 and halves <= 16);
    var bits: [2048]u1 = undefined; // 128 * halves <= 2048
    for (0..@as(usize, 128) * halves) |i| bits[i] = @truncate(r.int(u32) >> 17);
    var ends: [256]u32 = undefined;
    exl3_format.streamEnds(.{ .halves = halves }, &ends);
    const ring: u32 = @intCast(128 * @as(u32, halves));
    for (0..256) |p| {
        const start = (ends[p] + ring - 16) % ring;
        var s: u16 = 0;
        for (0..16) |bb| s = (s << 1) | @as(u16, bits[(start + @as(u32, @intCast(bb))) % ring]);
        states[p] = s;
    }
}

fn uniform01(r: std.Random) f32 {
    return (@as(f32, @floatFromInt(r.int(u32))) + 1.0) / 4294967296.0; // (0, 1]
}

/// Box-Muller standard normal: no dependency on a Gaussian distribution API.
fn gauss(r: std.Random) f32 {
    const za = uniform01(r);
    const zb = uniform01(r) - 0.5;
    return @sqrt(-2.0 * @log(za)) * @cos(2.0 * std.math.pi * zb);
}

/// Mirror of exl3_format.integerFp16Weight (exl3_format.zig:58-69): the exact-integer
/// weight r with value r * 2^-18, the form the kernels and spec section 1 use.
fn mirrorIntegerWeight(sum: u32) i32 {
    const n: i32 = @as(i32, @intCast(sum)) * 1774 - 2721792;
    const mag: u32 = @abs(n);
    std.debug.assert(mag != 0);
    const b: u32 = 31 - @clz(mag);
    const q: u32 = if (b > 10) @as(u32, 1) << @intCast(b - 10) else 1;
    const rem: u32 = mag & (q - 1);
    var r: u32 = mag - rem;
    if (rem > q >> 1 or (rem == q >> 1 and (r & q) != 0)) r += q;
    return if (n < 0) -@as(i32, @intCast(r)) else @as(i32, @intCast(r));
}

// ---------------------------------------------------------------------------
// T0: fresh bits

test "T0 freshBits equals the closed forms of section 1 and sums to the ring length for halves 2..16" {
    for (2..17) |h| {
        const w = Width{ .halves = @intCast(h) };
        var b: [256]u8 = undefined;
        freshBits(w, &b);
        var sum: usize = 0;
        for (0..256) |p| {
            const want: u8 = if (h % 2 == 0)
                @intCast(h / 2)
            else if (p % 2 == 0) @intCast((h - 1) / 2) else @intCast((h - 1) / 2 + 1);
            try expectEqual(want, b[p]);
            sum += b[p];
        }
        try expectEqual(@as(usize, 128) * h, sum);
    }
}

// ---------------------------------------------------------------------------
// T1: pack/decode round trip, and its failure alarm

test "T1 packStates then tileStates reproduces a closed sequence at every width; a flipped state breaks the round trip" {
    for (2..17) |h| {
        const w = Width{ .halves = @intCast(h) };
        var prng = std.Random.DefaultPrng.init(0xE4E5 + h);
        const r = prng.random();
        var states: [256]u16 = undefined;
        closedFromRing(r, @intCast(h), &states);
        // the sequence obeys the transition rule, including the tail-biting wrap
        var b: [256]u8 = undefined;
        freshBits(w, &b);
        for (0..256) |p| {
            const bb: u4 = @intCast(b[p]);
            const prev = states[(p + 255) % 256];
            const fresh = states[p] & ((@as(u16, 1) << bb) - 1);
            try expectEqual(states[p], ((prev << bb) & 0xFFFF) | fresh);
        }
        var tile: [128]i16 = undefined;
        var dec: [256]u32 = undefined;
        packStates(w, &states, tile[0..w.tileWords()]);
        exl3_format.tileStates(w, tile[0..w.tileWords()], &dec);
        for (0..256) |p| try expectEqual(@as(u32, states[p]), dec[p]);

        // flip the lowest fresh bit of one position: the ring bit changes inside the
        // windows of later positions too, so the decoded sequence cannot match
        const q: usize = @intCast(r.intRangeAtMost(u32, 0, 255));
        var flipped = states;
        flipped[q] ^= 1;
        packStates(w, &flipped, tile[0..w.tileWords()]);
        exl3_format.tileStates(w, tile[0..w.tileWords()], &dec);
        var differs = false;
        for (0..256) |p| {
            if (dec[p] != flipped[p]) differs = true;
        }
        try expect(differs);
    }
}

// ---------------------------------------------------------------------------
// T1b: the closure claim of section 3. A sequence obeying the transitions for
// p = 1..255 with state[0] outside succ(state[255]) fails the round trip only
// where the window leans on the wrap-around: position 0, and the first
// ceil(16/b) positions whose window reaches into the ring tail.

test "T1b a non-closed sequence mismatches only at position 0's closure and the ring-tail windows of the first ceil(16/b) positions" {
    for (2..17) |h| {
        const w = Width{ .halves = @intCast(h) };
        var b: [256]u8 = undefined;
        freshBits(w, &b);
        var ends: [256]u32 = undefined;
        exl3_format.streamEnds(w, &ends);
        var prng = std.Random.DefaultPrng.init(0x1B1E + h);
        const r = prng.random();
        const b0: u4 = @intCast(b[0]);

        // build forward: state[0] random, state[p] = (state[p-1] << b(p)) | fresh for
        // p = 1..255, so the transitions hold for p >= 1 by construction
        var states: [256]u16 = undefined;
        states[0] = r.int(u16);
        for (1..256) |p| {
            const bb: u4 = @intCast(b[p]);
            const mask: u16 = (@as(u16, 1) << bb) - 1;
            states[p] = ((states[p - 1] << bb) & 0xFFFF & ~mask) | (@as(u16, @truncate(r.int(u16))) & mask);
        }
        // force state[0] outside succ(state[255]): b(0) <= 8 < 16, so the shared prefix
        // owns bit 15 and flipping it escapes the successor set; b(1) >= 1, so bit 15 of
        // state[0] is shifted out of state[1] and the p = 1 transition survives
        const mask0: u16 = (@as(u16, 1) << b0) - 1;
        if ((states[0] & ~mask0) == ((states[255] << b0) & 0xFFFF & ~mask0)) states[0] ^= @as(u16, 1) << 15;
        try expect((states[0] & ~mask0) != ((states[255] << b0) & 0xFFFF & ~mask0));
        for (1..256) |p| {
            const bb: u4 = @intCast(b[p]);
            const mask: u16 = (@as(u16, 1) << bb) - 1;
            try expectEqual(states[p] & ~mask, (states[p - 1] << bb) & 0xFFFF & ~mask);
        }

        // position p >= 1 decodes from the fresh bits of the positions its window
        // covers; the transition-consistent states rebuild those exactly, so only a
        // window that wraps into the ring tail (E(p) < 16) can mismatch: there the
        // stream holds the tail positions' fresh bits where the forward chain kept
        // state[0]'s pre-shift bits. Position 0 is the closure claim itself: its
        // decoded prefix is the state[255] chain, equal to states[0] only when the
        // sequence closes - planted false here. All mismatches sit inside the first
        // ceil(16/b) positions (the E(p) < 16 region) plus that position-0 failure.
        var tile: [128]i16 = undefined;
        packStates(w, &states, tile[0..w.tileWords()]);
        var dec: [256]u32 = undefined;
        exl3_format.tileStates(w, tile[0..w.tileWords()], &dec);
        var n_bad: usize = 0;
        for (0..256) |p| {
            const bad = dec[p] != states[p];
            if (p == 0) try expect(bad) else if (ends[p] >= 16) try expect(!bad);
            if (bad) n_bad += 1;
        }
        var wraps: usize = 0;
        for (1..256) |p| {
            if (ends[p] < 16) wraps += 1;
        }
        try expect(n_bad <= wraps + 1); // position 0 plus the wrapped-window region only
    }
}

// ---------------------------------------------------------------------------
// T2: the Q12 table ties the oracle's fp16 weight to the kernels' integer path

test "T2 Metric q equals integerFp16Weight >> 6 on all 65536 states and the realized Q12 extremes are -14144 and 13712" {
    const m = Metric.init();
    var min_q: i16 = std.math.maxInt(i16);
    var max_q: i16 = std.math.minInt(i16);
    for (0..65536) |s| {
        const st: u16 = @intCast(s);
        const x: u32 = @truncate(@as(u64, st) *% exl3_format.Mul1Mul);
        const sum: u32 = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255);
        const r: i32 = mirrorIntegerWeight(sum);
        try expectEqual(@as(i16, @intCast(r >> 6)), m.q[st]); // arithmetic shift is floor
        // the oracle's fp16 weight quantizes to the same integer
        const f: f32 = @floatCast(exl3_format.codebookValue(.mul1, st));
        try expectEqual(@as(i16, @intFromFloat(@floor(f * q_scale))), m.q[st]);
        min_q = @min(min_q, m.q[st]);
        max_q = @max(max_q, m.q[st]);
    }
    try expectEqual(@as(i16, -14144), min_q);
    // Spec section 1's table hand-derives a max of 14128 at S = 1020, but no mul1
    // state reaches S = 1020: the byte sum of s * Mul1Mul is realizable only on
    // [0, 1005], where the oracle's codebookValue tops out at Q12 13712 (one state,
    // verified by the all-states equality above tying the table to the oracle).
    // The oracle wins (section 0), so the test asserts the realized extremes.
    try expectEqual(@as(i16, 13712), max_q);
    try expectEqual(@as(i16, -14144), m.q[0]); // S = 0 is state 0
    // the u32 cost bound of section 1: max |e| with the clamp is 14144 + 4*4096
    const qabs: i64 = @max(@abs(@as(i32, min_q)), @abs(@as(i32, max_q)));
    const emax: i64 = qabs + 16384;
    try expect(emax == 30528 and emax * emax < (1 << 30));
    try expectEqual(@as(i16, 16384), quantTarget(@bitCast(@as(u16, 0x4400)))); // fp16 4.0
    try expectEqual(@as(i16, -16384), quantTarget(@bitCast(@as(u16, 0xC400)))); // fp16 -4.0
}

// ---------------------------------------------------------------------------
// T3: planted path

test "T3 encoding a planted tile: open_cost is 0 always, the pinned state is the pass-1 argmin, and cost 0 plus optimality whenever no zero-cost end state sits below the planted one" {
    const m = Metric.init();
    var placement: [256]exl3_format.Placement = undefined;
    exl3_format.tilePlacement(&placement);
    var hijack_widths: usize = 0;
    for (2..17) |h| {
        const w = Width{ .halves = @intCast(h) };
        var scratch = try Scratch.init(std.testing.allocator, w);
        defer scratch.deinit(std.testing.allocator);
        var prng = std.Random.DefaultPrng.init(@as(u64, h) * 1_000_003 + 17);
        var states: [256]u16 = undefined;
        closedFromRing(prng.random(), @intCast(h), &states);
        var twq: [256]f16 = undefined;
        for (0..256) |p| twq[placement[p].row * 16 + placement[p].col] = exl3_format.codebookValue(.mul1, states[p]);

        // C[255] of pass 1, replicated by the test itself (the pass is deterministic):
        // pass 1's tie rule (lowest cost, then lowest state) pins pass 2 to the LOWEST
        // zero-cost end state, which need not be the planted one nor lie on any zero-cost
        // closed path - a legitimate heuristic gap (section 2 `closed`), so `cost == 0`
        // is asserted only when all of C[255] below the planted end state is positive.
        // The old group-only redraw hid every other lower zero state.
        var b: [256]u8 = undefined;
        freshBits(w, &b);
        var minb: u8 = 8;
        for (b) |v| minb = @min(minb, v);
        const gmax: usize = @as(usize, 1) << @intCast(@as(u8, 16) - minb);
        var qt: [256]i16 = undefined;
        for (0..256) |p| qt[p] = m.q[states[p]]; // floor(codebookValue * 4096) == m.q[s] (T2)
        @memset(scratch.cost[0], 0);
        _ = viterbiSteps(&scratch, &m, &b, &qt, null, gmax);
        var zero_below = false;
        var exp_pin: u16 = 0; // the pass-1 argmin over the replica: lowest cost, then lowest state
        var best: u32 = scratch.cost[0][0];
        for (scratch.cost[0], 0..) |c, s| {
            if (c < best) {
                best = c;
                exp_pin = @intCast(s);
            }
            if (s < states[255] and c == 0) zero_below = true;
        }

        const res = encodeTile(&m, w, &twq, &scratch);
        try expectEqual(@as(u64, 0), res.open_cost); // the planted zero path rebases by 0 at every step: A1 = 0 and R1[255][e*] = 0
        try expect(res.status != .failed);
        try expectEqual(@as(u32, exp_pin), @as(u32, res.pinned)); // the pass-1 tie rule, exercised on real ties
        var tile: [128]i16 = undefined;
        packStates(w, &res.states, tile[0..w.tileWords()]);
        var dec: [256]u32 = undefined;
        exl3_format.tileStates(w, tile[0..w.tileWords()], &dec);
        for (0..256) |p| try expectEqual(@as(u32, res.states[p]), dec[p]); // any non-failed result is closed
        if (zero_below) {
            hijack_widths += 1;
        } else {
            try expectEqual(@as(u32, 0), res.cost);
            try expectEqual(Status.optimal, res.status);
            // decoded values equal the targets (states need not match: many states share a value)
            var out: [256]f16 = undefined;
            exl3_format.unpack(w, .mul1, tile[0..w.tileWords()], 1, 1, &out);
            for (out, twq) |got, want| try expectEqual(@as(u16, @bitCast(got)), @as(u16, @bitCast(want)));
        }
    }
    std.debug.print("T3: {d}/15 widths had a zero-cost end state below the planted one (heuristic-gap branch taken there)\n", .{hijack_widths});
}

// ---------------------------------------------------------------------------
// T4: gaussian tiles round-trip through the packer at every width

test "T4 encodeTile at each width: packStates then tileStates equals Result.states, open_cost <= cost, not failed" {
    const m = Metric.init();
    for (2..17) |h| {
        const w = Width{ .halves = @intCast(h) };
        var scratch = try Scratch.init(std.testing.allocator, w);
        defer scratch.deinit(std.testing.allocator);
        var prng = std.Random.DefaultPrng.init(0xBEEF00 + h);
        const r = prng.random();
        for (0..2) |trial| {
            var twq: [256]f16 = undefined;
            for (&twq) |*v| v.* = @floatCast(gauss(r));
            const res = encodeTile(&m, w, &twq, &scratch);
            try expect(res.status != .failed);
            if (res.flags & 1 == 0) try expect(res.open_cost <= res.cost); // amendment: uncertified under a pass-1 flag
            if (res.status == .optimal) try expectEqual(res.cost, res.open_cost);
            var tile: [128]i16 = undefined;
            packStates(w, &res.states, tile[0..w.tileWords()]);
            var dec: [256]u32 = undefined;
            exl3_format.tileStates(w, tile[0..w.tileWords()], &dec);
            for (0..256) |p| try expectEqual(@as(u32, res.states[p]), dec[p]);
            _ = trial;
        }
    }
}

// ---------------------------------------------------------------------------
// T4+: clamped-outlier tiles under the amendment (fable review, ruling Q1/Q2).
// The all-+4, all--4, alternating +-4 and Hadamard-sign +-4 classes at halves
// 2, 3, 4, 6, 8, 16 encode closed (they are legal inside the clamp; failed is a
// bug signal), round-trip through the packer, and carry cost == the u64 sum of
// squared Q12 errors recomputed from the returned states - the exactness claim.

test "T4+ clamped-outlier tiles at halves 2,3,4,6,8,16: closed, round trip, cost equals the recomputed u64 path sum" {
    const m = Metric.init();
    var placement: [256]exl3_format.Placement = undefined;
    exl3_format.tilePlacement(&placement);
    for ([_]u16{ 2, 3, 4, 6, 8, 16 }) |hh| {
        const w = Width{ .halves = hh };
        var scratch = try Scratch.init(std.testing.allocator, w);
        defer scratch.deinit(std.testing.allocator);
        for (0..4) |shape| { // all +4, all -4, alternating, Hadamard signs
            var twq: [256]f16 = undefined;
            for (&twq, 0..) |*v, at| {
                const sign: f32 = switch (shape) {
                    0, 1 => 1.0,
                    2 => if (at % 2 == 0) @as(f32, 1.0) else -1.0,
                    else => if ((@popCount(@as(u8, @truncate(at / 16)) & @as(u8, @truncate(at))) & 1) == 0) @as(f32, 1.0) else -1.0,
                };
                v.* = @floatCast(if (shape == 1) @as(f32, -4.0) else 4.0 * sign);
            }
            const res = encodeTile(&m, w, &twq, &scratch);
            try expect(res.status != .failed); // legal input must not light the bug signal (ruling Q2)
            var sum: u64 = 0;
            for (0..256) |p| {
                const e: i64 = @as(i32, m.q[res.states[p]]) - @as(i32, quantTarget(twq[placement[p].row * 16 + placement[p].col]));
                sum += @as(u64, @intCast(e * e));
            }
            try expectEqual(sum, res.cost); // exact branch-metric sum of the returned path (the amendment)
            if (res.flags & 1 == 0) try expect(res.open_cost <= res.cost);
            var tile: [128]i16 = undefined;
            packStates(w, &res.states, tile[0..w.tileWords()]);
            var dec: [256]u32 = undefined;
            exl3_format.tileStates(w, tile[0..w.tileWords()], &dec);
            for (0..256) |p| try expectEqual(@as(u32, res.states[p]), dec[p]); // closed
        }
    }
}

// ---------------------------------------------------------------------------
// T5: pack then unpack on unit-variance grids at 3 bits

test "T5 pack then unpack of [128,128] and non-square [128,64] grids at 3 bits: RMS < 0.2 and the trellis decodes to the Results" {
    const m = Metric.init();
    const w = Width{ .halves = 6 }; // 3 bits per weight
    // the spec's square 8x8 grid, and a non-square 8x4 whose aspect makes a tk/tn
    // swap in the k-major order break the round trip (fable review)
    const Grid = struct { tiles_k: usize, tiles_n: usize };
    const grids = [_]Grid{ .{ .tiles_k = 8, .tiles_n = 8 }, .{ .tiles_k = 8, .tiles_n = 4 } };
    var scratch = try Scratch.init(std.testing.allocator, w);
    defer scratch.deinit(std.testing.allocator);
    var prng = std.Random.DefaultPrng.init(0x3B1C);
    const r = prng.random();

    for (grids) |grid| {
        const dim_k = grid.tiles_k * 16;
        const dim_n = grid.tiles_n * 16;
        const wq = try std.testing.allocator.alloc(f16, dim_k * dim_n);
        defer std.testing.allocator.free(wq);
        for (wq) |*v| v.* = @floatCast(gauss(r));
        const trellis = try std.testing.allocator.alloc(i16, grid.tiles_k * grid.tiles_n * w.tileWords());
        defer std.testing.allocator.free(trellis);
        const results = try std.testing.allocator.alloc(Result, grid.tiles_k * grid.tiles_n);
        defer std.testing.allocator.free(results);

        try pack(&m, w, wq, grid.tiles_k, grid.tiles_n, trellis, &scratch, results, null);

        const got = try std.testing.allocator.alloc(f16, dim_k * dim_n);
        defer std.testing.allocator.free(got);
        exl3_format.unpack(w, .mul1, trellis, grid.tiles_k, grid.tiles_n, got);
        var ss: f64 = 0;
        for (got, wq) |gv, tv| {
            const d: f64 = @floatCast(@as(f32, @floatCast(gv)) - @as(f32, @floatCast(tv)));
            ss += d * d;
        }
        const rms = @sqrt(ss / @as(f64, @floatFromInt(dim_k * dim_n)));
        try expect(rms < 0.2);

        for (0..grid.tiles_k * grid.tiles_n) |ti| {
            try expect(results[ti].status != .failed);
            if (results[ti].flags & 1 == 0) try expect(results[ti].open_cost <= results[ti].cost); // amendment: uncertified under a pass-1 flag
            var dec: [256]u32 = undefined;
            exl3_format.tileStates(w, trellis[ti * w.tileWords() ..][0..w.tileWords()], &dec);
            for (0..256) |p| try expectEqual(@as(u32, results[ti].states[p]), dec[p]);
        }
    }
}

// ---------------------------------------------------------------------------
// T10: the ruling's machine proof. An independent u64 Viterbi (same tie rules,
// no rebase, no saturation) must equal the rebased u32 encoder in every field
// where the flags are zero, and the rebased result must never be cheaper.

const RefResult = struct { pin: u16, open: u64, cost: u64, closed: bool, states: [256]u16 };

/// One u64 sweep mirroring viterbiSteps' group structure without rebasing or
/// saturating: unreachable states carry cinf and are never added to.
fn refSteps(bufs: [2][]u64, m: *const Metric, b: *const [256]u8, qt: *const [256]i16, bp: ?[]u8, gmax: usize) void {
    const cinf: u64 = std.math.maxInt(u64);
    var cur: usize = 0;
    for (0..256) |p| {
        const bb: u5 = @intCast(b[p]);
        const n_fresh: u32 = @as(u32, 1) << bb;
        const hshift: u5 = 16 - bb;
        const gcount: usize = @as(usize, 1) << @intCast(@as(u8, 16) - @as(u8, bb));
        const read = bufs[cur];
        const write = bufs[cur ^ 1];
        const qtp: i32 = qt[p];
        var g: usize = 0;
        while (g < gcount) : (g += 1) {
            var best_h: u32 = 0;
            var best: u64 = read[g];
            var h: u32 = 1;
            while (h < n_fresh) : (h += 1) {
                const c = read[g | (h << hshift)];
                if (c < best) {
                    best = c;
                    best_h = h;
                }
            }
            if (bp) |bpbuf| bpbuf[p * gmax + g] = @intCast(best_h);
            const group_base: usize = g << @intCast(bb);
            var n: u32 = 0;
            while (n < n_fresh) : (n += 1) {
                const t: usize = group_base | @as(usize, @intCast(n));
                const e: i64 = @as(i32, m.q[t]) - @as(i32, qtp);
                write[t] = if (best == cinf) cinf else best + @as(u64, @intCast(e * e));
            }
        }
        cur ^= 1;
    }
}

fn refEncode(m: *const Metric, b: *const [256]u8, qt: *const [256]i16, gmax: usize, gpa: std.mem.Allocator) !RefResult {
    const cinf: u64 = std.math.maxInt(u64);
    const c0 = try gpa.alloc(u64, 65536);
    defer gpa.free(c0);
    const c1 = try gpa.alloc(u64, 65536);
    defer gpa.free(c1);
    const bpr = try gpa.alloc(u8, 256 * gmax);
    defer gpa.free(bpr);
    const bufs = [2][]u64{ c0, c1 };
    @memset(bufs[0], 0);
    refSteps(bufs, m, b, qt, null, gmax);
    var pin: u16 = 0;
    var best: u64 = bufs[0][0];
    for (1..65536) |s| {
        if (bufs[0][s] < best) {
            best = bufs[0][s];
            pin = @intCast(s);
        }
    }
    const open = best;
    for (bufs[0], 0..) |*v, s| v.* = if (s == pin) 0 else cinf;
    refSteps(bufs, m, b, qt, bpr, gmax);
    const cost = bufs[0][pin];
    var states: [256]u16 = undefined;
    var e: usize = pin;
    var p: usize = 256;
    while (p > 0) {
        p -= 1;
        states[p] = @intCast(e);
        const bb: u5 = @intCast(b[p]);
        const g: usize = e >> bb;
        const h: u32 = bpr[p * gmax + g];
        e = g | (@as(usize, h) << (16 - bb));
    }
    return .{ .pin = pin, .open = open, .cost = cost, .closed = cost != cinf and e == pin, .states = states };
}

test "T10 the rebased u32 encoder equals an independent u64 Viterbi in every field where flags are zero and is never cheaper" {
    const m = Metric.init();
    var placement: [256]exl3_format.Placement = undefined;
    exl3_format.tilePlacement(&placement);
    for ([_]u16{ 2, 3, 4, 6, 8, 16 }) |hh| {
        const w = Width{ .halves = hh };
        var scratch = try Scratch.init(std.testing.allocator, w);
        defer scratch.deinit(std.testing.allocator);
        var b: [256]u8 = undefined;
        freshBits(w, &b);
        var minb: u8 = 8;
        for (b) |v| minb = @min(minb, v);
        const gmax: usize = @as(usize, 1) << @intCast(@as(u8, 16) - minb);
        var prng = std.Random.DefaultPrng.init(0x1010 + hh);
        const r = prng.random();
        for (0..5) |shape| { // gaussian, all +4, all -4, alternating, Hadamard signs
            var twq: [256]f16 = undefined;
            for (&twq, 0..) |*v, at| {
                if (shape == 0) {
                    v.* = @floatCast(gauss(r));
                    continue;
                }
                const sign: f32 = switch (shape) {
                    1, 2 => 1.0,
                    3 => if (at % 2 == 0) @as(f32, 1.0) else -1.0,
                    else => if ((@popCount(@as(u8, @truncate(at / 16)) & @as(u8, @truncate(at))) & 1) == 0) @as(f32, 1.0) else -1.0,
                };
                v.* = @floatCast(if (shape == 2) @as(f32, -4.0) else 4.0 * sign);
            }
            var qt: [256]i16 = undefined;
            for (0..256) |p| qt[p] = quantTarget(twq[placement[p].row * 16 + placement[p].col]);
            const res = encodeTile(&m, w, &twq, &scratch);
            try expect(res.status != .failed);
            const ref = try refEncode(&m, &b, &qt, gmax, std.testing.allocator);
            try expect(ref.closed); // reachability within 16 steps: an open reference is a bug
            try expectEqual(res.pinned, ref.pin); // pass-1 leader argument: the argmin path never saturates
            try expectEqual(res.open_cost, ref.open);
            if (res.flags == 0) {
                try expectEqual(res.cost, ref.cost);
                try std.testing.expectEqualSlices(u16, &ref.states, &res.states);
                try expectEqual(if (res.cost == res.open_cost) Status.optimal else Status.closed, res.status);
            } else {
                try expect(res.cost >= ref.cost); // never cheaper than the true optimum
                var sum: u64 = 0;
                for (0..256) |p| {
                    const e: i64 = @as(i32, m.q[res.states[p]]) - @as(i32, qt[p]);
                    sum += @as(u64, @intCast(e * e));
                }
                try expectEqual(sum, res.cost); // still the exact sum of the returned path
            }
        }
    }
}
