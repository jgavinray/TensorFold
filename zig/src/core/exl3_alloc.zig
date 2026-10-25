//! Host-only EXL3 bit allocator: choose per-part widths for a fresh quantization run.
//!
//! Given the quantizable tensors of a model (name, K, N, class, and the layer/role
//! position the tie-break needs), a target average bpw, the head's fixed width, and
//! optional class floors/promotions (hq-style), the allocator assigns each trellis
//! tensor a format `Width` (halves 2..16; `bits = halves / 2`, so odd halves are the
//! half-steps 2.5, 3.5, ... the format oracle admits at mul1 and build step 5 lets the
//! device decode) so the weighted average lands on the target as closely as the width
//! lattice allows.
//!
//! The interface is the contract of docs/eaq-review/eaq-halfstep-spec-fable.md section 3
//! (binding): the unit is the trellis tensor; `target_bpw` is trellis bits per *body*
//! weight, K*N-weighted, with the head fixed at `head_bits` outside the budget and
//! reported separately; the emitted artifact is a `Plan` (`PartWidth{base, halves}` per
//! trellis tensor plus `plan_sha`), which the driver consumes via `--plan <file>`.
//! The budget rule follows the ruling in docs/eaq-review/alloc-rule-fable.md (fable,
//! 2026-10-24): floors first, then a deterministic greedy water-fill; scale overhead
//! `(16K + 16N) / (K N)` is excluded from every average here, as the spec defines
//! `target_bpw` over trellis bits (eaq-halfstep-spec-fable.md section 3.1).
//!
//! Conventions, all deterministic and pure:
//!   * floors assign every quantized tensor its class floor first (floors are
//!     non-negotiable, exllamav3 --hq style);
//!   * the remaining budget then goes to the fitting tensor with the largest marginal
//!     error proxy `r = S * 2^-bits` (below), ties broken by the chain the ruling
//!     demands: smaller `S * K*N`, then fewer half-steps already taken in its
//!     (class, layer) group, then role priority (mlp down < gate < up; fused-slot
//!     v < k < q; gdn qkv < z < out), then the bit-reversed 6-bit layer index, then
//!     the tensor's position in the input order (section 2 of the ruling);
//!   * tensors the run must keep verbatim (norms, embedding, the width-less GDN a/b
//!     gates, conv1d, A_log, dt_bias, and anything the caller marks) are `dense`:
//!     assignment halves 0, no budget, no average, per ruling 1;
//!   * a quantizable tensor must tile 128 in K and N (ruling 4.2; the decode kernels'
//!     requirement at eaq-halfstep-spec-fable.md section 2.1, verified against the real
//!     27B and 35B inventories, whose quantized members are all 128 multiples).
//!
//! The error proxy (ruling 3): the per-tensor squared quantization error of a K x N
//! weight at `b` bits, summed over the block-scaled codebook, decays as `2^(-2b)`; its
//! marginal value per extra bit, for the greedy step that adds a half bit, is
//! proportional to `S * 2^(-b)` where `S` is the Hessian-weighted input-energy scale
//! from the calibration pass (eaq-ldlq-spec-fable.md's `sense_i`) with the format's
//! per-block max-abs normalization folded in. Documented assumption: the block scale
//! removes the weight-magnitude dependence, so every width in the domain sees the same
//! geometric decay and `r` orders candidates by (sensitivity, then width) alone. The
//! proxy is the ruling's; fable rules on it. `S` is documented as a log2 half-step
//! offset: sensitivities must lie in [1/16, 16] (|log2 S| <= 4), outside which
//! `allocate` refuses with `BadSensitivity`.
//!
//! Odd halves are emitted only when `allow_odd` (the fleet default, since step 5
//! admits them; false pins the run to the whole-bit lattice the current loader
//! accepts). `report.odd` lists every odd-halves tensor, and a width decimal is always
//! printed from the integer halves (`bitsDecimal`), never through a float formatter,
//! per section 3.3 of the spec.

const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

/// Role classes, matching the quantizer's part table (quant.zig's layer slot plan).
pub const Class = enum(u3) {
    /// self_attn q/k/v/o projections (plus the q/k norms, which stay dense).
    attn,
    /// linear_attn in_proj_qkv / in_proj_z / out_proj (a/b, conv, A_log, dt_bias stay dense).
    gdn,
    mlp,
    norm,
    embed,
    head,
    /// vision merger / block linears: kept, never quantized here.
    other,
};

pub const classNames = [_][]const u8{
    "attn", "gdn", "mlp", "norm", "embed", "head", "other",
};

pub fn isQuantizable(c: Class) bool {
    return switch (c) {
        .attn, .gdn, .mlp, .head => true,
        .norm, .embed, .other => false,
    };
}

/// The member's position inside its layer, for the role-priority tie-break the ruling
/// demands (section 2 of docs/eaq-review/alloc-rule-fable.md): within mlp, down before
/// gate before up; within the fused attention slot, v before k before q; within the
/// fused gdn input slot, qkv before z before out. The rank is only ever compared
/// inside one (class, layer) group, so the numbering only fixes the order inside each
/// family.
pub const Role = enum(u4) {
    down = 0,
    gate = 1,
    up = 2,
    v = 3,
    k = 4,
    q = 5,
    o = 6,
    qkv = 7,
    z = 8,
    out = 9,
    other = 10,

    pub fn rank(self: Role) u4 {
        return @intFromEnum(self);
    }
};

/// One candidate tensor. `k` and `n` are the stored GEMM shape; the allocator reads
/// them as the weight K*N and as the 128-multiple tile check (ruling 4.2). `layer` and
/// `role` feed the deterministic tie-break; a null layer falls back to input order
/// after spreading each tensor's own half-steps.
pub const Tensor = struct {
    name: []const u8,
    k: u32,
    n: u32,
    class: Class,
    /// True keeps the tensor verbatim at 16 bits (assignment halves 0): the dense copy path.
    dense: bool = false,
    layer: ?u32 = null,
    role: Role = .other,

    pub fn weight(self: Tensor) u64 {
        return @as(u64, self.k) * self.n;
    }
};

/// The format oracle's width: bits = halves / 2, halves 2..16, odd halves = mul1 half steps.
pub fn bitsOf(halves: u8) f64 {
    return @as(f64, @floatFromInt(halves)) / 2.0;
}

/// The bits decimal of a width, printed from the integer halves (3 -> "1.5",
/// 4 -> "2.0", 5 -> "2.5") and never through a float formatter, per section 3.3 of
/// eaq-halfstep-spec-fable.md. `buf` needs 4 bytes.
pub fn bitsDecimal(halves: u8, buf: *[4]u8) []const u8 {
    const whole = halves / 2;
    if (halves % 2 == 0) {
        return std.fmt.bufPrint(buf, "{d}.0", .{whole}) catch unreachable;
    }
    return std.fmt.bufPrint(buf, "{d}.5", .{whole}) catch unreachable;
}

/// The encoder BP sizing quantity of a width (spec 3.2): fresh bits `b(p)` at even
/// halves, and `ka = (halves - 1) / 2` at odd halves, which is integer `halves / 2`
/// either way.
pub fn freshBits(halves: u8) u8 {
    return halves / 2;
}

/// `tr` bytes of one trellis part (spec 3.2): `tiles * 8 * halves * 2`.
pub fn trBytes(tiles: u64, halves: u8) u64 {
    return tiles * 8 * @as(u64, halves) * 2;
}

/// A per-class minimum width (hq-style promotion floor), in bits; half-bit values are
/// legal and require `allow_odd`.
pub const Floor = struct {
    class: Class,
    min_bits: f64,
};

/// An hq-style promotion: raises the class floor above the run minimum by `boost_bits`
/// (what exllamav3's --hq does for its selected layers, per class here).
pub const Promotion = struct {
    class: Class,
    boost_bits: f64,
};

pub const Options = struct {
    /// Target trellis bits per BODY weight, K*N-weighted: the head rides at
    /// `head_bits` outside this average (spec 3.1; the ruling), dense members ride
    /// nothing (ruling 1). Scale overhead excluded, as the spec defines the field.
    target_bits: f64,
    /// The head's fixed width (exllamav3 --head_bits; the fleet runs 4). Not budgeted;
    /// reported as `Report.head_bits` and folded only into `trellis_bits_total`.
    head_bits: u4 = 4,
    /// Class floors; the first entry for a class wins. Half-bit values are legal.
    floors: []const Floor = &.{},
    /// Class promotions on top of the floors (the base run minimum is 1.0 bpw = 2 halves).
    promotions: []const Promotion = &.{},
    /// Odd halves are emitted (and flagged) unless false, which restricts every width
    /// to the whole-bit lattice — what the device loader admits before build step 5.
    allow_odd: bool = true,
    /// Per-tensor error sensitivity S (parallel to `tensors`); null is all-ones.
    /// Documented as a log2 half-step offset: S must lie in [1/16, 16] (ruling 4),
    /// outside which `allocate` refuses with `Error.BadSensitivity`.
    sensitivities: ?[]const f64 = null,
};

pub const Assignment = struct {
    /// 0 marks a dense (unquantized) tensor; otherwise the format halves, 2..16.
    halves: u8,
    /// The halves are odd: a mul1 half step (streamEnds b(p) alternation), refused by
    /// the device loader until build step 5 admits it.
    odd: bool,
};

pub const Bound = enum { ok, below_floor, above_maximum };

pub const ClassStat = struct {
    class: Class,
    count: usize = 0,
    /// Dense (norm, embed and caller-marked) members kept verbatim at 16 bits; they ride no
    /// budget and appear in no histogram below.
    dense_count: usize = 0,
    /// Quantized members at each whole-or-half width, index = halves - 2 (2..16).
    hist: [15]usize = @splat(0),
    /// Weighted average bits of this class's quantized members (0 when it has none).
    /// The head class holds the one fixed-width head, so its average is `head_bits`.
    avg_bits: f64 = 0,
};

pub const OddEntry = struct {
    tensor: usize, // index into the input list
    name: []const u8,
    halves: u8,
};

pub const Report = struct {
    requested_bits: f64,
    /// Trellis bits per BODY weight achieved (the quantity the target constrains):
    /// K*N-weighted over the quantized non-head members; equals `closest_bits` always.
    achieved_bits: f64,
    /// The closest achievable body average: the clamped bound when infeasible,
    /// otherwise the greedy's result (at or below the target; below it by less than one
    /// surviving step).
    closest_bits: f64,
    /// True when the requested body target is inside [floor average, 8].
    feasible: bool,
    /// True when achieved == requested as exactly as the lattice allows (the budget
    /// residue is smaller than the cheapest half-step the greedy could still take).
    exact: bool,
    /// Feasible but the lattice left the target unspent: `achieved_bits` strictly under
    /// `requested_bits`. (Ruling 12: the report names what bound the width.)
    budget_spill: bool = false,
    /// Which feasibility bound binds when `feasible` is false; `.ok` otherwise.
    bound: Bound,
    /// The weighted floor average of the body, and the whole-8-bit maximum of the body.
    /// The head rides neither: it is fixed outside the budget.
    min_bits: f64,
    max_bits: f64 = 8,
    /// Body member count (quantized, head excluded) and the K*N weights of the body and
    /// of the trellis ledger (body + head). Dense members are counted per class but
    /// ride no budget and no average.
    quantized: usize,
    body_weight: u64 = 0,
    total_weight: u64 = 0,
    /// The head's configured width, reported apart from the body average (spec 3.1).
    head_bits: f64 = 0,
    /// The byte-ledger figure: trellis bits per weight over body AND head together,
    /// for the manifest's cross-check only.
    trellis_bits_total: f64 = 0,
    /// What the averages above mean, stated in the report so a reader cannot misjoin
    /// it against a manifest field that counts scale overhead (spec 3.1/3.3).
    convention: []const u8 = "trellis bits, K*N-weighted over body tensors; head fixed and separate; scale overhead excluded",
    per_class: [num_classes]ClassStat,
    /// Every emitted odd-halves tensor: the widths the step-5 loader re-gate must admit.
    odd: []const OddEntry,
};

pub const Error = error{
    NoQuantizableTensors,
    BadTarget,
    HeadBitsOutOfRange,
    FloorOutOfRange,
    ZeroShape,
    BadTileShape,
    SensitivitiesLengthMismatch,
    BadSensitivity,
    BadPlan,
    OutOfMemory,
};

pub const num_classes = classNames.len;

/// Half bits of a requested bit count (4 = 2.0 bpw). Half-bit values require odd halves
/// to be admitted; with allow_odd false the floors clamp up to the next whole bit.
fn halvesOf(bits: f64, allow_odd: bool) !u8 {
    if (!(bits >= 1.0 and bits <= 8.0)) return Error.FloorOutOfRange;
    const x = @as(f64, @round(bits * 2.0));
    if (@abs(bits * 2.0 - x) > 1e-9) return Error.FloorOutOfRange; // widths are half-bit lattice points
    const h: u8 = @intFromFloat(x);
    if (!allow_odd and h % 2 == 1) return h + 1;
    return @min(h, 16);
}

// ---------------------------------------------------------------- budget rule

/// 2^(-h/2) for h in 0..16: the marginal-error decay of the proxy (header), built by
/// a pure multiply chain so the greedy's comparisons are exact and reproducible.
const pow2neghalf: [17]f64 = blk: {
    var t: [17]f64 = undefined;
    t[0] = 1.0;
    const s = 0.707106781186547524400844362104849039284835937688474036588339868; // 2^(-1/2)
    var i: usize = 1;
    while (i <= 16) : (i += 1) t[i] = t[i - 1] * s;
    break :blk t;
};

/// Tie-break group: the (class, layer) cohort whose members' half-steps must stay
/// balanced (ruling 9). A null layer makes each tensor its own group, which preserves
/// the round-robin the input-order tie alone would give equal twins.
const GroupKey = struct {
    class: Class,
    group: u64,

    fn of(t: Tensor, tix: usize) GroupKey {
        return .{
            .class = t.class,
            .group = if (t.layer) |l| @as(u64, l) else (@as(u64, 1) << 32) | @as(u64, @intCast(tix)),
        };
    }
};

const Ctx = struct {
    tensors: []const Tensor,
    opts: Options,
    /// Indices into `tensors` of the quantized BODY members (head excluded: it rides
    /// its fixed width outside the budget), in input order.
    body: []const usize,
    /// Parallel to `tensors`; 0 stays dense.
    halves: []u8,
    sens: ?[]const f64,
    /// Half bits per step: 1 with odd halves admitted, 2 pinned to the whole-bit lattice.
    step: u8,
    /// Parallel to `tensors`: the tie-break group of each tensor, and the half-steps
    /// taken so far per group (ruling 9's layer-balance key).
    groups: []const u32,
    group_taken: []u64,

    fn sensAt(self: Ctx, tix: usize) f64 {
        if (self.sens) |sv| return sv[tix];
        return 1.0;
    }

    fn w(self: Ctx, tix: usize) f64 {
        return @floatFromInt(self.tensors[tix].weight());
    }
};

/// One greedy step: the fitting (spend + cost <= budget) body tensor with the largest
/// marginal-error ratio r = S * 2^-bits, ties by the ruling's chain (section 2 of
/// alloc-rule-fable.md): smaller size-weighted sensitivity S*w, then fewer half-steps
/// already taken in its (class, layer) group, then role rank, then the bit-reversed
/// 6-bit layer index, then body-list position. Takes the step; false when nothing fits.
fn raiseFitting(ctx: *Ctx, budget: f64, spent: *f64) bool {
    var best: ?usize = null;
    var best_r: f64 = 0;
    var best_sw: f64 = 0;
    var best_bal: u64 = 0;
    var best_role: u4 = 0;
    var best_brev: u6 = 0;
    var best_cost: f64 = 0;
    for (ctx.body, 0..) |tix, qi| {
        const h = ctx.halves[tix];
        if (h + ctx.step > 16) continue; // 8 bits is the format ceiling
        const cost = ctx.w(tix) * @as(f64, @floatFromInt(ctx.step)) / 2.0;
        if (spent.* + cost > budget) continue;
        const t = ctx.tensors[tix];
        const s = ctx.sensAt(tix);
        const r = s * pow2neghalf[h];
        const sw = s * ctx.w(tix);
        const bal = ctx.group_taken[ctx.groups[tix]];
        const role = t.role.rank();
        const brev: u6 = if (t.layer) |l| @bitReverse(@as(u6, @truncate(l))) else 0;
        const take = if (best) |b| blk: {
            if (r != best_r) break :blk r > best_r;
            if (sw != best_sw) break :blk sw < best_sw;
            if (bal != best_bal) break :blk bal < best_bal;
            if (role != best_role) break :blk role < best_role;
            if (brev != best_brev) break :blk brev < best_brev;
            break :blk qi < b;
        } else true;
        if (take) {
            best = qi;
            best_r = r;
            best_sw = sw;
            best_bal = bal;
            best_role = role;
            best_brev = brev;
            best_cost = cost;
        }
    }
    const qi = best orelse return false;
    const tix = ctx.body[qi];
    ctx.halves[tix] += ctx.step;
    ctx.group_taken[ctx.groups[tix]] += ctx.step;
    spent.* += best_cost;
    return true;
}

/// The cheapest half-step the greedy could still take; the budget residue under this
/// means the target was hit as exactly as the lattice allows.
fn cheapestStep(ctx: *const Ctx) f64 {
    var m: f64 = std.math.inf(f64);
    for (ctx.body) |tix| {
        if (ctx.halves[tix] + ctx.step > 16) continue;
        const c = ctx.w(tix) * @as(f64, @floatFromInt(ctx.step)) / 2.0;
        if (c < m) m = c;
    }
    return if (m == std.math.inf(f64)) 0.0 else m;
}

fn floorFor(opts: Options, cls: Class) !u8 {
    var base: f64 = 1.0; // the run minimum, 1.0 bpw
    for (opts.floors) |f| {
        if (f.class == cls) {
            base = f.min_bits;
            break; // the first entry for a class wins
        }
    }
    for (opts.promotions) |p| {
        if (p.class == cls) base += p.boost_bits;
    }
    return try halvesOf(@min(base, 8.0), opts.allow_odd);
}

pub const Allocation = struct {
    /// Parallel to the input list.
    assignments: []const Assignment,
    report: Report,
    groups: []u32,
    group_taken: []u64,

    pub fn deinit(self: *Allocation, a: std.mem.Allocator) void {
        a.free(@constCast(self.assignments));
        for (self.report.odd) |o| a.free(@constCast(o.name));
        if (self.report.odd.len > 0) a.free(@constCast(self.report.odd));
        a.free(self.groups);
        a.free(self.group_taken);
    }
};

/// Assign per-part halves over `tensors` at the target, honoring floors, the fixed
/// head, and the whole-bit/half-step lattice. Deterministic: the same input list (in
/// the same order) always produces the same assignment, widths, and odd list.
pub fn allocate(a: std.mem.Allocator, tensors: []const Tensor, opts: Options) Error!Allocation {
    if (!(opts.target_bits >= 1.0 and opts.target_bits <= 8.0)) return Error.BadTarget;
    if (!(opts.head_bits >= 1 and opts.head_bits <= 8)) return Error.HeadBitsOutOfRange;
    if (opts.sensitivities) |sv| {
        if (sv.len != tensors.len) return Error.SensitivitiesLengthMismatch;
        for (sv) |s| {
            // Ruling 4: S is a log2 half-step offset capped at 4.
            if (!(s >= 1.0 / 16.0 and s <= 16.0)) return Error.BadSensitivity;
        }
    }

    // validation + membership
    var qlist: std.ArrayList(usize) = .empty; // quantized, head included (ledger, plan)
    errdefer qlist.deinit(a);
    var blist: std.ArrayList(usize) = .empty; // quantized body (budget, averages)
    errdefer blist.deinit(a);
    for (tensors, 0..) |t, i| {
        if (t.weight() == 0) return Error.ZeroShape;
        if (isQuantizable(t.class) and !t.dense) {
            // ruling 4.2: trellis tensors tile 128 in K and N; thin members (the GDN
            // a/b gates at N = 48) must arrive dense.
            if (t.k % 128 != 0 or t.n % 128 != 0) return Error.BadTileShape;
            try qlist.append(a, i);
            if (t.class != .head) try blist.append(a, i);
        }
    }
    if (blist.items.len == 0) return Error.NoQuantizableTensors;

    const halves = try a.alloc(u8, tensors.len);
    errdefer a.free(halves);
    @memset(halves, 0); // dense stays 0

    // floors, then the head's fixed width (outside the budget, spec 3.1)
    const head_halves: u8 = 2 * @as(u8, opts.head_bits);
    for (qlist.items) |tix| {
        const t = tensors[tix];
        halves[tix] = if (t.class == .head) head_halves else try floorFor(opts, t.class);
    }

    // body ledger: floors and the 8-bit ceiling bound the feasible interval
    var body_weight: u64 = 0;
    var head_weight: u64 = 0;
    var floor_weighted: f64 = 0;
    for (blist.items) |tix| {
        const w: f64 = @floatFromInt(tensors[tix].weight());
        body_weight += tensors[tix].weight();
        floor_weighted += w * @as(f64, @floatFromInt(halves[tix])) / 2.0;
    }
    for (qlist.items) |tix| {
        if (tensors[tix].class == .head) head_weight += tensors[tix].weight();
    }
    const body_f: f64 = @floatFromInt(body_weight);
    const min_bits = floor_weighted / body_f;
    const max_bits: f64 = 8.0;

    const feasible = opts.target_bits >= min_bits - 1e-9 and opts.target_bits <= max_bits + 1e-9;
    const bound: Bound = if (opts.target_bits < min_bits - 1e-9)
        .below_floor
    else if (opts.target_bits > max_bits + 1e-9) .above_maximum else .ok;
    const clamped = @min(@max(opts.target_bits, min_bits), max_bits);
    const budget = clamped * body_f;
    var spent = floor_weighted;

    // tie-break groups over the body, in input order
    const groups = try a.alloc(u32, tensors.len);
    errdefer a.free(groups);
    @memset(groups, 0);
    var gmap = std.AutoHashMap(GroupKey, u32).init(a);
    defer gmap.deinit();
    var gtaken: std.ArrayList(u64) = .empty;
    errdefer gtaken.deinit(a);
    for (blist.items) |tix| {
        const key = GroupKey.of(tensors[tix], tix);
        const gop = try gmap.getOrPut(key);
        if (!gop.found_existing) {
            gop.value_ptr.* = @intCast(gtaken.items.len);
            try gtaken.append(a, 0);
        }
        groups[tix] = gop.value_ptr.*;
    }
    const gtaken_items = try gtaken.toOwnedSlice(a); // exact-fit: the Allocation frees this
    errdefer a.free(gtaken_items); // (gtaken's own errdefer deinit is inert past toOwnedSlice)
    var ctx = Ctx{
        .tensors = tensors,
        .opts = opts,
        .body = blist.items,
        .halves = halves,
        .sens = opts.sensitivities,
        .step = if (opts.allow_odd) @as(u8, 1) else @as(u8, 2),
        .groups = groups,
        .group_taken = gtaken_items,
    };
    // the greedy always runs, at the clamped target: below the floor average no step
    // fits and the assignment sits at the floors; above 8 every tensor caps at 16.
    while (raiseFitting(&ctx, budget, &spent)) {}

    // ---- report
    var report = Report{
        .requested_bits = opts.target_bits,
        .achieved_bits = 0,
        .closest_bits = 0,
        .feasible = feasible,
        .exact = false,
        .bound = bound,
        .min_bits = min_bits,
        .max_bits = max_bits,
        .quantized = blist.items.len,
        .body_weight = body_weight,
        .total_weight = body_weight + head_weight,
        .head_bits = if (head_weight > 0) bitsOf(head_halves) else 0,
        .per_class = undefined,
        .odd = &.{},
    };
    for (&report.per_class, 0..) |*cs, ci| {
        cs.* = .{ .class = @enumFromInt(@as(u3, @intCast(ci))) };
    }
    const achieved = spent / body_f;
    report.achieved_bits = achieved;
    report.closest_bits = achieved;
    report.exact = feasible and (budget - spent) < cheapestStep(&ctx);
    report.budget_spill = feasible and (opts.target_bits - achieved) > 1e-9;

    var wbody_total: f64 = 0; // weighted bit-sum over the trellis ledger (body + head)
    for (qlist.items) |tix| {
        const t = tensors[tix];
        const cs = &report.per_class[@intFromEnum(t.class)];
        cs.count += 1;
        cs.hist[halves[tix] - 2] += 1;
        wbody_total += @as(f64, @floatFromInt(t.weight())) * bitsOf(halves[tix]);
    }
    for (tensors) |t| {
        if (t.dense) report.per_class[@intFromEnum(t.class)].dense_count += 1;
    }
    for (blist.items) |tix| {
        const t = tensors[tix];
        const cs = &report.per_class[@intFromEnum(t.class)];
        cs.avg_bits += ctx.w(tix) * bitsOf(halves[tix]);
    }
    for (&report.per_class) |*cs| {
        var cw: f64 = 0;
        for (blist.items) |tix| {
            if (tensors[tix].class == cs.class) cw += ctx.w(tix);
        }
        if (cw > 0) cs.avg_bits /= cw;
    }
    if (head_weight > 0) {
        const cs = &report.per_class[@intFromEnum(Class.head)];
        cs.avg_bits = bitsOf(head_halves);
        report.trellis_bits_total = wbody_total / @as(f64, @floatFromInt(body_weight + head_weight));
    } else {
        report.trellis_bits_total = achieved;
    }

    // odd list, in input order (the step-5 re-gate's admission checklist)
    var odd: std.ArrayList(OddEntry) = .empty;
    errdefer odd.deinit(a);
    for (blist.items) |tix| {
        if (halves[tix] % 2 == 1) {
            try odd.append(a, .{
                .tensor = tix,
                .name = try a.dupe(u8, tensors[tix].name),
                .halves = halves[tix],
            });
        }
    }
    report.odd = try odd.toOwnedSlice(a);

    const assignments = try a.alloc(Assignment, tensors.len);
    errdefer a.free(assignments);
    for (0..tensors.len) |i| {
        assignments[i] = .{ .halves = halves[i], .odd = halves[i] % 2 == 1 };
    }

    a.free(halves);
    qlist.deinit(a);
    blist.deinit(a);
    return .{
        .assignments = assignments,
        .report = report,
        .groups = groups,
        .group_taken = gtaken_items,
    };
}

// ---------------------------------------------------------------- the plan (spec 3.2, binding)

pub const method_name = "eaq-alloc/v1";

/// One trellis tensor's width, as the allocator decides it and the driver consumes it
/// (eaq-halfstep-spec-fable.md 3.2). `base` borrows the tensor name it applies to.
pub const PartWidth = struct {
    base: []const u8, // the PartPlan.base this applies to (quant-driver.zig's lookup)
    halves: u8, // 2..16, odd allowed
};

pub const Plan = struct {
    target_bpw: f64, // trellis bits per body weight, K*N-weighted (spec 3.1)
    head_halves: u8, // the lm_head width (even or odd)
    method: []const u8, // allocator name/version, e.g. "eaq-alloc/v1"
    parts: []const PartWidth, // every trellis member, head included (driver looks up by base)
    plan_sha: [32]u8, // sha256 of the canonical serialization below

    /// Encoder scratch is sized per site by the SMALLEST fresh-bit count of the plan,
    /// never its mean (spec 3.2): `BP = count * 256 * 2^(16 - min_b)`, `min_b` here.
    pub fn minFreshBits(self: Plan) u8 {
        if (self.parts.len == 0) return 0;
        var m: u8 = 8;
        for (self.parts) |p| m = @min(m, freshBits(p.halves));
        return m;
    }

    /// The plan's average trellis bits over the parts (a sanity read, K*N-free).
    pub fn meanHalves(self: Plan) f64 {
        if (self.parts.len == 0) return 0;
        var s: u64 = 0;
        for (self.parts) |p| s += p.halves;
        return @as(f64, @floatFromInt(s)) / @as(f64, @floatFromInt(self.parts.len));
    }
};

/// Canonical serialization of the plan fields — the `plan_sha` input and the resume
/// identity the driver stores (spec 3.2): fixed little-endian layout, integer widths,
/// and the target as its exact IEEE-754 bit pattern, so the bytes are reproducible
/// across runs and platforms and no float formatter is ever on the hash path.
pub fn planCanonicalBytes(a: std.mem.Allocator, p: Plan) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "eaq-plan-v1\n");
    try out.appendSlice(a, &.{ @truncate(p.method.len), @truncate(p.method.len >> 8) });
    try out.appendSlice(a, p.method);
    const tb: u64 = @bitCast(p.target_bpw);
    try out.appendSlice(a, &.{
        @truncate(tb),         @truncate(tb >> 8),  @truncate(tb >> 16),  @truncate(tb >> 24),
        @truncate(tb >> 32),   @truncate(tb >> 40), @truncate(tb >> 48),  @truncate(tb >> 56),
    });
    try out.append(a, p.head_halves);
    const n: u32 = @intCast(p.parts.len);
    try out.appendSlice(a, &.{ @truncate(n), @truncate(n >> 8), @truncate(n >> 16), @truncate(n >> 24) });
    for (p.parts) |part| {
        const l: u16 = @intCast(part.base.len);
        try out.appendSlice(a, &.{ @truncate(l), @truncate(l >> 8) });
        try out.appendSlice(a, part.base);
        try out.append(a, part.halves);
    }
    return out.toOwnedSlice(a);
}

pub fn planShaWith(a: std.mem.Allocator, p: Plan) ![32]u8 {
    const bytes = try planCanonicalBytes(a, p);
    defer a.free(bytes);
    var sha: [32]u8 = undefined;
    Sha256.hash(bytes, &sha, .{});
    return sha;
}

/// Build the driver-facing plan from an allocation: one part per trellis member
/// (head included, dense excluded — the copy path rides no plan entry), the requested
/// body target, the fixed head width, and the sha over the canonical bytes. `base`
/// strings are duped so the plan outlives nothing but the allocator.
pub fn makePlan(a: std.mem.Allocator, tensors: []const Tensor, al: Allocation, opts: Options) (Error || error{BadWidth})!Plan {
    var parts: std.ArrayList(PartWidth) = .empty;
    errdefer {
        for (parts.items) |p| a.free(@constCast(p.base));
        parts.deinit(a);
    }
    for (tensors, 0..) |t, i| {
        if (!isQuantizable(t.class) or t.dense) continue;
        const h = al.assignments[i].halves;
        if (h < 2 or h > 16) return error.BadWidth;
        try parts.append(a, .{ .base = try a.dupe(u8, t.name), .halves = h });
    }
    var p = Plan{
        .target_bpw = al.report.requested_bits,
        .head_halves = 2 * @as(u8, opts.head_bits),
        .method = method_name,
        .parts = try parts.toOwnedSlice(a),
        .plan_sha = undefined,
    };
    errdefer freePlan(a, &p);
    p.plan_sha = try planShaWith(a, p);
    return p;
}

pub fn freePlan(a: std.mem.Allocator, p: *Plan) void {
    for (p.parts) |part| a.free(@constCast(part.base));
    a.free(p.parts);
    p.parts = &.{};
}

fn jsonEscape(a: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    for (s) |c| {
        switch (c) {
            '"' => try out.appendSlice(a, "\""),
            '\\' => try out.appendSlice(a, "\\\\"),
            '\n' => try out.appendSlice(a, "\\n"),
            '\r' => try out.appendSlice(a, "\\r"),
            '\t' => try out.appendSlice(a, "\\t"),
            else => {
                if (c < 0x20) {
                    try out.print(a, "\\u{x:0>4}", .{c});
                    continue;
                }
                try out.append(a, c);
            },
        }
    }
}

/// JSON rendering of a plan for the driver's `--plan <file>`. Widths appear as integer
/// `halves` and as `bits` decimals printed from the integers (never float-formatted),
/// per spec 3.3; the sha is lowercase hex.
pub fn planToJson(a: std.mem.Allocator, p: Plan) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "{\n  \"target_bpw\": ");
    try out.print(a, "{d}", .{p.target_bpw});
    var hb: [4]u8 = undefined;
    try out.appendSlice(a, ",\n  \"head_bits\": \"");
    try out.appendSlice(a, bitsDecimal(p.head_halves, &hb));
    try out.appendSlice(a, "\",\n  \"method\": \"");
    try jsonEscape(a, &out, p.method);
    try out.appendSlice(a, "\",\n  \"plan_sha\": \"");
    for (p.plan_sha) |byte| try out.print(a, "{x:0>2}", .{byte});
    try out.appendSlice(a, "\",\n  \"parts\": [\n");
    for (p.parts, 0..) |part, i| {
        if (i > 0) try out.appendSlice(a, ",\n");
        try out.appendSlice(a, "    {\"base\": \"");
        try jsonEscape(a, &out, part.base);
        var pb: [4]u8 = undefined;
        try out.print(a, "\", \"halves\": {d}, \"bits\": \"{s}\"}}", .{ part.halves, bitsDecimal(part.halves, &pb) });
    }
    try out.appendSlice(a, "\n  ]\n}\n");
    return out.toOwnedSlice(a);
}

/// Read back a `planToJson` rendering (the driver's `--plan <file>` intake and the
/// resume check): rebuild the Plan and verify every self-consistency rule the writer
/// promised — the `bits` decimals recompute from the integer halves, `head_bits` from
/// `head_halves`, the sha hex decodes — so a hand-edited file refuses as `BadPlan`.
/// The caller owns the returned strings; `planShaWith` on the result must equal the
/// embedded sha for the file to be canonical.
pub fn planFromJson(a: std.mem.Allocator, s: []const u8) Error!Plan {
    var parsed = std.json.parseFromSlice(std.json.Value, a, s, .{}) catch return error.BadPlan;
    defer parsed.deinit();
    const root_v = parsed.value;
    if (root_v != .object) return error.BadPlan;
    const obj = root_v.object;

    const target_bpw = switch (obj.get("target_bpw") orelse return error.BadPlan) {
        .float => |f| f,
        .integer => |i| @as(f64, @floatFromInt(i)),
        .number_string => |str| std.fmt.parseFloat(f64, str) catch return error.BadPlan,
        else => return error.BadPlan,
    };
    const method = switch (obj.get("method") orelse return error.BadPlan) {
        .string => |x| try a.dupe(u8, x),
        else => return error.BadPlan,
    };
    errdefer a.free(method);
    const head_bits_str = switch (obj.get("head_bits") orelse return error.BadPlan) {
        .string => |x| x,
        else => return error.BadPlan,
    };
    const sha_str = switch (obj.get("plan_sha") orelse return error.BadPlan) {
        .string => |x| x,
        else => return error.BadPlan,
    };
    if (sha_str.len != 64) return error.BadPlan;
    const parts_v = switch (obj.get("parts") orelse return error.BadPlan) {
        .array => |x| x,
        else => return error.BadPlan,
    };

    var p = Plan{
        .target_bpw = target_bpw,
        .head_halves = 0,
        .method = method,
        .parts = &.{},
        .plan_sha = undefined,
    };
    var hb: [4]u8 = undefined;
    // head_bits "4.0" / "3.5" -> halves, printed from the integer and checked
    {
        const dot = std.mem.indexOfScalar(u8, head_bits_str, '.') orelse return error.BadPlan;
        const whole = std.fmt.parseInt(u8, head_bits_str[0..dot], 10) catch return error.BadPlan;
        const frac = head_bits_str[dot + 1 ..];
        if (!std.mem.eql(u8, frac, "0") and !std.mem.eql(u8, frac, "5")) return error.BadPlan;
        p.head_halves = whole * 2 + @intFromBool(frac[0] == '5');
    }
    if (!std.mem.eql(u8, head_bits_str, bitsDecimal(p.head_halves, &hb))) return error.BadPlan;
    var sha: [32]u8 = std.mem.zeroes([32]u8);
    for (sha_str, 0..) |c, i| {
        const n = std.fmt.parseInt(u8, sha_str[i .. i + 1], 16) catch return error.BadPlan;
        _ = c;
        sha[i / 2] |= n << @as(u3, @intCast(if (i % 2 == 0) @as(u8, 4) else 0));
    }
    p.plan_sha = sha;

    var parts: std.ArrayList(PartWidth) = .empty;
    errdefer {
        for (parts.items) |part| a.free(@constCast(part.base));
        parts.deinit(a);
    }
    for (parts_v.items) |pv| {
        if (pv != .object) return error.BadPlan;
        const po = pv.object;
        const base = switch (po.get("base") orelse return error.BadPlan) {
            .string => |x| try a.dupe(u8, x),
            else => return error.BadPlan,
        };
        errdefer a.free(base);
        const halves = switch (po.get("halves") orelse return error.BadPlan) {
            .integer => |i| (if (i >= 2 and i <= 16) @as(u8, @intCast(i)) else return error.BadPlan),
            else => return error.BadPlan,
        };
        const bits = switch (po.get("bits") orelse return error.BadPlan) {
            .string => |x| x,
            else => return error.BadPlan,
        };
        var pb: [4]u8 = undefined;
        if (!std.mem.eql(u8, bits, bitsDecimal(halves, &pb))) return error.BadPlan;
        try parts.append(a, .{ .base = base, .halves = halves });
    }
    p.parts = try parts.toOwnedSlice(a);
    return p;
}

// ---------------------------------------------------------------- the real 27B inventory

/// The conv1d channel count: the qkv (10240) and z (6144) projections' GEMM-tile
/// padding the source stores the depthwise kernel over (4 taps).
const GDN_CONV: u32 = 16640;

/// The kept-851 language inventory of the bf16 source (quant.zig's KEPT: 1199 source
/// tensors minus the 333 model.visual.* and the 15 mtp.* drops, verified against the
/// source index by the quantizer's own kept-count check at run time). Class roles
/// follow quant.zig's part table: attn 96 (q/k/v/o plus q_norm/k_norm over the 16
/// layers where (i + 1) % 4 == 0), gdn 432 (in_proj qkv/z/a/b, out_proj, conv1d,
/// A_log, dt_bias, gated norm over the 48 remaining layers), mlp 192, norm 129 (input
/// and post per layer plus the final), embed 1 and head 1. Denseness is the
/// quantizer's own: a/b, conv1d, A_log, dt_bias and the q/k norms stay bf16, so the
/// quantized set is 64 + 144 + 192 + 1 = 401 parts (400 body + head).
pub fn qwen27Inventory(a: std.mem.Allocator) ![]Tensor {
    const D: u32 = 5120;
    const FF: u32 = 17408;
    const VOCAB: u32 = 248320;
    const layers: usize = 64;

    var out: std.ArrayList(Tensor) = .empty;
    errdefer a.free(out.items);
    try out.append(a, .{ .name = try a.dupe(u8, "model.language_model.embed_tokens.weight"), .k = VOCAB, .n = D, .class = .embed, .dense = true });
    try out.append(a, .{ .name = try a.dupe(u8, "lm_head"), .k = D, .n = VOCAB, .class = .head });
    for (0..layers) |i| {
        const attn = (i + 1) % 4 == 0; // qwen27.isAttention
        const pre = try std.fmt.allocPrint(a, "model.language_model.layers.{d}.", .{i});
        const mk = struct {
            fn add(ll: *std.ArrayList(Tensor), aa: std.mem.Allocator, p: []const u8, tail: []const u8, k: u32, n: u32, cls: Class, dense: bool, layer: ?u32, role: Role) !void {
                try ll.append(aa, .{
                    .name = try std.fmt.allocPrint(aa, "{s}{s}", .{ p, tail }),
                    .k = k,
                    .n = n,
                    .class = cls,
                    .dense = dense,
                    .layer = layer,
                    .role = role,
                });
            }
        }.add;
        const L: u32 = @intCast(i);
        try mk(&out, a, pre, "input_layernorm.weight", D, 1, .norm, true, L, .other);
        try mk(&out, a, pre, "post_attention_layernorm.weight", D, 1, .norm, true, L, .other);
        if (attn) {
            try mk(&out, a, pre, "self_attn.q_proj.weight", D, 12288, .attn, false, L, .q);
            try mk(&out, a, pre, "self_attn.k_proj.weight", D, 1024, .attn, false, L, .k);
            try mk(&out, a, pre, "self_attn.v_proj.weight", D, 1024, .attn, false, L, .v);
            try mk(&out, a, pre, "self_attn.o_proj.weight", 6144, D, .attn, false, L, .o);
            try mk(&out, a, pre, "self_attn.q_norm.weight", 256, 1, .attn, true, L, .other);
            try mk(&out, a, pre, "self_attn.k_norm.weight", 256, 1, .attn, true, L, .other);
        } else {
            try mk(&out, a, pre, "linear_attn.in_proj_qkv.weight", D, 10240, .gdn, false, L, .qkv);
            try mk(&out, a, pre, "linear_attn.in_proj_z.weight", D, 6144, .gdn, false, L, .z);
            try mk(&out, a, pre, "linear_attn.in_proj_a.weight", D, 48, .gdn, true, L, .other);
            try mk(&out, a, pre, "linear_attn.in_proj_b.weight", D, 48, .gdn, true, L, .other);
            try mk(&out, a, pre, "linear_attn.out_proj.weight", 6144, D, .gdn, false, L, .out);
            try mk(&out, a, pre, "linear_attn.conv1d.weight", 4, GDN_CONV, .gdn, true, L, .other);
            try mk(&out, a, pre, "linear_attn.A_log", 48, 1, .gdn, true, L, .other);
            try mk(&out, a, pre, "linear_attn.dt_bias", 48, 1, .gdn, true, L, .other);
            try mk(&out, a, pre, "linear_attn.norm.weight", 128, 1, .gdn, true, L, .other);
        }
        try mk(&out, a, pre, "mlp.gate_proj.weight", D, FF, .mlp, false, L, .gate);
        try mk(&out, a, pre, "mlp.up_proj.weight", D, FF, .mlp, false, L, .up);
        try mk(&out, a, pre, "mlp.down_proj.weight", FF, D, .mlp, false, L, .down);
        a.free(pre);
    }
    try out.append(a, .{ .name = try a.dupe(u8, "model.language_model.norm.weight"), .k = D, .n = 1, .class = .norm, .dense = true });

    return out.toOwnedSlice(a);
}

/// Releases a list built by `qwen27Inventory` (the names ride the same allocator).
pub fn freeInventory(a: std.mem.Allocator, ts: []Tensor) void {
    for (ts) |t| a.free(@constCast(t.name));
    a.free(ts);
}

// ---------------------------------------------------------------- tests

const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectApproxEqAbs = std.testing.expectApproxEqAbs;

fn tensor(name: []const u8, k: u32, n: u32, cls: Class) Tensor {
    return .{ .name = name, .k = k, .n = n, .class = cls };
}

/// The body average (head and dense excluded) of an assignment, as the spec defines it.
fn avgBody(tensors: []const Tensor, as: []const Assignment, want: f64) !void {
    var wsum: f64 = 0;
    var total: u64 = 0;
    for (tensors, as) |t, x| {
        if (!isQuantizable(t.class) or t.dense or t.class == .head) continue;
        wsum += @as(f64, @floatFromInt(t.weight())) * bitsOf(x.halves);
        total += t.weight();
    }
    try expectApproxEqAbs(want, wsum / @as(f64, @floatFromInt(total)), 1e-9);
}

test "exact target: equal tensors at whole bits, waterfill hits 2.0 with no odd halves" {
    const ts = [_]Tensor{
        tensor("q", 128, 128, .mlp),
        tensor("k", 128, 128, .mlp),
        tensor("v", 128, 128, .mlp),
        tensor("o", 128, 128, .mlp),
    };
    var al = try allocate(std.testing.allocator, &ts, .{ .target_bits = 2.0 });
    defer al.deinit(std.testing.allocator);
    for (al.assignments) |x| {
        try expectEqual(@as(u8, 4), x.halves); // exactly 2.0 bits
        try expect(!x.odd);
    }
    try expect(al.report.feasible);
    try expect(al.report.exact);
    try expectApproxEqAbs(2.0, al.report.achieved_bits, 1e-12);
    try avgBody(&ts, al.assignments, 2.0);
}

test "exact target with odd emission: one tensor at 2.5 is flagged odd" {
    const ts = [_]Tensor{tensor("w", 128, 128, .gdn)};
    var al = try allocate(std.testing.allocator, &ts, .{ .target_bits = 2.5 });
    defer al.deinit(std.testing.allocator);
    try expectEqual(@as(u8, 5), al.assignments[0].halves);
    try expect(al.assignments[0].odd);
    try expectEqual(@as(usize, 1), al.report.odd.len);
    try expectEqual(@as(usize, 0), al.report.odd[0].tensor);
    try expectEqual(@as(u8, 5), al.report.odd[0].halves);
    try expect(al.report.exact);
    var buf: [4]u8 = undefined;
    try expectEqualStrings("2.5", bitsDecimal(5, &buf));
    try expectEqualStrings("2.0", bitsDecimal(4, &buf));
    try expectEqual(@as(u8, 2), freshBits(5)); // ka at odd halves
    try expectEqual(@as(u64, 512 * 8 * 5 * 2), trBytes(512, 5));
}

test "odd halves off: no width leaves the whole-bit lattice" {
    const ts = [_]Tensor{ tensor("a", 128, 128, .gdn), tensor("b", 128, 128, .attn) };
    var al = try allocate(std.testing.allocator, &ts, .{ .target_bits = 2.7, .allow_odd = false });
    defer al.deinit(std.testing.allocator);
    for (al.assignments) |x| {
        try expect(x.halves % 2 == 0);
        try expect(!x.odd);
    }
    try expectEqual(@as(usize, 0), al.report.odd.len);
    try expect(al.report.feasible);
    try expect(al.report.achieved_bits <= 2.7 + 1e-12); // the budget never overshoots
    try expect(al.report.budget_spill); // the lattice left the 2.7 target unspent
}

test "floors are respected: the class floor and the hq promotion both bind" {
    const ts = [_]Tensor{
        tensor("l0.attn.q", 512, 512, .attn),
        tensor("l0.attn.k", 512, 512, .attn),
        tensor("l0.mlp.gate", 512, 512, .mlp),
        tensor("l0.mlp.down", 512, 512, .mlp),
    };
    const floors = [_]Floor{.{ .class = .attn, .min_bits = 4.0 }};
    var al = try allocate(std.testing.allocator, &ts, .{ .target_bits = 2.5, .floors = &floors });
    defer al.deinit(std.testing.allocator);
    try expectEqual(@as(u8, 8), al.assignments[0].halves); // attn pinned at the 4.0 floor
    try expectEqual(@as(u8, 8), al.assignments[1].halves);
    for (al.assignments[2..]) |x| try expect(x.halves <= 8);
    try expectApproxEqAbs(2.5, al.report.achieved_bits, 1e-9);

    // an hq promotion raises the class floor by the boost
    const promos = [_]Promotion{.{ .class = .mlp, .boost_bits = 1.5 }};
    var al2 = try allocate(std.testing.allocator, &ts, .{ .target_bits = 3.0, .promotions = &promos });
    defer al2.deinit(std.testing.allocator);
    for (al2.assignments[2..]) |x| try expect(x.halves >= 5); // floor 1.0 + boost 1.5 = 5 halves
    try expectApproxEqAbs(3.0, al2.report.achieved_bits, 1e-9);
}

test "the head is fixed at head_bits and rides outside the budget" {
    const ts = [_]Tensor{
        tensor("lm_head", 512, 512, .head),
        tensor("l.mlp.down", 512, 512, .mlp),
        tensor("l.gdn.qkv", 512, 512, .gdn),
    };
    // two equal-weight body tensors reach quarter-bit averages on the half-step lattice
    for ([_]f64{ 1.5, 2.75, 6.0 }) |t| {
        var al = try allocate(std.testing.allocator, &ts, .{ .target_bits = t, .head_bits = 4 });
        defer al.deinit(std.testing.allocator);
        try expectEqual(@as(u8, 8), al.assignments[0].halves); // never moved
        try expect(!al.assignments[0].odd);
        try expectEqual(@as(f64, 4.0), al.report.head_bits); // reported separately
        try expectApproxEqAbs(t, al.report.achieved_bits, 1e-9); // body-only target
    }
    // the byte ledger folds the head in: (w.3 + w.3 + w.4) / 3w at body 3.0
    var al = try allocate(std.testing.allocator, &ts, .{ .target_bits = 3.0, .head_bits = 4 });
    defer al.deinit(std.testing.allocator);
    try expectApproxEqAbs(10.0 / 3.0, al.report.trellis_bits_total, 1e-9);
    try expectEqual(@as(usize, 2), al.report.quantized); // head counted apart
    try expectEqual(@as(usize, 3), al.report.total_weight / (512 * 512));
}

const expectEqualStrings = std.testing.expectEqualStrings;
const expectEqualSlices = std.testing.expectEqualSlices;

test "determinism: three fresh runs agree exactly, widths, odd list, and plan sha" {
    const ts = [_]Tensor{
        tensor("a.attn.q", 512, 512, .attn),
        tensor("a.gdn.qkv", 256 * 128, 2 * 128, .gdn),
        tensor("a.mlp.gate", 512, 512, .mlp),
        tensor("h", 512, 512, .head),
    };
    var a_first = try allocate(std.testing.allocator, &ts, .{ .target_bits = 2.9, .head_bits = 4 });
    defer a_first.deinit(std.testing.allocator);
    const first = a_first.assignments;
    const plan0 = try makePlan(std.testing.allocator, &ts, a_first, .{ .target_bits = 2.9, .head_bits = 4 });
    var p0 = plan0;
    defer freePlan(std.testing.allocator, &p0);
    for (0..2) |_| {
        var b = try allocate(std.testing.allocator, &ts, .{ .target_bits = 2.9, .head_bits = 4 });
        defer b.deinit(std.testing.allocator);
        try std.testing.expectEqualSlices(Assignment, first, b.assignments);
        try expectEqual(a_first.report.odd.len, b.report.odd.len);
        for (a_first.report.odd, b.report.odd) |x, y| {
            try expectEqual(x.tensor, y.tensor);
            try expectEqual(x.halves, y.halves);
        }
        try expectEqual(a_first.report.achieved_bits, b.report.achieved_bits);
        var bp = try makePlan(std.testing.allocator, &ts, b, .{ .target_bits = 2.9, .head_bits = 4 });
        defer freePlan(std.testing.allocator, &bp);
        try expectEqualSlices(u8, &p0.plan_sha, &bp.plan_sha);
    }
}

test "infeasible targets report the closest achievable body bpw and why" {
    const ts = [_]Tensor{
        tensor("h", 512, 512, .head), // fixed at 4, outside the body interval
        tensor("a", 512, 512, .mlp),
        tensor("b", 512, 512, .gdn),
    };
    const floors = [_]Floor{ .{ .class = .mlp, .min_bits = 3.0 }, .{ .class = .gdn, .min_bits = 3.0 } };
    // body floor average is 3.0; the head (4.0) does not move the interval
    var low = try allocate(std.testing.allocator, &ts, .{ .target_bits = 1.5, .head_bits = 4, .floors = &floors });
    defer low.deinit(std.testing.allocator);
    try expect(!low.report.feasible);
    try expectEqual(Bound.below_floor, low.report.bound);
    try expectApproxEqAbs(3.0, low.report.closest_bits, 1e-12);
    try expectApproxEqAbs(3.0, low.report.min_bits, 1e-12);
    try expectEqual(@as(u8, 8), low.assignments[0].halves); // head stays fixed regardless
    try expectEqual(@as(u8, 6), low.assignments[1].halves); // body at the floors: the closest end
    try expectEqual(@as(u8, 6), low.assignments[2].halves);
    try expect(!low.report.budget_spill); // infeasible, not a lattice miss

    // the interval top is 8 body bits: the head never caps it below 8
    var hi = try allocate(std.testing.allocator, &ts, .{ .target_bits = 8.0, .head_bits = 4 });
    defer hi.deinit(std.testing.allocator);
    try expect(hi.report.feasible);
    try expectApproxEqAbs(8.0, hi.report.achieved_bits, 1e-9);
    for (hi.assignments[1..]) |x| try expectEqual(@as(u8, 16), x.halves);
    try expectEqual(@as(u8, 8), hi.assignments[0].halves); // the head still rides at 4.0
    try expectApproxEqAbs(20.0 / 3.0, hi.report.trellis_bits_total, 1e-9); // (8+8+4)/3

    // out-of-domain requests are refused outright
    try std.testing.expectError(Error.BadTarget, allocate(std.testing.allocator, &ts, .{ .target_bits = 9.5 }));
    try std.testing.expectError(Error.BadTarget, allocate(std.testing.allocator, &ts, .{ .target_bits = 0.5 }));
    try std.testing.expectError(Error.HeadBitsOutOfRange, allocate(std.testing.allocator, &ts, .{ .target_bits = 3.0, .head_bits = 9 }));
}

test "dense members take no budget and ride no average" {
    var ts = [_]Tensor{
        tensor("embed", 4096, 4096, .embed), // huge but dense: must not bend the average
        tensor("l.n1", 512, 512, .norm),
        tensor("a", 512, 512, .mlp),
        tensor("b", 512, 512, .gdn),
        tensor("h", 512, 512, .head),
    };
    ts[0].dense = true;
    ts[1].dense = true;
    var al = try allocate(std.testing.allocator, &ts, .{ .target_bits = 3.0, .head_bits = 4 });
    defer al.deinit(std.testing.allocator);
    try expectEqual(@as(u8, 0), al.assignments[0].halves); // dense: 0 halves, the verbatim copy
    try expectEqual(@as(u8, 0), al.assignments[1].halves);
    try expectEqual(@as(usize, 2), al.report.quantized); // body only
    try expectApproxEqAbs(3.0, al.report.achieved_bits, 1e-9);
    try expectEqual(@as(usize, 1), al.report.per_class[@intFromEnum(Class.norm)].dense_count);
    try expectEqual(@as(usize, 1), al.report.per_class[@intFromEnum(Class.embed)].dense_count);
    try avgBody(&ts, al.assignments, 3.0);
}

test "per-class distribution reports what each class got" {
    const ts = [_]Tensor{
        tensor("q0", 128, 128, .attn),
        tensor("q1", 128, 128, .attn),
        tensor("g0", 128, 128, .gdn),
        tensor("m0", 128, 128, .mlp),
        tensor("h", 128, 128, .head),
    };
    var al = try allocate(std.testing.allocator, &ts, .{ .target_bits = 2.5, .head_bits = 4 });
    defer al.deinit(std.testing.allocator);
    const attn = al.report.per_class[@intFromEnum(Class.attn)];
    try expectEqual(@as(usize, 2), attn.count);
    for (attn.hist, 0..) |c, h| { // every width bucket holds exactly the attn tensors at that width
        var want: usize = 0;
        for (al.assignments[0..2]) |x| {
            if (x.halves == h + 2) want += 1;
        }
        try expectEqual(want, c);
    }
    const head = al.report.per_class[@intFromEnum(Class.head)];
    try expectEqual(@as(usize, 1), head.hist[8 - 2]); // fixed 4.0
    try expectApproxEqAbs(4.0, head.avg_bits, 1e-12);
    try expectApproxEqAbs(2.5, al.report.achieved_bits, 1e-9);
}

test "sensitivities steer the waterfill: a hot tensor climbs before the cold ones" {
    const ts = [_]Tensor{
        tensor("cold0", 256, 256, .mlp),
        tensor("l.norm", 256, 128, .norm), // dense between the twins: S rides tensor position
        tensor("cold1", 256, 256, .mlp),
        tensor("hot", 256, 256, .mlp),
    };
    var ts_mut = ts;
    ts_mut[1].dense = true;
    const sens = [_]f64{ 1.0, 1.0, 1.0, 16.0 }; // hot at the ruling's S cap
    var al = try allocate(std.testing.allocator, &ts_mut, .{ .target_bits = 2.0, .sensitivities = &sens });
    defer al.deinit(std.testing.allocator);
    try expectApproxEqAbs(2.0, al.report.achieved_bits, 1e-9);
    try expect(al.assignments[3].halves > al.assignments[0].halves); // the hot tensor got the bits
    try expectEqual(al.assignments[0].halves, al.assignments[2].halves); // cold twins stay even
    try expectEqual(@as(u8, 0), al.assignments[1].halves);
    // the ruling's S bound: |log2 S| <= 4
    const hot_sens = [_]f64{ 100.0, 1.0, 1.0, 1.0 };
    try std.testing.expectError(Error.BadSensitivity, allocate(std.testing.allocator, &ts_mut, .{ .target_bits = 2.0, .sensitivities = &hot_sens }));
    const cold_sens = [_]f64{ 0.01, 1.0, 1.0, 1.0 };
    try std.testing.expectError(Error.BadSensitivity, allocate(std.testing.allocator, &ts_mut, .{ .target_bits = 2.0, .sensitivities = &cold_sens }));
    const short_sens = [_]f64{ 1.0 };
    try std.testing.expectError(Error.SensitivitiesLengthMismatch, allocate(std.testing.allocator, &ts_mut, .{ .target_bits = 2.0, .sensitivities = &short_sens }));
}

test "tie-break the ruling demands: role priority inside a layer, balance across layers" {
    // one mlp layer, equal weights: every key ties until the role rank — down first
    const one = [_]Tensor{
        .{ .name = "L.mlp.gate", .k = 128, .n = 128, .class = .mlp, .layer = 7, .role = .gate },
        .{ .name = "L.mlp.up", .k = 128, .n = 128, .class = .mlp, .layer = 7, .role = .up },
        .{ .name = "L.mlp.down", .k = 128, .n = 128, .class = .mlp, .layer = 7, .role = .down },
    };
    // budget for exactly one half-step: it must land on down
    const t1 = 1.0 + 0.5 / 3.0 + 1e-12;
    var al = try allocate(std.testing.allocator, &one, .{ .target_bits = t1 });
    defer al.deinit(std.testing.allocator);
    try expectEqual(@as(u8, 3), al.assignments[2].halves); // down took the step
    try expectEqual(@as(u8, 2), al.assignments[0].halves);
    try expectEqual(@as(u8, 2), al.assignments[1].halves);

    // two equal layers, budget for three steps: the layer-balance key interleaves them
    const two = [_]Tensor{
        .{ .name = "A.gate", .k = 128, .n = 128, .class = .mlp, .layer = 3, .role = .gate },
        .{ .name = "A.up", .k = 128, .n = 128, .class = .mlp, .layer = 3, .role = .up },
        .{ .name = "A.down", .k = 128, .n = 128, .class = .mlp, .layer = 3, .role = .down },
        .{ .name = "B.gate", .k = 128, .n = 128, .class = .mlp, .layer = 9, .role = .gate },
        .{ .name = "B.up", .k = 128, .n = 128, .class = .mlp, .layer = 9, .role = .up },
        .{ .name = "B.down", .k = 128, .n = 128, .class = .mlp, .layer = 9, .role = .down },
    };
    const t3 = 1.0 + 3 * 0.5 / 6.0 + 1e-12; // three half-steps of budget over six members
    var al2 = try allocate(std.testing.allocator, &two, .{ .target_bits = t3 });
    defer al2.deinit(std.testing.allocator);
    var sum_a: u8 = 0;
    var sum_b: u8 = 0;
    for (al2.assignments[0..3], 0..) |x, i| {
        sum_a += x.halves - @as(u8, 2);
        _ = i;
    }
    for (al2.assignments[3..]) |x| sum_b += x.halves - @as(u8, 2);
    // steps split across the layers, never all on one
    try expect(sum_a >= 1 and sum_a <= 2);
    try expect(sum_b >= 1 and sum_b <= 2);
}

test "validation rejects degenerate and untileable inputs" {
    const bad_zero = [_]Tensor{tensor("z", 0, 128, .mlp)};
    try std.testing.expectError(Error.ZeroShape, allocate(std.testing.allocator, &bad_zero, .{ .target_bits = 3 }));

    var only_dense = [1]Tensor{ tensor("n", 128, 128, .mlp) };
    only_dense[0].dense = true;
    try std.testing.expectError(Error.NoQuantizableTensors, allocate(std.testing.allocator, &only_dense, .{ .target_bits = 3 }));

    const bad_floor = [_]Tensor{tensor("m", 128, 128, .mlp)};
    const floors = [_]Floor{.{ .class = .mlp, .min_bits = 0.5 }};
    try std.testing.expectError(Error.FloorOutOfRange, allocate(std.testing.allocator, &bad_floor, .{ .target_bits = 3, .floors = &floors }));

    // ruling 4.2: trellis tensors tile 128 in K and N (the decode kernels' rule)
    const untiled = [_]Tensor{tensor("m", 24, 16, .mlp)};
    try std.testing.expectError(Error.BadTileShape, allocate(std.testing.allocator, &untiled, .{ .target_bits = 3 }));
    const narrow = [_]Tensor{ tensor("a", 5120, 48, .gdn), tensor("m", 128, 128, .mlp) };
    try std.testing.expectError(Error.BadTileShape, allocate(std.testing.allocator, &narrow, .{ .target_bits = 3 }));
    // the same thin projection is fine marked dense (in_proj_a's route in the real inventory)
    var ok_list = narrow;
    ok_list[0].dense = true;
    var ok = try allocate(std.testing.allocator, &ok_list, .{ .target_bits = 3 });
    ok.deinit(std.testing.allocator);
}

test "the plan: parts cover the trellis set, sha is stable and width-sensitive" {
    const ts = [_]Tensor{
        tensor("embed", 4096, 4096, .embed),
        tensor("l.gdn.qkv", 512, 512, .gdn),
        tensor("l.mlp.down", 512, 512, .mlp),
        tensor("lm_head", 512, 512, .head),
    };
    var ts_mut = ts;
    ts_mut[0].dense = true;
    const opts = Options{ .target_bits = 2.9, .head_bits = 4 };
    var al = try allocate(std.testing.allocator, &ts_mut, opts);
    defer al.deinit(std.testing.allocator);
    var p = try makePlan(std.testing.allocator, &ts_mut, al, opts);
    defer freePlan(std.testing.allocator, &p);
    try expectEqual(@as(usize, 3), p.parts.len); // embed is dense: no part
    try expectEqualStrings("l.gdn.qkv", p.parts[0].base);
    try expectEqualStrings("lm_head", p.parts[2].base);
    try expectEqual(@as(u8, 8), p.parts[2].halves); // the head carries its fixed width
    try expectEqualStrings("eaq-alloc/v1", p.method);
    try expectApproxEqAbs(2.9, p.target_bpw, 1e-12);
    const sha = p.plan_sha;
    for (sha) |byte| try expect(byte == byte); // defined
    // the sha recomputes identically...
    const sha2 = try planShaWith(std.testing.allocator, p);
    try expectEqualSlices(u8, &sha, &sha2);
    // ...and changes with any width
    var p2 = p;
    const mut = try std.testing.allocator.dupe(PartWidth, p.parts);
    defer std.testing.allocator.free(mut);
    p2.parts = mut;
    mut[1].halves += 1;
    const sha3 = try planShaWith(std.testing.allocator, p2);
    try expect(!std.mem.eql(u8, &sha, &sha3));
    // fresh-bit sizing helper: odd halves floor to ka
    try expectEqual(@as(u8, 2), p.minFreshBits()); // floors at 1.0 (halves 2)
    const json = try planToJson(std.testing.allocator, p);
    defer std.testing.allocator.free(json);
    try expect(std.mem.indexOf(u8, json, "\"bits\": \"4.0\"") != null); // head decimal from the integer
    try expect(std.mem.indexOf(u8, json, "\"plan_sha\"") != null);
    try expect(std.mem.indexOf(u8, json, "2.000000000000") == null); // no float-formatted halves
}

test "the real 27B inventory: kept 851, classes 96/432/192/129/1/1, quantized 401" {
    const ts = try qwen27Inventory(std.testing.allocator);
    defer freeInventory(std.testing.allocator, ts);
    var counts: [num_classes]usize = @splat(0);
    var quantized: usize = 0;
    for (ts) |t| {
        counts[@intFromEnum(t.class)] += 1;
        if (isQuantizable(t.class) and !t.dense) quantized += 1;
    }
    try expectEqual(@as(usize, 851), ts.len);
    try expectEqual(@as(usize, 96), counts[@intFromEnum(Class.attn)]);
    try expectEqual(@as(usize, 432), counts[@intFromEnum(Class.gdn)]);
    try expectEqual(@as(usize, 192), counts[@intFromEnum(Class.mlp)]);
    try expectEqual(@as(usize, 129), counts[@intFromEnum(Class.norm)]);
    try expectEqual(@as(usize, 1), counts[@intFromEnum(Class.embed)]);
    try expectEqual(@as(usize, 1), counts[@intFromEnum(Class.head)]);
    try expectEqual(@as(usize, 401), quantized); // 400 body + the head
}

test "the real 27B at 2.9 and 3.9 body bpw with the head at 4 (acceptance numbers)" {
    const ts = try qwen27Inventory(std.testing.allocator);
    defer freeInventory(std.testing.allocator, ts);
    var head_w: u64 = 0;
    for (ts) |t| {
        if (t.class == .head) head_w = t.weight();
    }
    for ([_]f64{ 2.9, 3.9 }) |t| {
        const opts = Options{ .target_bits = t, .head_bits = 4 };
        var al = try allocate(std.testing.allocator, ts, opts);
        defer al.deinit(std.testing.allocator);
        try expect(al.report.feasible);
        try expect(al.report.achieved_bits <= t + 1e-9); // the budget never overshoots
        try expectApproxEqAbs(t, al.report.achieved_bits, 0.02); // within one realistic step
        try expectEqual(@as(f64, 4.0), al.report.head_bits);
        try expectEqual(@as(usize, 400), al.report.quantized);
        // the byte ledger identity: total = (body_w * body + head_w * head) / (body_w + head_w)
        const bw: f64 = @floatFromInt(al.report.body_weight);
        const hw: f64 = @floatFromInt(head_w);
        try expectApproxEqAbs(
            (bw * al.report.achieved_bits + hw * 4.0) / (bw + hw),
            al.report.trellis_bits_total,
            1e-9,
        );
        // the head rides fixed and the mlp spread stays inside one half-step (ruling 7)
        var hmin: u8 = 16;
        var hmax: u8 = 2;
        for (ts, al.assignments) |x, a| {
            if (x.class == .head) try expectEqual(@as(u8, 8), a.halves);
            if (x.class == .mlp and !x.dense) {
                hmin = @min(hmin, a.halves);
                hmax = @max(hmax, a.halves);
            }
        }
        try expect(hmax - hmin <= 1); // mlp layers interleave within one half-step
        std.debug.print("\n  27B @ body {d} bpw (head 4): achieved {d:.6}, total {d:.6}, odd {d} of {d} body, " ++
            "attn avg {d:.3}, gdn avg {d:.3}, mlp avg {d:.3} (spread {d} half), head {d:.1}\n", .{
            t,
            al.report.achieved_bits,
            al.report.trellis_bits_total,
            al.report.odd.len,
            al.report.quantized,
            al.report.per_class[@intFromEnum(Class.attn)].avg_bits,
            al.report.per_class[@intFromEnum(Class.gdn)].avg_bits,
            al.report.per_class[@intFromEnum(Class.mlp)].avg_bits,
            hmax - hmin,
            al.report.head_bits,
        });
    }
}

test "the real 27B is deterministic: two runs agree on widths and plan sha" {
    const ts = try qwen27Inventory(std.testing.allocator);
    defer freeInventory(std.testing.allocator, ts);
    const opts = Options{ .target_bits = 2.9, .head_bits = 4 };
    var a1 = try allocate(std.testing.allocator, ts, opts);
    defer a1.deinit(std.testing.allocator);
    var p1 = try makePlan(std.testing.allocator, ts, a1, opts);
    defer freePlan(std.testing.allocator, &p1);
    var a2 = try allocate(std.testing.allocator, ts, opts);
    defer a2.deinit(std.testing.allocator);
    var p2 = try makePlan(std.testing.allocator, ts, a2, opts);
    defer freePlan(std.testing.allocator, &p2);
    try std.testing.expectEqualSlices(Assignment, a1.assignments, a2.assignments);
    try expectEqualSlices(u8, &p1.plan_sha, &p2.plan_sha);
}

const EmitPlan = struct { target: f64, head_bits: u4, body: usize, odd_parts: usize };

/// The CLI's verification path, as a test: allocate the real inventory, render the
/// plan JSON, read it back through `planFromJson`, and require the resume identity —
/// the embedded plan_sha equals the sha recomputed over the canonical serialization of
/// both the emitted and the parsed-back plan.
fn emitRoundTrip(e: *EmitPlan, a: std.mem.Allocator, target: f64) !void {
    const ts = try qwen27Inventory(a);
    defer freeInventory(a, ts);
    const opts = Options{ .target_bits = target, .head_bits = e.head_bits };
    var al = try allocate(a, ts, opts);
    defer al.deinit(a);
    var p = try makePlan(a, ts, al, opts);
    defer freePlan(a, &p);

    const json = try planToJson(a, p);
    defer a.free(json);
    e.body = al.report.quantized;
    e.odd_parts = 0;
    for (p.parts) |part| {
        if (part.halves % 2 == 1) e.odd_parts += 1;
    }

    var rp = try planFromJson(a, json);
    defer {
        for (rp.parts) |part| a.free(@constCast(part.base));
        a.free(@constCast(rp.parts));
        a.free(@constCast(rp.method));
    }
    try expectEqual(p.parts.len, rp.parts.len);
    for (p.parts, rp.parts) |x, y| {
        try expectEqualStrings(x.base, y.base);
        try expectEqual(x.halves, y.halves);
    }
    try expectEqual(@as(u8, e.head_bits * 2), rp.head_halves);
    try expectEqualStrings(method_name, rp.method);
    try expectApproxEqAbs(target, rp.target_bpw, 0);

    const sha_emitted = try planShaWith(a, p);
    const sha_parsed = try planShaWith(a, rp);
    try expectEqualSlices(u8, &p.plan_sha, &sha_emitted);
    try expectEqualSlices(u8, &p.plan_sha, &rp.plan_sha);
    try expectEqualSlices(u8, &rp.plan_sha, &sha_parsed);

    // a hand-edited file refuses: flip a width without re-printing its bits decimal
    const dot = std.mem.indexOf(u8, json, "\"halves\": ") orelse return error.TestUnexpectedResult;
    var tampered = try a.dupe(u8, json);
    defer a.free(tampered);
    tampered[dot + "\"halves\": ".len] = '9';
    try std.testing.expectError(Error.BadPlan, planFromJson(a, tampered));
}

test "the 27B plan files round-trip and the sha identity holds (2.9, 3.9)" {
    for ([_]EmitPlan{
        .{ .target = 2.9, .head_bits = 4, .body = 400, .odd_parts = 0 },
        .{ .target = 3.9, .head_bits = 4, .body = 400, .odd_parts = 0 },
    }) |want| {
        var e = want;
        try emitRoundTrip(&e, std.testing.allocator, e.target);
        try expectEqual(e.body, 400);
        try expect(e.odd_parts > 0); // the odd-halves decode dependency (T-HS-6/9/10)
        std.debug.print("  plan @ {d}: body {d}, odd {d} (plan_sha identity holds)\n", .{ e.target, e.body, e.odd_parts });
    }
}
