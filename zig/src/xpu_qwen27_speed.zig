//! tf-xpu-qwen27-speed <model dir> [ref dir]: the S04 (Qwen 3.8 27B decode speed) measurement tool.
//!
//! One B70, the 100 W cap, one process. It answers the two questions S04 asks, in this order:
//!   1. *How fast is a decode token?* The gate's own measurement (one token, capacity 512, >= 60 timed steps,
//!      median ms) run three ways: the eager path (a submit per launch), the same step recorded once into a
//!      regular command list and replayed per token (S01's lever), and the recorded step with S02's fusions on.
//!      Every recorded variant is checked against the eager path for the *same* token and the *same* options
//!      before any of its timings is trusted.
//!   2. *How close is the biggest kernel to the card's bandwidth?* Per-kernel device time of one eager decode
//!      step (Level Zero event timestamps) and, for each projection shape, the weight bytes one launch reads
//!      divided by its device time - against the 590 GB/s ceiling measured at this cap (FINDINGS §9).
//!
//! Every phase runs on its own `Scratch` (own device buffers, own recorded lists, own `Replay` bucket). That is
//! deliberate: a recorded list bakes the *addresses* of the buffers and `dims` slots it was captured against, so
//! two variants that shared a scratch would overwrite each other's slots and two phases that shared one would
//! replay the first phase's graph. One scratch per variant costs device memory only (each is ~1.6 GiB at
//! capacity 512) and removes the whole class of aliasing bugs.
//!
//! Exactness of a recorded list is checked the hard way: the recorded step is replayed at a cache length *two
//! rows past* the one it was recorded at, against the eager path for that same token at that same length. A
//! recorder that failed to rewrite the position / cache-length scalars inside the captured `dims` arrays produces
//! the row for its own recorded position instead, which is a different row - so this check fails loudly rather
//! than timing a list that decodes the wrong token.
const std = @import("std");
const fwd = @import("qwen27_forward");
const rt = fwd.rt;
const sys = fwd.sys;
const cfg = fwd.cfg;
const xw = fwd.xw;
const xs = fwd.xs;
const rp = fwd.rp;

const gpa = std.heap.page_allocator;
const fixtures = [_][]const u8{"fibonacci-raw"};
const capacity = 512;
/// One row buffer, 248320 bf16 = 0.5 MiB: three of them hold the reference row, the row under test and the
/// variant's greedy row.
const row_bytes = 1 << 20;

fn bf(w: u16) f32 {
    return @bitCast(@as(u32, w) << 16);
}

fn secs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

fn readFile(comptime fmt: []const u8, args: anytype) ![]u8 {
    var b: [1024]u8 = undefined;
    const p = sys.zfmt(&b, fmt, args) orelse return error.PathTooLong;
    return sys.readFileAlloc(gpa, p, 1 << 30) orelse {
        std.debug.print("cannot read {s}\n", .{p});
        return error.NoFile;
    };
}

fn readNpyU32(dir: []const u8, name: []const u8) ![]u32 {
    const bytes = try readFile("{s}/{s}.npy", .{ dir, name });
    defer gpa.free(bytes);
    if (bytes.len < 10 or !std.mem.eql(u8, bytes[0..6], "\x93NUMPY")) return error.BadNpy;
    const hl = std.mem.readInt(u16, bytes[8..10], .little);
    if (10 + hl > bytes.len) return error.BadNpy;
    const data = bytes[10 + hl ..];
    if (data.len % 4 != 0) return error.BadNpy;
    const out = try gpa.alloc(u32, data.len / 4);
    for (out, 0..) |*o, i| o.* = std.mem.readInt(u32, data[i * 4 ..][0..4], .little);
    return out;
}

fn loadGroup(dir: []const u8) !usize {
    const text = try readFile("{s}/config.json", .{dir});
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const root = switch (parsed.value) {
        .object => |o| o,
        else => return error.BadConfig,
    };
    for ([_][]const u8{ "quantization", "quantization_config" }) |key| {
        const v = root.get(key) orelse continue;
        if (v != .object) continue;
        if (v.object.get("group_size")) |gs| {
            if (gs != .integer or gs.integer < 1) return error.BadAffine;
            return @intCast(gs.integer);
        }
    }
    return error.MissingAffineFormat;
}

fn loadConfig(dir: []const u8) !cfg.Config {
    const text = try readFile("{s}/config.json", .{dir});
    defer gpa.free(text);
    var gen: ?[]u8 = null;
    var gb: [1024]u8 = undefined;
    if (sys.zfmt(&gb, "{s}/generation_config.json", .{dir})) |gp| gen = sys.readFileAlloc(gpa, gp, 1 << 22);
    defer if (gen) |g| gpa.free(g);
    return cfg.parse(gpa, text, gen);
}

/// Argmax over a bf16 row; ties keep the lower id (strict >), matching the fixtures' argmax.
fn argmax(row: []const u16) usize {
    var best: usize = 0;
    var bv = bf(row[0]);
    for (row, 0..) |w, i| {
        const v = bf(w);
        if (v > bv) {
            bv = v;
            best = i;
        }
    }
    return best;
}

fn intList(v: std.json.Value, out: *std.ArrayList(u32)) !void {
    if (v != .array) return error.BadReference;
    for (v.array.items) |x| try out.append(gpa, @intCast(x.integer));
}

fn median(sorted: []const u64) f64 {
    return @as(f64, @floatFromInt(sorted[sorted.len / 2])) / 1e6;
}

fn report(label: []const u8, steps: *std.ArrayList(u64)) void {
    std.mem.sort(u64, steps.items, {}, std.sort.asc(u64));
    var sum: u64 = 0;
    for (steps.items) |x| sum += x;
    const mean = @as(f64, @floatFromInt(sum)) / 1e6 / @as(f64, @floatFromInt(steps.items.len));
    std.debug.print("MEASURE {s}: {d} steps, median {d:.2} ms, mean {d:.2} ms, min {d:.2} ms -> {d:.2} tokens/s (median)\n", .{
        label,
        steps.items.len,
        median(steps.items),
        mean,
        @as(f64, @floatFromInt(steps.items[0])) / 1e6,
        1e9 / (median(steps.items) * 1e6),
    });
}

/// The 27B shapes, keyed by the projection's (outputs, inputs): the largest tensors by bytes read per token.
const Shape = struct { outputs: usize, inputs: usize, label: []const u8 };

const top_shapes = [_]Shape{
    .{ .outputs = cfg.Pinned.intermediate, .inputs = cfg.Pinned.hidden, .label = "mlp.gate_proj / mlp.up_proj (17408 x 5120)" },
    .{ .outputs = cfg.Pinned.hidden, .inputs = cfg.Pinned.intermediate, .label = "mlp.down_proj (5120 x 17408)" },
    .{ .outputs = cfg.Pinned.gdn_value_dim, .inputs = cfg.Pinned.hidden, .label = "linear_attn.in_proj_z / out_proj (6144 x 5120)" },
    .{ .outputs = cfg.Pinned.gdn_qkv_dim, .inputs = cfg.Pinned.hidden, .label = "linear_attn.in_proj_qkv (10240 x 5120)" },
    .{ .outputs = cfg.Pinned.vocab, .inputs = cfg.Pinned.hidden, .label = "lm_head / embed_tokens (248320 x 5120)" },
};

/// Bytes one launch of a projection reads from device memory: int4 words (inputs/8 u32) + group scales and biases
/// (inputs/64 bf16 each). This is the denominator of the achieved-GB/s figure; the activation row it also reads is
/// negligible beside it.
fn projectionBytes(outputs: usize, inputs: usize, group: usize) usize {
    return outputs * (inputs / 8) * 4 + 2 * outputs * (inputs / group) * 2;
}

/// One phase's own buffers and recorded steps: a scratch per variant, and a `Replay` per scratch, because a
/// recorded list must be replayed against the exact scratch and fusion set it was captured with.
const Variant = struct {
    name: []const u8,
    fuse: fwd.Fuse,
    /// One recorded step per capacity; the decode step's grid depends only on the capacity, so one list serves
    /// every length and every phase of this variant.
    bucket: *rp.Replay,
};

fn shapeLabel(outputs: usize, inputs: usize) []const u8 {
    for (top_shapes) |s| {
        if (s.outputs == outputs and s.inputs == inputs) return s.label;
    }
    return "";
}

fn printProfile(sc: *fwd.Scratch, reps: usize, group: usize) !void {
    const p = &sc.prof.?;
    var total: f64 = 0;
    for (p.ns) |x| total += x / @as(f64, @floatFromInt(reps));
    std.debug.print("MEASURE per-kernel device time, one eager decode step (mean of {d}), total {d:.3} ms\n", .{ reps, total / 1e6 });
    std.debug.print("KERNEL            launches  us/step      %   weights_MiB     GB/s\n", .{});
    for (fwd.kernel_ids) |id| {
        const i = @intFromEnum(id);
        if (p.count[i] == 0) continue;
        const us = p.ns[i] / @as(f64, @floatFromInt(reps)) / 1e3;
        const launches = p.count[i] / reps;
        std.debug.print("KERNEL {s:<14} {d:>8} {d:>9.1} {d:>6.1}%  {s:>11} {s:>8}\n", .{
            @tagName(id),
            launches,
            us,
            100.0 * p.ns[i] / @as(f64, @floatFromInt(reps)) / total,
            "n/a",
            "n/a",
        });
    }
    // Per-shape qmm time, and the achieved bandwidth of the shapes that matter: the weight bytes a shape reads
    // once per token divided by the device time of one launch.
    var it = p.shape_ns.iterator();
    var rows: std.ArrayList(ShapeRow) = .empty;
    defer rows.deinit(gpa);
    while (it.next()) |kv| {
        const key = kv.key_ptr.*;
        const outputs = fwd.Profile.shapeOutputs(key);
        const inputs = fwd.Profile.shapeInputs(key);
        const count = p.shape_count.get(key) orelse 0;
        if (count == 0) continue;
        try rows.append(gpa, .{
            .outputs = outputs,
            .inputs = inputs,
            .us = kv.value_ptr.* / @as(f64, @floatFromInt(reps)) / 1e3 / @as(f64, @floatFromInt(count)),
            .bytes = projectionBytes(outputs, inputs, group),
        });
    }
    std.mem.sort(ShapeRow, rows.items, {}, struct {
        fn lt(_: void, a: ShapeRow, b: ShapeRow) bool {
            return a.bytes > b.bytes;
        }
    }.lt);
    std.debug.print("MEASURE qmm by shape, weight bytes per launch / device time of one launch (top 3 by bytes)\n", .{});
    for (rows.items[0..@min(rows.items.len, 6)], 0..) |r, i| {
        const gbs = @as(f64, @floatFromInt(r.bytes)) / (r.us * 1e3);
        std.debug.print("SHAPE {d:>7} x {d:<7} {d:>9.2} us  {d:>8.1} MiB  {d:>7.1} GB/s{s}  {s}\n", .{
            r.outputs, r.inputs, r.us, @as(f64, @floatFromInt(r.bytes)) / (1 << 20), gbs, if (i < 3) "  <- top 3" else "", shapeLabel(r.outputs, r.inputs),
        });
    }
}

const ShapeRow = struct { outputs: usize, inputs: usize, us: f64, bytes: usize };

/// One variant's timed decode: prefill, then 3 warm-up + 60 timed one-token steps through the recorded list.
/// `steps` receives the timed segments in submission order (the median is what the gate reads).
fn timeRecorded(s: *rt.session.Session, m: *fwd.Model, sc: *fwd.Scratch, v: Variant, ids: []const u32, steps: *std.ArrayList(u64)) !void {
    var cache = try xs.Cache.init(gpa, s, sc.cfg, capacity);
    defer cache.deinit();
    const row: []u16 = @ptrCast(@alignCast(try gpa.alloc(u8, row_bytes)));
    defer gpa.free(@as([]u8, @ptrCast(row)));
    var rq = try rt.cmdlist.Queue.init(&s.ctx);
    defer rq.deinit();
    try fwd.run(s, m, sc, &cache, ids, .{ .fuse = v.fuse });
    try cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
    var next: [1]u32 = undefined;
    next[0] = @intCast(argmax(row));
    var step = try rp.recordStep(s, m, sc, &cache, &next, v.fuse, gpa);
    defer step.deinit(gpa);
    try v.bucket.put(1, capacity, step);
    for (0..63) |i| {
        try cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
        next[0] = @intCast(argmax(row));
        const t1 = sys.nowNs();
        const vals: rp.Values = .{ .token = next[0], .position = @intCast(cache.len), .cache_len = @intCast(cache.len) };
        try rp.replayStep(v.bucket.find(1, capacity).?, &rq, vals);
        cache.len += 1;
        if (i >= 3) try steps.append(gpa, sys.nowNs() - t1);
    }
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    const dir: []const u8 = if (argv.len > 1) argv[1] else "/storage01/models/Qwen3.8-27B-MLX-4bit";
    const refdir: []const u8 = if (argv.len > 2) argv[2] else "/storage01/qwen-flash-on-intel/exp/ref/qwen38-27b-mlx4-fp32";

    var s: rt.session.Session = undefined;
    s.open(.{}) catch |e| {
        std.debug.print("cannot open a card: {s} ({s})\n", .{ @errorName(e), s.diag.message() });
        return 1;
    };
    defer s.close();
    std.debug.print("card {s} ({s})\n", .{ s.bdf(), s.ctx.dev.name() });

    const config = loadConfig(dir) catch |e| {
        std.debug.print("config error: {s}\n", .{@errorName(e)});
        return 1;
    };
    const group = loadGroup(dir) catch |e| {
        std.debug.print("quantization error: {s}\n", .{@errorName(e)});
        return 1;
    };
    var w = try xw.load(gpa, &s, dir, config, group);
    defer w.deinit();
    var m: fwd.Model = .{ .weights = &w, .kernels = undefined };
    try m.kernels.load(&s, gpa);
    defer m.kernels.deinit();
    std.debug.print("weights {d} tensors, {d:.1} MiB and kernels loaded (group {d})\n", .{ w.ledger.len, @as(f64, @floatFromInt(w.total_bytes)) / (1 << 20), group });

    const ids = try readNpyU32(refdir, fixtures[0]);
    defer gpa.free(ids);
    // The fixture's own greedy tokens: the reference the recorded variants must match token for token.
    var want: std.ArrayList(u32) = .empty;
    defer want.deinit(gpa);
    {
        const text = try readFile("{s}/greedy-{s}.json", .{ refdir, fixtures[0] });
        defer gpa.free(text);
        const doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer doc.deinit();
        const entry = doc.value.object.get(fixtures[0]).?;
        try intList(entry.object.get("tokens").?, &want);
    }

    const variants = [_]Variant{
        .{ .name = "decode step (recorded replay, 1 token, capacity 512)", .fuse = .none, .bucket = undefined },
        .{ .name = "decode step (recorded replay + qmm_pair + norm_qmm + mlp_act_down, 1 token, capacity 512)", .fuse = .all, .bucket = undefined },
    };
    var buckets: [variants.len]rp.Replay = undefined;
    var scs: [variants.len]fwd.Scratch = undefined;
    var made: usize = 0;
    defer {
        for (scs[0..made]) |*sc| sc.deinit();
        for (buckets[0..made]) |*b| b.deinit();
    }
    for (&buckets, &scs, 0..) |*b, *sc, i| {
        b.* = rp.Replay.init(gpa);
        sc.* = try fwd.Scratch.init(&s, config, ids.len, capacity, 1);
        made = i + 1;
    }

    // ---- 1. The gate's decode measurement, eager: one token per segment, a submit per launch.
    {
        const sc = &scs[0];
        var cache = try xs.Cache.init(gpa, &s, config, capacity);
        defer cache.deinit();
        const row: []u16 = @ptrCast(@alignCast(try gpa.alloc(u8, row_bytes)));
        defer gpa.free(@as([]u8, @ptrCast(row)));
        var eager: std.ArrayList(u64) = .empty;
        defer eager.deinit(gpa);
        try fwd.run(&s, &m, sc, &cache, ids, .{});
        var next: [1]u32 = undefined;
        for (0..63) |i| {
            try cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
            next[0] = @intCast(argmax(row));
            const t1 = sys.nowNs();
            try fwd.run(&s, &m, sc, &cache, &next, .{});
            if (i >= 3) try eager.append(gpa, sys.nowNs() - t1);
        }
        report("decode step (eager, 1 token, capacity 512)", &eager);
    }

    // ---- 2. The same step recorded once (S01) and replayed, unfused and fused. Each variant is checked against
    // the eager path at a *different* cache length than the one it was recorded at (see the file header), and its
    // greedy tokens are compared with the fixture's.
    for (variants, 0..) |v, vi| {
        const sc = &scs[vi];
        var rq = try rt.cmdlist.Queue.init(&s.ctx);
        defer rq.deinit();

        // 2a. Exactness of the recorded list itself, at a shifted cache length.
        {
            var rec_cache = try xs.Cache.init(gpa, &s, config, capacity);
            defer rec_cache.deinit();
            var ref_cache = try xs.Cache.init(gpa, &s, config, capacity);
            defer ref_cache.deinit();
            const row: []u16 = @ptrCast(@alignCast(try gpa.alloc(u8, row_bytes)));
            defer gpa.free(@as([]u8, @ptrCast(row)));
            const ref_row: []u16 = @ptrCast(@alignCast(try gpa.alloc(u8, row_bytes)));
            defer gpa.free(@as([]u8, @ptrCast(ref_row)));

            // The token decoded at the shifted length: what the eager path predicts two steps after the prompt.
            try fwd.run(&s, &m, sc, &ref_cache, ids, .{ .fuse = v.fuse });
            var tok: u32 = undefined;
            for (0..2) |_| {
                try ref_cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
                tok = @intCast(argmax(row));
                try fwd.run(&s, &m, sc, &ref_cache, &.{tok}, .{ .fuse = v.fuse });
            }
            // The recorded list is captured against the same scratch, at the same length.
            try fwd.run(&s, &m, sc, &rec_cache, ids, .{ .fuse = v.fuse });
            try fwd.run(&s, &m, sc, &rec_cache, &.{tok}, .{ .fuse = v.fuse });
            var step = try rp.recordStep(&s, &m, sc, &rec_cache, &.{tok}, v.fuse, gpa);
            defer step.deinit(gpa);
            try rp.replayStep(&step, &rq, .{ .token = tok, .position = @intCast(rec_cache.len), .cache_len = @intCast(rec_cache.len) });
            rec_cache.len += 1;
            try rec_cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
            try ref_cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(ref_row));
            const same = std.mem.eql(u8, std.mem.sliceAsBytes(row), std.mem.sliceAsBytes(ref_row));
            std.debug.print("GATE D {s} record-vs-eager logits row: {s}\n", .{ v.name, if (same) "bit-identical" else "DIFFERENT" });
            if (!same) {
                std.debug.print("GATE D {s}: FAIL (the recorded list did not reproduce the eager step)\n", .{v.name});
                return 1;
            }
        }

        // 2b. Greedy through the recorded list, token by token, against the fixture.
        {
            var cache = try xs.Cache.init(gpa, &s, config, capacity);
            defer cache.deinit();
            const row: []u16 = @ptrCast(@alignCast(try gpa.alloc(u8, row_bytes)));
            defer gpa.free(@as([]u8, @ptrCast(row)));
            try fwd.run(&s, &m, sc, &cache, ids, .{ .fuse = v.fuse });
            try cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
            var next: [1]u32 = undefined;
            next[0] = @intCast(argmax(row));
            var step = try rp.recordStep(&s, &m, sc, &cache, &next, v.fuse, gpa);
            defer step.deinit(gpa);
            try v.bucket.put(1, capacity, step);
            var matched: usize = 0;
            for (0..want.items.len) |i| {
                try cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
                const got: u32 = @intCast(argmax(row));
                if (got != want.items[i]) {
                    std.debug.print("GATE D {s} greedy: FIRST DIVERGENCE at step {d}: xpu {d}, reference {d}\n", .{ v.name, i, got, want.items[i] });
                    break;
                }
                matched += 1;
                if (i + 1 == want.items.len) break;
                const vals: rp.Values = .{ .token = got, .position = @intCast(cache.len), .cache_len = @intCast(cache.len) };
                try rp.replayStep(v.bucket.find(1, capacity).?, &rq, vals);
                cache.len += 1;
            }
            std.debug.print("GATE D {s} greedy {d}/{d} tokens: {s}\n", .{ v.name, matched, want.items.len, if (matched == want.items.len) "PASS" else "FAIL" });
            if (matched != want.items.len) {
                std.debug.print("GATE D {s}: FAIL (replay drifted from the reference)\n", .{v.name});
                return 1;
            }
        }

        // 2c. Timing: prefill fresh, record once for this length, then 3 warm-up + 60 timed replays.
        var steps: std.ArrayList(u64) = .empty;
        defer steps.deinit(gpa);
        timeRecorded(&s, &m, sc, v, ids, &steps) catch |e| {
            std.debug.print("MEASURE {s}: timing failed: {s}\n", .{ v.name, @errorName(e) });
            return 1;
        };
        report(v.name, &steps);
    }

    // ---- 3. Per-kernel device time of one eager decode step, with the achieved bandwidth of the big shapes.
    {
        const sc = &scs[0];
        var cache = try xs.Cache.init(gpa, &s, config, capacity);
        defer cache.deinit();
        const row: []u16 = @ptrCast(@alignCast(try gpa.alloc(u8, row_bytes)));
        defer gpa.free(@as([]u8, @ptrCast(row)));
        var prof = try fwd.Profile.init(&s, gpa, 8192);
        defer prof.deinit(gpa);
        sc.prof = prof;
        defer sc.prof = null;
        try fwd.run(&s, &m, sc, &cache, ids, .{});
        var next: [1]u32 = undefined;
        const reps = 3;
        for (0..reps) |_| {
            try cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
            next[0] = @intCast(argmax(row));
            try fwd.run(&s, &m, sc, &cache, &next, .{});
            try sc.prof.?.collect(gpa);
        }
        try printProfile(sc, reps, group);
    }

    // ---- 4. The largest projections by shape, from the model's own weight tensors (sanity for the table above).
    std.debug.print("MEASURE largest projection tensors (weight + scales + biases, int4 g{d})\n", .{group});
    for (top_shapes) |sh| {
        std.debug.print("SHAPE {s:<44} {d:>7.1} MiB\n", .{ sh.label, @as(f64, @floatFromInt(projectionBytes(sh.outputs, sh.inputs, group))) / (1 << 20) });
    }
    return 0;
}
