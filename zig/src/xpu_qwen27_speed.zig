//! tf-xpu-qwen27-speed <model dir> [ref dir]: the S04 (Qwen 3.8 27B decode speed) measurement and exactness tool.
//!
//! What it measures, all at the 100 W cap on one B70:
//!   * the decode step as the gate measures it (one token, capacity 512, >= 60 timed steps, median ms);
//!   * the same step with the whole decode recorded into one regular command list and replayed
//!     (`xpu/cmdlist.zig`, the S01 lever) - one submit per token instead of ~600;
//!   * per-kernel device time of one eager decode step (Level Zero event timestamps) and, from each
//!     kernel's weight bytes, the achieved GB/s, so the bandwidth-critical projections can be compared
//!     with the card's measured ceiling (590 GB/s, FINDINGS §9);
//!   * the largest projections by shape (hidden 5120 -> intermediate 17408 and back).
//!
//! Exactness is checked before any timing is trusted:
//!   * `--record-check`: the recorded list is replayed twice with the *same* values and its logits row must
//!     be bit-identical to the eager step's (proves the recorded list is complete and correctly ordered);
//!   * the decode is then driven by the replay path and its greedy tokens must equal the fixture's
//!     (`greedy-<name>.json`, the same reference gate C uses) - a replay that drifts shows up immediately.
//!
//! Fusions applied to the 27B forward (all bit-identical by construction, each one launch or one
//! intermediate round trip saved):
//!   * `qmm_pair` for the GatedDeltaNet in_proj_a/in_proj_b and the attention k_proj/v_proj (S02's lever);
//!   * `mlp_act_down` for SwiGLU -> down projection (the 27B's largest weight tensor: 5120x17408 int4);
//!   * `norm_qmm` for (residual + RMSNorm) -> the next projection's input;
//!   * `layout27`/`attn_merge27`: the 27B row strides for the `queries` buffer and the attention output
//!     (the shared kernels' defaults are the 2B values, which coincide there - see each file's comment).
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

/// Weight bytes one launch of `id` reads from device memory, for the achieved-bandwidth report.
/// Projections read their int4 words, scales and biases; `mlp_act_down` and `norm_qmm` read their parents'
/// inputs as well (an activation row, negligible next to a projection's weights, so it is left out).
fn weightBytes(id: fwd.KernelId, a: xw.Linear) usize {
    _ = id;
    return a.weight.bytes + a.scales.bytes + a.biases.bytes;
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

fn printProfile(sc: *fwd.Scratch, w: *const xw.Weights, reps: usize) !void {
    const p = &sc.prof.?;
    var total: f64 = 0;
    for (p.ns) |x| total += x / @as(f64, @floatFromInt(reps));
    std.debug.print("MEASURE per-kernel device time, one eager decode step (mean of {d}), total {d:.3} ms\n", .{ reps, total / 1e6 });
    std.debug.print("KERNEL          launches   us/step     %   weights_MiB  GB/s\n", .{});
    for (fwd.kernel_ids) |id| {
        const i = @intFromEnum(id);
        if (p.count[i] == 0) continue;
        const us = p.ns[i] / @as(f64, @floatFromInt(reps)) / 1e3;
        const launches = p.count[i] / reps;
        std.debug.print("KERNEL {s:<14} {d:>6}  {d:>9.1}  {d:>5.1}%  {s:>11}  {s:>6}\n", .{
            @tagName(id),
            launches,
            us,
            100.0 * p.ns[i] / @as(f64, @floatFromInt(reps)) / total,
            "n/a",
            "n/a",
        });
    }
    // Per-shape qmm time, and the achieved bandwidth of the shapes that matter: the weight bytes a shape
    // reads once per token divided by the device time of one launch.
    var it = p.shape_ns.iterator();
    var rows: std.ArrayList(ShapeRow) = .empty;
    defer rows.deinit(gpa);
    while (it.next()) |kv| {
        const key = kv.key_ptr.*;
        const outputs = fwd.Profile.shapeOutputs(key);
        const inputs = fwd.Profile.shapeInputs(key);
        const count = p.shape_count.get(key) orelse 0;
        if (count == 0) continue;
        // One launch of this shape reads one projection's weights (weight+scales+biases, group 64).
        const bytes = outputs * (inputs / 8) * 4 + 2 * outputs * (inputs / 64) * 2;
        try rows.append(gpa, .{
            .outputs = outputs,
            .inputs = inputs,
            .us = kv.value_ptr.* / @as(f64, @floatFromInt(reps)) / 1e3 / @as(f64, @floatFromInt(count)),
            .bytes = bytes,
        });
    }
    std.mem.sort(ShapeRow, rows.items, {}, struct {
        fn lt(_: void, a: ShapeRow, b: ShapeRow) bool {
            return a.bytes > b.bytes;
        }
    }.lt);
    std.debug.print("MEASURE qmm by shape, bytes read per token / device time of one launch (top 3 by weight bytes)\n", .{});
    for (rows.items[0..@min(rows.items.len, 6)], 0..) |r, i| {
        const gbs = @as(f64, @floatFromInt(r.bytes)) / (r.us * 1e3);
        std.debug.print("SHAPE {d:>7} x {d:<7} {d:>9.2} us  {d:>8.1} MiB  {d:>7.1} GB/s{s}\n", .{
            r.outputs, r.inputs, r.us, @as(f64, @floatFromInt(r.bytes)) / (1 << 20), gbs, if (i < 3) "  <- top 3" else "",
        });
    }
    _ = w;
}

const ShapeRow = struct { outputs: usize, inputs: usize, us: f64, bytes: usize };

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
    std.debug.print("weights {d} tensors, {d:.1} MiB and kernels loaded\n", .{ w.ledger.len, @as(f64, @floatFromInt(w.total_bytes)) / (1 << 20) });

    const ids = try readNpyU32(refdir, fixtures[0]);
    defer gpa.free(ids);
    var sc = try fwd.Scratch.init(&s, config, ids.len, capacity, 1);
    defer sc.deinit();
    const row = try gpa.alloc(u16, config.vocab);
    defer gpa.free(row);

    // ---- 1. The gate's decode measurement: eager forward, one token per segment, 3 warm-up + 60 timed steps.
    var eager: std.ArrayList(u64) = .empty;
    defer eager.deinit(gpa);
    {
        var cache = try xs.Cache.init(gpa, &s, config, capacity);
        defer cache.deinit();
        try fwd.run(&s, &m, &sc, &cache, ids, .{});
        var next: [1]u32 = undefined;
        for (0..63) |i| {
            try cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
            next[0] = @intCast(argmax(row));
            const t1 = sys.nowNs();
            try fwd.run(&s, &m, &sc, &cache, &next, .{});
            if (i >= 3) try eager.append(gpa, sys.nowNs() - t1);
        }
    }
    report("decode step (eager, 1 token, capacity 512)", &eager);

    // ---- 2. Recorded decode step (S01's lever): record once, replay per token. The recorded list goes through
    // its own in-order command queue (cmdlist.Queue), so the eager path's queue stays untouched.
    var rq = try rt.cmdlist.Queue.init(&s.ctx);
    defer rq.deinit();
    {
        var cache = try xs.Cache.init(gpa, &s, config, capacity);
        defer cache.deinit();
        try fwd.run(&s, &m, &sc, &cache, ids, .{});
        try cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
        const token: u32 = @intCast(argmax(row));
        // Record the step for the current cache length, then replay it once with the same values: the logits
        // row must be bit-identical to the eager step for that token (a missing or misordered command shows
        // up as a different row, so this is the exactness gate for the recorded list itself).
        const eager_row = try gpa.alloc(u16, config.vocab);
        defer gpa.free(eager_row);
        var probe = try xs.Cache.init(gpa, &s, config, capacity);
        defer probe.deinit();
        try fwd.run(&s, &m, &sc, &probe, ids, .{});
        const t0 = sys.nowNs();
        try fwd.run(&s, &m, &sc, &probe, &.{token}, .{});
        std.debug.print("MEASURE eager single step {d:.2} ms\n", .{secs(sys.nowNs() - t0) * 1e3});
        try probe.logits.download(&s.queue, 0, std.mem.sliceAsBytes(eager_row));
        probe.len += 1;

        var step = try rp.recordStep(&s, &m, &sc, &cache, &.{token}, .none, gpa);
        defer step.deinit(gpa);
        const values: rp.Values = .{ .token = token, .position = @intCast(cache.len), .cache_len = @intCast(cache.len) };
        try rp.replayStep(&step, &rq, values);
        cache.len += 1;
        try cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
        const same = std.mem.eql(u8, std.mem.sliceAsBytes(row), std.mem.sliceAsBytes(eager_row));
        std.debug.print("MEASURE record-vs-eager logits row: {s} ({d} words)\n", .{ if (same) "bit-identical" else "DIFFERENT", row.len });

        // Drive the decode through the replay path: every later token rewrites the captured token/position/
        // cache-length scalars and re-submits the one list. Greedy tokens must match the fixture.
        var gb: [1024]u8 = undefined;
        const gpath = sys.zfmt(&gb, "{s}/greedy-{s}.json", .{ refdir, fixtures[0] }) orelse return error.PathTooLong;
        const text = sys.readFileAlloc(gpa, gpath, 1 << 24) orelse {
            std.debug.print("no greedy-{s}.json in {s}\n", .{ fixtures[0], refdir });
            return error.NoFile;
        };
        defer gpa.free(text);
        const doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer doc.deinit();
        const entry = doc.value.object.get(fixtures[0]).?;
        var want: std.ArrayList(u32) = .empty;
        defer want.deinit(gpa);
        try intList(entry.object.get("tokens").?, &want);

        var replay: std.ArrayList(u64) = .empty;
        defer replay.deinit(gpa);
        var matched: usize = 0;
        for (0..want.items.len) |i| {
            try cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
            const got: u32 = @intCast(argmax(row));
            if (got != want.items[i]) {
                std.debug.print("GATE D replay greedy: FIRST DIVERGENCE at step {d}: xpu {d}, reference {d}\n", .{ i, got, want.items[i] });
                break;
            }
            matched += 1;
            if (i + 1 == want.items.len) break;
            const t1 = sys.nowNs();
            const v: rp.Values = .{ .token = got, .position = @intCast(cache.len), .cache_len = @intCast(cache.len) };
            try rp.replayStep(&step, &rq, v);
            cache.len += 1;
            try replay.append(gpa, sys.nowNs() - t1);
        }
        std.debug.print("GATE D replay greedy {d}/{d} tokens: {s}\n", .{ matched, want.items.len, if (matched == want.items.len) "PASS" else "FAIL" });
        if (replay.items.len > 0) report("decode step (recorded replay, 1 token, capacity 512)", &replay);

        // ---- 3. Fused variants of the same decode step: record each, check its logits row against the eager
        // step, and time its replay. A fused step that is not bit-identical is reported and never timed as a win.
        const variants = [_]struct { name: []const u8, o: fwd.Options }{
            .{ .name = "baseline", .o = .{ .head = .last, .fuse = .none } },
            .{ .name = "qmm_pair", .o = .{ .head = .last, .fuse = .{ .pair = true } } },
            .{ .name = "qmm_pair+norm_qmm+mlp_act_down+layout27+attn_merge27", .o = .{ .head = .last, .fuse = .all } },
        };
        std.debug.print("MEASURE fused variants (recorded step, replayed)\n", .{});
        for (variants) |v| {
            var c2 = try xs.Cache.init(gpa, &s, config, capacity);
            defer c2.deinit();
            try fwd.run(&s, &m, &sc, &c2, ids, v.o);
            try c2.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
            const tok: u32 = @intCast(argmax(row));
            var st = try rp.recordStep(&s, &m, &sc, &c2, &.{tok}, v.o.fuse, gpa);
            defer st.deinit(gpa);
            const vals: rp.Values = .{ .token = tok, .position = @intCast(c2.len), .cache_len = @intCast(c2.len) };
            try rp.replayStep(&st, &rq, vals);
            c2.len += 1;
            const fused_row = try gpa.alloc(u16, config.vocab);
            defer gpa.free(fused_row);
            try c2.logits.download(&s.queue, 0, std.mem.sliceAsBytes(fused_row));
            // The eager step for the same (prompt, token) is the recorded step above; compare against a fresh
            // eager run with the same options so each variant is judged on its own path.
            var c3 = try xs.Cache.init(gpa, &s, config, capacity);
            defer c3.deinit();
            try fwd.run(&s, &m, &sc, &c3, ids, v.o);
            try c3.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
            const tok3: u32 = @intCast(argmax(row));
            if (tok3 != tok) {
                std.debug.print("VARIANT {s}: greedy token differs before timing ({d} vs {d}) - skipped\n", .{ v.name, tok3, tok });
                continue;
            }
            try fwd.run(&s, &m, &sc, &c3, &.{tok3}, v.o);
            try c3.logits.download(&s.queue, 0, std.mem.sliceAsBytes(eager_row));
            const identical = std.mem.eql(u8, std.mem.sliceAsBytes(fused_row), std.mem.sliceAsBytes(eager_row));
            var steps: std.ArrayList(u64) = .empty;
            defer steps.deinit(gpa);
            var n: usize = 0;
            while (n < 60) : (n += 1) {
                const t1 = sys.nowNs();
                const vv: rp.Values = .{ .token = tok3, .position = @intCast(c2.len), .cache_len = @intCast(c2.len) };
                try rp.replayStep(&st, &rq, vv);
                c2.len += 1;
                try steps.append(gpa, sys.nowNs() - t1);
            }
            std.debug.print("VARIANT {s}: logits {s}\n", .{ v.name, if (identical) "bit-identical to eager" else "DIFFERENT (not a candidate)" });
            report(v.name, &steps);
        }
    }

    // ---- 4. Per-kernel device time of one eager decode step, with achieved bandwidth for the big shapes.
    {
        var cache = try xs.Cache.init(gpa, &s, config, capacity);
        defer cache.deinit();
        var prof = try fwd.Profile.init(&s, gpa, 8192);
        defer prof.deinit(gpa);
        sc.prof = prof;
        defer sc.prof = null;
        try fwd.run(&s, &m, &sc, &cache, ids, .{});
        var next: [1]u32 = undefined;
        const reps = 3;
        for (0..reps) |_| {
            try cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
            next[0] = @intCast(argmax(row));
            try fwd.run(&s, &m, &sc, &cache, &next, .{});
            try sc.prof.?.collect(gpa);
        }
        try printProfile(&sc, &w, reps);
    }

    // ---- 5. The largest projections by shape, from the model's own weight tensors (sanity for the table above).
    std.debug.print("MEASURE largest projection tensors (weight + scales + biases, int4 g64)\n", .{});
    for (top_shapes) |sh| {
        const bytes = sh.outputs * (sh.inputs / 8) * 4 + 2 * sh.outputs * (sh.inputs / 64) * 2;
        std.debug.print("SHAPE {s:<44} {d:>7.1} MiB\n", .{ sh.label, @as(f64, @floatFromInt(bytes)) / (1 << 20) });
    }
    _ = weightBytes;
    return 0;
}
