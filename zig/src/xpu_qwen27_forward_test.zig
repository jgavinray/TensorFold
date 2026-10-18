//! tf-xpu-qwen27-forward-test [model dir] [ref dir]: on-card gates for the Qwen 3.8 27B XPU forward (Phase 6.3/6.4).
//! Gate C: greedy tokens of the fixture's prompt vs greedy-<name>.json, one row per generated token, on the card.
//! The same run measures the decode step: one token per segment, capacity 512, >= 60 timed steps, median reported.
//! The reference directory holds the MLX fp32 fixtures (exp/ref/qwen38-27b-mlx4-fp32): <name>.npy (prompt ids),
//! greedy-<name>.json (prompt_tokens + tokens). Gate C needs only the greedy file; the teacher-forced logits rule is
//! the contract gate's job (gates/check_logits.py driving tf-xpu-qwen27-logits).
//! The forward is families/qwen27/xpu_forward.zig, the same module the tools use; the kernels come from the same
//! embedded 27B modules (zig/kernels/xpu/prebuilt-27b or ocloc, see zig/build/xpu.zig).
//! Expected runtime: minutes once a card is free (the runtime lock waits); kill at 2x.
const std = @import("std");
const fwd = @import("qwen27_forward");
const rt = fwd.rt;
const sys = fwd.sys;
const cfg = fwd.cfg;
const xw = fwd.xw;
const xs = fwd.xs;

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
    var cb: [1024]u8 = undefined;
    const cpath = sys.zfmt(&cb, "{s}/config.json", .{dir}) orelse return error.PathTooLong;
    const text = sys.readFileAlloc(gpa, cpath, 1 << 22) orelse return error.NoConfig;
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
    var cb: [1024]u8 = undefined;
    const cpath = sys.zfmt(&cb, "{s}/config.json", .{dir}) orelse return error.PathTooLong;
    const text = sys.readFileAlloc(gpa, cpath, 1 << 22) orelse return error.NoConfig;
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

/// Gate C on one fixture: prompt prefill, then one greedy row per reference token; also times each one-row decode
/// segment so the caller can print the decode-step median over the fixture's tokens.
fn gateC(s: *rt.session.Session, m: *fwd.Model, sc: *fwd.Scratch, refdir: []const u8, name: []const u8, steps: *std.ArrayList(u64)) !bool {
    var gb: [1024]u8 = undefined;
    const gpath = sys.zfmt(&gb, "{s}/greedy-{s}.json", .{ refdir, name }) orelse return error.PathTooLong;
    const text = sys.readFileAlloc(gpa, gpath, 1 << 24) orelse {
        std.debug.print("gate C {s}: no greedy-{s}.json in the reference directory, skipped\n", .{ name, name });
        return true;
    };
    defer gpa.free(text);
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer doc.deinit();
    const entry = doc.value.object.get(name) orelse return error.BadReference;
    var prompt: std.ArrayList(u32) = .empty;
    defer prompt.deinit(gpa);
    var want: std.ArrayList(u32) = .empty;
    defer want.deinit(gpa);
    try intList(entry.object.get("prompt_tokens").?, &prompt);
    try intList(entry.object.get("tokens").?, &want);

    var cache = try xs.Cache.init(gpa, s, sc.cfg, capacity);
    defer cache.deinit();
    const row = try gpa.alloc(u16, sc.cfg.vocab);
    defer gpa.free(row);
    var next: [1]u32 = undefined;
    var diverged: ?usize = null;
    var matched: usize = 0;
    const t0 = sys.nowNs();
    try fwd.run(s, m, sc, &cache, prompt.items, .{});
    const prefill_ns = sys.nowNs() - t0;
    var i: usize = 0;
    while (i < want.items.len) : (i += 1) {
        try cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
        const got: u32 = @intCast(argmax(row));
        if (got != want.items[i]) {
            diverged = i;
            std.debug.print("gate C {s}: FIRST DIVERGENCE at step {d}: xpu token {d}, reference token {d}, xpu logit of reference token {d:.4}, of its own {d:.4}\n", .{ name, i, got, want.items[i], bf(row[want.items[i]]), bf(row[got]) });
            break;
        }
        matched += 1;
        if (i + 1 == want.items.len) break;
        next[0] = got;
        const t1 = sys.nowNs();
        try fwd.run(s, m, sc, &cache, &next, .{});
        try steps.append(gpa, sys.nowNs() - t1);
    }
    const ok = diverged == null;
    std.debug.print("gate C {s}: prompt {d} tokens (prefill {d:.3} s), {d}/{d} greedy tokens match -> {s}\n", .{ name, prompt.items.len, secs(prefill_ns), matched, want.items.len, if (ok) "ok" else "DIVERGED" });
    return ok;
}

/// One-row decode segments, capacity 512, until >= 60 timed steps (the contract's measurement line). The prompt is
/// the fixture's, so the first step follows a real prefill; every later step is one token.
/// The timed steps go through the recorded-step replay (S01): the one-token step is recorded once into a regular
/// command list and each token re-submits that one list, instead of ~660 immediate-list submits per token. The
/// recorded list is checked against the eager step for the same token before any timing is trusted, so a replay
/// that dropped or reordered a command cannot be mistaken for a speedup.
fn decodeSteps(s: *rt.session.Session, m: *fwd.Model, sc: *fwd.Scratch, ids: []const u32, steps: *std.ArrayList(u64)) !void {
    var cache = try xs.Cache.init(gpa, s, sc.cfg, capacity);
    defer cache.deinit();
    const row = try gpa.alloc(u16, sc.cfg.vocab);
    defer gpa.free(row);
    try fwd.run(s, m, sc, &cache, ids, .{});
    var next: [1]u32 = undefined;
    try cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
    next[0] = @intCast(argmax(row));

    // The recorded list goes through its own in-order queue, so the eager path's queue stays untouched.
    var rq = try rt.cmdlist.Queue.init(&s.ctx);
    defer rq.deinit();
    var step = try fwd.rp.recordStep(s, m, sc, &cache, &next, .{}, gpa);
    defer step.deinit(gpa);

    // The recorded list must reproduce the eager step for the same token bit for bit; otherwise the replay is
    // not a valid measurement of the same work and the gate must not accept its timing.
    const eager_row = try gpa.alloc(u16, sc.cfg.vocab);
    defer gpa.free(eager_row);
    var probe = try xs.Cache.init(gpa, s, sc.cfg, capacity);
    defer probe.deinit();
    try fwd.run(s, m, sc, &probe, ids, .{});
    try fwd.run(s, m, sc, &probe, &next, .{});
    try probe.logits.download(&s.queue, 0, std.mem.sliceAsBytes(eager_row));
    const vals0: fwd.rp.Values = .{ .token = next[0], .position = @intCast(cache.len), .cache_len = @intCast(cache.len) };
    try fwd.rp.replayStep(&step, &rq, vals0);
    cache.len += 1;
    try cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
    const identical = std.mem.eql(u8, std.mem.sliceAsBytes(row), std.mem.sliceAsBytes(eager_row));
    std.debug.print("GATE D replay vs eager logits row: {s}\n", .{if (identical) "bit-identical" else "DIFFERENT"});
    if (!identical) return error.ReplayLogitsMismatch;

    // 3 warm-up steps, then as many timed steps as the capacity allows after the prompt (>= 60 whenever capacity
    // leaves room; the fixture prompt is 14 tokens, so 512 - 14 - 3 = 495 timed steps).
    var i: usize = 1;
    while (i < 63) : (i += 1) {
        try cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
        next[0] = @intCast(argmax(row));
        const t1 = sys.nowNs();
        const v: fwd.rp.Values = .{ .token = next[0], .position = @intCast(cache.len), .cache_len = @intCast(cache.len) };
        try fwd.rp.replayStep(&step, &rq, v);
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
    std.debug.print("config: hidden {d}, layers {d}, heads {d}, kv {d}, intermediate {d}, group {d}\n", .{ config.hidden, config.layers, config.heads, config.kv_heads, config.intermediate, group });
    var w = try xw.load(gpa, &s, dir, config, group);
    defer w.deinit();
    var m: fwd.Model = .{ .weights = &w, .kernels = undefined };
    try m.kernels.load(&s, gpa);
    defer m.kernels.deinit();
    std.debug.print("weights {d} tensors, {d:.1} MiB and kernels loaded\n", .{ w.ledger.len, @as(f64, @floatFromInt(w.total_bytes)) / (1 << 20) });

    // The scratch holds the whole prompt in one prefill segment; decode steps reuse the same buffers with rows = 1.
    const ids = try readNpyU32(refdir, fixtures[0]);
    defer gpa.free(ids);
    var sc = try fwd.Scratch.init(&s, config, ids.len, capacity, 1);
    defer sc.deinit();

    var ok = true;
    var steps: std.ArrayList(u64) = .empty;
    defer steps.deinit(gpa);
    for (fixtures) |name| {
        const r = gateC(&s, &m, &sc, refdir, name, &steps) catch |e| blk: {
            std.debug.print("gate C error: {s}\n", .{@errorName(e)});
            break :blk false;
        };
        ok = ok and r;
    }
    std.debug.print("GATE C greedy: {s}\n", .{if (ok) "PASS" else "FAIL"});

    // The contract's decode measurement: >= 60 timed one-token steps, median reported.
    var dec: std.ArrayList(u64) = .empty;
    defer dec.deinit(gpa);
    decodeSteps(&s, &m, &sc, ids, &dec) catch |e| std.debug.print("decode measurement error: {s}\n", .{@errorName(e)});
    if (dec.items.len > 0) {
        std.mem.sort(u64, dec.items, {}, std.sort.asc(u64));
        var sum: u64 = 0;
        for (dec.items) |x| sum += x;
        std.debug.print("decode step (1 token, capacity {d}): {d} steps, median {d:.1} ms, mean {d:.1} ms, min {d:.1} ms -> {d:.2} tokens/s\n", .{
            capacity,                                              dec.items.len,
            @as(f64, @floatFromInt(dec.items[dec.items.len / 2])) / 1e6, @as(f64, @floatFromInt(sum)) / 1e6 / @as(f64, @floatFromInt(dec.items.len)),
            @as(f64, @floatFromInt(dec.items[0])) / 1e6,           1e9 / (@as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(dec.items.len))),
        });
    }
    return if (ok) 0 else 1;
}
