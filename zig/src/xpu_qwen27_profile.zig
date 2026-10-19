//! tf-xpu-qwen27-profile <model dir> <prompt tokens> <steps> [rows]: where a Qwen 3.8 27B decode step's time goes on a B70.
//! [rows] (default 1): rows per verify round - a speculative window's or a shared round's width; plain stays 1 row.
//! Prefills a synthetic prompt in 128-row chunks (as the server does), then runs <steps> decode steps two ways and
//! prints wall time per step, device time per step, and device time per kernel and per projection shape:
//!   verify : the server's per-token path - a recording one-row round (`runSegments`, record = true, head = .all)
//!            followed by the logits row's download and a host argmax;
//!   plain  : a one-row `run` (record = false), the eager decode the logits tools use.
//! Wall minus device is host + launch overhead. Kernel timestamps are Level Zero events (`Profile`).
const std = @import("std");
const fwd = @import("qwen27_forward");
const rt = fwd.rt;
const sys = fwd.sys;
const cfg = fwd.cfg;
const xw = fwd.xw;
const xs = fwd.xs;

const gpa = std.heap.page_allocator;
const chunk = 128;

fn loadConfig(dir: []const u8) !cfg.Config {
    var cb: [1024]u8 = undefined;
    const cpath = sys.zfmt(&cb, "{s}/config.json", .{dir}) orelse return error.PathTooLong;
    const text = sys.readFileAlloc(gpa, cpath, 1 << 22) orelse return error.NoConfig;
    defer gpa.free(text);
    return cfg.parse(gpa, text, null);
}

fn argmax(row: []const u16) u32 {
    var best: usize = 0;
    var bv: f32 = @bitCast(@as(u32, row[0]) << 16);
    for (row, 0..) |w, i| {
        const v: f32 = @bitCast(@as(u32, w) << 16);
        if (v > bv) {
            bv = v;
            best = i;
        }
    }
    return @intCast(best);
}

const Mode = enum { verify, plain };

fn report(name: []const u8, p: *fwd.Profile, steps: usize, wall_ns: u64) void {
    var dev: f64 = 0;
    for (p.ns) |t| dev += t;
    const st: f64 = @floatFromInt(steps);
    std.debug.print("\n== {s}: wall {d:.2} ms/step, device {d:.2} ms/step ({d:.1} tok/s wall)\n", .{
        name, @as(f64, @floatFromInt(wall_ns)) / 1e6 / st, dev / 1e6 / st, st * 1e9 / @as(f64, @floatFromInt(wall_ns)),
    });
    // kernels by device time, descending
    var order: [fwd.kernel_ids.len]usize = undefined;
    for (&order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, &order, p, struct {
        fn lt(pp: *fwd.Profile, a: usize, b: usize) bool {
            return pp.ns[a] > pp.ns[b];
        }
    }.lt);
    for (order) |i| if (p.count[i] > 0) {
        std.debug.print("  {s:16} {d:8.3} ms/step  {d:6.1} launches/step  {d:7.1} us/launch\n", .{
            @tagName(fwd.kernel_ids[i]), p.ns[i] / 1e6 / st, @as(f64, @floatFromInt(p.count[i])) / st,
            p.ns[i] / 1e3 / @as(f64, @floatFromInt(p.count[i])),
        });
    };
    std.debug.print("  projections by shape (outputs x inputs):\n", .{});
    var it = p.shape_ns.iterator();
    while (it.next()) |e| {
        const n = p.shape_count.get(e.key_ptr.*) orelse 1;
        std.debug.print("    {d:6} x {d:5}: {d:8.3} ms/step  {d:5.1} launches/step  {d:7.1} us/launch\n", .{
            e.key_ptr.* >> 32, e.key_ptr.* & 0xffffffff, e.value_ptr.* / 1e6 / st,
            @as(f64, @floatFromInt(n)) / st, e.value_ptr.* / 1e3 / @as(f64, @floatFromInt(n)),
        });
    }
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    if (argv.len != 4 and argv.len != 5) {
        std.debug.print("usage: tf-xpu-qwen27-profile <model dir> <prompt tokens> <steps> [rows]\n", .{});
        return 2;
    }
    const dir = argv[1];
    const plen = try std.fmt.parseInt(usize, argv[2], 10);
    const steps = try std.fmt.parseInt(usize, argv[3], 10);
    const vrows: usize = if (argv.len == 5) try std.fmt.parseInt(usize, argv[4], 10) else 1;
    const capacity = plen + (vrows + 1) * (steps + 8) + 16;

    var s: rt.session.Session = undefined;
    s.open(.{}) catch |e| {
        std.debug.print("cannot open a card: {s} ({s})\n", .{ @errorName(e), s.diag.message() });
        return 1;
    };
    defer s.close();
    const config = try loadConfig(dir);
    var w = try xw.load(gpa, &s, dir, config);
    defer w.deinit();
    var m: fwd.Model = .{ .weights = &w, .kernels = undefined };
    try m.kernels.load(&s, gpa);
    defer m.kernels.deinit();
    var sc = try fwd.Scratch.initGpa(gpa, &s, config, chunk, capacity, @max(1, vrows));
    defer sc.deinit();
    var cache = try xs.Cache.init(gpa, &s, config, capacity);
    defer cache.deinit();

    // A synthetic prompt of ordinary token ids (the time per step does not depend on which ids they are).
    const prompt = try gpa.alloc(u32, plen);
    defer gpa.free(prompt);
    for (prompt, 0..) |*t, i| t.* = @intCast(1000 + (i * 7919) % 50000);
    const t0 = sys.nowNs();
    var at: usize = 0;
    while (at < plen) : (at += chunk) {
        const end = @min(plen, at + chunk);
        try fwd.run(&s, &m, &sc, &cache, prompt[at..end], .{});
    }
    try s.queue.synchronize();
    std.debug.print("prefill {d} tokens: {d:.1} ms ({d:.1} tok/s)\n", .{ plen, @as(f64, @floatFromInt(sys.nowNs() - t0)) / 1e6,
        @as(f64, @floatFromInt(plen)) * 1e9 / @as(f64, @floatFromInt(sys.nowNs() - t0)) });

    const row = try gpa.alloc(u16, config.vocab);
    defer gpa.free(row);
    var prof = try fwd.Profile.init(&s, gpa, 4096);
    defer prof.deinit(gpa);
    var tok: u32 = 1000;
    inline for (.{ Mode.verify, Mode.plain }) |mode| {
        prof.clear();
        var wall: u64 = 0;
        var i: usize = 0;
        while (i < steps + 3) : (i += 1) {
            const timed = i >= 3; // three warm-up steps
            sc.prof = if (timed) prof else null;
            const ts = sys.nowNs();
            switch (mode) {
                .verify => {
                    var win: [32]u32 = undefined;
                    for (win[0..vrows], 0..) |*t, k| t.* = tok +% @as(u32, @intCast(k));
                    try fwd.runSegments(&s, &m, &sc, &.{.{ .cache = &cache, .rows = vrows }}, win[0..vrows], true, .{ .head = .all });
                    try sc.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
                },
                .plain => {
                    try fwd.run(&s, &m, &sc, &cache, &.{tok}, .{});
                    try cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
                },
            }
            tok = argmax(row);
            if (timed) {
                wall += sys.nowNs() - ts;
                prof = sc.prof.?;
                try prof.collect(gpa);
            }
            sc.prof = null;
        }
        report(@tagName(mode), &prof, steps, wall);
    }
    return 0;
}
