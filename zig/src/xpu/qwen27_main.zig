//! Development runner of the Qwen3.8-27B XPU forward: greedy tokens and timings, to check against a fp32 reference.
const std = @import("std");
const q = @import("qwen27.zig");
const core = @import("core");
const dr_mod = @import("drafter.zig");
const Drafter = dr_mod.Drafter;
const fp64_ref = @import("fp64_ref.zig");
const enc = @import("exl3_encode.zig");

const alloc = std.heap.c_allocator;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len > 1 and std.mem.eql(u8, argv[1], "--enc-probe")) return encProbe();
    if (argv.len > 1 and std.mem.eql(u8, argv[1], "--calib-probe")) return calibProbe(argv);
    if (argv.len < 3) {
        std.debug.print("usage: xpu-qwen27 MODEL_DIR IDS_FILE [STEPS]\n", .{});
        return;
    }
    const steps: usize = if (argv.len > 3) try std.fmt.parseInt(usize, argv[3], 10) else 32;
    const ctx: u32 = if (std.c.getenv("XPU_CTX")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 4096;
    var ids: std.ArrayList(u32) = .empty;
    if (std.mem.startsWith(u8, argv[2], "--bench=")) {
        const n = try std.fmt.parseInt(usize, argv[2][8..], 10);
        for (0..n) |i| try ids.append(alloc, @intCast((i * 7919 + 13) % 200000));
    } else {
        const path = try std.fmt.allocPrintSentinel(alloc, "{s}", .{argv[2]}, 0);
        const fd = open(path.ptr, 0);
        if (fd < 0) return error.FileNotFound;
        const size: usize = @intCast(lseek(fd, 0, 2));
        const text: [*]const u8 = @ptrCast(mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed);
        var it = std.mem.tokenizeAny(u8, text[0..size], " \n,[]");
        while (it.next()) |w| try ids.append(alloc, try std.fmt.parseInt(u32, w, 10));
    }
    q.verbose = true;
    const t_load = q.now();
    const r = try q.Runner.init(argv[1], ctx);
    std.debug.print("loaded in {d:.1} s\n", .{q.now() - t_load});
    if (std.c.getenv("XPU_FIDELITY") != null) return fp64_ref.fidelity(r, ids.items);
    if (std.c.getenv("XPU_FP64_REF") != null) fp64_ref.arm(r);
    const t0 = q.now();
    try r.prompt(ids.items, true);
    const t1 = q.now();
    fp64_ref.disarm();
    q.profReport("prompt");
    var out: std.ArrayList(u32) = .empty;
    try out.append(alloc, r.last);
    for (1..steps) |_| {
        try r.step(r.last);
        try out.append(alloc, r.last);
    }
    const t2 = q.now();
    q.profReport("decode");
    if (std.c.getenv("XPU_OVERLAP_PROBE") != null) try overlapProbe(r);
    if (std.c.getenv("XPU_ROWS_PROBE") != null) try rowsProbe(r);
    if (std.c.getenv("XPU_EXL3_PROBE") != null) {
        try exl3Probe(r);
        try exl3PromptProbe(r);
    }
    if (std.c.getenv("XPU_MULTI_TEST")) |v| try multiTest(r, ids.items, try std.fmt.parseInt(u32, std.mem.span(v), 10));
    if (std.c.getenv("XPU_GEMM_ZEBIN")) |path| try gemmLoop(r, std.mem.span(path));
    if (std.c.getenv("XPU_A8_ZEBIN")) |path| try a8Loop(r, std.mem.span(path));
    if (std.c.getenv("XPU_VERIFY_TEST") != null) try verifyTest(r, ids.items, out.items);
    if (std.c.getenv("XPU_DRAFTER")) |dir| try drafted(r, std.mem.span(dir), ids.items, out.items);
    std.debug.print("prompt: {d} tokens in {d:.3} s ({d:.0} tok/s)\n", .{ ids.items.len, t1 - t0, @as(f64, @floatFromInt(ids.items.len)) / (t1 - t0) });
    std.debug.print("decode: {d} tokens in {d:.3} s ({d:.1} tok/s)\n", .{ steps - 1, t2 - t1, @as(f64, @floatFromInt(steps - 1)) / (t2 - t1) });
    std.debug.print("tokens:", .{});
    for (out.items) |t| std.debug.print(" {d}", .{t});
    std.debug.print("\n", .{});
}

/// The drafted-equals-serial contract with stand-in drafts: blocks of 8 rows, perfect and wrong drafts equal serial.
fn verifyTest(r: *q.Runner, prompt: []const u32, serial: []const u32) !void {
    for ([_]bool{ true, false }) |perfect| {
        try r.reset();
        try r.prompt(prompt, true);
        var got: std.ArrayList(u32) = .empty;
        try got.append(alloc, r.last);
        var rounds: usize = 0;
        while (got.items.len < serial.len) {
            var block: [8]u32 = undefined;
            block[0] = got.items[got.items.len - 1];
            for (1..8) |j| {
                const at = got.items.len - 1 + j;
                block[j] = if (perfect and at < serial.len) serial[at] else 0;
            }
            const greedy = try r.verify(&block);
            var n: u32 = 0; // accepted drafts
            while (n + 1 < 8 and greedy[n] == block[n + 1]) n += 1;
            try r.commit(n + 1);
            for (greedy[0 .. n + 1]) |t| try got.append(alloc, t);
            rounds += 1;
        }
        const same = std.mem.eql(u32, got.items[0..serial.len], serial);
        std.debug.print("verify test ({s} drafts): {d} rounds, equal to serial: {}\n", .{ if (perfect) "perfect" else "wrong", rounds, same });
        if (!same) {
            std.debug.print("  got:", .{});
            for (got.items[0..serial.len]) |t| std.debug.print(" {d}", .{t});
            std.debug.print("\n", .{});
        }
    }
}

fn onTaps(ctx: *anyopaque, taps: *anyopaque, n: u32) !void {
    const dr: *Drafter = @ptrCast(@alignCast(ctx));
    try dr.addTaps(taps, n);
}

/// Greedy decode with DFlash2 drafts (blocks of XPU_BLOCK rows, 8 by default): must equal the serial reply.
fn drafted(r: *q.Runner, dir: []const u8, prompt: []const u32, serial: []const u32) !void {
    const t_load = q.now();
    const dr = try Drafter.init(r, dir);
    std.debug.print("drafter loaded in {d:.1} s\n", .{q.now() - t_load});
    const L: u32 = if (std.c.getenv("XPU_BLOCK")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 8;
    try r.reset();
    dr.reset();
    r.taps_on = true;
    r.on_taps = .{ .ctx = dr, .f = onTaps };
    try r.prompt(prompt, true);
    var got: std.ArrayList(u32) = .empty;
    try got.append(alloc, r.last);
    var rounds: usize = 0;
    q.profReport("drafter prompt (discarded)");
    var t_draft: f64 = 0;
    var t_verify: f64 = 0;
    const t0 = q.now();
    const nodes: u32 = if (std.c.getenv("XPU_TREE")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 0;
    var accepted: usize = 0;
    var drafted_n: usize = 0;
    while (got.items.len < serial.len) {
        if (nodes > 0) { // tree rounds: drafting and verification timed together
            const a = q.now();
            var out: [16]u32 = undefined;
            const round = try dr_mod.treeRound(dr, r.last, L - 1, nodes, null, &out);
            t_verify += q.now() - a;
            for (round.tokens[1..]) |t| try got.append(alloc, t);
            try got.append(alloc, round.terminal);
            accepted += round.accepted;
            drafted_n += round.drafted;
            rounds += 1;
            continue;
        }
        var block: [16]u32 = undefined;
        block[0] = r.last;
        const a = q.now();
        try dr.propose(r.last, L - 1, block[1..L]);
        const b = q.now();
        const greedy = try r.verify(block[0..L]);
        var n: u32 = 0;
        while (n + 1 < L and greedy[n] == block[n + 1]) n += 1;
        try r.commit(n + 1);
        try dr.addTaps(r.acts.taps, n + 1);
        t_verify += q.now() - b;
        t_draft += b - a;
        for (greedy[0 .. n + 1]) |t| try got.append(alloc, t);
        rounds += 1;
    }
    const dt = q.now() - t0;
    q.profReport("drafted rounds");
    if (nodes > 0) std.debug.print("tree: {d} nodes a round, {d:.2} accepted of {d:.1} drafted\n", .{ nodes, @as(f64, @floatFromInt(accepted)) / @as(f64, @floatFromInt(rounds)), @as(f64, @floatFromInt(drafted_n)) / @as(f64, @floatFromInt(rounds)) });
    const len = @min(got.items.len, serial.len);
    const same = std.mem.eql(u32, got.items[0..len], serial[0..len]);
    std.debug.print("drafted: {d} tokens in {d} rounds ({d:.2} a round), {d:.1} tok/s, draft {d:.1} ms + verify {d:.1} ms a round, equal to serial: {}\n", .{ got.items.len - 1, rounds, @as(f64, @floatFromInt(got.items.len - 1)) / @as(f64, @floatFromInt(rounds)), @as(f64, @floatFromInt(got.items.len - 1)) / dt, t_draft * 1e3 / @as(f64, @floatFromInt(rounds)), t_verify * 1e3 / @as(f64, @floatFromInt(rounds)), same });
}

/// Can a prompt GEMM and the chunked GDN of another chunk share the card? Times each alone, chained, and unordered.
fn overlapProbe(r: *q.Runner) !void {
    const d = &r.d;
    const a = &r.acts;
    const T: u32 = q.CHUNK;
    const ordered = d.list;
    const free = try d.g.immediateUnordered(0);
    var gdn_layer: usize = 0;
    while (r.model.layers[gdn_layer].attn) gdn_layer += 1;
    const lg = &r.model.layers[gdn_layer];
    const l0 = &r.model.layers[0];
    const nc: i32 = @intCast(T / 64);
    const ncu: u32 = T / 64;
    const hk: i32 = 16;
    const hv: i32 = 48;
    const runA = struct {
        fn f(dd: *q.Dev, aa: *q.Acts, l: anytype, t: u32) !void {
            try q.gemm(dd, aa.pscr, aa.h, aa.ahi, aa.alo, l.gu, aa.p, t);
        }
    }.f;
    const runB = struct {
        fn f(dd: *q.Dev, aa: *q.Acts, l: anytype, ncv: i32, ncuv: u32, hkv: i32, hvv: i32) !void {
            try dd.launch(dd.k.gdn_prep, .{ .x = ncuv, .y = 48 }, .{ aa.k, aa.v, aa.g, aa.beta, aa.la, aa.wb, aa.u0, hkv, hvv, ncv, @as(i32, 0) });
            try dd.launch(dd.k.gdn_state, .{ .x = 48 }, .{ aa.k, aa.wb, aa.u0, aa.la, l.rec.?, l.rec.?, aa.hs, aa.vn, hkv, hvv, ncv });
            try dd.launch(dd.k.gdn_out, .{ .x = ncuv, .y = 48 }, .{ aa.q, aa.k, aa.la, aa.hs, aa.vn, aa.y, hkv, hvv, ncv });
        }
    }.f;
    var best = [5]f64{ 1e9, 1e9, 1e9, 1e9, 1e9 };
    for (0..6) |_| {
        for (0..5) |mode| {
            d.list = if (mode >= 3) free else ordered;
            try d.sync();
            const t0 = q.now();
            if (mode == 4) try runB(d, a, lg, nc, ncu, hk, hv);
            if (mode == 0 or mode >= 2) try runA(d, a, l0, T);
            if (mode >= 1 and mode <= 3) try runB(d, a, lg, nc, ncu, hk, hv);
            try d.sync();
            best[mode] = @min(best[mode], q.now() - t0);
        }
    }
    d.list = ordered;
    std.debug.print("overlap probe ({d} rows): GEMM gate|up {d:.2} ms, GDN chunk {d:.2} ms, chained {d:.2} ms, unordered A,B {d:.2} ms, unordered B,A {d:.2} ms\n", .{ T, best[0] * 1e3, best[1] * 1e3, best[2] * 1e3, best[3] * 1e3, best[4] * 1e3 });
}

/// Power study: the gate|up prompt GEMM of a zebin given by path, in a loop of CHUNK-row calls; time and TFLOPS.
fn gemmLoop(r: *q.Runner, path: []const u8) !void {
    const d = &r.d;
    const a = &r.acts;
    const fd = open(try std.fmt.allocPrintSentinel(alloc, "{s}", .{path}, 0), 0);
    if (fd < 0) return error.FileNotFound;
    const size: usize = @intCast(lseek(fd, 0, 2));
    const bin: [*]const u8 = @ptrCast(mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed);
    const copy = try alloc.alignedAlloc(u8, .@"8", size);
    @memcpy(copy, bin[0..size]);
    const k = try d.kernel(try d.g.module(copy), "w4g_dpb", 256);
    const w = r.model.layers[0].gu;
    const T = q.CHUNK;
    const tiles = (T + 63) / 64;
    const args = .{ a.h, w.w, w.s, w.bg, a.p, @as(i32, @intCast(T)), @as(i32, @intCast(w.n)), @as(i32, @intCast(w.k)), a.ahi, a.alo, @as(i32, @intCast((T + 7) / 8 * 8)), @as(i32, @intCast(w.k / w.gs)) };
    for (0..20) |_| try d.launch(k, .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 }, args);
    try d.sync();
    const t0 = q.now();
    for (0..4000) |_| try d.launch(k, .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 }, args);
    try d.sync();
    const dt = (q.now() - t0) / 4000;
    const flops = 2.0 * @as(f64, @floatFromInt(T)) * @as(f64, @floatFromInt(w.n)) * @as(f64, @floatFromInt(w.k));
    std.debug.print("gemm loop {s}: {d:.3} ms a call, {d:.1} TFLOPS\n", .{ path, dt * 1e3, flops / dt / 1e12 });
}

/// The W4A8 gate|up GEMM of a zebin given by path in a loop of CHUNK-row calls: time and TOPS per call.
fn a8Loop(r: *q.Runner, path: []const u8) !void {
    const d = &r.d;
    const a = &r.acts;
    const fd = open(try std.fmt.allocPrintSentinel(alloc, "{s}", .{path}, 0), 0);
    if (fd < 0) return error.FileNotFound;
    const size: usize = @intCast(lseek(fd, 0, 2));
    const bin: [*]const u8 = @ptrCast(mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed);
    const copy = try alloc.alignedAlloc(u8, .@"8", size);
    @memcpy(copy, bin[0..size]);
    const k = try d.kernel(try d.g.module(copy), "w4a8", 256);
    // XPU_A8_PROJ: gu (default), in (gdn.in), out (gdn.out), down
    const which = if (std.c.getenv("XPU_A8_PROJ")) |v| std.mem.span(v) else "gu";
    const l0 = &r.model.layers[0];
    const w = if (std.mem.eql(u8, which, "in")) l0.mixer_in else if (std.mem.eql(u8, which, "out")) l0.mixer_out else if (std.mem.eql(u8, which, "down")) l0.down else l0.gu;
    const T = q.CHUNK;
    const tiles = (T + 63) / 64;
    try d.launch(d.k.quant_a8, .{ .x = T }, .{ a.h, a.a8, a.sa, a.ahi, a.alo, @as(i32, @intCast(w.k)), @as(i32, @intCast(T + 32)), @as(i32, 0) });
    const args = .{ a.a8, w.w, w.s, w.bg, a.p, @as(i32, @intCast(T)), @as(i32, @intCast(w.n)), @as(i32, @intCast(w.k)), a.ahi, a.alo, @as(i32, @intCast(T + 32)), @as(i32, @intCast(w.k / 128)), a.sa };
    const groups: q.ze.GroupCount = .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 };
    for (0..20) |_| try d.launch(k, groups, args);
    try d.sync();
    const t0 = q.now();
    for (0..2000) |_| try d.launch(k, groups, args);
    try d.sync();
    const dt = (q.now() - t0) / 2000;
    const ops = 2.0 * @as(f64, @floatFromInt(T)) * @as(f64, @floatFromInt(w.n)) * @as(f64, @floatFromInt(w.k));
    std.debug.print("a8 loop {s} {s} (N {d}, K {d}): {d:.3} ms a call, {d:.1} TOPS\n", .{ path, which, w.n, w.k, dt * 1e3, ops / dt / 1e12 });
}

/// Multi-stream estimate: the cost of one forward of T rows (1 to 16) as a verification block; the state is left as is.
fn rowsProbe(r: *q.Runner) !void {
    var ids: [16]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(1000 + i);
    for ([_]u32{ 1, 2, 4, 8, 12, 16 }) |T| {
        var best: f64 = 1e9;
        for (0..12) |_| {
            const t0 = q.now();
            _ = try r.verify(ids[0..T]);
            best = @min(best, q.now() - t0);
            r.block_rows = 0;
        }
        std.debug.print("rows probe: {d:2} rows {d:6.2} ms ({d:5.1} rows/s at one row a stream)\n", .{ T, best * 1e3, @as(f64, @floatFromInt(T)) / best });
    }
}

/// Multi-stream contract: `n` slots at different positions decode serially, then in shared stepMulti rounds; equal.
fn multiTest(r: *q.Runner, prompt: []const u32, n: u32) !void {
    const steps = 24;
    var serial: [q.MAX_SLOTS][steps]u32 = undefined;
    for (0..n) |k| {
        try r.select(@intCast(k));
        try r.reset();
        try r.prompt(prompt[k % 8 ..], true);
        for (0..steps) |j| {
            serial[k][j] = r.last;
            try r.step(r.last);
        }
    }
    var shared: [q.MAX_SLOTS][steps]u32 = undefined;
    var slots: [q.MAX_SLOTS]u32 = undefined;
    var toks: [q.MAX_SLOTS]u32 = undefined;
    for (0..n) |k| {
        try r.select(@intCast(k));
        try r.reset();
        try r.prompt(prompt[k % 8 ..], true);
        slots[k] = @intCast(k);
    }
    for (0..steps) |j| {
        for (0..n) |k| {
            toks[k] = r.seqs[k].last;
            if (k == r.slot) toks[k] = r.last;
            shared[k][j] = toks[k];
        }
        try r.stepMulti(slots[0..n], toks[0..n]);
    }
    var equal = true;
    for (0..n) |k| equal = equal and std.mem.eql(u32, &serial[k], &shared[k]);
    std.debug.print("multi test: {d} slots, {d} tokens each, shared rounds equal to serial: {}\n", .{ n, steps, equal });
    if (!equal) for (0..n) |k| {
        std.debug.print("  slot {d} serial:", .{k});
        for (serial[k]) |t| std.debug.print(" {d}", .{t});
        std.debug.print("\n  slot {d} shared:", .{k});
        for (shared[k]) |t| std.debug.print(" {d}", .{t});
        std.debug.print("\n", .{});
    };
    // shared windows of 4 rows: perfect drafts (the serial reply) for even slots, wrong ones for odd slots
    var got: [q.MAX_SLOTS][steps + 8]u32 = undefined;
    var have: [q.MAX_SLOTS]usize = @splat(0);
    for (0..n) |k| {
        try r.select(@intCast(k));
        try r.reset();
        try r.prompt(prompt[k % 8 ..], true);
    }
    var rounds: usize = 0;
    while (true) {
        var wins: [16]q.Runner.Win = undefined;
        var ids: [16][4]u32 = undefined;
        var nw: usize = 0;
        var rows: usize = 0;
        for (0..n) |k| {
            if (have[k] >= steps or rows + 4 > 16) continue;
            const pending = if (k == r.slot) r.last else r.seqs[k].last;
            ids[k][0] = pending;
            for (1..4) |j| ids[k][j] = if (k % 2 == 0 and have[k] + j < steps) serial[k][have[k] + j] else 0;
            wins[nw] = .{ .slot = @intCast(k), .ids = &ids[k] };
            nw += 1;
            rows += 4;
        }
        if (nw == 0) break;
        try r.verifyMulti(wins[0..nw]);
        var keep: [16]u32 = undefined;
        var row: usize = 0;
        for (wins[0..nw], 0..) |w, i| {
            var a: u32 = 0; // accepted drafts
            while (a + 1 < 4 and r.block[row + a] == w.ids[a + 1]) a += 1;
            keep[i] = a + 1;
            const k = w.slot;
            for (w.ids[0 .. a + 1]) |t| {
                got[k][have[k]] = t;
                have[k] += 1;
            }
            row += 4;
        }
        try r.keepMulti(keep[0..nw]);
        rounds += 1;
    }
    var equal4 = true;
    for (0..n) |k| equal4 = equal4 and std.mem.eql(u32, got[k][0..steps], &serial[k]);
    std.debug.print("multi test: windows of 4 rows (perfect / wrong drafts), {d} rounds, equal to serial: {}\n", .{ rounds, equal4 });
    if (!equal4) for (0..n) |k| {
        const at = std.mem.indexOfDiff(u32, got[k][0..steps], &serial[k]);
        if (at) |x| std.debug.print("  slot {d} differs at token {d}\n", .{ k, x });
    };
    // shared tree windows of 7 rows (parents -1 0 0 2 1 3 3); the walk keeps a root path and keepPaths replays it
    @memset(&have, 0);
    for (0..n) |k| {
        try r.select(@intCast(k));
        try r.reset();
        try r.prompt(prompt[k % 8 ..], true);
    }
    const par = [7]i32{ -1, 0, 0, 2, 1, 3, 3 };
    var trounds: usize = 0;
    while (true) {
        var wins: [16]q.Runner.Win = undefined;
        var tids: [16][7]u32 = undefined;
        var nw: usize = 0;
        var rows: usize = 0;
        for (0..n) |k| {
            if (have[k] >= steps or rows + 7 > 16) continue;
            const x = &tids[k];
            x.* = .{ if (k == r.slot) r.last else r.seqs[k].last, 7, 0, 0, 11, 0, 13 };
            const at = have[k];
            const sv = struct {
                fn f(row: []const u32, i: usize) u32 {
                    return if (i < row.len) row[i] else 0;
                }
            }.f;
            if (k % 2 == 0) {
                x[2] = sv(&serial[k], at + 1);
                x[3] = sv(&serial[k], at + 2);
                x[5] = sv(&serial[k], at + 3);
            } else x[1] = sv(&serial[k], at + 1);
            wins[nw] = .{ .slot = @intCast(k), .ids = x, .parents = &par };
            nw += 1;
            rows += 7;
        }
        if (nw == 0) break;
        try r.verifyMulti(wins[0..nw]);
        var pbuf: [16][7]u32 = undefined;
        var paths: [16][]const u32 = undefined;
        var row0: usize = 0;
        for (wins[0..nw], 0..) |w, i| {
            var len: usize = 1;
            pbuf[i][0] = 0;
            walk: while (true) {
                const cur = pbuf[i][len - 1];
                for (1..7) |c| if (par[c] == @as(i32, @intCast(cur)) and w.ids[c] == r.block[row0 + cur]) {
                    pbuf[i][len] = @intCast(c);
                    len += 1;
                    continue :walk;
                };
                break;
            }
            paths[i] = pbuf[i][0..len];
            const k = w.slot;
            for (paths[i]) |c| if (have[k] < steps + 8) {
                got[k][have[k]] = w.ids[c];
                have[k] += 1;
            };
            row0 += 7;
        }
        try r.keepPaths(paths[0..nw]);
        trounds += 1;
    }
    var equalt = true;
    for (0..n) |k| equalt = equalt and std.mem.eql(u32, got[k][0..steps], &serial[k]);
    std.debug.print("multi test: tree windows of 7 rows, {d} rounds, equal to serial: {}\n", .{ trounds, equalt });
    if (!equalt) for (0..n) |k| {
        const at = std.mem.indexOfDiff(u32, got[k][0..steps], &serial[k]);
        if (at) |x| std.debug.print("  slot {d} differs at token {d}\n", .{ k, x });
    };
    var m: u32 = 1;
    while (m <= n) : (m *= 2) {
        var best: f64 = 1e9;
        for (0..8) |_| {
            for (0..m) |k| toks[k] = r.seqs[k].last;
            toks[r.slot] = r.last;
            const t0 = q.now();
            try r.stepMulti(slots[0..m], toks[0..m]);
            best = @min(best, q.now() - t0);
        }
        std.debug.print("multi rounds: {d:2} slots {d:6.2} ms a round, {d:6.1} tok/s together\n", .{ m, best * 1e3, @as(f64, @floatFromInt(m)) / best });
    }
}

/// EXL3 speed probe (no weights, random trellis bits): the mul1 decode GEMV at 4, 3 and 2 bits against our int4 GEMM.
fn exl3Probe(r: *q.Runner) !void {
    const d = &r.d;
    const a = &r.acts;
    const l0 = &r.model.layers[0];
    for ([_]q.Proj{ l0.gu, l0.down }) |w| {
        const K = w.k;
        const N = if (w.ilv) w.n else w.n;
        for ([_]u32{ 1, 8 }) |T| {
            var best: f64 = 1e9;
            for (0..200) |it| {
                const t0 = q.now();
                try q.gemm(d, a.pscr, a.h, a.ahi, a.alo, w, a.p, T);
                try d.sync();
                if (it > 10) best = @min(best, q.now() - t0);
            }
            const bytes4 = @as(f64, @floatFromInt(K)) * @as(f64, @floatFromInt(N)) / 2.0;
            std.debug.print("int4 (ours)  K {d:5} N {d:5} rows {d:2}: {d:6.3} ms, {d:5.0} GB/s\n", .{ K, N, T, best * 1e3, bytes4 / best / 1e9 });
            // bits=1 is excluded here: exl3_gemv lane-prefetches tw[lane] (nw >= 16 needed);
            // the indexed-read kernels (decode2/_t1, the _h clones, dequant128_c) are safe
            // and bits=1 is covered by exl3PromptProbe against the host oracle.
            for ([_]u32{ 4, 3, 2 }) |bits| {
                const bytes: usize = @as(usize, K / 16) * (N / 16) * 16 * bits * 2;
                const tr = try d.alloc_(bytes);
                const host = try alloc.alloc(u8, bytes);
                defer alloc.free(host);
                var prng = std.Random.DefaultPrng.init(bits);
                prng.random().bytes(host);
                try d.copyIn(tr, host);
                const ks: u32 = if (N / 16 >= 2048) 1 else @max(1, 2048 / (N / 16));
                var kslices = ks;
                while ((K / 16) % kslices != 0) kslices -= 1;
                var b2: f64 = 1e9;
                var b3: f64 = 1e9;
                for (0..100) |it| {
                    const t0 = q.now();
                    try d.launch(d.k.exl3_gemv, .{ .x = N / 16, .y = kslices }, .{ tr, a.h, a.pscr, @as(i32, @intCast(K)), @as(i32, @intCast(N)), @as(i32, @intCast(T)), @as(i32, @intCast(bits)), @as(i32, @intCast(kslices)) });
                    try d.sync();
                    if (it > 5) b2 = @min(b2, q.now() - t0);
                }
                for (0..100) |it| {
                    const t0 = q.now();
                    if (T == 1)
                        try d.launch(d.k.exl3_decode2_t1, .{ .x = N / 16, .y = kslices }, .{ tr, a.h, a.pscr, @as(i32, @intCast(K)), @as(i32, @intCast(N)), @as(i32, @intCast(T)), @as(i32, @intCast(bits)), @as(i32, @intCast(kslices)) })
                    else
                        try d.launch(d.k.exl3_decode2, .{ .x = N / 16, .y = kslices }, .{ tr, a.h, a.pscr, @as(i32, @intCast(K)), @as(i32, @intCast(N)), @as(i32, @intCast(T)), @as(i32, @intCast(bits)), @as(i32, @intCast(kslices)) });
                    try d.sync();
                    if (it > 5) b3 = @min(b3, q.now() - t0);
                }
                // seed vs v2 outputs: the same states and decode math, fp32 summation order differs.
                // The seed runs twice: its xr[t] is runtime-indexed and shuffled, so a miscompile
                // would show as seed != seed. The general kernel runs at every T; t1 only at T == 1.
                const yn: usize = kslices * T * N;
                const snap = try alloc.alloc(f32, yn);
                defer alloc.free(snap);
                const snap2 = try alloc.alloc(f32, yn);
                defer alloc.free(snap2);
                const hp = try d.g.hostAlloc(yn * 4);
                defer d.g.free(hp);
                const v2k = if (T == 1) d.k.exl3_decode2_t1 else d.k.exl3_decode2;
                const args = .{ tr, a.h, a.pscr, @as(i32, @intCast(K)), @as(i32, @intCast(N)), @as(i32, @intCast(T)), @as(i32, @intCast(bits)), @as(i32, @intCast(kslices)) };
                const groups: q.ze.GroupCount = .{ .x = N / 16, .y = kslices };
                try d.launch(d.k.exl3_gemv, .{ .x = N / 16, .y = kslices }, .{ tr, a.h, a.pscr, @as(i32, @intCast(K)), @as(i32, @intCast(N)), @as(i32, @intCast(T)), @as(i32, @intCast(bits)), @as(i32, @intCast(kslices)) });
                try d.sync();
                try q.copy(d, hp, a.pscr, yn * 4);
                try d.sync(); // the copy list is async: the host read must wait for it
                @memcpy(snap, @as([*]const f32, @ptrCast(@alignCast(hp)))[0..yn]);
                try d.launch(d.k.exl3_gemv, .{ .x = N / 16, .y = kslices }, .{ tr, a.h, a.pscr, @as(i32, @intCast(K)), @as(i32, @intCast(N)), @as(i32, @intCast(T)), @as(i32, @intCast(bits)), @as(i32, @intCast(kslices)) });
                try d.sync();
                try q.copy(d, hp, a.pscr, yn * 4);
                try d.sync(); // as above
                var seed_deterministic = true;
                for (snap, @as([*]const f32, @ptrCast(@alignCast(hp)))[0..yn]) |x, y| if (x != y) {
                    seed_deterministic = false;
                    break;
                };
                try d.launch(v2k, groups, args);
                try d.sync();
                try q.copy(d, hp, a.pscr, yn * 4);
                try d.sync(); // as above
                @memcpy(snap2, @as([*]const f32, @ptrCast(@alignCast(hp)))[0..yn]);
                var worst: f32 = 0;
                var mism: usize = 0;
                for (snap, snap2) |x, y| {
                    const dd = @abs(x - y);
                    if (dd > worst) worst = dd;
                    if (dd != 0) mism += 1;
                }
                try d.launch(d.k.exl3_decode2, groups, args);
                try d.sync();
                try q.copy(d, hp, a.pscr, yn * 4);
                try d.sync(); // as above
                var gw: f32 = 0;
                var gm: usize = 0;
                for (snap, @as([*]const f32, @ptrCast(@alignCast(hp)))[0..yn]) |x, y| {
                    const dd = @abs(x - y);
                    if (dd > gw) gw = dd;
                    if (dd != 0) gm += 1;
                }
                std.debug.print("seed==seed: {} | v2({s}) vs seed: max |delta| {d:.4}, {d}/{d} differ | general vs seed: max {d:.4}, {d} differ\n", .{ seed_deterministic, if (T == 1) "t1" else "gen", worst, mism, yn, gw, gm });
                if (T == 1) {
                    // host reference from the same random trellis: tileStates + codebookValue +
                    // tilePlacement, fp32-summed — anchors the card kernels without the seed.
                    const tiles_n: usize = N / 16;
                    const words: usize = @as(usize, 2 * bits) * 8;
                    const per: usize = K / 16 / kslices;
                    const xp = try d.g.hostAlloc(K * 2);
                    defer d.g.free(xp);
                    try q.copy(d, xp, a.h, K * 2);
                    try d.sync();
                    const xh = @as([*]const u16, @ptrCast(@alignCast(xp)))[0..K];
                    var placement: [256]core.exl3_format.Placement = undefined;
                    core.exl3_format.tilePlacement(&placement);
                    var tile: [128]i16 = undefined;
                    var states: [256]u32 = undefined;
                    const w16 = core.exl3_format.Width{ .halves = @intCast(2 * bits) };
                    const yref = try alloc.alloc(f32, kslices * N);
                    defer alloc.free(yref);
                    const yrefr = try alloc.alloc(f32, kslices * N);
                    defer alloc.free(yrefr);
                    @memset(yref, 0);
                    @memset(yrefr, 0);
                    const c1: f32 = @floatCast(@as(f16, @bitCast(core.exl3_format.Mul1ScaleBits)));
                    const c0: f32 = @floatCast(@as(f16, @bitCast(core.exl3_format.Mul1BiasBits)));
                    var tk: usize = 0;
                    while (tk < K / 16) : (tk += 1) {
                        var tn: usize = 0;
                        while (tn < tiles_n) : (tn += 1) {
                            const off2 = (tk * tiles_n + tn) * words * 2;
                            @memcpy(std.mem.sliceAsBytes(tile[0..words]), host[off2 .. off2 + words * 2]);
                            core.exl3_format.tileStates(w16, tile[0..words], &states);
                            for (0..256) |p| {
                                const st: u16 = @intCast(states[p]);
                                const x: u32 = st *% core.exl3_format.Mul1Mul;
                                const s: u32 = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255);
                                const wgt: f32 = (1024.0 + @as(f32, @floatFromInt(s))) * c1 + c0; // the kernel's unrounded fp32 weight
                                const xv: f32 = @as(f32, @as(f16, @bitCast(xh[tk * 16 + placement[p].row])));
                                yref[(tk / per) * N + tn * 16 + placement[p].col] += xv * wgt;
                                yrefr[(tk / per) * N + tn * 16 + placement[p].col] += xv * @as(f32, @floatCast(core.exl3_format.codebookValue(.mul1, st)));
                            }
                        }
                    }
                    var hworst: f32 = 0;
                    var hrel: f32 = 0;
                    for (yref, snap2) |x, y| {
                        const dd = @abs(x - y);
                        if (dd > hworst) hworst = dd;
                        hrel = @max(hrel, dd / @max(1.0, @abs(x)));
                    }
                    var rworst: f32 = 0;
                    var rrel: f32 = 0;
                    for (yrefr, snap2) |x, y| {
                        const dd = @abs(x - y);
                        if (dd > rworst) rworst = dd;
                        rrel = @max(rrel, dd / @max(1.0, @abs(x)));
                    }
                    std.debug.print("host(unrounded) vs v2: max {d:.6}, rel {d:.7} | host(fp16 w) vs v2: max {d:.6}, rel {d:.7}\n", .{ hworst, hrel, rworst, rrel });
                }
                std.debug.print("exl3 {d} bits  K {d:5} N {d:5} rows {d:2}: {d:6.3} ms, {d:5.0} GB/s ({d} K slices) | v2: {d:6.3} ms, {d:5.0} GB/s\n", .{ bits, K, N, T, b2 * 1e3, @as(f64, @floatFromInt(bytes)) / b2 / 1e9, kslices, b3 * 1e3, @as(f64, @floatFromInt(bytes)) / b3 / 1e9 });
                d.g.free(tr);
            }
        }
    }
}

/// Dequant check: device fp16 scratch vs the fp64 oracle for the row block
/// [row0, row0+rows); host_tr is the raw trellis bytes, wref the device copy (NT-pitched).
/// Reports the max abs error and the ref RMS so the caller can apply a relative bound.
fn exl3OracleCheck(host_tr: []const u8, hsuh: []const f16, hsvh: []const f16, wref: []const u16,
                   K: usize, N: usize, NT: usize, bits: u32, row0: usize, rows: usize,
                   worst: *f32, rel: *f32, rms: *f32) !void {
    _ = K; // the oracle slices carry their own shapes; K pins nothing here
    const words: usize = @as(usize, 16 * bits);
    const out64 = try alloc.alloc(f64, rows * N);
    defer alloc.free(out64);
    const scratch64 = try alloc.alloc(f64, 2 * rows * N);
    defer alloc.free(scratch64);
    const tk0 = row0 / 16;
    const trellis = @as([*]const i16, @ptrCast(@alignCast(host_tr.ptr + tk0 * (N / 16) * words * 2)))[0 .. (rows / 16) * (N / 16) * words];
    core.exl3_format.dequantize(.{ .halves = @intCast(2 * bits) }, .mul1, trellis, rows / 16, N / 16, hsuh[row0..], hsvh, out64, scratch64);
    var sumsq: f64 = 0;
    for (0..rows) |k| {
        for (0..N) |n| {
            const ref: f32 = @floatCast(out64[k * N + n]);
            const dv = @abs(ref - @as(f32, @as(f16, @bitCast(wref[(row0 + k) * NT + n]))));
            if (dv > worst.*) worst.* = dv;
            rel.* = @max(rel.*, dv / @max(@abs(ref), 1e-6));
            sumsq += @as(f64, ref) * ref;
        }
    }
    rms.* = @floatCast(@sqrt(sumsq / @as(f64, @floatFromInt(rows * N))));
}

/// GEMM check: the device fp16 tile at rows [m0, m0+128) x cols [n0, n0+128) vs a host
/// fp32 matmul over the device-dequantized weights; ha is the [T][K] activation source,
/// wref the full [K][NT] scratch, and cref the copied row block STARTING at crow0.
fn exl3GemmTileCheck(ha: []const f16, wref: []const u16, cref: []const u16, K: usize, NT: usize,
                     m0: usize, n0: usize, crow0: usize, worst: *f32, rel: *f32, rms: *f32) void {
    var sumsq: f64 = 0;
    for (0..128) |m| {
        for (0..128) |n| {
            var s: f32 = 0;
            for (0..K) |k| s += @as(f32, ha[(m0 + m) * K + k]) * @as(f32, @as(f16, @bitCast(wref[k * NT + n0 + n])));
            const dv = @abs(s - @as(f32, @as(f16, @bitCast(cref[(m0 + m - crow0) * NT + n0 + n]))));
            if (dv > worst.*) worst.* = dv;
            rel.* = @max(rel.*, dv / @max(@abs(s), 1e-6));
            sumsq += @as(f64, s) * s;
        }
    }
    rms.* = @floatCast(@sqrt(sumsq / @as(f64, 128 * 128)));
}

/// EXL3 prompt-path probe (no weights, random trellis/scales/activations): the [128,128]
/// block dequant plus the fp16 DPAS GEMM at a full chunk of rows, against the host oracle
/// (exl3_format.dequantize) and a host fp32 matmul over the device-dequantized weights.
/// The full fp64 oracle runs for `down` (17408x5120; gu's would be ~4.3 GB) and on the
/// first 128 k-rows for every shape; the GEMM is checked on the first 128x128 output tile
/// of each shape. Timings compare against the seed int4 prompt GEMM on the same shapes.
fn exl3PromptProbe(r: *q.Runner) !void {
    const d = &r.d;
    const a = &r.acts;
    const l0 = &r.model.layers[0];
    const T: u32 = q.CHUNK;
    for ([_]q.Proj{ l0.gu, l0.down }) |w| {
        const K: usize = w.k;
        const N: usize = w.n;
        const NT: usize = N;
        const name: []const u8 = if (N == 2 * q.FF) "gu" else "down";
        const full_oracle = K == q.FF; // down: host scratch for the fp64 oracle stays ~2.1 GB
        for ([_]u32{ 4, 3, 2, 1 }) |bits| {
            const tbytes: usize = (K / 16) * (N / 16) * @as(usize, 16 * bits) * 2;
            const tr = try d.alloc_(tbytes);
            const host_tr = try alloc.alloc(u8, tbytes);
            defer alloc.free(host_tr);
            var prng = std.Random.DefaultPrng.init(0x5137 + bits + @as(u64, K));
            prng.random().bytes(host_tr);
            try d.copyIn(tr, host_tr);
            const suh = try d.alloc_(K * 2);
            const svh = try d.alloc_(N * 2);
            const hsuh = try alloc.alloc(f16, K);
            defer alloc.free(hsuh);
            const hsvh = try alloc.alloc(f16, N);
            defer alloc.free(hsvh);
            var vr = std.Random.DefaultPrng.init(0x9137 + bits + @as(u64, N));
            for (hsuh) |*s| s.* = @floatCast(vr.random().float(f32) * 0.1 - 0.05);
            for (hsvh) |*s| s.* = @floatCast(vr.random().float(f32) * 0.1 - 0.05);
            try d.copyIn(suh, std.mem.sliceAsBytes(hsuh));
            try d.copyIn(svh, std.mem.sliceAsBytes(hsvh));
            // random fp16 activations in a.h's first T*K slots; the GEMM reads [T][K] there
            const ha = try alloc.alloc(f16, T * K);
            defer alloc.free(ha);
            for (ha) |*x| x.* = @floatCast(vr.random().float(f32) * 2.0 - 1.0);
            try d.copyIn(a.h, std.mem.sliceAsBytes(ha));

            const wscr = try d.alloc_(K * NT * 2);
            var bd: f64 = 1e9;
            for (0..30) |it| {
                const t0 = q.now();
                try d.launch(d.k.exl3_dequant128, .{ .x = @intCast(K / 128), .y = @intCast(NT / 128) }, .{ tr, suh, svh, wscr, @as(i32, @intCast(K)), @as(i32, @intCast(N)), @as(i32, @intCast(NT)), @as(i32, @intCast(bits)) });
                try d.sync();
                if (it > 3) bd = @min(bd, q.now() - t0);
            }
            const C = try d.alloc_(T * NT * 2);
            var bg: f64 = 1e9;
            for (0..30) |it| {
                const t0 = q.now();
                try d.launch(d.k.exl3_prompt16, .{ .x = @intCast((T + 127) / 128), .y = @intCast(NT / 128) }, .{ a.h, wscr, C, @as(i32, @intCast(T)), @as(i32, @intCast(NT)), @as(i32, @intCast(K)) });
                try d.sync();
                if (it > 3) bg = @min(bg, q.now() - t0);
            }
            var bs4: f64 = 0; // 0 = not measured: the seed int4 row only exists on int4 packs
            if (w.exl3 == null) { // the seed int4 kernels cannot read trellis buffers; the comparison row exists for the int4 packs only
                for (0..30) |it| {
                    const t0 = q.now();
                    try q.gemm(d, a.pscr, a.h, a.ahi, a.alo, w, a.p, T);
                    try d.sync();
                    if (it > 3) bs4 = @min(bs4, q.now() - t0);
                }
            }
            var bc: f64 = 1e9; // the path's real cost: dequant + GEMM back to back
            for (0..30) |it| {
                const t0 = q.now();
                try d.launch(d.k.exl3_dequant128, .{ .x = @intCast(K / 128), .y = @intCast(NT / 128) }, .{ tr, suh, svh, wscr, @as(i32, @intCast(K)), @as(i32, @intCast(N)), @as(i32, @intCast(NT)), @as(i32, @intCast(bits)) });
                try d.launch(d.k.exl3_prompt16, .{ .x = @intCast((T + 127) / 128), .y = @intCast(NT / 128) }, .{ a.h, wscr, C, @as(i32, @intCast(T)), @as(i32, @intCast(NT)), @as(i32, @intCast(K)) });
                try d.sync();
                if (it > 3) bc = @min(bc, q.now() - t0);
            }

            // dequant check: device fp16 scratch vs the fp64 oracle. down: all K rows
            // (host scratch ~2.1 GB); gu: the first and last 128-row blocks (a bug in the
            // bk indexing would show in any block; both ends pin it).
            const wref = try alloc.alloc(u16, K * NT); // the device fp16 scratch, host side
            defer alloc.free(wref);
            const wp = try d.g.hostAlloc(K * NT * 2);
            defer d.g.free(wp);
            try q.copy(d, wp, wscr, K * NT * 2);
            try d.sync();
            @memcpy(wref, @as([*]const u16, @ptrCast(@alignCast(wp)))[0 .. K * NT]);
            var dworst: f32 = 0;
            var drel: f32 = 0;
            var drms: f32 = 0;
            if (full_oracle) {
                try exl3OracleCheck(host_tr, hsuh, hsvh, wref, K, N, NT, bits, 0, K, &dworst, &drel, &drms);
            } else {
                try exl3OracleCheck(host_tr, hsuh, hsvh, wref, K, N, NT, bits, 0, 128, &dworst, &drel, &drms);
                try exl3OracleCheck(host_tr, hsuh, hsvh, wref, K, N, NT, bits, K - 128, 128, &dworst, &drel, &drms);
            }
            if (dworst > 0.02 * drms) return error.Exl3GateFailed;

            // GEMM check: two tiles — the first and the last (rows T-128.., cols NT-128..) —
            // against a host fp32 matmul over the device weights.
            const cref = try alloc.alloc(u16, 2 * 128 * NT);
            defer alloc.free(cref);
            const cp = try d.g.hostAlloc(2 * 128 * NT * 2);
            defer d.g.free(cp);
            try q.copy(d, cp, C, 128 * NT * 2);
            try q.copy(d, @ptrFromInt(@intFromPtr(cp) + 128 * NT * 2), @ptrFromInt(@intFromPtr(C) + @as(usize, T - 128) * NT * 2), 128 * NT * 2);
            try d.sync();
            @memcpy(cref, @as([*]const u16, @ptrCast(@alignCast(cp)))[0 .. 2 * 128 * NT]);
            var gworst: f32 = 0;
            var grel: f32 = 0;
            var grms: f32 = 0;
            exl3GemmTileCheck(ha, wref, cref[0 .. 128 * NT], K, NT, 0, 0, 0, &gworst, &grel, &grms);
            exl3GemmTileCheck(ha, wref, cref[128 * NT ..], K, NT, T - 128, NT - 128, T - 128, &gworst, &grel, &grms);
            if (gworst > 0.02 * grms) return error.Exl3GateFailed;
            const dq_gb: f64 = @as(f64, @floatFromInt(K * N * 2 + tbytes + K * 2 + N * 2)) / bd / 1e9;
            const mm_flops: f64 = 2.0 * @as(f64, @floatFromInt(K)) * @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(T));
            std.debug.print("exl3 prompt {s} {d} bits K {d} N {d} T {d}: dequant {d:.3} ms ({d:.0} GB/s) | gemm {d:.3} ms ({d:.2} TFLOP/s) | dequant+gemm {d:.3} ms | seed int4 {d:.3} ms | dequant-vs-oracle max {d:.6} rel {d:.4} | gemm-vs-ref max {d:.4} rel {d:.5}\n", .{ name, bits, K, N, T, bd * 1e3, dq_gb, bg * 1e3, mm_flops / bg / 1e12, bc * 1e3, bs4 * 1e3, dworst, drel, gworst, grel });
            d.g.free(wscr);
            d.g.free(C);
            d.g.free(suh);
            d.g.free(svh);
        }
        // Edge-case pass (down, 4 bits, correctness only): T = 1093 makes the last M-tile's
        // subgroups straddle the M boundary (partial 8-row store blocks) and the tail check
        // sits on REAL columns; NT = N+128 exercises the dequant's SVH guard and the zero
        // pad columns; unit-magnitude scales make the phase-B intermediate rounding visible.
        // A second trellis with N' = N+16 exercises the partial-N contract (the block that
        // straddles N must stay UNROTATED, matching the host oracle).
        if (full_oracle) {
            const bits: u32 = 4;
            const T2: u32 = 1093;
            const NT2: usize = N + 128;
            const tbytes: usize = (K / 16) * (N / 16) * @as(usize, 16 * bits) * 2;
            const tr = try d.alloc_(tbytes);
            defer d.g.free(tr);
            const host_tr = try alloc.alloc(u8, tbytes);
            defer alloc.free(host_tr);
            var prng = std.Random.DefaultPrng.init(0x5137 + bits + @as(u64, K));
            prng.random().bytes(host_tr);
            try d.copyIn(tr, host_tr);
            const suh = try d.alloc_(K * 2);
            const svh = try d.alloc_(NT2 * 2);
            const hsuh = try alloc.alloc(f16, K);
            defer alloc.free(hsuh);
            const hsvh = try alloc.alloc(f16, NT2);
            defer alloc.free(hsvh);
            var vr = std.Random.DefaultPrng.init(0x2711);
            for (hsuh) |*s| s.* = @floatCast(vr.random().float(f32) * 2.0 - 1.0);
            for (hsvh) |*s| s.* = @floatCast(vr.random().float(f32) * 2.0 - 1.0);
            try d.copyIn(suh, std.mem.sliceAsBytes(hsuh));
            try d.copyIn(svh, std.mem.sliceAsBytes(hsvh));
            const ha = try alloc.alloc(f16, T2 * K);
            defer alloc.free(ha);
            for (ha) |*x| x.* = @floatCast(vr.random().float(f32) * 2.0 - 1.0);
            try d.copyIn(a.h, std.mem.sliceAsBytes(ha));
            const wscr = try d.alloc_(K * NT2 * 2);
            defer d.g.free(wscr);
            try d.launch(d.k.exl3_dequant128, .{ .x = @intCast(K / 128), .y = @intCast(NT2 / 128) }, .{ tr, suh, svh, wscr, @as(i32, @intCast(K)), @as(i32, @intCast(N)), @as(i32, @intCast(NT2)), @as(i32, @intCast(bits)) });
            const C2 = try d.alloc_(T2 * NT2 * 2);
            defer d.g.free(C2);
            try d.launch(d.k.exl3_prompt16, .{ .x = @intCast((T2 + 127) / 128), .y = @intCast(NT2 / 128) }, .{ a.h, wscr, C2, @as(i32, @intCast(T2)), @as(i32, @intCast(NT2)), @as(i32, @intCast(K)) });
            try d.sync();
            const wref = try alloc.alloc(u16, K * NT2);
            defer alloc.free(wref);
            const wp = try d.g.hostAlloc(K * NT2 * 2);
            defer d.g.free(wp);
            try q.copy(d, wp, wscr, K * NT2 * 2);
            try d.sync();
            @memcpy(wref, @as([*]const u16, @ptrCast(@alignCast(wp)))[0 .. K * NT2]);
            var dworst: f32 = 0;
            var drel: f32 = 0;
            var drms: f32 = 0;
            try exl3OracleCheck(host_tr, hsuh, hsvh, wref, K, N, NT2, bits, 0, 128, &dworst, &drel, &drms);
            try exl3OracleCheck(host_tr, hsuh, hsvh, wref, K, N, NT2, bits, K - 128, 128, &dworst, &drel, &drms);
            if (dworst > 0.02 * drms) return error.Exl3GateFailed;
            var padw: f32 = 0; // pad columns of the scratch must be exactly zero
            for (0..K) |k| {
                for (0..128) |n| {
                    padw = @max(padw, @abs(@as(f32, @as(f16, @bitCast(wref[k * NT2 + N + n])))));
                }
            }
            const cref = try alloc.alloc(u16, 2 * 128 * NT2);
            defer alloc.free(cref);
            const cp = try d.g.hostAlloc(2 * 128 * NT2 * 2);
            defer d.g.free(cp);
            try q.copy(d, cp, C2, 128 * NT2 * 2);
            try q.copy(d, @ptrFromInt(@intFromPtr(cp) + 128 * NT2 * 2), @ptrFromInt(@intFromPtr(C2) + @as(usize, T2 - 128) * NT2 * 2), 128 * NT2 * 2);
            try d.sync();
            @memcpy(cref, @as([*]const u16, @ptrCast(@alignCast(cp)))[0 .. 2 * 128 * NT2]);
            var gworst: f32 = 0;
            var grel: f32 = 0;
            var grms: f32 = 0;
            exl3GemmTileCheck(ha, wref, cref[0 .. 128 * NT2], K, NT2, 0, 0, 0, &gworst, &grel, &grms);
            var pworst: f32 = 0;
            var prel: f32 = 0;
            var prms: f32 = 0;
            exl3GemmTileCheck(ha, wref, cref[128 * NT2 ..], K, NT2, T2 - 128, N, T2 - 128, &pworst, &prel, &prms); // all-pad tile: ref == 0
            if (gworst > 0.02 * grms or pworst > 1e-6) return error.Exl3GateFailed;
            std.debug.print("exl3 prompt edge down 4 bits T {d} NT {d} (+128 pad): dequant-vs-oracle max {d:.5} rms {d:.4} | pad-scratch max {d:.9} | gemm real-tile max {d:.4} rms {d:.3} | tail-pad-tile max {d:.9}\n", .{ T2, NT2, dworst, drms, padw, gworst, grms, pworst });

            // Partial-N contract: N' = N+16 tiles; the block straddling N' stays unrotated.
            const N3: usize = N + 16;
            const NT3: usize = N3 + (128 - N3 % 128);
            const tb3: usize = (K / 16) * (N3 / 16) * @as(usize, 16 * bits) * 2;
            const tr3 = try d.alloc_(tb3);
            defer d.g.free(tr3);
            const host_tr3 = try alloc.alloc(u8, tb3);
            defer alloc.free(host_tr3);
            var pr3 = std.Random.DefaultPrng.init(0x3131 + @as(u64, K));
            pr3.random().bytes(host_tr3);
            try d.copyIn(tr3, host_tr3);
            const svh3 = try d.alloc_(N3 * 2);
            defer d.g.free(svh3);
            const hsvh3 = try alloc.alloc(f16, N3);
            defer alloc.free(hsvh3);
            var vr3 = std.Random.DefaultPrng.init(0x2712);
            for (hsvh3) |*s| s.* = @floatCast(vr3.random().float(f32) * 2.0 - 1.0);
            try d.copyIn(svh3, std.mem.sliceAsBytes(hsvh3));
            const wscr3 = try d.alloc_(K * NT3 * 2);
            defer d.g.free(wscr3);
            try d.launch(d.k.exl3_dequant128, .{ .x = @intCast(K / 128), .y = @intCast(NT3 / 128) }, .{ tr3, suh, svh3, wscr3, @as(i32, @intCast(K)), @as(i32, @intCast(N3)), @as(i32, @intCast(NT3)), @as(i32, @intCast(bits)) });
            try d.sync();
            const wref3 = try alloc.alloc(u16, K * NT3);
            defer alloc.free(wref3);
            const wp3 = try d.g.hostAlloc(K * NT3 * 2);
            defer d.g.free(wp3);
            try q.copy(d, wp3, wscr3, K * NT3 * 2);
            try d.sync();
            @memcpy(wref3, @as([*]const u16, @ptrCast(@alignCast(wp3)))[0 .. K * NT3]);
            var dworst3: f32 = 0;
            var drel3: f32 = 0;
            var drms3: f32 = 0;
            try exl3OracleCheck(host_tr3, hsuh, hsvh3, wref3, K, N3, NT3, bits, 0, 128, &dworst3, &drel3, &drms3);
            try exl3OracleCheck(host_tr3, hsuh, hsvh3, wref3, K, N3, NT3, bits, K - 128, 128, &dworst3, &drel3, &drms3);
            if (dworst3 > 0.02 * drms3) return error.Exl3GateFailed;
            std.debug.print("exl3 prompt edge down 4 bits partial-N N {d}: dequant-vs-oracle max {d:.5} rms {d:.4}\n", .{ N3, dworst3, drms3 });
        }
    }
}

// ---- EXL3 trellis encoder probe (exl3_enc.cl, fable spec section 6's device half) ----
// T6 logs the Level Zero limits through xpu/exl3_encode.zig's queryLimits; the
// self-consistency check encodes seeded unit-Gaussian fp16 tiles at 3 bits on the card,
// reads the ring words back and holds them against the oracle decoder: tileStates repacks
// to the very words (the closed-path identity of spec section 3) and unpack's values meet
// the RMS bar for unit-variance input; T8 times tiles per second at 3 bits for the
// candidate counts. The reference equality (T7) joins once core/exl3_encode.zig lands.

fn gaussPair(rnd: std.Random) [2]f32 {
    while (true) {
        const r1 = rnd.float(f32);
        const r2 = rnd.float(f32);
        if (!(r1 > 0.0)) continue;
        const r = @sqrt(-2.0 * @log(r1));
        const a = r * @cos(6.283185307179586 * r2);
        const b = r * @sin(6.283185307179586 * r2);
        if (@abs(a) <= 4.0 and @abs(b) <= 4.0) return .{ a, b };
    }
}

/// The section 3 identity run backwards on one tile: decode the states, pack their fresh
/// bits into ring words, and the words must come back unchanged (a closed path decodes to
/// itself; an open one differs at the low positions).
fn encRoundTrip(w: enc.Width, words: []const i16) bool {
    std.debug.assert(words.len == w.tileWords());
    var ends: [256]u32 = undefined;
    core.exl3_format.streamEnds(w, &ends);
    var states: [256]u32 = undefined;
    core.exl3_format.tileStates(w, words, &states);
    var ring: [64]u32 = @splat(0); // MSB-first bits in little-endian word pairs, streamBit's view
    var prev_end: u32 = 0;
    for (0..256) |p| {
        const e = ends[p];
        const b: u32 = e - prev_end;
        prev_end = e;
        const n: u32 = states[p] & (@as(u32, 1) << @intCast(b)) - 1;
        for (0..b) |k| {
            const pos = e - b + @as(u32, @intCast(k));
            if (((n >> @intCast(b - 1 - @as(u32, @intCast(k)))) & 1) != 0)
                ring[pos >> 5] |= @as(u32, 1) << @intCast(31 - (pos & 31));
        }
    }
    const want: [*]const u32 = @ptrCast(@alignCast(words.ptr));
    for (0..w.tileWords() / 2) |i| if (want[i] != ring[i]) return false;
    return true;
}

fn encProbe() !void {
    var d: q.Dev = .{ .g = try q.ze.Gpu.open(0xe223), .list = undefined, .k = undefined };
    errdefer d.g.close();
    d.list = try d.g.immediate(0, 0);
    var namez: [256]u8 = d.g.props.name;
    const mem = try d.g.memory();
    std.debug.print("enc probe: {s}: {d} of {d} card bytes free\n", .{ std.mem.span(@as([*:0]const u8, @ptrCast(&namez))), mem.free, mem.size });
    _ = try enc.queryLimits(&d);

    // one [16, 2048] seeded Gaussian matrix: its k-major tiles are columns tn*16..tn*16+15,
    // so any leading tile set is a [16, count*16] matrix and every run shares the seed
    const w: enc.Width = .{ .halves = 6 };
    const tiles_max: u32 = 128;
    const host = try alloc.alloc(f16, @as(usize, 16) * tiles_max * 16);
    defer alloc.free(host);
    var prng = std.Random.DefaultPrng.init(0x5EEDC0DE);
    var i: usize = 0;
    while (i < host.len) {
        const g = gaussPair(prng.random());
        host[i] = @floatCast(g[0]);
        host[i + 1] = @floatCast(g[1]);
        i += 2;
    }
    const wq = try d.upload(std.mem.sliceAsBytes(host));

    // self-consistency at 3 bits: kernel words against the oracle (tileStates round-trip
    // tile by tile) and against the targets (unpack RMS), plus the status certificates
    const tiles_check: u32 = 64;
    {
        var e = try enc.Encoder.init(&d, w, tiles_check, tiles_max);
        const tr = try d.alloc_(@as(usize, tiles_check) * w.tileWords() * 2);
        try e.encode(&d, wq, 16, tiles_max * 16, tr, 0, tiles_check, null);
        try d.sync();
        const stats = try alloc.alloc(enc.TileStat, tiles_check);
        defer alloc.free(stats);
        try e.readStats(&d, stats, null);
        var n_opt: u32 = 0;
        var n_closed: u32 = 0;
        var n_failed: u32 = 0;
        var n_pruned1: u32 = 0;
        var n_pruned2: u32 = 0;
        var gap_sum: u64 = 0;
        var gap_max: u64 = 0;
        var gap_hist: [40]u64 = std.mem.zeroes([40]u64); // 40 log2 buckets (fable rebase ruling)
        for (stats, 0..) |s, ti| {
            if (s.flags & 1 != 0) n_pruned1 += 1;
            if (s.flags & 2 != 0) n_pruned2 += 1;
            switch (s.status) {
                0 => n_opt += 1,
                1 => {
                    n_closed += 1;
                    const gap = s.cost - s.open_cost;
                    gap_sum += gap;
                    gap_max = @max(gap_max, gap);
                    gap_hist[if (gap == 0) 0 else @min(@as(u64, 39), 63 - @clz(gap))] += 1;
                },
                else => {
                    n_failed += 1;
                    std.debug.print("enc self: tile {d} FAILED cost={x} open={x} pinned={x} flags={x}\n", .{ ti, s.cost, s.open_cost, s.pinned, s.flags });
                },
            }
            // open_cost <= cost is asserted only with pass-1 pruned clear (ruling): bit0
            // tiles had a finite branch pruned by the saturating add, so their ordering
            // is flag-qualified, not certified
            if ((s.flags & 1) == 0 and s.cost < s.open_cost) std.debug.print("enc self: tile {d} cost {d} below its open_cost {d}\n", .{ ti, s.cost, s.open_cost });
        }
        std.debug.print("enc self: {d} tiles at 3 bits: {d} optimal, {d} closed (heuristic gap sum {d}, max {d}), {d} failed, pruned1 {d} pruned2 {d}\n", .{ tiles_check, n_opt, n_closed, gap_sum, gap_max, n_failed, n_pruned1, n_pruned2 });
        std.debug.print("enc self: gap histogram (40 log2 buckets):", .{});
        for (gap_hist, 0..) |c, bk| {
            if (c != 0) std.debug.print(" {d}:{d}", .{ bk, c });
        }
        std.debug.print("\n", .{});
        const tbuf = try alloc.alloc(i16, @as(usize, tiles_check) * w.tileWords());
        defer alloc.free(tbuf);
        try e.readTrellis(&d, tr, tbuf, null);
        var round_ok: u32 = 0;
        var dec: [256]f16 = undefined;
        var sq: f64 = 0;
        for (0..tiles_check) |ti| {
            const tile = tbuf[ti * w.tileWords() ..][0 .. w.tileWords()];
            if (encRoundTrip(w, tile)) round_ok += 1;
            core.exl3_format.unpack(w, .mul1, tile, 1, 1, &dec);
            for (dec, 0..) |got, at| {
                const df = @as(f64, @floatCast(got)) - @as(f64, @floatCast(host[(at / 16) * @as(usize, tiles_max * 16) + ti * 16 + at % 16]));
                sq += df * df;
            }
        }
        const rms = @sqrt(sq / @as(f64, @floatFromInt(@as(usize, tiles_check) * 256)));
        const pass = n_failed == 0 and round_ok == tiles_check and rms < 0.2;
        std.debug.print("enc self: tileStates round trip {d}/{d}, unpack RMS {d:.4} (bar 0.2): {s}\n", .{ round_ok, tiles_check, rms, if (pass) "PASS" else "FAIL" });
        if (!pass) return error.EncoderSelfCheck;
        // WHAT (LDLQ driver spec 4.3, the kernel's 10th arg): the same launch with WHAT
        // live writes codebookValue(.mul1, state[p]) at each placement slot, bit for bit
        // the oracle's value on the very trellis words the kernel wrote (tbuf)
        const what_dev = try d.alloc_(@as(usize, 16) * tiles_max * 16 * 2);
        const what_host = try d.g.hostAlloc(@as(usize, 16) * tiles_max * 16 * 2);
        defer d.g.free(what_host);
        try e.encodeWhat(&d, wq, 16, tiles_max * 16, tr, 0, tiles_check, what_dev, null);
        try d.sync();
        try q.copy(&d, what_host, what_dev, @as(usize, 16) * tiles_max * 16 * 2);
        try d.sync();
        const what16: [*]const u16 = @ptrCast(@alignCast(what_host));
        var what_ok: u32 = 0;
        var what_first: ?usize = null;
        var wf_got: u16 = 0;
        var wf_want: u16 = 0;
        for (0..tiles_check) |ti| {
            var states: [256]u32 = undefined;
            core.exl3_format.tileStates(w, tbuf[ti * w.tileWords() ..][0 .. w.tileWords()], &states);
            for (0..256) |p| {
                const l = p >> 3;
                const j = p & 7;
                const row = 2 * (l & 3) + (j & 1) + 8 * ((j >> 1) & 1);
                const col = (l >> 2) + 8 * (j >> 2);
                const want: u16 = @bitCast(core.exl3_format.codebookValue(.mul1, @intCast(states[p])));
                const at = row * @as(usize, tiles_max * 16) + ti * 16 + col;
                if (what16[at] == want) what_ok += 1 else if (what_first == null) {
                    what_first = ti * 256 + p;
                    wf_got = what16[at];
                    wf_want = want;
                }
            }
        }
        const passw = what_ok == tiles_check * 256;
        std.debug.print("enc self: WHAT {d}/{d} words bit-identical to codebookValue on the kernel trellis: {s}\n", .{ what_ok, tiles_check * 256, if (passw) "PASS" else "FAIL" });
        if (what_first) |ix| std.debug.print("  first WHAT diff at tile {d} pos {d}: device {x:0>4} want {x:0>4}\n", .{ ix / 256, ix % 256, wf_got, wf_want });
        if (!passw) return error.EncoderWhat;
    }

    // T7: reference parity (core/exl3_encode.zig): the SAME seeded tiles go to the
    // reference pack (which gathers through exl3_format.tilePlacement exactly as the
    // kernel does) and to the kernel; trellis words must match byte for byte and the full
    // 32-byte TileStat must be bit-identical to the host mirror of Result (fable rebase
    // ruling: "T7 gains the T4+ tiles and compares the full 32-byte TileStat").
    // Non-square 16x8 grid: the tk = tile / tn16 decomposition and the gather are pinned
    // only when tiles_k > 1 (fable review). Tie rules beyond Gaussian's near-tie-free
    // totals: tn 2 is the constant all-zero tile, tn 3 is planted exactly on codebook
    // values - they tie at almost every group and both sides must still agree. T4+
    // saturation tiles: tn 4 all +4, tn 5 all -4, tn 6 checkerboard +/-4, tn 7
    // Hadamard-sign +/-4 (seeded): legal clamped-outlier tiles that saturate under the
    // plain saturating add and must encode finitely, identically, under the rebase.
    // count is sized per width: BP alone is 8 MiB per tile at halves 2/3 (1 GiB at count
    // 128), so the scratch stays under one budget and the launches chunk.
    const eref = core.exl3_encode;
    const mref = eref.Metric.init();
    var t7_pass = true;
    for ([_]u16{ 2, 3, 4, 5, 6, 8, 16 }) |hh| {
        const wt: enc.Width = .{ .halves = hh };
        const tk7: u32 = 16; // tiles_k > tiles_n: a square grid would not pin the tk/tn split
        const tn7: u32 = 8; // tn 0..1 Gaussian, 2 all-zero, 3 planted on codebook,
                            // 4 all +4, 5 all -4, 6 checkerboard +/-4, 7 Hadamard-sign +/-4
        const tiles7: u32 = tk7 * tn7;
        const K7: u32 = tk7 * 16;
        const N7: u32 = tn7 * 16;
        const hwq = try alloc.alloc(f16, @as(usize, K7) * N7);
        defer alloc.free(hwq);
        var prng7 = std.Random.DefaultPrng.init(0x7E700 + @as(u64, hh));
        var pos7: usize = 0;
        while (pos7 < hwq.len) {
            const g7 = gaussPair(prng7.random());
            hwq[pos7] = @floatCast(g7[0]);
            hwq[pos7 + 1] = @floatCast(g7[1]);
            pos7 += 2;
        }
        for (0..K7) |row| for (0..16) |col| {
            const base_at = @as(usize, row) * N7 + col;
            hwq[base_at + @as(usize, 2) * 16] = 0;
            hwq[base_at + @as(usize, 4) * 16] = 4;
            hwq[base_at + @as(usize, 5) * 16] = -4;
            hwq[base_at + @as(usize, 6) * 16] = if (((row + col) & 1) == 0) @as(f16, 4) else @as(f16, -4);
            hwq[base_at + @as(usize, 7) * 16] = if (prng7.random().boolean()) @as(f16, 4) else @as(f16, -4);
        };
        var place7: [256]core.exl3_format.Placement = undefined;
        core.exl3_format.tilePlacement(&place7);
        for (0..tk7) |tkr| {
            var planted: [256]u16 = undefined;
            eref.closedFromRing(prng7.random(), hh, &planted);
            for (0..256) |p| {
                const row: usize = place7[p].row;
                const col: usize = place7[p].col;
                hwq[(@as(usize, tkr) * 16 + row) * N7 + @as(usize, 3) * 16 + col] = core.exl3_format.codebookValue(.mul1, planted[p]);
            }
        }
        var scratch7 = try eref.Scratch.init(alloc, wt);
        defer scratch7.deinit(alloc);
        const ref_tr = try alloc.alloc(i16, @as(usize, tiles7) * wt.tileWords());
        defer alloc.free(ref_tr);
        const ref_res = try alloc.alloc(eref.Result, tiles7);
        defer alloc.free(ref_res);
        try eref.pack(&mref, wt, hwq, tk7, tn7, ref_tr, &scratch7, ref_res, null);
        const wq7 = try d.upload(std.mem.sliceAsBytes(hwq));
        const gmax7: usize = @as(usize, 1) << @intCast(@as(u8, 16) - enc.minFreshBits(wt));
        const per_tile: usize = 2 * 65536 * 4 + 256 * gmax7; // cost pair + BP bytes per tile
        const count7: u32 = @intCast(@max(@as(usize, 8), @min(@as(usize, tiles7), 335_544_320 / per_tile)));
        var e7 = try enc.Encoder.init(&d, wt, count7, tiles7);
        const tr7 = try d.alloc_(@as(usize, tiles7) * wt.tileWords() * 2);
        var first7: u32 = 0;
        while (first7 < tiles7) : (first7 += count7) {
            try e7.encode(&d, wq7, K7, N7, tr7, first7, @min(count7, tiles7 - first7), null);
        }
        try d.sync();
        const kn_tr = try alloc.alloc(i16, ref_tr.len);
        defer alloc.free(kn_tr);
        try e7.readTrellis(&d, tr7, kn_tr, null);
        const st7 = try alloc.alloc(enc.TileStat, tiles7);
        defer alloc.free(st7);
        try e7.readStats(&d, st7, null);
        var words_ok: u32 = 0;
        var stats_ok: u32 = 0;
        var first_bad: ?usize = null;
        for (0..tiles7) |ti| {
            const wr = ref_tr[ti * wt.tileWords() ..][0 .. wt.tileWords()];
            const wk = kn_tr[ti * wt.tileWords() ..][0 .. wt.tileWords()];
            if (std.mem.eql(u8, std.mem.sliceAsBytes(wr), std.mem.sliceAsBytes(wk))) words_ok += 1 else if (first_bad == null) first_bad = ti;
            const r = ref_res[ti];
            const mirror = enc.TileStat{
                .cost = @intCast(r.cost),
                .open_cost = @intCast(r.open_cost),
                .pinned = @intCast(r.pinned),
                .status = @intFromEnum(r.status),
                .flags = @intCast(r.flags),
                .reserved = 0,
            };
            if (std.mem.eql(u8, std.mem.asBytes(&mirror), std.mem.asBytes(&st7[ti]))) stats_ok += 1 else if (first_bad == null) first_bad = ti;
        }
        const ok7 = words_ok == tiles7 and stats_ok == tiles7;
        t7_pass = t7_pass and ok7;
        std.debug.print("enc T7 halves={d}: {d}/{d} tiles words identical, full 32-byte stats {d}/{d} equal to the reference mirror (non-square 16x8 grid: Gaussian, zero, planted-codebook tie, and T4+ +/-4 saturation tiles, count {d}): {s}\n", .{ hh, words_ok, tiles7, stats_ok, tiles7, count7, if (ok7) "PASS" else "FAIL" });
        if (first_bad) |fb| { // dump both sides for the first divergent tile: report, never patch
            std.debug.print("  tile {d} (tk {d}, tn {d}) halves {d}: kernel cost={x} open={x} pinned={x} status={d} flags={x}, reference cost={x} open={x} pinned={x} status={d} flags={x}\n", .{
                fb, fb / tn7, fb % tn7, hh, st7[fb].cost, st7[fb].open_cost, st7[fb].pinned, st7[fb].status, st7[fb].flags, ref_res[fb].cost, ref_res[fb].open_cost, ref_res[fb].pinned, @intFromEnum(ref_res[fb].status), @as(u32, @intCast(ref_res[fb].flags)) });
            var sk: [256]u32 = undefined;
            core.exl3_format.tileStates(wt, kn_tr[fb * wt.tileWords() ..][0 .. wt.tileWords()], &sk);
            var sr: [256]u32 = undefined;
            core.exl3_format.tileStates(wt, ref_tr[fb * wt.tileWords() ..][0 .. wt.tileWords()], &sr);
            for (0..32) |row| {
                std.debug.print("  states {d}:", .{row * 8});
                for (0..8) |c| std.debug.print(" ref {x:0>4}", .{ref_res[fb].states[row * 8 + c]});
                std.debug.print(" |", .{});
                for (0..8) |c| std.debug.print(" kern {x:0>4} refdec {x:0>4}", .{ sk[row * 8 + c], sr[row * 8 + c] });
                std.debug.print("\n", .{});
            }
        }
    }
    if (!t7_pass) return error.EncoderRefParity;

    // T7c: the device contract (fable review). The kernel's workgroup-uniform arg trap
    // returns without writing STAT or TR, and alloc_ content is unspecified, so a trapped
    // launch once read back as a perfect cost-0 optimal encode. The host now validates
    // halves, k, n_cols, first and n against the buffer extents before anything queues:
    // every row of this table must come back an error, and one legal launch of the same
    // small encoder must still write its (pre-filled status-2) certificates.
    {
        const wc: enc.Width = .{ .halves = 6 };
        var ec = try enc.Encoder.init(&d, wc, 4, 2); // stat_bytes = 64 B: two 32-byte tile certificates
        var ew = try enc.Encoder.init(&d, .{ .halves = 17 }, 4, 8); // outside the mul1 range
        const zeros = try alloc.alloc(u8, 16 * 64 * 2); // the legal launch reads a [16, 64] fp16 matrix
        defer alloc.free(zeros);
        @memset(zeros, 0);
        const wqc = try d.upload(zeros);
        const trc = try d.alloc_(1024);
        const bad = [_]struct { k: u32, n: u32, first: u32, len: u32 }{
            .{ .k = 120, .n = 64, .first = 0, .len = 2 }, // k % 16 != 0
            .{ .k = 16, .n = 0, .first = 0, .len = 2 }, // empty matrix
            .{ .k = 16, .n = 63, .first = 0, .len = 2 }, // n_cols % 16 != 0
            .{ .k = 16, .n = 64, .first = 1, .len = 4 }, // tiles [1,5) past a 1x4 grid
            .{ .k = 16, .n = 128, .first = 0, .len = 5 }, // n > count
            .{ .k = 16, .n = 128, .first = 0, .len = 0 }, // n == 0
            .{ .k = 16, .n = 128, .first = 1, .len = 2 }, // (1+2)*32 > stat_bytes 64
        };
        var refusals: u32 = 0;
        for (bad) |tc| {
            if ((ec.encode(&d, wqc, tc.k, tc.n, trc, tc.first, tc.len, null))) |_| {
                std.debug.print("  T7c: launch k={d} n={d} first={d} len={d} was ACCEPTED\n", .{ tc.k, tc.n, tc.first, tc.len });
            } else |_| refusals += 1;
        }
        if ((ew.encode(&d, wqc, 16, 64, trc, 0, 1, null))) |_| {
            std.debug.print("  T7c: halves 17 launch was ACCEPTED\n", .{});
        } else |_| refusals += 1;
        try ec.encode(&d, wqc, 16, 64, trc, 0, 2, null); // legal: tiles [0,2) of a 1x4 grid, two certificates
        try d.sync();
        const stc = try alloc.alloc(enc.TileStat, 2);
        defer alloc.free(stc);
        try ec.readStats(&d, stc, null);
        var certs: u32 = 0;
        for (stc, 0..) |s, ti| {
            if (s.status < 2) certs += 1 else std.debug.print("  T7c: tile {d} kept the pre-filled status {d} after a legal launch\n", .{ ti, s.status });
        }
        const passc = refusals == 8 and certs == 2;
        std.debug.print("enc T7c: device contract {d}/8 bad launches refused on the host, {d}/2 pre-filled certificates written by the legal one: {s}\n", .{ refusals, certs, if (passc) "PASS" else "FAIL" });
        if (!passc) return error.EncoderContract;
    }

    // T-LDLQ-6 (eaq-ldlq spec 4.3, section 5.4): the encoder's WHAT output equals the host
    // unpack of the kernel's own trellis words, for one [16, N] strip at halves 4, 6, 8.
    // WHAT is the LDLQ driver's feedback value; bit-identity here pins the placement
    // closed form and the exact integer weight against the oracle's unpack.
    var tld_pass = true;
    for ([_]u16{ 4, 6, 8 }) |hh| {
        const wt6: enc.Width = .{ .halves = hh };
        const tn6: u32 = 8; // one strip: K = 16, eight tile columns
        const n6: usize = @as(usize, 16) * tn6 * 16;
        const hw6 = try alloc.alloc(f16, n6);
        defer alloc.free(hw6);
        var prng6 = std.Random.DefaultPrng.init(0x1D1007 + @as(u64, hh));
        var pos6: usize = 0;
        while (pos6 < hw6.len) {
            const g6 = gaussPair(prng6.random());
            hw6[pos6] = @floatCast(g6[0]);
            hw6[pos6 + 1] = @floatCast(g6[1]);
            pos6 += 2;
        }
        const wq6 = try d.upload(std.mem.sliceAsBytes(hw6));
        var e6 = try enc.Encoder.init(&d, wt6, tn6, tn6);
        const tr6 = try d.alloc_(@as(usize, tn6) * wt6.tileWords() * 2);
        const what6 = try d.alloc_(n6 * 2);
        try e6.encodeWhat(&d, wq6, 16, tn6 * 16, tr6, 0, tn6, what6, null);
        try d.sync();
        const wh6 = try d.g.hostAlloc(n6 * 2);
        defer d.g.free(wh6);
        try q.copy(&d, wh6, what6, n6 * 2);
        try d.sync();
        const got6: [*]const u16 = @ptrCast(@alignCast(wh6));
        const tbuf6 = try alloc.alloc(i16, @as(usize, tn6) * wt6.tileWords());
        defer alloc.free(tbuf6);
        try e6.readTrellis(&d, tr6, tbuf6, null);
        const dec6 = try alloc.alloc(f16, n6);
        defer alloc.free(dec6);
        core.exl3_format.unpack(wt6, .mul1, tbuf6, 1, tn6, dec6);
        var what_ok: u32 = 0;
        var first6: ?usize = null;
        for (0..n6) |at| {
            if (@as(u16, @bitCast(dec6[at])) == got6[at]) what_ok += 1 else if (first6 == null) first6 = at;
        }
        const pass6 = what_ok == n6;
        tld_pass = tld_pass and pass6;
        std.debug.print("enc T-LDLQ-6 halves={d}: {d}/{d} WHAT words bit-identical to the host unpack of the kernel trellis: {s}\n", .{ hh, what_ok, n6, if (pass6) "PASS" else "FAIL" });
        if (first6) |at| std.debug.print("  first diff at [row {d}, n {d}]: WHAT {x:0>4} unpack {x:0>4}\n", .{ at / (tn6 * 16), at % (tn6 * 16), got6[at], @as(u16, @bitCast(dec6[at])) });
    }
    if (!tld_pass) return error.EncoderWhat;

    // T9: the serving path reads what the encoder wrote. A seeded [128, 128] Gaussian
    // matrix at 3 bits goes to the kernel; exl3_dequant128_c decodes its trellis on the
    // device with suh=svh=1, and the result must be bit-identical to the host oracle's
    // unpack of the same words run through an exact-order fp32 replica of the kernel's
    // rotations (same ±1 butterfly pairing and order, one v*inv rounding per phase,
    // phase B's half re-round in between): equal outputs mean the weights the serving
    // kernel decoded from the encoder's words equal the oracle's on those words.
    {
        const w9: enc.Width = .{ .halves = 6 };
        const hw9 = try alloc.alloc(f16, 128 * 128);
        defer alloc.free(hw9);
        var prng9 = std.Random.DefaultPrng.init(0xA917);
        var pos9: usize = 0;
        while (pos9 < hw9.len) {
            const g9 = gaussPair(prng9.random());
            hw9[pos9] = @floatCast(g9[0]);
            hw9[pos9 + 1] = @floatCast(g9[1]);
            pos9 += 2;
        }
        const wq9 = try d.upload(std.mem.sliceAsBytes(hw9));
        var e9 = try enc.Encoder.init(&d, w9, 64, 64);
        const tr9 = try d.alloc_(64 * w9.tileWords() * 2);
        try e9.encode(&d, wq9, 128, 128, tr9, 0, 64, null);
        try d.sync();
        const tbuf9 = try alloc.alloc(i16, 64 * w9.tileWords());
        defer alloc.free(tbuf9);
        try e9.readTrellis(&d, tr9, tbuf9, null);
        const wq_dec = try alloc.alloc(f16, 128 * 128); // the oracle's pre-rotation weights
        defer alloc.free(wq_dec);
        core.exl3_format.unpack(w9, .mul1, tbuf9, 8, 8, wq_dec);
        const dq = try enc.dequantKernel(&d);
        const ones: [128]f16 = @splat(1.0);
        const suh9 = try d.upload(std.mem.sliceAsBytes(&ones));
        const svh9 = try d.upload(std.mem.sliceAsBytes(&ones));
        const wout9 = try d.alloc_(128 * 128 * 2);
        try d.launch(dq, .{ .x = 1, .y = 1 }, .{ tr9, suh9, svh9, wout9, @as(i32, 128), @as(i32, 128), @as(i32, 128), @as(i32, 3), @as(i32, 0) });
        try d.sync();
        const wdev = try d.g.hostAlloc(128 * 128 * 2);
        defer d.g.free(wdev);
        try q.copy(&d, wdev, wout9, 128 * 128 * 2);
        try d.sync();
        const wdev16: [*]const u16 = @ptrCast(@alignCast(wdev));
        const inv = 0.08838834764831845;
        var blkB = try alloc.alloc(f16, 128 * 128); // blk as the kernel leaves it after phase B
        defer alloc.free(blkB);
        var v: [128]f32 = undefined;
        for (0..128) |k| {
            for (0..128) |n| v[n] = @floatCast(wq_dec[k * 128 + n]);
            rotExact128(&v);
            for (0..128) |n| blkB[k * 128 + n] = @floatCast(v[n] * inv);
        }
        const out9 = try alloc.alloc(f16, 128 * 128);
        defer alloc.free(out9);
        for (0..128) |n| {
            for (0..128) |k| v[k] = @floatCast(blkB[k * 128 + n]);
            rotExact128(&v);
            for (0..128) |k| out9[k * 128 + n] = @floatCast(v[k] * inv);
        }
        var rot_ok: u32 = 0;
        var first9: ?usize = null;
        for (0..128 * 128) |at| {
            if (@as(u16, @bitCast(out9[at])) == wdev16[at]) rot_ok += 1 else if (first9 == null) first9 = at;
        }
        const pass9 = rot_ok == 128 * 128;
        std.debug.print("enc T9 3 bits [128,128]: exl3_dequant128_c on the kernel trellis {d}/16384 words bit-identical to the host unpack through the exact rotations: {s}\n", .{ rot_ok, if (pass9) "PASS" else "FAIL" });
        if (first9) |at| std.debug.print("  first diff at [k {d}, n {d}]: device {x:0>4} host {x:0>4}\n", .{ at / 128, at % 128, wdev16[at], @as(u16, @bitCast(out9[at])) });
        if (!pass9) return error.EncoderDequant;
    }

    // T8: tiles per second at 3 bits for the three candidate counts; count * 512 KiB
    // against the T6 cache is the knob the layout work turns
    for ([_]u32{ 32, 64, 128 }) |count| {
        var e = try enc.Encoder.init(&d, w, count, count);
        const tr = try d.alloc_(@as(usize, count) * w.tileWords() * 2);
        const reps: u32 = @max(4, 2048 / count);
        try e.encode(&d, wq, 16, tiles_max * 16, tr, 0, count, null);
        try d.sync();
        const t0 = q.now();
        for (0..reps) |_| try e.encode(&d, wq, 16, tiles_max * 16, tr, 0, count, null);
        try d.sync();
        const dt = (q.now() - t0) / @as(f64, @floatFromInt(reps));
        const tps = @as(f64, @floatFromInt(count)) / dt;
        std.debug.print("enc T8 3 bits count={d} (working set {d} KiB): {d:.3} ms a launch, {d:.1} tiles/s, full model (~100 M tiles) ~{d:.2} h\n", .{ count, count * 512, dt * 1e3, tps, 100_000_000.0 / tps / 3600.0 });
    }
    d.deinit();
}

/// A bit-exact mirror of exl3_h.cl's rot128_lane over the contiguous 128-vector
/// (element index = lane*8 + j): the in-lane butterflies m=1,2,4 are masks 1,2,4 on
/// the element index, the cross-lane shuffle_xor m=1,2,4,8 are masks 8,16,32,64 in
/// the same order, and both sides pair (a, b) as (a+b, a-b) by the bit-test side.
fn rotExact128(v: *[128]f32) void {
    for ([_]usize{ 1, 2, 4, 8, 16, 32, 64 }) |m| {
        var i: usize = 0;
        while (i < 128) : (i += 2 * m) {
            for (0..m) |j| {
                const a = v[i + j];
                const b = v[i + j + m];
                v[i + j] = a + b;
                v[i + j + m] = a - b;
            }
        }
    }
}

// ---- fresh-quant calibration probe (calib.cl, eaq-ldlq-spec-fable.md sections 6 and 7.1):
// the device gates T-CAP-1 (layer-major identity against Runner.prompt + XPU_DUMP_H on the
// pack, proving hazards H1-H8 including the H6 residual pairing), T-CAP-2 (row-order
// independence), T-HESS-1..4 (the Hessian syrk: determinism, fp64 agreement, upper-tile
// untouched, stale-row immunity), and the per-layer timing budget of 7.1. Usage:
//   xpu-qwen27 --calib-probe PACK_DIR IDS_FILE
// IDS_FILE is read by the runner's usual grammar and must hold at least 4096 ids (the two
// T-CAP rows). XPU_CHUNK defaults to 2048 (spec 1.2: one row is one forward); XPU_CTX to 4096.
// Any red gate exits nonzero naming itself.
extern "c" fn creat(path: [*:0]const u8, mode: c_uint) c_int;
const cwrite = @extern(*const fn (fd: c_int, buf: [*]const u8, n: usize) callconv(.c) isize, .{ .name = "write", .linkage = .strong });
const cclose = @extern(*const fn (fd: c_int) callconv(.c) c_int, .{ .name = "close", .linkage = .strong });
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

const calib_mod = @import("calib.zig");
const fwd_mod = @import("forward.zig");

fn calibProbe(argv: []const [:0]const u8) !void {
    if (argv.len < 4) {
        std.debug.print("usage: xpu-qwen27 --calib-probe PACK_DIR IDS_FILE\n", .{});
        return error.CalibUsage;
    }
    if (std.c.getenv("XPU_CHUNK") == null) _ = setenv("XPU_CHUNK", "2048", 0); // one row per forward (1.2)
    if (std.c.getenv("XPU_CTX") == null) _ = setenv("XPU_CTX", "4096", 0);
    var ids: std.ArrayList(u32) = .empty;
    const text = try calibRead(argv[3]);
    var it = std.mem.tokenizeAny(u8, text, " \n,[]");
    while (it.next()) |w| try ids.append(alloc, try std.fmt.parseInt(u32, w, 10));
    if (ids.items.len < 4096) return error.CalibNeedRows; // two T-CAP rows of 2048
    const t_load = q.now();
    const r = try q.Runner.init(argv[2], 4096);
    defer r.deinit();
    std.debug.print("calib: pack loaded in {d:.1} s (chunk {d}, maxRows {d})\n", .{ q.now() - t_load, q.CHUNK, q.maxRows() });
    var fails: u32 = 0;
    if (!try tSlab(r)) fails += 1;
    if (!try tCap1(r, ids.items)) fails += 1;
    if (!try tCap2(r, ids.items)) fails += 1;
    if (!try tHess12(r)) fails += 1;
    if (!try tHess34(r)) fails += 1;
    try timedLayers(r, ids.items);
    std.debug.print("calib probe: {s}\n", .{if (fails == 0) "ALL GATES GREEN" else "GATE FAILURE(S) ABOVE"});
    if (fails != 0) return error.CalibGateFailed;
}

/// Whole file into fresh memory through the C syscall trio (this runner keeps std.fs out
/// of the probe path; the serving reader mmaps the same way).
fn calibRead(path: []const u8) ![]u8 {
    const sent = try alloc.dupeSentinel(u8, path, 0);
    defer alloc.free(sent);
    const fd = open(sent.ptr, 0);
    if (fd < 0) return error.FileNotFound;
    defer _ = cclose(fd);
    const size: usize = @intCast(lseek(fd, 0, 2));
    const map: [*]const u8 = @ptrCast(mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed);
    return alloc.dupe(u8, map[0..size]);
}

/// Device -> host with the queue drained.
fn calibDl(d: *q.Dev, p: *anyopaque, dst: []u8) !void {
    try q.ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dst.ptr, p, dst.len, null, 0, null));
    try d.sync();
}


/// T-SLAB: the dense slab upload round-trip (the review's blocker — a pre-fix kernel stores
/// every 16x16 tile transposed and nothing else in the gates touches uploadDense). A bf16
/// pattern with one distinct value per (n, k) goes up through the kernel and back; the slab
/// must read DST[k * NT + n0 + n] = (half) SRC[n * K + k] exactly, the pad columns the call
/// owns must be zero, nothing outside [n0, n0 + npad) may move, and an n0 that is not a
/// multiple of 16 must be refused (the kernel's own trap would leave the slab stale).
fn tSlab(r: *q.Runner) !bool {
    const d = &r.d;
    const N: u32 = 20;
    const K: u32 = 32;
    const NT: u32 = 128;
    const n0: u32 = 16; // N straddles a tile edge; n0 sits off the slab base
    const npad: u32 = (n0 + N + 15) / 16 * 16 - n0; // 32: the columns this call owns, inside [n0, n0 + npad)
    const src16 = try alloc.alloc(u16, @as(usize, N) * K);
    defer alloc.free(src16);
    for (0 .. @as(usize, N) * K) |i| src16[i] = @intCast((118 + i / 128) << 7 | i % 128); // distinct normal bf16 per element (1+8+7: mantissa is 7 bits, 5 exponents cover 640), ~2^-9..2^-3
    const slab = try d.alloc_(@as(usize, K) * NT * 2);
    defer d.g.free(slab);
    try d.fill(slab, @as(u16, 0), @as(usize, K) * NT * 2);
    try calib_mod.uploadDense(d, std.mem.sliceAsBytes(src16), N, K, NT, n0, slab);
    const got = try alloc.alloc(u16, @as(usize, K) * NT);
    defer alloc.free(got);
    try calibDl(d, slab, std.mem.sliceAsBytes(got));
    var ok = true;
    for (0 .. @as(usize, K) * N) |i| { // the contract, cell by cell; the fp16 oracle is analytic, not the kernel's
        const n = i / K;
        const k = i % K;
        const exp: u16 = @intCast(118 + i / 128);
        const mant: u16 = @intCast(i % 128);
        const want: u16 = @intCast((exp - 127 + 15) << 10 | mant << 3); // the same value in fp16 (1+5+10): normals, the 7-bit mantissa kept exactly
        if (got[k * NT + n0 + n] != want) ok = false;
    }
    for (0 .. @as(usize, K)) |k| { // pad columns zero, and nothing outside the owned range moved
        for (N .. npad) |np| {
            if (got[k * NT + n0 + @as(u32, @intCast(np))] != 0) ok = false;
        }
        for (0 .. n0) |c| {
            if (got[k * NT + c] != 0) ok = false;
        }
        for (n0 + npad .. NT) |c| {
            if (got[k * NT + c] != 0) ok = false;
        }
    }
    var refused = false;
    if (calib_mod.uploadDense(d, std.mem.sliceAsBytes(src16), N, K, NT, n0 + 1, slab)) |_| {} else |e| refused = e == error.CalibShape;
    std.debug.print("T-SLAB uploadDense round-trip N=20 K=32 n0=16 (contract, pads, foreign columns, unaligned-n0 refusal): {s}\n", .{if (ok and refused) "PASS" else "FAIL"});
    return ok and refused;
}

/// The four fused slots of one layer through the serving dequant into the resident slabs.
fn dequantLayer(r: *q.Runner, i: usize, slabs: calib_mod.Slabs) !void {
    const l = &r.model.layers[i];
    for ([_]struct { p: q.Proj, slab: *anyopaque }{
        .{ .p = l.mixer_in, .slab = slabs.mixer_in },
        .{ .p = l.mixer_out, .slab = slabs.mixer_out },
        .{ .p = l.gu, .slab = slabs.gu },
        .{ .p = l.down, .slab = slabs.down },
    }) |s| {
        const hx = s.p.exl3 orelse return error.CalibNoTrellis; // the probe runs on a trellis pack
        try calib_mod.dequantSlot(&r.d, hx, s.p.k, s.p.n, s.slab);
    }
}

/// embedStream + 64 × layerForward (dequantized slabs, fresh per row) + finalNorm, into `out`
/// (host fp16 [T, D] — the hidden the fleet's dump would carry).
fn sweepCalib(r: *q.Runner, slabs: calib_mod.Slabs, row: []const u32, sbuf: *anyopaque, hid: *anyopaque, out: []u8) !void {
    const T: u32 = @intCast(row.len);
    try r.calibEmbed(row, sbuf, 0);
    for (0..q.LAYERS) |i| {
        try dequantLayer(r, i, slabs);
        try r.calibLayer(i, slabs, sbuf, sbuf, T, 0, true, null);
    }
    try r.calibFinalNorm(sbuf, hid, T, null);
    try calibDl(&r.d, hid, out);
}

/// T-CAP-1: the layer-major path against Runner.prompt with XPU_DUMP_H on the pack and row —
/// hidden states bit-identical (H1-H8, and the H6 pairing, are proved together).
fn tCap1(r: *q.Runner, ids: []const u32) !bool {
    const d = &r.d;
    const T: u32 = 2048;
    if (T > q.maxRows()) {
        std.debug.print("T-CAP-1: SKIP (XPU_CHUNK {d} admits no 2048-row forward)\n", .{q.CHUNK});
        return false;
    }
    const path: [*:0]const u8 = "/tmp/calib_serving.f16";
    const fd = creat(path, 0o644);
    if (fd < 0) return error.CalibNoTmp;
    q.dump_h = fd; // the fleet's own per-row hidden dump
    errdefer q.dump_h = null;
    try r.reset();
    try r.prompt(ids[0..T], true);
    q.dump_h = null;
    _ = cclose(fd);
    const srv = try calibRead("/tmp/calib_serving.f16");
    if (srv.len != @as(usize, T) * q.D * 2) return error.CalibDumpSize;
    const cal = try alloc.alloc(u8, srv.len);
    defer alloc.free(cal);
    var slabs = try calib_mod.Slabs.init(d);
    defer slabs.deinit(d);
    const sbuf = try d.alloc_(@as(usize, T) * q.D * 4);
    defer d.g.free(sbuf);
    const hid = try d.alloc_(@as(usize, T) * q.D * 2);
    defer d.g.free(hid);
    try sweepCalib(r, slabs, ids[0..T], sbuf, hid, cal);
    var first: usize = std.math.maxInt(usize);
    if (!std.mem.eql(u8, srv, cal)) for (cal, 0..) |c, k| {
        if (c != srv[k]) {
            first = k;
            break;
        }
    };
    const ok = first == std.math.maxInt(usize);
    std.debug.print("T-CAP-1 layer-major identity vs Runner.prompt+XPU_DUMP_H: {s} ({d} bytes", .{ if (ok) "PASS (bit-identical)" else "FAIL", srv.len });
    if (!ok) std.debug.print(", first diff at byte {d}", .{first});
    std.debug.print(")\n", .{});
    return ok;
}

/// T-CAP-2: two rows through the sweep with fresh = true each equal the same rows run in the
/// opposite order — the per-row state reset (H1-H3, H5) carries nothing across rows.
fn tCap2(r: *q.Runner, ids: []const u32) !bool {
    const d = &r.d;
    const T: u32 = 2048;
    if (T > q.maxRows()) {
        std.debug.print("T-CAP-2: SKIP (XPU_CHUNK {d})\n", .{q.CHUNK});
        return false;
    }
    var slabs = try calib_mod.Slabs.init(d);
    defer slabs.deinit(d);
    const sbuf = try d.alloc_(@as(usize, T) * q.D * 4);
    defer d.g.free(sbuf);
    const hid = try d.alloc_(@as(usize, T) * q.D * 2);
    defer d.g.free(hid);
    const rowa = ids[0..T];
    const rowb = ids[T .. 2 * T];
    const a1 = try alloc.alloc(u8, @as(usize, T) * q.D * 2);
    defer alloc.free(a1);
    const b1 = try alloc.alloc(u8, a1.len);
    defer alloc.free(b1);
    const a2 = try alloc.alloc(u8, a1.len);
    defer alloc.free(a2);
    const b2 = try alloc.alloc(u8, a1.len);
    defer alloc.free(b2);
    try sweepCalib(r, slabs, rowa, sbuf, hid, a1);
    try sweepCalib(r, slabs, rowb, sbuf, hid, b1);
    try sweepCalib(r, slabs, rowb, sbuf, hid, b2);
    try sweepCalib(r, slabs, rowa, sbuf, hid, a2);
    const ok = std.mem.eql(u8, a1, a2) and std.mem.eql(u8, b1, b2);
    std.debug.print("T-CAP-2 row-order independence (fresh rows, {d}x{d}): {s}\n", .{ T, q.D, if (ok) "PASS" else "FAIL" });
    return ok;
}

fn randHalf(rnd: std.Random) f16 {
    return @floatCast(rnd.float(f32) * 2.0 - 1.0);
}


/// One worker's ticket range over the lower tiles of the fp64 oracle (linear tile index:
/// rows bi, then bj <= bi), strided across the pool.
const HessJob = struct {
    x: []const f16,
    ref: []f64,
    tp: usize,
    K: usize,
    first: usize,
    stride: usize,
};

fn hessWorker(job: HessJob) void {
    const K = job.K;
    const tp = job.tp;
    const tiles = K / 128;
    const total = tiles * (tiles + 1) / 2;
    var idx = job.first;
    while (idx < total) : (idx += job.stride) {
        var rem = idx;
        var bi: usize = 0;
        while (rem >= bi + 1) : (bi += 1) rem -= bi + 1;
        const bj = rem; // the linear ticket -> (bi, bj) with bi >= bj
        const r0 = bi * 128;
        const c0 = bj * 128;
        var i = r0;
        while (i < r0 + 128) : (i += 1) {
            var j = c0;
            while (j < c0 + 128) : (j += 1) {
                var acc: f64 = 0;
                for (0..tp) |t| {
                    const r = t * K;
                    acc += @as(f64, @floatCast(job.x[r + i])) * @as(f64, @floatCast(job.x[r + j]));
                }
                job.ref[i * K + j] = acc;
            }
        }
    }
}

const hess_pool_max = 16;

/// T-HESS-1 (capture of 3 rows twice: H bit-identical — no atomics, one owner per tile,
/// fixed kk order, spec 2.3) and T-HESS-2 (T = 1000, K = 5120, random fp16 X, against the
/// host fp64 X^T X: the max error over the lower tiles below 1e-5 relative to their RMS;
/// the staging+syrk GFLOP/s is logged for the 7.1 Hessian budget).
fn tHess12(r: *q.Runner) !bool {
    const d = &r.d;
    const a = &r.acts;
    const K: usize = q.D;
    const tiles = K / 128;
    const hbytes = K * K * 4;
    var prng = std.Random.DefaultPrng.init(0xC41B);
    const rnd = prng.random();

    // T-HESS-1
    const T1: u32 = 3;
    const x1 = try alloc.alloc(f16, T1 * K);
    defer alloc.free(x1);
    for (x1) |*v| v.* = randHalf(rnd);
    try d.copyIn(a.h, std.mem.sliceAsBytes(x1));
    var cap = try calib_mod.initCapture(d, 64);
    defer calib_mod.deinitCapture(&cap, d);
    const hA = try d.alloc_(hbytes);
    defer d.g.free(hA);
    const hB = try d.alloc_(hbytes);
    defer d.g.free(hB);
    try d.fill(hA, @as(f32, 0), hbytes);
    try d.fill(hB, @as(f32, 0), hbytes);
    cap.h[0] = hA;
    try fwd_mod.capture(d, &cap, 0, K, a.h, T1); // the same 3 rows, twice, into two clean H's
    cap.h[0] = hB;
    try fwd_mod.capture(d, &cap, 0, K, a.h, T1);
    const A = try alloc.alloc(u8, hbytes);
    defer alloc.free(A);
    const B = try alloc.alloc(u8, hbytes);
    defer alloc.free(B);
    try calibDl(d, hA, A);
    try calibDl(d, hB, B);
    const ok1 = std.mem.eql(u8, A, B) and cap.rows[0] == 2 * T1;
    std.debug.print("T-HESS-1 capture of 3 rows twice, H bit-identical: {s} (rows[0]={d})\n", .{ if (ok1) "PASS" else "FAIL", cap.rows[0] });

    // T-HESS-2
    const T2: u32 = 1000;
    const Tp: u32 = (T2 + 63) / 64 * 64;
    const x2 = try alloc.alloc(f16, T2 * K);
    defer alloc.free(x2);
    for (x2) |*v| v.* = randHalf(rnd);
    try d.copyIn(a.h, std.mem.sliceAsBytes(x2));
    var cap2 = try calib_mod.initCapture(d, Tp);
    defer calib_mod.deinitCapture(&cap2, d);
    const h = try d.alloc_(hbytes);
    defer d.g.free(h);
    try d.fill(h, @as(f32, 0), hbytes);
    cap2.h[0] = h;
    const t0 = q.now();
    try fwd_mod.capture(d, &cap2, 0, K, a.h, T2); // staging copy + zero-fill + transpose + syrk
    try d.sync();
    const t_cap = q.now() - t0;
    const dev_h = try alloc.alloc(f32, K * K);
    defer alloc.free(dev_h);
    try calibDl(d, h, std.mem.sliceAsBytes(dev_h));
    const ref = try alloc.alloc(f64, K * K);
    defer alloc.free(ref);
    const ncpu = @max(1, std.Thread.getCpuCount() catch 1);
    const nw = @min(hess_pool_max, ncpu);
    var ths: [hess_pool_max]std.Thread = undefined;
    for (0..nw) |w| ths[w] = try std.Thread.spawn(.{}, hessWorker, .{HessJob{
        .x = x2, .ref = ref, .tp = @intCast(T2), .K = K, .first = w, .stride = nw,
    }}); // the oracle sums the T2 real rows only: the staging's Tp - T2 pad rows are exact zeros, so they add nothing to H
    for (ths[0..nw]) |t| t.join();
    var nz_dev: usize = 0;
    for (0..tiles) |bi| for (0..bi + 1) |bj| for (bi * 128..bi * 128 + 128) |i| for (bj * 128..bj * 128 + 128) |j| {
        if (dev_h[i * K + j] != 0) nz_dev += 1;
    };
    var maxerr: f64 = 0;
    var maxabs: f64 = 0; // the largest reference entry: the accumulation scale (dominant diagonal)
    for (0..tiles) |bi| for (0..bi + 1) |bj| {
        for (bi * 128..bi * 128 + 128) |i| for (bj * 128..bj * 128 + 128) |j| {
            const rr = ref[i * K + j];
            const e = @abs(@as(f64, dev_h[i * K + j]) - rr);
            if (e > maxerr) maxerr = e;
            const ar = @abs(rr);
            if (ar > maxabs) maxabs = ar;
        };
    };
    const rel = maxerr / @max(maxabs, 1e-30);
    const ok2 = rel < 1e-5; // max relative error on the lower tiles below 1e-5 (hand-derived bound 3e-6, spec 2.3)
    const gflops = @as(f64, @floatFromInt(@as(usize, tiles) * (tiles + 1) / 2 * 128 * 128 * Tp * 2)) / t_cap / 1e9;
    std.debug.print("T-HESS-2 T=1000 K=5120 vs host fp64: {s} (max |d-r| {d:.2}e-6, max |ref| {d:.1}, rel {d:.3}e-6 vs gate 1e-5; dev lower nonzero {d}/{d}; capture {d:.1} ms, {d:.0} GFLOP/s)\n", .{ if (ok2) "PASS" else "FAIL", maxerr * 1e6, maxabs, rel * 1e6, nz_dev, @as(usize, tiles) * (tiles + 1) / 2 * 128 * 128, t_cap * 1e3, gflops });
    return ok1 and ok2;
}

const nan32: u32 = 0x7FC0_0001; // a quiet NaN with a payload the syrk could never produce
const nan16: u16 = 0x7E01; // its fp16 kin, for the stale acts rows

/// T-HESS-3: upper 128x128 tiles pre-filled with NaN stay NaN (the kernel exits before
/// touching them) and the host mirror of the lower tiles is exactly symmetric; T-HESS-4:
/// Acts rows T..Tp filled with NaN before the hook leave H finite (the staging copies T
/// rows and zero-fills Tp - T, and transpose16 reads no row past T).
fn tHess34(r: *q.Runner) !bool {
    const d = &r.d;
    const a = &r.acts;
    const K: usize = q.D;
    const tiles = K / 128;
    const hbytes = K * K * 4;
    const T: u32 = 3;
    var prng = std.Random.DefaultPrng.init(0xA354);
    const rnd = prng.random();
    const x = try alloc.alloc(f16, T * K);
    defer alloc.free(x);
    for (x) |*v| v.* = randHalf(rnd);
    try d.copyIn(a.h, std.mem.sliceAsBytes(x));

    // T-HESS-3
    const pat = try alloc.alloc(u32, K * K);
    defer alloc.free(pat);
    for (0..tiles) |bi| for (0..tiles) |bj| {
        const v: u32 = if (bi < bj) nan32 else 0;
        for (bi * 128..bi * 128 + 128) |i| for (bj * 128..bj * 128 + 128) |j| { pat[i * K + j] = v; };
    };
    const h = try d.alloc_(hbytes);
    defer d.g.free(h);
    try d.copyIn(h, std.mem.sliceAsBytes(pat));
    var cap = try calib_mod.initCapture(d, 64);
    defer calib_mod.deinitCapture(&cap, d);
    cap.h[0] = h;
    try fwd_mod.capture(d, &cap, 0, K, a.h, T);
    const dev_h = try alloc.alloc(u32, K * K);
    defer alloc.free(dev_h);
    try calibDl(d, h, std.mem.sliceAsBytes(dev_h));
    var ok3 = true;
    for (0..tiles) |bi| for (0..tiles) |bj| {
        for (bi * 128..bi * 128 + 128) |i| for (bj * 128..bj * 128 + 128) |j| {
            if (bi < bj) { // upper: planted bits must survive untouched
                if (dev_h[i * K + j] != nan32) ok3 = false;
            } else if (dev_h[i * K + j] & 0x7F80_0000 == 0x7F80_0000) ok3 = false; // lower: finite
        };
    };
    const mir = try alloc.alloc(f64, K * K);
    defer alloc.free(mir);
    for (0..tiles) |bi| for (0..bi + 1) |bj| for (bi * 128..bi * 128 + 128) |i| for (bj * 128..bj * 128 + 128) |j| {
        const v: f64 = @floatCast(@as(f32, @bitCast(dev_h[i * K + j])));
        mir[i * K + j] = v;
        mir[j * K + i] = v;
    };
    for (0..tiles) |bi| for (0..tiles) |bj| {
        for (bi * 128..bi * 128 + 128) |i| for (bj * 128..bj * 128 + 128) |j| {
            if (@as(u64, @bitCast(mir[i * K + j])) != @as(u64, @bitCast(mir[j * K + i]))) ok3 = false;
        };
    };
    std.debug.print("T-HESS-3 NaN-planted upper tiles untouched, mirror exactly symmetric: {s}\n", .{if (ok3) "PASS" else "FAIL"});

    // T-HESS-4: rows T..Tp of the acts input hold NaN fp16 before the hook
    const Tp: usize = 64;
    const x4 = try alloc.alloc(f16, Tp * K);
    defer alloc.free(x4);
    for (0..Tp) |t| for (0..K) |kx| {
        x4[t * K + kx] = if (t < T) randHalf(rnd) else @bitCast(nan16);
    };
    try d.copyIn(a.h, std.mem.sliceAsBytes(x4));
    try d.fill(h, @as(f32, 0), hbytes);
    cap.h[0] = h;
    try fwd_mod.capture(d, &cap, 0, K, a.h, T);
    try calibDl(d, h, std.mem.sliceAsBytes(dev_h));
    var ok4 = true;
    for (0..tiles) |bi| for (0..bi + 1) |bj| for (bi * 128..bi * 128 + 128) |i| for (bj * 128..bj * 128 + 128) |j| {
        if (dev_h[i * K + j] & 0x7F80_0000 == 0x7F80_0000) ok4 = false;
    };
    std.debug.print("T-HESS-4 stale rows T..Tp planted with NaN, H finite: {s}\n", .{if (ok4) "PASS" else "FAIL"});
    return ok3 and ok4;
}

/// 7.1 item 1: layerForward at T = 2048 with dense slabs, one GDN and one attention layer;
/// the per-layer capture-pass budget (100 rows of the calibration set) follows.
fn timedLayers(r: *q.Runner, ids: []const u32) !void {
    const d = &r.d;
    const T: u32 = 2048;
    if (T > q.maxRows()) {
        std.debug.print("7.1 timings: SKIP (XPU_CHUNK {d} admits no 2048-row forward)\n", .{q.CHUNK});
        return;
    }
    var slabs = try calib_mod.Slabs.init(d);
    defer slabs.deinit(d);
    const sbuf = try d.alloc_(@as(usize, T) * q.D * 4);
    defer d.g.free(sbuf);
    try r.calibEmbed(ids[0..T], sbuf, 0);
    for ([_]struct { i: usize, kind: []const u8 }{
        .{ .i = 0, .kind = "GDN" },
        .{ .i = 3, .kind = "attention" },
    }) |t| {
        try dequantLayer(r, t.i, slabs);
        var best: f64 = 1e9;
        for (0..3) |_| {
            const t0 = q.now();
            try r.calibLayer(t.i, slabs, sbuf, sbuf, T, 0, true, null);
            try d.sync();
            best = @min(best, q.now() - t0);
        }
        std.debug.print("7.1 layerForward T=2048 L{d} ({s}): {d:.1} ms; capture pass (100 rows) ~{d:.1} s, capture+apply ~{d:.1} s per layer\n", .{ t.i, t.kind, best * 1e3, best * 100.0, 2.0 * best * 100.0 });
    }
}
