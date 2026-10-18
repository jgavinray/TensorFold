//! TEMP probe (throwaway): mlp_act + qmm (down) vs the fused mlp_act_down, bit-for-bit + timing.
//! Not a gate; delete after the S02 investigation.
const std = @import("std");
const rt = @import("xpu/runtime.zig");
const sys = @import("xpu/sys.zig");
const cfg = @import("families/qwen3_5/config.zig");
const xw = @import("families/qwen3_5/xpu_weights.zig");
const geo = @import("families/qwen3_5/xpu_geometry.zig");
const kbins = @import("xpu_qwen35_kernels");

const gpa = std.heap.page_allocator;
const Buffer = rt.memory.Buffer;

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    const dir: []const u8 = if (argv.len > 1) argv[1] else "/storage01/models/Qwen3.5-2B-MLX-4bit";

    var s: rt.session.Session = undefined;
    s.open(.{}) catch |e| {
        std.debug.print("cannot open a card: {s}\n", .{@errorName(e)});
        return 1;
    };
    defer s.close();
    std.debug.print("card {s}\n", .{s.bdf()});

    var cb: [1024]u8 = undefined;
    const cpath = sys.zfmt(&cb, "{s}/config.json", .{dir}) orelse return 1;
    const ctext = sys.readFileAlloc(gpa, cpath, 1 << 22) orelse return 1;
    defer gpa.free(ctext);
    const config = try cfg.parse(gpa, ctext);
    var w = try xw.load(gpa, &s, dir, config.tied);
    defer w.deinit();

    var l1: rt.module.BuildLog = .{};
    var m_act = try rt.module.Module.load(&s.ctx, &kbins.mlp_act, &l1);
    defer m_act.unload();
    var k_act = try m_act.kernel("qwen35_mlp_act");
    defer k_act.deinit();
    var m_qmm = try rt.module.Module.load(&s.ctx, &kbins.qmm, &l1);
    defer m_qmm.unload();
    var k_qmm = try m_qmm.kernel("qwen35_qmm");
    defer k_qmm.deinit();
    var m_ad = try rt.module.Module.load(&s.ctx, &kbins.mlp_act_down, &l1);
    defer m_ad.unload();
    var k_ad = try m_ad.kernel("qwen35_mlp_act_down");
    defer k_ad.deinit();

    const inter: u32 = cfg.intermediate;
    const blk = w.blocks[0];
    const lin = blk.down; // N = hidden, K = intermediate
    const N: u32 = @intCast(lin.outputs);
    const K: u32 = @intCast(lin.inputs);
    std.debug.print("down projection N={d} K={d} (intermediate {d})\n", .{ N, K, inter });
    if (K != inter) {
        std.debug.print("K {d} != QW_INTERMEDIATE {d}\n", .{ K, inter });
        return 1;
    }

    var gate = try Buffer.alloc(&s.ctx, .device, inter * 2);
    defer gate.free();
    var up = try Buffer.alloc(&s.ctx, .device, inter * 2);
    defer up.free();
    var act = try Buffer.alloc(&s.ctx, .device, inter * 2);
    defer act.free();
    var out_ref = try Buffer.alloc(&s.ctx, .device, N * 2);
    defer out_ref.free();
    var out_fus = try Buffer.alloc(&s.ctx, .device, N * 2);
    defer out_fus.free();

    {
        var upb = try Buffer.alloc(&s.ctx, .host, inter * 2);
        defer upb.free();
        const ub = try upb.bytes();
        var prng = std.Random.DefaultPrng.init(0xabcdef);
        const rnd = prng.random();
        var i: usize = 0;
        while (i < inter) : (i += 1) {
            const v: f32 = (rnd.float(f32) - 0.5) * 8.0;
            const u: u32 = @bitCast(v);
            std.mem.writeInt(u16, ub[i * 2 ..][0..2], @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16), .little);
        }
        try gate.upload(&s.queue, 0, ub);
        i = 0;
        while (i < inter) : (i += 1) {
            const v: f32 = (rnd.float(f32) - 0.5) * 8.0;
            const u: u32 = @bitCast(v);
            std.mem.writeInt(u16, ub[i * 2 ..][0..2], @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16), .little);
        }
        try up.upload(&s.queue, 0, ub);
    }

    // Reference: mlp_act then qmm.
    {
        const ga = geo.mlpAct(1, inter);
        try rt.launch.run(&s.queue, &k_act, .{ .global = .{ .x = ga.gx, .y = ga.gy }, .local = .{ .x = ga.lx } }, .{ gate.ptr, up.ptr, act.ptr }, null);
        const gq = geo.qmm(1, N);
        try rt.launch.run(&s.queue, &k_qmm, .{ .global = .{ .x = gq.gx, .y = gq.gy }, .local = .{ .x = gq.lx } }, .{
            act.ptr, @as(i32, 1), w.ptr(lin.weight), w.ptr(lin.scales), w.ptr(lin.biases), out_ref.ptr, @as(i32, @intCast(N)), @as(i32, @intCast(K)),
        }, null);
        try s.queue.synchronize();
    }
    // Fused.
    {
        const gq = geo.qmm(1, N);
        try rt.launch.run(&s.queue, &k_ad, .{ .global = .{ .x = gq.gx, .y = gq.gy }, .local = .{ .x = gq.lx } }, .{
            gate.ptr, up.ptr, w.ptr(lin.weight), w.ptr(lin.scales), w.ptr(lin.biases), out_fus.ptr, @as(i32, 1), @as(i32, @intCast(N)), @as(i32, @intCast(K)),
        }, null);
        try s.queue.synchronize();
    }

    var oa = try gpa.alloc(u8, N * 2);
    defer gpa.free(oa);
    var ob = try gpa.alloc(u8, N * 2);
    defer gpa.free(ob);
    try out_ref.download(&s.queue, 0, oa);
    try out_fus.download(&s.queue, 0, ob);
    var diff: usize = 0;
    var maxd: u16 = 0;
    var first: ?usize = null;
    for (0..N) |i| {
        const x = std.mem.readInt(u16, oa[i * 2 ..][0..2], .little);
        const y = std.mem.readInt(u16, ob[i * 2 ..][0..2], .little);
        if (x != y) {
            diff += 1;
            if (first == null) first = i;
            const d = if (x > y) x - y else y - x;
            if (d > maxd) maxd = d;
        }
    }
    std.debug.print("OUT: {d}/{d} bf16 words differ (max ulp delta {d}), first at {?}\n", .{ diff, N, maxd, first });
    if (first) |i| {
        std.debug.print("  col {d}: ref 0x{x:0>4} fused 0x{x:0>4}\n", .{ i, std.mem.readInt(u16, oa[i * 2 ..][0..2], .little), std.mem.readInt(u16, ob[i * 2 ..][0..2], .little) });
    }

    // Timing: act+qmm vs fused, both batched.
    {
        const ga = geo.mlpAct(1, inter);
        const gq = geo.qmm(1, N);
        const acfg = rt.launch.Config{ .global = .{ .x = ga.gx, .y = ga.gy }, .local = .{ .x = ga.lx } };
        const qcfg = rt.launch.Config{ .global = .{ .x = gq.gx, .y = gq.gy }, .local = .{ .x = gq.lx } };
        var best2: u64 = std.math.maxInt(u64);
        var best1: u64 = std.math.maxInt(u64);
        for (0..20) |_| {
            const t0 = sys.nowNs();
            try rt.launch.run(&s.queue, &k_act, acfg, .{ gate.ptr, up.ptr, act.ptr }, null);
            try rt.launch.run(&s.queue, &k_qmm, qcfg, .{ act.ptr, @as(i32, 1), w.ptr(lin.weight), w.ptr(lin.scales), w.ptr(lin.biases), out_ref.ptr, @as(i32, @intCast(N)), @as(i32, @intCast(K)) }, null);
            try s.queue.synchronize();
            const d = sys.nowNs() - t0;
            if (d < best2) best2 = d;

            const t1 = sys.nowNs();
            try rt.launch.run(&s.queue, &k_ad, qcfg, .{ gate.ptr, up.ptr, w.ptr(lin.weight), w.ptr(lin.scales), w.ptr(lin.biases), out_fus.ptr, @as(i32, 1), @as(i32, @intCast(N)), @as(i32, @intCast(K)) }, null);
            try s.queue.synchronize();
            const d1 = sys.nowNs() - t1;
            if (d1 < best1) best1 = d1;
        }
        std.debug.print("timing: act+qmm best {d:.1} us, fused best {d:.1} us, saving {d:.1} us\n", .{
            @as(f64, @floatFromInt(best2)) / 1e3, @as(f64, @floatFromInt(best1)) / 1e3, (@as(f64, @floatFromInt(best2)) - @as(f64, @floatFromInt(best1))) / 1e3,
        });
    }
    return 0;
}
