//! TEMP probe (throwaway): compare fused norm_qmm against norm+qmm on real 2B weights, bit-for-bit.
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
const Dim = rt.launch.Dim3;

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

    // Modules: norm, qmm, norm_qmm.
    var l1: rt.module.BuildLog = .{};
    var m_norm = try rt.module.Module.load(&s.ctx, &kbins.norm, &l1);
    defer m_norm.unload();
    var k_norm = try m_norm.kernel("qwen35_norm");
    defer k_norm.deinit();
    var m_qmm = try rt.module.Module.load(&s.ctx, &kbins.qmm, &l1);
    defer m_qmm.unload();
    var k_qmm = try m_qmm.kernel("qwen35_qmm");
    defer k_qmm.deinit();
    var m_fused = try rt.module.Module.load(&s.ctx, &kbins.norm_qmm, &l1);
    defer m_fused.unload();
    var k_fused = try m_fused.kernel("qwen35_norm_qmm");
    defer k_fused.deinit();
    var m_act = try rt.module.Module.load(&s.ctx, &kbins.mlp_act, &l1);
    defer m_act.unload();
    var k_act = try m_act.kernel("qwen35_mlp_act");
    defer k_act.deinit();
    var m_ad = try rt.module.Module.load(&s.ctx, &kbins.mlp_act_down, &l1);
    defer m_ad.unload();
    var k_ad = try m_ad.kernel("qwen35_mlp_act_down");
    defer k_ad.deinit();

    const h: u32 = cfg.hidden;
    // Use a real layer-0 projection: post_attention_layernorm -> mlp.gate_proj.
    const blk = w.blocks[0];
    const norm_w = w.ptr(blk.post_norm);
    const lin = blk.gate;
    const N: u32 = if (argv.len > 2) try std.fmt.parseInt(u32, argv[2], 10) else @intCast(lin.outputs);
    const K: u32 = @intCast(lin.inputs);
    std.debug.print("projection N={d} K={d}\n", .{ N, K });
    // For an N smaller than the real projection, use a dummy weight/scale/bias so the geometry is still valid.
    const dummy_w = N != @as(u32, @intCast(lin.outputs));
    var dw = try Buffer.alloc(&s.ctx, .device, @as(usize, N) * (K / 8) * 4);
    defer dw.free();
    var dsc = try Buffer.alloc(&s.ctx, .device, @as(usize, N) * (K / 64) * 4);
    defer dsc.free();
    const wptr: *anyopaque = if (dummy_w) dw.ptr.? else w.ptr(lin.weight);
    const scptr: *anyopaque = if (dummy_w) dsc.ptr.? else w.ptr(lin.scales);
    const biptr: *anyopaque = if (dummy_w) dsc.ptr.? else w.ptr(lin.biases);

    // Inputs: h and r (both bf16, hidden each), one row.
    var hin = try Buffer.alloc(&s.ctx, .device, h * 2);
    defer hin.free();
    var rin = try Buffer.alloc(&s.ctx, .device, h * 2);
    defer rin.free();
    var ho = try Buffer.alloc(&s.ctx, .device, h * 2);
    defer ho.free();
    var xo = try Buffer.alloc(&s.ctx, .device, h * 2);
    defer xo.free();
    var out_ref = try Buffer.alloc(&s.ctx, .device, N * 2);
    defer out_ref.free();
    var out_fus = try Buffer.alloc(&s.ctx, .device, N * 2);
    defer out_fus.free();

    // Deterministic pseudo-random bf16 inputs (finite, moderate range).
    {
        var up = try Buffer.alloc(&s.ctx, .host, h * 2);
        defer up.free();
        const ub = try up.bytes();
        var prng = std.Random.DefaultPrng.init(0xfeed);
        const rnd = prng.random();
        var i: usize = 0;
        while (i < h) : (i += 1) {
            const v: f32 = (rnd.float(f32) - 0.5) * 4.0;
            const u: u32 = @bitCast(v);
            const b: u16 = @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
            std.mem.writeInt(u16, ub[i * 2 ..][0..2], b, .little);
        }
        try hin.upload(&s.queue, 0, ub);
        // Second independent row.
        i = 0;
        while (i < h) : (i += 1) {
            const v: f32 = (rnd.float(f32) - 0.5) * 4.0;
            const u: u32 = @bitCast(v);
            const b: u16 = @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
            std.mem.writeInt(u16, ub[i * 2 ..][0..2], b, .little);
        }
        try rin.upload(&s.queue, 0, ub);
    }

    // Reference: norm then qmm.
    {
        const g = geo.norm(1, h);
        try rt.launch.run(&s.queue, &k_norm, .{ .global = .{ .x = g.gx, .y = g.gy }, .local = .{ .x = g.lx } }, .{
            hin.ptr, rin.ptr, norm_w, cfg.eps, ho.ptr, xo.ptr,
        }, null);
        const gq = geo.qmm(1, N);
        try rt.launch.run(&s.queue, &k_qmm, .{ .global = .{ .x = gq.gx, .y = gq.gy }, .local = .{ .x = gq.lx } }, .{
            xo.ptr, @as(i32, 1), wptr, scptr, biptr, out_ref.ptr, @as(i32, @intCast(N)), @as(i32, @intCast(K)),
        }, null);
        try s.queue.synchronize();
    }

    // Fused: one launch. HO and OUT.
    {
        var ho2 = try Buffer.alloc(&s.ctx, .device, h * 2);
        defer ho2.free();
        const gf = geo.qmm(1, N);
        try rt.launch.run(&s.queue, &k_fused, .{ .global = .{ .x = gf.gx, .y = gf.gy }, .local = .{ .x = gf.lx } }, .{
            hin.ptr, rin.ptr, norm_w, cfg.eps, ho2.ptr,
            wptr, scptr, biptr, out_fus.ptr, @as(i32, 1), @as(i32, @intCast(N)), @as(i32, @intCast(K)),
        }, null);
        try s.queue.synchronize();

        // Compare HO.
        var a = try gpa.alloc(u8, h * 2);
        defer gpa.free(a);
        var b2 = try gpa.alloc(u8, h * 2);
        defer gpa.free(b2);
        try ho.download(&s.queue, 0, a);
        try ho2.download(&s.queue, 0, b2);
        var ho_diff: usize = 0;
        for (0..h) |i| {
            const x = std.mem.readInt(u16, a[i * 2 ..][0..2], .little);
            const y = std.mem.readInt(u16, b2[i * 2 ..][0..2], .little);
            if (x != y) ho_diff += 1;
        }
        std.debug.print("HO: {d}/{d} bf16 words differ\n", .{ ho_diff, h });
    }

    // Compare OUT.
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
        const x = std.mem.readInt(u16, oa[i * 2 ..][0..2], .little);
        const y = std.mem.readInt(u16, ob[i * 2 ..][0..2], .little);
        std.debug.print("  col {d}: ref 0x{x:0>4} fused 0x{x:0>4}\n", .{ i, x, y });
    }

    // Timing: norm+qmm (two launches) vs fused (one launch), batched as the forward does.
    {
        const g = geo.norm(1, h);
        const gq = geo.qmm(1, N);
        const ncfg = rt.launch.Config{ .global = .{ .x = g.gx, .y = g.gy }, .local = .{ .x = g.lx } };
        const qcfg = rt.launch.Config{ .global = .{ .x = gq.gx, .y = gq.gy }, .local = .{ .x = gq.lx } };
        var best2: u64 = std.math.maxInt(u64);
        var best1: u64 = std.math.maxInt(u64);
        for (0..10) |_| {
            const t0 = sys.nowNs();
            try rt.launch.run(&s.queue, &k_norm, ncfg, .{ hin.ptr, rin.ptr, norm_w, cfg.eps, ho.ptr, xo.ptr }, null);
            try rt.launch.run(&s.queue, &k_qmm, qcfg, .{ xo.ptr, @as(i32,1), wptr, scptr, biptr, out_ref.ptr, @as(i32,@intCast(N)), @as(i32,@intCast(K)) }, null);
            try s.queue.synchronize();
            const d = sys.nowNs() - t0;
            if (d < best2) best2 = d;

            const t1 = sys.nowNs();
            try rt.launch.run(&s.queue, &k_fused, qcfg, .{ hin.ptr, rin.ptr, norm_w, cfg.eps, ho.ptr, wptr, scptr, biptr, out_fus.ptr, @as(i32,1), @as(i32,@intCast(N)), @as(i32,@intCast(K)) }, null);
            try s.queue.synchronize();
            const d1 = sys.nowNs() - t1;
            if (d1 < best1) best1 = d1;
        }
        std.debug.print("timing: norm+qmm best {d:.1} us, fused best {d:.1} us, saving {d:.1} us\n", .{@as(f64,@floatFromInt(best2))/1e3, @as(f64,@floatFromInt(best1))/1e3, (@as(f64,@floatFromInt(best2))-@as(f64,@floatFromInt(best1)))/1e3});
    }
    return 0;
}
