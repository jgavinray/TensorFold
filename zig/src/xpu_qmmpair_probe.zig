//! TEMP probe (throwaway): qwen35_qmm_pair vs two qwen35_qmm calls on the real a/b weights, bit-for-bit.
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

    var log: rt.module.BuildLog = .{};
    var m_qmm = try rt.module.Module.load(&s.ctx, &kbins.qmm, &log);
    defer m_qmm.unload();
    var m_pair = try rt.module.Module.load(&s.ctx, &kbins.qmm_pair, &log);
    defer m_pair.unload();
    var k_qmm = try m_qmm.kernel("qwen35_qmm");
    defer k_qmm.deinit();
    var k_pair = try m_pair.kernel("qwen35_qmm_pair");
    defer k_pair.deinit();

    const h: u32 = cfg.hidden;
    const n: u32 = cfg.linear_heads; // a/b: 16
    const nkv: u32 = cfg.kv_heads * cfg.head_dim; // k/v: 512
    const d = w.blocks[0].mixer.delta;

    // A realistic x row (bf16 random).
    var x = try Buffer.alloc(&s.ctx, .device, h * 2);
    defer x.free();
    {
        var hb = try Buffer.alloc(&s.ctx, .host, h * 2);
        defer hb.free();
        const ub = try hb.bytes();
        var prng = std.Random.DefaultPrng.init(0x5eed);
        const rnd = prng.random();
        for (0..h) |i| {
            const v: f32 = (rnd.float(f32) - 0.5) * 6.0;
            const u: u32 = @bitCast(v);
            std.mem.writeInt(u16, ub[i * 2 ..][0..2], @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16), .little);
        }
        try x.upload(&s.queue, 0, ub);
    }

    var ref_a = try Buffer.alloc(&s.ctx, .device, nkv * 2);
    defer ref_a.free();
    var ref_b = try Buffer.alloc(&s.ctx, .device, nkv * 2);
    defer ref_b.free();
    var fus_a = try Buffer.alloc(&s.ctx, .device, nkv * 2);
    defer fus_a.free();
    var fus_b = try Buffer.alloc(&s.ctx, .device, nkv * 2);
    defer fus_b.free();

    var pool = try rt.event.Pool.init(&s.ctx, 4);
    defer pool.deinit();
    var ev0 = try pool.event(0);
    defer ev0.deinit();
    var ev1 = try pool.event(1);
    defer ev1.deinit();

    const attn = w.blocks[3].mixer.attention; // block 3 is a full-attention layer (layer_types[3])
    for ([_]struct { name: []const u8, la: xw.Linear, lb: xw.Linear, N: u32 }{
        .{ .name = "a/b", .la = d.a, .lb = d.b, .N = n },
        .{ .name = "k/v", .la = attn.k, .lb = attn.v, .N = nkv },
    }) |pair| {
        const N: i32 = @intCast(pair.N);
        const gq = geo.qmm(1, pair.N);
        const cq = rt.launch.Config{ .global = .{ .x = gq.gx, .y = gq.gy }, .local = .{ .x = gq.lx } };
        const gp = geo.qmmPair(1, pair.N);
        const cp = rt.launch.Config{ .global = .{ .x = gp.gx, .y = gp.gy }, .local = .{ .x = gp.lx } };
        const la = pair.la;
        const lb = pair.lb;

        // Reference: two separate qmm launches.
        try rt.launch.run(&s.queue, &k_qmm, cq, .{ x.ptr, @as(i32, 1), w.ptr(la.weight), w.ptr(la.scales), w.ptr(la.biases), ref_a.ptr, N, @as(i32, h) }, null);
        try rt.launch.run(&s.queue, &k_qmm, cq, .{ x.ptr, @as(i32, 1), w.ptr(lb.weight), w.ptr(lb.scales), w.ptr(lb.biases), ref_b.ptr, N, @as(i32, h) }, null);
        // Fused pair.
        try rt.launch.run(&s.queue, &k_pair, cp, .{
            x.ptr,         @as(i32, 1),
            w.ptr(la.weight), w.ptr(la.scales), w.ptr(la.biases),
            w.ptr(lb.weight), w.ptr(lb.scales), w.ptr(lb.biases),
            fus_a.ptr,     fus_b.ptr,
            N,             @as(i32, h),
        }, null);
        try s.queue.synchronize();

        const oa = try gpa.alloc(u8, pair.N * 2);
        defer gpa.free(oa);
        const ob = try gpa.alloc(u8, pair.N * 2);
        defer gpa.free(ob);
        const fa = try gpa.alloc(u8, pair.N * 2);
        defer gpa.free(fa);
        const fb = try gpa.alloc(u8, pair.N * 2);
        defer gpa.free(fb);
        try ref_a.download(&s.queue, 0, oa);
        try ref_b.download(&s.queue, 0, ob);
        try fus_a.download(&s.queue, 0, fa);
        try fus_b.download(&s.queue, 0, fb);

        for ([_]struct { nm: []const u8, r: []const u8, f: []const u8 }{
            .{ .nm = "A", .r = oa, .f = fa },
            .{ .nm = "B", .r = ob, .f = fb },
        }) |c| {
            var diff: usize = 0;
            var maxd: u16 = 0;
            var first: ?usize = null;
            for (0..pair.N) |i| {
                const xr = std.mem.readInt(u16, c.r[i * 2 ..][0..2], .little);
                const xf = std.mem.readInt(u16, c.f[i * 2 ..][0..2], .little);
                if (xr != xf) {
                    diff += 1;
                    if (first == null) first = i;
                    const dd = if (xr > xf) xr - xf else xf - xr;
                    if (dd > maxd) maxd = dd;
                }
            }
            std.debug.print("{s} {s}: {d}/{d} bf16 words differ (max ulp delta {d}), first at {?}\n", .{ pair.name, c.nm, diff, pair.N, maxd, first });
        }

        // Timing: two qmm launches vs one pair launch (device time, batched).
        var best_two: f64 = std.math.inf(f64);
        var best_one: f64 = std.math.inf(f64);
        for (0..50) |_| {
            try rt.launch.run(&s.queue, &k_qmm, cq, .{ x.ptr, @as(i32, 1), w.ptr(la.weight), w.ptr(la.scales), w.ptr(la.biases), ref_a.ptr, N, @as(i32, h) }, &ev0);
            try rt.launch.run(&s.queue, &k_qmm, cq, .{ x.ptr, @as(i32, 1), w.ptr(lb.weight), w.ptr(lb.scales), w.ptr(lb.biases), ref_b.ptr, N, @as(i32, h) }, &ev1);
            try s.queue.synchronize();
            const t2 = try ev0.kernelNs() + try ev1.kernelNs();
            try ev0.reset();
            try ev1.reset();
            if (t2 < best_two) best_two = t2;

            try rt.launch.run(&s.queue, &k_pair, cp, .{
                x.ptr,         @as(i32, 1),
                w.ptr(la.weight), w.ptr(la.scales), w.ptr(la.biases),
                w.ptr(lb.weight), w.ptr(lb.scales), w.ptr(lb.biases),
                fus_a.ptr,     fus_b.ptr,
                N,             @as(i32, h),
            }, &ev0);
            try s.queue.synchronize();
            const t1 = try ev0.kernelNs();
            try ev0.reset();
            if (t1 < best_one) best_one = t1;
        }
        std.debug.print("{s}: two qmm launches {d:.2} us, one pair launch {d:.2} us, saving {d:.2} us\n", .{
            pair.name, best_two / 1e3, best_one / 1e3, (best_two - best_one) / 1e3,
        });
    }
    return 0;
}
