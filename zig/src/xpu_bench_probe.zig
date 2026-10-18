//! TEMP bench probe (throwaway): time qwen35_qmm at each 2B decode shape, R=1, to separate launch
//! overhead from weight-bandwidth time. Not a gate; delete after the S02 investigation.
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

fn now() u64 {
    return sys.nowNs();
}

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

    // Load the qmm module and its kernel.
    var log: rt.module.BuildLog = .{};
    var mod = try rt.module.Module.load(&s.ctx, &kbins.qmm, &log);
    defer mod.unload();
    var k = try mod.kernel("qwen35_qmm");
    defer k.deinit();

    const h: u32 = cfg.hidden;
    const inter: u32 = cfg.intermediate;
    const shapes = [_]struct { name: []const u8, N: u32, K: u32 }{
        .{ .name = "delta.qkv", .N = 4096, .K = h },
        .{ .name = "delta.z", .N = 2048, .K = h },
        .{ .name = "mlp.gate", .N = inter, .K = h },
        .{ .name = "mlp.down", .N = h, .K = inter },
        .{ .name = "attn.k", .N = 512, .K = h },
        .{ .name = "head", .N = cfg.vocab, .K = h },
    };

    // Activation buffer (K bf16) and an output buffer big enough for any N.
    var x = try Buffer.alloc(&s.ctx, .device, 8192 * 2);
    defer x.free();
    var out = try Buffer.alloc(&s.ctx, .device, cfg.vocab * 2);
    defer out.free();

    for (shapes) |sh| {
        // Reuse the real weight tensors where names match; else allocate a dummy int4 buffer of the right size.
        const wbytes = sh.N * sh.K / 2;
        var dummy = try Buffer.alloc(&s.ctx, .device, wbytes);
        defer dummy.free();
        const scbytes = sh.N * (sh.K / 64) * 4;
        var dsc = try Buffer.alloc(&s.ctx, .device, scbytes);
        defer dsc.free();

        const g = geo.qmm(1, sh.N);
        const local: Dim = .{ .x = g.lx };
        const global: Dim = .{ .x = g.gx, .y = g.gy };
        const lcfg = rt.launch.Config{ .global = global, .local = local };

        // warm-up
        for (0..3) |_| {
            try rt.launch.run(&s.queue, &k, lcfg, .{ x.ptr, @as(i32, 1), dummy.ptr, dsc.ptr, dsc.ptr, out.ptr, @as(i32, @intCast(sh.N)), @as(i32, @intCast(sh.K)) }, null);
        }
        try s.queue.synchronize();
        var best: u64 = std.math.maxInt(u64);
        var sum: u64 = 0;
        const reps = 50;
        for (0..reps) |_| {
            const t0 = now();
            try rt.launch.run(&s.queue, &k, lcfg, .{ x.ptr, @as(i32, 1), dummy.ptr, dsc.ptr, dsc.ptr, out.ptr, @as(i32, @intCast(sh.N)), @as(i32, @intCast(sh.K)) }, null);
            try s.queue.synchronize();
            const dt = now() - t0;
            sum += dt;
            if (dt < best) best = dt;
        }
        // Queued batch: N launches submitted back-to-back, one sync at the end (the forward's real shape).
        var batch_best: u64 = std.math.maxInt(u64);
        const batch = 50;
        for (0..5) |_| {
            const t0 = now();
            for (0..batch) |_| try rt.launch.run(&s.queue, &k, lcfg, .{ x.ptr, @as(i32, 1), dummy.ptr, dsc.ptr, dsc.ptr, out.ptr, @as(i32, @intCast(sh.N)), @as(i32, @intCast(sh.K)) }, null);
            try s.queue.synchronize();
            const dt = now() - t0;
            if (dt < batch_best) batch_best = dt;
        }
        const bytes = wbytes + scbytes * 2 + sh.K * 2 + sh.N * 2;
        std.debug.print("qmm {s:<10} N={d:<7} K={d:<5} sync-each best {d:8.1} us  mean {d:8.1} us  {d:7.1} GB/s  ({d:.2} MB)  | batched {d:8.1} us/launch ({d:7.1} GB/s)\n", .{
            sh.name, sh.N, sh.K, @as(f64, @floatFromInt(best)) / 1e3, @as(f64, @floatFromInt(sum)) / 1e3 / reps,
            @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(best)), @as(f64, @floatFromInt(bytes)) / 1e6,
            @as(f64, @floatFromInt(batch_best)) / 1e3 / batch, @as(f64, @floatFromInt(bytes)) * batch / @as(f64, @floatFromInt(batch_best)),
        });
    }

    // Pure launch overhead: an empty-ish tiny launch (N=64,K=64).
    {
        var dummy = try Buffer.alloc(&s.ctx, .device, 64 * 64 / 2);
        defer dummy.free();
        var dsc = try Buffer.alloc(&s.ctx, .device, 64 * 4);
        defer dsc.free();
        const g = geo.qmm(1, 64);
        const lcfg = rt.launch.Config{ .global = .{ .x = g.gx, .y = g.gy }, .local = .{ .x = g.lx } };
        for (0..3) |_| try rt.launch.run(&s.queue, &k, lcfg, .{ x.ptr, @as(i32, 1), dummy.ptr, dsc.ptr, dsc.ptr, out.ptr, @as(i32, 64), @as(i32, 64) }, null);
        try s.queue.synchronize();
        var best: u64 = std.math.maxInt(u64);
        for (0..50) |_| {
            const t0 = now();
            try rt.launch.run(&s.queue, &k, lcfg, .{ x.ptr, @as(i32, 1), dummy.ptr, dsc.ptr, dsc.ptr, out.ptr, @as(i32, 64), @as(i32, 64) }, null);
            try s.queue.synchronize();
            const dt = now() - t0;
            if (dt < best) best = dt;
        }
        std.debug.print("qmm tiny     N=64 K=64 (launch+sync floor): best {d:.1} us\n", .{@as(f64, @floatFromInt(best)) / 1e3});
    }
    return 0;
}
