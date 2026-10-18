//! TEMP probe (throwaway): device-time comparison of fused vs unfused kernel pairs at R=1, batched.
//! Unlike xpu_normqmm_probe/xpu_actdown_probe (which sync per launch), this measures Level Zero event
//! timestamps with all launches queued back-to-back - the forward's real submission pattern.
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

const reps = 20;

const Ctx = struct {
    s: *rt.session.Session,
    pool: rt.event.Pool,
    events: [8]rt.event.Event = undefined,

    fn init(s: *rt.session.Session) !Ctx {
        var pool = try rt.event.Pool.init(&s.ctx, 8);
        var out = Ctx{ .s = s, .pool = pool };
        for (&out.events, 0..) |*e, i| e.* = try pool.event(@intCast(i));
        return out;
    }
    fn deinit(self: *Ctx) void {
        for (&self.events) |*e| e.deinit();
        self.pool.deinit();
    }
    /// Best total device time (ns) over `reps` rounds of the two queued launches (or one when `two` is false).
    fn pair(self: *Ctx, two: bool, c0: rt.launch.Config, k0: *rt.module.Kernel, a0: anytype, c1: rt.launch.Config, k1: *rt.module.Kernel, a1: anytype) !f64 {
        var best: f64 = std.math.inf(f64);
        for (0..reps) |_| {
            try rt.launch.run(&self.s.queue, k0, c0, a0, &self.events[0]);
            if (two) try rt.launch.run(&self.s.queue, k1, c1, a1, &self.events[1]);
            try self.s.queue.synchronize();
            var total: f64 = try self.events[0].kernelNs();
            if (two) total += try self.events[1].kernelNs();
            try self.events[0].reset();
            try self.events[1].reset();
            if (total < best) best = total;
        }
        return best;
    }
};

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
    var m_norm = try rt.module.Module.load(&s.ctx, &kbins.norm, &log);
    defer m_norm.unload();
    var m_qmm = try rt.module.Module.load(&s.ctx, &kbins.qmm, &log);
    defer m_qmm.unload();
    var m_nq = try rt.module.Module.load(&s.ctx, &kbins.norm_qmm, &log);
    defer m_nq.unload();
    var m_act = try rt.module.Module.load(&s.ctx, &kbins.mlp_act, &log);
    defer m_act.unload();
    var m_ad = try rt.module.Module.load(&s.ctx, &kbins.mlp_act_down, &log);
    defer m_ad.unload();
    var m_pair = try rt.module.Module.load(&s.ctx, &kbins.qmm_pair, &log);
    defer m_pair.unload();

    var k_norm = try m_norm.kernel("qwen35_norm");
    defer k_norm.deinit();
    var k_qmm = try m_qmm.kernel("qwen35_qmm");
    defer k_qmm.deinit();
    var k_nq = try m_nq.kernel("qwen35_norm_qmm");
    defer k_nq.deinit();
    var k_act = try m_act.kernel("qwen35_mlp_act");
    defer k_act.deinit();
    var k_ad = try m_ad.kernel("qwen35_mlp_act_down");
    defer k_ad.deinit();
    var k_pair = try m_pair.kernel("qwen35_qmm_pair");
    defer k_pair.deinit();

    const h: u32 = cfg.hidden;
    const inter: u32 = cfg.intermediate;
    const blk = w.blocks[0];
    var ctx = try Ctx.init(&s);
    defer ctx.deinit();

    // ---- Case 1: post-attention norm (residual) + mlp.gate qmm (N=6144, K=2048). ----
    {
        var hb = try Buffer.alloc(&s.ctx, .device, h * 2);
        defer hb.free();
        var rb = try Buffer.alloc(&s.ctx, .device, h * 2);
        defer rb.free();
        var ho = try Buffer.alloc(&s.ctx, .device, h * 2);
        defer ho.free();
        var xo = try Buffer.alloc(&s.ctx, .device, h * 2);
        defer xo.free();
        const gate = blk.gate;
        var out = try Buffer.alloc(&s.ctx, .device, gate.outputs * 2);
        defer out.free();
        const N: i32 = @intCast(gate.outputs);
        const K: i32 = @intCast(gate.inputs);

        const gn = geo.norm(1, h);
        const ncfg = rt.launch.Config{ .global = .{ .x = gn.gx, .y = gn.gy }, .local = .{ .x = gn.lx } };
        const gq = geo.qmm(1, @intCast(gate.outputs));
        const qcfg = rt.launch.Config{ .global = .{ .x = gq.gx, .y = gq.gy }, .local = .{ .x = gq.lx } };
        const wn = blk.post_norm;

        const unfused = try ctx.pair(true, ncfg, &k_norm, .{ hb.ptr, rb.ptr, w.ptr(wn), cfg.eps, ho.ptr, xo.ptr }, qcfg, &k_qmm, .{
            xo.ptr, @as(i32, 1), w.ptr(gate.weight), w.ptr(gate.scales), w.ptr(gate.biases), out.ptr, N, K,
        });
        const fused = try ctx.pair(false, qcfg, &k_nq, .{
            hb.ptr, rb.ptr, w.ptr(wn), cfg.eps, ho.ptr, w.ptr(gate.weight), w.ptr(gate.scales), w.ptr(gate.biases), out.ptr, @as(i32, 1), N, K,
        }, qcfg, &k_qmm, .{ xo.ptr, @as(i32, 1), w.ptr(gate.weight), w.ptr(gate.scales), w.ptr(gate.biases), out.ptr, N, K });
        std.debug.print("norm+qmm (N={d} K={d}): unfused {d:.2} us, fused {d:.2} us, saving {d:.2} us\n", .{
            gate.outputs, gate.inputs, unfused / 1e3, fused / 1e3, (unfused - fused) / 1e3,
        });
    }

    // ---- Case 1b: the same norm_qmm fusion at N=2048 (32 work-groups instead of 96): the fused kernel
    // recomputes the row's RMSNorm once per work-group, so a smaller grid means less redundant work. ----
    {
        var hb = try Buffer.alloc(&s.ctx, .device, h * 2);
        defer hb.free();
        var rb = try Buffer.alloc(&s.ctx, .device, h * 2);
        defer rb.free();
        var ho = try Buffer.alloc(&s.ctx, .device, h * 2);
        defer ho.free();
        var xo = try Buffer.alloc(&s.ctx, .device, h * 2);
        defer xo.free();
        const lin = blk.mixer.delta.out; // N=2048 K=2048
        var out = try Buffer.alloc(&s.ctx, .device, lin.outputs * 2);
        defer out.free();
        const N: i32 = @intCast(lin.outputs);
        const K: i32 = @intCast(lin.inputs);

        const gn = geo.norm(1, h);
        const ncfg = rt.launch.Config{ .global = .{ .x = gn.gx, .y = gn.gy }, .local = .{ .x = gn.lx } };
        const gq = geo.qmm(1, @intCast(lin.outputs));
        const qcfg = rt.launch.Config{ .global = .{ .x = gq.gx, .y = gq.gy }, .local = .{ .x = gq.lx } };
        const wn = blk.post_norm;

        const unfused = try ctx.pair(true, ncfg, &k_norm, .{ hb.ptr, rb.ptr, w.ptr(wn), cfg.eps, ho.ptr, xo.ptr }, qcfg, &k_qmm, .{
            xo.ptr, @as(i32, 1), w.ptr(lin.weight), w.ptr(lin.scales), w.ptr(lin.biases), out.ptr, N, K,
        });
        const fused = try ctx.pair(false, qcfg, &k_nq, .{
            hb.ptr, rb.ptr, w.ptr(wn), cfg.eps, ho.ptr, w.ptr(lin.weight), w.ptr(lin.scales), w.ptr(lin.biases), out.ptr, @as(i32, 1), N, K,
        }, qcfg, &k_qmm, .{ xo.ptr, @as(i32, 1), w.ptr(lin.weight), w.ptr(lin.scales), w.ptr(lin.biases), out.ptr, N, K });
        std.debug.print("norm+qmm (N={d} K={d}): unfused {d:.2} us, fused {d:.2} us, saving {d:.2} us\n", .{
            lin.outputs, lin.inputs, unfused / 1e3, fused / 1e3, (unfused - fused) / 1e3,
        });
    }

    // ---- Case 2: mlp_act + down (N=2048, K=6144). ----
    {
        const lin = blk.down;
        const N: i32 = @intCast(lin.outputs);
        const K: i32 = @intCast(lin.inputs);
        var gb = try Buffer.alloc(&s.ctx, .device, inter * 2);
        defer gb.free();
        var ub = try Buffer.alloc(&s.ctx, .device, inter * 2);
        defer ub.free();
        var ab = try Buffer.alloc(&s.ctx, .device, inter * 2);
        defer ab.free();
        var out = try Buffer.alloc(&s.ctx, .device, lin.outputs * 2);
        defer out.free();

        const ga = geo.mlpAct(1, inter);
        const acfg = rt.launch.Config{ .global = .{ .x = ga.gx, .y = ga.gy }, .local = .{ .x = ga.lx } };
        const gq = geo.qmm(1, @intCast(lin.outputs));
        const qcfg = rt.launch.Config{ .global = .{ .x = gq.gx, .y = gq.gy }, .local = .{ .x = gq.lx } };

        const unfused = try ctx.pair(true, acfg, &k_act, .{ gb.ptr, ub.ptr, ab.ptr }, qcfg, &k_qmm, .{
            ab.ptr, @as(i32, 1), w.ptr(lin.weight), w.ptr(lin.scales), w.ptr(lin.biases), out.ptr, N, K,
        });
        const fused = try ctx.pair(false, qcfg, &k_ad, .{
            gb.ptr, ub.ptr, w.ptr(lin.weight), w.ptr(lin.scales), w.ptr(lin.biases), out.ptr, @as(i32, 1), N, K,
        }, qcfg, &k_qmm, .{ ab.ptr, @as(i32, 1), w.ptr(lin.weight), w.ptr(lin.scales), w.ptr(lin.biases), out.ptr, N, K });
        std.debug.print("mlp_act+down (N={d} K={d}): unfused {d:.2} us, fused {d:.2} us, saving {d:.2} us\n", .{
            lin.outputs, lin.inputs, unfused / 1e3, fused / 1e3, (unfused - fused) / 1e3,
        });
    }

    // ---- Case 3: bare qmm device time at each shape (batched, one sync per round). ----
    {
        var x = try Buffer.alloc(&s.ctx, .device, 8192 * 2);
        defer x.free();
        var out = try Buffer.alloc(&s.ctx, .device, cfg.vocab * 2);
        defer out.free();
        const shapes = [_]struct { name: []const u8, N: i32, K: i32 }{
            .{ .name = "gate/up", .N = 6144, .K = 2048 },
            .{ .name = "down", .N = 2048, .K = 6144 },
            .{ .name = "a/b", .N = 16, .K = 2048 },
            .{ .name = "head", .N = cfg.vocab, .K = 2048 },
        };
        for (shapes) |sh| {
            const gq = geo.qmm(1, @intCast(sh.N));
            const qcfg = rt.launch.Config{ .global = .{ .x = gq.gx, .y = gq.gy }, .local = .{ .x = gq.lx } };
            const lin = if (sh.N == 16) blk.mixer.delta.a else if (sh.N == 6144) blk.gate else if (sh.N == 2048) blk.down else w.head();
            const dt = try ctx.pair(false, qcfg, &k_qmm, .{
                x.ptr, @as(i32, 1), w.ptr(lin.weight), w.ptr(lin.scales), w.ptr(lin.biases), out.ptr, sh.N, sh.K,
            }, qcfg, &k_qmm, .{});
            std.debug.print("qmm {s:<8} N={d:<7} K={d:<5} device {d:7.2} us\n", .{ sh.name, sh.N, sh.K, dt / 1e3 });
        }
    }

    // ---- Case 4: the a/b pair (both N=16 K=2048, same input): two launches vs one N=32 launch. ----
    // Traffic is identical (the same 2x16 rows of W are read either way); the win is one launch instead of two.
    {
        var x = try Buffer.alloc(&s.ctx, .device, cfg.hidden * 2);
        defer x.free();
        var out32 = try Buffer.alloc(&s.ctx, .device, 32 * 2);
        defer out32.free();
        const d = blk.mixer.delta;
        const gq16 = geo.qmm(1, 16);
        const c16 = rt.launch.Config{ .global = .{ .x = gq16.gx, .y = gq16.gy }, .local = .{ .x = gq16.lx } };
        const gq32 = geo.qmm(1, 32);
        const c32 = rt.launch.Config{ .global = .{ .x = gq32.gx, .y = gq32.gy }, .local = .{ .x = gq32.lx } };

        const two = try ctx.pair(true, c16, &k_qmm, .{
            x.ptr, @as(i32, 1), w.ptr(d.a.weight), w.ptr(d.a.scales), w.ptr(d.a.biases), out32.ptr, @as(i32, 16), @as(i32, cfg.hidden),
        }, c16, &k_qmm, .{
            x.ptr, @as(i32, 1), w.ptr(d.b.weight), w.ptr(d.b.scales), w.ptr(d.b.biases), out32.ptr, @as(i32, 16), @as(i32, cfg.hidden),
        });
        // N=32 with a's weights (only the launch/traffic shape matters for timing; a and b have identical shapes).
        const one = try ctx.pair(false, c32, &k_qmm, .{
            x.ptr, @as(i32, 1), w.ptr(d.a.weight), w.ptr(d.a.scales), w.ptr(d.a.biases), out32.ptr, @as(i32, 32), @as(i32, cfg.hidden),
        }, c32, &k_qmm, .{});
        std.debug.print("a+b pair: two N=16 launches {d:.2} us, one N=32 launch {d:.2} us, saving {d:.2} us\n", .{
            two / 1e3, one / 1e3, (two - one) / 1e3,
        });
    }
    // ---- Case 5: the mlp gate/up pair (both N=6144 K=2048, same x): two launches vs one pair launch.
    // Same total work-groups either way; the win, if any, is one launch's overhead per layer. ----
    {
        var x2 = try Buffer.alloc(&s.ctx, .device, h * 2);
        defer x2.free();
        var out64 = try Buffer.alloc(&s.ctx, .device, blk.gate.outputs * 2);
        defer out64.free();
        const N: i32 = @intCast(blk.gate.outputs);
        const gq = geo.qmm(1, @intCast(blk.gate.outputs));
        const cq = rt.launch.Config{ .global = .{ .x = gq.gx, .y = gq.gy }, .local = .{ .x = gq.lx } };
        const gp = geo.qmmPair(1, @intCast(blk.gate.outputs));
        const cp = rt.launch.Config{ .global = .{ .x = gp.gx, .y = gp.gy }, .local = .{ .x = gp.lx } };
        const ga = blk.gate;
        const ua = blk.up;

        const two = try ctx.pair(true, cq, &k_qmm, .{
            x2.ptr, @as(i32, 1), w.ptr(ga.weight), w.ptr(ga.scales), w.ptr(ga.biases), out64.ptr, N, @as(i32, h),
        }, cq, &k_qmm, .{
            x2.ptr, @as(i32, 1), w.ptr(ua.weight), w.ptr(ua.scales), w.ptr(ua.biases), out64.ptr, N, @as(i32, h),
        });
        const one = try ctx.pair(false, cp, &k_pair, .{
            x2.ptr,            @as(i32, 1),
            w.ptr(ga.weight),  w.ptr(ga.scales), w.ptr(ga.biases),
            w.ptr(ua.weight),  w.ptr(ua.scales), w.ptr(ua.biases),
            out64.ptr,         out64.ptr,
            N,                 @as(i32, h),
        }, cp, &k_pair, .{});
        std.debug.print("gate+up pair: two launches {d:.2} us, one pair launch {d:.2} us, saving {d:.2} us\n", .{
            two / 1e3, one / 1e3, (two - one) / 1e3,
        });
    }
    return 0;
}
