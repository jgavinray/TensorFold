//! The Intel XPU engine a native server opens: Qwen3.5-2B (MLX affine-4) on a B70 through Level Zero, served by the
//! same lane host and scheduler as the Metal path (native/qwen35.zig). Rooted at zig/src so the family and runtime
//! files import each other by relative path.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const rt = @import("xpu/runtime.zig");
const cfg = @import("families/qwen3_5/config.zig");
const xw = @import("families/qwen3_5/xpu_weights.zig");
const fwd = @import("families/qwen3_5/xpu_forward.zig");
const xb = @import("families/qwen3_5/xpu_backend.zig");
const Allocator = std.mem.Allocator;

pub const backends: []const []const u8 = &.{"xpu"};
pub const families: []const api.Family = &.{
    .{ .model_type = "qwen3_5", .formats = &.{"mlx-q4g64"} },
};
/// `serve --backend xpu` is honoured by this binary (server/cli.zig reads this from the root).
pub const serves: []const []const u8 = &.{"--backend=xpu"};

/// Serve defaults this backend changes (server/serve.zig reads them from the root): the XPU 2B reasons past a short
/// max_tokens, so a served default with an open think block answers every chat request in reasoning_content and
/// leaves content empty. A closed block (`enable_thinking` false) is what the Metal serving examples pass too
/// (RUNBOOK.md `--no-thinking`); a request may still ask for thinking with chat_template_kwargs.
pub const defaults: api.ServeDefaults = .{ .enable_thinking = false };

/// Prompt plus reply tokens one request may use when --context is not given (the model's window is 262,144;
/// the KV cache of a stream is allocated at this size).
const default_window: i64 = 32768;
const chunk = 128;

pub const Device = struct { name: []const u8, bdf: []const u8 };

/// The allow-listed B70s the Level Zero loader shows (names and PCI addresses); empty when there is no driver or card.
/// Takes no card lock and does no device work beyond property reads.
pub fn devices(a: Allocator) []const Device {
    var runtime = rt.Runtime.open() catch return &.{};
    defer runtime.close();
    var out: std.ArrayList(Device) = .empty;
    for (rt.device.cards) |bdf| {
        const d = rt.device.select(&runtime, bdf) catch continue;
        const name = a.dupe(u8, d.name()) catch continue;
        const addr = a.dupe(u8, bdf) catch continue;
        out.append(a, .{ .name = name, .bdf = addr }) catch continue;
    }
    return out.items;
}

pub fn chip(a: Allocator) ?[]const u8 {
    const d = devices(a);
    if (d.len == 0) return null;
    return "intel-b70";
}

const Host = struct {
    gpa: Allocator,
    session: *rt.session.Session,
    weights: *xw.Weights,
    model: *fwd.Model,
    xpu: *xb.Xpu,
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        h.core.deinit();
        h.config.deinit(h.gpa);
        h.xpu.deinit();
        h.model.kernels.deinit();
        h.weights.deinit();
        h.session.close();
        h.gpa.destroy(h.model);
        h.gpa.destroy(h.weights);
        h.gpa.destroy(h.session);
        h.gpa.destroy(h);
    }
};

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (!std.mem.eql(u8, o.model_type, "qwen3_5")) {
        problem.* = try std.fmt.allocPrint(a, "--backend xpu serves Qwen3.5-2B (qwen3_5) only, not {s}", .{o.model_type});
        return null;
    }
    const config = cfg.Config.read(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU Qwen engine cannot read this checkpoint's config ({s})", .{@errorName(err)});
        return null;
    };
    const window: i64 = o.context orelse @min(default_window, @as(i64, @intCast(config.context)));
    if (window < 1 or window > config.context) {
        problem.* = "--context must fit the Qwen checkpoint's positive context window";
        return null;
    }
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.session = try gpa.create(rt.session.Session);
    errdefer gpa.destroy(h.session);
    std.log.info("xpu: waiting for a free B70 (the card lock is shared with other jobs)", .{});
    h.session.open(.{}) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "cannot open a B70 ({s}: {s})", .{ @errorName(err), h.session.diag.message() });
        return null;
    };
    errdefer h.session.close();
    std.log.info("xpu: card {s} ({s})", .{ h.session.bdf(), h.session.ctx.dev.name() });
    h.weights = try gpa.create(xw.Weights);
    errdefer gpa.destroy(h.weights);
    h.weights.* = xw.load(gpa, h.session, o.dir, config.tied) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU Qwen engine cannot load this checkpoint ({s})", .{@errorName(err)});
        return null;
    };
    errdefer h.weights.deinit();
    h.model = try gpa.create(fwd.Model);
    errdefer gpa.destroy(h.model);
    h.model.* = .{ .weights = h.weights, .kernels = undefined };
    h.model.kernels.load(h.session, gpa) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU Qwen kernels did not load ({s})", .{@errorName(err)});
        return null;
    };
    errdefer h.model.kernels.deinit();
    h.xpu = xb.Xpu.init(gpa, h.session, h.model, .{ .capacity = @intCast(window), .chunk = chunk }) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU Qwen backend did not start ({s})", .{@errorName(err)});
        return null;
    };
    errdefer h.xpu.deinit();
    h.config = try lanes.Config.init(gpa, h.xpu.facts(), xb.window_rows, xb.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, h.xpu.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(window), .prefill_step = chunk, .structures = true });
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}
