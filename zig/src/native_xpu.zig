//! The Intel XPU engine a native server opens: Qwen3.5-2B and Qwen3.8-27B (both MLX affine-4) on a B70 through
//! Level Zero, served by the same lane host and scheduler as the Metal path (native/qwen35.zig, native/qwen27.zig).
//! Rooted at zig/src so the family and runtime files import each other by relative path.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const rt = @import("xpu/runtime.zig");
const Allocator = std.mem.Allocator;

pub const backends: []const []const u8 = &.{"xpu"};
pub const families: []const api.Family = &.{
    .{ .model_type = "qwen3_5", .formats = &.{"mlx-q4g64"} },
    .{ .model_type = "qwen27", .formats = &.{"mlx-q4g64"} },
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
    family: Family,
    weights: Family.Weights,
    model: Family.Model,
    xpu: Family.Xpu,
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        h.core.deinit();
        h.config.deinit(h.gpa);
        switch (h.family) {
            .qwen35 => {
                h.xpu.qwen35.deinit();
                h.model.qwen35.kernels.deinit();
                h.weights.qwen35.deinit();
                h.gpa.destroy(h.model.qwen35);
                h.gpa.destroy(h.weights.qwen35);
            },
            .qwen27 => {
                h.xpu.qwen27.deinit();
                h.model.qwen27.kernels.deinit();
                h.weights.qwen27.deinit();
                h.gpa.destroy(h.model.qwen27);
                h.gpa.destroy(h.weights.qwen27);
            },
        }
        h.session.close();
        h.gpa.destroy(h.session);
        h.gpa.destroy(h);
    }
};

/// The two families this binary serves, each with its own config, weights, forward and backend types.
const Family = enum {
    qwen35,
    qwen27,

    const Config = union(Family) {
        qwen35: @import("families/qwen3_5/config.zig").Config,
        qwen27: @import("families/qwen27/config.zig").Config,
    };
    const Weights = union(Family) {
        qwen35: *@import("families/qwen3_5/xpu_weights.zig").Weights,
        qwen27: *@import("families/qwen27/xpu_weights.zig").Weights,
    };
    const Model = union(Family) {
        qwen35: *@import("families/qwen3_5/xpu_forward.zig").Model,
        qwen27: *@import("families/qwen27/xpu_forward.zig").Model,
    };
    const Xpu = union(Family) {
        qwen35: *@import("families/qwen3_5/xpu_backend.zig").Xpu,
        qwen27: *@import("families/qwen27/xpu_backend.zig").Xpu,
    };

    /// The family a checkpoint belongs to. Both the 2B and the 27B report `model_type` `qwen3_5`, so `model_type`
    /// alone cannot tell them apart: the 27B's `text_config` is a different geometry, and running it through the 2B
    /// family's forward (compiled for the 2B constants) is not a shape error the kernels can report - it reads the
    /// wrong buffers. The config decides: the 27B layout (`Geometry.qwen27`) selects `qwen27`, the 2B layout
    /// (`Geometry.qwen35`) selects `qwen35`; a checkpoint whose layout is neither is refused with `problem` set.
    fn route(a: Allocator, gpa: Allocator, io: std.Io, model_type: []const u8, dir: []const u8, problem: *[]const u8) !?Family {
        const cfg = @import("families/qwen3_5/config.zig");
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22)) catch {
            problem.* = try std.fmt.allocPrint(a, "the XPU Qwen engine cannot read {s}/config.json", .{dir});
            return null;
        };
        defer gpa.free(bytes);
        const config = cfg.parse(gpa, bytes) catch |err| {
            problem.* = try std.fmt.allocPrint(a, "the XPU Qwen engine cannot read this checkpoint's config ({s})", .{@errorName(err)});
            return null;
        };
        const geometry = config.geometry();
        if (geometry.eql(cfg.Geometry.qwen27)) return .qwen27;
        if (geometry.eql(cfg.Geometry.qwen35)) return .qwen35;
        problem.* = try std.fmt.allocPrint(a, "--backend xpu serves Qwen3.5-2B (qwen3_5) and Qwen3.8-27B (qwen27) only, not {s}", .{model_type});
        return null;
    }

    /// The checkpoint's own context window, read before any card work: both families' configs read themselves the
    /// same way, so one call serves either (the 2B states it as `context`, the 27B as `max_position`).
    fn context(family: Family, gpa: Allocator, io: std.Io, dir: []const u8) !usize {
        return switch (family) {
            .qwen35 => (try @import("families/qwen3_5/config.zig").Config.read(gpa, io, dir)).context,
            .qwen27 => (try @import("families/qwen27/config.zig").Config.read(gpa, io, dir)).max_position,
        };
    }
};

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const family = try Family.route(a, gpa, io, o.model_type, o.dir, problem) orelse return null;
    const context: i64 = @intCast(family.context(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU Qwen engine cannot read this checkpoint's config ({s})", .{@errorName(err)});
        return null;
    });
    const window: i64 = o.context orelse @min(default_window, context);
    if (window < 1 or window > context) {
        problem.* = "--context must fit the Qwen checkpoint's positive context window";
        return null;
    }
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.family = family;
    h.session = try gpa.create(rt.session.Session);
    errdefer gpa.destroy(h.session);
    std.log.info("xpu: waiting for a free B70 (the card lock is shared with other jobs)", .{});
    h.session.open(.{}) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "cannot open a B70 ({s}: {s})", .{ @errorName(err), h.session.diag.message() });
        // A `return null` is not an error return, so the `errdefer`s above do not run: free here.
        gpa.destroy(h.session);
        gpa.destroy(h);
        return null;
    };
    errdefer h.session.close();
    std.log.info("xpu: card {s} ({s})", .{ h.session.bdf(), h.session.ctx.dev.name() });
    switch (family) {
        .qwen35 => try openQwen35(h, a, gpa, io, o, problem),
        .qwen27 => try openQwen27(h, a, gpa, io, o, problem),
    }
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// Builds a Qwen3.5-2B host: config, weights, kernels, backend, lane config and the serving host.
fn openQwen35(h: *Host, a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !void {
    const cfg = @import("families/qwen3_5/config.zig");
    const xw = @import("families/qwen3_5/xpu_weights.zig");
    const fwd = @import("families/qwen3_5/xpu_forward.zig");
    const xb = @import("families/qwen3_5/xpu_backend.zig");
    const config = try cfg.Config.read(gpa, io, o.dir);
    const window: i64 = o.context orelse @min(default_window, @as(i64, @intCast(config.context)));
    const weights = try gpa.create(xw.Weights);
    var weights_live = false;
    errdefer {
        if (weights_live) weights.deinit();
        gpa.destroy(weights);
    }
    weights.* = xw.load(gpa, h.session, o.dir, config.tied) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU Qwen engine cannot load this checkpoint ({s})", .{@errorName(err)});
        return err;
    };
    weights_live = true;
    h.weights = .{ .qwen35 = weights };
    const model = try gpa.create(fwd.Model);
    var model_live = false;
    errdefer {
        if (model_live) model.kernels.deinit();
        gpa.destroy(model);
    }
    model.* = .{ .weights = weights, .kernels = undefined };
    model.kernels.load(h.session, gpa) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU Qwen kernels did not load ({s})", .{@errorName(err)});
        return err;
    };
    model_live = true;
    h.model = .{ .qwen35 = model };
    h.xpu = .{ .qwen35 = xb.Xpu.init(gpa, h.session, h.model.qwen35, .{ .capacity = @intCast(window), .chunk = chunk }) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU Qwen backend did not start ({s})", .{@errorName(err)});
        return err;
    } };
    errdefer h.xpu.qwen35.deinit();
    h.config = try lanes.Config.init(gpa, h.xpu.qwen35.facts(), xb.window_rows, xb.window_rows - 1);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, h.xpu.qwen35.backend(), h.clock.clock());
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(window), .prefill_step = chunk, .structures = true });
    try h.host.start();
}

/// Builds a Qwen3.8-27B host: the same shape, its own 27B config/weights/forward/backend.
fn openQwen27(h: *Host, a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !void {
    const cfg = @import("families/qwen27/config.zig");
    const xw = @import("families/qwen27/xpu_weights.zig");
    const fwd = @import("families/qwen27/xpu_forward.zig");
    const xb = @import("families/qwen27/xpu_backend.zig");
    const config = try cfg.Config.read(gpa, io, o.dir);
    const window: i64 = o.context orelse @min(default_window, @as(i64, @intCast(config.max_position)));
    const weights = try gpa.create(xw.Weights);
    var weights_live = false;
    errdefer {
        if (weights_live) weights.deinit();
        gpa.destroy(weights);
    }
    weights.* = xw.load(gpa, h.session, o.dir, config) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU Qwen engine cannot load this checkpoint ({s})", .{@errorName(err)});
        return err;
    };
    weights_live = true;
    h.weights = .{ .qwen27 = weights };
    const model = try gpa.create(fwd.Model);
    var model_live = false;
    errdefer {
        if (model_live) model.kernels.deinit();
        gpa.destroy(model);
    }
    model.* = .{ .weights = weights, .kernels = undefined };
    model.kernels.load(h.session, gpa) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU Qwen kernels did not load ({s})", .{@errorName(err)});
        return err;
    };
    model_live = true;
    h.model = .{ .qwen27 = model };
    h.xpu = .{ .qwen27 = xb.Xpu.init(gpa, h.session, h.model.qwen27, config, .{ .capacity = @intCast(window), .chunk = chunk }) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU Qwen backend did not start ({s})", .{@errorName(err)});
        return err;
    } };
    h.config = try lanes.Config.init(gpa, h.xpu.qwen27.facts(), xb.window_rows, xb.window_rows - 1);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, h.xpu.qwen27.backend(), h.clock.clock());
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(window), .prefill_step = chunk, .structures = true });
    try h.host.start();
}
