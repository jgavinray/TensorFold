//! Kernel launches on an in-order immediate list, OpenCL-style: global size and local (group) size.
const std = @import("std");
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const Kernel = @import("module.zig").Kernel;
const Queue = @import("queue.zig").Queue;
const Event = @import("event.zig").Event;

pub const Dim3 = struct { x: u32, y: u32 = 1, z: u32 = 1 };
pub const Config = struct {
    /// Work-items per dimension (OpenCL global size), a multiple of `local`.
    global: Dim3,
    local: Dim3,

    pub fn groups(self: Config) runtime.Error!abi.GroupCount {
        const g = self.global;
        const l = self.local;
        if (g.x == 0 or g.y == 0 or g.z == 0 or l.x == 0 or l.y == 0 or l.z == 0) return error.Invalid;
        if (g.x % l.x != 0 or g.y % l.y != 0 or g.z % l.z != 0) return error.Invalid;
        return .{ .x = g.x / l.x, .y = g.y / l.y, .z = g.z / l.z };
    }
};

/// Sets the group size and appends the launch; arguments must already be set (see `run` for the tuple form).
pub fn launch(q: *Queue, k: *Kernel, cfg: Config, signal: ?*const Event) runtime.Error!void {
    const g = try cfg.groups();
    try k.setGroupSize(cfg.local.x, cfg.local.y, cfg.local.z);
    const sig: abi.Event = if (signal) |e| e.handle else null;
    try runtime.check(q.ctx.api().zeCommandListAppendLaunchKernel(q.list, k.handle, &g, sig, 0, null));
}

/// Sets every argument of the tuple in order, then launches.
pub fn run(q: *Queue, k: *Kernel, cfg: Config, args: anytype, signal: ?*const Event) runtime.Error!void {
    inline for (args, 0..) |a, i| try k.setArg(i, a);
    try launch(q, k, cfg, signal);
}

test "launch geometry refuses zero and non-divisible sizes" {
    const g = try (Config{ .global = .{ .x = 128, .y = 33 }, .local = .{ .x = 128 } }).groups();
    try std.testing.expectEqual(@as(u32, 33), g.y);
    try std.testing.expectError(error.Invalid, (Config{ .global = .{ .x = 0 }, .local = .{ .x = 1 } }).groups());
    try std.testing.expectError(error.Invalid, (Config{ .global = .{ .x = 100 }, .local = .{ .x = 64 } }).groups());
}
