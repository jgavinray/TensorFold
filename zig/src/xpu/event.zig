//! Event pools with kernel timestamps; elapsed time comes from event timestamps, never host clocks.
const std = @import("std");
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const Context = @import("context.zig").Context;

pub const Pool = struct {
    ctx: *const Context,
    handle: abi.EventPool,
    count: u32,

    pub fn init(ctx: *const Context, count: u32) runtime.Error!Pool {
        if (count == 0) return error.Invalid;
        var h: abi.EventPool = null;
        var dev = ctx.dev.handle;
        const desc: abi.EventPoolDesc = .{ .flags = abi.event_pool_host_visible | abi.event_pool_kernel_timestamp, .count = count };
        try runtime.check(ctx.api().zeEventPoolCreate(ctx.handle, &desc, 1, @ptrCast(&dev), &h));
        if (h == null) return error.Invalid;
        return .{ .ctx = ctx, .handle = h, .count = count };
    }

    pub fn deinit(self: *Pool) void {
        _ = self.ctx.api().zeEventPoolDestroy(self.handle);
        self.* = undefined;
    }

    pub fn event(self: *const Pool, index: u32) runtime.Error!Event {
        if (index >= self.count) return error.Invalid;
        var h: abi.Event = null;
        const desc: abi.EventDesc = .{ .index = index, .signal = abi.event_scope_host, .wait = abi.event_scope_host };
        try runtime.check(self.ctx.api().zeEventCreate(self.handle, &desc, &h));
        if (h == null) return error.Invalid;
        return .{ .ctx = self.ctx, .handle = h };
    }
};

pub const Event = struct {
    ctx: *const Context,
    handle: abi.Event,

    pub fn deinit(self: *Event) void {
        _ = self.ctx.api().zeEventDestroy(self.handle);
        self.* = undefined;
    }

    pub fn wait(self: *const Event) runtime.Error!void {
        try runtime.check(self.ctx.api().zeEventHostSynchronize(self.handle, abi.wait_forever));
    }

    pub fn reset(self: *const Event) runtime.Error!void {
        try runtime.check(self.ctx.api().zeEventHostReset(self.handle));
    }

    /// Device-side kernel duration in nanoseconds, from the event's start/end timestamps (counter wrap handled).
    pub fn kernelNs(self: *const Event) runtime.Error!f64 {
        var ts: abi.TimestampResult = .{};
        try runtime.check(self.ctx.api().zeEventQueryKernelTimestamp(self.handle, &ts));
        const props = self.ctx.dev.props;
        return ticksToNs(ts.global.start, ts.global.end, props.kernelTimestampValidBits, props.timerResolution);
    }
};

pub fn ticksToNs(start: u64, end: u64, valid_bits: u32, ns_per_tick: u64) f64 {
    const bits: u7 = @intCast(@min(valid_bits, 64));
    const mask: u64 = if (bits >= 64 or bits == 0) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(bits)) - 1;
    const ticks = (end -% start) & mask;
    return @as(f64, @floatFromInt(ticks)) * @as(f64, @floatFromInt(ns_per_tick));
}

test "timestamp ticks convert with counter wrap" {
    try std.testing.expectEqual(@as(f64, 1000), ticksToNs(100, 110, 32, 100));
    // counter wrapped at 32 valid bits
    try std.testing.expectEqual(@as(f64, 19), ticksToNs(0xFFFFFFFF - 4, 14, 32, 1));
}
