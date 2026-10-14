//! A Level Zero context on one selected Intel device, plus its compute queue-group ordinal.
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const device = @import("device.zig");

pub const Context = struct {
    dev: device.Device,
    handle: abi.Context,
    compute_ordinal: u32,

    pub fn init(dev: device.Device) runtime.Error!Context {
        const r = dev.r;
        var handle: abi.Context = null;
        try runtime.check(r.api.zeContextCreate(dev.driver, &.{}, &handle));
        if (handle == null) return error.Invalid;
        errdefer _ = r.api.zeContextDestroy(handle);
        var n: u32 = 0;
        try runtime.check(r.api.zeDeviceGetCommandQueueGroupProperties(dev.handle, &n, null));
        if (n == 0 or n > 16) return error.Invalid;
        var groups: [16]abi.QueueGroupProperties = @splat(.{});
        try runtime.check(r.api.zeDeviceGetCommandQueueGroupProperties(dev.handle, &n, &groups));
        for (groups[0..n], 0..) |g, i| {
            if (g.flags & abi.queue_group_compute != 0) return .{ .dev = dev, .handle = handle, .compute_ordinal = @intCast(i) };
        }
        return error.Invalid;
    }

    pub fn deinit(self: *Context) void {
        _ = self.dev.r.api.zeContextDestroy(self.handle);
        self.* = undefined;
    }

    pub fn api(self: *const Context) *const abi.Api {
        return &self.dev.r.api;
    }
};
