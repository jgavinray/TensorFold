//! A Level Zero context on one selected Intel device, plus its compute queue-group ordinal and the device-memory
//! accounting every allocation goes through.
const std = @import("std");
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const device = @import("device.zig");

/// What the card can hold and what this process holds on it. The driver will happily *create* buffers past VRAM - it
/// evicts to system memory - but a long-running compute context that touches them every step then stalls forever
/// (2026-10-19: a 27B serving plan of ~58 GiB on a 31.9 GiB B70 hung decode). So the runtime knows the card's real
/// capacity and refuses an allocation that would exceed it, instead of letting the driver thrash.
pub const Usage = struct {
    /// Device memory the driver reports (zeDeviceGetMemoryProperties, all regions).
    total: u64,
    /// What `memory.Buffer.alloc` may hand out: `total` minus a reserve for kernel binaries, command buffers and the
    /// driver's own allocations, none of which go through `Buffer`.
    budget: u64,
    /// Device and shared bytes currently allocated through `Buffer` in this context.
    used: std.atomic.Value(u64) = .init(0),

    pub fn reserveFor(total: u64) u64 {
        return @max(512 << 20, total / 32);
    }
};

/// One accounting record per live context; contexts are few (one per opened card), so a small ring is enough.
var usage_slots: [16]Usage = undefined;
var usage_next: std.atomic.Value(usize) = .init(0);

pub const Context = struct {
    dev: device.Device,
    handle: abi.Context,
    compute_ordinal: u32,
    usage: *Usage,

    pub fn init(dev: device.Device) runtime.Error!Context {
        const r = dev.r;
        const total = try deviceMemory(dev);
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
            if (g.flags & abi.queue_group_compute != 0) {
                const u = &usage_slots[usage_next.fetchAdd(1, .monotonic) % usage_slots.len];
                u.* = .{ .total = total, .budget = total - Usage.reserveFor(total) };
                return .{ .dev = dev, .handle = handle, .compute_ordinal = @intCast(i), .usage = u };
            }
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

/// The device's memory in bytes, summed over its regions. A B70 reports one region of ~32 GiB; anything under 1 GiB
/// means the query (or its struct layout) is wrong, and the runtime refuses to run blind rather than guess.
fn deviceMemory(dev: device.Device) runtime.Error!u64 {
    const api = &dev.r.api;
    var n: u32 = 0;
    try runtime.check(api.zeDeviceGetMemoryProperties(dev.handle, &n, null));
    if (n == 0 or n > 8) return error.Invalid;
    var props: [8]abi.DeviceMemoryProperties = @splat(.{});
    try runtime.check(api.zeDeviceGetMemoryProperties(dev.handle, &n, &props));
    var total: u64 = 0;
    for (props[0..n]) |p| total += p.totalSize;
    if (total < 1 << 30) return error.Invalid;
    return total;
}

test "the reserve keeps at least 512 MiB and grows with the card" {
    try std.testing.expectEqual(@as(u64, 512 << 20), Usage.reserveFor(8 << 30));
    try std.testing.expectEqual(@as(u64, 1 << 30), Usage.reserveFor(32 << 30));
}
