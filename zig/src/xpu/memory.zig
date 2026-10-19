//! Device, host and shared allocations in a context; synchronous host<->device helpers go through a Queue.
const std = @import("std");
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const Context = @import("context.zig").Context;
const Queue = @import("queue.zig").Queue;

pub const Kind = enum { device, host, shared };

pub const Buffer = struct {
    ctx: *const Context,
    ptr: ?*anyopaque,
    len: usize,
    kind: Kind,

    pub fn alloc(ctx: *const Context, kind: Kind, len: usize) runtime.Error!Buffer {
        if (len == 0 or len > ctx.dev.props.maxMemAllocSize) return error.Invalid;
        // Device and shared allocations are charged against the card's real capacity (context.Usage): an allocation
        // that would overcommit VRAM is refused here, loudly, instead of being created and evicted by the driver.
        const charged = kind != .host;
        if (charged) {
            const u = ctx.usage;
            const before = u.used.fetchAdd(len, .monotonic);
            if (before + len > u.budget) {
                _ = u.used.fetchSub(len, .monotonic);
                var bb: [12]u8 = undefined;
                std.log.warn("xpu: {s}: device memory budget exceeded - asked {d} MiB with {d} MiB already allocated; " ++
                    "budget {d} MiB of {d} MiB on the card. The plan does not fit this card: shrink it, never overcommit.", .{
                    ctx.dev.bdf.format(&bb), (len + (1 << 20) - 1) >> 20, before >> 20, u.budget >> 20, u.total >> 20,
                });
                return error.OutOfDeviceMemory;
            }
        }
        errdefer if (charged) {
            _ = ctx.usage.used.fetchSub(len, .monotonic);
        };
        var p: ?*anyopaque = null;
        const api = ctx.api();
        const dd: abi.DeviceMemDesc = .{};
        const hd: abi.HostMemDesc = .{};
        switch (kind) {
            .device => try runtime.check(api.zeMemAllocDevice(ctx.handle, &dd, len, 64, ctx.dev.handle, &p)),
            .host => try runtime.check(api.zeMemAllocHost(ctx.handle, &hd, len, 64, &p)),
            .shared => try runtime.check(api.zeMemAllocShared(ctx.handle, &dd, &hd, len, 64, ctx.dev.handle, &p)),
        }
        if (p == null) return error.Invalid;
        return .{ .ctx = ctx, .ptr = p, .len = len, .kind = kind };
    }

    pub fn free(self: *Buffer) void {
        _ = self.ctx.api().zeMemFree(self.ctx.handle, self.ptr);
        if (self.kind != .host) _ = self.ctx.usage.used.fetchSub(self.len, .monotonic);
        self.* = undefined;
    }

    pub fn bytes(self: *const Buffer) runtime.Error![]u8 {
        if (self.kind == .device) return error.Invalid; // not host-addressable
        return @as([*]u8, @ptrCast(self.ptr.?))[0..self.len];
    }

    /// Queued host->buffer copy of `src` (src must stay valid until the queue is synchronized).
    pub fn uploadAsync(self: *Buffer, q: *Queue, offset: usize, src: []const u8) runtime.Error!void {
        if (offset > self.len or src.len > self.len - offset) return error.Invalid;
        const dst: *anyopaque = @ptrFromInt(@intFromPtr(self.ptr.?) + offset);
        try q.copy(dst, src.ptr, src.len, null);
    }

    pub fn downloadAsync(self: *const Buffer, q: *Queue, offset: usize, dst: []u8) runtime.Error!void {
        if (offset > self.len or dst.len > self.len - offset) return error.Invalid;
        const src: *const anyopaque = @ptrFromInt(@intFromPtr(self.ptr.?) + offset);
        try q.copy(dst.ptr, src, dst.len, null);
    }

    pub fn upload(self: *Buffer, q: *Queue, offset: usize, src: []const u8) runtime.Error!void {
        try self.uploadAsync(q, offset, src);
        try q.synchronize();
    }

    pub fn download(self: *const Buffer, q: *Queue, offset: usize, dst: []u8) runtime.Error!void {
        try self.downloadAsync(q, offset, dst);
        try q.synchronize();
    }
};
