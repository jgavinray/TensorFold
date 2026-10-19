//! In-order immediate command lists only. An out-of-order list gave stale readbacks on this hardware
//! (exp/xmx-probe/dpas_ooo_queue_REPRO.cpp), so the in-order flag is not optional and there is no other constructor.
const std = @import("std");
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const Context = @import("context.zig").Context;
const Event = @import("event.zig").Event;

pub const Queue = struct {
    ctx: *const Context,
    list: abi.CommandList,

    pub fn init(ctx: *const Context) runtime.Error!Queue {
        var list: abi.CommandList = null;
        const desc: abi.QueueDesc = .{
            .ordinal = ctx.compute_ordinal,
            .index = 0,
            .flags = abi.queue_flag_in_order,
            .mode = abi.queue_mode_asynchronous,
        };
        try runtime.check(ctx.api().zeCommandListCreateImmediate(ctx.handle, ctx.dev.handle, &desc, &list));
        if (list == null) return error.Invalid;
        return .{ .ctx = ctx, .list = list };
    }

    pub fn deinit(self: *Queue) void {
        _ = self.ctx.api().zeCommandListDestroy(self.list);
        self.* = undefined;
    }

    fn handle(e: ?*const Event) abi.Event {
        return if (e) |ev| ev.handle else null;
    }

    pub fn copy(self: *Queue, dst: ?*anyopaque, src: ?*const anyopaque, len: usize, sig: ?*const Event) runtime.Error!void {
        if (len == 0) return;
        try runtime.live();
        try runtime.submit(self.ctx.api().zeCommandListAppendMemoryCopy(self.list, dst, src, len, handle(sig), 0, null));
    }

    /// Fills `len` bytes with a repeating `pattern` (len must be a multiple of pattern.len).
    pub fn fill(self: *Queue, dst: ?*anyopaque, pattern: []const u8, len: usize, sig: ?*const Event) runtime.Error!void {
        if (pattern.len == 0 or len % pattern.len != 0) return error.Invalid;
        if (len == 0) return;
        try runtime.live();
        try runtime.submit(self.ctx.api().zeCommandListAppendMemoryFill(self.list, dst, pattern.ptr, pattern.len, len, handle(sig), 0, null));
    }

    pub fn barrier(self: *Queue, sig: ?*const Event) runtime.Error!void {
        try runtime.live();
        try runtime.submit(self.ctx.api().zeCommandListAppendBarrier(self.list, handle(sig), 0, null));
    }

    pub fn signalEvent(self: *Queue, ev: *const Event) runtime.Error!void {
        try runtime.live();
        try runtime.submit(self.ctx.api().zeCommandListAppendSignalEvent(self.list, ev.handle));
    }

    pub fn synchronize(self: *Queue) runtime.Error!void {
        try runtime.live();
        try runtime.submit(self.ctx.api().zeCommandListHostSynchronize(self.list, abi.wait_forever));
    }
};
