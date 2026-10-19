//! Regular (non-immediate) command lists and their queue: the CUDA-graph analogue for decode.
//!
//! An immediate list replays its commands into the hardware queue as each append happens, so every kernel of a
//! decode step costs a host->device submit (FINDINGS §9: a 2048x2048 int4 GEMV is ~4 us of real traffic but ~27 us
//! wall). A *regular* list instead records the appends into a closed command buffer, and
//! `zeCommandQueueExecuteCommandLists` submits the whole step in one call.
//!
//! `Recorder` builds one such list; `Recorded` holds it closed and replays it. Both go through an in-order
//! command queue so copy<->compute ordering stays explicit (the hardware rule learned from
//! exp/xmx-probe/dpas_ooo_queue_REPRO.cpp). Kernel *arguments* are baked into the recorded list, so a replay must
//! not depend on `zeKernelSetArgumentValue` between executes; the changing per-token values travel through a
//! device buffer the kernels read (see `Params`).
const std = @import("std");
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const Context = @import("context.zig").Context;
const Kernel = @import("module.zig").Kernel;
const Event = @import("event.zig").Event;
const Dim3 = @import("launch.zig").Dim3;

/// In-order command queue: the only queue kind this backend uses (out-of-order gave stale readbacks).
pub const Queue = struct {
    ctx: *const Context,
    handle: abi.CommandQueue,

    pub fn init(ctx: *const Context) runtime.Error!Queue {
        var h: abi.CommandQueue = null;
        const desc: abi.CommandQueueDesc = .{
            .ordinal = ctx.compute_ordinal,
            .index = 0,
            .flags = abi.queue_flag_in_order,
            .mode = abi.queue_mode_asynchronous,
        };
        try runtime.check(ctx.api().zeCommandQueueCreate(ctx.handle, ctx.dev.handle, &desc, &h));
        if (h == null) return error.Invalid;
        return .{ .ctx = ctx, .handle = h };
    }

    pub fn deinit(self: *Queue) void {
        _ = self.ctx.api().zeCommandQueueDestroy(self.handle);
        self.* = undefined;
    }

    /// Submits `lists` in one call; with `signal` the queue's completion signals that event.
    pub fn execute(self: *Queue, lists: []const abi.CommandList, signal: ?*const Event) runtime.Error!void {
        if (lists.len == 0) return error.Invalid;
        const sig: abi.Event = if (signal) |e| e.handle else null;
        var handles: [8]abi.CommandList = undefined;
        if (lists.len > handles.len) return error.Invalid;
        @memcpy(handles[0..lists.len], lists);
        try runtime.live();
        try runtime.submit(self.ctx.api().zeCommandQueueExecuteCommandLists(self.handle, @intCast(lists.len), &handles, sig));
    }

    pub fn synchronize(self: *Queue) runtime.Error!void {
        try runtime.live();
        try runtime.submit(self.ctx.api().zeCommandQueueSynchronize(self.handle, abi.wait_forever));
    }
};

/// Builds one regular command list: appends are recorded, not executed.
pub const Recorder = struct {
    ctx: *const Context,
    handle: abi.CommandList,

    pub fn init(ctx: *const Context) runtime.Error!Recorder {
        var h: abi.CommandList = null;
        // In order: a recorded step's kernels each read what the previous one wrote. (Created without the flag, the
        // driver may overlap appended launches; nothing in the list orders them.)
        const desc: abi.CommandListDesc = .{ .commandQueueGroupOrdinal = ctx.compute_ordinal, .flags = abi.list_flag_in_order };
        try runtime.check(ctx.api().zeCommandListCreate(ctx.handle, ctx.dev.handle, &desc, &h));
        if (h == null) return error.Invalid;
        return .{ .ctx = ctx, .handle = h };
    }

    pub fn deinit(self: *Recorder) void {
        _ = self.ctx.api().zeCommandListDestroy(self.handle);
        self.* = undefined;
    }

    pub fn copy(self: *Recorder, dst: ?*anyopaque, src: ?*const anyopaque, len: usize) runtime.Error!void {
        if (len == 0) return;
        try runtime.check(self.ctx.api().zeCommandListAppendMemoryCopy(self.handle, dst, src, len, null, 0, null));
    }

    pub fn fill(self: *Recorder, dst: ?*anyopaque, pattern: []const u8, len: usize) runtime.Error!void {
        if (pattern.len == 0 or len % pattern.len != 0) return error.Invalid;
        if (len == 0) return;
        try runtime.check(self.ctx.api().zeCommandListAppendMemoryFill(self.handle, dst, pattern.ptr, pattern.len, len, null, 0, null));
    }

    pub fn barrier(self: *Recorder) runtime.Error!void {
        try runtime.check(self.ctx.api().zeCommandListAppendBarrier(self.handle, null, 0, null));
    }

    /// Sets `k`'s arguments from the tuple, then records the launch. The argument values live in the kernel object
    /// and are captured by the append, so the tuple need not outlive this call.
    pub fn run(self: *Recorder, k: *Kernel, cfg: anytype, args: anytype) runtime.Error!void {
        const g = try cfg.groups();
        try k.setGroupSize(cfg.local.x, cfg.local.y, cfg.local.z);
        inline for (args, 0..) |a, i| try k.setArg(i, a);
        try runtime.check(self.ctx.api().zeCommandListAppendLaunchKernel(self.handle, k.handle, &g, null, 0, null));
    }

    /// Closes the list for submission. No further appends are allowed until `reset`.
    pub fn close(self: *Recorder) runtime.Error!void {
        try runtime.check(self.ctx.api().zeCommandListClose(self.handle));
    }
};

/// A closed list that can be re-submitted as often as wanted.
pub const Recorded = struct {
    ctx: *const Context,
    handle: abi.CommandList,

    /// Closes `r` and takes ownership of its handle (the recorder must not be used afterwards).
    pub fn take(r: *Recorder) runtime.Error!Recorded {
        try r.close();
        const out: Recorded = .{ .ctx = r.ctx, .handle = r.handle };
        r.* = undefined;
        return out;
    }

    pub fn deinit(self: *Recorded) void {
        _ = self.ctx.api().zeCommandListDestroy(self.handle);
        self.* = undefined;
    }

    /// Re-submits the recorded commands on `q`. The caller synchronizes (or waits on an event) before reading.
    pub fn replay(self: *const Recorded, q: *Queue, signal: ?*const Event) runtime.Error!void {
        try q.execute(&.{self.handle}, signal);
    }

    /// Re-submits and waits for completion: the per-token path.
    pub fn replaySync(self: *const Recorded, q: *Queue) runtime.Error!void {
        try self.replay(q, null);
        try q.synchronize();
    }
};

/// The changing values of a decode step (token id, position, cache length), read by the kernels from device memory
/// instead of passed as arguments: a recorded list bakes its arguments, so per-token values must live somewhere
/// writable between replays. Host-visible and 64-byte aligned, so a replay can be preceded by a plain store.
pub const Params = struct {
    buffer: @import("memory.zig").Buffer,

    pub const slots = 16;
    pub const bytes = slots * @sizeOf(u32);

    /// Offsets of the well-known slots; a kernel reads `params[SLOT]`.
    pub const Slot = enum(u32) { token_id = 0, position = 1, cache_len = 2, rows = 3, capacity = 4 };

    pub fn init(ctx: *const Context) runtime.Error!Params {
        return .{ .buffer = try @import("memory.zig").Buffer.alloc(ctx, .shared, bytes) };
    }

    pub fn deinit(self: *Params) void {
        self.buffer.free();
        self.* = undefined;
    }

    /// Writes one slot; visible to the device without a copy because the buffer is shared.
    pub fn set(self: *Params, slot: Slot, value: u32) void {
        const p: [*]u32 = @ptrCast(@alignCast(self.buffer.ptr.?));
        p[@intFromEnum(slot)] = value;
    }

    pub fn get(self: *const Params, slot: Slot) u32 {
        const p: [*]const u32 = @ptrCast(@alignCast(self.buffer.ptr.?));
        return p[@intFromEnum(slot)];
    }

    pub fn ptr(self: *const Params) *anyopaque {
        return self.buffer.ptr.?;
    }
};

test "a recorder and a queue are distinct objects and both tear down" {
    // No card here: the geometry and slot arithmetic are the parts that can be checked without a driver.
    try std.testing.expectEqual(@as(usize, 64), Params.bytes);
    try std.testing.expectEqual(@as(u32, 2), @intFromEnum(Params.Slot.cache_len));
    const cfg = @import("launch.zig").Config{ .global = .{ .x = 256, .y = 3 }, .local = .{ .x = 256 } };
    try std.testing.expectEqual(@as(u32, 3), (try cfg.groups()).y);
}
