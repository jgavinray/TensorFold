//! One decode step recorded into a replayable command list.
//!
//! A decode step launches ~100 kernels for one token. Through the immediate list every launch is its own
//! host->device submit, and the step's own weight traffic is only ~4 us per small projection (FINDINGS §9), so the
//! submits dominate. This records the whole step once into a regular list (see xpu/cmdlist.zig) and replays it with
//! one `zeCommandQueueExecuteCommandLists` per token.
//!
//! Replay works because a decode step's only changing inputs are three integers - the token id, the RoPE position
//! and the cache length - and the forward already passes every per-launch scalar through the host-visible
//! `Scratch.pool` (its `dims` slots, `ids`, `windows`), never as a kernel argument. A recorded list bakes the
//! *addresses* of those pool slots, and the forward allocates its `dims` slots in call order, so for a fixed shape
//! each slot index holds the same field every token. The recorder therefore captures, per `dims` slot, which
//! element carries the position or the cache length, and rewrites exactly those before each replay.
//!
//! Scope: one token (`rows == 1`), one segment, `head = .last`. A different capacity or row count is a different
//! shape and is recorded afresh, which is what the per-shape table is for.
const std = @import("std");
const rt = @import("../../xpu/runtime.zig");
const c = @import("config.zig");
const xs = @import("xpu_state.zig");
const fwd = @import("xpu_forward.zig");

const Recorded = rt.cmdlist.Recorded;
const Recorder = rt.cmdlist.Recorder;

/// The changing scalars of one decode token, as the kernels read them.
pub const Values = struct {
    /// The token being decoded.
    token: u32,
    /// RoPE position: the cache length before this token.
    position: u32,
    /// Cache length before this token.
    cache_len: u32,
};

/// A `dims` slot the recorded step reads, and which of its elements carries a value that moves per token.
const DimsSlot = struct {
    /// Byte offset of the slot from the pool's start.
    offset: usize,
    /// Index of the element inside the slot that holds the RoPE position, or null.
    position_at: ?usize,
    /// Index of the element inside the slot that holds the cache length, or null.
    cache_len_at: ?usize,
};

/// One recorded decode step for a fixed (capacity, rows) shape.
pub const Step = struct {
    list: Recorded,
    /// The `Scratch` the step was recorded against: its pool is written on every replay, so a replay is only valid
    /// while this exact scratch lives. Kept so a mismatched scratch is refused rather than misrun.
    sc: *fwd.Scratch,
    slots: []DimsSlot,
    /// The values the step was recorded with; the replays that follow are checked against them.
    recorded: Values,
    /// Replays performed so far: the caller's cache length must be `recorded.cache_len + replays`.
    replays: usize = 0,

    pub fn deinit(self: *Step, gpa: std.mem.Allocator) void {
        self.list.deinit();
        gpa.free(self.slots);
        self.* = undefined;
    }
};

/// One recorded step per (capacity, rows) shape; `find` returns the live one, `put` records or replaces it.
pub const Replay = struct {
    gpa: std.mem.Allocator,
    buckets: std.ArrayList(Bucket) = .empty,

    const Bucket = struct {
        capacity: usize,
        rows: usize,
        step: ?Step = null,
    };

    pub fn init(gpa: std.mem.Allocator) Replay {
        return .{ .gpa = gpa };
    }

    pub fn deinit(self: *Replay) void {
        for (self.buckets.items) |*b| if (b.step) |*st| st.deinit(self.gpa);
        self.buckets.deinit(self.gpa);
        self.* = undefined;
    }

    /// The recorded step for this shape, or null when none has been recorded yet.
    pub fn find(self: *Replay, capacity: usize, rows: usize) ?*Step {
        for (self.buckets.items) |*b| {
            if (b.capacity == capacity and b.rows == rows) return if (b.step) |*st| st else null;
        }
        return null;
    }

    /// Registers `step` under (capacity, rows), replacing any earlier one (whose list is freed).
    pub fn put(self: *Replay, capacity: usize, rows: usize, step: Step) !void {
        for (self.buckets.items) |*b| {
            if (b.capacity == capacity and b.rows == rows) {
                if (b.step) |*old| old.deinit(self.gpa);
                b.step = step;
                return;
            }
        }
        try self.buckets.append(self.gpa, .{ .capacity = capacity, .rows = rows, .step = step });
    }
};

/// Rewrites the step's moving values in place: the token id and the position/cache-length elements of every `dims`
/// slot. The recorded list's pointers do not move, so a replay picks up the new values.
pub fn writeValues(step: *Step, v: Values) void {
    const pool: [*]u8 = @ptrCast(step.sc.pool.ptr.?);
    const ids: [*]u32 = @ptrCast(@alignCast(pool));
    ids[0] = v.token;
    for (step.slots) |slot| {
        const d: [*]i32 = @ptrCast(@alignCast(pool + slot.offset));
        if (slot.position_at) |i| d[i] = @intCast(v.position);
        if (slot.cache_len_at) |i| d[i] = @intCast(v.cache_len);
    }
}

/// Replays a recorded step: rewrites the moving values, submits the one closed list, and waits for it.
pub fn replayStep(s: *rt.session.Session, step: *Step, q: *rt.cmdlist.Queue, v: Values) !void {
    _ = s;
    if (v.cache_len != step.recorded.cache_len + @as(u32, @intCast(step.replays))) return error.ReplayValueOutOfOrder;
    writeValues(step, v);
    try step.list.replaySync(q);
    step.replays += 1;
}

/// Records the one-token decode step for `cache`'s current length into a closed list over `sc`'s buffers.
/// `tokens` is the single token the step decodes; `sc.dims_used` is reset and every `dims` slot is captured with the
/// fields the replay must rewrite.
pub fn recordStep(
    s: *rt.session.Session,
    m: *fwd.Model,
    sc: *fwd.Scratch,
    cache: *xs.Cache,
    tokens: []const u32,
    gpa: std.mem.Allocator,
) !Step {
    if (tokens.len != 1) return error.ReplayNeedsOneRow;
    var rec = try Recorder.init(&s.ctx);
    errdefer rec.deinit();
    const recorded: Values = .{ .token = tokens[0], .position = @intCast(cache.len), .cache_len = @intCast(cache.len) };
    var slots: std.ArrayList(DimsSlot) = .empty;
    errdefer slots.deinit(gpa);
    try fwd.recordDecode(&rec, m, sc, cache, tokens, recorded.position, &slots);
    const list = try Recorded.take(&rec);
    const owned = try slots.toOwnedSlice(gpa);
    return .{ .list = list, .sc = sc, .slots = owned, .recorded = recorded };
}
