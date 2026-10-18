//! One 27B decode step recorded into a replayable command list (S01's lever applied to Qwen 3.8 27B).
//!
//! A 27B decode step launches ~660 kernels for one token; through the immediate list every launch is its own
//! host->device submit, and the step's own weight traffic is ~4 us per small projection (FINDINGS §9), so the
//! submits dominate. This records the whole step once into a regular list (`xpu/cmdlist.zig`) and replays it with
//! one `zeCommandQueueExecuteCommandLists` per token.
//!
//! Replay works because a decode step's only changing inputs are three integers - the token id, the RoPE position
//! and the cache length - and the forward passes every per-launch scalar through the host-visible `Scratch.pool`
//! (its `dims` slots, `ids`), never as a kernel argument. A recorded list bakes the *addresses* of those pool
//! slots, and the forward allocates its `dims` slots in call order, so for a fixed shape each slot index holds the
//! same field every token. The recorder therefore captures, per `dims` slot, which element carries the position or
//! the cache length, and rewrites exactly those before each replay.
//!
//! Scope: one token (`rows == 1`), one segment, `head = .last`. The grid of every 27B kernel depends only on the
//! capacity (`attn_partial`'s chunk count is derived from it), so one recorded step per capacity serves every
//! length; a different capacity is a different shape and is recorded afresh, which is what the table is for.
//! The recorded step must be produced with the same `Fuse` set as the eager path it replaces: a fused variant
//! captures different kernels (and, for `norm_qmm`, one fewer launch), so mixing them would replay a different
//! graph than the one that was verified.
const std = @import("std");
const rt = @import("../../xpu/runtime.zig");
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

const DimsSlot = fwd.DimsSlot;

/// The shape a step was recorded for: a decode row count (always 1 here) and the cache capacity.
pub const Shape = struct { rows: usize, capacity: usize };

/// One recorded decode step for a fixed (rows, capacity) shape.
pub const Step = struct {
    list: Recorded,
    /// The `Scratch` the step was recorded against: its pool is written on every replay, so a replay is only valid
    /// while this exact scratch lives. Kept so a mismatched scratch is refused rather than misrun.
    sc: *fwd.Scratch,
    slots: []DimsSlot,
    /// The fusions the recorded launches were captured with; a replay must match the eager path that was gated.
    fuse: fwd.Fuse,
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

/// One recorded step per (rows, capacity) shape; `find` returns the live one, `put` records or replaces it.
pub const Replay = struct {
    gpa: std.mem.Allocator,
    buckets: std.ArrayList(Bucket) = .empty,

    const Bucket = struct {
        rows: usize,
        capacity: usize,
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
    pub fn find(self: *Replay, rows: usize, capacity: usize) ?*Step {
        for (self.buckets.items) |*b| {
            if (b.rows == rows and b.capacity == capacity) return if (b.step) |*st| st else null;
        }
        return null;
    }

    /// Registers `step` under (rows, capacity), replacing any earlier one (whose list is freed).
    pub fn put(self: *Replay, rows: usize, capacity: usize, step: Step) !void {
        for (self.buckets.items) |*b| {
            if (b.rows == rows and b.capacity == capacity) {
                if (b.step) |*old| old.deinit(self.gpa);
                b.step = step;
                return;
            }
        }
        try self.buckets.append(self.gpa, .{ .rows = rows, .capacity = capacity, .step = step });
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
pub fn replayStep(step: *Step, q: *rt.cmdlist.Queue, v: Values) !void {
    if (v.cache_len != step.recorded.cache_len + @as(u32, @intCast(step.replays))) return error.ReplayValueOutOfOrder;
    writeValues(step, v);
    try step.list.replaySync(q);
    step.replays += 1;
}

/// Records the one-token decode step for `cache`'s current length into a closed list over `sc`'s buffers.
/// `tokens` is the single token the step decodes; `sc.dims_used` is reset and every `dims` slot is captured with
/// the fields the replay must rewrite. `fuse` must equal the eager path's fusion set.
pub fn recordStep(
    s: *rt.session.Session,
    m: *fwd.Model,
    sc: *fwd.Scratch,
    cache: *xs.Cache,
    tokens: []const u32,
    fuse: fwd.Fuse,
    gpa: std.mem.Allocator,
) !Step {
    if (tokens.len != 1) return error.ReplayNeedsOneRow;
    var rec = try Recorder.init(&s.ctx);
    errdefer rec.deinit();
    const recorded: Values = .{ .token = tokens[0], .position = @intCast(cache.len), .cache_len = @intCast(cache.len) };
    var slots: std.ArrayList(DimsSlot) = .empty;
    errdefer slots.deinit(gpa);
    try fwd.recordDecode(&rec, m, sc, cache, tokens, recorded.position, &slots, fuse);
    const list = try Recorded.take(&rec);
    const owned = try slots.toOwnedSlice(gpa);
    return .{ .list = list, .sc = sc, .slots = owned, .fuse = fuse, .recorded = recorded };
}
