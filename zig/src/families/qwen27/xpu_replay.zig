//! One Qwen 3.8 27B decode step recorded into a replayable command list (the qwen3_5 module's port to the 27B forward).
//!
//! A decode step launches ~200 kernels for one token, so the submits dominate. This records the whole step once into a
//! regular list and replays it with one execute per token. Replay works because a decode step's only changing inputs
//! are three integers - the token id, the RoPE position and the cache length - and the forward passes every per-launch
//! scalar through the host-visible `Scratch.pool` (its `dims` slots, `ids`), never as a kernel argument.
//!
//! Scope: one token (`rows == 1`), one segment, `head = .last`. A different capacity or row count is a different shape
//! and is recorded afresh, which is what the per-shape table is for.
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

/// The `dims`-slot classification lives with the forward that produces it.
const DimsSlot = fwd.DimsSlot;

/// The shape a step was recorded for: a decode row count (always 1 here) and the cache capacity.
pub const Shape = struct { rows: usize, capacity: usize };

/// One recorded decode step for a fixed (rows, capacity) shape.
pub const Step = struct {
    list: Recorded,
    /// The `Scratch` the step was recorded against: its pool is written on every replay, so a replay is only valid
    /// while this exact scratch lives.
    sc: *fwd.Scratch,
    slots: []DimsSlot,
    /// The host pool as the recording left it (ids, conv windows, every `dims` slot): the kernels read the pool when
    /// the list executes, and any eager forward in between (this stream's first token, another stream) rewrites it.
    image: []u8,
    /// The values the step was recorded with; the replays that follow are checked against them.
    recorded: Values,
    /// Replays performed so far: the caller's cache length must be `recorded.cache_len + replays`.
    replays: usize = 0,

    pub fn deinit(self: *Step, gpa: std.mem.Allocator) void {
        self.list.deinit();
        gpa.free(self.slots);
        gpa.free(self.image);
        self.* = undefined;
    }
};

/// One recorded step per (rows, capacity) shape; `find` returns the live one, `stepFor` makes a bucket.
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

    /// The mutable step pointer for this shape, creating the bucket (with no step yet) when absent. Null only when the
    /// bucket exists but has never held a step.
    pub fn stepFor(self: *Replay, rows: usize, capacity: usize) ?*Step {
        for (self.buckets.items) |*b| {
            if (b.rows == rows and b.capacity == capacity) return if (b.step) |*st| st else null;
        }
        self.buckets.append(self.gpa, .{ .rows = rows, .capacity = capacity }) catch return null;
        return null;
    }
};

/// Restores the step's pool image, then writes the moving values: the token id and the position/cache-length elements
/// of the slots the forward marked (`Scratch.dimsMoving`).
pub fn writeValues(step: *Step, v: Values) void {
    const pool: [*]u8 = @ptrCast(step.sc.pool.ptr.?);
    @memcpy(pool[0..step.image.len], step.image);
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
    try fwd.recordDecode(s, &rec, m, sc, cache, tokens, recorded.position, &slots);
    const pool: [*]const u8 = @ptrCast(sc.pool.ptr.?);
    const image = try gpa.dupe(u8, pool[0..sc.poolUsed()]);
    errdefer gpa.free(image);
    const list = try Recorded.take(&rec);
    const owned = try slots.toOwnedSlice(gpa);
    return .{ .list = list, .sc = sc, .slots = owned, .image = image, .recorded = recorded };
}
