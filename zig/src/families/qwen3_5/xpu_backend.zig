//! Qwen3.5-2B on an Intel B70 behind the lane engine's backend interface (`struct Xpu`, as backend.zig's `Metal`).
//! The XPU forward appends one segment per call and records no per-row snapshots, so a multi-row window saves the
//! recurrence/conv state first and `keep` of a shorter prefix restores it and re-runs the kept rows.
const std = @import("std");
const lanes = @import("lanes");
const rt = @import("../../xpu/runtime.zig");
const c = @import("config.zig");
const xs = @import("xpu_state.zig");
const fwd = @import("xpu_forward.zig");
const be = lanes.backend;

pub const window_rows = 16;
pub const Options = struct { capacity: usize, chunk: usize = 128, streams: u32 = 8, qualify: bool = true };

/// A stream's cache; the rollback a multi-row window needs lives in the round scratch (the forward records the
/// window's snapshots, and `Cache.last` names the row range `keep` reads back).
pub const Entry = struct {
    cache: xs.Cache,

    pub fn deinit(self: *Entry) void {
        self.cache.deinit();
    }

    /// Forward `ids` (pending then drafts) as one window with every row's logits in `sc.logits`. `Cache.commit`
    /// remembers the window (row range and snapshots) so a later `keep` can roll it back.
    pub fn window(self: *Entry, s: *rt.session.Session, m: *fwd.Model, sc: *fwd.Scratch, ids: []const u32) !void {
        if (ids.len == 0 or ids.len > window_rows) return error.UnsupportedQwenWindow;
        try fwd.runSegments(s, m, sc, &.{.{ .cache = &self.cache, .rows = ids.len }}, ids, ids.len > 1, .{ .head = .all });
    }

    /// Roll back to the first `n` rows of the last recorded window (logits row and recurrence as after exactly those
    /// rows). The window's row range comes from `cache.last` (set by `Cache.commit`), so this serves both the
    /// single-stream `window` path and a shared round's `verify` (which commits through `runSegments`).
    pub fn keep(self: *Entry, s: *rt.session.Session, m: *fwd.Model, sc: *fwd.Scratch, path: []const u32) !void {
        _ = m;
        const last = self.cache.last orelse return error.NothingToKeep;
        if (path.len == 0 or path.len > last.rows) return error.NothingToKeep;
        for (path, 0..) |row, i| if (row != i) return error.UnsupportedQwenTree;
        if (path.len != last.rows) {
            const row = last.base + path.len - 1;
            var rec: [c.layers]?*const anyopaque = @splat(null);
            var cv: [c.layers]?*const anyopaque = @splat(null);
            for (&rec, &cv, 0..) |*r, *v, i| if (c.linear(i)) {
                r.* = sc.snapDelta(i, row);
                v.* = sc.snapConv(i, row);
            };
            try self.cache.keep(s, path, rec, cv, offLogits(sc, row));
        }
    }
};

fn offLogits(sc: *fwd.Scratch, row: usize) *const anyopaque {
    return @ptrFromInt(@intFromPtr(sc.logits.ptr.?) + row * c.vocab * 2);
}

pub const Xpu = struct {
    gpa: std.mem.Allocator,
    session: *rt.session.Session,
    model: *fwd.Model,
    options: Options,
    scratch: fwd.Scratch,
    width: u32 = 1,
    entries: std.AutoHashMapUnmanaged(*lanes.Stream, *Entry) = .empty,
    row: []u16, // host copy of one logits row
    rows: []u16, // host copy of a window's logits

    pub fn init(gpa: std.mem.Allocator, session: *rt.session.Session, model: *fwd.Model, options: Options) !*Xpu {
        const out = try gpa.create(Xpu);
        errdefer gpa.destroy(out);
        const row = try gpa.alloc(u16, c.vocab);
        errdefer gpa.free(row);
        const rows = try gpa.alloc(u16, xs.max_segments * window_rows * c.vocab);
        errdefer gpa.free(rows);
        out.* = .{ .gpa = gpa, .session = session, .model = model, .options = options, .row = row, .rows = rows, .scratch = try fwd.Scratch.init(session, options.chunk, options.capacity, xs.max_segments * window_rows) };
        errdefer out.scratch.deinit();
        if (options.qualify) {
            @import("xpu_qualify.zig").check(gpa, session, model, &out.scratch) catch |err| switch (err) {
                error.QwenCacheLengthMismatch, error.QwenRecurrenceMismatch, error.QwenAttentionCacheMismatch, error.QwenWindowLogitsMismatch, error.QwenCommittedLogitsMismatch => {
                    std.log.warn("Qwen wider lanes failed the local exactness check ({s}); keeping one-row lane rounds", .{@errorName(err)});
                    return out;
                },
                else => return err,
            };
            out.width = window_rows;
        }
        return out;
    }

    pub fn deinit(self: *Xpu) void {
        var it = self.entries.valueIterator();
        while (it.next()) |e| {
            e.*.deinit();
            self.gpa.destroy(e.*);
        }
        self.entries.deinit(self.gpa);
        self.scratch.deinit();
        self.gpa.free(self.row);
        self.gpa.free(self.rows);
        self.gpa.destroy(self);
    }

    pub fn facts(self: *const Xpu) lanes.Model {
        return .{ .exact_width = self.width, .hidden_rows = self.width > 1, .batch_rows = xs.max_segments * window_rows, .max_streams = self.options.streams, .streams_exact = true };
    }

    pub fn backend(self: *Xpu) be.Backend {
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Xpu {
        return @ptrCast(@alignCast(ptr));
    }

    fn entryOf(self: *Xpu, stream: *lanes.Stream) !*Entry {
        return self.entries.get(stream) orelse error.UnknownQwenStream;
    }

    fn logits(self: *Xpu, e: *Entry) ![]const u16 {
        try e.cache.logits.download(&self.session.queue, 0, std.mem.sliceAsBytes(self.row));
        return self.row;
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        const ids = stream.prompt();
        if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity or self.entries.contains(stream)) return error.InvalidQwenPrompt;
        const e = try self.gpa.create(Entry);
        errdefer self.gpa.destroy(e);
        const capacity = @min(self.options.capacity, ids.len + stream.max_new + window_rows);
        e.* = .{ .cache = try xs.Cache.init(self.session, capacity) };
        errdefer e.deinit();
        var at: usize = 0;
        while (at < ids.len) {
            var end: usize = @min(ids.len, at + self.options.chunk);
            for (stream.chunks) |boundary| if (boundary > at) {
                end = @min(end, boundary);
                break;
            };
            try fwd.run(self.session, self.model, &self.scratch, &e.cache, ids[at..end], .{ .head = .last });
            at = end;
        }
        try self.entries.put(self.gpa, stream, e);
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        const self = cast(ptr);
        const e = try self.entryOf(stream);
        if (position != e.cache.len) return error.InvalidQwenPosition;
        return try draw(self.gpa, try self.logits(e), stream.sampling, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const self = cast(ptr);
        const e = try self.entryOf(stream);
        if (position != e.cache.len) return error.InvalidQwenPosition;
        return try draw(self.gpa, try self.logits(e), stream.sampling, position, mask);
    }

    fn queueFn(ptr: *anyopaque, stream: *lanes.Stream, feed: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        const e = try self.entryOf(stream);
        if (position != e.cache.len + 1) return error.InvalidQwenPosition;
        const token: u32 = switch (feed) {
            .value => |v| v,
            .handle => |h| @intCast(h),
        };
        try fwd.run(self.session, self.model, &self.scratch, &e.cache, &.{token}, .{ .head = .last });
        return try draw(self.gpa, try self.logits(e), stream.sampling, position, &.{});
    }

    fn readFn(_: *anyopaque, handle: u64) !u32 {
        if (handle >= c.vocab) return error.InvalidQwenToken;
        return @intCast(handle);
    }

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        if (windows.len != out.len or windows.len > xs.max_segments) return error.InvalidQwenWindows;
        var segments: [xs.max_segments]fwd.Segment = undefined;
        var ids: [window_rows * xs.max_segments]u32 = undefined;
        var total: usize = 0;
        for (windows, out, 0..) |w, o, i| {
            const n = w.rows();
            if (w.parents != null or w.held != 0 or n > window_rows or w.positions.len != n) return error.UnsupportedQwenWindow;
            if (w.masks.len != 0 and w.masks.len != n * mask_words) return error.InvalidQwenMasks;
            if (o.sampled.len != n or o.drafts.len != w.tokens.len) return error.InvalidQwenOutputs;
            const e = try self.entryOf(w.stream);
            for (w.positions, 0..) |p, r| if (p != e.cache.len + r + 1) return error.InvalidQwenPosition;
            // The row budget is `max_segments * window_rows` (one window a stream, each at most `window_rows` rows):
            // checked before the writes so a round of narrow windows can never run past `ids`.
            if (total + n > ids.len) return error.InvalidQwenWindows;
            segments[i] = .{ .cache = &e.cache, .rows = n };
            ids[total] = w.pending;
            @memcpy(ids[total + 1 ..][0..w.tokens.len], w.tokens);
            total += n;
        }
        // One forward over every window: each stream's rows are its own segment, snapshots recorded so `keep` can roll back.
        try fwd.runSegments(self.session, self.model, &self.scratch, segments[0..windows.len], ids[0..total], true, .{ .head = .all });
        const bytes = std.mem.sliceAsBytes(self.rows[0 .. total * c.vocab]);
        try self.scratch.logits.download(&self.session.queue, 0, bytes);
        var base: usize = 0;
        for (windows, out) |w, o| {
            const n = w.rows();
            for (o.sampled, w.positions, 0..) |*t, p, r| {
                const mask = if (w.masks.len > 0) w.masks[r * mask_words ..][0..mask_words] else &.{};
                t.* = try draw(self.gpa, self.rows[(base + r) * c.vocab ..][0..c.vocab], w.stream.sampling, p, mask);
            }
            @memcpy(o.drafts, w.tokens);
            base += n;
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        for (windows, paths) |w, path| try (try self.entryOf(w.stream)).keep(self.session, self.model, &self.scratch, path);
    }

    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) !void {
        for (requests) |r| if (r.depth != 0) return error.QwenCheckpointHasNoDraftHead;
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        if (self.entries.fetchRemove(stream)) |kv| {
            kv.value.deinit();
            self.gpa.destroy(kv.value);
        }
    }
};

/// A row's allowed-token bits (structured output): one bit a logit.
const mask_words = (c.vocab + 31) / 32;

pub fn value(word: u16) f64 {
    return @as(f32, @bitCast(@as(u32, word) << 16));
}

/// The unchanged top-k candidates for the shared fp64 keyed sampler (same rule as backend.zig's `draw`).
pub fn draw(gpa: std.mem.Allocator, logits: []const u16, settings: ?lanes.Sampling, position: u64, mask: []const u32) !u32 {
    var best: ?usize = null;
    var allowed: usize = 0;
    for (logits, 0..) |word, id| {
        const x = value(word);
        if (!std.math.isFinite(x)) return error.NonfiniteQwenLogits;
        if (mask.len > 0 and !lanes.grammar.allows(mask, id)) continue;
        allowed += 1;
        if (best == null or x > value(logits[best.?])) best = id;
    }
    if (best == null) return error.NoAllowedQwenToken;
    const sampling = settings orelse return @intCast(best.?);
    if (sampling.temperature <= 0) return @intCast(best.?);
    const k = if (sampling.top_k == 0) allowed else @min(allowed, sampling.top_k);
    const values = try gpa.alloc(f64, k);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, k);
    defer gpa.free(ids);
    var filled: usize = 0;
    for (logits, 0..) |word, id| {
        if (mask.len > 0 and !lanes.grammar.allows(mask, id)) continue;
        const x = value(word);
        if (k == allowed) {
            values[filled] = x;
            ids[filled] = id;
            filled += 1;
            continue;
        }
        if (filled == k and x <= values[k - 1]) continue;
        var at = @min(filled, k - 1);
        while (at > 0 and x > values[at - 1]) : (at -= 1) {
            values[at] = values[at - 1];
            ids[at] = ids[at - 1];
        }
        values[at] = x;
        ids[at] = id;
        filled = @min(filled + 1, k);
    }
    return @intCast(try lanes.sampling.choose(gpa, values, ids, position, sampling));
}
