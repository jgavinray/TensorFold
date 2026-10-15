//! Per-stream committed state on an Intel B70, same layout and sizes as state.zig's Cache: per GatedDeltaNet layer the fp32
//! recurrence [16 heads][128 dv][128 dk] and the bf16 conv state [3][6144] (both start at zero), per attention layer bf16 key
//! and value caches [kv_head][capacity][256] (the layout the layout.cl kernels write), plus the bf16 logits row.
const std = @import("std");
const rt = @import("../../xpu/runtime.zig");
const c = @import("config.zig");

pub const delta_bytes = c.linear_heads * c.linear_dim * c.linear_dim * 4;
pub const conv_bytes = (c.conv_taps - 1) * c.conv_dim * 2;

/// A window's rows at most (a stream's verify window), and the streams a shared round serves at most (one row a
/// stream at least, so `window_rows` segments). The snapshot scratch is indexed by absolute scratch row, as state.zig.
pub const window_rows = 16;
pub const max_segments = window_rows;

pub const Delta = struct { recurrence: rt.memory.Buffer, conv: rt.memory.Buffer };
pub const Attention = struct { keys: rt.memory.Buffer, values: rt.memory.Buffer };
pub const Mixer = union(enum) { delta: Delta, attention: Attention };

pub fn kvBytes(capacity: usize) usize {
    return capacity * c.kv_heads * c.head_dim * 2;
}

pub const Cache = struct {
    capacity: usize,
    len: usize = 0,
    /// The last recorded window: where its cache started, the scratch row of its first row, and its row count. `keep`
    /// rolls the recurrence/conv back to the first `path.len` rows of it (null when the last round did not record).
    last: ?struct { start: usize, base: usize, rows: usize } = null,
    blocks: [c.layers]Mixer,
    logits: rt.memory.Buffer,
    total_bytes: usize,
    made: usize = 0,
    has_logits: bool = false,

    /// Allocates everything on `s`'s card and zero-fills the recurrence and conv state (KV and logits are left unwritten, as in state.zig).
    pub fn init(s: *rt.session.Session, capacity: usize) rt.Error!Cache {
        var out: Cache = .{ .capacity = capacity, .blocks = undefined, .logits = undefined, .total_bytes = 0 };
        errdefer out.deinit();
        out.logits = try rt.memory.Buffer.alloc(&s.ctx, .device, c.vocab * 2);
        out.has_logits = true;
        out.total_bytes += c.vocab * 2;
        for (&out.blocks, 0..) |*b, i| {
            if (c.linear(i)) {
                var r = try rt.memory.Buffer.alloc(&s.ctx, .device, delta_bytes);
                errdefer r.free();
                var cv = try rt.memory.Buffer.alloc(&s.ctx, .device, conv_bytes);
                errdefer cv.free();
                const zero = [_]u8{0};
                try s.queue.fill(r.ptr, &zero, delta_bytes, null);
                try s.queue.fill(cv.ptr, &zero, conv_bytes, null);
                b.* = .{ .delta = .{ .recurrence = r, .conv = cv } };
                out.total_bytes += delta_bytes + conv_bytes;
            } else {
                var k = try rt.memory.Buffer.alloc(&s.ctx, .device, kvBytes(capacity));
                errdefer k.free();
                const v = try rt.memory.Buffer.alloc(&s.ctx, .device, kvBytes(capacity));
                b.* = .{ .attention = .{ .keys = k, .values = v } };
                out.total_bytes += 2 * kvBytes(capacity);
            }
            out.made = i + 1;
        }
        try s.queue.synchronize();
        return out;
    }

    pub fn deinit(self: *Cache) void {
        for (self.blocks[0..self.made]) |*b| switch (b.*) {
            .delta => |*d| {
                d.recurrence.free();
                d.conv.free();
            },
            .attention => |*x| {
                x.keys.free();
                x.values.free();
            },
        };
        if (self.has_logits) self.logits.free();
    }

    /// The segment finished its rows: with `record` the forward wrote every row's recurrence to `snap_rec` and the
    /// conv tail to `snap_conv`, so the committed state (the window's last row) is copied back into `cache.recurrence`
    /// and `cache.conv` — the recurrence is otherwise untouched (state_in == state_out read the old one). Remembers the
    /// window so `keep` can roll it back. Without `record` the kernel wrote the final state in place and there is
    /// nothing to roll back.
    pub fn commit(self: *Cache, s: *rt.session.Session, base: usize, rows: usize, record: bool, snap_rec: [c.layers]?*const anyopaque, snap_conv: *const anyopaque) rt.Error!void {
        if (!record) {
            self.last = null;
            self.len += rows;
            return;
        }
        for (&self.blocks, 0..) |*b, i| if (c.linear(i)) {
            try s.queue.copy(b.delta.recurrence.ptr, snap_rec[i].?, delta_bytes, null);
            try s.queue.copy(b.delta.conv.ptr, snap_conv, conv_bytes, null);
        };
        self.last = .{ .start = self.len, .base = base, .rows = rows };
        self.len += rows;
    }

    /// Roll back to the first `path.len` rows of the last recorded window: the recurrence and conv state as after
    /// exactly those rows, and the logits row of that row. The caller supplies the window row's snapshot addresses
    /// (`snapRec` per layer, `snapConv`) and its logits row, read from the round scratch.
    pub fn keep(self: *Cache, s: *rt.session.Session, path: []const u32, snap_rec: [c.layers]?*const anyopaque, snap_conv: *const anyopaque, logits_row: *const anyopaque) !void {
        const last = self.last orelse return error.NothingToKeep;
        if (path.len == 0 or path.len > last.rows) return error.NothingToKeep;
        for (path, 0..) |row, i| if (row != i) return error.UnsupportedQwenTree;
        for (&self.blocks, 0..) |*b, i| if (c.linear(i)) {
            try s.queue.copy(b.delta.recurrence.ptr, snap_rec[i].?, delta_bytes, null);
            try s.queue.copy(b.delta.conv.ptr, snap_conv, conv_bytes, null);
        };
        self.len = last.start + path.len;
        try s.queue.copy(self.logits.ptr, logits_row, c.vocab * 2, null);
        self.last = null;
    }
};
