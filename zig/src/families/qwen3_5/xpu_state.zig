//! Per-stream committed state on an Intel B70, same layout and sizes as state.zig's Cache: per GatedDeltaNet layer the fp32
//! recurrence [16 heads][128 dv][128 dk] and the bf16 conv state [3][6144] (both start at zero), per attention layer bf16 key
//! and value caches [kv_head][capacity][256] (the layout the layout.cl kernels write), plus the bf16 logits row.
const std = @import("std");
const rt = @import("../../xpu/runtime.zig");
const c = @import("config.zig");

pub const delta_bytes = c.linear_heads * c.linear_dim * c.linear_dim * 4;
pub const conv_bytes = (c.conv_taps - 1) * c.conv_dim * 2;

/// A window's rows at most, and the streams a shared round serves at most (one segment a stream).
pub const window_rows = 16;
pub const max_segments = window_rows;

/// Bytes a stream's window snapshots take in the round scratch: `window_rows` recurrence rows then `window_rows` conv tails.
pub const snap_delta_bytes = window_rows * delta_bytes;
pub const snap_conv_bytes = window_rows * conv_bytes;

pub const Delta = struct { recurrence: rt.memory.Buffer, conv: rt.memory.Buffer };
pub const Attention = struct { keys: rt.memory.Buffer, values: rt.memory.Buffer };
pub const Mixer = union(enum) { delta: Delta, attention: Attention };

pub fn kvBytes(capacity: usize) usize {
    return capacity * c.kv_heads * c.head_dim * 2;
}

pub const Cache = struct {
    capacity: usize,
    len: usize = 0,
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
};
