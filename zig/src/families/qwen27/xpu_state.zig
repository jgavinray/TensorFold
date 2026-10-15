//! Qwen 3.8 27B per-stream committed state on an Intel B70, same layout and sizes as the Metal path's state owners:
//! per GatedDeltaNet layer the fp32 recurrence [v_heads][dv][dk] (48 x 128 x 128) and the bf16 conv state
//! [conv_kernel - 1][gdnQkvDim] (3 x 10240), both starting at zero (gdn_pool.zig Budget.state / .history);
//! per attention layer bf16 key and value caches [kv_heads][capacity][head_dim] (attention.zig Cache, head stride
//! `capacity * 256`), plus the bf16 logits row.
const std = @import("std");
const rt = @import("../../xpu/runtime.zig");
const cfg = @import("config.zig");

pub fn deltaBytes(c: cfg.Config) usize {
    return c.v_heads * c.dv * c.dk * 4;
}

pub fn convBytes(c: cfg.Config) usize {
    return (c.conv_kernel - 1) * c.gdnQkvDim() * 2;
}

/// One attention layer's key or value cache: `capacity * kvDim()` bf16 elements.
pub fn kvBytes(c: cfg.Config, capacity: usize) usize {
    return capacity * c.kvDim() * 2;
}

pub fn logitsBytes(c: cfg.Config) usize {
    return c.vocab * 2;
}

pub const Gdn = struct { recurrence: rt.memory.Buffer, conv: rt.memory.Buffer };
pub const Attention = struct { keys: rt.memory.Buffer, values: rt.memory.Buffer };
pub const Mixer = union(cfg.Kind) { linear: Gdn, attention: Attention };

pub const Cache = struct {
    gpa: std.mem.Allocator,
    c: cfg.Config,
    capacity: usize,
    len: usize = 0,
    blocks: []Mixer,
    logits: rt.memory.Buffer,
    total_bytes: usize,
    made: usize = 0,
    has_logits: bool = false,

    /// Allocates everything on `s`'s card and zero-fills the recurrence and conv state
    /// (KV and logits are left unwritten, as in the Metal path's slotCache).
    pub fn init(gpa: std.mem.Allocator, s: *rt.session.Session, c: cfg.Config, capacity: usize) (rt.Error || std.mem.Allocator.Error)!Cache {
        if (capacity == 0 or c.layers == 0) return error.Invalid;
        const blocks = try gpa.alloc(Mixer, c.layers);
        errdefer gpa.free(blocks);
        var out: Cache = .{
            .gpa = gpa,
            .c = c,
            .capacity = capacity,
            .blocks = blocks,
            .logits = undefined,
            .total_bytes = 0,
        };
        errdefer out.deinit();
        out.logits = try rt.memory.Buffer.alloc(&s.ctx, .device, logitsBytes(c));
        out.has_logits = true;
        out.total_bytes += logitsBytes(c);
        for (out.blocks, 0..) |*b, i| {
            switch (c.kind(i)) {
                .linear => {
                    var r = try rt.memory.Buffer.alloc(&s.ctx, .device, deltaBytes(c));
                    errdefer r.free();
                    var cv = try rt.memory.Buffer.alloc(&s.ctx, .device, convBytes(c));
                    errdefer cv.free();
                    const zero = [_]u8{0};
                    try s.queue.fill(r.ptr, &zero, deltaBytes(c), null);
                    try s.queue.fill(cv.ptr, &zero, convBytes(c), null);
                    b.* = .{ .linear = .{ .recurrence = r, .conv = cv } };
                    out.total_bytes += deltaBytes(c) + convBytes(c);
                },
                .attention => {
                    var k = try rt.memory.Buffer.alloc(&s.ctx, .device, kvBytes(c, capacity));
                    errdefer k.free();
                    const v = try rt.memory.Buffer.alloc(&s.ctx, .device, kvBytes(c, capacity));
                    b.* = .{ .attention = .{ .keys = k, .values = v } };
                    out.total_bytes += 2 * kvBytes(c, capacity);
                },
            }
            out.made = i + 1;
        }
        try s.queue.synchronize();
        return out;
    }

    pub fn deinit(self: *Cache) void {
        for (self.blocks[0..self.made]) |*b| switch (b.*) {
            .linear => |*d| {
                d.recurrence.free();
                d.conv.free();
            },
            .attention => |*x| {
                x.keys.free();
                x.values.free();
            },
        };
        if (self.has_logits) self.logits.free();
        self.gpa.free(self.blocks);
        self.made = 0;
        self.has_logits = false;
    }

    /// Counts the buffers the cache owns: two per layer, plus the logits row.
    pub fn bufferCount(self: *const Cache) usize {
        return 2 * self.made + @intFromBool(self.has_logits);
    }

    /// Iterates every owned buffer, in layer order, for a pattern write and read-back.
    pub fn each(self: *Cache, out: []*rt.memory.Buffer) usize {
        var n: usize = 0;
        for (self.blocks[0..self.made]) |*b| switch (b.*) {
            .linear => |*d| {
                out[n] = &d.recurrence;
                out[n + 1] = &d.conv;
                n += 2;
            },
            .attention => |*x| {
                out[n] = &x.keys;
                out[n + 1] = &x.values;
                n += 2;
            },
        };
        if (self.has_logits) {
            out[n] = &self.logits;
            n += 1;
        }
        return n;
    }
};

test "27B state sizes follow the Metal path's slot layouts" {
    const c = try cfg.parse(std.testing.allocator,
        \\{"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text","hidden_size":5120,"intermediate_size":17408,
        \\"num_hidden_layers":64,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"vocab_size":248320,
        \\"linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,"linear_value_head_dim":128,
        \\"linear_conv_kernel_dim":4,"partial_rotary_factor":0.25,"eos_token_id":248044,"tie_word_embeddings":false},
        \\"quantization":{"bits":4,"group_size":64},"tie_word_embeddings":false}
    , null);
    try std.testing.expectEqual(@as(usize, 48 * 128 * 128 * 4), deltaBytes(c));
    try std.testing.expectEqual(@as(usize, 3 * 10240 * 2), convBytes(c));
    try std.testing.expectEqual(@as(usize, 4096 * 1024 * 2), kvBytes(c, 4096));
    try std.testing.expectEqual(@as(usize, 248320 * 2), logitsBytes(c));
}
