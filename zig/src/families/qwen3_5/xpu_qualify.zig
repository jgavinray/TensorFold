//! XPU port of qualify.zig: admit wider lanes only after real-weight window rows and committed caches match one-row
//! execution byte for byte on this card (including a partial `keep`).
const std = @import("std");
const rt = @import("../../xpu/runtime.zig");
const xs = @import("xpu_state.zig");
const fwd = @import("xpu_forward.zig");
const xb = @import("xpu_backend.zig");
const c = @import("config.zig");

fn same(gpa: std.mem.Allocator, s: *rt.session.Session, a: *const rt.memory.Buffer, b: *const rt.memory.Buffer, a_off: usize, b_off: usize, bytes: usize) !bool {
    const x = try gpa.alloc(u8, bytes);
    defer gpa.free(x);
    const y = try gpa.alloc(u8, bytes);
    defer gpa.free(y);
    try a.download(&s.queue, a_off, x);
    try b.download(&s.queue, b_off, y);
    return std.mem.eql(u8, x, y);
}

pub fn equal(gpa: std.mem.Allocator, s: *rt.session.Session, a: *const xs.Cache, b: *const xs.Cache) !void {
    if (a.len != b.len) return error.QwenCacheLengthMismatch;
    if (!try same(gpa, s, &a.logits, &b.logits, 0, 0, c.vocab * 2)) return error.QwenCommittedLogitsMismatch;
    for (a.blocks, b.blocks) |x, y| switch (x) {
        .delta => |d| {
            if (!try same(gpa, s, &d.recurrence, &y.delta.recurrence, 0, 0, xs.delta_bytes) or
                !try same(gpa, s, &d.conv, &y.delta.conv, 0, 0, xs.conv_bytes)) return error.QwenRecurrenceMismatch;
        },
        .attention => |kv| {
            for ([_]*const rt.memory.Buffer{ &kv.keys, &kv.values }, [_]*const rt.memory.Buffer{ &y.attention.keys, &y.attention.values }) |ab, bb| {
                for (0..c.kv_heads) |h| {
                    const ao = h * a.capacity * c.head_dim * 2;
                    const bo = h * b.capacity * c.head_dim * 2;
                    if (!try same(gpa, s, ab, bb, ao, bo, a.len * c.head_dim * 2)) return error.QwenAttentionCacheMismatch;
                }
            }
        },
    };
}

pub fn check(gpa: std.mem.Allocator, s: *rt.session.Session, m: *fwd.Model, wide: *fwd.Scratch) !void {
    var solo = try fwd.Scratch.init(s, 1, 128, 1);
    defer solo.deinit();
    var caches: [4]xb.Entry = undefined;
    var initialized: usize = 0;
    defer for (caches[0..initialized]) |*e| e.deinit();
    for (&caches) |*e| {
        e.* = .{ .cache = try xs.Cache.init(s, 64 + initialized) };
        initialized += 1;
    }
    const ids = [_]u32{ 825, 264, 15352, 4829, 9944, 13, 271, 279, 854, 22373, 825, 264, 15352, 4829, 9944, 13, 42, 264, 271, 279, 15352, 854, 13, 4829, 825, 22373, 9944, 264, 271, 42, 13, 279 };
    for (&caches, 0..) |*e, i| {
        const rows: usize = if (i % 2 == 0) 1 else 3;
        try fwd.run(s, m, wide, &e.cache, ids[0..rows], .{ .head = .last });
    }
    const got = try gpa.alloc(u16, 16 * c.vocab);
    defer gpa.free(got);
    const one = try gpa.alloc(u16, c.vocab);
    defer gpa.free(one);
    for (0..2) |stream| {
        try caches[stream].window(s, m, wide, ids[stream * 16 ..][0..16]);
        try wide.logits.download(&s.queue, 0, std.mem.sliceAsBytes(got));
        for (0..16) |r| {
            try fwd.run(s, m, &solo, &caches[stream + 2].cache, ids[stream * 16 + r ..][0..1], .{ .head = .all });
            try solo.logits.download(&s.queue, 0, std.mem.sliceAsBytes(one));
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(got[r * c.vocab ..][0..c.vocab]), std.mem.sliceAsBytes(one))) return error.QwenWindowLogitsMismatch;
        }
        try equal(gpa, s, &caches[stream].cache, &caches[stream + 2].cache);
    }
    try caches[0].keep(s, m, wide, &.{ 0, 1, 2 });
    try caches[1].keep(s, m, wide, &.{0});
    for (0..2) |stream| {
        var fresh = try xs.Cache.init(s, 64);
        defer fresh.deinit();
        const prefix: usize = if (stream == 0) 1 else 3;
        const kept: usize = if (stream == 0) 3 else 1;
        for (0..prefix) |r| try fwd.run(s, m, &solo, &fresh, ids[r..][0..1], .{ .head = .last });
        for (0..kept) |r| try fwd.run(s, m, &solo, &fresh, ids[stream * 16 + r ..][0..1], .{ .head = .all });
        try equal(gpa, s, &caches[stream].cache, &fresh);
    }
}
