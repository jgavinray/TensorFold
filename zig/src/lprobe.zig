//! TEMP probe: long-prompt prefill divergence — chunked 128 (the server's recipe) vs one-shot rows.
const std = @import("std");
const lanes = @import("lanes");
const rt = @import("xpu/runtime.zig");
const sys = @import("xpu/sys.zig");
const cfg = @import("families/qwen3_5/config.zig");
const xw = @import("families/qwen3_5/xpu_weights.zig");
const fwd = @import("families/qwen3_5/xpu_forward.zig");
const xb = @import("families/qwen3_5/xpu_backend.zig");
const c = @import("families/qwen3_5/config.zig");

const gpa = std.heap.page_allocator;

fn chunked(s: *rt.session.Session, m: *fwd.Model, x: *xb.Xpu, ids: []const u32, chunk: usize, cap: usize) ![]u16 {
    const e = try gpa.create(xb.Entry);
    e.* = .{ .cache = try @import("families/qwen3_5/xpu_state.zig").Cache.init(s, cap) };
    var at: usize = 0;
    while (at < ids.len) {
        const end: usize = @min(ids.len, at + chunk);
        try fwd.run(s, m, &x.scratch, &e.cache, ids[at..end], .{ .head = .last });
        at = end;
    }
    const row = try gpa.alloc(u16, c.vocab);
    try e.cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
    const len = e.cache.len;
    e.cache.deinit();
    gpa.destroy(e);
    std.debug.print("  chunk {d}: cache.len={d}\n", .{ chunk, len });
    return row;
}

fn topk(row: []const u16, k: usize) void {
    var idx: [16]usize = undefined;
    var vals: [16]f64 = undefined;
    const n = @min(k, idx.len);
    for (0..n) |i| {
        idx[i] = i;
        vals[i] = @as(f32, @bitCast(@as(u32, row[i]) << 16));
    }
    for (row, 0..) |w, id| {
        const v = @as(f32, @bitCast(@as(u32, w) << 16));
        for (0..n) |i| if (v > vals[i]) {
            var j: usize = n - 1;
            while (j > i) : (j -= 1) {
                vals[j] = vals[j - 1];
                idx[j] = idx[j - 1];
            }
            vals[i] = v;
            idx[i] = id;
            break;
        };
    }
    for (0..n) |i| std.debug.print("    #{d}: {d} {d:.4}\n", .{ i, idx[i], vals[i] });
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    const dir: []const u8 = if (argv.len > 1) argv[1] else "/storage01/models/Qwen3.5-2B-MLX-4bit";
    const ntok: usize = if (argv.len > 2) try std.fmt.parseInt(usize, argv[2], 10) else 10939;
    var s: rt.session.Session = undefined;
    s.open(.{}) catch |e| { std.debug.print("no card: {s}\n", .{@errorName(e)}); return 1; };
    defer s.close();
    var cb: [1024]u8 = undefined;
    const cpath = sys.zfmt(&cb, "{s}/config.json", .{dir}) orelse return 1;
    const ctext = sys.readFileAlloc(gpa, cpath, 1 << 22) orelse return 1;
    const config = try cfg.parse(gpa, ctext);
    var w = try xw.load(gpa, &s, dir, config.tied);
    defer w.deinit();
    var m: fwd.Model = .{ .weights = &w, .kernels = undefined };
    try m.kernels.load(&s, gpa);
    defer m.kernels.deinit();
    const cap = ntok + 256;
    const x = try xb.Xpu.init(gpa, &s, &m, .{ .capacity = cap, .qualify = false });
    defer x.deinit();

    const ids = try gpa.alloc(u32, ntok);
    for (ids, 0..) |*t, i| t.* = @intCast(1000 + (i % 30000));

    const a = try chunked(&s, &m, x, ids, 128, cap);
    std.debug.print("chunked 128 top logits (pos {d}):\n", .{ntok});
    topk(a, 5);
    const b = try chunked(&s, &m, x, ids, 64, cap);
    std.debug.print("chunked 64 top logits:\n", .{});
    topk(b, 5);
    var diff: f64 = 0;
    for (a, b) |p, q| {
        const xv = @as(f32, @bitCast(@as(u32, p) << 16));
        const yv = @as(f32, @bitCast(@as(u32, q) << 16));
        diff += @abs(xv - yv);
    }
    std.debug.print("sum|diff| chunk128 vs chunk64 = {d:.6}\n", .{diff});
    return 0;
}
