const std = @import("std");
const lanes = @import("lanes");
const rt = @import("xpu/runtime.zig");
const sys = @import("xpu/sys.zig");
const cfg = @import("families/qwen3_5/config.zig");
const xw = @import("families/qwen3_5/xpu_weights.zig");
const fwd = @import("families/qwen3_5/xpu_forward.zig");
const xs = @import("families/qwen3_5/xpu_state.zig");
const xb = @import("families/qwen3_5/xpu_backend.zig");
const gpa = std.heap.page_allocator;

var rowbuf: [200000]u16 = undefined;

fn rowOf(s: *rt.session.Session, cache: *xs.Cache) ![]const u16 {
    try cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(rowbuf[0..151936]));
    return rowbuf[0..151936];
}

fn case(x: *xb.Xpu, m: *fwd.Model, label: []const u8, prompt: []const u32, want: []const u32, record: bool, head: fwd.Head) !void {
    const s = x.session;
    var cache = try xs.Cache.init(s, 512);
    defer cache.deinit();
    try fwd.run(s, m, &x.scratch, &cache, prompt, .{ .head = .last });
    var pending = try xb.draw(gpa, try rowOf(s, &cache), null, prompt.len, &.{});
    std.debug.print("{s}: first {d} want {d} {s}\n", .{ label, pending, want[0], if (pending == want[0]) "ok" else "BAD" });
    for (1..8) |i| {
        const ids = [_]u32{pending};
        try fwd.runSegments(s, m, &x.scratch, &.{.{ .cache = &cache, .rows = 1 }}, &ids, record, .{ .head = head });
        pending = try xb.draw(gpa, try rowOf(s, &cache), null, prompt.len + i, &.{});
        std.debug.print("{s} step {d}: got {d} want {d} {s}\n", .{ label, i, pending, want[i], if (pending == want[i]) "ok" else "BAD" });
        if (i == 1) {
            const rec = cache.blocks[0].delta.recurrence;
            var h: [16]u8 = undefined;
            try rec.download(&s.queue, 0, h[0..16]);
            std.debug.print("   rec[0..16]: {any}\n", .{h});
            var cv: [16]u8 = undefined;
            try cache.blocks[0].delta.conv.download(&s.queue, 0, cv[0..16]);
            std.debug.print("   conv[0..16]: {any}\n", .{cv});
        }
    }
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    const dir: []const u8 = if (argv.len > 1) argv[1] else "/storage01/models/Qwen3.5-2B-MLX-4bit";
    const refdir: []const u8 = if (argv.len > 2) argv[2] else "/storage01/qwen-flash-on-intel/exp/ref/qwen35-2b-mlx4-fp32";
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
    const x = try xb.Xpu.init(gpa, &s, &m, .{ .capacity = 512, .qualify = false });
    defer x.deinit();
    var gb: [1024]u8 = undefined;
    const gpath = sys.zfmt(&gb, "{s}/greedy-fibonacci-raw.json", .{refdir}) orelse return 1;
    const text = sys.readFileAlloc(gpa, gpath, 1 << 24) orelse return 1;
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    const entry = doc.value.object.get("fibonacci-raw").?;
    var prompt: std.ArrayList(u32) = .empty;
    var want: std.ArrayList(u32) = .empty;
    for (entry.object.get("prompt_tokens").?.array.items) |v| try prompt.append(gpa, @intCast(v.integer));
    for (entry.object.get("tokens").?.array.items) |v| try want.append(gpa, @intCast(v.integer));

    // qualify-style: 16-row window vs 16 one-row runs, compare committed logits
    {
        var wid = try xs.Cache.init(&s, 64 + 0);
        defer wid.deinit();
        var one = try xs.Cache.init(&s, 64 + 1);
        defer one.deinit();
        var w16: [16]u32 = undefined;
        for (0..16) |k| w16[k] = prompt.items[k % prompt.items.len];
        // prior rows, as xpu_qualify's first loop leaves them (even index -> 1 row, odd -> 3)
        try fwd.run(&s, &m, &x.scratch, &wid, prompt.items[0..1], .{ .head = .last });
        try fwd.run(&s, &m, &x.scratch, &one, prompt.items[0..1], .{ .head = .last });
        var e = xb.Entry{ .cache = wid };
        try e.window(&s, &m, &x.scratch, &w16);
        var got: [151936]u16 = undefined;
        try x.scratch.logits.download(&s.queue, 0, std.mem.sliceAsBytes(got[0..151936]));
        var row: [151936]u16 = undefined;
        for (0..16) |r| {
            try fwd.run(&s, &m, &x.scratch, &one, w16[r..][0..1], .{ .head = .all });
            try x.scratch.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row[0..151936]));
        }
        var a: [151936]u16 = undefined;
        var b: [151936]u16 = undefined;
        try wid.logits.download(&s.queue, 0, std.mem.sliceAsBytes(a[0..151936]));
        try one.logits.download(&s.queue, 0, std.mem.sliceAsBytes(b[0..151936]));
        std.debug.print("qualify logits: window {any}\n              one-row {any}\n", .{ a[0..6], b[0..6] });
    }
    try case(x, &m, "run last/rec0 ", prompt.items, want.items, false, .last);
    try case(x, &m, "run all /rec0 ", prompt.items, want.items, false, .all);
    try case(x, &m, "run all /rec1 ", prompt.items, want.items, true, .all);
    try case(x, &m, "run last/rec1 ", prompt.items, want.items, true, .last);
    return 0;
}
