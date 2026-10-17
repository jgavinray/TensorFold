//! TEMP probe: does the committed row after a long chunked prefill depend on the cache CAPACITY (with the same
//! prompt length)? If it does, the bug is capacity-derived (scratch/geometry), not length- or chunk-derived.
const std = @import("std");
const rt = @import("xpu/runtime.zig");
const sys = @import("xpu/sys.zig");
const cfg = @import("families/qwen3_5/config.zig");
const xw = @import("families/qwen3_5/xpu_weights.zig");
const fwd = @import("families/qwen3_5/xpu_forward.zig");
const xb = @import("families/qwen3_5/xpu_backend.zig");
const xs = @import("families/qwen3_5/xpu_state.zig");
const tk = @import("core").tokenizer;

const gpa = std.heap.page_allocator;

fn val(w: u16) f32 {
    return @as(f32, @bitCast(@as(u32, w) << 16));
}

fn argmax(row: []const u16) usize {
    var best: usize = 0;
    for (row, 0..) |w, i| if (val(w) > val(row[best])) {
        best = i;
    };
    return best;
}

fn run(s: *rt.session.Session, m: *fwd.Model, x: *xb.Xpu, ids: []const u32, cap: usize) !usize {
    const e = try gpa.create(xb.Entry);
    e.* = .{ .cache = try xs.Cache.init(s, cap) };
    defer {
        e.cache.deinit();
        gpa.destroy(e);
    }
    var at: usize = 0;
    while (at < ids.len) {
        const end: usize = @min(ids.len, at + 128);
        try fwd.run(s, m, &x.scratch, &e.cache, ids[at..end], .{ .head = .last });
        at = end;
    }
    const row = try gpa.alloc(u16, cfg.vocab);
    defer gpa.free(row);
    try e.cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
    return argmax(row);
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    const dir: []const u8 = if (argv.len > 1) argv[1] else "/storage01/models/Qwen3.5-2B-MLX-4bit";
    const chars: usize = if (argv.len > 2) try std.fmt.parseInt(usize, argv[2], 10) else 65536;
    var s: rt.session.Session = undefined;
    s.open(.{}) catch |e| {
        std.debug.print("no card: {s}\n", .{@errorName(e)});
        return 1;
    };
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
    const src = sys.readFileAlloc(gpa, "/storage01/tensorfold-zig/CONTRIBUTING.md", 1 << 24) orelse return 1;
    var text: std.ArrayList(u8) = .empty;
    while (text.items.len < chars + src.len) try text.appendSlice(gpa, src);
    const body = text.items[0..chars];
    var toker = tk.loadTokenizer(init.io, gpa, dir) catch |e| {
        std.debug.print("tokenizer: {s}\n", .{@errorName(e)});
        return 1;
    };
    defer toker.deinit();
    const ids = try toker.encode(gpa, body);
    std.debug.print("text {d} chars -> {d} tokens\n", .{ body.len, ids.len });
    // Same prompt, capacities from tight (len+16) up to 32768: the served path uses min(32768, len+max_new+16).
    const caps = [_]usize{ ids.len + 16, 8192, 16384, 24576, 32768 };
    // The Xpu scratch is sized by `capacity`, so one Xpu per capacity (the served path does the same per request).
    for (caps) |cap| {
        if (cap < ids.len + 16) continue;
        const x = try xb.Xpu.init(gpa, &s, &m, .{ .capacity = @intCast(cap), .chunk = 128, .qualify = false });
        defer x.deinit();
        const a = try run(&s, &m, x, ids, cap);
        std.debug.print("cap={d}: argmax={d}  (eos=248044)\n", .{ cap, a });
    }
    return 0;
}
