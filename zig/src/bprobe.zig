//! TEMP probe: the bench's 16K cell. Reproduces bench_serve.py's prompt, prefills it the way the served
//! backend does (chunk 128), and reports the drawn first token plus the logits row at the last position, then
//! compares against the same text truncated to the short cells' sizes. Also runs the engine path (prefill via
//! Xpu.prefill + first) so the serve signature (tokens=1, finish=stop, rounds=0) is reproducible in isolation.
const std = @import("std");
const lanes = @import("lanes");
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

fn stats(row: []const u16) void {
    var best: usize = 0;
    var second: usize = 0;
    var sum: f64 = 0;
    var finite: usize = 0;
    for (row, 0..) |w, i| {
        const v = val(w);
        if (std.math.isFinite(v)) finite += 1 else continue;
        sum += v;
        if (i == 0 or v > val(row[best])) {
            second = best;
            best = i;
        } else if (i != best and (second == 0 or v > val(row[second]))) second = i;
    }
    std.debug.print("  finite={d}/{d} mean={d:.4} argmax={d} ({d:.4}) second={d} ({d:.4})\n", .{ finite, row.len, sum / @as(f64, @floatFromInt(row.len)), best, val(row[best]), second, val(row[second]) });
}

/// Direct backend prefill: chunk 128, exactly as `Xpu.prefill` does it, then read the committed logits row.
fn direct(s: *rt.session.Session, m: *fwd.Model, x: *xb.Xpu, ids: []const u32, cap: usize, label: []const u8) !void {
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
    const drawn = try xb.draw(gpa, row, null, ids.len, &.{});
    std.debug.print("{s}: {d} tokens, cache.len={d}, drawn={d}\n", .{ label, ids.len, e.cache.len, drawn });
    stats(row);
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

    // bench_serve.py's prompt: CONTRIBUTING.md repeated, then truncated to p_tok*4 chars.
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
    std.debug.print("text {d} chars -> {d} tokens (first5 {any} last5 {any})\n", .{ body.len, ids.len, ids[0..5], ids[ids.len - 5 ..] });

    const cap = @min(@as(usize, 32768), ids.len + 128 + 16);
    const x = try xb.Xpu.init(gpa, &s, &m, .{ .capacity = @intCast(cap), .chunk = 128, .qualify = false });
    defer x.deinit();
    try direct(&s, &m, x, ids, cap, "full");
    // The same text at the two short-cell lengths, for a same-text reference.
    try direct(&s, &m, x, ids[0..@min(ids.len, 474)], cap, "prefix-474");
    try direct(&s, &m, x, ids[0..@min(ids.len, 3679)], cap, "prefix-3679");
    return 0;
}
