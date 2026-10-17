//! TEMP probe: sweep prompt length on the bench text through the served prefill path, reporting the drawn token,
//! so the length at which the argmax flips to EOS (248044) is visible. Also compares chunked 128 against one-shot
//! rows at the same length: if they agree, chunking is not the cause.
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

fn top(row: []const u16, k: usize) void {
    var idx: [8]usize = undefined;
    var vals: [8]f32 = undefined;
    const n = @min(k, idx.len);
    for (0..n) |i| {
        idx[i] = i;
        vals[i] = val(row[i]);
    }
    for (row, 0..) |w, id| {
        const v = val(w);
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
    for (0..n) |i| std.debug.print(" {d}:{d}/{d:.2}", .{ i, idx[i], vals[i] });
}

/// Prefill `ids` in `chunk`-sized pieces (0 = one shot) and report the committed row's top tokens.
fn run(s: *rt.session.Session, m: *fwd.Model, x: *xb.Xpu, ids: []const u32, chunk: usize, label: []const u8) !void {
    const cap = @min(@as(usize, 32768), ids.len + 256);
    const e = try gpa.create(xb.Entry);
    e.* = .{ .cache = try xs.Cache.init(s, cap) };
    defer {
        e.cache.deinit();
        gpa.destroy(e);
    }
    var at: usize = 0;
    while (at < ids.len) {
        const end: usize = if (chunk == 0) ids.len else @min(ids.len, at + chunk);
        try fwd.run(s, m, &x.scratch, &e.cache, ids[at..end], .{ .head = .last });
        at = end;
    }
    const row = try gpa.alloc(u16, cfg.vocab);
    defer gpa.free(row);
    try e.cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
    std.debug.print("{s} n={d} chunk={d} cache.len={d}:", .{ label, ids.len, chunk, e.cache.len });
    top(row, 5);
    std.debug.print("\n", .{});
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    const dir: []const u8 = if (argv.len > 1) argv[1] else "/storage01/models/Qwen3.5-2B-MLX-4bit";
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
    var toker = tk.loadTokenizer(init.io, gpa, dir) catch |e| {
        std.debug.print("tokenizer: {s}\n", .{@errorName(e)});
        return 1;
    };
    defer toker.deinit();
    // One long token stream: prefixes of it are the sweep points (same text, growing length).
    var text: std.ArrayList(u8) = .empty;
    while (text.items.len < 65536 + src.len) try text.appendSlice(gpa, src);
    const ids = try toker.encode(gpa, text.items[0..65536]);
    std.debug.print("bench text -> {d} tokens\n", .{ids.len});
    const cap = @min(@as(usize, 32768), ids.len + 256);
    const x = try xb.Xpu.init(gpa, &s, &m, .{ .capacity = @intCast(cap), .chunk = 128, .qualify = false });
    defer x.deinit();
    // Sweep lengths that are multiples of the 128-token chunk, so chunked and one-shot runs are comparable.
    for ([_]usize{ 128, 512, 1024, 2048, 4096, 8192, 12288, 14559 }) |n| {
        if (n > ids.len) continue;
        try run(&s, &m, x, ids[0..n], 128, "chunked");
        if (n <= 128) try run(&s, &m, x, ids[0..n], 0, "oneshot");
    }
    return 0;
}
