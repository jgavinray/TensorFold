//! TEMP probe: the serve path (Xpu + lane engine) on the real bench prompt text, to isolate the EOS-first-token bug.
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

fn topk(row: []const u16, k: usize) void {
    var idx: [8]usize = undefined;
    var vals: [8]f64 = undefined;
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

/// Direct backend prefill (chunked 128 like the server) + first draw, no lane engine.
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
    try e.cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
    std.debug.print("{s}: direct prefill cache.len={d}, top logits:\n", .{ label, e.cache.len });
    topk(row, 5);
    gpa.free(row);
}

fn engineRun(x: *xb.Xpu, ids: []const u32, max_new: u32, drafts: bool, label: []const u8) !void {
    var lc = try lanes.Config.init(gpa, x.facts(), xb.window_rows, xb.window_rows - 1);
    defer lc.deinit(gpa);
    var fixed: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &lc, x.backend(), fixed.clock());
    defer engine.deinit();
    var prop = try lanes.SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer prop.deinit();
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = ids, .max_new = max_new, .eos = &.{}, .sampling = null, .drafts = drafts, .proposer = if (drafts) prop.proposer() else null });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    while (engine.activeCount() > 0) try engine.step();
    std.debug.print("{s}: engine emitted {d} tokens, first={d} last={d}, rounds {d}\n", .{ label, s.emitted().len, s.emitted()[0], s.emitted()[s.emitted().len - 1], s.rounds });
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    const dir: []const u8 = if (argv.len > 1) argv[1] else "/storage01/models/Qwen3.5-2B-MLX-4bit";
    const chars: usize = if (argv.len > 2) try std.fmt.parseInt(usize, argv[2], 10) else 49152;
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

    // The bench prompt: CONTRIBUTING.md repeated then truncated, exactly as bench_serve.py builds it.
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
    const x = try xb.Xpu.init(gpa, &s, &m, .{ .capacity = @intCast(cap), .chunk = 128, .qualify = true });
    defer x.deinit();
    try direct(&s, &m, x, ids, cap, "direct");
    try engineRun(x, ids, 8, false, "engine nodrafts");
    try engineRun(x, ids, 8, true, "engine drafts");
    return 0;
}
