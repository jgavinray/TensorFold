const std = @import("std");
const lanes = @import("lanes");
const rt = @import("xpu/runtime.zig");
const sys = @import("xpu/sys.zig");
const cfg = @import("families/qwen3_5/config.zig");
const xw = @import("families/qwen3_5/xpu_weights.zig");
const fwd = @import("families/qwen3_5/xpu_forward.zig");
const xb = @import("families/qwen3_5/xpu_backend.zig");

const gpa = std.heap.page_allocator;

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

    // reference
    var gb: [1024]u8 = undefined;
    const gpath = sys.zfmt(&gb, "{s}/greedy-fibonacci-raw.json", .{refdir}) orelse return 1;
    const text = sys.readFileAlloc(gpa, gpath, 1 << 24) orelse return 1;
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    const entry = doc.value.object.get("fibonacci-raw").?;
    var prompt: std.ArrayList(u32) = .empty;
    var want: std.ArrayList(u32) = .empty;
    for (entry.object.get("prompt_tokens").?.array.items) |v| try prompt.append(gpa, @intCast(v.integer));
    for (entry.object.get("tokens").?.array.items) |v| try want.append(gpa, @intCast(v.integer));

    const b = x.backend();
    var st = try lanes.Stream.init(gpa, .{ .id = "q", .prompt = prompt.items, .max_new = 8, .eos = &.{}, .sampling = null, .drafts = false });
    defer st.deinit(gpa);
    try b.prefill(&st);
    defer b.release(&st);
    const first = try b.read(try b.first(&st, prompt.items.len));
    std.debug.print("first: got {d} want {d} {s}\n", .{ first, want.items[0], if (first == want.items[0]) "ok" else "BAD" });
    var pending: u32 = first;
    var got: std.ArrayList(u32) = .empty;
    try got.append(gpa, first);
    for (1..8) |i| {
        const t: u32 = @intCast(try b.queue(&st, .{ .value = pending }, prompt.items.len + i));
        pending = t;
        try got.append(gpa, t);
        std.debug.print("step {d}: got {d} want {d} {s}\n", .{ i, t, want.items[i], if (t == want.items[i]) "ok" else "BAD" });
    }
    return 0;
}
