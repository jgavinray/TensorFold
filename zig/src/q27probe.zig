//! TEMP (I05 investigation): the 27B engine's queue path (prefill/first/queue) against the greedy reference, to
//! isolate a serve divergence from the forward itself. Delete with the I05 investigation.
const std = @import("std");
const lanes = @import("lanes");
const rt = @import("xpu/runtime.zig");
const sys = @import("xpu/sys.zig");
const cfg = @import("families/qwen27/config.zig");
const xw = @import("families/qwen27/xpu_weights.zig");
const fwd = @import("families/qwen27/xpu_forward.zig");
const xb = @import("families/qwen27/xpu_backend.zig");

const gpa = std.heap.page_allocator;

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    const dir: []const u8 = if (argv.len > 1) argv[1] else "/storage01/models/Qwen3.8-27B-MLX-4bit";
    const refdir: []const u8 = if (argv.len > 2) argv[2] else "/storage01/qwen-flash-on-intel/exp/ref/qwen38-27b-mlx4-fp32";
    const steps: usize = if (argv.len > 3) try std.fmt.parseInt(usize, argv[3], 10) else 8;
    var s: rt.session.Session = undefined;
    s.open(.{}) catch |e| {
        std.debug.print("no card: {s}\n", .{@errorName(e)});
        return 1;
    };
    defer s.close();
    var cb: [1024]u8 = undefined;
    const cpath = sys.zfmt(&cb, "{s}/config.json", .{dir}) orelse return 1;
    const ctext = sys.readFileAlloc(gpa, cpath, 1 << 22) orelse return 1;
    const config = try cfg.parse(gpa, ctext, null);
    var w = try xw.load(gpa, &s, dir, config);
    defer w.deinit();
    var m: fwd.Model = .{ .weights = &w, .kernels = undefined };
    try m.kernels.load(&s, gpa);
    defer m.kernels.deinit();
    const x = try xb.Xpu.init(gpa, &s, &m, config, .{ .capacity = 512, .qualify = false });
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

    const b = x.backend();
    var st = try lanes.Stream.init(gpa, .{ .id = "q", .prompt = prompt.items, .max_new = @intCast(steps), .eos = &.{}, .sampling = null, .drafts = false });
    defer st.deinit(gpa);
    const t0 = sys.nowNs();
    try b.prefill(&st);
    defer b.release(&st);
    const first = try b.read(try b.first(&st, prompt.items.len));
    std.debug.print("prefill {d} rows + first token: {d:.2} s\n", .{ prompt.items.len, @as(f64, @floatFromInt(sys.nowNs() - t0)) / 1e9 });
    std.debug.print("first: got {d} want {d} {s}\n", .{ first, want.items[0], if (first == want.items[0]) "ok" else "BAD" });
    var pending: u32 = first;
    for (1..steps) |i| {
        const t1 = sys.nowNs();
        const t: u32 = @intCast(try b.queue(&st, .{ .value = pending }, prompt.items.len + i));
        std.debug.print("step {d}: got {d} want {d} {s} ({d:.1} ms)\n", .{ i, t, want.items[i], if (t == want.items[i]) "ok" else "BAD", @as(f64, @floatFromInt(sys.nowNs() - t1)) / 1e6 });
        pending = t;
    }
    return 0;
}
