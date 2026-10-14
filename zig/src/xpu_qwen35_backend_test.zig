//! tf-xpu-qwen35-backend-test [model dir] [ref dir]: on-card test of `Xpu` (the lane engine's backend on a B70).
//! 1. xpu_qualify: wide windows vs one-row execution, byte-equal logits/caches, incl. a partial keep.
//! 2. per fixture: one stream through the lane engine (greedy, 32 tokens), no drafts; then with a suffix-copy proposer;
//!    tokens must equal greedy-<name>.json.
//! 3. per fixture: the backend driven directly with reference-derived draft windows (some corrupted) to force verify + keep.
//! Expected runtime: a few minutes once a card is free (the runtime's lock waits for one); kill at 2x.
const std = @import("std");
const lanes = @import("lanes");
const rt = @import("xpu/runtime.zig");
const sys = @import("xpu/sys.zig");
const cfg = @import("families/qwen3_5/config.zig");
const xw = @import("families/qwen3_5/xpu_weights.zig");
const fwd = @import("families/qwen3_5/xpu_forward.zig");
const xb = @import("families/qwen3_5/xpu_backend.zig");
const xq = @import("families/qwen3_5/xpu_qualify.zig");

const gpa = std.heap.page_allocator;
const fixtures = [_][]const u8{ "fibonacci-raw", "gpu-chat-no-think", "public-contribution-guide-64" };
const capacity = 512;

fn intList(v: std.json.Value, out: *std.ArrayList(u32)) !void {
    if (v != .array) return error.BadReference;
    for (v.array.items) |x| try out.append(gpa, @intCast(x.integer));
}

fn secs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

const Fixture = struct {
    prompt: std.ArrayList(u32) = .empty,
    want: std.ArrayList(u32) = .empty,

    fn load(refdir: []const u8, name: []const u8) !Fixture {
        var gb: [1024]u8 = undefined;
        const gpath = sys.zfmt(&gb, "{s}/greedy-{s}.json", .{ refdir, name }) orelse return error.PathTooLong;
        const text = sys.readFileAlloc(gpa, gpath, 1 << 24) orelse return error.NoFile;
        const doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        const entry = doc.value.object.get(name) orelse return error.BadReference;
        var f: Fixture = .{};
        try intList(entry.object.get("prompt_tokens").?, &f.prompt);
        try intList(entry.object.get("tokens").?, &f.want);
        return f;
    }
};

fn report(label: []const u8, name: []const u8, got: []const u32, want: []const u32, ns: u64) bool {
    var matched: usize = 0;
    while (matched < got.len and matched < want.len and got[matched] == want[matched]) matched += 1;
    const ok = matched == want.len and got.len == want.len;
    std.debug.print("{s} {s}: {d}/{d} tokens equal greedy reference ({d:.2} s) -> {s}\n", .{ label, name, matched, want.len, secs(ns), if (ok) "PASS" else "FAIL" });
    if (!ok and matched < @min(got.len, want.len)) std.debug.print("  first divergence at {d}: got {d}, want {d}\n", .{ matched, got[matched], want[matched] });
    return ok;
}

/// One stream through lanes.Engine.
fn engineRun(x: *xb.Xpu, f: Fixture, drafts: bool) ![]u32 {
    var lc = try lanes.Config.init(gpa, x.facts(), xb.window_rows, xb.window_rows - 1);
    defer lc.deinit(gpa);
    var fixed: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &lc, x.backend(), fixed.clock());
    defer engine.deinit();
    var prop = try lanes.SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer prop.deinit();
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = f.prompt.items, .max_new = @intCast(f.want.items.len), .eos = &.{}, .sampling = null, .drafts = drafts, .proposer = if (drafts) prop.proposer() else null });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    while (engine.activeCount() > 0) try engine.step();
    std.debug.print("  engine (drafts {s}): {d} rounds, {d} drafted, {d} accepted, exact width {d}\n", .{ if (drafts) "on" else "off", s.rounds, s.drafted, s.accepted, lc.family_width });
    return try gpa.dupe(u32, s.emitted());
}

/// The backend driven directly: windows of the reference's next tokens, one corrupted per round, then kept by prefix.
fn windowRun(x: *xb.Xpu, f: Fixture, depth: usize) ![]u32 {
    const b = x.backend();
    var s = try lanes.Stream.init(gpa, .{ .id = "w", .prompt = f.prompt.items, .max_new = @intCast(f.want.items.len), .eos = &.{}, .sampling = null, .drafts = false });
    defer s.deinit(gpa);
    try b.prefill(&s);
    defer b.release(&s);
    var out: std.ArrayList(u32) = .empty;
    var pending = try b.read(try b.first(&s, f.prompt.items.len));
    try out.append(gpa, pending);
    var round: usize = 0;
    var kept_partial: usize = 0;
    while (out.items.len < f.want.items.len) : (round += 1) {
        const room = f.want.items.len - out.items.len; // cache holds prompt + out.len - 1 rows
        const nd = @min(depth, room - 1);
        var drafts: [xb.window_rows]u32 = undefined;
        for (0..nd) |i| drafts[i] = f.want.items[out.items.len + i];
        if (nd > 0 and round % 2 == 1) drafts[round % nd] +%= 1; // wrong draft: partial acceptance
        const base = f.prompt.items.len + out.items.len - 1;
        var positions: [xb.window_rows]u64 = undefined;
        for (positions[0 .. nd + 1], 0..) |*p, r| p.* = base + r + 1;
        var sampled: [xb.window_rows]u32 = undefined;
        var dout: [xb.window_rows]u32 = undefined;
        const w: lanes.backend.Window = .{ .stream = &s, .pending = pending, .held = 0, .tokens = drafts[0..nd], .parents = null, .positions = positions[0 .. nd + 1] };
        var res: lanes.backend.Verified = .{ .sampled = sampled[0 .. nd + 1], .drafts = dout[0..nd] };
        try b.verify(&.{w}, (&res)[0..1]);
        var acc: usize = 0;
        while (acc < nd and sampled[acc] == drafts[acc]) acc += 1;
        var path: [xb.window_rows]u32 = undefined;
        for (path[0 .. acc + 1], 0..) |*p, i| p.* = @intCast(i);
        try b.keep(&.{w}, &.{path[0 .. acc + 1]});
        if (acc < nd) kept_partial += 1;
        for (sampled[0 .. acc + 1]) |t| try out.append(gpa, t);
        pending = sampled[acc];
        _ = &pending;
    }
    out.shrinkRetainingCapacity(f.want.items.len);
    std.debug.print("  window run: depth {d}, {d} rounds, {d} partial keeps\n", .{ depth, round, kept_partial });
    return out.items;
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    const dir: []const u8 = if (argv.len > 1) argv[1] else "/storage01/models/Qwen3.5-2B-MLX-4bit";
    const refdir: []const u8 = if (argv.len > 2) argv[2] else "/storage01/qwen-flash-on-intel/exp/ref/qwen35-2b-mlx4-fp32";

    var s: rt.session.Session = undefined;
    s.open(.{}) catch |e| {
        std.debug.print("cannot open a card: {s} ({s})\n", .{ @errorName(e), s.diag.message() });
        return 1;
    };
    defer s.close();
    std.debug.print("card {s} ({s})\n", .{ s.bdf(), s.ctx.dev.name() });

    var cb: [1024]u8 = undefined;
    const cpath = sys.zfmt(&cb, "{s}/config.json", .{dir}) orelse return 1;
    const ctext = sys.readFileAlloc(gpa, cpath, 1 << 22) orelse return 1;
    const config = try cfg.parse(gpa, ctext);
    var w = try xw.load(gpa, &s, dir, config.tied);
    defer w.deinit();
    var m: fwd.Model = .{ .weights = &w, .kernels = undefined };
    try m.kernels.load(&s, gpa);
    defer m.kernels.deinit();

    var ok = true;
    const x = try xb.Xpu.init(gpa, &s, &m, .{ .capacity = capacity, .qualify = false });
    defer x.deinit();
    const q0 = sys.nowNs();
    if (xq.check(gpa, &s, &m, &x.scratch)) {
        x.width = xb.window_rows;
        std.debug.print("xpu_qualify: wide windows (16 rows, partial keep) byte-equal to one-row execution ({d:.1} s) -> PASS\n", .{secs(sys.nowNs() - q0)});
    } else |e| {
        std.debug.print("xpu_qualify: {s} -> FAIL (width stays 1)\n", .{@errorName(e)});
        ok = false;
    }

    for (fixtures) |name| {
        var f = Fixture.load(refdir, name) catch |e| {
            std.debug.print("{s}: no usable greedy-{s}.json ({s}), skipped\n", .{ name, name, @errorName(e) });
            continue;
        };
        defer f.prompt.deinit(gpa);
        defer f.want.deinit(gpa);
        for ([_]bool{ false, true }) |drafts| {
            const t0 = sys.nowNs();
            const got = engineRun(x, f, drafts) catch |e| {
                std.debug.print("engine {s}: error {s}\n", .{ name, @errorName(e) });
                ok = false;
                continue;
            };
            defer gpa.free(got);
            ok = report(if (drafts) "engine+drafts" else "engine", name, got, f.want.items, sys.nowNs() - t0) and ok;
        }
        for ([_]usize{ 1, 7, 15 }) |depth| {
            const t0 = sys.nowNs();
            const got = windowRun(x, f, depth) catch |e| {
                std.debug.print("window {s}: error {s}\n", .{ name, @errorName(e) });
                ok = false;
                continue;
            };
            ok = report("windows", name, got, f.want.items, sys.nowNs() - t0) and ok;
        }
    }
    std.debug.print("OVERALL: {s}\n", .{if (ok) "PASS" else "FAIL"});
    return if (ok) 0 else 1;
}
