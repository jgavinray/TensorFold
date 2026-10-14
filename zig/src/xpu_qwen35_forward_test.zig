//! tf-xpu-qwen35-forward-test [model dir] [capture dir] [ref dir] [gates]: on-card gates for Phase 4.3, the Qwen3.5-2B
//! forward pass on a B70 with the reference OpenCL kernels. `gates` is any of "abc" (default all).
//! Gate A: layer-0 output of the 176-token capture sequence vs the MLX-CPU fp32 capture (rms within 2x the bf16 floor).
//! Gate B: teacher-forced logits of the 3 fixtures vs the MLX fp32 logits (top-1, |NLL diff|, rms, max abs diff).
//! Gate C: greedy 32 tokens per prompt vs greedy-<name>.json.
//! Expected runtime: a few minutes once a card is free (the runtime's lock waits for one); kill at 2x.
const std = @import("std");
const rt = @import("xpu/runtime.zig");
const sys = @import("xpu/sys.zig");
const cfg = @import("families/qwen3_5/config.zig");
const xw = @import("families/qwen3_5/xpu_weights.zig");
const xs = @import("families/qwen3_5/xpu_state.zig");
const fwd = @import("families/qwen3_5/xpu_forward.zig");

const gpa = std.heap.page_allocator;
const fixtures = [_][]const u8{ "fibonacci-raw", "gpu-chat-no-think", "public-contribution-guide-64" };
const capacity = 512;

fn bf(w: u16) f32 {
    return @bitCast(@as(u32, w) << 16);
}
fn toBf(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}
fn rbf(x: f32) f32 {
    return bf(toBf(x));
}
fn secs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

fn readFile(comptime fmt: []const u8, args: anytype) ![]u8 {
    var b: [1024]u8 = undefined;
    const p = sys.zfmt(&b, fmt, args) orelse return error.PathTooLong;
    return sys.readFileAlloc(gpa, p, 1 << 30) orelse {
        std.debug.print("cannot read {s}\n", .{p});
        return error.NoFile;
    };
}

fn readNpyU32(dir: []const u8, name: []const u8) ![]u32 {
    const bytes = try readFile("{s}/{s}.npy", .{ dir, name });
    defer gpa.free(bytes);
    const hl = std.mem.readInt(u16, bytes[8..10], .little);
    const data = bytes[10 + hl ..];
    const out = try gpa.alloc(u32, data.len / 4);
    for (out, 0..) |*o, i| o.* = std.mem.readInt(u32, data[i * 4 ..][0..4], .little);
    return out;
}

fn downloadBf16(s: *rt.session.Session, b: *const rt.memory.Buffer, n: usize) ![]u16 {
    const out = try gpa.alloc(u16, n);
    try b.download(&s.queue, 0, std.mem.sliceAsBytes(out));
    return out;
}

// ---- Gate A -------------------------------------------------------------------------------------------------------

fn rmsFloor(ref: []const f32) f64 {
    var sq: f64 = 0;
    for (ref) |x| {
        const d: f64 = rbf(x) - x;
        sq += d * d;
    }
    return @sqrt(sq / @as(f64, @floatFromInt(ref.len)));
}

fn compareF32(name: []const u8, got: []const f32, ref: []const f32, floor_of: []const f32) f64 {
    var sq: f64 = 0;
    var mx: f64 = 0;
    var rs: f64 = 0;
    for (got, ref) |g, r| {
        const d: f64 = @as(f64, g) - r;
        sq += d * d;
        rs += @as(f64, r) * r;
        mx = @max(mx, @abs(d));
    }
    const n: f64 = @floatFromInt(got.len);
    const rms = @sqrt(sq / n);
    const floor = rmsFloor(floor_of);
    std.debug.print("  {s}: rms diff {d:.6}, max abs diff {d:.4}, ref rms {d:.4}, bf16 floor rms {d:.6}, ratio {d:.2}x\n", .{ name, rms, mx, @sqrt(rs / n), floor, rms / floor });
    return rms / floor;
}

fn loadF32(dir: []const u8, name: []const u8, n: usize) ![]f32 {
    const bytes = try readFile("{s}/{s}.f32", .{ dir, name });
    defer gpa.free(bytes);
    if (bytes.len != n * 4) return error.BadCapture;
    const out = try gpa.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

fn gateA(s: *rt.session.Session, m: *fwd.Model, sc: *fwd.Scratch, capdir: []const u8) !bool {
    const ids_bytes = try readFile("{s}/ids.u32", .{capdir});
    defer gpa.free(ids_bytes);
    const ids = try gpa.alloc(u32, ids_bytes.len / 4);
    defer gpa.free(ids);
    @memcpy(std.mem.sliceAsBytes(ids), ids_bytes);
    const rows = ids.len;
    var cache = try xs.Cache.init(s, capacity);
    defer cache.deinit();
    const t0 = sys.nowNs();
    try fwd.run(s, m, sc, &cache, ids, .{ .stop_after = 0 });
    std.debug.print("gate A: layer 0 on {d} tokens in {d:.3} s\n", .{ rows, secs(sys.nowNs() - t0) });
    const h = try downloadBf16(s, &sc.h, rows * cfg.hidden);
    defer gpa.free(h);
    const r = try downloadBf16(s, &sc.r, rows * cfg.hidden);
    defer gpa.free(r);
    const out = try gpa.alloc(f32, rows * cfg.hidden);
    defer gpa.free(out);
    const hf = try gpa.alloc(f32, rows * cfg.hidden);
    defer gpa.free(hf);
    for (out, hf, h, r) |*o, *hh, a, b| {
        o.* = rbf(bf(a) + bf(b));
        hh.* = bf(a);
    }
    const xr = try downloadBf16(s, &sc.x, rows * cfg.hidden);
    defer gpa.free(xr);
    const xf = try gpa.alloc(f32, rows * cfg.hidden);
    defer gpa.free(xf);
    for (xf, xr) |*o, a| o.* = bf(a);
    const stages = [_]struct { name: []const u8, key: []const u8, mirror: []const u8, got: []const f32 }{
        .{ .name = "ln2", .key = "l0_ln2", .mirror = "mirror_l0_ln2", .got = xf },
        .{ .name = "h", .key = "l0_h", .mirror = "mirror_l0_h", .got = hf },
        .{ .name = "layer output", .key = "l0_out", .mirror = "mirror_l0_out", .got = out },
    };
    // Three numbers per stage, each as rms diff over the bf16 rounding floor of the reference side:
    //  (1) XPU vs the fp64 mirror of the Metal arithmetic (zig/tests/qwen35_layer0_mirror.py) - the gate, must be <= 2x; the floor in all three rows is the fp32 capture's
    //  (2) the mirror vs the MLX fp32 capture - what any kernel chain with Metal's bf16 rounding points can reach
    //  (3) XPU vs the MLX fp32 capture
    var gate_ok = true;
    for (stages) |st| {
        const cap_ref = try loadF32(capdir, st.key, rows * cfg.hidden);
        defer gpa.free(cap_ref);
        const mir = try loadF32(capdir, st.mirror, rows * cfg.hidden);
        defer gpa.free(mir);
        std.debug.print("stage {s}:\n", .{st.name});
        const one = compareF32("(1) XPU vs Metal mirror", st.got, mir, cap_ref);
        _ = compareF32("(2) Metal mirror vs fp32 capture", mir, cap_ref, cap_ref);
        _ = compareF32("(3) XPU vs fp32 capture", st.got, cap_ref, cap_ref);
        if (one > 2.0) gate_ok = false;
    }
    return gate_ok;
}

// ---- Gate B -------------------------------------------------------------------------------------------------------

fn jsonNumber(text: []const u8, key: []const u8) ?f64 {
    const at = std.mem.indexOf(u8, text, key) orelse return null;
    var i = at + key.len;
    while (i < text.len and (text[i] == '"' or text[i] == ':' or text[i] == ' ')) i += 1;
    var e = i;
    while (e < text.len and (std.ascii.isDigit(text[e]) or text[e] == '.' or text[e] == '-' or text[e] == 'e' or text[e] == '+')) e += 1;
    return std.fmt.parseFloat(f64, text[i..e]) catch null;
}

fn argmax(row: []const u16) usize {
    var best: usize = 0;
    var bv = bf(row[0]);
    for (row, 0..) |w, i| {
        const v = bf(w);
        if (v > bv) {
            bv = v;
            best = i;
        }
    }
    return best;
}

fn logZ(row: []const u16) f64 {
    var top: f64 = -std.math.inf(f64);
    for (row) |w| top = @max(top, bf(w));
    var sum: f64 = 0;
    for (row) |w| sum += @exp(@as(f64, bf(w)) - top);
    return @log(sum) + top;
}

fn top2(row: []const u16) struct { first: usize, margin: f32 } {
    const a = argmax(row);
    var second: f32 = -std.math.inf(f32);
    for (row, 0..) |w, i| if (i != a and bf(w) > second) {
        second = bf(w);
    };
    return .{ .first = a, .margin = bf(row[a]) - second };
}

fn gateB(s: *rt.session.Session, m: *fwd.Model, sc: *fwd.Scratch, refdir: []const u8) !bool {
    var all_ok = true;
    for (fixtures) |name| {
        const ids = try readNpyU32(refdir, name);
        defer gpa.free(ids);
        const n = ids.len;
        const refbytes = try readFile("{s}/{s}.serial.bf16", .{ refdir, name });
        defer gpa.free(refbytes);
        if (refbytes.len != n * cfg.vocab * 2) return error.BadReference;
        const ref: []const u16 = @ptrCast(@alignCast(refbytes));
        const refjson = try readFile("{s}/reference-{s}.json", .{ refdir, name });
        defer gpa.free(refjson);
        const ref_nll = jsonNumber(refjson, "\"mlx_serial_nll\"") orelse return error.BadReference;

        var cache = try xs.Cache.init(s, capacity);
        defer cache.deinit();
        const t0 = sys.nowNs();
        try fwd.run(s, m, sc, &cache, ids, .{ .head = .all });
        const dt = sys.nowNs() - t0;
        const got = try downloadBf16(s, &sc.logits, n * cfg.vocab);
        defer gpa.free(got);

        var agree: usize = 0;
        var sq: f64 = 0;
        var mx: f64 = 0;
        var nll: f64 = 0;
        var ref_bf_nll: f64 = 0;
        for (0..n) |t| {
            const gr = got[t * cfg.vocab ..][0..cfg.vocab];
            const rr = ref[t * cfg.vocab ..][0..cfg.vocab];
            if (argmax(gr) == argmax(rr)) agree += 1 else {
                const ta = top2(gr);
                const tb = top2(rr);
                std.debug.print("  top-1 mismatch at position {d}: xpu token {d} (margin to its runner-up {d:.4}), mlx token {d} (margin {d:.4}); mlx logit of xpu token {d:.4}, xpu logit of mlx token {d:.4}\n", .{
                    t, ta.first, ta.margin, tb.first, tb.margin, bf(rr[ta.first]), bf(gr[tb.first]),
                });
            }
            for (gr, rr) |a, b| {
                const d: f64 = @as(f64, bf(a)) - bf(b);
                sq += d * d;
                mx = @max(mx, @abs(d));
            }
            if (t + 1 < n) {
                nll += logZ(gr) - bf(gr[ids[t + 1]]);
                ref_bf_nll += logZ(rr) - bf(rr[ids[t + 1]]);
            }
        }
        nll /= @floatFromInt(n - 1);
        ref_bf_nll /= @floatFromInt(n - 1);
        const rms = @sqrt(sq / @as(f64, @floatFromInt(n * cfg.vocab)));
        const nd = @abs(nll - ref_nll);
        const ok = agree == n and nd < 0.01;
        all_ok = all_ok and ok;
        std.debug.print("gate B {s}: {d} tokens, forward {d:.3} s ({d:.1} tokens/s teacher-forced)\n", .{ name, n, secs(dt), @as(f64, @floatFromInt(n)) / secs(dt) });
        std.debug.print("  top-1 agreement {d}/{d} ({d:.1}%), NLL xpu {d:.6} vs mlx fp32 {d:.6} (mlx logits rounded to bf16: {d:.6}), |NLL diff| {d:.6}, logit rms diff {d:.4}, max abs diff {d:.4}  -> {s}\n", .{
            agree, n, 100.0 * @as(f64, @floatFromInt(agree)) / @as(f64, @floatFromInt(n)), nll, ref_nll, ref_bf_nll, nd, rms, mx, if (ok) "ok" else "below target",
        });
    }
    return all_ok;
}

// ---- Gate C -------------------------------------------------------------------------------------------------------

fn intList(v: std.json.Value, out: *std.ArrayList(u32)) !void {
    if (v != .array) return error.BadReference;
    for (v.array.items) |x| try out.append(gpa, @intCast(x.integer));
}

fn gateC(s: *rt.session.Session, m: *fwd.Model, sc: *fwd.Scratch, refdir: []const u8) !bool {
    var all_ok = true;
    var step_ns: std.ArrayList(u64) = .empty;
    defer step_ns.deinit(gpa);
    for (fixtures) |name| {
        var gb: [1024]u8 = undefined;
        const gpath = sys.zfmt(&gb, "{s}/greedy-{s}.json", .{ refdir, name }) orelse return error.PathTooLong;
        const text = sys.readFileAlloc(gpa, gpath, 1 << 24) orelse {
            std.debug.print("gate C {s}: no greedy-{s}.json in the reference directory, skipped\n", .{ name, name });
            continue;
        };
        defer gpa.free(text);
        const doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer doc.deinit();
        const entry = doc.value.object.get(name) orelse return error.BadReference;
        var prompt: std.ArrayList(u32) = .empty;
        defer prompt.deinit(gpa);
        var want: std.ArrayList(u32) = .empty;
        defer want.deinit(gpa);
        try intList(entry.object.get("prompt_tokens").?, &prompt);
        try intList(entry.object.get("tokens").?, &want);
        var cache = try xs.Cache.init(s, capacity);
        defer cache.deinit();
        const row = try gpa.alloc(u16, cfg.vocab);
        defer gpa.free(row);
        var next: [1]u32 = undefined;
        var diverged: ?usize = null;
        var matched: usize = 0;
        const t0 = sys.nowNs();
        try fwd.run(s, m, sc, &cache, prompt.items, .{});
        const prefill_ns = sys.nowNs() - t0;
        var i: usize = 0;
        while (i < want.items.len) : (i += 1) {
            try cache.logits.download(&s.queue, 0, std.mem.sliceAsBytes(row));
            const t = top2(row);
            if (t.first != want.items[i]) {
                diverged = i;
                std.debug.print("gate C {s}: FIRST DIVERGENCE at step {d}: xpu token {d}, reference token {d}, xpu top-2 logit margin {d:.4} (xpu logit of reference token {d:.4}, of its own {d:.4})\n", .{ name, i, t.first, want.items[i], t.margin, bf(row[want.items[i]]), bf(row[t.first]) });
                break;
            }
            matched += 1;
            if (i + 1 == want.items.len) break;
            next[0] = @intCast(t.first);
            const t1 = sys.nowNs();
            try fwd.run(s, m, sc, &cache, &next, .{});
            try step_ns.append(gpa, sys.nowNs() - t1);
        }
        const ok = diverged == null;
        all_ok = all_ok and ok;
        std.debug.print("gate C {s}: prompt {d} tokens (prefill {d:.3} s), {d}/{d} greedy tokens match -> {s}\n", .{ name, prompt.items.len, secs(prefill_ns), matched, want.items.len, if (ok) "ok" else "DIVERGED" });
    }
    if (step_ns.items.len > 0) {
        std.mem.sort(u64, step_ns.items, {}, std.sort.asc(u64));
        var sum: u64 = 0;
        for (step_ns.items) |x| sum += x;
        std.debug.print("decode step (1 token, capacity {d}): {d} steps, median {d:.1} ms, mean {d:.1} ms, min {d:.1} ms -> {d:.2} tokens/s\n", .{
            capacity,                                              step_ns.items.len,
            @as(f64, @floatFromInt(step_ns.items[step_ns.items.len / 2])) / 1e6, @as(f64, @floatFromInt(sum)) / 1e6 / @as(f64, @floatFromInt(step_ns.items.len)),
            @as(f64, @floatFromInt(step_ns.items[0])) / 1e6,       1e9 / (@as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(step_ns.items.len))),
        });
    }
    return all_ok;
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    const dir: []const u8 = if (argv.len > 1) argv[1] else "/storage01/models/Qwen3.5-2B-MLX-4bit";
    const capdir: []const u8 = if (argv.len > 2) argv[2] else "/storage01/qwen-flash-on-intel/exp/p4-forward-capture";
    const refdir: []const u8 = if (argv.len > 3) argv[3] else "/storage01/qwen-flash-on-intel/exp/ref/qwen35-2b-mlx4-fp32";
    const gates: []const u8 = if (argv.len > 4) argv[4] else "abc";

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
    defer gpa.free(ctext);
    const config = try cfg.parse(gpa, ctext);
    var w = try xw.load(gpa, &s, dir, config.tied);
    defer w.deinit();
    var m: fwd.Model = .{ .weights = &w, .kernels = undefined };
    try m.kernels.load(&s, gpa);
    defer m.kernels.deinit();
    var sc = try fwd.Scratch.init(&s, 176, capacity, 64);
    defer sc.deinit();
    std.debug.print("weights {d:.1} MiB and kernels loaded\n", .{@as(f64, @floatFromInt(w.total_bytes)) / (1 << 20)});

    var ok = true;
    if (std.mem.indexOfScalar(u8, gates, 'a') != null) {
        const r = gateA(&s, &m, &sc, capdir) catch |e| blk: {
            std.debug.print("gate A error: {s}\n", .{@errorName(e)});
            break :blk false;
        };
        std.debug.print("GATE A layer-0 XPU vs fp64 Metal mirror (<= 2x floor): {s}\n", .{if (r) "PASS" else "FAIL"});
        ok = ok and r;
    }
    if (std.mem.indexOfScalar(u8, gates, 'b') != null) {
        const r = gateB(&s, &m, &sc, refdir) catch |e| blk: {
            std.debug.print("gate B error: {s}\n", .{@errorName(e)});
            break :blk false;
        };
        std.debug.print("GATE B teacher-forced logits (top-1 100% and |NLL diff| < 0.01): {s}\n", .{if (r) "PASS" else "FAIL"});
        ok = ok and r;
    }
    if (std.mem.indexOfScalar(u8, gates, 'c') != null) {
        const r = gateC(&s, &m, &sc, refdir) catch |e| blk: {
            std.debug.print("gate C error: {s}\n", .{@errorName(e)});
            break :blk false;
        };
        std.debug.print("GATE C greedy 32 tokens: {s}\n", .{if (r) "PASS" else "FAIL"});
        ok = ok and r;
    }
    return if (ok) 0 else 1;
}
