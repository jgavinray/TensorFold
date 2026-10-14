//! tf-xpu-kernel-exact <kernel.cl|kernel.bin> <case dir>: generic on-card gate runner.
//!
//! The case dir holds case.json and the input .bin files; the runner writes each output as <name>.bin next to them.
//! Comparison logic stays in each case's own checker (the runner knows nothing about the kernel's meaning).
//!
//! case.json:
//!   { "kernel": "name", "global": [x, y, z], "local": [x, y, z],
//!     "args": [ {"buffer": "H.bin"},                 // input: device buffer filled from the file
//!               {"out": "HO.bin", "bytes": 135168},  // output: device buffer, optional "fill": 0..255 byte, saved after the run
//!               {"f32": 1e-6}, {"i32": 33}, {"u32": 7}, {"u64": 9}, {"i64": -1}, {"u8": 1}, {"u16": 2} ] }
const std = @import("std");
const rt = @import("runtime.zig");
const Session = @import("session.zig").Session;
const sys = @import("sys.zig");

const Arg = struct {
    buffer: ?[]const u8 = null,
    out: ?[]const u8 = null,
    bytes: ?usize = null,
    fill: ?u8 = null,
    f32: ?f32 = null,
    i32: ?i32 = null,
    u32: ?u32 = null,
    i64: ?i64 = null,
    u64: ?u64 = null,
    u16: ?u16 = null,
    u8: ?u8 = null,
};
const Case = struct {
    kernel: []const u8,
    global: []const u32,
    local: []const u32,
    args: []const Arg,
};

fn fail(comptime fmt: []const u8, args: anytype) u8 {
    std.debug.print("tf-xpu-kernel-exact: " ++ fmt ++ "\n", args);
    return 1;
}

fn z(buf: []u8, comptime fmt: []const u8, args: anytype) ?[:0]u8 {
    return sys.zfmt(buf, fmt, args);
}

/// Single-quote for /bin/sh.
fn quoted(buf: []u8, s: []const u8) ?[]const u8 {
    var n: usize = 0;
    if (n >= buf.len) return null;
    buf[n] = '\'';
    n += 1;
    for (s) |c| {
        const need: usize = if (c == '\'') 4 else 1;
        if (n + need + 1 >= buf.len) return null;
        if (c == '\'') {
            @memcpy(buf[n..][0..4], "'\\''");
        } else buf[n] = c;
        n += need;
    }
    buf[n] = '\'';
    return buf[0 .. n + 1];
}

fn dupez(a: std.mem.Allocator, text: []const u8) ![:0]u8 {
    const out = try a.allocSentinel(u8, text.len, 0);
    @memcpy(out, text);
    return out;
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    if (argv.len != 3) return fail("usage: tf-xpu-kernel-exact <kernel.cl|kernel.bin> <case dir>", .{});
    const kernel_path: []const u8 = argv[1];
    const dir: []const u8 = argv[2];

    var pb: [1024]u8 = undefined;
    const case_json = sys.readFileAlloc(arena, z(&pb, "{s}/case.json", .{dir}) orelse return fail("path too long", .{}), 1 << 20) orelse
        return fail("cannot read {s}/case.json", .{dir});
    const parsed = std.json.parseFromSliceLeaky(Case, arena, case_json, .{ .ignore_unknown_fields = true }) catch |e|
        return fail("bad case.json: {s}", .{@errorName(e)});
    if (parsed.global.len != 3 or parsed.local.len != 3) return fail("global and local need 3 entries", .{});

    // .cl sources are compiled with ocloc for bmg into <case dir>/ocloc-out; anything else is taken as a native binary.
    var bin_path: []const u8 = kernel_path;
    if (std.mem.endsWith(u8, kernel_path, ".cl")) {
        var q1: [1024]u8 = undefined;
        var q2: [1100]u8 = undefined;
        var cmd: [3300]u8 = undefined;
        const out_dir = std.fmt.allocPrint(arena, "{s}/ocloc-out", .{dir}) catch return fail("oom", .{});
        const c = z(&cmd, "ocloc compile -file {s} -device bmg -options '-cl-std=CL3.0' -out_dir {s} > {s}/ocloc.log 2>&1", .{
            quoted(&q1, kernel_path) orelse return fail("path too long", .{}),
            quoted(&q2, out_dir) orelse return fail("path too long", .{}),
            dir,
        }) orelse return fail("path too long", .{});
        if (sys.shell(c) != 0) return fail("ocloc failed; see {s}/ocloc.log", .{dir});
        const stem = std.fs.path.stem(kernel_path);
        bin_path = std.fmt.allocPrint(arena, "{s}/{s}_bmg.bin", .{ out_dir, stem }) catch return fail("oom", .{});
    }
    const bin_z = try dupez(arena, bin_path);
    const binary = sys.readFileAlloc(arena, bin_z, 256 << 20) orelse return fail("cannot read kernel binary {s}", .{bin_path});

    var s: Session = undefined;
    s.open(.{ .wait_ms = 3_600_000 }) catch |e| return fail("cannot open a card: {s} ({s})", .{ @errorName(e), s.diag.message() });
    defer s.close();
    std.debug.print("card {s} ({s})\n", .{ s.bdf(), s.ctx.dev.name() });

    var log: rt.module.BuildLog = .{};
    var m = rt.module.Module.load(&s.ctx, binary, &log) catch |e| return fail("module load: {s}: {s}", .{ @errorName(e), log.text() });
    defer m.unload();
    const kname = try dupez(arena, parsed.kernel);
    var k = m.kernel(kname) catch |e| return fail("kernel {s}: {s}", .{ parsed.kernel, @errorName(e) });
    defer k.deinit();

    // Buffers: every buffer/out argument becomes a device allocation; outputs are read back through host memory.
    var bufs = try arena.alloc(?rt.memory.Buffer, parsed.args.len);
    for (bufs) |*b| b.* = null;
    defer for (bufs) |*b| if (b.*) |*x| x.free();
    var stage = try arena.alloc(?rt.memory.Buffer, parsed.args.len);
    for (stage) |*b| b.* = null;
    defer for (stage) |*b| if (b.*) |*x| x.free();

    for (parsed.args, 0..) |a, i| {
        const idx: u32 = @intCast(i);
        if (a.buffer) |name| {
            const data = sys.readFileAlloc(arena, z(&pb, "{s}/{s}", .{ dir, name }) orelse return fail("path too long", .{}), 2 << 30) orelse
                return fail("cannot read input {s}", .{name});
            var b = rt.memory.Buffer.alloc(&s.ctx, .device, data.len) catch |e| return fail("alloc {s}: {s}", .{ name, @errorName(e) });
            bufs[i] = b;
            b.upload(&s.queue, 0, data) catch |e| return fail("upload {s}: {s}", .{ name, @errorName(e) });
            try k.setArg(idx, b.ptr);
        } else if (a.out) |name| {
            const n = a.bytes orelse return fail("output {s} needs \"bytes\"", .{name});
            const b = rt.memory.Buffer.alloc(&s.ctx, .device, n) catch |e| return fail("alloc {s}: {s}", .{ name, @errorName(e) });
            bufs[i] = b;
            const pat = [_]u8{a.fill orelse 0};
            try s.queue.fill(b.ptr, &pat, n, null);
            try k.setArg(idx, b.ptr);
        } else if (a.f32) |v| try k.setArg(idx, v) else if (a.i32) |v| try k.setArg(idx, v) else if (a.u32) |v| try k.setArg(idx, v) else if (a.i64) |v| try k.setArg(idx, v) else if (a.u64) |v| try k.setArg(idx, v) else if (a.u16) |v| try k.setArg(idx, v) else if (a.u8) |v| try k.setArg(idx, v) else return fail("argument {d} has no recognised kind", .{i});
    }
    try s.queue.synchronize();

    var pool = try rt.event.Pool.init(&s.ctx, 1);
    defer pool.deinit();
    var ev = try pool.event(0);
    defer ev.deinit();
    rt.launch.launch(&s.queue, &k, .{
        .global = .{ .x = parsed.global[0], .y = parsed.global[1], .z = parsed.global[2] },
        .local = .{ .x = parsed.local[0], .y = parsed.local[1], .z = parsed.local[2] },
    }, &ev) catch |e| return fail("launch: {s}", .{@errorName(e)});
    // Queue the read-backs behind the kernel on the in-order list, then one synchronize.
    for (parsed.args, 0..) |a, i| {
        if (a.out == null) continue;
        const n = a.bytes.?;
        stage[i] = try rt.memory.Buffer.alloc(&s.ctx, .host, n);
        try s.queue.copy(stage[i].?.ptr, bufs[i].?.ptr, n, null);
    }
    try s.queue.synchronize();
    std.debug.print("kernel {s}: {d:.0} ns on device\n", .{ parsed.kernel, try ev.kernelNs() });

    for (parsed.args, 0..) |a, i| {
        const name = a.out orelse continue;
        const out_path = z(&pb, "{s}/{s}", .{ dir, name }) orelse return fail("path too long", .{});
        if (!sys.writeFile(out_path, try stage[i].?.bytes())) return fail("cannot write {s}", .{name});
    }
    return 0;
}
