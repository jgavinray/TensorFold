//! tf-xpu-qwen27-test [model dir] [capacity]: on-card gates for Qwen 3.8 27B weights and state on a B70.
//! Gate 1: every language-model tensor of the multi-shard checkpoint loaded to the card reads back with the SHA-256
//! of its host bytes. Gate 2: the recurrence/conv/KV buffers for the given capacity (default 4096) start zeroed,
//! take a pattern and read it back exactly. Prints tensor count, bytes, load time and card memory used.
//! Expected runtime: about 1-3 min once a card is free (the runtime's lock waits for one). Exit status 0 only if both pass.
const std = @import("std");
const rt = @import("xpu/runtime.zig");
const sys = @import("xpu/sys.zig");
const cfg = @import("families/qwen27/config.zig");
const xw = @import("families/qwen27/xpu_weights.zig");
const xs = @import("families/qwen27/xpu_state.zig");

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

const MemState = extern struct { stype: u32 = 0x1e, pNext: ?*anyopaque = null, health: u32 = 0, free: u64 = 0, size: u64 = 0 };

/// Card memory in use as Level Zero Sysman reports it (ZES_ENABLE_SYSMAN=1); null when Sysman is not available.
fn cardUsedBytes(s: *rt.session.Session, lib: *std.DynLib) ?u64 {
    const C = std.builtin.CallingConvention.c;
    const Enum = *const fn (?*anyopaque, *u32, ?[*]?*anyopaque) callconv(C) u32;
    const Get = *const fn (?*anyopaque, *MemState) callconv(C) u32;
    const enum_fn = lib.lookup(Enum, "zesDeviceEnumMemoryModules") orelse {
        std.debug.print("sysman: zesDeviceEnumMemoryModules not exported\n", .{});
        return null;
    };
    const get_fn = lib.lookup(Get, "zesMemoryGetState") orelse return null;
    if (lib.lookup(*const fn (u32) callconv(C) u32, "zesInit")) |init_fn| _ = init_fn(0);
    var n: u32 = 0;
    const r0 = enum_fn(s.ctx.dev.handle, &n, null);
    if (r0 != 0 or n == 0 or n > 8) {
        std.debug.print("sysman: enum result 0x{x}, {d} modules\n", .{ r0, n });
        return null;
    }
    var mods: [8]?*anyopaque = @splat(null);
    if (enum_fn(s.ctx.dev.handle, &n, &mods) != 0) return null;
    var used: u64 = 0;
    for (mods[0..n]) |m| {
        var ms: MemState = .{};
        if (get_fn(m, &ms) != 0 or ms.size == 0) return null;
        used += ms.size - ms.free;
    }
    return used;
}

fn mib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 20);
}

fn loadConfig(gpa: std.mem.Allocator, dir: []const u8) !cfg.Config {
    var cb: [1024]u8 = undefined;
    const cpath = sys.zfmt(&cb, "{s}/config.json", .{dir}) orelse return error.PathTooLong;
    const text = sys.readFileAlloc(gpa, cpath, 1 << 22) orelse return error.NoConfig;
    defer gpa.free(text);
    var gen: ?[]u8 = null;
    var gb: [1024]u8 = undefined;
    if (sys.zfmt(&gb, "{s}/generation_config.json", .{dir})) |gp| gen = sys.readFileAlloc(gpa, gp, 1 << 22);
    defer if (gen) |g| gpa.free(g);
    return cfg.parse(gpa, text, gen);
}

/// The checkpoint's affine group size from config.json's quantization block (this loader assumes one global format).
fn loadGroup(gpa: std.mem.Allocator, dir: []const u8) !usize {
    var cb: [1024]u8 = undefined;
    const cpath = sys.zfmt(&cb, "{s}/config.json", .{dir}) orelse return error.PathTooLong;
    const text = sys.readFileAlloc(gpa, cpath, 1 << 22) orelse return error.NoConfig;
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const root = switch (parsed.value) {
        .object => |o| o,
        else => return error.BadConfig,
    };
    for ([_][]const u8{ "quantization", "quantization_config" }) |key| {
        const v = root.get(key) orelse continue;
        if (v != .object) continue;
        const q = v.object;
        if (q.get("group_size")) |gs| {
            if (gs != .integer or gs.integer < 1) return error.BadAffine;
            return @intCast(gs.integer);
        }
    }
    return error.MissingAffineFormat;
}

fn weightsGate(gpa: std.mem.Allocator, s: *rt.session.Session, lib: *std.DynLib, dir: []const u8) !bool {
    const config = try loadConfig(gpa, dir);
    const group = try loadGroup(gpa, dir);

    const before = cardUsedBytes(s, lib);
    var w = try xw.load(gpa, s, dir, config, group);
    defer w.deinit();
    const after = cardUsedBytes(s, lib);
    std.debug.print("weights: {d} tensors from {d} shards, {d} bytes ({d:.1} MiB, {d:.2} GiB), affine 4-bit g{d}, layers {d}, load {d:.2} s ({d:.2} GB/s incl. hashing)\n", .{
        w.ledger.len,
        w.shards,
        w.total_bytes,
        mib(w.total_bytes),
        @as(f64, @floatFromInt(w.total_bytes)) / (1 << 30),
        group,
        config.layers,
        @as(f64, @floatFromInt(w.load_ns)) / 1e9,
        @as(f64, @floatFromInt(w.total_bytes)) / @as(f64, @floatFromInt(w.load_ns)),
    });
    std.debug.print("  load split: file read {d:.2} s, host hash {d:.2} s, device upload {d:.2} s\n", .{ @as(f64, @floatFromInt(w.read_ns)) / 1e9, @as(f64, @floatFromInt(w.hash_ns)) / 1e9, @as(f64, @floatFromInt(w.upload_ns)) / 1e9 });
    if (before != null and after != null)
        std.debug.print("card memory used (Sysman): before {d:.1} MiB, after {d:.1} MiB, delta {d:.1} MiB\n", .{ mib(before.?), mib(after.?), mib(after.? - before.?) })
    else
        std.debug.print("card memory used: Sysman unavailable; our device allocation is {d:.1} MiB\n", .{mib(w.arena.len)});
    const t0 = sys.nowNs();
    const bad = try w.verify(s, true);
    std.debug.print("readback: {d} of {d} tensor hashes differ ({d:.2} s)\n", .{ bad, w.ledger.len, @as(f64, @floatFromInt(sys.nowNs() - t0)) / 1e9 });
    // Spot-print three ledger lines so the hashes are visible in the log.
    for ([_]usize{ 0, w.ledger.len / 2, w.ledger.len - 1 }) |i| {
        const e = w.ledger[i];
        std.debug.print("  {s} @{d} {d} B sha256={x}\n", .{ e.name, e.tensor.offset, e.tensor.bytes, e.sha256[0..8] });
    }
    return bad == 0;
}

fn fillPattern(buf: []u8, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    prng.random().bytes(buf);
}

fn stateGate(gpa: std.mem.Allocator, s: *rt.session.Session, c: cfg.Config, capacity: usize) !bool {
    var cache = try xs.Cache.init(gpa, s, c, capacity);
    defer cache.deinit();
    const n_bufs = cache.bufferCount();
    std.debug.print("state: capacity {d}, {d} bytes ({d:.1} MiB): {d} linear layers x ({d} recurrence + {d} conv), {d} attention layers x 2 x {d} KV, logits {d}\n", .{
        capacity, cache.total_bytes, mib(cache.total_bytes),
        countKind(c, .linear), xs.deltaBytes(c), xs.convBytes(c),
        countKind(c, .attention), xs.kvBytes(c, capacity), xs.logitsBytes(c),
    });
    const bufs = try gpa.alloc(*rt.memory.Buffer, n_bufs);
    defer gpa.free(bufs);
    const n = cache.each(bufs);
    std.debug.assert(n == n_bufs);
    var max_len: usize = 0;
    for (bufs[0..n]) |b| max_len = @max(max_len, b.len);
    var up = try rt.memory.Buffer.alloc(&s.ctx, .host, max_len);
    defer up.free();
    var down = try rt.memory.Buffer.alloc(&s.ctx, .host, max_len);
    defer down.free();
    const ub = try up.bytes();
    const db = try down.bytes();

    var bad: usize = 0;
    // Recurrence and conv state must start at zero.
    for (cache.blocks[0..cache.made]) |*b| if (b.* == .linear) for ([_]*rt.memory.Buffer{ &b.linear.recurrence, &b.linear.conv }) |z| {
        try z.download(&s.queue, 0, db[0..z.len]);
        for (db[0..z.len]) |x| if (x != 0) {
            bad += 1;
            break;
        };
    };
    std.debug.print("state: {d} recurrence/conv buffers not zero at start\n", .{bad});
    // Pattern write, exact read-back, and no buffer bleeding into another (each has its own seed).
    for (bufs[0..n], 0..) |b, i| {
        fillPattern(ub[0..b.len], 0x5eed0000 + i);
        try b.upload(&s.queue, 0, ub[0..b.len]);
    }
    var wrong: usize = 0;
    for (bufs[0..n], 0..) |b, i| {
        fillPattern(ub[0..b.len], 0x5eed0000 + i);
        try b.download(&s.queue, 0, db[0..b.len]);
        if (!std.mem.eql(u8, ub[0..b.len], db[0..b.len])) {
            wrong += 1;
            std.debug.print("  buffer {d} ({d} bytes) differs\n", .{ i, b.len });
        }
    }
    std.debug.print("state: {d} buffers pattern-written, {d} read back different\n", .{ n, wrong });
    return bad == 0 and wrong == 0;
}

fn countKind(c: cfg.Config, k: cfg.Kind) usize {
    var n: usize = 0;
    for (0..c.layers) |i| if (c.kind(i) == k) {
        n += 1;
    };
    return n;
}

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    const dir: []const u8 = if (argv.len > 1) argv[1] else "/storage01/models/Qwen3.8-27B-MLX-4bit";
    const capacity: usize = if (argv.len > 2) try std.fmt.parseInt(usize, argv[2], 10) else 4096;
    const gpa = std.heap.page_allocator;

    _ = setenv("ZES_ENABLE_SYSMAN", "1", 1);
    var s: rt.session.Session = undefined;
    s.open(.{}) catch |e| {
        std.debug.print("cannot open a card: {s} ({s})\n", .{ @errorName(e), s.diag.message() });
        return 1;
    };
    defer s.close();
    var lib = std.DynLib.open(rt.loader_name) catch return 1;
    defer lib.close();
    std.debug.print("card {s} ({s}), maxAlloc {d} MiB\n", .{ s.bdf(), s.ctx.dev.name(), s.ctx.dev.props.maxMemAllocSize >> 20 });

    const config = loadConfig(gpa, dir) catch |e| {
        std.debug.print("config error: {s}\n", .{@errorName(e)});
        return 1;
    };
    const w_ok = weightsGate(gpa, &s, &lib, dir) catch |e| {
        std.debug.print("weights gate error: {s}\n", .{@errorName(e)});
        return 1;
    };
    std.debug.print("GATE weights: {s}\n", .{if (w_ok) "PASS" else "FAIL"});
    const s_ok = stateGate(gpa, &s, config, capacity) catch |e| {
        std.debug.print("state gate error: {s}\n", .{@errorName(e)});
        return 1;
    };
    std.debug.print("GATE state: {s}\n", .{if (s_ok) "PASS" else "FAIL"});
    return if (w_ok and s_ok) 0 else 1;
}
