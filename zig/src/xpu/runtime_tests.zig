//! On-card Level Zero qualification (opt-in: `zig build xpu-test`). Real drivers only; failures never fall back to a mock.
//! Each test takes the card lock (shared with the Python harness) and runs the harness preflight first.
//! Expected runtime: about 10-20 s for the whole file with both cards free.
const std = @import("std");
const rt = @import("runtime.zig");
const Session = @import("session.zig").Session;
const kernels = @import("xpu_kernels");
const sys = @import("sys.zig");

const gpa = std.heap.page_allocator;

fn open(s: *Session, card: usize) !void {
    if (!try openCard(s, card)) return error.CardNotBound;
}

/// Opens a session on `card`, returning false when the card is parked (no driver bound): such a card is not in the
/// lock pool, so `only` would wait out `wait_ms` and time out. This suite tests the usable cards; the host may
/// legitimately hold a parked one.
fn openCard(s: *Session, card: usize) !bool {
    if (!rt.preflight.isBound(.{}, rt.device.cards[card])) {
        std.debug.print("card {d} ({s}) has no driver bound; skipped\n", .{ card, rt.device.cards[card] });
        return false;
    }
    s.open(.{ .only = card, .wait_ms = 300_000 }) catch |e| {
        std.debug.print("session on card {d} refused: {s} ({s})\n", .{ card, @errorName(e), s.diag.message() });
        return e;
    };
    return true;
}

fn loadBinary(path: []const u8) ![]u8 {
    var z: [512:0]u8 = undefined;
    @memcpy(z[0..path.len], path);
    z[path.len] = 0;
    return sys.readFileAlloc(gpa, &z, 16 << 20) orelse error.KernelBinaryMissing;
}

fn loadModule(s: *Session, path: []const u8) !rt.module.Module {
    const bin = try loadBinary(path);
    defer gpa.free(bin);
    var log: rt.module.BuildLog = .{};
    return rt.module.Module.load(&s.ctx, bin, &log) catch |e| {
        std.debug.print("module load failed: {s}\n", .{log.text()});
        return e;
    };
}

// ---- requirement 1 + 3: selection by PCI address, refusal of anything else, shared lock -----------------------------

test "xpu req1/req3: select both B70s by PCI address, refuse others, hold the shared card lock" {
    for (0..rt.device.cards.len) |card| {
        var s: Session = undefined;
        open(&s, card) catch |e| switch (e) {
            error.CardNotBound => continue, // a parked card: not in the pool, nothing to test here
            else => return e,
        };
        defer s.close();
        const d = s.ctx.dev;
        var buf: [12]u8 = undefined;
        std.debug.print("card {d}: {s} bdf={s} vendor=0x{x} device=0x{x} timer={d}ns maxAlloc={d} MiB compute_ordinal={d}\n", .{
            card, d.name(), d.bdf.format(&buf), d.props.vendorId, d.props.deviceId, d.props.timerResolution, d.props.maxMemAllocSize >> 20, s.ctx.compute_ordinal,
        });
        try std.testing.expectEqualStrings(rt.device.cards[card], d.bdf.format(&buf));
        try std.testing.expectEqual(@as(u32, 0x8086), d.props.vendorId);
        // While this session holds the card, its lock is taken.
        try std.testing.expect((try rt.lock.tryCard(rt.lock.default_dir, card, .{})) == null);
        for ([_][]const u8{ "0000:c1:00.0", "0000:01:00.0", "0000:00:02.0" }) |bad|
            try std.testing.expectError(error.ForbiddenCard, rt.device.select(&s.rt, bad));
    }
    // Locks are free again after close (a parked card is not in the pool, so `tryCard` returns null for it).
    for (0..rt.device.cards.len) |card| {
        var held = try rt.lock.tryCard(rt.lock.default_dir, card, .{});
        if (held) |*h| h.release();
    }
}

test "xpu req2: preflight on the live host reads 100 W caps and 24 GiB" {
    var d: rt.preflight.Diag = .{};
    // Every bound card must pass preflight; a parked card is skipped by the sweep, so it is not asserted.
    for (rt.device.cards) |card| {
        if (!rt.preflight.isBound(.{}, card)) {
            std.debug.print("card {s} has no driver bound; skipped\n", .{card});
            continue;
        }
        rt.preflight.check(.{}, card, &d) catch |e| {
            std.debug.print("preflight refused {s}: {s}\n", .{ card, d.message() });
            return e;
        };
    }
    std.debug.print("MemAvailable {d} MiB\n", .{rt.preflight.memAvailableKiB("/proc/meminfo").? >> 10});
}

// ---- requirement 4: alloc/free/copies, 256 MiB round trip ----------------------------------------------------------

test "xpu req4: 256 MiB host-device-host round trip is bit-exact on both cards; host and shared allocations work" {
    const size: usize = 256 << 20;
    for (0..rt.device.cards.len) |card| {
        var s: Session = undefined;
        if (!try openCard(&s, card)) continue; // a parked card is not in the pool
        defer s.close();
        var src = try rt.memory.Buffer.alloc(&s.ctx, .host, size);
        defer src.free();
        var dst = try rt.memory.Buffer.alloc(&s.ctx, .host, size);
        defer dst.free();
        var dev = try rt.memory.Buffer.alloc(&s.ctx, .device, size);
        defer dev.free();
        var prng = std.Random.DefaultPrng.init(0x7f4a7c15 + card);
        prng.random().bytes(try src.bytes());
        @memset(try dst.bytes(), 0);
        const t0 = sys.nowNs();
        try dev.upload(&s.queue, 0, try src.bytes());
        const t1 = sys.nowNs();
        const up_ns = t1 - t0;
        try dev.download(&s.queue, 0, try dst.bytes());
        const down_ns = sys.nowNs() - t1;
        try std.testing.expect(std.mem.eql(u8, try src.bytes(), try dst.bytes()));
        std.debug.print("card {d}: 256 MiB round trip bit-exact; up {d:.2} GB/s down {d:.2} GB/s\n", .{
            card,
            @as(f64, @floatFromInt(size)) / @as(f64, @floatFromInt(up_ns)),
            @as(f64, @floatFromInt(size)) / @as(f64, @floatFromInt(down_ns)),
        });

        // shared memory is host-addressable and device-visible
        var sh = try rt.memory.Buffer.alloc(&s.ctx, .shared, 1 << 20);
        defer sh.free();
        const sb = try sh.bytes();
        @memset(sb, 0x5a);
        try s.queue.fill(sh.ptr, &[_]u8{0xc3}, sb.len / 2, null);
        try s.queue.synchronize();
        try std.testing.expectEqual(@as(u8, 0xc3), sb[0]);
        try std.testing.expectEqual(@as(u8, 0xc3), sb[sb.len / 2 - 1]);
        try std.testing.expectEqual(@as(u8, 0x5a), sb[sb.len / 2]);
    }
}

// ---- requirement 5: in-order regression ---------------------------------------------------------------------------

test "xpu req5: fill sentinel, kernel writes, copy back - 100/100 exact with no host sync between the three" {
    const n: u32 = 4096;
    const runs = 100;
    for (0..rt.device.cards.len) |card| {
        var s: Session = undefined;
        if (!try openCard(&s, card)) continue; // a parked card is not in the pool
        defer s.close();
        var m = try loadModule(&s, kernels.runtime_tests);
        defer m.unload();
        var k = try m.kernel("tf_xpu_write");
        defer k.deinit();
        var dev = try rt.memory.Buffer.alloc(&s.ctx, .device, n * 4);
        defer dev.free();
        var host = try rt.memory.Buffer.alloc(&s.ctx, .host, n * 4);
        defer host.free();
        const out: []u32 = @alignCast(std.mem.bytesAsSlice(u32, try host.bytes()));
        var exact: usize = 0;
        for (0..runs) |run| {
            // Sentinel differs per run so a stale readback of the previous result also fails.
            const sentinel: u32 = 0xDEADBE00 + @as(u32, @intCast(run));
            @memset(out, 0xFFFFFFFF);
            try s.queue.fill(dev.ptr, std.mem.asBytes(&sentinel), n * 4, null);
            try rt.launch.run(&s.queue, &k, .{ .global = .{ .x = n }, .local = .{ .x = 256 } }, .{ dev.ptr, n }, null);
            try s.queue.copy(host.ptr, dev.ptr, n * 4, null);
            try s.queue.synchronize();
            var ok = true;
            for (out, 0..) |v, i| {
                if (v != @as(u32, @intCast(i)) * 3 + 7) ok = false;
            }
            if (ok) exact += 1;
        }
        std.debug.print("card {d}: in-order regression {d}/{d} exact\n", .{ card, exact, runs });
        try std.testing.expectEqual(@as(usize, runs), exact);
    }
}

// ---- requirement 6: module, args, group size, launch, events, timing ---------------------------------------------

test "xpu req6: native module, mixed-width kernel args, group size, launch, event timestamps" {
    const n: u32 = 1025;
    for (0..rt.device.cards.len) |card| {
        var s: Session = undefined;
        if (!try openCard(&s, card)) continue; // a parked card is not in the pool
        defer s.close();
        var m = try loadModule(&s, kernels.runtime_tests);
        defer m.unload();
        var k = try m.kernel("tf_xpu_probe");
        defer k.deinit();
        try std.testing.expectError(error.ZeFailed, m.kernel("no_such_kernel"));
        var dev = try rt.memory.Buffer.alloc(&s.ctx, .device, n * 4);
        defer dev.free();
        var host = try rt.memory.Buffer.alloc(&s.ctx, .host, n * 4);
        defer host.free();
        const out: []u32 = @alignCast(std.mem.bytesAsSlice(u32, try host.bytes()));
        var pool = try rt.event.Pool.init(&s.ctx, 4);
        defer pool.deinit();
        var ev = try pool.event(0);
        defer ev.deinit();
        const zero: u32 = 0;
        try s.queue.fill(dev.ptr, std.mem.asBytes(&zero), n * 4, null);
        const global: u32 = 1280; // 5 groups of 256; the kernel guards i < n
        try rt.launch.run(&s.queue, &k, .{ .global = .{ .x = global }, .local = .{ .x = 256 } }, .{ dev.ptr, n, @as(u8, 17), @as(u64, 0x0000000300000000) }, &ev);
        try s.queue.copy(host.ptr, dev.ptr, n * 4, null);
        try ev.wait();
        try s.queue.synchronize();
        for (out, 0..) |v, i| try std.testing.expectEqual(@as(u32, @intCast(i + 20)), v);
        const ns = try ev.kernelNs();
        std.debug.print("card {d}: probe kernel exact; event-timestamp duration {d:.0} ns (timer {d} ns/tick)\n", .{ card, ns, s.ctx.dev.props.timerResolution });
        try std.testing.expect(ns > 0 and ns < 1e9);
        // a bad geometry is refused before the driver sees it
        try std.testing.expectError(error.Invalid, rt.launch.launch(&s.queue, &k, .{ .global = .{ .x = 100 }, .local = .{ .x = 64 } }, null));
        // an event can be reset and reused
        try ev.reset();
    }
}

// ---- requirement 7: DPAS int8 / int4 probe -------------------------------------------------------------------------

fn pack(comptime bits: u5, vals: []const i32) u32 {
    var w: u32 = 0;
    for (vals, 0..) |v, i| w |= (@as(u32, @bitCast(v)) & ((@as(u32, 1) << bits) - 1)) << @intCast(i * bits);
    return w;
}

/// One DPAS check: random A (1xK) and B (Kx16), run on the card, compare C[0][0..16] with a CPU reference.
fn dpasRun(s: *Session, k: *rt.module.Kernel, comptime K: u32, comptime bits: u5, seed: u64, da: *rt.memory.Buffer, db: *rt.memory.Buffer, dc: *rt.memory.Buffer, hc: *rt.memory.Buffer) !bool {
    const per = 16 / bits;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const lo: i32 = if (bits == 4) -8 else -128;
    const hi: i32 = if (bits == 4) 7 else 127;
    var a: [K]i32 = undefined;
    var b: [K * 16]i32 = undefined;
    for (&a) |*x| x.* = rnd.intRangeAtMost(i32, lo, hi);
    for (&b) |*x| x.* = rnd.intRangeAtMost(i32, lo, hi);
    var A: [16]i16 = undefined;
    var B: [128]u32 = undefined;
    for (0..16) |w| A[w] = @bitCast(@as(u16, @truncate(pack(bits, a[w * per ..][0..per]))));
    for (0..16) |n| for (0..8) |j| {
        var col: [2 * per]i32 = undefined;
        for (0..2 * per) |i| col[i] = b[(j * 2 * per + i) * 16 + n];
        B[n * 8 + j] = pack(bits, &col);
    };
    const sentinel: u32 = 0x7eadbeef;
    try da.upload(&s.queue, 0, std.mem.asBytes(&A));
    try db.upload(&s.queue, 0, std.mem.asBytes(&B));
    try s.queue.fill(dc.ptr, std.mem.asBytes(&sentinel), 64, null);
    try rt.launch.run(&s.queue, k, .{ .global = .{ .x = 16 }, .local = .{ .x = 16 } }, .{ da.ptr, db.ptr, dc.ptr }, null);
    try s.queue.copy(hc.ptr, dc.ptr, 64, null);
    try s.queue.synchronize();
    const got: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, (try hc.bytes())[0..64]));
    for (0..16) |n| {
        var ref: i64 = 0;
        for (0..K) |kk| ref += @as(i64, a[kk]) * b[kk * 16 + n];
        if (got[n] != ref) return false;
    }
    return true;
}

test "xpu req7: DPAS int8 (K=32, 0x33) and int4 (K=64, 0xC3) bit-exact vs CPU, 100/100, both cards" {
    const runs = 100;
    for (0..rt.device.cards.len) |card| {
        var s: Session = undefined;
        if (!try openCard(&s, card)) continue; // a parked card is not in the pool
        defer s.close();
        var m = try loadModule(&s, kernels.dpas);
        defer m.unload();
        var k8 = try m.kernel("tf_dpas_int8");
        defer k8.deinit();
        var k4 = try m.kernel("tf_dpas_int4");
        defer k4.deinit();
        var da = try rt.memory.Buffer.alloc(&s.ctx, .device, 32);
        defer da.free();
        var db = try rt.memory.Buffer.alloc(&s.ctx, .device, 512);
        defer db.free();
        var dc = try rt.memory.Buffer.alloc(&s.ctx, .device, 64);
        defer dc.free();
        var hc = try rt.memory.Buffer.alloc(&s.ctx, .host, 64);
        defer hc.free();
        var ok8: usize = 0;
        var ok4: usize = 0;
        for (0..runs) |r| {
            // alternate order like the original probe (int8 first / int4 first)
            if (r % 2 == 0) {
                if (try dpasRun(&s, &k8, 32, 8, 1000 + r, &da, &db, &dc, &hc)) ok8 += 1;
                if (try dpasRun(&s, &k4, 64, 4, 5000 + r, &da, &db, &dc, &hc)) ok4 += 1;
            } else {
                if (try dpasRun(&s, &k4, 64, 4, 5000 + r, &da, &db, &dc, &hc)) ok4 += 1;
                if (try dpasRun(&s, &k8, 32, 8, 1000 + r, &da, &db, &dc, &hc)) ok8 += 1;
            }
        }
        std.debug.print("card {d}: DPAS int8 {d}/{d} exact, int4 {d}/{d} exact\n", .{ card, ok8, runs, ok4, runs });
        try std.testing.expectEqual(@as(usize, runs), ok8);
        try std.testing.expectEqual(@as(usize, runs), ok4);
    }
}

// ---- requirement: a recorded regular command list replays bit-exact vs the eager immediate-list run ----------------

test "xpu cmdlist: a recorded DPAS step replays bit-for-bit equal to the eager launch, 100/100" {
    const runs = 100;
    for (0..rt.device.cards.len) |card| {
        var s: Session = undefined;
        if (!try openCard(&s, card)) continue; // a parked card is not in the pool
        defer s.close();
        var m = try loadModule(&s, kernels.dpas);
        defer m.unload();
        var k = try m.kernel("tf_dpas_int8");
        defer k.deinit();
        var da = try rt.memory.Buffer.alloc(&s.ctx, .device, 32);
        defer da.free();
        var db = try rt.memory.Buffer.alloc(&s.ctx, .device, 512);
        defer db.free();
        var dc = try rt.memory.Buffer.alloc(&s.ctx, .device, 64);
        defer dc.free();
        var hc = try rt.memory.Buffer.alloc(&s.ctx, .host, 64);
        defer hc.free();

        // The step's inputs are read-only device buffers whose *contents* change between replays while their addresses
        // (baked into the recorded list) do not - the same contract the decode step's token/position/cache-length
        // values rely on. Record one launch of the DPAS step, then replay it against fresh inputs each round and check
        // every round equals the eager immediate-list launch on the same inputs.
        var rec = try rt.cmdlist.Recorder.init(&s.ctx);
        errdefer rec.deinit();
        try rec.run(&k, rt.launch.Config{ .global = .{ .x = 16 }, .local = .{ .x = 16 } }, .{ da.ptr, db.ptr, dc.ptr });
        // `take` closes the list and moves the handle into `list`; the recorder must not be destroyed afterwards.
        var list = try rt.cmdlist.Recorded.take(&rec);
        defer list.deinit();
        var q = try rt.cmdlist.Queue.init(&s.ctx);
        defer q.deinit();

        var exact: usize = 0;
        var eager_buf: [16]i32 = undefined;
        var replay_buf: [16]i32 = undefined;
        for (0..runs) |r| {
            var prng = std.Random.DefaultPrng.init(9000 + r);
            const rnd = prng.random();
            var a: [16]i16 = undefined;
            var b: [128]i32 = undefined;
            for (&a) |*x| x.* = @bitCast(@as(u16, rnd.int(u16)));
            for (&b) |*x| x.* = rnd.int(i32);
            // Eager path: immediate in-order list.
            try da.upload(&s.queue, 0, std.mem.asBytes(&a));
            try db.upload(&s.queue, 0, std.mem.asBytes(&b));
            try rt.launch.run(&s.queue, &k, .{ .global = .{ .x = 16 }, .local = .{ .x = 16 } }, .{ da.ptr, db.ptr, dc.ptr }, null);
            try s.queue.copy(hc.ptr, dc.ptr, 64, null);
            try s.queue.synchronize();
            @memcpy(std.mem.sliceAsBytes(&eager_buf), (try hc.bytes())[0..64]);
            // Replay path: same buffers, same recorded list, fresh contents uploaded between replays.
            try da.upload(&s.queue, 0, std.mem.asBytes(&a));
            try db.upload(&s.queue, 0, std.mem.asBytes(&b));
            try list.replaySync(&q);
            try s.queue.copy(hc.ptr, dc.ptr, 64, null);
            try s.queue.synchronize();
            @memcpy(std.mem.sliceAsBytes(&replay_buf), (try hc.bytes())[0..64]);
            if (std.mem.eql(i32, &eager_buf, &replay_buf)) exact += 1;
        }
        std.debug.print("card {d}: recorded-list replay vs eager {d}/{d} bit-exact\n", .{ card, exact, runs });
        try std.testing.expectEqual(@as(usize, runs), exact);
    }
}
