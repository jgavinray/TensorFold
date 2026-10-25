//! Device tests for the block-LDLQ driver (fable LDLQ spec section 4.7, T-LDLQ-2..7 and
//! the strip-mode timing of measurements 7.4/7.5). The arbiter is the host fp64 reference
//! core/exl3_ldlq_ref.zig (T-LDLQ-1), re-run here over the same synthetic slot: K=64,
//! N=32 at halves 3 (3 bits, the spec's comparison width). Skips without a Level Zero
//! device, like every tf-xpu-qwen27-*-test.
//!
//! T-LDLQ-2  the device loop reproduces the reference trellis bit for bit and every
//!           Result (cost, open_cost, pinned, status) per tile.
//! T-LDLQ-3  the kept eta strips read back bit-identical to the reference's f32(q)-f32(s).
//! T-LDLQ-4  the device eta_sq (deterministic partials summed in fp64) equals the
//!           reference's eta_sq within relative 1e-3; the gap histogram matches exactly;
//!           the host proxy (want_proxy) is byte-deterministic against proxyLoss.
//! T-LDLQ-5  planted |target| > 4 entries make the clamp counter nonzero and match the
//!           reference's count over the same T32 evolution.
//! T-LDLQ-6  the raw strip decoder (strip_decode) reproduces the codebook values of the
//!           trellis bit for bit - the WHAT oracle and the 4.6 proxy dequant, no
//!           rotation and no scale (exl3_dequant128_c is unusable there).
//! T-LDLQ-7  quantizePart twice with identical input returns identical trellis bytes and
//!           identical reports (the whole pipeline is deterministic).
//! Timing    benchPart reps times, with and without the rank-16 feedback: tiles/s and
//!           the update share, printed against the free-running T8 encoder number.
const std = @import("std");
const q = @import("qwen27.zig");
const core = @import("core");
const ldlq = @import("ldlq.zig");
const exl3_enc = @import("exl3_encode.zig");
const devm = @import("dev.zig");
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
const gpa = std.heap.c_allocator;
const K: u32 = 64; // four 16-row blocks
const N: u32 = 32; // two tiles wide, one part
const HALVES: u16 = 3; // 3 bits: the spec's comparison width

fn gauss(r: std.Random) f64 { // the box-muller the reference test uses
    const u = @max(r.float(f64), 1e-300);
    return @sqrt(-2.0 * @log(u)) * @cos(2.0 * std.math.pi * r.float(f64));
}

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len > 2 and std.mem.eql(u8, argv[1], "--real")) return realT2(argv[2..]);
    if (argv.len > 2 and std.mem.eql(u8, argv[1], "--bench")) return benchReal(argv[2..]);
    const g = q.ze.Gpu.open(0xe223) catch {
        std.debug.print("skipped: no Level Zero device\n", .{});
        return;
    };
    var d: q.Dev = .{ .g = g, .list = undefined, .k = undefined };
    errdefer d.g.close();
    d.list = try d.g.immediate(0, 0); // the encProbe pattern (qwen27_main.zig:955): the command list the copies append to
    const w: core.exl3_format.Width = .{ .halves = HALVES };
    const k: usize = K;
    const n: usize = N;
    const kblks = K / 16;
    const nt = N / 16;
    const words = w.tileWords();

    // The synthetic slot: SPD H' = A A^T + K*I, one part; the driver consumes W_q and L
    // as given (the 4.2 sign/RMS/rotation stage is CalibStage's capture, not the loop).
    var prng = std.Random.DefaultPrng.init(0x2D1A_9B97 ^ 0x4C9A_1F13); // fixed seed: T-LDLQ-7 input stability
    const r = prng.random();
    const a = try gpa.alloc(f64, k * k);
    defer gpa.free(a);
    for (a) |*x| x.* = gauss(r) * 0.25;
    const h = try gpa.alloc(f64, k * k);
    defer gpa.free(h);
    for (0..k) |i| for (0..k) |cj| {
        var acc: f64 = 0;
        for (0..k) |p| acc += a[i * k + p] * a[cj * k + p];
        h[i * k + cj] = acc + if (i == cj) @as(f64, @floatFromInt(k)) else 0;
    };
    var pivot: usize = 0;
    var bldl = try core.exl3_ldl.blockLdl(gpa, h, k, 0.01, 0.001, &pivot);
    defer bldl.deinit(gpa);
    const wq64 = try gpa.alloc(f64, k * n);
    defer gpa.free(wq64);
    for (wq64) |*x| x.* = gauss(r);
    const sv = try gpa.alloc(i8, n);
    defer gpa.free(sv);
    @memset(sv, 1);
    const s = try gpa.alloc(f32, n);
    defer gpa.free(s);
    @memset(s, 1.0);
    const svh = try gpa.alloc(f16, n);
    defer gpa.free(svh);
    @memset(svh, 0);
    const slot: ldlq.Slot = .{ .k = K, .su = &.{}, .suh = &.{}, .ldl = &bldl, .parts = &.{} };
    const part: ldlq.Part = .{ .n = N, .n0 = 0, .sv = sv, .s = s, .svh = svh };

    // T-LDLQ-1's arbiter on identical inputs, eta strips included for T-LDLQ-3. The
    // trellis is duplicated now: T-LDLQ-5 re-runs the reference on planted data and
    // would clobber the buffer T-LDLQ-6 reads as its oracle.
    var metric = core.exl3_encode.Metric.init();
    const tr_ref = try gpa.alloc(i16, kblks * nt * words);
    defer gpa.free(tr_ref);
    const res_ref = try gpa.alloc(core.exl3_encode.Result, kblks * nt);
    defer gpa.free(res_ref);
    const eta_ref = try gpa.alloc(f32, k * n);
    defer gpa.free(eta_ref);
    const rep_ref = try core.exl3_ldlq_ref.ldlqRef(gpa, &metric, w, wq64, k, n, bldl.l, tr_ref, res_ref, eta_ref, "ref_dump.txt");
    const tr_oracle = try gpa.dupe(i16, tr_ref);
    defer gpa.free(tr_oracle);

    var dr = try ldlq.Driver.init(&d, w, K, N, 16);
    dr.want_eta = true;
    dr.want_proxy = true;
    dr.dbg_dump = "dev_dump.txt"; // per-block S16/Q16/T32/STAT evidence for the bisect
    try dr.beginSlot(&d, &slot);

    const tr = try gpa.alloc(i16, kblks * nt * words);
    defer gpa.free(tr);
    const stat = try gpa.alloc(ldlq.TileStat, kblks * nt);
    defer gpa.free(stat);
    const rep = try dr.quantizePart(&d, &slot, &part, wq64, tr, stat);

    // Evidence before the verdict: per tile, trellis-word agreement and the two
    // certificates side by side, so the run log names the first divergent tile and
    // splits cost-mismatch from chain-mismatch without another card cycle.
    for (0..kblks * nt) |t| {
        var same = true;
        for (tr[t * words ..][0..words], tr_ref[t * words ..][0..words]) |x, y| {
            if (x != y) same = false;
        }
        const hr = res_ref[t];
        const ds = stat[t];
        std.debug.print("T2 tile {d} (j={d} tn={d}) words {s}: dev cost {x} open {x} pin {d} st {d} | ref cost {x} open {x} pin {d}\n", .{
            t,       t / nt,       t % nt,    @as([]const u8, if (same) "=" else "X"),
            ds.cost, ds.open_cost, ds.pinned, ds.status,
            hr.cost, hr.open_cost, hr.pinned,
        });
    }
    try std.testing.expectEqualSlices(i16, tr_ref, tr);
    for (res_ref, stat) |hr, ds| {
        try std.testing.expectEqual(@as(u64, hr.cost), ds.cost);
        try std.testing.expectEqual(@as(u64, hr.open_cost), ds.open_cost);
        try std.testing.expectEqual(@as(u32, hr.pinned), ds.pinned);
        const want: u32 = switch (hr.status) {
            .optimal => 0,
            .closed => 1,
            .failed => 2,
        };
        try std.testing.expectEqual(want, ds.status);
    }
    try std.testing.expectEqual(rep_ref.failed, rep.failed);
    std.debug.print("T-LDLQ-2 ok: {d} tiles bit-identical trellis and certificates\n", .{rep.tiles});

    // T-LDLQ-3: eta strips, bit for bit (one f32 rounding of the same fp64 difference).
    const eta = try gpa.alloc(f32, k * n);
    defer gpa.free(eta);
    try dr.readEta(&d, eta, K, N);
    try std.testing.expectEqualSlices(f32, eta_ref, eta);
    std.debug.print("T-LDLQ-3 ok: {d} eta entries bit-identical\n", .{k * n});

    // T-LDLQ-4: eta_sq tolerance gate, exact histogram, deterministic host proxy.
    const rel = @abs(rep.eta_sq - rep_ref.eta_sq) / @max(rep_ref.eta_sq, 1e-30);
    if (rel > 1e-3) return error.EtaSqDrift;
    try std.testing.expectEqualSlices(u32, &rep_ref.gap_hist, &rep.gap_hist);
    const proxy_ref = try core.exl3_ldlq_ref.proxyLoss(gpa, w, bldl.l, bldl.d, tr, k, n, wq64);
    const got_proxy = rep.proxy orelse return error.ProxyMissing;
    try std.testing.expectEqual(@as(u64, @bitCast(proxy_ref)), @as(u64, @bitCast(got_proxy)));
    std.debug.print("T-LDLQ-4 ok: eta_sq device {d:.3} vs host {d:.3} (rel {d:.6}), proxy {d:.6} bit-exact\n", .{ rep.eta_sq, rep_ref.eta_sq, rel, got_proxy });

    // T-LDLQ-6 first: decodePart reads tr_dev, which the next quantizePart clobbers.
    const dec = try gpa.alloc(f16, k * n);
    defer gpa.free(dec);
    try dr.decodePart(&d, dec, N);
    const tss = try gpa.alloc([256]u32, kblks * nt);
    defer gpa.free(tss);
    for (0..kblks) |bj| {
        for (0..nt) |tn| {
            core.exl3_format.tileStates(w, tr_oracle[(bj * nt + tn) * words ..][0..words], &tss[bj * nt + tn]);
        }
    }
    var placement: [256]core.exl3_format.Placement = undefined;
    core.exl3_format.tilePlacement(&placement);
    for (0..kblks) |bj| {
        for (0..nt) |tn| {
            for (0..256) |p| {
                const pl = placement[p];
                const want = core.exl3_format.codebookValue(.mul1, @intCast(tss[bj * nt + tn][p]));
                const got = dec[(bj * 16 + @as(usize, @intCast(pl.row))) * n + @as(usize, @intCast(tn)) * 16 + @as(usize, @intCast(pl.col))];
                try std.testing.expectEqual(@as(u16, @bitCast(want)), @as(u16, @bitCast(got)));
            }
        }
    }
    std.debug.print("T-LDLQ-6 ok: strip_decode reproduces {d} codebook values bit for bit\n", .{k * n});

    // T-LDLQ-5: planted |target| > 4 entries; the device counter must match the
    // reference's count over the same target evolution, and be nonzero to prove it fired.
    const wq_clamp = try gpa.alloc(f64, k * n);
    defer gpa.free(wq_clamp);
    for (wq_clamp, 0..) |*x, i| x.* = if (i % 97 == 0) 4.25 else wq64[i];
    const tr2 = try gpa.alloc(i16, kblks * nt * words);
    defer gpa.free(tr2);
    const stat2 = try gpa.alloc(ldlq.TileStat, kblks * nt);
    defer gpa.free(stat2);
    const rep2 = try dr.quantizePart(&d, &slot, &part, wq_clamp, tr2, stat2);
    const rep2_ref = try core.exl3_ldlq_ref.ldlqRef(gpa, &metric, w, wq_clamp, k, n, bldl.l, tr_ref, res_ref, null, null);
    try std.testing.expectEqual(rep2_ref.clamped, rep2.clamped);
    if (rep2.clamped == 0) return error.ClampCounterDead;
    std.debug.print("T-LDLQ-5 ok: clamp counter {d} matches the host count over the same targets\n", .{rep2.clamped});

    // T-LDLQ-7: byte-identical reproducibility, two fresh runs of the original part.
    const tr_x = try gpa.alloc(i16, kblks * nt * words);
    defer gpa.free(tr_x);
    const stat_x = try gpa.alloc(ldlq.TileStat, kblks * nt);
    defer gpa.free(stat_x);
    const tr_y = try gpa.alloc(i16, kblks * nt * words);
    defer gpa.free(tr_y);
    const stat_y = try gpa.alloc(ldlq.TileStat, kblks * nt);
    defer gpa.free(stat_y);
    const rep_x = try dr.quantizePart(&d, &slot, &part, wq64, tr_x, stat_x);
    const rep_y = try dr.quantizePart(&d, &slot, &part, wq64, tr_y, stat_y);
    try std.testing.expectEqualSlices(i16, tr_x, tr_y);
    try std.testing.expectEqualSlices(ldlq.TileStat, stat_x, stat_y);
    try std.testing.expectEqual(@as(u64, @bitCast(rep_x.eta_sq)), @as(u64, @bitCast(rep_y.eta_sq)));
    try std.testing.expectEqual(rep_x.optimal, rep_y.optimal);
    try std.testing.expectEqual(rep_x.closed, rep_y.closed);
    try std.testing.expectEqual(rep_x.clamped, rep_y.clamped);
    std.debug.print("T-LDLQ-7 ok: two runs byte-identical\n", .{});

    // Strip-mode timing (measurements 7.4/7.5), printed, not asserted: tiles/s with the
    // full loop and the update-free variant; the T8 comparison is the report's job.
    const reps: u32 = 50;
    const full_ns = try dr.benchPart(&d, wq_clamp, N, reps, false);
    const skip_ns = try dr.benchPart(&d, wq_clamp, N, reps, true);
    const tiles: f64 = @floatFromInt(kblks * nt * @as(usize, reps));
    const full_s: f64 = @as(f64, @floatFromInt(full_ns)) / 1e9;
    const skip_s: f64 = @as(f64, @floatFromInt(skip_ns)) / 1e9;
    std.debug.print("strip-mode: {d:.1} tiles/s full, {d:.1} tiles/s without ldlq_update, update share {d:.2}% (against the free-running T8 number)\n", .{
        tiles / full_s, tiles / skip_s, 100.0 * (full_s - skip_s) / full_s,
    });
    std.debug.print("ok\n", .{});
}

/// Spec T-LDLQ-2 at real size: quantizePart on one real pack part's decoded W_q with an
/// identity L — no feedback, so the loop machinery must reproduce the calibration-free
/// Encoder byte for byte on the same fp16 strip. Usage: --real PACK_DIR [LAYER] [PART].
/// The W_q comes from the serving dequant (calib.dequantSlot), so the strip the driver
/// builds is f16-exact and T32 never moves (identity L's strictly-lower entries are 0).
fn realT2(args: []const [:0]const u8) !void {
    const calib = @import("calib.zig");
    if (std.c.getenv("XPU_CHUNK") == null) _ = setenv("XPU_CHUNK", "2048", 0);
    if (std.c.getenv("XPU_CTX") == null) _ = setenv("XPU_CTX", "4096", 0);
    const layer_i = if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else 0;
    const part_i = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 0;
    const r = try q.Runner.init(args[0], 4096);
    defer r.deinit();
    const p = r.model.layers[layer_i].mixer_in;
    const hx = p.exl3 orelse return error.NoTrellis;
    const part = hx.parts[part_i];
    if (part.dense) return error.DensePart;
    const w: core.exl3_format.Width = .{ .halves = part.halves };
    const k: usize = p.k;
    const n: usize = part.n;
    const nt = n / 16;
    const words = w.tileWords();
    std.debug.print("real part: layer {d} part {d} K={d} N={d} n0={d} halves {d} ({d} blocks x {d} tiles)\n", .{ layer_i, part_i, k, n, part.n0, w.halves, k / 16, nt });

    // the pack's own decoded weights: W_q for the driver, the f16 strip for the baseline
    const slab = try r.d.alloc_(@as(usize, k) * hx.nt * 2);
    defer r.d.g.free(slab);
    try calib.dequantSlot(&r.d, hx, p.k, hx.nt, slab);
    const slab_h = try gpa.alloc(f16, k * hx.nt);
    defer gpa.free(slab_h);
    try readDev(&r.d, slab, std.mem.sliceAsBytes(slab_h));
    const wq64 = try gpa.alloc(f64, k * n);
    defer gpa.free(wq64);
    const strip16 = try gpa.alloc(f16, k * n);
    defer gpa.free(strip16);
    for (0..k) |row| for (0..n) |c| {
        const v = slab_h[row * hx.nt + part.n0 + c];
        strip16[row * n + c] = v;
        wq64[row * n + c] = @floatCast(v);
    };

    // baseline: the calibration-free encoder over the same device strip, same launch
    // geometry as quantizePart's encodeWhat chunk (count 16, tile0 stepping in [0, nt))
    const strip_dev = try r.d.alloc_(k * n * 2);
    defer r.d.g.free(strip_dev);
    try r.d.copyIn(strip_dev, std.mem.sliceAsBytes(strip16));
    var base = try exl3_enc.Encoder.init(&r.d, w, 16, @intCast((k / 16) * nt));
    const tr_base = try gpa.alloc(i16, (k / 16) * nt * words);
    defer gpa.free(tr_base);
    const tr_base_dev = try r.d.alloc_(@as(usize, tr_base.len) * 2);
    defer r.d.g.free(tr_base_dev);
    var first: u32 = 0;
    const tt = (k / 16) * nt; // tiles run k-major across the full strip: tile = tk*nt + tn
    while (first < tt) : (first += 16) {
        if (first & 2047 == 0) std.debug.print("base: tile {d} of {d}\n", .{ first, tt });
        try base.encode(&r.d, strip_dev, @intCast(k), @intCast(n), tr_base_dev, first, @min(16, tt - first), null);
    }
    try exl3_enc.Encoder.readTrellis(&base, &r.d, tr_base_dev, tr_base, null);

    // the driver: identity L, the same part; the trellis must come back byte-identical
    const bldl = try gpa.create(core.exl3_ldl.BlockLdl);
    defer gpa.destroy(bldl);
    bldl.* = .{
        .k = @intCast(k),
        .l = try gpa.alloc(f32, k * k),
        .d = try gpa.alloc(f64, (k / 16) * 256),
        .lambda = 0,
        .mu = 0,
    };
    defer gpa.free(bldl.l);
    defer gpa.free(bldl.d);
    @memset(bldl.l, 0);
    for (0..k) |i| bldl.l[i * k + i] = 1;
    @memset(bldl.d, 1);
    const sv = try gpa.alloc(i8, n);
    defer gpa.free(sv);
    @memset(sv, 1);
    const s = try gpa.alloc(f32, n);
    defer gpa.free(s);
    @memset(s, 1);
    const svh = try gpa.alloc(f16, n);
    defer gpa.free(svh);
    @memset(svh, 0);
    const slot: ldlq.Slot = .{ .k = @intCast(k), .su = &.{}, .suh = &.{}, .ldl = bldl, .parts = &.{} };
    const partp: ldlq.Part = .{ .n = @intCast(n), .n0 = part.n0, .sv = sv, .s = s, .svh = svh };
    var dr = try ldlq.Driver.init(&r.d, w, @intCast(k), @intCast(n), 16);
    try dr.beginSlot(&r.d, &slot);
    const tr = try gpa.alloc(i16, (k / 16) * nt * words);
    defer gpa.free(tr);
    const stat = try gpa.alloc(ldlq.TileStat, (k / 16) * nt);
    defer gpa.free(stat);
    const rep = try dr.quantizePart(&r.d, &slot, &partp, wq64, tr, stat);

    try std.testing.expectEqualSlices(i16, tr_base, tr);
    for (stat) |st| {
        if (st.status == 2) return error.TileFailed;
    }
    std.debug.print("T-LDLQ-2 real ok: layer {d} part {d} K={d} N={d} halves {d}: {d} tiles, {d} B trellis identical to the calibration-free encoder (clamped {d})\n", .{
        layer_i, part_i, k, n, w.halves, rep.tiles, tr.len * 2, rep.clamped,
    });

    // Measurement 7.4/7.5 at real size: the block loop reps times with and without the
    // rank-16 feedback, no readbacks (benchPart queues the whole pass; card must be
    // quiet — run with LDLQ_BENCH=<reps>). The update share is the 7.5 number.
    if (std.c.getenv("LDLQ_BENCH")) |v| {
        const reps: u32 = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 2;
        const full_ns = try dr.benchPart(&r.d, wq64, @intCast(n), reps, false);
        const skip_ns = try dr.benchPart(&r.d, wq64, @intCast(n), reps, true);
        const tiles: f64 = @floatFromInt((k / 16) * nt * @as(usize, reps));
        const full_s: f64 = @as(f64, @floatFromInt(full_ns)) / 1e9;
        const skip_s: f64 = @as(f64, @floatFromInt(skip_ns)) / 1e9;
        std.debug.print("strip-mode real: {d:.1} tiles/s full, {d:.1} tiles/s without ldlq_update, update share {d:.2}% ({d} reps, K={d} N={d})\n", .{
            tiles / full_s, tiles / skip_s, 100.0 * (full_s - skip_s) / full_s, reps, k, n,
        });
    }
}

fn readDev(dd: *q.Dev, src: *anyopaque, out: []u8) !void {
    const host = try dd.g.hostAlloc(out.len);
    defer dd.g.free(host);
    try devm.copy(dd, host, src, out.len);
    try dd.sync();
    @memcpy(out, @as([*]const u8, @ptrCast(host))[0..out.len]);
}

/// Measurements 7.4 (tiles/s through the full block loop) and 7.5 (the ldlq_update
/// share) at real size, without the correctness runs so the quiet-card window is
/// spent on the loop itself. Reports card bytes free at entry, after the pack load,
/// after the timed passes, and after teardown (the release proof). The card must be
/// idle: contended numbers get discarded, not reported. Usage:
///   --bench PACK_DIR [LAYER] [PART] [REPS] [COUNT]
fn benchReal(args: []const [:0]const u8) !void {
    const calib = @import("calib.zig");
    if (std.c.getenv("XPU_CHUNK") == null) _ = setenv("XPU_CHUNK", "2048", 0);
    if (std.c.getenv("XPU_CTX") == null) _ = setenv("XPU_CTX", "4096", 0);
    const layer_i = if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else 0;
    const part_i = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 0;
    const reps: u32 = if (args.len > 3) try std.fmt.parseInt(u32, args[3], 10) else 1;
    const count: u32 = if (args.len > 4) try std.fmt.parseInt(u32, args[4], 10) else 16;
    var g0 = try q.ze.Gpu.open(0xe223);
    const mem0 = try g0.memory();
    std.debug.print("bench: card bytes free {d} of {d} (entry)\n", .{ mem0.free, mem0.size });
    const r = try q.Runner.init(args[0], 4096);
    const mem1 = try g0.memory();
    std.debug.print("bench: card bytes free {d} of {d} (pack loaded)\n", .{ mem1.free, mem1.size });
    const p = r.model.layers[layer_i].mixer_in;
    const hx = p.exl3 orelse return error.NoTrellis;
    const part = hx.parts[part_i];
    if (part.dense) return error.DensePart;
    const w: core.exl3_format.Width = .{ .halves = part.halves };
    const k: usize = p.k;
    const n: usize = part.n;
    const nt = n / 16;
    std.debug.print("bench: layer {d} part {d} K={d} N={d} halves {d} ({d} blocks x {d} tiles), reps {d} count {d}\n", .{
        layer_i, part_i, k, n, w.halves, k / 16, nt, reps, count,
    });

    const slab = try r.d.alloc_(@as(usize, k) * hx.nt * 2);
    defer r.d.g.free(slab);
    try calib.dequantSlot(&r.d, hx, p.k, hx.nt, slab);
    const slab_h = try gpa.alloc(f16, k * hx.nt);
    defer gpa.free(slab_h);
    try readDev(&r.d, slab, std.mem.sliceAsBytes(slab_h));
    const wq64 = try gpa.alloc(f64, k * n);
    defer gpa.free(wq64);
    for (0..k) |row| {
        for (0..n) |c| wq64[row * n + c] = @floatCast(slab_h[row * hx.nt + part.n0 + c]);
    }

    // identity L: the bench exercises the full launch pattern (5 kernels per block,
    // nt/count encode launches at the given count); the L values ride the same bytes
    const bldl = try gpa.create(core.exl3_ldl.BlockLdl);
    defer gpa.destroy(bldl);
    bldl.* = .{ .k = k, .l = try gpa.alloc(f32, k * k), .d = try gpa.alloc(f64, (k / 16) * 256), .lambda = 0, .mu = 0 };
    defer gpa.free(bldl.l);
    defer gpa.free(bldl.d);
    @memset(bldl.l, 0);
    for (0..k) |i| bldl.l[i * k + i] = 1;
    @memset(bldl.d, 1);
    const slot: ldlq.Slot = .{ .k = @intCast(k), .su = &.{}, .suh = &.{}, .ldl = bldl, .parts = &.{} };
    var dr = try ldlq.Driver.init(&r.d, w, @intCast(k), @intCast(n), count);
    try dr.beginSlot(&r.d, &slot);

    const full_ns = try dr.benchPart(&r.d, wq64, @intCast(n), reps, false);
    const skip_ns = try dr.benchPart(&r.d, wq64, @intCast(n), reps, true);
    const tiles: f64 = @floatFromInt((k / 16) * nt * @as(usize, reps));
    const full_s: f64 = @as(f64, @floatFromInt(full_ns)) / 1e9;
    const skip_s: f64 = @as(f64, @floatFromInt(skip_ns)) / 1e9;
    const mem2 = try g0.memory();
    std.debug.print("bench: card bytes free {d} of {d} (timed passes done)\n", .{ mem2.free, mem2.size });
    std.debug.print("strip-mode real: {d:.1} tiles/s full, {d:.1} tiles/s without ldlq_update, update share {d:.2}% ({d} reps, K={d} N={d})\n", .{
        tiles / full_s, tiles / skip_s, 100.0 * (full_s - skip_s) / full_s, reps, k, n,
    });
    r.deinit();
    const mem3 = try g0.memory();
    std.debug.print("bench: card bytes free {d} of {d} (after teardown)\n", .{ mem3.free, mem3.size });
    g0.close();
}
