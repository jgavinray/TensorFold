//! Device gate for the step-5 odd-halves decode (eaq-halfstep-spec-fable.md section 4,
//! "Device parity"), built on the tf-xpu-qwen27-*-test pattern (skips, exit 0, without a
//! Level Zero device). The arbiters are the core/exl3_halfstep.zig transcriptions (the
//! T-HS-0..3 host gate already pins them against the exl3_format oracle) and the oracle
//! itself (dequantize for T-HS-4, unpack for T-HS-7); the launches mirror forward_exl3.zig
//! argument-for-argument, kernel-for-kernel.
//!
//!   T-HS-4  exl3_dequant128_c at halves 3, 5, 7, 9 against host dequantize with
//!           suh = svh = 1 on a [128, 256] part (the gate-2 construction): the odd widths
//!           must sit inside the even envelope measured in the same run (gate 2 saw 4 to
//!           5e-6 at these shapes with small scales; the in-run even max is the bar).
//!   T-HS-5  exl3_decode2_t1_h (m = 1) and exl3_decode2_h (m = 8, then rows 8..16 via
//!           the second launch) at halves 3, 5, 7, 9 with exl3_xform_h and exl3_yscale_h
//!           around them — the forward's exact sequence — against the host transcription
//!           pipeline (xform replica, hs.gemvTranscribe/gemvT1Transcribe, yscale replica):
//!           inside the envelope measured at halves 4, 6, 8 in the same run. The device
//!           Xt must also equal the host xform replica bit for bit (the extraction check).
//!   T-HS-6  the even-halves golden: --golden PATH captures the fixed-input Y and dequant
//!           slab at halves 2, 4, 6, 8, 16 (run it in the HEAD worktree with --bits-arg,
//!           where the kernels still take `int bits`); --golden-check PATH replays on the
//!           step-5 kernels and demands byte-identical records. A launch site still
//!           passing bits shows up here as a mismatch.
//!   T-HS-7  (step-1 T9 at the odd widths) the device encoder's trellis at halves 3, 5,
//!           7, 9 decoded by the device exl3_dequant128_c equals the host unpack through
//!           the exact rotations, bit for bit: producer and consumer agree on the odd ring.
//!
//! --time logs (does not assert) the per-launch ms of the three kernels at the slot
//! shapes (spec section 5's before/after row): --reps N the repeat count, --body skips
//! the head, --all-widths covers halves 2..9, --bits-arg passes bits (the HEAD kernels).
const std = @import("std");
const q = @import("qwen27.zig");
const core = @import("core");
const kb = @import("xpu_qwen27_kernels");
const devm = @import("dev.zig");
const enc = @import("exl3_encode.zig");

const hs = core.exl3_halfstep;
const exl3_format = core.exl3_format;
const gpa = std.heap.c_allocator;

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;

const odd_halves = [_]u32{ 3, 5, 7, 9 };
const even_halves = [_]u32{ 2, 4, 6, 8, 16 }; // T-HS-4's envelope and T-HS-6's golden widths
const en5 = [_]u32{ 4, 6, 8 }; // T-HS-5's envelope widths (spec: measured at 4, 6, 8)

/// The five exl3_h.cl kernels of the forward, handles taken once per run.
const Kernels = struct {
    decode2: q.ze.H,
    t1: q.ze.H,
    dequant: q.ze.H,
    xform: q.ze.H,
    yscale: q.ze.H,
};

fn openKernels(d: *q.Dev) !Kernels {
    const mxh = try d.g.module(&kb.exl3_h);
    return .{
        .decode2 = try d.kernel(mxh, "exl3_decode2_h", 16),
        .t1 = try d.kernel(mxh, "exl3_decode2_t1_h", 16),
        .dequant = try d.kernel(mxh, "exl3_dequant128_c", 256),
        .xform = try d.kernel(mxh, "exl3_xform_h", 16),
        .yscale = try d.kernel(mxh, "exl3_yscale_h", 16),
    };
}

fn devOpen() !q.Dev {
    const g = q.ze.Gpu.open(0xe223) catch return error.NoDevice;
    var d: q.Dev = .{ .g = g, .list = undefined, .k = undefined };
    errdefer d.g.close();
    d.list = try d.g.immediate(0, 0);
    return d;
}

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len > 1 and std.mem.eql(u8, argv[1], "--time")) {
        return timing(argv, std.mem.eql(u8, argv[argv.len - 1], "--bits-arg"));
    }
    const bits_arg = argv.len > 1 and std.mem.eql(u8, argv[argv.len - 1], "--bits-arg");
    if (argv.len > 2 and std.mem.eql(u8, argv[1], "--golden")) return goldenMode(argv[2], false, bits_arg);
    if (argv.len > 2 and std.mem.eql(u8, argv[1], "--golden-check")) return goldenMode(argv[2], true, bits_arg);
    var d = devOpen() catch {
        std.debug.print("skipped: no Level Zero device\n", .{});
        return;
    };
    errdefer d.deinit();
    const ks = try openKernels(&d);
    var fails: u32 = 0;
    fails += try tHS4(&d, ks.dequant, false);
    fails += try tHS5(&d, ks, false);
    fails += try tHS7(&d, ks.dequant);
    d.deinit();
    if (fails != 0) return error.HalfStepDeviceGate;
    std.debug.print("ok\n", .{});
}

// ---------------------------------------------------------------------------
// helpers

/// A bit-exact mirror of exl3_h.cl's rot128_lane over the contiguous 128-vector — the
/// twin of qwen27_main.zig's rotExact128 (the step-1 T9 replica): the in-lane butterflies
/// m=1,2,4 are masks 1,2,4 on the element index, the cross-lane shuffle_xor m=1,2,4,8 are
/// masks 8,16,32,64 in the same order, and both sides pair (a, b) as (a+b, a-b).
fn rotExact128(v: *[128]f32) void {
    for ([_]usize{ 1, 2, 4, 8, 16, 32, 64 }) |m| {
        var i: usize = 0;
        while (i < 128) : (i += 2 * m) {
            for (0..m) |j| {
                const a = v[i + j];
                const b = v[i + j + m];
                v[i + j] = a + b;
                v[i + j + m] = a - b;
            }
        }
    }
}

const inv_f32: f32 = @floatCast(0.08838834764831845); // the kernel's 0.08838834764831845f

/// exl3_xform_h's exact fp16 output: per row, per 128-block, f32(x*suh) through the exact
/// butterfly order, one rounding of v*inv into f16 (the kernel's store).
fn xformReplica(x: []const f16, rows: u32, k: u32, suh: []const f16, xt: []f16) void {
    std.debug.assert(x.len == @as(usize, rows) * k and xt.len == x.len and suh.len == k);
    var v: [128]f32 = undefined;
    for (0..rows) |r| {
        var kb0: u32 = 0;
        while (kb0 < k) : (kb0 += 128) {
            for (0..128) |j| v[j] = @as(f32, @floatCast(x[r * k + kb0 + j])) * @as(f32, @floatCast(suh[kb0 + j]));
            rotExact128(&v);
            for (0..128) |j| xt[r * k + kb0 + j] = @floatCast(v[j] * inv_f32);
        }
    }
}

/// exl3_yscale_h's exact fp16 output over the GEMV's y: per row, per 128-block of the
/// part's columns, the f32 butterfly, one rounding of (v*inv)*svh[n] — the kernel's store
/// is `v[j] * 0.08838834764831845f * (float)SVH[n]`, left-associative.
fn yscaleReplica(y: []const f16, rows: u32, nt: u32, n: u32, svh: []const f16, yfin: []f16) void {
    std.debug.assert(y.len == @as(usize, rows) * nt and yfin.len == y.len and svh.len >= n);
    var v: [128]f32 = undefined;
    for (0..rows) |r| {
        var nb0: u32 = 0;
        while (nb0 < n) : (nb0 += 128) {
            for (0..128) |j| v[j] = @as(f32, @floatCast(y[r * nt + nb0 + j]));
            rotExact128(&v);
            for (0..128) |j| {
                const c = nb0 + j;
                yfin[r * nt + c] = @floatCast((v[j] * inv_f32) * @as(f32, @floatCast(svh[c])));
            }
        }
    }
}

fn readDev(dd: *q.Dev, src: *anyopaque, out: []u8) !void {
    const host = try dd.g.hostAlloc(out.len);
    defer dd.g.free(host);
    try devm.copy(dd, host, src, out.len);
    try dd.sync();
    @memcpy(out, @as([*]const u8, @ptrCast(host))[0..out.len]);
}

fn widthArg(bits_arg: bool, halves: u32) i32 {
    return @intCast(if (bits_arg) halves / 2 else halves);
}

// ---------------------------------------------------------------------------
// the fixed seeded parts T-HS-4/5/6 build on (seeded, so capture and check agree)

/// The T-HS-4 [128, 256] part: closed-sequence tiles, suh = svh = 1 (the spec's scales).
const Deq4 = struct {
    tr: []i16,
    suh: []f16,
    svh: []f16,

    fn init(seed: u64, halves: u32) !Deq4 {
        var self: Deq4 = undefined;
        const words = @as(usize, halves) * 8;
        self.tr = try gpa.alloc(i16, @as(usize, K4) / 16 * (N4 / 16) * words);
        errdefer gpa.free(self.tr);
        self.suh = try gpa.alloc(f16, K4);
        errdefer gpa.free(self.suh);
        self.svh = try gpa.alloc(f16, N4);
        errdefer gpa.free(self.svh);
        var prng = std.Random.DefaultPrng.init(seed);
        hs.fillClosedTiles(prng.random(), halves, @as(usize, K4 / 16) * (N4 / 16), self.tr);
        @memset(self.suh, @as(f16, @floatCast(1.0))); // the spec's suh = svh = 1
        @memset(self.svh, @as(f16, @floatCast(1.0)));
        return self;
    }

    /// golden variant: seeded random scales in (0.5, 1.5) so the slab pins phases B and C
    fn initScaled(seed: u64, halves: u32) !Deq4 {
        const self = try Deq4.init(seed, halves);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x5CA1);
        for (self.suh) |*s| s.* = @as(f16, @floatCast(0.5 + prng.random().float(f32)));
        for (self.svh) |*s| s.* = @as(f16, @floatCast(0.5 + prng.random().float(f32)));
        return self;
    }

    fn deinit(self: *Deq4) void {
        gpa.free(self.tr);
        gpa.free(self.suh);
        gpa.free(self.svh);
    }
};

/// The T-HS-5 [256, 256] part with its [16][K] activation rows and per-part scale vectors.
const Gemv5 = struct {
    tr: []i16,
    x: []f16, // [16][K]
    suh: []f16,
    svh: []f16,

    fn init(seed: u64, halves: u32) !Gemv5 {
        var self: Gemv5 = undefined;
        const words = @as(usize, halves) * 8;
        self.tr = try gpa.alloc(i16, @as(usize, K5 / 16) * (N5 / 16) * words);
        errdefer gpa.free(self.tr);
        self.x = try gpa.alloc(f16, 16 * K5);
        errdefer gpa.free(self.x);
        self.suh = try gpa.alloc(f16, K5);
        errdefer gpa.free(self.suh);
        self.svh = try gpa.alloc(f16, N5);
        errdefer gpa.free(self.svh);
        var prng = std.Random.DefaultPrng.init(seed);
        hs.fillClosedTiles(prng.random(), halves, @as(usize, K5 / 16) * (N5 / 16), self.tr);
        hs.fillRandomX(prng.random(), self.x);
        for (self.suh) |*s| s.* = @as(f16, @floatCast(0.5 + prng.random().float(f32)));
        for (self.svh) |*s| s.* = @as(f16, @floatCast(0.5 + prng.random().float(f32)));
        return self;
    }

    fn deinit(self: *Gemv5) void {
        gpa.free(self.tr);
        gpa.free(self.x);
        gpa.free(self.suh);
        gpa.free(self.svh);
    }
};

// ---------------------------------------------------------------------------
// T-HS-4: the device dequant against the host oracle, [128, 256] with unit scales.

const K4: u32 = 128;
const N4: u32 = 256;

/// run + compare one (halves, part): the device slab against core.exl3_format.dequantize
/// on the same trellis words.
fn deqRun(d: *q.Dev, k_dq: q.ze.H, p: *Deq4, halves: u32, bits_arg: bool) !struct { worst: f32, rel: f32, rms: f32 } {
    const tr_dev = try d.upload(std.mem.sliceAsBytes(p.tr));
    defer d.g.free(tr_dev);
    const suh_dev = try d.upload(std.mem.sliceAsBytes(p.suh));
    defer d.g.free(suh_dev);
    const svh_dev = try d.upload(std.mem.sliceAsBytes(p.svh));
    defer d.g.free(svh_dev);
    const slab = try d.alloc_(@as(usize, K4) * N4 * 2);
    defer d.g.free(slab);
    try d.launch(k_dq, .{ .x = K4 / 128, .y = N4 / 128 }, .{
        tr_dev, suh_dev, svh_dev, slab,
        @as(i32, @intCast(K4)), @as(i32, @intCast(N4)), @as(i32, @intCast(N4)),
        widthArg(bits_arg, halves), @as(i32, 0),
    });
    try d.sync();
    const got = try gpa.alloc(u16, @as(usize, K4) * N4);
    defer gpa.free(got);
    try readDev(d, slab, std.mem.sliceAsBytes(got));

    const want = try gpa.alloc(f64, @as(usize, K4) * N4);
    defer gpa.free(want);
    const scratch = try gpa.alloc(f64, 2 * @as(usize, K4) * N4);
    defer gpa.free(scratch);
    exl3_format.dequantize(.{ .halves = @intCast(halves) }, .mul1, p.tr, K4 / 16, N4 / 16, p.suh, p.svh, want, scratch);

    var worst: f32 = 0;
    var rel: f32 = 0;
    var sumsq: f64 = 0;
    for (0..@as(usize, K4) * N4) |at| {
        const ref: f32 = @floatCast(want[at]);
        const dv = @abs(ref - @as(f32, @as(f16, @bitCast(got[at]))));
        if (dv > worst) worst = dv;
        const r = dv / @max(@abs(ref), 1e-6);
        if (r > rel) rel = r;
        sumsq += want[at] * want[at];
    }
    return .{ .worst = worst, .rel = rel, .rms = @floatCast(@sqrt(sumsq / @as(f64, @floatFromInt(K4 * N4)))) };
}

fn tHS4(d: *q.Dev, k_dq: q.ze.H, bits_arg: bool) !u32 {
    var env: f32 = 0;
    for (even_halves) |hh| {
        var part = try Deq4.init(0x4548_0000 + hh, hh);
        defer part.deinit();
        const m = try deqRun(d, k_dq, &part, hh, bits_arg);
        std.debug.print("T-HS-4 halves={d} (even) [128,256] suh=svh=1: dequant-vs-oracle max {d:.6} rel {d:.4} rms {d:.4}\n", .{ hh, m.worst, m.rel, m.rms });
        env = @max(env, m.worst);
    }
    var fails: u32 = 0;
    for (odd_halves) |hh| {
        var part = try Deq4.init(0x4548_0000 + hh, hh);
        defer part.deinit();
        const m = try deqRun(d, k_dq, &part, hh, bits_arg);
        const pass = m.worst <= env and m.worst <= 0.02 * m.rms;
        std.debug.print("T-HS-4 halves={d} [128,256] suh=svh=1: dequant-vs-oracle max {d:.6} rel {d:.4} rms {d:.4} (even envelope {d:.6}): {s}\n", .{ hh, m.worst, m.rel, m.rms, env, if (pass) "PASS" else "FAIL" });
        if (!pass) fails += 1;
    }
    return fails;
}

// ---------------------------------------------------------------------------
// T-HS-5: the forward's decode sequence against the transcription pipeline.

const K5: u32 = 256;
const N5: u32 = 256;

/// one (halves, m) cell: launch the forward's sequence, mirror it host-side, and return
/// (max ulp device-vs-mirror, xt bit-identity count, xt words).
fn gemvCell(d: *q.Dev, ks: Kernels, g: *Gemv5, halves: u32, m: u32, bits_arg: bool) !struct { ulp: u32, xt_ok: u32, xt_all: u32 } {
    const tr_dev = try d.upload(std.mem.sliceAsBytes(g.tr));
    defer d.g.free(tr_dev);
    const suh_dev = try d.upload(std.mem.sliceAsBytes(g.suh));
    defer d.g.free(suh_dev);
    const svh_dev = try d.upload(std.mem.sliceAsBytes(g.svh));
    defer d.g.free(svh_dev);
    const x_dev = try d.upload(std.mem.sliceAsBytes(g.x[0 .. @as(usize, m) * K5]));
    defer d.g.free(x_dev);
    const xt_dev = try d.alloc_(@as(usize, m) * K5 * 2);
    defer d.g.free(xt_dev);
    const y_dev = try d.alloc_(@as(usize, m) * N5 * 2);
    defer d.g.free(y_dev);
    const mi: i32 = @intCast(m);
    const w = widthArg(bits_arg, halves);

    // the forward's exact sequence (forward_exl3.zig gemvs): xform, the GEMV (t1 at m = 1,
    // the eight-row kernel, twice at m = 16), yscale. kslices = 1, NT = N (one part).
    try d.launch(ks.xform, .{ .x = K5 / 128, .y = m }, .{ x_dev, suh_dev, xt_dev, @as(i32, @intCast(K5)), mi });
    if (m == 1) {
        try d.launch(ks.t1, .{ .x = N5 / 16, .y = 1 }, .{
            tr_dev, xt_dev, y_dev, @as(i32, @intCast(K5)), @as(i32, @intCast(N5)),
            @as(i32, 1), w, @as(i32, 1), @as(i32, @intCast(N5)),
        });
    } else {
        try d.launch(ks.decode2, .{ .x = N5 / 16, .y = 1 }, .{
            tr_dev, xt_dev, y_dev, @as(i32, @intCast(K5)), @as(i32, @intCast(N5)),
            @as(i32, @intCast(@min(m, 8))), w, @as(i32, 1), @as(i32, @intCast(N5)),
        });
        if (m > 8) {
            const x2 = devm.off(xt_dev, 8 * @as(usize, K5) * 2);
            const y2 = devm.off(y_dev, 8 * @as(usize, N5) * 2);
            try d.launch(ks.decode2, .{ .x = N5 / 16, .y = 1 }, .{
                tr_dev, x2, y2, @as(i32, @intCast(K5)), @as(i32, @intCast(N5)),
                @as(i32, @intCast(m - 8)), w, @as(i32, 1), @as(i32, @intCast(N5)),
            });
        }
    }
    try d.launch(ks.yscale, .{ .x = N5 / 128, .y = m }, .{ y_dev, svh_dev, @as(i32, @intCast(N5)), @as(i32, @intCast(N5)), mi });
    try d.sync();

    // host mirror: xform replica -> GEMV transcription -> yscale replica.
    const xt_host = try gpa.alloc(f16, @as(usize, m) * K5);
    defer gpa.free(xt_host);
    xformReplica(g.x[0 .. @as(usize, m) * K5], m, K5, g.suh, xt_host);
    const xt_dev_host = try gpa.alloc(u16, @as(usize, m) * K5);
    defer gpa.free(xt_dev_host);
    try readDev(d, xt_dev, std.mem.sliceAsBytes(xt_dev_host));
    var xt_ok: u32 = 0;
    for (xt_host, xt_dev_host) |h, dv| {
        if (@as(u16, @bitCast(h)) == dv) xt_ok += 1;
    }

    const y_tr = try gpa.alloc(f16, @as(usize, m) * N5);
    defer gpa.free(y_tr);
    if (m == 1) {
        hs.gemvT1Transcribe(g.tr, halves, K5, N5, N5, xt_host, y_tr);
    } else {
        hs.gemvTranscribe(g.tr, halves, K5, N5, N5, xt_host, m, y_tr, null);
    }
    const y_fin = try gpa.alloc(f16, @as(usize, m) * N5);
    defer gpa.free(y_fin);
    yscaleReplica(y_tr, m, N5, N5, g.svh, y_fin);

    const y_dev_host = try gpa.alloc(u16, @as(usize, m) * N5);
    defer gpa.free(y_dev_host);
    try readDev(d, y_dev, std.mem.sliceAsBytes(y_dev_host));
    var ulp: u32 = 0;
    for (y_fin, y_dev_host) |h, dv| ulp = @max(ulp, hs.f16UlpDist(h, @as(f16, @bitCast(dv))));
    return .{ .ulp = ulp, .xt_ok = xt_ok, .xt_all = @intCast(xt_host.len) };
}

fn tHS5(d: *q.Dev, ks: Kernels, bits_arg: bool) !u32 {
    var env: u32 = 0;
    for (en5) |hh| {
        var g = try Gemv5.init(0x4855_0000 + hh, hh);
        defer g.deinit();
        for ([_]u32{ 1, 8, 16 }) |m| {
            const r = try gemvCell(d, ks, &g, hh, m, bits_arg);
            env = @max(env, r.ulp);
            std.debug.print("T-HS-5 halves={d} (even envelope) m={d}: Y-vs-transcription max {d} ulp, xt bit-identical {d}/{d}\n", .{ hh, m, r.ulp, r.xt_ok, r.xt_all });
        }
    }
    var fails: u32 = 0;
    for (odd_halves) |hh| {
        var g = try Gemv5.init(0x4855_0000 + hh, hh);
        defer g.deinit();
        for ([_]u32{ 1, 8, 16 }) |m| {
            const r = try gemvCell(d, ks, &g, hh, m, bits_arg);
            const pass = r.ulp <= env and r.xt_ok == r.xt_all;
            std.debug.print("T-HS-5 halves={d} m={d} ({s}): Y-vs-transcription max {d} ulp (envelope {d}), xt bit-identical {d}/{d}: {s}\n", .{
                hh, m, if (m == 1) "t1" else if (m == 8) "decode2 T=8" else "decode2 2x T=8",
                r.ulp, env, r.xt_ok, r.xt_all, if (pass) "PASS" else "FAIL",
            });
            if (!pass) fails += 1;
        }
    }
    return fails;
}

// ---------------------------------------------------------------------------
// T-HS-7: the device encoder's odd-halves output through the device dequant vs the host
// unpack (step-1 T9's construction at halves 3, 5, 7, 9): bit-identical through an exact
// replica of the kernel's rotations with suh = svh = 1.

const K7: u32 = 128;

fn gaussPair(r: std.Random) [2]f64 {
    const za = @max(r.float(f64), 1e-300);
    const zb = r.float(f64);
    const rr = @sqrt(-2.0 * @log(za));
    return .{ rr * @cos(2 * std.math.pi * zb), rr * @sin(2 * std.math.pi * zb) };
}

fn tHS7(d: *q.Dev, k_dq: q.ze.H) !u32 {
    var fails: u32 = 0;
    for (odd_halves) |hh| {
        const w: exl3_format.Width = .{ .halves = @intCast(hh) };
        const words = w.tileWords();
        const hw = try gpa.alloc(f16, @as(usize, K7) * K7);
        defer gpa.free(hw);
        var prng = std.Random.DefaultPrng.init(0xA917 + hh);
        for (hw) |*t| {
            const g = gaussPair(prng.random());
            t.* = @floatCast(g[0]);
        }
        const wq = try d.upload(std.mem.sliceAsBytes(hw));
        defer d.g.free(wq);
        var e = try enc.Encoder.init(d, w, 64, 64);
        const tr_dev = try d.alloc_(64 * words * 2);
        defer d.g.free(tr_dev);
        try e.encode(d, wq, @intCast(K7), @intCast(K7), tr_dev, 0, 64, null);
        try d.sync();
        const tr_host = try gpa.alloc(i16, 64 * words);
        defer gpa.free(tr_host);
        try e.readTrellis(d, tr_dev, tr_host, null);

        // device: dequant128_c on the encoder's own trellis buffer, suh = svh = 1
        const ones = try gpa.alloc(f16, K7);
        defer gpa.free(ones);
        @memset(ones, @as(f16, @floatCast(1.0)));
        const suh_dev = try d.upload(std.mem.sliceAsBytes(ones));
        defer d.g.free(suh_dev);
        const svh_dev = try d.upload(std.mem.sliceAsBytes(ones));
        defer d.g.free(svh_dev);
        const wout = try d.alloc_(@as(usize, K7) * K7 * 2);
        defer d.g.free(wout);
        try d.launch(k_dq, .{ .x = 1, .y = 1 }, .{
            tr_dev, suh_dev, svh_dev, wout,
            @as(i32, @intCast(K7)), @as(i32, @intCast(K7)), @as(i32, @intCast(K7)),
            @as(i32, @intCast(hh)), @as(i32, 0),
        });
        try d.sync();
        const wdev = try gpa.alloc(u16, @as(usize, K7) * K7);
        defer gpa.free(wdev);
        try readDev(d, wout, std.mem.sliceAsBytes(wdev));

        // host: the oracle's unpack of the same words through the exact rotations
        const wq_dec = try gpa.alloc(f16, @as(usize, K7) * K7);
        defer gpa.free(wq_dec);
        exl3_format.unpack(w, .mul1, tr_host, K7 / 16, K7 / 16, wq_dec);
        const blkB = try gpa.alloc(f16, @as(usize, K7) * K7);
        defer gpa.free(blkB);
        var v: [128]f32 = undefined;
        for (0..K7) |row| {
            for (0..128) |n| v[n] = @floatCast(wq_dec[row * 128 + n]);
            rotExact128(&v);
            for (0..128) |n| blkB[row * 128 + n] = @floatCast(v[n] * inv_f32);
        }
        const out9 = try gpa.alloc(f16, @as(usize, K7) * K7);
        defer gpa.free(out9);
        for (0..K7) |n| {
            for (0..128) |row| v[row] = @floatCast(blkB[row * 128 + n]);
            rotExact128(&v);
            for (0..128) |row| out9[row * 128 + n] = @floatCast(v[row] * inv_f32);
        }
        var ok: u32 = 0;
        var first: ?usize = null;
        for (out9, wdev) |h, dv| {
            if (@as(u16, @bitCast(h)) == dv) ok += 1 else if (first == null) first = @intCast(ok);
        }
        const pass = ok == K7 * K7;
        std.debug.print("T-HS-7 halves={d} [128,128]: exl3_dequant128_c on the encoder trellis {d}/{d} words bit-identical to the host unpack through the exact rotations: {s}\n", .{ hh, ok, K7 * K7, if (pass) "PASS" else "FAIL" });
        if (first) |at| std.debug.print("  first diff at {d}: device {x:0>4} host {x:0>4}\n", .{ at, wdev[at], @as(u16, @bitCast(out9[at])) });
        if (!pass) fails += 1;
    }
    return fails;
}

// ---------------------------------------------------------------------------
// T-HS-6: the even-halves golden capture / check. Fixed seeded inputs at halves
// 2, 4, 6, 8, 16: the dequant slab ([128, 256], seeded random scales so phases B and C
// are pinned too) and the decode-path Y (m = 1 through t1, m = 8 and 16 through the
// eight-row kernel, second launch for rows 8..16) with the m = 1 Xt. Capture runs on
// the HEAD kernels with --bits-arg (their width argument is `int bits` = halves / 2);
// --golden-check replays on the step-5 kernels and compares every record byte for byte.

const golden_magic = "TFHSGOLD";
const golden_version: u32 = 1;

const Writer = struct {
    list: std.ArrayList(u8),

    fn put(w: *Writer, bytes: []const u8) !void {
        try w.list.appendSlice(gpa, bytes);
    }
    fn putInt(w: *Writer, comptime T: type, v: T) !void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, v, .little);
        try w.put(&b);
    }
    fn dump(w: *Writer, tag: u32, bytes: []const u8) !void {
        try w.putInt(u32, tag);
        try w.putInt(u64, bytes.len);
        try w.put(bytes);
    }
};

const Reader = struct {
    bytes: []const u8,
    at: usize = 0,
    fn take(r: *Reader, n: usize) ![]const u8 {
        if (r.at + n > r.bytes.len) return error.GoldenTruncated;
        defer r.at += n;
        return r.bytes[r.at .. r.at + n];
    }
    fn takeInt(r: *Reader, comptime T: type) !T {
        const b = try r.take(@sizeOf(T));
        return std.mem.readInt(T, b[0..@sizeOf(T)], .little);
    }
    fn expect(r: *Reader, tag: u32, want: []const u8, name: []const u8, fails: *u32) !void {
        const t = try r.takeInt(u32);
        const n = try r.takeInt(u64);
        if (t != tag or n != want.len) {
            std.debug.print("T-HS-6 {s}: record mismatch (golden tag {d} len {d}, run tag {d} len {d})\n", .{ name, t, n, tag, want.len });
            fails.* += 1;
            if (n <= r.bytes.len - r.at) _ = try r.take(@intCast(n));
            return;
        }
        const got = try r.take(want.len);
        if (!std.mem.eql(u8, got, want)) {
            var i: usize = 0;
            while (i < got.len and got[i] == want[i]) i += 1;
            std.debug.print("T-HS-6 {s}: NOT byte-identical, first diff at byte {d} (golden {x:0>2} vs run {x:0>2})\n", .{ name, i, got[i], want[i] });
            fails.* += 1;
        }
    }
};

/// tag = kind + halves in the high byte, so a missing width cannot hide behind a tag reuse.
fn gtag(kind: u32, halves: u32) u32 {
    return kind | (halves << 8);
}

fn goldenMode(path: [:0]const u8, check: bool, bits_arg: bool) !void {
    var d = devOpen() catch {
        std.debug.print("skipped: no Level Zero device\n", .{});
        return;
    };
    errdefer d.deinit();
    const ks = try openKernels(&d);

    var w: Writer = .{ .list = .empty };
    var rdr: Reader = .{ .bytes = &.{} };
    if (check) {
        rdr = try readFile(path);
        const magic = try rdr.take(8);
        if (!std.mem.eql(u8, magic, golden_magic)) return error.GoldenBadMagic;
        if (try rdr.takeInt(u32) != golden_version) return error.GoldenBadVersion;
        const captured_bits = try rdr.takeInt(u32);
        std.debug.print("T-HS-6: golden captured with kernels taking {s} (this run passes {s})\n", .{
            if (captured_bits != 0) "int bits (HEAD)" else "int halves (step-5)",
            if (bits_arg) "bits" else "halves",
        });
    } else {
        try w.put(golden_magic);
        try w.putInt(u32, golden_version);
        try w.putInt(u32, @intFromBool(bits_arg)); // 1 = the HEAD kernels took bits
    }

    var fails: u32 = 0;
    for (even_halves) |hh| {
        var deq = try Deq4.initScaled(0x4744_0000 + hh, hh);
        defer deq.deinit();
        var gemv = try Gemv5.init(0x4747_0000 + hh, hh);
        defer gemv.deinit();

        const slab = try d.alloc_(@as(usize, K4) * N4 * 2);
        defer d.g.free(slab);
        {
            const tr_dev = try d.upload(std.mem.sliceAsBytes(deq.tr));
            defer d.g.free(tr_dev);
            const suh_dev = try d.upload(std.mem.sliceAsBytes(deq.suh));
            defer d.g.free(suh_dev);
            const svh_dev = try d.upload(std.mem.sliceAsBytes(deq.svh));
            defer d.g.free(svh_dev);
            try d.launch(ks.dequant, .{ .x = K4 / 128, .y = N4 / 128 }, .{
                tr_dev, suh_dev, svh_dev, slab,
                @as(i32, @intCast(K4)), @as(i32, @intCast(N4)), @as(i32, @intCast(N4)),
                widthArg(bits_arg, hh), @as(i32, 0),
            });
            try d.sync();
        }
        const slab_h = try gpa.alloc(u16, @as(usize, K4) * N4);
        defer gpa.free(slab_h);
        try readDev(&d, slab, std.mem.sliceAsBytes(slab_h));

        // the decode-path Y at m = 1 (t1), 8 and 16 (eight-row kernel, second launch for
        // rows 8..16); the m = 1 Xt rides along — its bit-identity is part of the golden.
        const tr5 = try d.upload(std.mem.sliceAsBytes(gemv.tr));
        defer d.g.free(tr5);
        const suh5 = try d.upload(std.mem.sliceAsBytes(gemv.suh));
        defer d.g.free(suh5);
        const svh5 = try d.upload(std.mem.sliceAsBytes(gemv.svh));
        defer d.g.free(svh5);
        const xt_h = try gpa.alloc(u16, K5);
        defer gpa.free(xt_h);
        const y1 = try gpa.alloc(u16, N5);
        defer gpa.free(y1);
        const y8 = try gpa.alloc(u16, 8 * N5);
        defer gpa.free(y8);
        const y16 = try gpa.alloc(u16, 16 * N5);
        defer gpa.free(y16);
        const wd = widthArg(bits_arg, hh);
        for ([_]u32{ 1, 8, 16 }, 0..) |m, i| {
            const x_dev = try d.upload(std.mem.sliceAsBytes(gemv.x[0 .. @as(usize, m) * K5]));
            defer d.g.free(x_dev);
            const xt_dev = try d.alloc_(@as(usize, m) * K5 * 2);
            defer d.g.free(xt_dev);
            const y_dev = try d.alloc_(@as(usize, m) * N5 * 2);
            defer d.g.free(y_dev);
            const mi: i32 = @intCast(m);
            try d.launch(ks.xform, .{ .x = K5 / 128, .y = m }, .{ x_dev, suh5, xt_dev, @as(i32, @intCast(K5)), mi });
            if (m == 1) {
                try d.launch(ks.t1, .{ .x = N5 / 16, .y = 1 }, .{
                    tr5, xt_dev, y_dev, @as(i32, @intCast(K5)), @as(i32, @intCast(N5)),
                    @as(i32, 1), wd, @as(i32, 1), @as(i32, @intCast(N5)),
                });
            } else {
                try d.launch(ks.decode2, .{ .x = N5 / 16, .y = 1 }, .{
                    tr5, xt_dev, y_dev, @as(i32, @intCast(K5)), @as(i32, @intCast(N5)),
                    @as(i32, @intCast(@min(m, 8))), wd, @as(i32, 1), @as(i32, @intCast(N5)),
                });
                if (m > 8) {
                    const x2 = devm.off(xt_dev, 8 * @as(usize, K5) * 2);
                    const y2 = devm.off(y_dev, 8 * @as(usize, N5) * 2);
                    try d.launch(ks.decode2, .{ .x = N5 / 16, .y = 1 }, .{
                        tr5, x2, y2, @as(i32, @intCast(K5)), @as(i32, @intCast(N5)),
                        @as(i32, @intCast(m - 8)), wd, @as(i32, 1), @as(i32, @intCast(N5)),
                    });
                }
            }
            try d.launch(ks.yscale, .{ .x = N5 / 128, .y = m }, .{ y_dev, svh5, @as(i32, @intCast(N5)), @as(i32, @intCast(N5)), mi });
            try d.sync();
            if (i == 0) {
                try readDev(&d, xt_dev, std.mem.sliceAsBytes(xt_h));
                try readDev(&d, y_dev, std.mem.sliceAsBytes(y1));
            } else if (i == 1) {
                try readDev(&d, y_dev, std.mem.sliceAsBytes(y8));
            } else {
                try readDev(&d, y_dev, std.mem.sliceAsBytes(y16));
            }
        }

        if (check) {
            try rdr.expect(gtag(1, hh), std.mem.sliceAsBytes(deq.tr), "dequant trellis", &fails);
            try rdr.expect(gtag(2, hh), std.mem.sliceAsBytes(deq.suh), "dequant suh", &fails);
            try rdr.expect(gtag(3, hh), std.mem.sliceAsBytes(deq.svh), "dequant svh", &fails);
            try rdr.expect(gtag(4, hh), std.mem.sliceAsBytes(slab_h), "dequant slab", &fails);
            try rdr.expect(gtag(5, hh), std.mem.sliceAsBytes(gemv.tr), "gemv trellis", &fails);
            try rdr.expect(gtag(6, hh), std.mem.sliceAsBytes(gemv.x), "gemv x", &fails);
            try rdr.expect(gtag(7, hh), std.mem.sliceAsBytes(gemv.suh), "gemv suh", &fails);
            try rdr.expect(gtag(8, hh), std.mem.sliceAsBytes(gemv.svh), "gemv svh", &fails);
            try rdr.expect(gtag(9, hh), std.mem.sliceAsBytes(xt_h), "gemv xt (m=1)", &fails);
            try rdr.expect(gtag(10, hh), std.mem.sliceAsBytes(y1), "Y m=1", &fails);
            try rdr.expect(gtag(11, hh), std.mem.sliceAsBytes(y8), "Y m=8", &fails);
            try rdr.expect(gtag(12, hh), std.mem.sliceAsBytes(y16), "Y m=16", &fails);
            std.debug.print("T-HS-6 halves={d}: {d} records compared against the golden\n", .{ hh, 12 });
        } else {
            try w.dump(gtag(1, hh), std.mem.sliceAsBytes(deq.tr));
            try w.dump(gtag(2, hh), std.mem.sliceAsBytes(deq.suh));
            try w.dump(gtag(3, hh), std.mem.sliceAsBytes(deq.svh));
            try w.dump(gtag(4, hh), std.mem.sliceAsBytes(slab_h));
            try w.dump(gtag(5, hh), std.mem.sliceAsBytes(gemv.tr));
            try w.dump(gtag(6, hh), std.mem.sliceAsBytes(gemv.x));
            try w.dump(gtag(7, hh), std.mem.sliceAsBytes(gemv.suh));
            try w.dump(gtag(8, hh), std.mem.sliceAsBytes(gemv.svh));
            try w.dump(gtag(9, hh), std.mem.sliceAsBytes(xt_h));
            try w.dump(gtag(10, hh), std.mem.sliceAsBytes(y1));
            try w.dump(gtag(11, hh), std.mem.sliceAsBytes(y8));
            try w.dump(gtag(12, hh), std.mem.sliceAsBytes(y16));
            std.debug.print("T-HS-6 halves={d}: captured the dequant slab and Y (m=1,8,16)\n", .{hh});
        }
    }
    if (!check) try writeFile(path, w.list.items);
    d.deinit();
    if (fails != 0) return error.GoldenMismatch;
    if (check) std.debug.print("T-HS-6: golden byte-identical at halves 2, 4, 6, 8, 16\n", .{});
}

fn readFile(path: [:0]const u8) !Reader {
    const fd = open(path, 0);
    if (fd < 0) return error.GoldenFileNotFound;
    defer _ = close(fd);
    const size: usize = @intCast(lseek(fd, 0, 2));
    const buf = try gpa.alloc(u8, size);
    var got: usize = 0;
    while (got < size) {
        const n = read(fd, buf.ptr + got, size - got);
        if (n <= 0) return error.GoldenShortRead;
        got += @intCast(n);
    }
    return .{ .bytes = buf };
}

fn writeFile(path: [:0]const u8, bytes: []const u8) !void {
    const O_WRONLY_CREAT_TRUNC: c_int = 0x4001 | 0o100 | 0o1000; // Linux O_WRONLY|O_CREAT|O_TRUNC
    const fd = open(path, O_WRONLY_CREAT_TRUNC, @as(c_uint, 0o644));
    if (fd < 0) return error.GoldenFileUnwritable;
    defer _ = close(fd);
    var sent: usize = 0;
    while (sent < bytes.len) {
        const n = write(fd, bytes.ptr + sent, bytes.len - sent);
        if (n <= 0) return error.GoldenShortWrite;
        sent += @intCast(n);
    }
}

// ---------------------------------------------------------------------------
// --time: the section-5 per-launch rows (logged, never asserted): the three fp16 kernels
// the forward launches, at the slot shapes, min over --reps after 5 warm launches. Run
// it in both trees for the before/after record.

const Shape = struct { name: []const u8, k: u32, n: u32 };
const shapes_all = [_]Shape{
    .{ .name = "gu", .k = 5120, .n = 34816 },
    .{ .name = "down", .k = 17408, .n = 5120 },
    .{ .name = "qkv", .k = 5120, .n = 14336 },
    .{ .name = "mixer_in", .k = 5120, .n = 16640 },
    .{ .name = "mixer_out", .k = 6144, .n = 5120 },
    .{ .name = "head", .k = 5120, .n = 248320 },
};

fn timing(argv: []const [:0]const u8, bits_arg: bool) !void {
    var reps: u32 = 50;
    var body_only = false;
    const widths_all = [_]u32{ 2, 3, 4, 5, 6, 7, 8, 9 };
    const widths_small = [_]u32{ 4, 6 };
    var widths: []const u32 = &widths_small;
    var it: usize = 2;
    while (it < argv.len) : (it += 1) {
        if (std.mem.eql(u8, argv[it], "--reps") and it + 1 < argv.len) {
            reps = try std.fmt.parseInt(u32, argv[it + 1], 10);
            it += 1;
        } else if (std.mem.eql(u8, argv[it], "--body")) {
            body_only = true;
        } else if (std.mem.eql(u8, argv[it], "--all-widths")) {
            widths = &widths_all;
        }
    }
    var d = devOpen() catch {
        std.debug.print("skipped: no Level Zero device\n", .{});
        return;
    };
    errdefer d.deinit();
    const ks = try openKernels(&d);
    const nshapes: usize = shapes_all.len - @intFromBool(body_only);
    for (shapes_all[0..nshapes]) |sh| {
        const trb: usize = @as(usize, sh.k / 16) * (sh.n / 16) * (8 * 16) * 2; // the widest trellis (halves 16)
        const tr_dev = try d.alloc_(trb);
        defer d.g.free(tr_dev);
        try d.fillOn(d.list, tr_dev, @as(u16, 0x5a5a), trb);
        const slab = try d.alloc_(@as(usize, sh.k) * sh.n * 2);
        defer d.g.free(slab);
        const xt = try d.alloc_(16 * @as(usize, sh.k) * 2);
        defer d.g.free(xt);
        try d.fillOn(d.list, xt, @as(u16, 0x3c00), 16 * @as(usize, sh.k) * 2);
        const y = try d.alloc_(16 * @as(usize, sh.n) * 2);
        defer d.g.free(y);
        const ones16 = try gpa.alloc(f16, @max(sh.k, sh.n));
        defer gpa.free(ones16);
        @memset(ones16, @as(f16, @floatCast(1.0)));
        const suh_dev = try d.upload(std.mem.sliceAsBytes(ones16[0..sh.k]));
        defer d.g.free(suh_dev);
        const svh_dev = try d.upload(std.mem.sliceAsBytes(ones16[0..sh.n]));
        defer d.g.free(svh_dev);
        try d.sync();
        for (widths) |hh| {
            const w = widthArg(bits_arg, hh);
            var best: f64 = 1e9;
            for (0..reps + 5) |i| {
                const t0 = q.now();
                try d.launch(ks.dequant, .{ .x = sh.k / 128, .y = sh.n / 128 }, .{
                    tr_dev, suh_dev, svh_dev, slab,
                    @as(i32, @intCast(sh.k)), @as(i32, @intCast(sh.n)), @as(i32, @intCast(sh.n)), w, @as(i32, 0),
                });
                try d.sync();
                if (i >= 5) best = @min(best, q.now() - t0);
            }
            const dq_ms = best * 1e3;
            best = 1e9;
            for (0..reps + 5) |i| {
                const t0 = q.now();
                try d.launch(ks.t1, .{ .x = sh.n / 16, .y = 1 }, .{
                    tr_dev, xt, y, @as(i32, @intCast(sh.k)), @as(i32, @intCast(sh.n)), @as(i32, 1), w, @as(i32, 1), @as(i32, @intCast(sh.n)),
                });
                try d.sync();
                if (i >= 5) best = @min(best, q.now() - t0);
            }
            const t1_ms = best * 1e3;
            best = 1e9;
            for (0..reps + 5) |i| {
                const t0 = q.now();
                try d.launch(ks.decode2, .{ .x = sh.n / 16, .y = 1 }, .{
                    tr_dev, xt, y, @as(i32, @intCast(sh.k)), @as(i32, @intCast(sh.n)), @as(i32, 8), w, @as(i32, 1), @as(i32, @intCast(sh.n)),
                });
                try d.sync();
                if (i >= 5) best = @min(best, q.now() - t0);
            }
            std.debug.print("TIMING {s} K={d} N={d} halves={d}: dequant {d:.3} ms, t1 m=1 {d:.3} ms, decode2 m=8 {d:.3} ms\n", .{ sh.name, sh.k, sh.n, hh, dq_ms, t1_ms, best * 1e3 });
        }
    }
    d.deinit();
}
