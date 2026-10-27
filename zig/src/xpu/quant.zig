//! xpu-eaq-quant — the fresh-quant integration driver (eaq-ldlq-spec-fable.md §1.2-§1.5,
//! §4.2-§4.6, §5.5; the §Q3 manifest of eaq-rebase-ruling-fable.md). One binary, one pass:
//!
//!   bf16 source shards (keep 851 tensors, drop visual/mtp; dense copied verbatim — vecTensor
//!   takes BF16 — except where the loader needs F16: the a/b gates convert host-side) ->
//!   embed the stream for `rows` x `cols` with calib.embedStream -> per layer: capture pass
//!   (calib.layerForward with dense slabs over the rows, the four site Hessians through the
//!   capture hooks), finalize each site's H (finiteness -> flipSigns -> rotateBoth -> damping
//!   lambda*mu, spec 2.4), blockLdl per slot (core.exl3_ldl, spec 3), per part: sv and the
//!   column RMS s, W_q fp64 (spec 4.2 step 3), ldlq.quantizePart (spec 4.3) -> trellis; then
//!   the apply pass (dequant slabs + layerForward -> S_{i+1}). The head is quantized last at
//!   --head-bits, in column windows (the full head part is 5 GB at fp32; windows of
//!   head_window columns reassemble into one trellis). The pack is written through
//!   core/eaq_write.zig (EAQ container, stock entry names); the §Q3 manifest fields per part
//!   (tiles/optimal/closed/failed/pruned1/pruned2/clamped, the 40-bucket gap histogram, the
//!   contract string, the producing encoder) live in the run report
//!   `<out>/quant_report.json` — the container manifest is the repacker's own, derived from
//!   bytes, and carries what it always carried.
//!
//! Checkpointing: the work dir holds the per-layer fragments (safetensors shards the reader
//! itself admits — T-LDLQ-5 reads them back through exl3_model.projectionExl3), the site
//! Hessians (fp32 K x K as captured) and the residual stream S_i after every layer;
//! `--resume` skips layers the progress file records as complete (each layer's completion
//! carries a sha256 over its fragment bytes, and the progress header pins the run
//! parameters that produced them — mismatched parameters refuse the resume rather than mix
//! half-quantized state).
//!
//! `--proxy` adds the spec 4.6 loss per part (tr(deltaW^T H' deltaW), the host fp64 form of
//! core.exl3_ldlq_ref.proxyLoss threaded over the columns — the reference's guard is a
//! single-threaded test-size guard, not an arithmetic one), the D-ratio from the eta strips,
//! and the T-LDLQ-4 comparison: when the run covers a single layer every part is encoded
//! twice — once with the slot's L, once with L = I — both losses under the true H', and the
//! LDLQ run must be the lower one (and must differ, or L never reached the loop).
//! A T-LDLQ-5 probe runs over every processed layer part: the fragment read back through
//! projectionExl3, calib.dequantSlot into the serving slab, then the probe-down ruling's
//! G1-G4 gates plus the pack-vs-source correlation and the heavy-tail census, all measured
//! against the pack's bytes or the captured source (see probeSlot's doc). The feedback and
//! the eta strip advance from the final packed codes: ldlq.zig regenerates each block's
//! Q16 with strip_decode over its own final trellis before strip_diff reads it.
//!
//! The device entries are the calibration slice's (calib.zig / ldlq.zig / exl3_encode.zig —
//! T-CAP, T-HESS, T-LDLQ all green); this file never touches their kernels, only drives
//! them. Card discipline: free device memory is reported (ze sysman, Gpu.memory) before any
//! allocation and the run refuses to start when the driver's own footprint would not fit —
//! the tf-main serve is reported, never touched.

const std = @import("std");
const q = @import("qwen27");
const ze = @import("ze");
const core = @import("core");
const devm = q.dev;
const fwd = q.fwd;
const load = q.loader;
const exl3_model = q.exl3_model;
const calib = q.calib;
const ldlq = q.ldlq;
const kb = @import("xpu_qwen27_kernels");

const alloc = q.alloc;
const D = q.D;
const FF = q.FF;
const LAYERS = q.LAYERS;
const VOCAB = q.VOCAB;
const GDN_IN = q.GDN_IN;
const ATT_IN = q.ATT_IN;
const Dev = devm.Dev;
const Model = load.Model;
const Proj = load.Proj;
const Checkpoint = devm.Checkpoint;
const ldl = core.exl3_ldl;
const ldlq_ref = core.exl3_ldlq_ref;
const efmt = core.exl3_format;
const eaq_write = @import("eaq_write");

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;
extern "c" fn mkdir(path: [*:0]const u8, mode: c_uint) c_int;
extern "c" fn unlink(path: [*:0]const u8) c_int;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;
extern "c" fn munmap(addr: *anyopaque, len: usize) c_int;
extern "c" fn __errno_location() [*]c_int;
const PROT_READ = 1;
const MAP_PRIVATE = 2;

/// The §Q3 contract string: every tile of this run certifies against it (the rebasing
/// ruling's settled state: u32-rebased Q12, sentinel 0xFFFFFFFF, clamp at 4.0).
pub const contract = "exl3-enc/v1: q12 u32-rebased inf=0xFFFFFFFF clamp=4.0";

/// The kept-tensor count of the source after the visual/mtp drop (spec 1.1); a source that
/// disagrees is a different model than the tables here were written for.
pub const KEPT = 851;

pub const Error = error{
    Usage,
    BadCalib,
    InsufficientDeviceMemory,
    CholeskyFailed,
    TileFailed,
    ClampHigh,
    ProxyOrdering,
    ProbeFailed,
    PublishFailed,
};

pub const Options = struct {
    src: []const u8 = "",
    out: []const u8 = "",
    calib_path: []const u8 = "",
    rows: u32 = 100,
    cols: u32 = 2048,
    bits: u32 = 3,
    head_bits: u32 = 4,
    seed: u64 = 0,
    seed_given: bool = false,
    lambda: f64 = 0.025,
    scale: f64 = 1.0,
    layer_a: usize = 0,
    layer_b: usize = LAYERS - 1,
    save_h: ?[]const u8 = null,
    proxy: bool = false,
    allow_identity_fallback: bool = false,
    work: []const u8 = "",
    resume_run: bool = false,
    dry_run: bool = false,
    device_check: bool = false, // the card probe mode (Main 2026-10-26, the device-memory gate)
    plan_path: []const u8 = "", // the allocator's plan JSON (--plan); empty synthesizes the degenerate uniform plan
};

// ---------------------------------------------------------------- file plumbing

fn dupeSentinel(path: []const u8) ![:0]u8 {
    return alloc.dupeSentinel(u8, path, 0);
}

fn openRead(path: []const u8) !c_int {
    const sent = try dupeSentinel(path);
    defer alloc.free(sent);
    const fd = open(sent.ptr, 0);
    if (fd < 0) return error.FileNotFound;
    return fd;
}

const MappedFile = struct {
    bytes: []const u8,

    pub fn open(path: []const u8) !MappedFile {
        const fd = try openRead(path);
        defer _ = close(fd);
        const size: usize = @intCast(lseek(fd, 0, 2));
        const base = mmap(null, size, PROT_READ, MAP_PRIVATE, fd, 0) orelse return error.MmapFailed;
        if (@intFromPtr(base) == @as(usize, @bitCast(@as(isize, -1)))) return error.MmapFailed;
        return .{ .bytes = @as([*]const u8, @ptrCast(base))[0..size] };
    }

    pub fn unmap(self: *MappedFile) void {
        if (self.bytes.len == 0) return;
        _ = munmap(@ptrCast(@constCast(self.bytes.ptr)), self.bytes.len);
        self.bytes = &.{};
    }
};

fn readFile(path: []const u8) ![]u8 {
    var mf = try MappedFile.open(path);
    defer mf.unmap();
    return alloc.dupe(u8, mf.bytes);
}

fn mkdirp(path: []const u8) !void {
    const sent = try dupeSentinel(path);
    defer alloc.free(sent);
    var at: usize = 1; // every prefix of the absolute path, in order (the failed_H dump lives under OUT, which a dry run never creates)
    while (std.mem.indexOfScalarPos(u8, path, at, '/')) |sl| {
        if (sl != path.len - 1) {
            const pre = try dupeSentinel(path[0..sl]);
            defer alloc.free(pre);
            _ = mkdir(pre.ptr, 0o755); // EEXIST is fine; the later open is the real verdict
        }
        at = sl + 1;
    }
    _ = mkdir(sent.ptr, 0o755);
}

fn writeFile(path: []const u8, bytes: []const u8) !void {
    const sent = try dupeSentinel(path);
    defer alloc.free(sent);
    const fd = open(sent.ptr, @as(c_int, 65 | 8), @as(c_int, 0o644)); // O_CREAT | O_WRONLY | O_TRUNC
    if (fd < 0) {
        std.debug.print("quant: create {s}: errno {d}\n", .{ path, __errno_location()[0] });
        return error.FileNotFound;
    }
    defer _ = close(fd);
    var done: usize = 0;
    while (done < bytes.len) {
        const n = write(fd, bytes.ptr + done, bytes.len - done);
        if (n < 0) {
            const e = __errno_location()[0];
            if (e == 4) continue; // EINTR: restart the syscall, do not abort a 35-minute layer
            std.debug.print("quant: write {s}: {d} of {d} bytes done, errno {d}\n", .{ path, done, bytes.len, e });
            return error.FileNotFound;
        }
        if (n == 0) {
            std.debug.print("quant: write {s}: {d} of {d} bytes done, short write\n", .{ path, done, bytes.len });
            return error.FileNotFound;
        }
        done += @intCast(n);
    }
}

fn fileExists(path: []const u8) bool {
    const sent = alloc.dupeSentinel(u8, path, 0) catch return false;
    defer alloc.free(sent);
    const fd = open(sent.ptr, 0);
    if (fd < 0) return false;
    _ = close(fd);
    return true;
}

fn removeFile(path: []const u8) void {
    const sent = alloc.dupeSentinel(u8, path, 0) catch return;
    defer alloc.free(sent);
    _ = unlink(sent.ptr);
}

fn sha256Hex(bytes: []const u8) [64]u8 {
    const Sha256 = std.crypto.hash.sha2.Sha256;
    var d: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

fn sha256File(path: []const u8) !struct { hex: [64]u8, bytes: []u8 } {
    var mf = try MappedFile.open(path);
    defer mf.unmap();
    return .{ .hex = sha256Hex(mf.bytes), .bytes = try alloc.dupe(u8, mf.bytes) };
}

fn fmtStr(comptime f: []const u8, args: anytype) ![]u8 {
    return std.fmt.allocPrint(alloc, f, args);
}

fn off(p: *anyopaque, bytes: usize) *anyopaque {
    return @ptrFromInt(@intFromPtr(p) + bytes);
}

extern "c" fn rename(old: [*:0]const u8, new: [*:0]const u8) c_int;

fn renameAt(from: []const u8, to: []const u8) !void {
    const t0 = try dupeSentinel(from);
    defer alloc.free(t0);
    const f0 = try dupeSentinel(to);
    defer alloc.free(f0);
    if (rename(t0.ptr, f0.ptr) != 0) return error.FileNotFound;
}

/// Source name -> pack name: the source carries the multimodal naming (the 851 kept tensors
/// are model.language_model.* plus lm_head); the reader's loader (load.zig:428,441) reads
/// the pack's stock names — the same tensors without the language_model hop.
fn packName(name: []const u8) ![]u8 {
    if (std.mem.startsWith(u8, name, "model.language_model.layers."))
        return fmtStr("model.layers.{s}", .{name["model.language_model.layers.".len..]});
    if (std.mem.eql(u8, name, "model.language_model.embed_tokens.weight")) return try alloc.dupe(u8, "model.embed_tokens.weight");
    if (std.mem.eql(u8, name, "model.language_model.norm.weight")) return try alloc.dupe(u8, "model.norm.weight");
    return alloc.dupe(u8, name); // lm_head and anything else keeps its name
}

/// Device bytes to host, chunked and event-signalled (the Dev.copyIn ruling, read-side
/// twin): one 4.19 GB append let a transient exec-queue reset surface as a failed
/// HostSynchronize on an in-flight DMA and take a GREEN run down at settle. Each chunk
/// rides a host-visible event and the caller blocks until the card really signals, so
/// d.sync afterwards drains only an idle list; a card that stays stuck past the wall
/// deadline returns error.DeviceLost instead of killing the process from the guard.
var dl_ev: ze.H = null;
fn dlslice(d: *Dev, p: *anyopaque, bytes: usize) ![]u8 {
    const b = try alloc.alloc(u8, bytes);
    errdefer alloc.free(b);
    if (dl_ev == null) dl_ev = (try d.g.events(1))[0];
    const chunk: usize = 256 << 20;
    var done: usize = 0;
    while (done < bytes) {
        const n = @min(chunk, bytes - done);
        _ = d.g.api.zeEventHostReset(dl_ev.?);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, off(b.ptr, done), off(p, done), n, dl_ev.?, 0, null));
        const deadline = q.now() + 600;
        while (d.g.api.zeEventHostSynchronize(dl_ev.?, 300_000_000_000) != 0) {
            if (q.now() > deadline) return error.DeviceLost;
        }
        try d.sync();
        done += n;
    }
    return b;
}

// ---------------------------------------------------------------- the part table

/// One quantized tensor of a fused slot: the stored-name base (the source's name minus
/// `.weight`), the part's N, and its column base inside the fused slot (spec 1.1's table).
pub const PartPlan = struct {
    base: []const u8, // e.g. model.language_model.layers.12.mlp.down_proj
    tail: []const u8, // the same without the layer prefix, for projectionExl3's part list
    n: u32,
    n0: u32,
};

pub const SlotPlan = struct {
    k: u32, // the site's K (the Hessian side)
    nt: u32, // the fused Proj.n (the slab pitch; the 256-padded N)
    wide: bool, // the Proj.wide flag (which serving GEMM the fused column group feeds)
    parts: []const PartPlan,
};

/// The four fused slots of one decoder layer in part order — the twins of load.zig's
/// exl3_model.projectionExl3 calls (load.zig:447-457): mixer_in q 12288 / k 1024 / v 1024
/// (attention) or in_proj_qkv 10240 / in_proj_z 6144 (GDN), mixer_out o_proj / out_proj
/// 5120, gu gate+up 17408 x2, down 5120. a and b stay dense (N = 48 fails the 128-multiple
/// the decode path requires); the visual tower and the mtp draft have no reader slot and
/// are dropped with the kept-851 filter.
pub fn layerSlots(a: std.mem.Allocator, i: usize) !([4]SlotPlan) {
    const p = try fmtStr("model.language_model.layers.{d}.", .{i});
    const attn = q.isAttention(i);
    const mk = struct {
        fn pp(aa: std.mem.Allocator, pref: []const u8, tail: []const u8, n: u32, n0: u32) !PartPlan {
            return .{ .base = try fmtStr("{s}{s}", .{ pref, tail }), .tail = try aa.dupe(u8, tail), .n = n, .n0 = n0 };
        }
    }.pp;
    const mi: SlotPlan = if (attn) .{
        .k = D, .nt = ATT_IN, .wide = true,
        .parts = try a.dupe(PartPlan, &.{
            try mk(a, p, "self_attn.q_proj", 12288, 0),
            try mk(a, p, "self_attn.k_proj", 1024, 12288),
            try mk(a, p, "self_attn.v_proj", 1024, 13312),
        }),
    } else .{
        .k = D, .nt = GDN_IN, .wide = true,
        .parts = try a.dupe(PartPlan, &.{
            try mk(a, p, "linear_attn.in_proj_qkv", 10240, 0),
            try mk(a, p, "linear_attn.in_proj_z", 6144, 10240),
        }),
    };
    const mo: SlotPlan = .{
        .k = 6144, .nt = D, .wide = false,
        .parts = try a.dupe(PartPlan, &.{try mk(a, p, if (attn) "self_attn.o_proj" else "linear_attn.out_proj", D, 0)}),
    };
    const gu: SlotPlan = .{
        .k = D, .nt = 2 * FF, .wide = true,
        .parts = try a.dupe(PartPlan, &.{
            try mk(a, p, "mlp.gate_proj", FF, 0),
            try mk(a, p, "mlp.up_proj", FF, FF),
        }),
    };
    const dn: SlotPlan = .{
        .k = FF, .nt = D, .wide = false,
        .parts = try a.dupe(PartPlan, &.{try mk(a, p, "mlp.down_proj", D, 0)}),
    };
    return .{ mi, mo, gu, dn };
}

pub fn headSlotPlan(a: std.mem.Allocator) !SlotPlan {
    return .{
        .k = D, .nt = VOCAB, .wide = true,
        .parts = try a.dupe(PartPlan, &.{.{ .base = try a.dupe(u8, "lm_head"), .tail = try a.dupe(u8, "lm_head"), .n = VOCAB, .n0 = 0 }}),
    };
}

/// The head streams in column windows (spec 1.5's ledger: the full 5120 x 248320 fp32 part
/// is 5 GB; windows of this many columns keep every driver buffer bounded). 248320 =
/// 7 x 32768 + 18944: every window is a multiple of 16.
const head_window: u32 = 32768;

// ---------------------------------------------------------------- the dense model

fn bf16ToF16(src: []const u8, out: []u16) void {
    for (out, 0..) |*o, e| {
        // bf16 is the top half of fp32: expand by shifting left, then narrow (the twin of the kernel-side bf2f)
        const bits: u32 = @as(u32, src[2 * e]) | (@as(u32, src[2 * e + 1]) << 8);
        o.* = @bitCast(@as(f16, @floatCast(@as(f32, @bitCast(bits << 16)))));
    }
}

/// The quantizer's own Model (spec 1.4): the small tensors loaded from the bf16 source the
/// way the dense loader loads them (vecTensor admits BF16; the a/b gates must be F16 at
/// load.zig:460, so the source's BF16 converts host-side), plus the device state the
/// forward needs (kc/vc, rec/rec_save/stash/conv — the load.zig:453-475 sequence). The
/// Proj slots are placeholders with the right n/k/wide; calib.layerForward swaps in the
/// dense16 slab views at every call. The head is never GEMM'd here: finalNorm reads only
/// m.norm and m.head.gs.
pub fn buildDenseModel(d: *Dev, c: *const Checkpoint) !Model {
    var m: Model = undefined;
    const e = try c.get("model.embed_tokens.weight");
    m.embed_w = try d.alloc_(@as(usize, VOCAB) * D * 2);
    if (std.mem.eql(u8, e.dtype, "F16")) {
        try d.copyIn(m.embed_w, e.bytes);
    } else if (std.mem.eql(u8, e.dtype, "BF16")) {
        if (e.bytes.len != @as(usize, VOCAB) * D * 2) return error.UnexpectedShape;
        const out = try alloc.alloc(u16, @as(usize, VOCAB) * D);
        defer alloc.free(out);
        bf16ToF16(e.bytes, out);
        try d.copyIn(m.embed_w, std.mem.sliceAsBytes(out));
    } else return error.UnexpectedDtype;
    m.embed_s = null;
    m.embed_b = null;
    m.norm = try load.vecTensor(d, c, "model.language_model.norm.weight", true); // the Qwen3_5 norm is x·rsqrt·(1+w): the source stores gamma − 1
    const dummy = try d.alloc_(64);
    m.head = .{ .w = dummy, .s = dummy, .bg = dummy, .n = VOCAB, .k = D, .wide = true, .gs = q.GS };
    var invf: [32]f32 = undefined;
    for (&invf, 0..) |*f, i| f.* = @floatCast(std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(i)) / 32.0));
    m.invf = try d.upload(std.mem.sliceAsBytes(&invf));
    const stage = try d.alloc_(800 << 20);
    defer {
        d.g.free(stage);
        d.allocated -= 800 << 20;
    }
    for (&m.layers, 0..) |*l, i| {
        const p = try fmtStr("model.language_model.layers.{d}.", .{i});
        l.* = .{
            .in_norm = try load.vecTensor(d, c, try fmtStr("{s}input_layernorm.weight", .{p}), true),
            .post_norm = try load.vecTensor(d, c, try fmtStr("{s}post_attention_layernorm.weight", .{p}), true),
            .attn = q.isAttention(i),
            .mixer_in = .{ .w = stage, .s = stage, .bg = stage, .n = if (q.isAttention(i)) ATT_IN else GDN_IN, .k = D, .wide = true, .gs = q.GS },
            .mixer_out = .{ .w = stage, .s = stage, .bg = stage, .n = D, .k = 6144, .wide = false, .gs = q.GS },
            .gu = .{ .w = stage, .s = stage, .bg = stage, .n = 2 * FF, .k = D, .wide = true, .gs = q.GS },
            .down = .{ .w = stage, .s = stage, .bg = stage, .n = D, .k = FF, .wide = false, .gs = q.GS },
        };
        if (l.attn) {
            l.qn = try load.vecTensor(d, c, try fmtStr("{s}self_attn.q_norm.weight", .{p}), true);
            l.kn = try load.vecTensor(d, c, try fmtStr("{s}self_attn.k_norm.weight", .{p}), true);
            l.kc = try d.alloc_(fwd.kvBytes());
            l.vc = try d.alloc_(fwd.kvBytes());
        } else {
            const ta = try c.get(try fmtStr("{s}linear_attn.in_proj_a.weight", .{p}));
            const tb = try c.get(try fmtStr("{s}linear_attn.in_proj_b.weight", .{p}));
            if (ta.bytes.len != 48 * D * 2 or tb.bytes.len != 48 * D * 2) return error.UnexpectedShape;
            l.ab = try d.alloc_(96 * D * 2);
            const ab_h = try alloc.alloc(u16, 96 * D);
            defer alloc.free(ab_h);
            if (std.mem.eql(u8, ta.dtype, "F16")) {
                @memcpy(ab_h[0 .. 48 * D], std.mem.bytesAsSlice(u16, ta.bytes));
                @memcpy(ab_h[48 * D ..], std.mem.bytesAsSlice(u16, tb.bytes));
            } else if (std.mem.eql(u8, ta.dtype, "BF16")) {
                bf16ToF16(ta.bytes, ab_h[0 .. 48 * D]);
                bf16ToF16(tb.bytes, ab_h[48 * D ..]);
            } else return error.UnexpectedDtype;
            try d.copyIn(l.ab.?, std.mem.sliceAsBytes(ab_h));
            l.conv_w = try load.vecTensor(d, c, try fmtStr("{s}linear_attn.conv1d.weight", .{p}), false);
            l.alog = try load.vecTensor(d, c, try fmtStr("{s}linear_attn.A_log", .{p}), false);
            l.dt = try load.vecTensor(d, c, try fmtStr("{s}linear_attn.dt_bias", .{p}), false);
            l.gnorm = try load.vecTensor(d, c, try fmtStr("{s}linear_attn.norm.weight", .{p}), false);
            l.rec = try d.alloc_(48 * 128 * 128 * 4);
            l.rec_save = try d.alloc_(48 * 128 * 128 * 4);
            l.stash = try d.alloc_(q.STASH);
            try d.fill(l.rec.?, @as(u32, 0), 48 * 128 * 128 * 4);
            for (&l.conv) |*cv| {
                cv.* = try d.alloc_(3 * 10240 * 2);
                try d.fill(cv.*.?, @as(u16, 0), 3 * 10240 * 2);
            }
        }
        if (i % 16 == 15) std.debug.print("  quant: small tensors loaded {d}/{d} layers\n", .{ i + 1, LAYERS });
    }
    try d.sync();
    return m;
}

/// The kernel table (the Runner.init sequence, qwen27.zig:260-336 — layerForward and the
/// capture hooks consume d.k through the shared Dev).
pub fn initKernelTable(d: *Dev) !void {
    const mg = try d.g.module(&kb.glue);
    const md = try d.g.module(&kb.decode);
    const mx = try d.g.module(&kb.exl3);
    const mxh = try d.g.module(&kb.exl3_h);
    const mm = try d.g.module(&kb.gemm);
    const mgd = try d.g.module(&kb.gdn);
    const mw = try d.g.module(&kb.ngen_wide);
    const mn = try d.g.module(&kb.ngen_narrow);
    const mm128 = try d.g.module(&kb.gemm128);
    const mw128 = try d.g.module(&kb.ngen_wide128);
    const mn128 = try d.g.module(&kb.ngen_narrow128);
    const mcl = try d.g.module(&kb.calib);
    d.k = .{
        .embed = try d.kernel(mg, "embed", 256),
        .add_rmsnorm = try d.kernel(mg, "add_rmsnorm", 128),
        .add_rmsnorm_dec = try d.kernel(mg, "add_rmsnorm_dec", 512),
        .gdn_pre = try d.kernel(mg, "gdn_pre", 256),
        .gdn_pre_rows = try d.kernel(mg, "gdn_pre_rows", 256),
        .gated_norm = try d.kernel(mg, "gated_norm", 256),
        .attn_prep = try d.kernel(mg, "attn_prep", 256),
        .attn_gate = try d.kernel(mg, "attn_gate", 256),
        .swiglu = try d.kernel(mg, "swiglu", 256),
        .argmax = try d.kernel(mg, "argmax", 256),
        .argmax_part = try d.kernel(mg, "argmax_part", 256),
        .argmax_part_masked = try d.kernel(mg, "argmax_part_masked", 256),
        .tap = try d.kernel(mg, "tap", 256),
        .conv_commit = try d.kernel(mg, "conv_commit", 256),
        .repack = try d.kernel(mg, "repack", 16),
        .gdn_step = try d.kernel(md, "gdn_step", 16),
        .attn_decode = try d.kernel(md, "attn_decode", 16),
        .attn_merge = try d.kernel(md, "attn_merge", 16),
        .kv_move = try d.kernel(md, "kv_move", 16),
        .exl3_gemv = try d.kernel(mx, "exl3_gemv", 16),
        .exl3_decode2 = try d.kernel(mx, "exl3_decode2", 16),
        .exl3_decode2_t1 = try d.kernel(mx, "exl3_decode2_t1", 16),
        .exl3_dequant128 = try d.kernel(mx, "exl3_dequant128", 256),
        .exl3_prompt16 = try d.kernel(mx, "exl3_prompt16", 256),
        .exl3_decode2_h = try d.kernel(mxh, "exl3_decode2_h", 16),
        .exl3_decode2_t1_h = try d.kernel(mxh, "exl3_decode2_t1_h", 16),
        .exl3_dequant128_c = try d.kernel(mxh, "exl3_dequant128_c", 256),
        .exl3_xform_h = try d.kernel(mxh, "exl3_xform_h", 16),
        .exl3_yscale_h = try d.kernel(mxh, "exl3_yscale_h", 16),
        .rot_heads = try d.kernel(mg, "rot_heads", 16),
        .attn_prompt = try d.kernel(md, "attn_prompt", 128),
        .attn_prompt8 = try d.kernel(md, "attn_prompt8", 128),
        .embed_dense = try d.kernel(mg, "embed_dense", 256),
        .dense_rows = try d.kernel(mg, "dense_rows", 256),
        .repack_gptq = try d.kernel(mg, "repack_gptq", 16),
        .quantize_mlx = try d.kernel(mg, "quantize_mlx", 16),
        .dec_wide_g128 = try d.kernel(mm128, "w4k1_decode", 64),
        .dec_wide64_g128 = try d.kernel(mm128, "w4k1_d64", 64),
        .dec_narrow_g128 = try d.kernel(mm128, "w4k_dsplit", 64),
        .ngen_wide_g128 = try d.kernel(mw128, "w4g_dpb", 256),
        .ngen_narrow_g128 = try d.kernel(mn128, "w4g_s4dpb", 256),
        .ngen_swiglu = try d.kernel(try d.g.module(&kb.ngen_swiglu), "w4g_dpb_swiglu", 256),
        .ngen_swiglu_g128 = try d.kernel(try d.g.module(&kb.ngen_swiglu128), "w4g_dpb_swiglu", 256),
        .rowsums_ff = try d.kernel(mg, "rowsums_ff", 128),
        .fakequant_rows = try d.kernel(mg, "fakequant_rows", 256),
        .a8 = try d.kernel(try d.g.module(&kb.ngen_a8), "w4a8", 256),
        .a8_swiglu = try d.kernel(try d.g.module(&kb.ngen_a8_swiglu), "w4a8_swiglu", 256),
        .quant_a8 = try d.kernel(mg, "quant_a8", 128),
        .quant_a4 = try d.kernel(mg, "quant_a4", 128),
        .a4 = try d.kernel(try d.g.module(&kb.ngen_a4), "w4a4", 256),
        .dec_wide = try d.kernel(mm, "w4k1_decode", 64),
        .dec_wide64 = try d.kernel(mm, "w4k1_d64", 64),
        .dec_narrow = try d.kernel(mm, "w4k_dsplit", 64),
        .gdn_prep = try d.kernel(mgd, "gdn_prep", 64),
        .gdn_state = try d.kernel(mgd, "gdn_state", 256),
        .gdn_out = try d.kernel(mgd, "gdn_out", 128),
        .ngen_wide = try d.kernel(mw, "w4g_dpb", 256),
        .ngen_narrow = try d.kernel(mn, "w4g_s4dpb", 256),
        .bf16_to_slab = try d.kernel(mcl, "bf16_to_slab", 256),
        .transpose16 = try d.kernel(mcl, "transpose16", 256),
        .exl3_syrk32 = try d.kernel(mcl, "exl3_syrk32", 256),
    };
    for ([_]ze.H{ d.k.attn_prep, d.k.attn_decode, d.k.gdn_pre, d.k.gdn_step }) |k| try d.g.indirect(k);
}

/// The names the pack copies verbatim: every kept source tensor (visual/mtp dropped) except
/// the 400 quantized linear weights and lm_head. The a/b gates convert BF16 -> F16 on the
/// way in (the loader requires F16 48 x D gates, load.zig:460); everything else — embed and
/// norms included (vecTensor takes BF16, load.zig:176) — is byte-exact.
const DenseEntry = struct { name: []const u8, dtype: []const u8, convert_to_f16: bool };

fn quantizedName(i: usize, s: []const u8) bool {
    const p = std.fmt.allocPrint(alloc, "model.language_model.layers.{d}.", .{i}) catch return false;
    defer alloc.free(p);
    const tails = [_][]const u8{
        "self_attn.q_proj.weight",     "self_attn.k_proj.weight",        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",     "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
        "linear_attn.out_proj.weight", "mlp.gate_proj.weight",           "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    };
    for (tails) |t| {
        const full = std.fmt.allocPrint(alloc, "{s}{s}", .{ p, t }) catch return false;
        defer alloc.free(full);
        if (std.mem.eql(u8, s, full)) return true;
    }
    return false;
}

fn isKept(name: []const u8) bool {
    return !(std.mem.startsWith(u8, name, "model.visual.") or std.mem.startsWith(u8, name, "mtp."));
}

fn denseRegistry(c: *const Checkpoint) ![]DenseEntry {
    var list: std.ArrayList(DenseEntry) = .empty;
    for (c.tensors.items) |t| {
        if (!isKept(t.name)) continue;
        if (std.mem.eql(u8, t.name, "lm_head.weight")) continue; // replaced by the trellis head
        var skip = false;
        for (0..LAYERS) |i| if (quantizedName(i, t.name)) {
            skip = true;
            break;
        };
        if (skip) continue;
        const gate = std.mem.endsWith(u8, t.name, "linear_attn.in_proj_a.weight") or
            std.mem.endsWith(u8, t.name, "linear_attn.in_proj_b.weight");
        try list.append(alloc, .{ .name = try packName(t.name), .dtype = t.dtype, .convert_to_f16 = gate and std.mem.eql(u8, t.dtype, "BF16") });
    }
    return list.items;
}

// ---------------------------------------------------------------- the records

const TileItem = struct { // the §Q3 itemised tile: a nonzero flag or a non-primary encoder
    prefix: []const u8,
    tk: u32,
    tn: u32,
    encoder: []const u8,
    status: u32,
    flags: u32,
    cost: u64,
    open_cost: u64,
    pinned: u32,
    reason: []const u8,
};

pub const PartRecord = struct { // pub for the host merge test (PartWork carries it)
    prefix: []const u8,
    encoder: []const u8 = "xpu-kernel",
    k: u32 = 0,
    n: u32 = 0,
    halves: u16 = 0,
    tiles: u32 = 0,
    optimal: u32 = 0,
    closed: u32 = 0,
    failed: u32 = 0,
    pruned1: u32 = 0,
    pruned2: u32 = 0,
    clamped: u64 = 0,
    gap_hist: [40]u32 = @splat(0),
    eta_sq: f64 = 0,
    proxy: ?f64 = null,
    proxy_identity: ?f64 = null, // the same part encoded with L = I (the T-LDLQ-4 comparison)
    dratio: ?f64 = null, // tr(eta^T D eta) / (mean diag D * eta_sq)
    identity_fallback: bool = false,
    items: []const TileItem = &.{},
    trellis_sha256: ?[]const u8 = null, // Q3 binding: the trellis payload's sha256, checked by the writer against the staged input
};

/// The 40 log2 buckets of cost - open_cost (the §Q3 histogram; the ruling widened the
/// PartReport's 16 to 40). Bucket 0 is the zero gap; bucket k (1..39) is [2^(k-1), 2^k).
fn gapBucket40(diff: u64) usize {
    if (diff == 0) return 0;
    const l: usize = 63 - @as(usize, @clz(diff));
    return @min(39, l + 1);
}

fn slotSeed(seed: u64, layer: usize, slot: usize, part: ?usize) u64 {
    var h = std.hash.Wyhash.init(seed);
    h.update(std.mem.asBytes(&layer));
    h.update(std.mem.asBytes(&slot));
    if (part) |p| h.update(std.mem.asBytes(&p));
    return h.final();
}

fn drawSigns(seed: u64, n: usize) ![]i8 {
    // deterministic per-(seed, position) Rademacher stream: no RNG state to outlive
    const s = try alloc.alloc(i8, n);
    for (s, 0..) |*e, i| e.* = if (std.hash.Wyhash.hash(seed, std.mem.asBytes(&i)) & 1 != 0) @as(i8, 1) else @as(i8, -1);
    return s;
}

/// The identity twin of a blockLdl result: L = I (unit 16x16 diagonal blocks, everything
/// else exactly zero — beginSlot's admitted form), D = the lower-triangular blocks of the
/// rotated damped H in BlockLdl's own d[b*256 + a*16 + c] indexing. For the T-LDLQ-4
/// comparison, and — gated on --allow-identity-fallback — for the Cholesky-failure
/// experiment. Never the shipped form of a slot on its own.
fn identityLdl(h64: []const f64, k: usize) !ldl.BlockLdl {
    const l = try alloc.alloc(f32, k * k);
    errdefer alloc.free(l);
    @memset(l, 0);
    for (0..k) |r| l[r * k + r] = 1;
    const d = try alloc.alloc(f64, (k / 16) * 256);
    errdefer alloc.free(d);
    for (0..k / 16) |b| for (0..16) |ai| for (0..16) |cc| {
        const r = b * 16 + ai;
        const c2 = b * 16 + cc;
        d[b * 256 + ai * 16 + cc] = if (ai >= cc) h64[r * k + c2] else 0;
    };
    return .{ .k = k, .l = l, .d = d, .lambda = 0, .mu = 0 };
}

// ---------------------------------------------------------------- the proxy (section 4.6, T-LDLQ-4)

/// The column-tile scratch for one proxy loss pass (tm columns wide, k rows): V the
/// delta against the source, U = L^T V, DU the block-diagonal D's response.
const ProxyTile = struct {
    v: []f64,
    u: []f64,
    du: []f64,
    tm: usize,
};

const ProxyJob = struct {
    got_a: []const f16,
    got_b: []const f16, // the L = I run's unpack; empty when there is no comparison
    wq: []const f64,
    l: []const f32, // the slot's real L: BOTH losses score under the true H' = L D L^T
    d: []const f64,
    k: usize,
    n: usize,
    lo: usize,
    hi: usize,
    loss_a: f64 = 0,
    loss_b: f64 = 0,

    fn colLoss(j: *ProxyJob, got: []const f16, tl: *ProxyTile) f64 {
        // Tiled over column blocks with L2-resident (row, column) sub-blocks: the dense
        // K x K L streams ONCE per tile (a 1.2 GB down-proj L streamed once per COLUMN
        // is what the measured 1e9 MAC/s came from), and every accumulation keeps the
        // serial pass's order — U by ascending r, acc by ascending p, the dot by
        // ascending r with columns ascending; the per-tile lossb fold re-associates
        // the running sum, so totals agree to rounding, not bit-for-bit.
        const k = j.k;
        const tm = tl.tm;
        const br: usize = 64;
        const bp: usize = 128;
        var loss: f64 = 0;
        var c0 = j.lo;
        while (c0 < j.hi) : (c0 += tm) {
            const cw = @min(tm, j.hi - c0);
            for (0..k) |r| { // the tile V[r][t] = Q16 - W_q (both runs score under the true H')
                for (0..cw) |t| tl.v[r * tm + t] = @as(f64, @floatCast(got[r * j.n + c0 + t])) - j.wq[r * j.n + c0 + t];
            }
            @memset(tl.u[0 .. k * tm], 0);
            var rb: usize = 0;
            while (rb < k) : (rb += br) { // U[p][t] += L[r][p] * V[r][t], r ascending
                const r1 = @min(k, rb + br);
                var pb: usize = 0;
                while (pb < k) : (pb += bp) {
                    const p1 = @min(k, pb + bp);
                    for (rb..r1) |r| {
                        const row = j.l[r * k ..][0..k];
                        const vr = tl.v[r * tm ..][0..cw];
                        for (pb..p1) |p| {
                            const lp: f64 = @as(f64, @floatCast(row[p]));
                            if (lp == 0) continue; // the exact zeros above the Cholesky factor's diagonal add nothing
                            const ut = tl.u[p * tm ..][0..cw];
                            for (0..cw) |t| ut[t] += lp * vr[t];
                        }
                    }
                }
            }
            for (0..cw) |t| { // (blockdiag D) u, column by column
                for (0..k / 16) |bj| for (0..16) |ai| {
                    var acc: f64 = 0;
                    for (0..16) |b| acc += j.d[(bj * 16 + ai) * 16 + b] * tl.u[(bj * 16 + b) * tm + t];
                    tl.du[(bj * 16 + ai) * tm + t] = acc;
                };
            }
            var lossb: [32]f64 = @splat(0); // tm pinned
            var accb: [64 * 32]f64 = undefined; // br x tm, hot per row block
            rb = 0;
            while (rb < k) : (rb += br) { // acc[r][t] += L[r][p] * DU[p][t], p ascending
                const r1 = @min(k, rb + br);
                @memset(&accb, 0);
                var pb: usize = 0;
                while (pb < k) : (pb += bp) {
                    const p1 = @min(k, pb + bp);
                    for (rb..r1) |r| {
                        const row = j.l[r * k ..][0..k];
                        const ac = accb[(r - rb) * tm ..][0..cw];
                        for (pb..p1) |p| {
                            const lp: f64 = @as(f64, @floatCast(row[p]));
                            if (lp == 0) continue;
                            const du = tl.du[p * tm ..][0..cw];
                            for (0..cw) |t| ac[t] += lp * du[t];
                        }
                    }
                }
                for (rb..r1) |r| { // the per-column dot, ascending r — the serial accumulation order
                    for (0..cw) |t| lossb[t] += tl.v[r * tm + t] * accb[(r - rb) * tm + t];
                }
            }
            for (0..cw) |t| loss += lossb[t]; // columns fold in order
        }
        return loss;
    }

    fn run(j: *ProxyJob) void {
        const tm: usize = 32; // the column tile: L streams once per tile, V/U/DU stay cache-hot
        const v = alloc.alloc(f64, j.k * tm) catch @panic("oom");
        defer alloc.free(v);
        const u = alloc.alloc(f64, j.k * tm) catch @panic("oom");
        defer alloc.free(u);
        const du = alloc.alloc(f64, j.k * tm) catch @panic("oom");
        defer alloc.free(du);
        var tl = ProxyTile{ .v = v, .u = u, .du = du, .tm = tm };
        j.loss_a = j.colLoss(j.got_a, &tl);
        if (j.got_b.len > 0) j.loss_b = j.colLoss(j.got_b, &tl);
    }
};

fn proxyJobRun(j: *ProxyJob) void {
    j.run();
}

// ------------------------------------------------------------- host threading
//
// Every heavy host-side pass in the driver (the proxy loss, the T-LDLQ-5 target build,
// the oracle decode, the metric scan) splits its work over the machine's cores. The
// thread count follows core.exl3_ldl's convention: EAQ_LDL_THREADS overrides the core
// count (std.Thread.getCpuCount) — no hardcoded cap. Chunks are whole and contiguous,
// each job writes only its own partials, and the totals fold in chunk order: the result
// is deterministic run to run and matches the serial pass up to fp64 associativity.
fn envUsizeQ(name: [*:0]const u8) ?usize {
    const v = std.c.getenv(name) orelse return null;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch null;
}

fn quantThreads(cap: usize) usize {
    const want = envUsizeQ("EAQ_LDL_THREADS") orelse (std.Thread.getCpuCount() catch 1);
    return std.math.clamp(want, 1, @max(1, cap));
}

const Split = struct { chunk: usize, njobs: usize };

fn chunkSplit(cap: usize, total: usize) Split {
    const nproc = quantThreads(cap);
    const chunk = @max(1, (total + nproc - 1) / nproc);
    return .{ .chunk = chunk, .njobs = (total + chunk - 1) / chunk };
}

const ChunkEntry = *const fn (ctx: *const anyopaque, lo: usize, hi: usize, idx: usize) void;

/// One spawn-each, join-all pass over [0, total) in contiguous chunks: the last working
/// job runs inline on the caller (the proxyLosses trick), a failed spawn runs that range
/// inline too — the totals always hold.
fn chunkRun(ctx: *const anyopaque, total: usize, split: Split, entry: ChunkEntry) void {
    const Job = struct {
        ctx: *const anyopaque,
        lo: usize,
        hi: usize,
        idx: usize,
        entry: ChunkEntry,
        fn go(j: *@This()) void { j.entry(j.ctx, j.lo, j.hi, j.idx); }
    };
    const jobs = alloc.alloc(Job, split.njobs) catch @panic("oom");
    defer alloc.free(jobs);
    const threads = alloc.alloc(std.Thread, split.njobs) catch @panic("oom");
    defer alloc.free(threads);
    var spawned: usize = 0;
    var made: usize = 0;
    while (made < split.njobs) : (made += 1) {
        const lo = made * split.chunk;
        if (lo >= total) break;
        jobs[made] = .{ .ctx = ctx, .lo = lo, .hi = @min(total, lo + split.chunk), .idx = made, .entry = entry };
        if (made + 1 == split.njobs or lo + split.chunk >= total) { // the last chunk runs inline
            jobs[made].go();
            made += 1;
            break;
        }
        threads[spawned] = std.Thread.spawn(.{}, Job.go, .{&jobs[made]}) catch {
            jobs[made].go(); // no thread available: the range runs inline, the total holds
            continue; // the loop's made += 1 is the only increment — a manual one here double-steps
        };
        spawned += 1;
    }
    for (threads[0..spawned]) |t| t.join();
}

/// The T-LDLQ-5 metric scan, one chunk of rows: every accumulator private; the driver
/// (probeSlot) folds the partials in chunk order — blk sums add, worst values merge by
/// argmax, the tail census re-inserts top-1000 entries in chunk order.
const ProbeJob = struct {
    tt: []const f64 = &.{},
    out64: []const f64 = &.{},
    zc: []const bool = &.{},
    bytes: []const u8 = &.{},
    svh: []const f16 = &.{},
    k: usize = 0,
    n: usize = 0,
    n0: usize = 0,
    j0: usize = 0,
    p_sq: []f64 = &.{},
    p_worst: []f64 = &.{},
    p_wr: []u32 = &.{},
    p_wc: []u32 = &.{},
    p_clamp: []u32 = &.{},
    p_col: []f64 = &.{},
    p_col_all: []f64 = &.{}, // whole-column ratio energy over ALL rows (g2-colfloor ruling 2026-10-27)
    hist: [31]u64 = @splat(0),
    g3_uncl: usize = 0,
    g3_clmp: usize = 0,
    g3_worst: f64 = 0,
    g3_row: usize = 0,
    g3_col: usize = 0,
    g3_bound: f64 = 0,
    src_sq: f64 = 0,
    dec_sq: f64 = 0,
    err_abs_sq: f64 = 0,
    st: f64 = 0,
    sq: f64 = 0,
    stq: f64 = 0,
    stt: f64 = 0,
    sqq: f64 = 0,
    rot_t: f64 = 0,
    rot_tt: f64 = 0,
    rot_tq: f64 = 0,
    ht_src: [1000]f32 = @splat(0),
    ht_err: [1000]f64 = @splat(0),
    ht_n: usize = 0,
    ht_min: usize = 0,

    fn run(j: *ProbeJob, r0: usize, r1: usize) void {
        for (r0..r1) |rr| {
            const jb = rr / 16;
            for (0..j.n) |cc| {
                if (j.zc[cc]) continue; // dead columns: target zero by construction, not a gate element
                const tv: f32 = @floatCast(j.tt[rr * j.n + cc]); // the rotated CODE target (the domain the encoder clamps in)
                const qv: f32 = @floatCast(j.out64[rr * j.n + cc]); // dequantize's reconstruction = the pack's WEIGHT-space W
                const idx = ((j.n0 + cc) * j.k + rr) * 2;
                const sbits: u32 = @as(u32, j.bytes[idx]) | (@as(u32, j.bytes[idx + 1]) << 8);
                const sv_src: f32 = @as(f32, @bitCast(sbits << 16)); // the true bf16 checkpoint weight (read-only; the source-hash invariant above pins it)
                // The G2-G4 distortion is the pack decode against the TRUE SOURCE WEIGHT, both in
                // the weight domain (efmt.dequantize reconstructs W; the ruling: "RMS against the true
                // checkpoint" = ~0.14 of scale). Comparing the weight-space decode against the ROTATED
                // code target (tt) instead is a domain mismatch that reads ~sqrt(2) scale units.
                const ef: f64 = @abs(@as(f64, qv) - @as(f64, sv_src)) / @max(@as(f64, @abs(j.svh[cc])), 1e-30); // scale units
                const ef_raw: f64 = @abs(@as(f64, qv) - @as(f64, sv_src)); // weight units (the absolute RMS)
                j.p_sq[jb] += ef * ef;
                if (ef > j.p_worst[jb]) {
                    j.p_worst[jb] = ef;
                    j.p_wr[jb] = @intCast(rr);
                    j.p_wc[jb] = @intCast(cc);
                }
                if (@abs(tv) > 3.775) j.p_clamp[jb] += 1; // the encoder's clamp predicate on the rotated target (the driver's own clamp rate 1.649e-4 and clamp_exp 3.172e-4 both sit at the two-sided 3.775-sigma tail)
                j.dec_sq += @as(f64, qv) * qv;
                j.src_sq += @as(f64, sv_src) * @as(f64, sv_src);
                j.st += @as(f64, sv_src); // the pack-vs-source correlation, WEIGHT space (decode out64 vs source): the >=0.98 signal
                j.sq += @as(f64, qv);
                j.stq += @as(f64, sv_src) * qv;
                j.stt += @as(f64, sv_src) * @as(f64, sv_src);
                j.sqq += @as(f64, qv) * qv;
                j.rot_t += @as(f64, tv); // informational: decode-vs-rotated-target correlation (cross-domain, expected ~0)
                j.rot_tt += @as(f64, tv) * @as(f64, tv);
                j.rot_tq += @as(f64, qv) * @as(f64, tv);
                const asrc = @abs(sv_src);
                if (j.ht_n < j.ht_src.len) {
                    j.ht_src[j.ht_n] = asrc;
                    j.ht_err[j.ht_n] = ef;
                    if (j.ht_n == 0 or asrc < j.ht_src[j.ht_min]) j.ht_min = j.ht_n;
                    j.ht_n += 1;
                } else if (asrc > j.ht_src[j.ht_min]) {
                    j.ht_src[j.ht_min] = asrc;
                    j.ht_err[j.ht_min] = ef;
                    var m: usize = 0;
                    for (j.ht_src, 0..) |v, ti| {
                        if (v < j.ht_src[m]) m = ti;
                    }
                    j.ht_min = m;
                }
                j.p_col_all[cc] += ef * ef; // the shipped-bytes column gate reads this, not the 16-row window (g2-colfloor ruling)
                if (jb == j.j0) {
                    j.p_col[cc] += ef * ef;
                    j.hist[@min(@as(usize, @intFromFloat(@floor(ef * 10))), j.hist.len - 1)] += 1; // scale units, the ruling's 0.1-bin table
                    j.err_abs_sq += ef_raw * ef_raw;
                    if (@abs(tv) > 3.775) {
                        // the ruling's clamp predicate: beyond the two-sided
                        // 3.775-sigma tail the code saturates, so the residual against
                        // the rotated target is |T| - 3.775 (+ the ruling's 1.0 slack);
                        // the predicate and the bound both live in the rotated domain
                        // the encoder clamped in - scoring the bound off the
                        // pre-rotation source misclassifies elements that rotation
                        // pushed past 3.775 (bound <= 0 = auto-violation)
                        const bound = @as(f64, @abs(tv)) - 3.775 + 1.0;
                        if (ef > @max(bound, 0)) {
                            j.g3_clmp += 1;
                            if (ef > j.g3_worst) {
                                j.g3_worst = ef;
                                j.g3_row = rr;
                                j.g3_col = cc;
                                j.g3_bound = bound;
                            }
                        }
                    } else if (ef > 2.0) { // the ruling's own arithmetic: the max Gaussian target clamped leaves ~1.9 units, so the unclamped Viterbi bound gets the same 2.0 ceiling
                        j.g3_uncl += 1;
                        if (ef > j.g3_worst) {
                            j.g3_worst = ef;
                            j.g3_row = rr;
                            j.g3_col = cc;
                            j.g3_bound = 2.0;
                        }
                    }
                }
            }
        }
    }
};

const ProbeCtx = struct {
    base: ProbeJob = .{},
    jobs: []ProbeJob = &.{},
    p_sq: []f64 = &.{},
    p_worst: []f64 = &.{},
    p_wr: []u32 = &.{},
    p_wc: []u32 = &.{},
    p_clamp: []u32 = &.{},
    p_col: []f64 = &.{},
    p_col_all: []f64 = &.{},
    kblks: usize = 0,

    fn entry(ctx: *const anyopaque, b0: usize, b1: usize, idx: usize) void {
        const c: *const ProbeCtx = @ptrCast(@alignCast(ctx));
        const j = &c.jobs[idx];
        j.* = c.base; // inputs shared, partials re-zeroed from the template
        j.p_sq = c.p_sq[idx * c.kblks ..][0..c.kblks];
        j.p_worst = c.p_worst[idx * c.kblks ..][0..c.kblks];
        j.p_wr = c.p_wr[idx * c.kblks ..][0..c.kblks];
        j.p_wc = c.p_wc[idx * c.kblks ..][0..c.kblks];
        j.p_clamp = c.p_clamp[idx * c.kblks ..][0..c.kblks];
        j.p_col = c.p_col[idx * c.base.n ..][0..c.base.n];
        j.p_col_all = c.p_col_all[idx * c.base.n ..][0..c.base.n];
        j.run(b0 * 16, b1 * 16);
    }
};

/// The oracle decode split over whole 128-row block strips: rotation is block-local,
/// so strips of full-width blocks are independent; each worker owns its scratch.
const DqCtx = struct {
    w: efmt.Width = .{ .halves = 2 },
    tr: []const i16 = &.{},
    suh: []const f16 = &.{},
    svh: []const f16 = &.{},
    out: []f64 = &.{},
    n: usize = 0,
    words: usize = 0,

    fn exec(self: *const DqCtx, s0: usize, s1: usize) void {
        const tn = self.n / 16;
        const scratch = alloc.alloc(f64, 2 * 128 * self.n) catch @panic("oom");
        defer alloc.free(scratch);
        for (s0..s1) |s| {
            const row0 = s * 128;
            const strip = self.tr[(row0 / 16) * tn * self.words ..][0..8 * tn * self.words];
            efmt.dequantize(self.w, .mul1, strip, 8, tn, self.suh[row0 ..][0..128], self.svh, self.out[row0 * self.n ..][0..128 * self.n], scratch);
        }
    }
};

fn dqChunk(ctx: *const anyopaque, s0: usize, s1: usize, _: usize) void {
    const c: *const DqCtx = @ptrCast(@alignCast(ctx));
    c.exec(s0, s1);
}

/// The qf -> chained-decode build (G1a): the elementwise phases split over rows; the
/// rotations stay whole-plane (block-local inside, no ordering question).
const QfCtx = struct {
    qf: []f64 = &.{},
    strip: []const f16 = &.{},
    suh: []const f16 = &.{},
    svh: []const f16 = &.{},
    n: usize = 0,
    phase: usize = 0, // 0 convert, 1 *= suh, 2 *= svh

    fn exec(self: *const QfCtx, r0: usize, r1: usize) void {
        for (r0..r1) |i| {
            const base = i * self.n;
            switch (self.phase) {
                0 => for (0..self.n) |jj| {
                    self.qf[base + jj] = @floatCast(self.strip[base + jj]);
                },
                1 => {
                    const s: f64 = @as(f64, @floatCast(self.suh[i]));
                    for (0..self.n) |j| self.qf[base + j] *= s;
                },
                else => for (0..self.n) |jj| {
                    self.qf[base + jj] *= @as(f64, @floatCast(self.svh[jj]));
                },
            }
        }
    }
};

fn qfChunk(ctx: *const anyopaque, r0: usize, r1: usize, _: usize) void {
    const c: *const QfCtx = @ptrCast(@alignCast(ctx));
    c.exec(r0, r1);
}

/// The target build (buildWqWin) split over rows: every element independent until the
/// whole-plane rotateBoth, which stays serial behind the join.
const WqCtx = struct {
    wq: []f64 = &.{},
    t: []const u8 = &.{},
    su: []const i8 = &.{},
    svh: []const f16 = &.{},
    zero: []const bool = &.{},
    k: usize = 0,
    n: usize = 0,
    n0: usize = 0,

    fn exec(self: *const WqCtx, r0: usize, r1: usize) void {
        for (r0..r1) |kk| {
            const s: f64 = @floatFromInt(self.su[kk]);
            for (0..self.n) |j| {
                if (self.zero[j]) {
                    self.wq[kk * self.n + j] = 0;
                    continue;
                }
                const idx = ((self.n0 + j) * self.k + kk) * 2;
                const bits: u32 = @as(u32, self.t[idx]) | (@as(u32, self.t[idx + 1]) << 8);
                const v: f64 = @as(f64, @floatCast(@as(f32, @bitCast(bits << 16)))); // bf16 = top half of fp32
                self.wq[kk * self.n + j] = v * s / @as(f64, @floatCast(self.svh[j]));
            }
        }
    }
};

fn wqChunk(ctx: *const anyopaque, r0: usize, r1: usize, _: usize) void {
    const c: *const WqCtx = @ptrCast(@alignCast(ctx));
    c.exec(r0, r1);
}

/// tr(deltaW^T H' deltaW) for one or two finished trellises at real size: the same fp64
/// arithmetic as core.exl3_ldlq_ref.proxyLoss (whose size guard protects the single-
/// threaded test path, not the formula), threaded over whole column tiles. Unpacks through
/// the oracle's own exl3_format.unpack; both runs score under the slot's true H' (the
/// T-LDLQ-1 convention, exl3_ldlq_ref.zig:292-294).
fn proxyLosses(w: efmt.Width, bld: *const ldl.BlockLdl, tr_a: []const i16, tr_b: []const i16,
               wq64: []const f64, k: usize, n: usize) struct { a: f64, b: ?f64 } {
    const got_a = alloc.alloc(f16, k * n) catch @panic("oom");
    defer alloc.free(got_a);
    efmt.unpack(w, .mul1, tr_a, k / 16, n / 16, got_a);
    var got_b: []f16 = &.{};
    if (tr_b.len > 0) {
        got_b = alloc.alloc(f16, k * n) catch @panic("oom");
        efmt.unpack(w, .mul1, tr_b, k / 16, n / 16, got_b);
    }
    defer if (got_b.len > 0) alloc.free(got_b);
    const nt: usize = n / 16; // whole column tiles per worker
    const nthreads = quantThreads(nt);
    const tiles_per = (nt + nthreads - 1) / nthreads;
    const jobs = alloc.alloc(ProxyJob, nthreads) catch @panic("oom");
    defer alloc.free(jobs);
    const threads = alloc.alloc(std.Thread, nthreads) catch @panic("oom");
    defer alloc.free(threads);
    var spawned: usize = 0;
    var made: usize = 0;
    while (made < nthreads) : (made += 1) {
        const t0 = made * tiles_per;
        if (t0 >= nt) break;
        const t1 = @min(nt, t0 + tiles_per);
        jobs[made] = .{
            .got_a = got_a, .got_b = got_b, .wq = wq64, .l = bld.l, .d = bld.d,
            .k = k, .n = n, .lo = t0 * 16, .hi = t1 * 16,
        };
        if (made + 1 == nthreads or t0 + tiles_per >= nt) { // the last working job runs inline
            jobs[made].run();
            made += 1;
            break;
        }
        threads[spawned] = std.Thread.spawn(.{}, proxyJobRun, .{&jobs[made]}) catch {
            jobs[made].run(); // no thread available: the range runs inline, the total holds
            continue; // the loop's made += 1 is the only increment — a manual one here double-steps
        };
        spawned += 1; // made advances via the loop's continue clause alone
    }
    for (threads[0..spawned]) |t| t.join();
    var a: f64 = 0;
    var b: f64 = 0;
    for (jobs[0..made]) |*j| {
        a += j.loss_a;
        b += j.loss_b;
    }
    return .{ .a = a, .b = if (tr_b.len > 0) b else null };
}

// ---------------------------------------------------------------- the fragments

/// One tensor of a fragment safetensors shard: the §4.5 entry set is trellis/suh/svh/mul1
/// per part, stock entry names — the reader (Checkpoint.addShard / admitGroup /
/// projectionExl3) admits the fragment verbatim, which is the T-LDLQ-5 bundle gate.
const FragT = struct {
    name: []const u8,
    dtype: []const u8,
    shape: []const usize,
    payload: []const u8,
};

fn writeFragment(path: []const u8, ts: []const FragT) !void {
    var head: std.ArrayList(u8) = .empty;
    try head.append(alloc, '{');
    var offb: usize = 0;
    for (ts, 0..) |t, i| {
        if (i != 0) try head.append(alloc, ',');
        var dims: std.ArrayList(u8) = .empty;
        try dims.append(alloc, '[');
        for (t.shape, 0..) |d, di| {
            if (di != 0) try dims.append(alloc, ',');
            try dims.appendSlice(alloc, try fmtStr("{d}", .{d}));
        }
        try dims.append(alloc, ']');
        const end = offb + t.payload.len;
        try head.appendSlice(alloc, try fmtStr("\"{s}\":{{\"dtype\":\"{s}\",\"shape\":{s},\"data_offsets\":[{d},{d}]}}", .{
            t.name, t.dtype, dims.items, offb, end,
        }));
        offb = end;
    }
    try head.append(alloc, '}');
    const padded = (head.items.len + 8) / 8 * 8;
    var prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &prefix, @intCast(padded), .little); // the padded JSON length (eaq_write.zig:152 convention), not 8 + it
    var file: std.ArrayList(u8) = .empty;
    try file.appendSlice(alloc, &prefix);
    try file.appendSlice(alloc, head.items);
    try file.appendNTimes(alloc, ' ', padded - head.items.len);
    for (ts) |t| try file.appendSlice(alloc, t.payload);
    try writeFile(path, file.items);
}

// ---------------------------------------------------------------- the progress bookkeeping

pub const ProgressLayer = struct { i: usize, sha: [64]u8, frag: []const u8, identity_fallback: bool = false };

pub const Progress = struct {
    // the parameters a resume must match: a mismatch refuses rather than mix a half-quantized
    // work dir into a different run
    src: []const u8 = "",
    calib_sha: [64]u8 = @splat(0),
    rows: u32 = 0,
    cols: u32 = 0,
    bits: u32 = 0,
    head_bits: u32 = 0,
    seed: u64 = 0,
    lambda: f64 = 0,
    scale: f64 = 0,
    layers: std.ArrayList(ProgressLayer) = .empty,
    head_sha: ?[64]u8 = null,
    stream_sha: ?[64]u8 = null, // S5: the hash of the newest saved stream file
    stream_i: usize = 0, // which boundary that stream is (S_i)
    head_idfb: bool = false, // P1-4: the head's fin was the L = I fallback — a resumed run must still see the pack gate close
    plan_sha: ?[64]u8 = null, // --plan provenance (spec 3.2): a resume under a different plan refuses

    fn filePath(work: []const u8) ![]u8 {
        return fmtStr("{s}/progress.json", .{work});
    }

    fn toJson(p: *const Progress) ![]u8 {
        var aw: std.Io.Writer.Allocating = .init(alloc);
        var w = &aw.writer;
        try w.print("{{\"src\":\"{s}\",\"calib_sha\":\"{s}\",\"rows\":{d},\"cols\":{d},\"bits\":{d},\"head_bits\":{d},\"seed\":{d},\"lambda\":{d},\"scale\":{d},\"layers\":[", .{
            p.src, p.calib_sha, p.rows, p.cols, p.bits, p.head_bits, p.seed, p.lambda, p.scale,
        });
        for (p.layers.items, 0..) |l, i| {
            if (i != 0) try w.writeByte(',');
            try w.print("{{\"i\":{d},\"sha\":\"{s}\",\"frag\":\"{s}\",\"idfb\":{s}}}", .{ l.i, l.sha, l.frag, if (l.identity_fallback) "true" else "false" });
        }
        try w.writeAll("],\"head\":");
        if (p.head_sha) |hs| try w.print("\"{s}\"", .{hs}) else try w.writeAll("null");
        try w.print(",\"head_idfb\":{s}", .{if (p.head_idfb) "true" else "false"});
        try w.writeAll(",\"stream_sha\":");
        if (p.stream_sha) |ss| try w.print("\"{s}\"", .{ss}) else try w.writeAll("null");
        try w.print(",\"stream_i\":{d}", .{p.stream_i});
        try w.writeAll(",\"plan_sha\":");
        if (p.plan_sha) |ps| try w.print("\"{s}\"", .{ps}) else try w.writeAll("null");
        try w.writeAll("}");
        return aw.toOwnedSlice();
    }

    pub fn save(p: *const Progress, work: []const u8) !void { // pub for the host save-then-load test
        const txt = try p.toJson();
        defer alloc.free(txt);
        const pp = try filePath(work);
        defer alloc.free(pp);
        // tmp+rename: a crash mid-write leaves the previous good file, not a truncated one
        const tmp = try fmtStr("{s}.tmp", .{pp});
        defer alloc.free(tmp);
        try writeFile(tmp, txt);
        try renameAt(tmp, pp);
    }

    pub fn load(work: []const u8) !?Progress { // pub for the host save-then-load test
        const pp = try filePath(work);
        defer alloc.free(pp);
        if (!fileExists(pp)) return null;
        const txt = try readFile(pp);
        const parsed = try std.json.parseFromSlice(std.json.Value, alloc, txt, .{});
        if (parsed.value != .object) return error.BadCalib;
        const o = parsed.value.object;
        const gs = struct {
            fn s(map: std.json.ObjectMap, key: []const u8) ![]const u8 {
                const v = map.get(key) orelse return error.BadCalib;
                if (v != .string) return error.BadCalib;
                return v.string;
            }
            fn num(map: std.json.ObjectMap, key: []const u8) !u64 {
                const v = map.get(key) orelse return error.BadCalib;
                return switch (v) {
                    .integer => |ii| @intCast(ii),
                    .string => |st| std.fmt.parseInt(u64, st, 10) catch error.BadCalib,
                    else => error.BadCalib,
                };
            }
            fn flt(map: std.json.ObjectMap, key: []const u8) !f64 {
                const v = map.get(key) orelse return error.BadCalib;
                return switch (v) {
                    .float => |f| f,
                    .integer => |ii| @floatFromInt(ii),
                    else => error.BadCalib,
                };
            }
            fn hex(map: std.json.ObjectMap, key: []const u8) ![64]u8 {
                // the stored form is the 64-CHARACTER hex TEXT (sha256Hex/bytesToHex output),
                // kept as text: decoding it here would compare 32 raw bytes against a hex
                // string and refuse every resume (the B1 bug)
                const hx = try s(map, key);
                if (hx.len != 64) return error.BadCalib;
                for (hx) |ch| switch (ch) { '0'...'9', 'a'...'f', 'A'...'F' => {}, else => return error.BadCalib };
                var out: [64]u8 = undefined;
                @memcpy(&out, hx);
                return out;
            }
            fn boo(map: std.json.ObjectMap, key: []const u8) bool {
                const v = map.get(key) orelse return false;
                return v == .bool and v.bool;
            }
        };
        var p: Progress = .{
            .src = try alloc.dupe(u8, try gs.s(o, "src")),
            .calib_sha = try gs.hex(o, "calib_sha"),
            .rows = @intCast(try gs.num(o, "rows")),
            .cols = @intCast(try gs.num(o, "cols")),
            .bits = @intCast(try gs.num(o, "bits")),
            .head_bits = @intCast(try gs.num(o, "head_bits")),
            .seed = try gs.num(o, "seed"),
            .lambda = try gs.flt(o, "lambda"),
            .scale = try gs.flt(o, "scale"),
        };
        const lv = o.get("layers") orelse return error.BadCalib;
        if (lv != .array) return error.BadCalib;
        var list: std.ArrayList(ProgressLayer) = .empty;
        for (lv.array.items) |item| {
            if (item != .object) return error.BadCalib;
            const lo = item.object;
            try list.append(alloc, .{
                .i = @intCast(try gs.num(lo, "i")),
                .sha = try gs.hex(lo, "sha"),
                .frag = try alloc.dupe(u8, try gs.s(lo, "frag")),
                .identity_fallback = gs.boo(lo, "idfb"),
            });
        }
        p.layers = list;
        if (o.get("head")) |hv| if (hv == .string) {
            p.head_sha = try gs.hex(o, "head");
        };
        p.head_idfb = gs.boo(o, "head_idfb");
        if (o.get("stream_sha")) |sv| if (sv == .string and sv.string.len == 64) {
            var out: [64]u8 = undefined;
            for (sv.string) |ch| switch (ch) { '0'...'9', 'a'...'f', 'A'...'F' => {}, else => return error.BadCalib };
            @memcpy(&out, sv.string);
            p.stream_sha = out;
        };
        p.stream_i = @intCast(gs.num(o, "stream_i") catch 0);
        if (o.get("plan_sha")) |pv| if (pv == .string and pv.string.len == 64) {
            var out: [64]u8 = undefined;
            for (pv.string) |ch| switch (ch) { '0'...'9', 'a'...'f', 'A'...'F' => {}, else => return error.BadCalib };
            @memcpy(&out, pv.string);
            p.plan_sha = out;
        };
        return p;
    }
};

fn streamPath(work: []const u8, i: usize) ![]u8 {
    return fmtStr("{s}/stream_{d}.f32", .{ work, i });
}

fn fragPathLayer(work: []const u8, i: usize) ![]u8 {
    return fmtStr("{s}/frag_layer{d}.safetensors", .{ work, i });
}

// ---------------------------------------------------------------- the quantizer

pub const PartWork = struct { // one part, kept from quantize until its fragment is written (pub for the host merge test)
    spec: PartPlan,
    win_n: u32 = 0, // the ENCODED width: the head streams windows narrower than the part
    tr: []i16, // the part's trellis i16 [k/16, win_n/16, words] (owned)
    suh: []f16, // the slot's row signs as fp16 (shared across the slot's parts)
    svh: []f16, // this part's fp16(sv * s * g) (owned)
    halves: u16,
    rec: PartRecord,
    stats: ?[]ldlq.TileStat = null, // the kernel's per-tile stats, kept for the T-LDLQ-5 RED dossier (owned)
    eta: ?[]f32 = null, // the eta plane (proxy parts), kept for the RED dossier (owned)
    tr_i: ?[]i16 = null, // the L = I twin trellis (single-layer proxy), kept for the RED dossier (owned)
    qf: ?[]f16 = null, // the final-code decode strips (proxy parts), read at capture (owned)
    what: ?[]f16 = null, // encodeWhat's WHAT as recorded (proxy parts), for the G1 dossier, (owned)
    zc: ?[]const bool = null, // the zero/subnormal column mask (owned)
    src_hash: ?[64]u8 = null, // sha256 over the source weight bytes at load (capture invariant)
};

const SiteFin = struct {
    h: []f64, // the captured H through spec 2.4: symmetric fp64, signs flipped, rotated, damped
    ldl: ldl.BlockLdl,
    ldl_i: ?ldl.BlockLdl = null, // the built L = I twin (proxy comparisons only)
    su: []i8,
    suh: []f16,
    identity: bool, // the --allow-identity-fallback L = I (never publishable)
    mu: f64,
    gen: u64 = 0, // P0-2: fresh identity token per finished factorization (finalizeSite)

    /// Frees the finalize copies. The fragment writers and the probe run before the owner's
    /// scope ends (runLayer's defer), so the borrowed suh is still valid at every use.
    fn deinit(self: *SiteFin) void {
        alloc.free(self.h);
        alloc.free(self.su);
        alloc.free(self.suh);
        alloc.free(self.ldl.l);
        alloc.free(self.ldl.d);
        if (self.ldl_i) |*bi| {
            alloc.free(bi.l);
            alloc.free(bi.d);
        }
    }
};

/// Stitch the head's window trellises into the full-part buffer: each chunk's tr is
/// [k/16, win_n/16, words] and lands at the running column-tile base. Pure so the
/// two-window synthetic test can drive it without a device.
pub fn mergeWindowTrellises(chunks: []const PartWork, k: usize, nt_total: usize, words: usize) ![]i16 { // pub for the two-window synthetic test
    const tr_full = try alloc.alloc(i16, (k / 16) * (nt_total / 16) * words);
    errdefer alloc.free(tr_full);
    var base: u32 = 0;
    for (chunks) |ch| {
        const cnt: usize = ch.win_n / 16;
        for (0..k / 16) |blk| {
            const src = ch.tr[(blk * cnt) * words ..][0..cnt * words];
            const dst_i = (blk * (nt_total / 16) + base / 16) * words;
            @memcpy(tr_full[dst_i..][0..cnt * words], src);
        }
        base += ch.win_n;
    }
    return tr_full;
}

pub const Quantizer = struct {
    o: Options,
    d: *Dev,
    m: *Model,
    a: *fwd.Acts,
    c: *Checkpoint,
    slabs: calib.Slabs,
    cap: calib.Capture,
    stream: *anyopaque,
    rowbuf: *anyopaque, // the capture pass's throwaway x_out (one row block of fp32 [cols, D])
    ids_dev: *anyopaque,
    ids: []const u32,
    hbuf: [5]?*anyopaque = @splat(null),
    drivers: [5][17]?*ldlq.Driver = @splat(@splat(null)), // one driver per slot AND width (the plan may mix halves)
    plan: ?core.exl3_alloc.Plan = null, // --plan (or the synthesized uniform plan)
    plan_map: std.StringHashMapUnmanaged(u8) = .{}, // PartPlan.base -> halves
    records: std.ArrayList(PartRecord) = .empty,
    done: std.ArrayList(usize) = .empty, // layers recorded complete in progress
    head_done: bool = false,
    prog: ?Progress = null, // the live progress state being accumulated
    any_identity_fallback: bool = false,
    gen_ctr: u64 = 0, // P0-2: monotonic SiteFin identity source — stack frames get reused, gens do not
    single_layer: bool = false, // set when the run covers exactly one layer: the T-LDLQ-4 comparisons
    stream_valid: bool = false, // the device stream holds the stream of the last completed boundary
    calib_sha: [64]u8 = @splat(0),

    fn siteK(site: usize) u32 {
        return switch (site) {
            0, 2, 4 => D,
            1 => 6144,
            3 => FF,
            else => unreachable,
        };
    }

    fn slabOf(self: *Quantizer, site: usize) *anyopaque {
        return switch (site) {
            0 => self.slabs.mixer_in,
            1 => self.slabs.mixer_out,
            2 => self.slabs.gu,
            3 => self.slabs.down,
            else => unreachable,
        };
    }

    fn dl(d: *Dev, p: *anyopaque, dst: []u8) !void {
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dst.ptr, p, dst.len, null, 0, null));
        try d.sync();
    }

    /// The run-wide widest part width per site — the same table estimateBytes uses at
    /// :1939-1945. A driver must be sized from this, never from the layer that happened
    /// to arrive first: layer 0 is GDN (site 0 at 10240) but layer 3's attention q_proj
    /// needs 12288, and an under-sized stat/scratch set is the B3 overflow.
    fn siteRunNMax(self: *Quantizer, site: usize) u32 {
        _ = self;
        return switch (site) {
            0 => 12288, // attn q_proj, wider than the GDN's 10240
            1 => D,
            2 => FF,
            3 => D,
            4 => head_window,
            else => unreachable,
        };
    }

    fn drvOf(self: *Quantizer, site: usize, n_max: u32, halves: u8) !*ldlq.Driver {
        if (self.drivers[site][halves]) |dr| return dr; // cached: sized by the width-aware prepass
        const w: efmt.Width = .{ .halves = halves };
        const dr = try alloc.create(ldlq.Driver);
        dr.* = try ldlq.Driver.init(self.d, w, siteK(site), @max(n_max, self.siteRunNMax(site)), 32); // count 32: the measured strip-mode optimum (604.7 tiles/s; cost footprint past it)
        if (self.o.proxy and site < 4) dr.want_eta = true; // the eta strips feed the D-ratio
        if (self.o.proxy and site < 4) dr.want_what = true; // the WHAT strips name the violating tiles (probe-down ruling)
        self.drivers[site][halves] = dr;
        return dr;
    }

    /// --plan (allocator contract 3.2): parse the allocator's plan JSON and verify its
    /// plan_sha against the canonical serialization — or synthesize the degenerate uniform
    /// plan from --bits/--head-bits. Part widths then look up by PartPlan.base; the sha
    /// rides the progress record (a resume under another plan refuses) and the §Q3 report.
    fn loadPlan(self: *Quantizer) !void {
        const al = core.exl3_alloc;
        var pw: std.ArrayList(al.PartWidth) = .empty;
        var method: []const u8 = "uniform";
        var target: f64 = @floatFromInt(self.o.bits);
        var head_halves: u8 = @intCast(2 * self.o.head_bits);
        var declared: ?[32]u8 = null;
        if (self.o.plan_path.len > 0) {
            const txt = try readFile(self.o.plan_path);
            defer alloc.free(txt);
            const pj = try std.json.parseFromSliceLeaky(std.json.Value, alloc, txt, .{ .ignore_unknown_fields = true });
            if (pj != .object) return error.BadPlan;
            const tv = pj.object.get("target_bpw") orelse return error.BadPlan;
            target = switch (tv) { .float => |f| f, .integer => |ii| @floatFromInt(ii), else => return error.BadPlan };
            const hv = pj.object.get("head_halves") orelse return error.BadPlan;
            head_halves = switch (hv) { .integer => |ii| @intCast(ii), else => return error.BadPlan };
            const mv = pj.object.get("method") orelse return error.BadPlan;
            method = switch (mv) { .string => |s| s, else => return error.BadPlan };
            const pv = pj.object.get("parts") orelse return error.BadPlan;
            if (pv != .array) return error.BadPlan;
            for (pv.array.items) |item| {
                if (item != .object) return error.BadPlan;
                const bv = item.object.get("base") orelse return error.BadPlan;
                const base = switch (bv) { .string => |s| s, else => return error.BadPlan };
                const hnv = item.object.get("halves") orelse return error.BadPlan;
                const h: u8 = switch (hnv) { .integer => |ii| @intCast(ii), else => return error.BadPlan };
                if (h < 2 or h > 16) return error.BadPlan;
                try pw.append(alloc, .{ .base = try alloc.dupe(u8, base), .halves = h });
            }
            const sv = pj.object.get("plan_sha") orelse return error.BadPlan;
            const hx = switch (sv) { .string => |s| s, else => return error.BadPlan };
            if (hx.len != 64) return error.BadPlan;
            var d: [32]u8 = @splat(0);
            for (hx, 0..) |ch, ci| {
                const v: u8 = std.fmt.charToDigit(ch, 16) catch return error.BadPlan;
                d[ci / 2] = @as(u8, d[ci / 2] << 4) | v;
            }
            declared = d;
        } else {
            for (0..LAYERS) |i| {
                const slots = try layerSlots(alloc, i);
                for (0..4) |s| for (slots[s].parts) |pp|
                    try pw.append(alloc, .{ .base = try alloc.dupe(u8, pp.base), .halves = @intCast(2 * self.o.bits) });
            }
            const sp = try headSlotPlan(alloc);
            for (sp.parts) |pp| try pw.append(alloc, .{ .base = try alloc.dupe(u8, pp.base), .halves = head_halves });
        }
        var p = al.Plan{ .target_bpw = target, .head_halves = head_halves, .method = method, .parts = pw.items, .plan_sha = undefined };
        p.plan_sha = try al.planShaWith(alloc, p);
        if (declared) |want| if (!std.mem.eql(u8, &want, &p.plan_sha)) {
            std.debug.print("quant: {s}: plan_sha \"{s}\" does not match the canonical serialization (\"{s}\") — refusing\n", .{ self.o.plan_path, std.fmt.bytesToHex(want, .lower), std.fmt.bytesToHex(p.plan_sha, .lower) });
            return error.BadPlan;
        };
        self.plan = p;
        for (p.parts) |part| try self.plan_map.put(alloc, part.base, part.halves);
        std.debug.print("quant: plan {s} method {s} target {d} bpw, head {d} halves, {d} parts, sha {s}\n", .{
            if (self.o.plan_path.len > 0) self.o.plan_path else "(uniform)", method, target, head_halves, p.parts.len, std.fmt.bytesToHex(p.plan_sha, .lower),
        });
    }


    // ------------------------------------------------------------- the capture pass

    /// The dense slabs for one layer's four slots from the bf16 source (spec 1.4:
    /// uploadDense logs the sub-6.1e-5 source counts per tensor).
    fn uploadDenseSlot(self: *Quantizer, sp: *const SlotPlan, slab: *anyopaque) !void {
        for (sp.parts) |pp| {
            const t = try self.c.get(try fmtStr("{s}.weight", .{pp.base}));
            if (!std.mem.eql(u8, t.dtype, "BF16")) return error.UnexpectedDtype;
            if (t.rank != 2 or t.shape[0] != pp.n or t.shape[1] != sp.k) return error.UnexpectedShape;
            try calib.uploadDense(self.d, t.bytes, pp.n, sp.k, sp.nt, pp.n0, slab);
        }
    }

    /// One layer's capture pass (spec 1.2): dense slabs, then layerForward per row with
    /// fresh = true (H1..H3: every row starts from zero state; H6: the rows leave fully
    /// summed, so the body's first norm skips the add). The outputs land in the throwaway
    /// rowbuf — the capture pass feeds nothing forward.
    fn capturePass(self: *Quantizer, i: usize, sp: *[4]SlotPlan) !void {
        const d = self.d;
        for (0..4) |s| {
            try self.uploadDenseSlot(&sp[s], self.slabOf(s));
            const kk = siteK(s);
            try d.fill(self.hbuf[s].?, @as(f32, 0), @as(usize, kk) * kk * 4);
        }
        self.cap.rows = @splat(0);
        const T = self.o.cols;
        for (0..self.o.rows) |r| {
            const bo: usize = @as(usize, r) * T * D * 4;
            // the runner's prompt batching (q.chunkEnd): a token batch runs at most
            // maxRows() rows; GDN state is zeroed on the first chunk of the row only
            var at: u32 = 0;
            while (at < T) {
                const end: u32 = @intCast(q.chunkEnd(at, T));
                try calib.layerForward(d, self.m, self.a, i, self.slabs,
                    off(self.stream, bo + @as(usize, at) * D * 4), off(self.rowbuf, @as(usize, at) * D * 4),
                    end - at, at, at == 0, &self.cap);
                at = end;
            }
        }
        for (0..4) |s| {
            const want = @as(u64, self.o.rows) * T;
            if (self.cap.rows[s] != want) {
                std.debug.print("quant: layer {d} site {d}: captured {d} rows, expected {d} — aborting\n", .{ i, s, self.cap.rows[s], want });
                return error.Usage;
            }
        }
    }

    /// The head capture (spec 4.5's last act): the final norm of the stream into fp16
    /// hidden, row by row, with cap.h[4] set — finalNorm's site-4 hook is the T-CAP
    /// head-pass twin; the head's own dequant slab is the KL tool's job, not ours.
    fn captureHead(self: *Quantizer) !void {
        const d = self.d;
        const kk = siteK(4);
        try d.fill(self.hbuf[4].?, @as(f32, 0), @as(usize, kk) * kk * 4);
        self.cap.rows[4] = 0;
        const T = self.o.cols;
        for (0..self.o.rows) |r| {
            const bo: usize = @as(usize, r) * T * D * 4;
            var at: u32 = 0;
            while (at < T) {
                const end: u32 = @intCast(q.chunkEnd(at, T));
                try calib.finalNorm(d, self.m, self.a, off(self.stream, bo + @as(usize, at) * D * 4),
                    off(self.rowbuf, @as(usize, at) * D * 2), end - at, &self.cap);
                at = end;
            }
        }
        if (self.cap.rows[4] != @as(u64, self.o.rows) * T) return error.Usage;
    }

    /// The apply pass (spec 1.3): the dequantized slabs are already in place; layerForward
    /// per row in place over the stream, fresh per row, no capture hooks.
    fn applyPass(self: *Quantizer, i: usize) !void {
        const d = self.d;
        const T = self.o.cols;
        for (0..self.o.rows) |r| {
            const bo: usize = @as(usize, r) * T * D * 4;
            var at: u32 = 0;
            while (at < T) {
                const end: u32 = @intCast(q.chunkEnd(at, T));
                const co = bo + @as(usize, at) * D * 4;
                try calib.layerForward(d, self.m, self.a, i, self.slabs, off(self.stream, co), off(self.stream, co),
                    end - at, at, at == 0, null);
                at = end;
            }
        }
    }

    // ------------------------------------------------------------- finalize (2.4) and LDL (3)

    /// Download, mirror, divide, finalize and factorize one site: the spec 2.4 sequence
    /// (finiteness first — a non-finite captured H is the forward poisoning a row, a bug
    /// signal naming layer and site, never papered over; flipSigns(su); rotateBoth; damping
    /// lambda*mu) and the section 3 blockLdl. The fp32 capture goes to the work dir (and to
    /// --save-h when given) before any of it, so a red run leaves the evidence. On a
    /// non-positive pivot the damped matrix is dumped to <out>/failed_H with the JSON
    /// sidecar (spec 3.2); the run aborts unless --allow-identity-fallback is set, in which
    /// case L = I, D = blockdiag(H') encodes and the pack gate refuses the pack at publish.
    fn finalizeSite(self: *Quantizer, tag: []const u8, site: usize, layer_i: ?usize) !SiteFin {
        const d = self.d;
        const k = siteK(site);
        const h32 = try alloc.alloc(f32, @as(usize, k) * k);
        defer alloc.free(h32);
        try dl(d, self.hbuf[site].?, std.mem.sliceAsBytes(h32));
        if (self.o.save_h) |dir| {
            try mkdirp(dir);
            const p2 = if (layer_i) |li|
                try fmtStr("{s}/layer{d}_site{d}.f32", .{ dir, li, site })
            else
                try fmtStr("{s}/head_site{d}.f32", .{ dir, site });
            defer alloc.free(p2);
            try writeFile(p2, std.mem.sliceAsBytes(h32));
        }
        // the syrk wrote the lower tiles; mirror them into a symmetric fp64 H and divide by
        // the captured rows (spec 2.1's accumulator is the plain sum over real rows)
        const rows: f64 = @as(f64, @floatFromInt(self.cap.rows[site]));
        const h64 = try alloc.alloc(f64, @as(usize, k) * k);
        errdefer alloc.free(h64);
        for (0..k) |r| for (0..k) |cc| {
            h64[r * k + cc] = if (r >= cc)
                @as(f64, h32[r * k + cc]) / rows
            else
                @as(f64, h32[cc * k + r]) / rows;
        };
        for (h64) |v| if (!std.math.isFinite(v)) {
            std.debug.print("quant: {s} site {d}: the captured H holds a non-finite value — the forward poisoned a row (spec 2.4 step 2)\n", .{ tag, site });
            return error.Usage;
        };
        const su = try drawSigns(slotSeed(self.o.seed, layer_i orelse LAYERS, site, null), k);
        errdefer alloc.free(su);
        ldl.flipSigns(h64, k, su);
        ldl.rotateBoth(h64, k, k);
        var mu: f64 = 0;
        for (0..k) |x| mu += h64[x * k + x];
        mu /= @as(f64, @floatFromInt(k));
        var dead: usize = 0;
        var dmin: f64 = std.math.inf(f64);
        var dmax: f64 = -std.math.inf(f64);
        for (0..k) |x| {
            const v = h64[x * k + x];
            if (v < 1e-6 * mu) dead += 1;
            dmin = @min(dmin, v);
            dmax = @max(dmax, v);
        }
        for (0..k) |x| h64[x * k + x] += self.o.lambda * mu; // spec 2.4 step 4
        std.debug.print("quant: {s} site {d}: K {d} mu {e} lambda*mu {e} diag [{e},{e}] dead {d} ({d:.2} pct)\n", .{
            tag, site, k, mu, self.o.lambda * mu, dmin, dmax, dead, @as(f64, @floatFromInt(dead)) * 100 / @as(f64, @floatFromInt(k)),
        });
        const suh = try alloc.alloc(f16, k);
        errdefer alloc.free(suh);
        for (suh, su) |*s, sign| s.* = @floatFromInt(sign);
        var pivot: usize = undefined;
        var fin: SiteFin = undefined;
        if (ldl.blockLdl(alloc, h64, k, self.o.lambda, mu, &pivot)) |got| {
            fin = .{ .h = h64, .ldl = got, .ldl_i = null, .su = su, .suh = suh, .identity = false, .mu = mu };
        } else |e| {
            if (e != error.NotPositive) return e;
            // spec 3.2: dump the matrix and the sidecar, print the verdict
            const dir = try fmtStr("{s}/failed_H", .{self.o.out});
            try mkdirp(dir);
            const raw = if (layer_i) |li|
                try fmtStr("{s}/layer{d}_site{d}.f64", .{ dir, li, site })
            else
                try fmtStr("{s}/head_site{d}.f64", .{ dir, site });
            defer alloc.free(raw);
            try writeFile(raw, std.mem.sliceAsBytes(h64));
            const side = try fmtStr("{s}.json", .{raw});
            defer alloc.free(side);
            const sj = try fmtStr(
                \\{{"tag":"{s}","site":{d},"k":{d},"lambda":{d},"mu":{e},"pivot":{d},"pivot_block":{d},"rows":{d},"diag_min":{e},"diag_max":{e}}}
            , .{ tag, site, k, self.o.lambda, mu, pivot, pivot / 16, self.cap.rows[site], dmin, dmax });
            try writeFile(side, sj);
            std.debug.print("quant: {s} site {d}: the damped rotated H failed at pivot {d} (block {d}) — dumped to {s}\n", .{ tag, site, pivot, pivot / 16, raw });
            if (!self.o.allow_identity_fallback) return error.CholeskyFailed;
            std.debug.print("quant: {s} site {d}: --allow-identity-fallback — encoding with L = I, D = blockdiag(H') (never for a shipped pack)\n", .{ tag, site });
            self.any_identity_fallback = true;
            const id = try identityLdl(h64, k);
            fin = .{ .h = h64, .ldl = id, .ldl_i = null, .su = su, .suh = suh, .identity = true, .mu = mu };
        }
        self.gen_ctr += 1; // P0-2: a finished fin is never confused with a later stack twin
        fin.gen = self.gen_ctr;
        if (self.o.proxy and !fin.identity and site != 4) {
            // the T-LDLQ-4 twin, built once per slot: the comparison encodes this part set with L = I
            var f2 = fin;
            f2.ldl_i = try identityLdl(h64, k);
            return f2;
        }
        return fin;
    }

    // ------------------------------------------------------------- W_q (4.2 step 3)

    /// The column RMS of the stored part over rows n0..n0+n: s[j] = sqrt(mean_k
    /// W[n0+j,k]^2) (spec 4.2 step 2; the head streams its columns in windows, layers
    /// pass n0 = 0 and the full width).
    fn colRmsWin(t: []const u8, k: usize, n: usize, n0: usize, s: []f32) void {
        for (0..n) |j| {
            var sum: f64 = 0;
            for (0..k) |kk| {
                const idx = ((n0 + j) * k + kk) * 2;
                const bits: u32 = @as(u32, t[idx]) | (@as(u32, t[idx + 1]) << 8);
                const v: f64 = @as(f64, @floatCast(@as(f32, @bitCast(bits << 16)))); // bf16 = top half of fp32
                sum += v * v;
            }
            s[j] = @floatCast(@sqrt(sum / @as(f64, @floatFromInt(k))));
        }
    }

    /// The part's W_q fp64 [k, n] for the window n0..n0+n: f64(W[n,k]) flipped by su along
    /// k and divided by the stored svh[j] (the s*g column scale carries the sv signs —
    /// svh enters the target; the H is already in the su domain, spec 4.2 steps 2-4), then
    /// rotateBoth: the host fp64 FWHT pair, provably the oracle's pre-rotation. Columns
    /// flagged in zero (zero/subnormal source RMS, S1) enter as exact zeros so the divisor
    /// is never consulted and the rotation cannot spread a 0/0 NaN across the 128 block.
    fn buildWqWin(wq: []f64, t: []const u8, k: usize, n: usize, n0: usize, su: []const i8, svh: []const f16, zero: []const bool) void {
        const c = WqCtx{ .wq = wq, .t = t, .su = su, .svh = svh, .zero = zero, .k = k, .n = n, .n0 = n0 };
        chunkRun(&c, k, chunkSplit(k, k), wqChunk); // rows are independent until the rotateBoth below (the machine's full core count, EAQ_LDL_THREADS override)
        ldl.rotateBoth(wq, k, n);
    }

    // ------------------------------------------------------------- part quantization

    /// The part's width: the plan's word by PartPlan.base, else the uniform --bits default.
    fn halvesOf(self: *Quantizer, site: usize, base: []const u8) u8 {
        if (self.plan_map.get(base)) |h| return h;
        return @intCast(2 * (if (site == 4) self.o.head_bits else self.o.bits));
    }

    /// One part (or one head window of a part): the caller draws sv and s (per-part PRNG
    /// seed hash(run_seed, layer, slot, part), spec 4.1) and passes the slices; this builds
    /// svh, W_q, runs the driver's loop, and fills the §Q3 record — with --proxy also the
    /// 4.6 loss and the D-ratio, and on a single-layer run the T-LDLQ-4 comparison (the
    /// same part encoded with L = I, both losses under the true H', LDLQ strictly lower and
    /// different, or the run stops red). The trellis lands in pw.tr (the fragment payload).
    fn quantizePartFull(
        self: *Quantizer,
        site: usize,
        layer_i: ?usize,
        sp: *const SlotPlan,
        fin: *SiteFin,
        pp: PartPlan,
        sv: []const i8,
        s: []const f32,
        n_win: u32, // the windowed n (the head streams; layers pass pp.n)
        n0_win: u32, // the window's row base in the stored part (column base for the head)
        svh: []f16, // the n-wide store-scale buffer (the caller owns it; the fragment borrows it)
        pw: *PartWork,
    ) !void {
        const d = self.d;
        _ = layer_i;
        const w: efmt.Width = .{ .halves = self.halvesOf(site, pp.base) };
        const k: usize = sp.k;
        const n: usize = n_win;
        const base_name = pp.base;
        const t = try self.c.get(try fmtStr("{s}.weight", .{pp.base}));
        if (!std.mem.eql(u8, t.dtype, "BF16")) return error.UnexpectedDtype;
        if (t.rank != 2 or t.shape[1] != k or t.shape[0] < n0_win + n) return error.UnexpectedShape;
        const words = w.tileWords();
        var zero_cols: usize = 0;
        const zc = try alloc.alloc(bool, n); // owned by pw.zc: the probe's zero-column mask, freed in runLayer's cleanup
        for (svh, 0..) |*o, j| {
            const sj: f64 = @as(f64, s[j]);
            // S1: a dead column has no usable scale — svh = sv (exact) and the W_q column
            // enters as zero, never the 0/0 that would poison the whole 128-column block
            if (!std.math.isFinite(sj) or !(sj > 1e-30)) {
                zero_cols += 1;
                zc[j] = true;
                o.* = @floatFromInt(sv[j]);
                continue;
            }
            zc[j] = false;
            const v: f32 = @as(f32, @floatFromInt(sv[j])) * s[j] * @as(f32, @floatCast(self.o.scale));
            o.* = @floatCast(v);
        }
        const wq = try alloc.alloc(f64, k * n);
        defer alloc.free(wq);
        buildWqWin(wq, t.bytes, k, n, n0_win, fin.su, svh, zc);
        for (wq) |x| if (!std.math.isFinite(x)) {
            std.debug.print("quant: {s}: the W_q target holds a non-finite value — the column scales are broken; refusing the upload\n", .{base_name});
            return error.Usage;
        };
        if (zero_cols != 0) std.debug.print("quant: {s}: {d} zero/subnormal columns encoded as zeros\n", .{ base_name, zero_cols });
        const tr = try alloc.alloc(i16, (k / 16) * (n / 16) * words);
        errdefer alloc.free(tr);
        const results = try alloc.alloc(ldlq.TileStat, (k / 16) * (n / 16)); // owned by pw.stats once the part lands; a mid-part error aborts the run anyway
        const dr = try self.drvOf(site, @intCast(n), @intCast(w.halves)); // halves <= 16: the plan guard already bounds it
        const part = ldlq.Part{ .n = @intCast(n), .n0 = n0_win, .sv = sv, .s = s, .svh = svh };
        var parts = [_]ldlq.Part{part};
        const slot = ldlq.Slot{ .k = @intCast(k), .su = fin.su, .suh = fin.suh, .ldl = &fin.ldl, .gen = fin.gen, .parts = &parts };
        try dr.beginSlot(d, &slot);
        const rep = try dr.quantizePart(d, &slot, &part, wq, tr, results);

        var rec = PartRecord{ .prefix = base_name, .k = @intCast(k), .n = @intCast(n), .halves = w.halves };
        rec.tiles = rep.tiles;
        rec.optimal = rep.optimal;
        rec.closed = rep.closed;
        rec.failed = rep.failed;
        rec.clamped = rep.clamped;
        rec.eta_sq = rep.eta_sq;
        rec.identity_fallback = fin.identity;
        var items: std.ArrayList(TileItem) = .empty;
        const kblks = k / 16;
        var ti: usize = 0;
        for (0..kblks) |tk| {
            for (0..n / 16) |tn| {
                const st = results[ti];
                ti += 1;
            if (st.status > 2) return error.Usage; // corrupted stat word: fail closed (ldlq.zig:236)
            rec.gap_hist[gapBucket40(st.cost -% st.open_cost)] += 1;
            rec.pruned1 += @as(u32, @intCast(st.flags & 1));
            rec.pruned2 += @as(u32, @intCast((st.flags >> 1) & 1));
            if (st.flags != 0 or st.status == 2) { // S2: itemise only pruned or failed tiles — heuristic closes are counted, not listed
                const reason: []const u8 = if (st.flags & 1 != 0)
                    "pruned-pass1"
                else if (st.flags & 2 != 0)
                    "pruned-pass2"
                else
                    "failed"; // status 1 with clean flags is a counted heuristic close, not an item
                try items.append(alloc, .{
                    .prefix = base_name, .tk = @intCast(tk), .tn = @intCast(n0_win / 16 + tn),
                    .encoder = if (st.status == 1) "xpu-kernel+heuristic" else if (st.status == 2) "none" else "xpu-kernel",
                    .status = st.status, .flags = st.flags, .cost = st.cost,
                    .open_cost = st.open_cost, .pinned = st.pinned, .reason = reason,
                });
            }
            }
        }
        rec.items = items.items;
        if (rec.failed != 0) {
            for (items.items) |it| if (it.status == 2)
                std.debug.print("quant: {s}: tile (tk {d}, tn {d}) FAILED cost {d} open {d} flags {x}\n", .{ base_name, it.tk, it.tn, it.cost, it.open_cost, it.flags });
            std.debug.print("quant: {s}: {d} failed tiles of {d} — the pack cannot be trusted (T-LDLQ-4: zero failed required)\n", .{ base_name, rec.failed, rec.tiles });
            return error.TileFailed;
        }
        const clamp_frac = @as(f64, @floatFromInt(rec.clamped)) / @as(f64, rec.tiles * 256);
        if (clamp_frac >= 1e-4) {
            std.debug.print("quant: {s}: clamp fraction {e:.3} is above 1e-4 — investigate g before trusting this part\n", .{ base_name, clamp_frac });
            if (self.o.proxy and site == 3) return error.ClampHigh; // the T-LDLQ-4 gate is the DOWN slot's (spec 418); other parts record the fraction and continue, as the full run does
        }
        std.debug.print("quant: {s}: tiles {d} (optimal {d}, closed {d}) clamped {d} ({e:.2}) eta_sq {e}\n", .{ base_name, rec.tiles, rec.optimal, rec.closed, rec.clamped, clamp_frac, rec.eta_sq });

        var keep_tr_i: ?[]i16 = null;
        var keep_eta: ?[]f32 = null;
        var keep_qf: ?[]f16 = null;
        var keep_what: ?[]f16 = null;
        if (self.o.proxy and !fin.identity and site != 4) { // the head has no L = I twin and no eta plane (L1)
            const losses = proxyLosses(w, &fin.ldl, tr, &.{}, wq, k, n);
            rec.proxy = losses.a;
            // the encoder-side planes are read BEFORE the L = I twin re-encode overwrites
            // the driver's qf/eta/what planes — the gates must see the SHIPPED part's codes
            const eta = try alloc.alloc(f32, k * n); // owned by pw.eta (the RED dossier reads the feedback at the worst element)
            keep_eta = eta;
            try dr.readEta(d, eta, @intCast(k), @intCast(n));
            keep_qf = try alloc.alloc(f16, k * n); // the final-code decode and the encoder's WHAT
            try dr.readQf(d, keep_qf.?, @intCast(k), @intCast(n));
            keep_what = try alloc.alloc(f16, k * n);
            try dr.readWhat(d, keep_what.?, @intCast(k), @intCast(n));
            if (self.single_layer) {
                const tr_i = try alloc.alloc(i16, tr.len); // owned by pw.tr_i (the RED dossier re-dequantizes this element from the twin)
                const res_i = try alloc.alloc(ldlq.TileStat, results.len);
                defer alloc.free(res_i);
                var slot_i = slot;
                slot_i.ldl = &fin.ldl_i.?;
                slot_i.gen = fin.gen + 1; // P0-2: the L = I twin is its own factorization, not the shipped one's
                keep_tr_i = tr_i;
                try dr.beginSlot(d, &slot_i);
                _ = try dr.quantizePart(d, &slot_i, &part, wq, tr_i, res_i);
                try dr.beginSlot(d, &slot); // restore the shipped L for whatever comes next
                const li = proxyLosses(w, &fin.ldl, tr_i, &.{}, wq, k, n);
                rec.proxy_identity = li.a;
                const gain = (li.a - losses.a) / li.a; // the LDLQ gain (spec 4.6's number)
                std.debug.print("quant: {s}: T-LDLQ-4 loss(LDLQ) {e} loss(L=I) {e} gain {d:.3} pct\n", .{ base_name, losses.a, li.a, gain * 100 });
                var trd: f64 = 0;
                for (0..k / 16) |b| {
                    for (0..16) |ai| {
                        trd += fin.ldl.d[b * 256 + ai * 16 + ai];
                    }
                }
                var trh: f64 = 0;
                for (0..k) |x| trh += fin.h[@as(usize, x) * (k + 1)];
                std.debug.print("quant: {s}: T-LDLQ-4 1-gain {d:.6} vs tr(D)/tr(H) {d:.6} (tr D {e}, tr H {e})\n", .{ base_name, 1 - gain, trd / trh, trd, trh });
                if (!(losses.a < li.a)) {
                    std.debug.print("quant: {s}: the LDLQ run is NOT below the L = I run — T-LDLQ-4 red\n", .{base_name});
                    return error.ProxyOrdering;
                }
                if (std.mem.eql(i16, tr, tr_i)) {
                    std.debug.print("quant: {s}: the L = I encode produced an identical trellis — L never reached the loop\n", .{base_name});
                    return error.ProxyOrdering;
                }
            }
            var dsum: f64 = 0;
            for (0..k / 16) |b| {
                for (0..16) |ai| dsum += fin.ldl.d[b * 256 + ai * 16 + ai];
            }
            const md = dsum / @as(f64, @floatFromInt(k));
            rec.dratio = ldlq_ref.etaWeighted(fin.ldl.d, eta, k, n) / (md * rec.eta_sq);
        }

        pw.* = .{ .spec = pp, .win_n = n_win, .tr = tr, .suh = fin.suh, .svh = svh, .halves = w.halves, .rec = rec, .stats = results, .eta = keep_eta, .tr_i = keep_tr_i, .qf = keep_qf, .what = keep_what, .zc = zc, .src_hash = sha256Hex(t.bytes) };
    }

    /// The T-LDLQ-5 RED forensics (Main 2026-10-25): the worst element's full trail —
    /// (row, col, k-block, strip), the source bf16 weight, the pre-rotation encoder target
    /// (the buildWqWin mirror), the rotated target through the 128-block FWHT, the kernel's
    /// clamp decision at |T| > 4, the tile stat and trellis words, the device reconstruction,
    /// the host dequant of the same trellis, the eta feedback at the element and its column
    /// tail, the same element from the L = I twin, and the sample counts over 5e-3 / 2e-3
    /// of the global RMS. Prints only on a red metric: a green part behaves exactly as before.
    fn dumpWorstElement(
        self: *Quantizer,
        site: usize,
        sp: *const SlotPlan,
        pw: *const PartWork,
        su: []const i8,
        row: usize,
        col: usize,
        ref_rms: f32,
    ) !void {
        const d = self.d;
        const n: usize = pw.spec.n;
        const k: usize = sp.k;
        const w: efmt.Width = .{ .halves = pw.halves };
        const words = w.tileWords();
        const t = try self.c.get(try fmtStr("{s}.weight", .{pw.spec.base}));
        const srcAt = struct {
            fn f(tb: []const u8, kk: usize, jj: usize, kd: usize) f64 { // bf16 at stored (row kk, tensor-local column jj) - column-major (buildWqWin); the slab's n0 must NOT be folded in here
                const idx = (jj * kd + kk) * 2;
                const b: u32 = @as(u32, tb[idx]) | (@as(u32, tb[idx + 1]) << 8);
                return @as(f64, @floatCast(@as(f32, @bitCast(b << 16))));
            }
        }.f;
        const src = srcAt(t.bytes, row, col, k); // tensor-local column: pw.spec.n0 is the FUSED-SLAB column base (sp.nt pitch), not an offset inside this part's own tensor - the encoder encodes it from column 0 (quantizePartFull's n0_win)
        const hd: usize = 128; // hadamard_dim: rotateBoth's blocks are 128 on both axes, so the element needs its whole block
        const rb0 = (row / hd) * hd;
        const cb0 = (col / hd) * hd;
        const blk = try alloc.alloc(f64, hd * hd);
        defer alloc.free(blk);
        const zcm = pw.zc;
        for (0..hd) |rr| for (0..hd) |cc| {
            const kk = rb0 + rr;
            const jj = cb0 + cc;
            if (zcm) |z| if (z[jj]) { // buildWqWin zeroes dead columns before the rotate
                blk[rr * hd + cc] = 0;
                continue;
            };
            const v = srcAt(t.bytes, kk, jj, k); // tensor-local, as above
            blk[rr * hd + cc] = v * @as(f64, @floatFromInt(su[kk])) / @as(f64, @floatCast(pw.svh[jj]));
        };
        const pre = blk[(row - rb0) * hd + (col - cb0)];
        ldl.rotateBoth(blk, hd, hd);
        const rot = blk[(row - rb0) * hd + (col - cb0)];
        const clamped = @abs(rot) > 4.0;
        const tgt: f64 = if (!clamped) rot else if (rot > 0) 4 else -4;
        const tk = row / 16;
        const tn = col / 16;
        const kblks = k / 16;
        const strip: usize = if (tk < kblks) tk / (1 + (kblks - 1) / 32) else 0; // the driver's 32 strips (quantizePartFull:1444's measured optimum)
        const trows: usize = 128; // the rotation's block: a tile-level decode must carry its whole 128x128 neighbourhood (a (1,1) call silently skips rotate — 128/128 = 0 blocks)
        const tk0 = (tk / 8) * 8;
        const tn0 = (tn / 8) * 8;
        const ttr = try alloc.alloc(i16, 8 * 8 * words);
        defer alloc.free(ttr);
        for (0..8) |a| for (0..8) |b_| {
            const src_t = pw.tr[((tk0 + a) * (n / 16) + tn0 + b_) * words ..][0..words];
            @memcpy(ttr[(a * 8 + b_) * words ..][0..words], src_t);
        };
        var outb: [128 * 128]f64 = undefined;
        var scr: [2 * 128 * 128]f64 = undefined;
        efmt.dequantize(w, .mul1, ttr, 8, 8, pw.suh[tk0 * 16 ..][0..trows], pw.svh[tn0 * 16 ..][0..trows], &outb, &scr);
        const host_recon = outb[(row % 128) * 128 + (col % 128)];
        var li_recon: ?f64 = null;
        if (pw.tr_i) |ti| {
            for (0..8) |a| for (0..8) |b_| {
                const src_t = ti[((tk0 + a) * (n / 16) + tn0 + b_) * words ..][0..words];
                @memcpy(ttr[(a * 8 + b_) * words ..][0..words], src_t);
            };
            efmt.dequantize(w, .mul1, ttr, 8, 8, pw.suh[tk0 * 16 ..][0..trows], pw.svh[tn0 * 16 ..][0..trows], &outb, &scr);
            li_recon = outb[(row % 128) * 128 + (col % 128)];
        }
        var eta_at: ?f32 = null;
        var eta_max_col: f32 = 0;
        var eta_sum_col: f64 = 0;
        if (pw.eta) |et| {
            eta_at = et[row * n + col];
            for (0..k) |kk| {
                const e = @abs(et[kk * n + col]);
                eta_max_col = @max(eta_max_col, e);
                eta_sum_col += et[kk * n + col] * et[kk * n + col];
            }
        }
        var over_5e3: u64 = 0;
        var over_2e3: u64 = 0;
        var dev_bits: ?u16 = null;
        {
            const rows: usize = 128;
            const wref = try alloc.alloc(u16, rows * n);
            defer alloc.free(wref);
            const out64 = try alloc.alloc(f64, rows * n);
            defer alloc.free(out64);
            const scratch64 = try alloc.alloc(f64, 2 * rows * n);
            defer alloc.free(scratch64);
            for ([_]usize{ 0, k - 128 }) |row0| {
                for (0..rows) |r| {
                    const byte_off = (@as(usize, row0 + r) * sp.nt + pw.spec.n0) * 2;
                    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, wref[r * n ..][0..n].ptr,
                        off(self.slabOf(site), byte_off), n * 2, null, 0, null));
                }
                try d.sync();
                const trellis = pw.tr[(row0 / 16) * (n / 16) * words ..][0..(rows / 16) * (n / 16) * words];
                efmt.dequantize(w, .mul1, trellis, rows / 16, n / 16, pw.suh[row0..][0..rows], pw.svh, out64, scratch64);
                for (0..rows) |r| for (0..n) |cc| {
                    const ref: f32 = @floatCast(out64[r * n + cc]);
                    const gotv: f32 = @as(f32, @as(f16, @bitCast(wref[r * n + cc])));
                    const rel = @as(f64, @abs(ref - gotv)) / @max(@as(f64, ref_rms), 1e-30);
                    if (rel > 5e-3) over_5e3 += 1;
                    if (rel > 2e-3) over_2e3 += 1;
                };
                if (row >= row0 and row < row0 + rows) dev_bits = wref[(row - row0) * n + col];
            }
        }
        var trbuf: std.Io.Writer.Allocating = .init(alloc);
        defer trbuf.deinit();
        for (pw.tr[(tk * (n / 16) + tn) * words ..][0..words], 0..) |code, wi| {
            if (wi != 0) trbuf.writer.print(", ", .{}) catch {};
            trbuf.writer.print("{d}", .{code}) catch {};
        }
        // the dossier coordinates are band-resident by construction (g3 scores only the
        // rows-0..15 block; the G1 scan samples the two 128-row bands), so dev_bits
        // resolves for every element this dossier can be called on
        const bits = dev_bits orelse 0;
        std.debug.print("quant: T-LDLQ-5 {s}: RED dossier for the worst element (row {d}, col {d}): k-block tk {d} of {d} (strip {d} of 32), tile ({d}, {d})\n", .{ pw.spec.base, row, col, tk, kblks, strip, tk, tn });
        std.debug.print("quant: T-LDLQ-5 {s}:   source bf16 {e}; pre-rotation target {e}; rotated target {e} (initial-target mirror: exact only in the feedback-free block; the eta plane below carries the encoder's feedback); clamp |T|>4 {s} -> encoded target {e}; su {d}, svh {e}\n", .{
            pw.spec.base, src, pre, rot, if (clamped) "FIRED" else "no", tgt, su[row], pw.svh[col],
        });
        if (pw.stats) |sx| {
            const st = sx[tk * (n / 16) + tn];
            std.debug.print("quant: T-LDLQ-5 {s}:   tile status {d} flags {x} cost {d} open {d} pinned {d}; trellis words [{s}]\n", .{ pw.spec.base, st.status, st.flags, st.cost, st.open_cost, st.pinned, trbuf.written() });
        } else std.debug.print("quant: T-LDLQ-5 {s}:   trellis words [{s}]\n", .{ pw.spec.base, trbuf.written() });
        std.debug.print("quant: T-LDLQ-5 {s}:   device slab bits {x} (f16 {e}); host dequant of the same trellis {e}; L = I twin {s}{e}\n", .{
            pw.spec.base, bits, @as(f32, @as(f16, @bitCast(bits))), host_recon,
            if (li_recon) |_| "re-dequant " else "twin absent ", if (li_recon) |v| v else 0.0,
        });
        if (pw.what) |wh| if (pw.qf) |qfp| { // the target the Viterbi searched vs the shipped code
            const wv: f32 = @floatCast(wh[row * n + col]);
            const qv2: f32 = @floatCast(qfp[row * n + col]);
            std.debug.print("quant: T-LDLQ-5 {s}:   encoder WHAT {e} vs shipped qf {e} at (row {d}, col {d})\n", .{ pw.spec.base, wv, qv2, row, col });
        };
        std.debug.print("quant: T-LDLQ-5 {s}:   eta at (row, col) {s}; column eta max {e}, column eta RMS {e}; sampled elements over 5e-3 of RMS: {d}, over 2e-3: {d}\n", .{
            pw.spec.base, if (eta_at) |e| std.fmt.allocPrint(alloc, "{e}", .{e}) catch "?(fmt)" else "unavailable (no eta plane)",
            eta_max_col, @sqrt(eta_sum_col / @as(f64, @floatFromInt(k))), over_5e3, over_2e3,
        });
    }

    /// The dequant probe (T-LDLQ-5): the fragment's parts back through the reader's own
    /// admission (exl3_model.projectionExl3) and the serving dequant into the slab
    /// (calib.dequantSlot — the bytes the serving prompt GEMM reads from wscr), then the
    /// probe-down ruling's gates, every one measured against the pack's own bytes or the
    /// capture-side source (never an encoder-internal buffer — that blindness is what the
    /// stale-WHAT bug slipped through):
    ///   bpw + source hash   the shipped width vs --bits; sha256 of the source weight bytes
    ///                       at probe time vs the load-time hash (capture invariance);
    ///   G1                  host efmt.dequantize of the packed trellis is the oracle; the
    ///                       encoder's own final decode qf_dev (proxy parts) must agree with
    ///                       it within 2 fp16 ulp on the sampled bands - HARD. The capture-
    ///                       time slab is scored and printed but INFORMATIONAL: it is
    ///                       decoded before the post-Viterbi clamp and tail-biting closure,
    ///                       so disagreement there is recorded staleness, not pack error
    ///                       (the published artifact is the trellis; load-time
    ///                       dequantSlot re-decodes it fresh under the exl3_format oracle);
    ///   G2                  E = |decode(pack) - source bf16| / |svh| in scale units on the
    ///                       feedback-free block (the first processed, j descending): both
    ///                       sides weight space (dequantize reconstructs W; scoring it
    ///                       against the rotated code target instead is a domain mismatch
    ///                       at ~sqrt(2)); RMS in [0.10, 0.16] — [0.10, 0.20] for the
    ///                       o_proj/out_proj head-strip family (g2-band ruling
    ///                       2026-10-27: K-per-column 624 vs 80, observed family drift
    ///                       0.1269..0.1852, ~3 sigma; hard RED above 0.20); the ruling's per-column
    ///                       [0.08, 0.20] count prints as a diagnostic; the column RED
    ///                       reads WHOLE-COLUMN ratios (all k rows of the file-decoded
    ///                       pack) with floor 0.06 — the 16-row window is diagnostic on
    ///                       the low side (nu = 16 noise puts a healthy column's window
    ///                       at ~5e-6 per column below 0.04, ~1 false gate per 11 parts;
    ///                       g2-colfloor ruling 2026-10-27) and stays RED above 0.40;
    ///                       eta_sq/(K*N) in [0.0156, 0.026] (proxy);
    ///                       absolute src/decoded/error RMS printed beside the ratios;
    ///   G3                  same block: unclamped elements |E| <= 2.0 (the ruling's own
    ///                       5.6-sigma-clamped-to-3.77 arithmetic leaves ~1.9 units);
    ///                       clamped (|T| > 3.775, the predicate behind the driver's own
    ///                       clamp rate and clamp_exp's 3.775-sigma tail) |E| <=
    ///                       |T| - 3.775 + 1; the clamp count vs the tail expectation
    ///                       (2x ceiling); 0.1-bin error histogram printed;
    ///   G4                  the last-processed block's RMS/worst and its ratio to G2 —
    ///                       reported, not gated; per-block table for all k/16 blocks;
    ///   corr                Pearson r of the pack's decode vs the source bf16 over the live
    ///                       columns of the whole part (weight space): >= 0.98 (the Shannon
    ///                       ceiling at the 0.16 RMS edge is 1/sqrt(1+0.16^2) = 0.9874, so
    ///                       the ruling's 0.99 predates the 3.0bpw budget; a domain-mixing
    ///                       bug still measures ~0); the decode-vs-rotated-target
    ///                       cross-domain r printed alongside, informational;
    ///   heavy tail          the top-1000 |source| elements' median |E| in scale units:
    ///                       <= 2.0 (LDLQ must protect the outliers first).
    /// SAMPLING: G1 uses the first and last 128-k-row blocks (the exl3OracleCheck
    /// convention); everything else scans the whole part. Any violation runs the RED
    /// dossier (dumpWorstElement) and stops the run.
    fn probeSlot(self: *Quantizer, layer_i: usize, site: usize, sp: *const SlotPlan, pws: []const PartWork, su: []const i8) !void {
        const d = self.d;
        var frag = Checkpoint{};
        defer frag.deinit();
        const fp = try fragPathLayer(self.o.work, layer_i);
        defer alloc.free(fp);
        const f0 = try alloc.dupeSentinel(u8, fp, 0);
        defer alloc.free(f0);
        try frag.addShard(f0);
        const base = try fmtStr("model.layers.{d}.", .{layer_i});
        var names: std.ArrayList([]const u8) = .empty;
        for (pws) |pw| try names.append(alloc, pw.spec.tail);
        const min_n: u32 = if (site == 0 and !q.isAttention(layer_i)) GDN_IN else 0;
        const proj = try exl3_model.projectionExl3(d, &frag, base, names.items, sp.wide, min_n, false);
        defer {
            for (proj.exl3.?.parts) |p| {
                d.g.free(p.tr);
                d.g.free(p.suh);
                d.g.free(p.svh);
            }
            alloc.free(proj.exl3.?.parts);
            alloc.destroy(proj.exl3.?);
        }
        try calib.dequantSlot(d, proj.exl3.?, sp.k, proj.n, self.slabOf(site));
        for (pws) |pw| {
            const w: efmt.Width = .{ .halves = pw.halves }; // the part's OWN width — the plan may mix them
            const k: usize = sp.k;
            const n: usize = pw.spec.n;
            const n0: usize = pw.spec.n0; // the fused-SLAB column base (sp.nt pitch) - slab addressing only; every source-tensor read below indexes the part's own tensor from its column 0, exactly as quantizePartFull's n0_win = 0 did
            const kblks = k / 16;
            const j0 = kblks - 1; // the FIRST processed block (j descends): the only feedback-free one

            // the ruling's bpw invariant: the shipped width against the requested one
            std.debug.print("quant: T-LDLQ-5 {s}: pack {d} halves ({d:.1} bpw), run requested {d} bpw\n", .{
                pw.spec.base, pw.halves, @as(f64, @floatFromInt(pw.halves)) / 2, self.o.bits,
            });

            // capture invariant: the bytes the part was encoded from must BE the bytes the
            // probe hashes now - the same tensor the fragment is checked against
            const t = try self.c.get(try fmtStr("{s}.weight", .{pw.spec.base}));
            if (pw.src_hash) |sh| {
                const h_now = sha256Hex(t.bytes);
                if (!std.mem.eql(u8, &h_now, &sh)) {
                    std.debug.print("quant: T-LDLQ-5 {s}: RED - the source tensor changed between capture and probe (at load {s}, at probe {s})\n", .{
                        pw.spec.base, std.fmt.bytesToHex(sh, .lower), std.fmt.bytesToHex(h_now, .lower),
                    });
                    return error.ProbeFailed;
                }
            }
            const zc = pw.zc orelse return error.ProbeFailed;

            // publish equality (Main 2026-10-26, the laundered-pack ruling): the oracle below
            // decodes the WRITTEN fragment's own bytes - and those bytes must BE the codes the
            // encoder produced. A file that differs from the in-memory plane at any point the
            // gates read is a pack the probe never verified: RED, and the run stops before
            // publish. The fragcheck referee proved file == encoder bytes on gates4's artifact;
            // this gate keeps that true by construction, every run, every part.
            const fbase = try packName(pw.spec.base);
            defer alloc.free(fbase);
            const ftr_t = try frag.get(try fmtStr("{s}.trellis", .{fbase}));
            const fsh_t = try frag.get(try fmtStr("{s}.suh", .{fbase}));
            const fsv_t = try frag.get(try fmtStr("{s}.svh", .{fbase}));
            const ftr: []const i16 = @alignCast(std.mem.bytesAsSlice(i16, ftr_t.bytes));
            const fsh: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, fsh_t.bytes));
            const fsv: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, fsv_t.bytes));
            if ((ftr.len != pw.tr.len or !std.mem.eql(i16, ftr, pw.tr)) or
                (fsh.len != pw.suh.len or !std.mem.eql(f16, fsh, pw.suh)) or
                (fsv.len != pw.svh.len or !std.mem.eql(f16, fsv, pw.svh)))
            {
                std.debug.print("quant: T-LDLQ-5 {s}: RED - the written fragment does not equal the encoder's plane (trellis {s} vs {s}; the pack launders bytes the probe never verified)\n", .{
                    pw.spec.base, std.fmt.bytesToHex(sha256Hex(ftr_t.bytes), .lower), std.fmt.bytesToHex(sha256Hex(std.mem.sliceAsBytes(pw.tr)), .lower),
                });
                return error.ProbeFailed;
            }
            // the host mirror of the encoder's target: the same buildWqWin (same su, the
            // fragment's own svh, the capture's zero-column mask) - T in scale units - and
            // ONE host decode of the packed trellis, the oracle every gate compares to
            const tt = try alloc.alloc(f64, k * n);
            defer alloc.free(tt);
            buildWqWin(tt, t.bytes, k, n, 0, su, pw.svh, zc); // tensor-local: the encoder built the real target with n0_win = 0 (runLayer's quantizePartFull call); pw.spec.n0 is the slab base
            const out64 = try alloc.alloc(f64, k * n);
            defer alloc.free(out64);
            const words = w.tileWords();
            const dqc = DqCtx{ .w = w, .tr = ftr, .suh = fsh, .svh = fsv, .out = out64, .n = n, .words = words }; // the FILE's bytes, equality-checked above - the probe scores the artifact the pack ships
            chunkRun(&dqc, k / 128, chunkSplit(k / 128, k / 128), dqChunk); // the whole-part oracle decode over 128-row strips (rotation-local, scratch-per-worker)

            // G1: the host oracle against the device's pack decode (the serving slab,
            // dequantSlot's bytes) and - proxy parts - against the encoder's own final
            // decode qf_dev. The bound is 2 fp16 ulp; the violators name (k, n, tile,
            // offset in tile), the closure/clamp signature the ruling demands.
            var viol_a: [8][3]u32 = undefined;
            var na_p: usize = 0;
            var na_all: usize = 0;
            var viol_b: [8][3]u32 = undefined;
            var nb_p: usize = 0;
            var nb_all: usize = 0;
            var vb_bits: [8]u16 = @splat(0);
            var max_ulp: u32 = 0;
            var wa_u: u32 = 0; // the worst-ulp encoder-decode violator (the dossier's coordinate)
            var wa_r: usize = 0;
            var wa_c: usize = 0;
            var qf64: ?[]f64 = null; // the encoder's final-code decode, chained into the slab's domain (rotate, suh, rotate, svh — the dequantize body's arithmetic) for G1a
            var qf_keep: ?[]f64 = null; // the chain lives to the part's last G1a read - a block-scoped defer would free it while qf64 still points there
            defer if (qf_keep) |p| alloc.free(p);
            if (pw.qf) |strip| {
                const qf6 = try alloc.alloc(f64, k * n);
                qf_keep = qf6;
                var qctx = QfCtx{ .qf = qf6, .strip = strip, .suh = pw.suh, .svh = pw.svh, .n = n, .phase = 0 };
                chunkRun(&qctx, k, chunkSplit(k, k), qfChunk);
                efmt.rotate(qf6, k, n, 0);
                qctx.phase = 1;
                chunkRun(&qctx, k, chunkSplit(k, k), qfChunk);
                efmt.rotate(qf6, k, n, 1);
                qctx.phase = 2;
                chunkRun(&qctx, k, chunkSplit(k, k), qfChunk);
                qf64 = qf6;
            }
            for ([_]usize{ 0, k - 128 }) |row0| { // the exl3OracleCheck sampling convention
                const rows: usize = 128;
                const wref = try alloc.alloc(u16, rows * n);
                defer alloc.free(wref);
                for (0..rows) |r| {
                    const byte_off = (@as(usize, row0 + r) * sp.nt + n0) * 2;
                    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, wref[r * n ..][0..n].ptr,
                        off(self.slabOf(site), byte_off), n * 2, null, 0, null));
                }
                try d.sync();
                for (0..rows) |r| for (0..n) |cc| {
                    const oracle: f16 = @floatCast(out64[(row0 + r) * n + cc]);
                    const rbits: u16 = @bitCast(oracle);
                    const rc: i32 = if (rbits & 0x8000 != 0) 0x8000 - @as(i32, rbits) else @intCast(rbits);
                    const gbits = wref[r * n + cc];
                    const gc: i32 = if (gbits & 0x8000 != 0) 0x8000 - @as(i32, gbits) else @intCast(gbits);
                    const u: u32 = @intCast(@abs(gc - rc));
                    if (u > max_ulp) max_ulp = u;
                    if (u > 2) {
                        nb_all += 1;
                        if (nb_p < viol_b.len) {
                            viol_b[nb_p] = .{ @intCast(row0 + r), @intCast(cc), u };
                            vb_bits[nb_p] = gbits;
                            nb_p += 1;
                        }
                    }
                    if (qf64) |chain| {
                        const qc16: f16 = @floatCast(chain[(row0 + r) * n + cc]);
                        const qbits: u16 = @bitCast(qc16);
                        const qc: i32 = if (qbits & 0x8000 != 0) 0x8000 - @as(i32, qbits) else @as(i32, qbits);
                        const v: u32 = @intCast(@abs(qc - rc));
                        if (v > 2) {
                            na_all += 1;
                            if (v > wa_u) {
                                wa_u = v;
                                wa_r = row0 + r;
                                wa_c = cc;
                            }
                            if (na_p < viol_a.len) {
                                viol_a[na_p] = .{ @intCast(row0 + r), @intCast(cc), v };
                                na_p += 1;
                            }
                        }
                    }
                };
            }
            std.debug.print("quant: T-LDLQ-5 {s}: G1 host oracle vs the encoder's final decode (hard): {d} violators >2 ulp; vs the serving slab (the device dequantSlot decode of THIS part's shipped fragment - fresh at read, the serving-kernel cross-check): {d} (max ulp {d})\n", .{ pw.spec.base, na_all, nb_all, max_ulp });
            for (viol_b[0..nb_p], vb_bits[0..nb_p]) |v, b| std.debug.print("quant: T-LDLQ-5 {s}:   slab(informational) violator (k,n) ({d},{d}) tile ({d},{d}) offset ({d},{d}) {d} ulp; slab {x} (f16 {e}) vs host decode {e}\n", .{
                pw.spec.base, v[0], v[1], v[0] / 16, v[1] / 16, v[0] % 16, v[1] % 16, v[2],
                b, @as(f32, @as(f16, @bitCast(b))), out64[@as(usize, v[0]) * n + @as(usize, v[1])],
            });
            for (viol_a[0..na_p]) |v| std.debug.print("quant: T-LDLQ-5 {s}:   encoder-decode violator (k,n) ({d},{d}) tile ({d},{d}) offset ({d},{d}) {d} ulp\n", .{
                pw.spec.base, v[0], v[1], v[0] / 16, v[1] / 16, v[0] % 16, v[1] % 16, v[2],
            });

            // G2/G3/G4 + the per-block table: E = the device's own strip_diff form,
            // f32(Q) - W32 in scale units, host-recomputed from the packed trellis
            var blk_sq = try alloc.alloc(f64, kblks);
            defer alloc.free(blk_sq);
            var blk_worst = try alloc.alloc(f64, kblks);
            defer alloc.free(blk_worst);
            var blk_wr = try alloc.alloc(u32, kblks);
            defer alloc.free(blk_wr);
            var blk_wc = try alloc.alloc(u32, kblks);
            defer alloc.free(blk_wc);
            var blk_clamp = try alloc.alloc(u32, kblks);
            defer alloc.free(blk_clamp);
            @memset(blk_sq, 0);
            @memset(blk_worst, 0);
            @memset(blk_wr, 0);
            @memset(blk_wc, 0);
            @memset(blk_clamp, 0);
            const col_sq = try alloc.alloc(f64, n); // the feedback-free block's per-column sums
            defer alloc.free(col_sq);
            @memset(col_sq, 0);
            const col_all_sq = try alloc.alloc(f64, n); // whole-column sums, all k rows of each live column
            defer alloc.free(col_all_sq);
            @memset(col_all_sq, 0);
            var hist: [31]u64 = @splat(0);
            var g3_uncl: usize = 0; // unclamped |E| over 2.0 (the Viterbi bound, with the closure allowance)
            var g3_clmp: usize = 0; // clamped |E| over |T| - 3.775 + 1 (the clamp recovery bound, rotated domain)
            var g3_worst: f64 = 0;
            var g3_row: usize = 0;
            var g3_col: usize = 0;
            var g3_bound: f64 = 0;
            var src_sq: f64 = 0;
            var dec_sq: f64 = 0;
            var err_abs_sq: f64 = 0; // the G2 block's absolute error energy (source units)
            var st: f64 = 0; // the pack-vs-source correlation, WEIGHT space (decode out64 vs source bf16): the >=0.98 signal
            var sq: f64 = 0;
            var stq: f64 = 0;
            var stt: f64 = 0;
            var sqq: f64 = 0;
            var rot_t: f64 = 0; // informational: decode-vs-rotated-target Pearson (cross-domain, expected ~0)
            var rot_tt: f64 = 0;
            var rot_tq: f64 = 0;
            var ht_src: [1000]f32 = undefined; // the heavy-tail census: top-1000 |source| elements
            var ht_err: [1000]f64 = undefined;
            var ht_n: usize = 0;
            var ht_min: usize = 0;
            const split = chunkSplit(kblks, kblks); // whole 16-row blocks per job (the jb index the table keys on)
            std.debug.print("quant: probe: {d} threads\n", .{split.njobs});
            const p_sq = try alloc.alloc(f64, split.njobs * kblks);
            defer alloc.free(p_sq);
            const p_worst = try alloc.alloc(f64, split.njobs * kblks);
            defer alloc.free(p_worst);
            const p_wr = try alloc.alloc(u32, split.njobs * kblks);
            defer alloc.free(p_wr);
            const p_wc = try alloc.alloc(u32, split.njobs * kblks);
            defer alloc.free(p_wc);
            const p_clamp = try alloc.alloc(u32, split.njobs * kblks);
            defer alloc.free(p_clamp);
            const p_col = try alloc.alloc(f64, split.njobs * n);
            defer alloc.free(p_col);
            @memset(p_sq, 0);
            @memset(p_worst, 0);
            @memset(p_wr, 0);
            @memset(p_wc, 0);
            @memset(p_clamp, 0);
            @memset(p_col, 0);
            const p_col_all = try alloc.alloc(f64, split.njobs * n);
            defer alloc.free(p_col_all);
            @memset(p_col_all, 0);
            const pjobs = try alloc.alloc(ProbeJob, split.njobs);
            defer alloc.free(pjobs);
            const pctx = ProbeCtx{
                .base = .{ .tt = tt, .out64 = out64, .zc = zc, .bytes = t.bytes, .svh = pw.svh, .k = k, .n = n, .n0 = 0, .j0 = j0 }, // n0 here indexes the part's OWN tensor (the source-bf16 read); the slab reads keep pw.spec.n0
                .jobs = pjobs,
                .p_sq = p_sq, .p_worst = p_worst, .p_wr = p_wr, .p_wc = p_wc, .p_clamp = p_clamp, .p_col = p_col, .p_col_all = p_col_all,
                .kblks = kblks,
            };
            chunkRun(&pctx, kblks, split, ProbeCtx.entry);
            for (pjobs[0..split.njobs]) |*jm| { // fold the partials in chunk order: sums add, worst entries merge by argmax, the census re-inserts in order
                for (0..kblks) |b| {
                    blk_sq[b] += jm.p_sq[b];
                    blk_clamp[b] += jm.p_clamp[b];
                    if (jm.p_worst[b] > blk_worst[b]) {
                        blk_worst[b] = jm.p_worst[b];
                        blk_wr[b] = jm.p_wr[b];
                        blk_wc[b] = jm.p_wc[b];
                    }
                }
                for (0..n) |cc| {
                    col_sq[cc] += jm.p_col[cc];
                    col_all_sq[cc] += jm.p_col_all[cc];
                }
                for (0..hist.len) |bi| hist[bi] += jm.hist[bi];
                g3_uncl += jm.g3_uncl;
                g3_clmp += jm.g3_clmp;
                if (jm.g3_worst > g3_worst) {
                    g3_worst = jm.g3_worst;
                    g3_row = jm.g3_row;
                    g3_col = jm.g3_col;
                    g3_bound = jm.g3_bound;
                }
                src_sq += jm.src_sq;
                dec_sq += jm.dec_sq;
                err_abs_sq += jm.err_abs_sq;
                st += jm.st;
                sq += jm.sq;
                stq += jm.stq;
                stt += jm.stt;
                sqq += jm.sqq;
                rot_t += jm.rot_t;
                rot_tt += jm.rot_tt;
                rot_tq += jm.rot_tq;
                for (0..jm.ht_n) |hi| {
                    const asrc = jm.ht_src[hi];
                    const aer = jm.ht_err[hi];
                    if (ht_n < ht_src.len) {
                        ht_src[ht_n] = asrc;
                        ht_err[ht_n] = aer;
                        if (ht_n == 0 or asrc < ht_src[ht_min]) ht_min = ht_n;
                        ht_n += 1;
                    } else if (asrc > ht_src[ht_min]) {
                        ht_src[ht_min] = asrc;
                        ht_err[ht_min] = aer;
                        var m: usize = 0;
                        for (ht_src, 0..) |v, ti| {
                            if (v < ht_src[m]) m = ti;
                        }
                        ht_min = m;
                    }
                }
            }
            var n_zero: usize = 0;
            for (zc) |dead| {
                if (dead) n_zero += 1;
            }
            const n_live = n - n_zero;
            const nn: f64 = @floatFromInt(k * n_live);
            const cov = nn * stq - st * sq;
            const den = @sqrt(@max(nn * stt - st * st, 0.0) * @max(nn * sqq - sq * sq, 0.0));
            const r_pack = if (den > 1e-30) cov / den else 0.0;
            const cov_rot = nn * rot_tq - sq * rot_t; // Pearson(qv, tv): decode vs the rotated target - cross-domain, informational
            const den_rot = @sqrt(@max(nn * rot_tt - rot_t * rot_t, 0.0) * @max(nn * sqq - sq * sq, 0.0));
            const r_rot = if (den_rot > 1e-30) cov_rot / den_rot else 0.0;
            var ht_med: f64 = 0;
            std.mem.sort(f64, ht_err[0..ht_n], {}, std.sort.asc(f64));
            if (ht_n != 0) ht_med = ht_err[ht_n / 2];
            const cnt_blk: f64 = @floatFromInt(16 * n_live);
            const j0_rms = @sqrt(blk_sq[j0] / cnt_blk);
            // G2 head-strip family (self_attn.o_proj / linear_attn.out_proj): the fable
            // ruling (eaq-review/g2-band-ruling-packet.md, 2026-10-27) lifts the block
            // ceiling to 0.20 — K-per-column 624 vs 80 makes the max-scale order
            // statistic lag the heavy-tailed error numerator; the 8-point series
            // 0.1269..0.1852 is a smooth family offset, ~mean+3sigma at 0.202. Above
            // 0.20 stays hard RED; the wide family keeps [0.10, 0.16] unchanged.
            const head_strip = std.mem.endsWith(u8, pw.spec.base, "o_proj") or
                std.mem.endsWith(u8, pw.spec.base, "out_proj");
            const g2_hi: f64 = if (head_strip) 0.20 else 0.16;
            var col_out: usize = 0; // the ruling's [0.08,0.20] count: diagnostic only, printed
            var col_first: usize = 0;
            var col_bad: usize = 0; // the gate: a 16-row per-column RMS estimate carries ~18 pct sampling noise, so the ruling's tight band is unmeetable by sampling alone at the 0.16 block-RMS ceiling - gate wide
            var col_badfirst: usize = 0;
            var col_lo: f64 = std.math.inf(f64);
            var col_hi: f64 = 0;
            var col_all_lo: f64 = std.math.inf(f64);
            var col_ship_bad: usize = 0; // whole-column ratio under the 0.06 collapse floor
            var col_ship_first: usize = 0;
            const krows: f64 = @floatFromInt(k);
            for (0..n) |cc| {
                if (zc[cc]) continue;
                const cr = @sqrt(col_sq[cc] / 16);
                col_lo = @min(col_lo, cr);
                col_hi = @max(col_hi, cr);
                if (cr < 0.08 or cr > 0.20) {
                    if (col_out == 0) col_first = cc;
                    col_out += 1;
                }
                if (cr > 0.40) { // high side only: the nu = 16 upper tail at 0.40 against a 0.135 population is ~1e-17 per column - safe as RED
                    if (col_bad == 0) col_badfirst = cc;
                    col_bad += 1;
                }
                const cr_all = @sqrt(col_all_sq[cc] / krows);
                col_all_lo = @min(col_all_lo, cr_all);
                if (cr_all < 0.06) { // the collapse floor: healthy whole-column minima measure 0.156 (54 parts); a zero-error column reads ~0
                    if (col_ship_bad == 0) col_ship_first = cc;
                    col_ship_bad += 1;
                }
            }
            const nelems: f64 = @floatFromInt(k * n);
            const src_rms_abs = @sqrt(src_sq / @as(f64, @floatFromInt(k * n_live)));
            const dec_rms_abs = @sqrt(dec_sq / @as(f64, @floatFromInt(k * n_live))); // dec_sq counts live elements only (the zc skip)
            const err_abs_rms = @sqrt(err_abs_sq / cnt_blk);
            const clamp_exp = @as(f64, @floatFromInt(16 * n)) * 3.172e-4; // the two-sided 3.775-sigma tail rate (the ruling's table; matches the driver's own 1.649e-4 qkv rate)
            var eta_ratio: ?f64 = null;
            if (pw.rec.eta_sq > 0) { // the device's own squared eta partials (the eta PLANE is a signed residual — summing it un-squared was the negative "eta" of the first RED)
                eta_ratio = pw.rec.eta_sq / nelems;
            }
            var eta_txt: [40]u8 = undefined;
            const eta_str: []const u8 = if (eta_ratio) |r| std.fmt.bufPrint(&eta_txt, "{d:.6}", .{r}) catch "?" else "-";
            std.debug.print("quant: T-LDLQ-5 {s}: G2 feedback-free block {d} (rows {d}..{d}): err RMS {d:.6} of scale (bounds [0.10,{d:.2}]); corr(pack, source) {d:.4} weight space (gate 0.98 - 1/sqrt(1+0.16^2)=0.9874 is the Shannon ceiling at the band's upper edge); corr(pack, rotated target) {d:.4} cross-domain (informational, expected ~0); heavy tail top-{d} median err {d:.3} scale units on |source| rank (bound 2.0); per-column window min {d:.4} max {d:.4}, outside [0.08,0.20]: {d} (first {d}); window-gated >0.40: {d} (first {d}); whole-col min {d:.4} gated <0.06: {d} (first {d}); eta_sq/(K*N) {s} (bounds [0.0156,0.026]); absolute RMS: src {e:.6}, decoded {e:.6}, block error {e:.6}\n", .{
                pw.spec.base, j0, k - 16, k - 1, j0_rms, g2_hi, r_pack, r_rot, ht_src.len, ht_med, col_lo, col_hi, col_out, col_first, col_bad, col_badfirst, col_all_lo, col_ship_bad, col_ship_first,
                eta_str, // the eta plane is the encoder's own per-element energy
                src_rms_abs, dec_rms_abs, err_abs_rms,
            });
            var histw: std.Io.Writer.Allocating = .init(alloc);
            defer histw.deinit();
            for (hist, 0..) |h, bi| {
                if (bi != 0) histw.writer.print(",", .{}) catch {};
                histw.writer.print("{d}", .{h}) catch {};
            }
            var bump: u64 = 0;
            for (hist[7..]) |h| bump += h;
            std.debug.print("quant: T-LDLQ-5 {s}: G2 hist {s} (31 bins of 0.1 over [0,3]; last bin >3.0); bump >0.6: {d} vs clamped in block {d} (expected {d:.1})\n", .{ pw.spec.base, histw.written(), bump, blk_clamp[j0], clamp_exp });
            std.debug.print("quant: T-LDLQ-5 {s}: G3 unclamped |E|>2.0: {d}; clamped beyond |T|-3.775+1: {d} (worst {d:.4} vs bound {d:.4} at ({d},{d}))\n", .{
                pw.spec.base, g3_uncl, g3_clmp, g3_worst, g3_bound, g3_row, g3_col,
            });
            const j0_last = @sqrt(blk_sq[0] / cnt_blk); // the LAST processed block: the most feedback consumed
            std.debug.print("quant: T-LDLQ-5 {s}: G4 last-processed block 0 (rows 0..15): RMS {d:.6}, worst {d:.4} at ({d},{d}), ratio to the feedback-free block {d:.3} (reported, not gated)\n", .{
                pw.spec.base, j0_last, blk_worst[0], blk_wr[0], blk_wc[0], j0_last / @max(j0_rms, 1e-30),
            });
            std.debug.print("quant: T-LDLQ-5 {s}: per-block (strip order, all {d}): block:RMS/worst@(row,col):clamp0 (the initial target's |T|>3.775 count; exact only in the feedback-free block)\n", .{ pw.spec.base, kblks });
            var bb: usize = 0;
            while (bb < kblks) : (bb += 12) {
                for (bb..@min(bb + 12, kblks)) |b| {
                    std.debug.print(" {d}:{d:.4}/{d:.2}@({d},{d}):{d}", .{ b, @sqrt(blk_sq[b] / cnt_blk), blk_worst[b], blk_wr[b], blk_wc[b], blk_clamp[b] });
                }
                std.debug.print("\n", .{});
            }

            // the gates (fable's probe-down ruling, re-calibrated on the 2026-10-25 layer-0
            // log): G1 hard-gates ONLY the encoder's own final decode against the host oracle -
            // the published artifact is the trellis and the capture-time slab is decoded before
            // the post-Viterbi clamp/tail-biting closure, so its violators are informational
            // (load-time dequantSlot re-decodes the fragment fresh; the exl3_format oracle
            // covers that path). G2 block RMS + eta; the ruling's per-column count prints as a
            // diagnostic and the gate runs on the wide [0.04,0.40] band (16-row estimator
            // noise); the corr gate is 0.98, the Shannon ceiling at the 0.16 RMS edge times a
            // margin, and a domain-mixing bug still measures ~0; G3 clamps on |T| > 3.775 with
            // the bound |T|-3.775+1, unclamped at 2.0 (the ruling's 5.6-sigma clamp arithmetic).
            const eta_bad = if (eta_ratio) |r| !(r >= 0.0156 and r <= 0.026) else false;
            const red = na_all != 0 or
                !(j0_rms >= 0.10 and j0_rms <= g2_hi) or col_bad != 0 or col_ship_bad != 0 or eta_bad or
                r_pack < 0.98 or ht_med > 2.0 or
                g3_uncl != 0 or g3_clmp != 0 or blk_clamp[j0] > @as(u32, @intFromFloat(2 * clamp_exp));
            if (red) {
                var drow: usize = g3_row; // dossier coordinates: an encoder-decode violator leads when G3 itself came clean
                var dcol: usize = g3_col;
                if (g3_uncl == 0 and g3_clmp == 0 and na_all != 0) {
                    drow = wa_r; // the worst-ulp encoder-decode violator, not the first seen
                    dcol = wa_c;
                }
                try self.dumpWorstElement(site, sp, &pw, su, drow, dcol, @floatCast(src_rms_abs)); // the red trail (Main 2026-10-25); green parts never reach it
                std.debug.print("quant: T-LDLQ-5 {s}: RED (G1 encoder-decode {d}, slab {d} informational, G2 block RMS {d:.6}, corr {d:.4}, tail med {d:.3}, window-gated {d} (diagnostic {d}), whole-col min {d:.4} gated {d}, eta {s}, G3 {d}+{d}, clamps {d}/{d:.1})\n", .{
                    pw.spec.base, na_all, nb_all, j0_rms, r_pack, ht_med, col_bad, col_out, col_all_lo, col_ship_bad, eta_str,
                    g3_uncl, g3_clmp, blk_clamp[j0], clamp_exp,
                });
                return error.ProbeFailed;
            }
            if (head_strip and j0_rms > 0.18 and r_pack < 0.99) // joint amber, ruling 3: corr and G2 share the outlier-column symptom — both near ceiling at once is a review marker, not an independent pass
                std.debug.print("quant: T-LDLQ-5 {s}: G2 JOINT AMBER: block RMS {d:.6} > 0.18 with corr {d:.4} < 0.99 — head-strip outliers, review at the pack self-check (g2-band ruling 2026-10-27)\n", .{ pw.spec.base, j0_rms, r_pack });
            std.debug.print("quant: T-LDLQ-5 {s}: GREEN\n", .{pw.spec.base});
        }
    }

    // ------------------------------------------------------------- fragments

    /// The layer's fragments from the quantized parts (stock names: base.trellis I16
    /// [K/16, N/16, 8*halves], base.suh F16 [K], base.svh F16 [N], base.mul1 I32 scalar —
    /// exactly what admitGroup accepts). Returns the sha256 of the written bytes.
    fn writeLayerFragment(self: *Quantizer, layer_i: usize, slots: *[4]SlotPlan, slot_pws: *[4][]PartWork) ![64]u8 {
        var ts: std.ArrayList(FragT) = .empty;
        var mul1: [4]u8 = undefined;
        std.mem.writeInt(u32, &mul1, efmt.Mul1Mul, .little);
        for (0..4) |s| for (slot_pws[s]) |pw| {
            const k = slots[s].k;
            const words = pw.halves * 8;
            const pk = try packName(pw.spec.base); // the reader reads the pack's stock names
            try ts.append(alloc, .{
                .name = try fmtStr("{s}.trellis", .{pk}),
                .dtype = "I16",
                .shape = try alloc.dupe(usize, &[_]usize{ k / 16, pw.spec.n / 16, words }),
                .payload = std.mem.sliceAsBytes(pw.tr),
            });
            try ts.append(alloc, .{
                .name = try fmtStr("{s}.suh", .{pk}),
                .dtype = "F16",
                .shape = try alloc.dupe(usize, &[_]usize{k}),
                .payload = std.mem.sliceAsBytes(pw.suh),
            });
            try ts.append(alloc, .{
                .name = try fmtStr("{s}.svh", .{pk}),
                .dtype = "F16",
                .shape = try alloc.dupe(usize, &[_]usize{pw.spec.n}),
                .payload = std.mem.sliceAsBytes(pw.svh),
            });
            try ts.append(alloc, .{
                .name = try fmtStr("{s}.mul1", .{pk}),
                .dtype = "I32",
                .shape = &[_]usize{},
                .payload = &mul1,
            });
        };
        const fp = try fragPathLayer(self.o.work, layer_i);
        defer alloc.free(fp);
        const tmp = try fmtStr("{s}.tmp", .{fp});
        defer alloc.free(tmp);
        try writeFragment(tmp, ts.items);
        const s = try sha256File(tmp);
        defer alloc.free(s.bytes);
        try renameAt(tmp, fp);
        return s.hex;
    }

    /// The head fragment: one trellis tensor [K/16, VOCAB/16, words] assembled from the
    /// window encodes (the window tr buffers are blocks of tiles_n, interleaved per block
    /// row — the merge below), plus the single suh/svh/mul1 set under `lm_head`.
    fn writeHeadFragment(self: *Quantizer, sp: *const SlotPlan, pw: PartWork) ![64]u8 {
        var mul1: [4]u8 = undefined;
        std.mem.writeInt(u32, &mul1, efmt.Mul1Mul, .little);
        const k = sp.k;
        const words = pw.halves * 8;
        // chunks' tr buffers each hold [k/16, chunk_nt, words]; the full part wants
        // [k/16, sum_nt, words]: copy chunk tile-rows into the merged buffer
        var ts: std.ArrayList(FragT) = .empty;
        try ts.append(alloc, .{
            .name = try fmtStr("lm_head.trellis", .{}),
            .dtype = "I16",
            .shape = try alloc.dupe(usize, &[_]usize{ k / 16, sp.nt / 16, words }),
            .payload = std.mem.sliceAsBytes(pw.tr[0 .. (k / 16) * (sp.nt / 16) * words]),
        });
        try ts.append(alloc, .{ .name = "lm_head.suh", .dtype = "F16", .shape = try alloc.dupe(usize, &[_]usize{k}), .payload = std.mem.sliceAsBytes(pw.suh) });
        try ts.append(alloc, .{ .name = "lm_head.svh", .dtype = "F16", .shape = try alloc.dupe(usize, &[_]usize{sp.nt}), .payload = std.mem.sliceAsBytes(pw.svh) });
        try ts.append(alloc, .{ .name = "lm_head.mul1", .dtype = "I32", .shape = &[_]usize{}, .payload = &mul1 });
        const fp = try fmtStr("{s}/frag_head.safetensors", .{self.o.work});
        defer alloc.free(fp);
        const tmp = try fmtStr("{s}.tmp", .{fp});
        defer alloc.free(tmp);
        try writeFragment(tmp, ts.items);
        const s = try sha256File(tmp);
        defer alloc.free(s.bytes);
        try renameAt(tmp, fp);
        return s.hex;
    }

    // ------------------------------------------------------------- the stream files

    fn saveStream(self: *Quantizer, i: usize) ![64]u8 {
        const bytes = @as(usize, self.o.rows) * self.o.cols * D * 4;
        const p = try streamPath(self.o.work, i);
        defer alloc.free(p);
        const host = try dlslice(self.d, self.stream, bytes);
        defer alloc.free(host);
        try writeFile(p, std.mem.sliceAsBytes(host));
        return sha256Hex(std.mem.sliceAsBytes(host)); // S5: recorded in progress, verified by loadStream
    }

    fn loadStream(self: *Quantizer, i: usize) !void {
        const p = try streamPath(self.o.work, i);
        defer alloc.free(p);
        const bytes = try readFile(p);
        defer alloc.free(bytes);
        if (bytes.len != @as(usize, self.o.rows) * self.o.cols * D * 4) return error.BadCalib;
        if (self.prog) |prg| { // S5: the recorded hash, not the length, decides trust
            if (prg.stream_sha) |ss| if (prg.stream_i == i) {
                const cur = sha256Hex(bytes);
                if (!std.mem.eql(u8, cur[0..], ss[0..])) {
                    std.debug.print("quant: {s} hashes {s}, progress recorded {s} — refusing\n", .{ p, cur, ss });
                    return error.BadCalib;
                }
            };
        }
        try self.d.copyIn(self.stream, bytes);
    }

    /// The embed pass (spec 1.2's head): the calibration ids row by row through
    /// calib.embedStream into S_0.
    fn embedStreamPass(self: *Quantizer) !void {
        const T = self.o.cols;
        for (0..self.o.rows) |r| {
            const ids_off = @as(usize, r) * T * 4;
            const out_off = @as(usize, r) * T * D * 4;
            var at: u32 = 0;
            while (at < T) {
                const end: u32 = @intCast(q.chunkEnd(at, T));
                try calib.embedStream(self.d, self.m, self.a, off(self.ids_dev, ids_off + @as(usize, at) * 4),
                    off(self.stream, out_off + @as(usize, at) * D * 4), end - at);
                at = end;
            }
        }
        try self.d.sync();
    }

    // ------------------------------------------------------------- one layer

    fn runLayer(self: *Quantizer, i: usize) !void {
        const t0 = q.now();
        var slots = try layerSlots(alloc, i);
        var pws: [4][]PartWork = undefined;
        var fins: [4]?SiteFin = @splat(null);
        defer for (&fins) |*f| if (f.*) |*fin| fin.deinit();
        const rec_mark: usize = self.records.items.len; // Q3 binding: this layer's records slice, persisted below
        var t_ph = q.now();
        try self.capturePass(i, &slots);
        std.debug.print("quant: layer {d}: capture in {d:.1} s\n", .{ i, q.now() - t_ph });
        t_ph = q.now();
        for (0..4) |s| {
            const tag = try fmtStr("layer {d}", .{i});
            fins[s] = try self.finalizeSite(tag, s, i);
            const fin = &fins[s].?;
            var mmaxw: [17]u32 = @splat(0); // one driver per slot and width, sized to that width's widest part
            for (slots[s].parts) |pp| {
                const h = self.halvesOf(s, pp.base);
                mmaxw[h] = @max(mmaxw[h], pp.n);
            }
            for (mmaxw[2..], 2..) |mm, h| {
                if (mm > 0) _ = try self.drvOf(s, mm, @intCast(h)); // one driver per slot and width, sized at that width's widest part
            }
            const pwslot = try alloc.alloc(PartWork, slots[s].parts.len);
            for (slots[s].parts, 0..) |pp, pi| {
                const sv = try drawSigns(slotSeed(self.o.seed, i, s, pi), pp.n);
                defer alloc.free(sv);
                const srow = try alloc.alloc(f32, pp.n);
                defer alloc.free(srow);
                const svh_b = try alloc.alloc(f16, pp.n); // owned until the fragment is written
                const t = try self.c.get(try fmtStr("{s}.weight", .{pp.base}));
                colRmsWin(t.bytes, slots[s].k, pp.n, 0, srow);
                pwslot[pi].spec = pp;
                try self.quantizePartFull(s, i, &slots[s], fin, pp, sv, srow, pp.n, 0, svh_b, &pwslot[pi]);
                try self.records.append(alloc, pwslot[pi].rec);
            }
            pws[s] = pwslot;
        }
        std.debug.print("quant: layer {d}: finalize+encode in {d:.1} s\n", .{ i, q.now() - t_ph });
        // fragments first: the T-LDLQ-5 probe reads the pack's own bytes back through the
        // reader's admission, dequantizes into the serving slab, and checks the oracle
        const t_frag = q.now();
        const sha = try self.writeLayerFragment(i, &slots, &pws);
        std.debug.print("quant: layer {d}: fragments in {d:.1} s\n", .{ i, q.now() - t_frag });
        const t_probe = q.now();
        for (0..4) |s| try self.probeSlot(i, s, &slots[s], pws[s], fins[s].?.su);
        std.debug.print("quant: layer {d}: probe in {d:.1} s\n", .{ i, q.now() - t_probe });
        // the probe's dequant left the trellis bytes in the slabs — the apply pass runs on them
        const t_apply = q.now();
        try self.applyPass(i);
        std.debug.print("quant: layer {d}: apply in {d:.1} s\n", .{ i, q.now() - t_apply });
        // bookkeeping: S_{i+1} on disk, S_{i-1} gone, the progress line
        const ssha = try self.saveStream(i + 1);
        if (self.prog) |*prg| {
            prg.stream_sha = ssha;
            prg.stream_i = i + 1; // saved with the layer's record line below
        }
        self.stream_valid = true;
        if (i >= 1) {
            const old = try streamPath(self.o.work, i - 1);
            defer alloc.free(old);
            removeFile(old);
        }
        try self.done.append(alloc, i);
        std.debug.print("quant: layer {d} done in {d:.1} s (fragment sha {s})\n", .{ i, q.now() - t0, sha });
        const needle = try fmtStr("layers.{d}.", .{i}); // S3: the layer's own records, by base-name needle
        defer alloc.free(needle);
        var layer_idfb = false;
        for (self.records.items) |r| {
            if (!r.identity_fallback) continue;
            if (std.mem.indexOf(u8, r.prefix, needle) != null) layer_idfb = true;
        }
        try self.recordLayer(i, sha, layer_idfb);
        try self.persistRecords(i, self.records.items[rec_mark..]); // Q3 binding: the per-layer records survive the resume
        for (0..4) |s| {
            for (pws[s]) |pw| {
                alloc.free(pw.tr);
                alloc.free(pw.svh);
                if (pw.stats) |st| alloc.free(st);
                if (pw.eta) |et| alloc.free(et);
                if (pw.tr_i) |ti| alloc.free(ti);
                if (pw.zc) |z| alloc.free(z);
                if (pw.qf) |x| alloc.free(x);
                if (pw.what) |x| alloc.free(x);
            }
            alloc.free(pws[s]);
        }
    }

    // Q3 countersign: the §Q3 report survives a resume — the per-layer part records persist
    // beside the progress file and reload into records[] so a resumed run can rebuild it.
    fn recordsPath(work: []const u8, i: ?usize) ![]u8 {
        return if (i) |li| fmtStr("{s}/records_{d}.json", .{ work, li }) else fmtStr("{s}/records_head.json", .{work});
    }

    fn persistRecords(self: *Quantizer, i: ?usize, recs: []const PartRecord) !void {
        if (self.o.dry_run) return; // a dry run never publishes; nothing to bind
        const p = try recordsPath(self.o.work, i);
        defer alloc.free(p);
        const txt = try std.json.Stringify.valueAlloc(alloc, recs, .{});
        defer alloc.free(txt);
        try writeFile(p, txt);
        std.debug.print("quant: records {s} ({d} parts)\n", .{ p, recs.len });
    }

    fn reloadRecords(self: *Quantizer, i: ?usize) !void {
        const p = try recordsPath(self.o.work, i);
        defer alloc.free(p);
        if (!fileExists(p)) {
            std.debug.print("quant: {s} missing — the §Q3 report cannot be rebuilt from this work dir; re-run {s} or start a fresh one\n", .{ p, if (i == null) "the head" else "the missing layer" });
            return error.Usage;
        }
        const txt = try readFile(p);
        defer alloc.free(txt);
        const got = try std.json.parseFromSliceLeaky([]PartRecord, alloc, txt, .{ .allocate = .alloc_always, .ignore_unknown_fields = true });
        for (got) |r| try self.records.append(alloc, r);
    }

    /// The §Q3 report bytes: the §Q3 fields per part (the PartRecord verbatim, plus the
    /// per-part trellis_sha256 over the fragment payload), the contract string and the
    /// encoder. Built BEFORE eaq_write.write so the writer can bind the exact bytes.
    fn buildReport(self: *Quantizer, frags: []const Checkpoint) ![]u8 {
        const Report = struct {
            src: []const u8,
            out: []const u8,
            rows: u32,
            cols: u32,
            bits: u32,
            head_bits: u32,
            seed: u64,
            lambda: f64,
            scale: f64,
            calib_sha: [64]u8,
            contract: []const u8,
            encoder: []const u8,
            layers: []const usize,
            parts: []const PartRecord,
        };
        const parts = try alloc.dupe(PartRecord, self.records.items);
        for (parts) |*r| {
            const tn = try fmtStr("{s}.trellis", .{r.prefix});
            defer alloc.free(tn);
            hit: {
                for (frags) |ck| for (ck.tensors.items) |t| if (std.mem.eql(u8, t.name, tn)) {
                    const hx = sha256Hex(t.bytes);
                    r.trellis_sha256 = try alloc.dupe(u8, hx[0..]);
                    break :hit;
                };
            }
            if (r.trellis_sha256 == null) {
                std.debug.print("quant: report: {s} has no .trellis tensor in any fragment — refusing to certify\n", .{r.prefix});
                return error.PublishFailed;
            }
        }
        const rep = Report{
            .src = self.o.src, .out = self.o.out, .rows = self.o.rows, .cols = self.o.cols,
            .bits = self.o.bits, .head_bits = self.o.head_bits, .seed = self.o.seed,
            .lambda = self.o.lambda, .scale = self.o.scale, .calib_sha = self.calib_sha,
            .contract = contract, .encoder = "xpu-kernel", .layers = self.done.items, .parts = parts,
        };
        return std.json.Stringify.valueAlloc(alloc, rep, .{});
    }

    /// The layer's completion line: recorded in memory always (a dry run prints the shape
    /// it would publish) and persisted immediately only for a real run — the crash-restart
    /// contract the summary hands the operator.
    fn recordLayer(self: *Quantizer, i: usize, sha: [64]u8, identity_fallback: bool) !void {
        // The entry lives for the process lifetime: prog.layers is re-serialized on every
        // later commit, so the path must NOT be freed here (the original defer-free left
        // entries 1..N dangling; the allocator recycled the chunks over tensor names and
        // raw heap words, and the next commit's save wrote non-UTF-8 bytes into the JSON).
        const fp = try fragPathLayer(self.o.work, i);
        const prog = &(self.prog orelse return error.Usage);
        try prog.layers.append(alloc, .{ .i = i, .sha = sha, .frag = fp, .identity_fallback = identity_fallback });
        if (!self.o.dry_run) try prog.save(self.o.work);
    }

    fn isDone(self: *Quantizer, i: usize) bool {
        for (self.done.items) |x| if (x == i) return true;
        return false;
    }

    fn isHeadDone(self: *Quantizer) bool {
        return self.head_done;
    }

    // ------------------------------------------------------------- the head (4.5, last at 4 bits)

    /// The head is captured once (finalNorm over the stream after layer 63, the T-CAP
    /// head-pass twin of the site-4 hook) and encoded in column windows — the full part is
    /// 5 GB at fp32 (spec 1.5's ledger), so windows of head_window columns stream through
    /// the driver and reassemble into one lm_head trellis. One §Q3 record for the whole
    /// part: counters summed, the 40-bucket histogram merged, tile items across windows.
    fn headFlow(self: *Quantizer) !void {
        const t0 = q.now();
        try self.captureHead();
        const tag = "head";
        var fin = try self.finalizeSite(tag, 4, null);
        defer fin.deinit();
        const sp = try headSlotPlan(alloc);
        const pp = sp.parts[0];
        const k: usize = sp.k;
        const sv = try drawSigns(slotSeed(self.o.seed, LAYERS, 4, 0), sp.nt); // L2: the part in the hash — (…,4,null) is the head su's own seed
        defer alloc.free(sv);
        const sfull = try alloc.alloc(f32, sp.nt);
        defer alloc.free(sfull);
        const t = try self.c.get("lm_head.weight");
        colRmsWin(t.bytes, k, sp.nt, 0, sfull);
        const svh_full = try alloc.alloc(f16, sp.nt);
        defer alloc.free(svh_full);
        const w: efmt.Width = .{ .halves = self.halvesOf(4, pp.base) };
        const words = w.tileWords();
        var agg = PartRecord{ .prefix = "lm_head", .k = @intCast(k), .n = sp.nt, .halves = w.halves };
        var items: std.ArrayList(TileItem) = .empty;
        var chunks: std.ArrayList(PartWork) = .empty;
        var win: usize = 0;
        while (win * head_window < sp.nt) : (win += 1) {
            const n0w: u32 = @intCast(win * head_window);
            const nn: u32 = @min(head_window, sp.nt - n0w);
            var pw = PartWork{ .spec = pp, .tr = &.{}, .suh = fin.suh, .svh = svh_full[n0w .. n0w + nn], .halves = w.halves, .rec = .{ .prefix = "lm_head" } };
            try self.quantizePartFull(4, null, &sp, &fin, pp, sv[n0w .. n0w + nn], sfull[n0w .. n0w + nn], nn, n0w, pw.svh[0..nn], &pw);
            agg.tiles += pw.rec.tiles;
            agg.optimal += pw.rec.optimal;
            agg.closed += pw.rec.closed;
            agg.failed += pw.rec.failed;
            agg.pruned1 += pw.rec.pruned1;
            agg.pruned2 += pw.rec.pruned2;
            agg.clamped += pw.rec.clamped;
            agg.eta_sq += pw.rec.eta_sq;
            for (0..40) |b| agg.gap_hist[b] += pw.rec.gap_hist[b];
            if (pw.rec.proxy) |p| agg.proxy = (agg.proxy orelse 0) + p;
            try items.appendSlice(alloc, pw.rec.items);
            try chunks.append(alloc, pw);
        }
        agg.items = items.items;
        agg.identity_fallback = fin.identity; // P1-4: the head's §Q3 record says what it is
        // merge the window trellises: chunk tr is [k/16, win_n/16, words] — the window's
        // OWN width, not the plan part's 248320 (the B2 overflow: cnt 15520 read the chunk
        // buffers off the end from block row 3); the whole part is [k/16, VOCAB/16, words]
        const tr_full = try mergeWindowTrellises(chunks.items, k, sp.nt, words);
        defer alloc.free(tr_full);
        const pw_full = PartWork{ .spec = pp, .win_n = sp.nt, .tr = tr_full, .suh = fin.suh, .svh = svh_full, .halves = w.halves, .rec = agg };
        const sha = try self.writeHeadFragment(&sp, pw_full);
        for (chunks.items) |ch| {
            alloc.free(ch.tr);
            if (ch.stats) |st| alloc.free(st); // the head is not probed; its stats were kept only for symmetry
            if (ch.zc) |z| alloc.free(z); // the zero-column mask (quantizePartFull's; runLayer frees the layer's)
        }
        try self.records.append(alloc, agg);
        try self.persistRecords(null, self.records.items[self.records.items.len - 1 ..]); // Q3 binding: the head's record too
        const prog = &(self.prog orelse return error.Usage);
        prog.head_sha = sha;
        prog.head_idfb = fin.identity; // P1-4: the fallback verdict survives the resume
        if (fin.identity) self.any_identity_fallback = true;
        if (!self.o.dry_run) try prog.save(self.o.work);
        self.head_done = true;
        std.debug.print("quant: head done in {d:.1} s ({d} windows, tiles {d}, optimal {d}, closed {d}, sha {s})\n", .{ q.now() - t0, chunks.items.len, agg.tiles, agg.optimal, agg.closed, sha });
    }

    // ------------------------------------------------------------- publish (4.5)

    /// The pack: the kept dense tensors byte-exact (a/b gates BF16->F16, the only
    /// conversion — vecTensor takes BF16, so embed and norms copy verbatim) plus the 64
    /// layer fragments and the head fragment, through core/eaq_write.zig (EAQ container,
    /// stock entry names; the §Q3 manifest is this run's quant_report.json — the container
    /// manifest is the repacker's, derived from bytes). An identity-fallback slot never
    /// ships; the pack opens through the reader's own openEaq as the last gate.
    fn publish(self: *Quantizer) !void {
        std.debug.print("quant: publishing {s}\n", .{self.o.out});
        if (self.any_identity_fallback) {
            std.debug.print("quant: a slot was encoded with the L = I fallback — the pack gate refuses to publish (re-damp or fix the capture)\n", .{});
            return error.PublishFailed;
        }
        if (self.records.items.len == 0 and !(self.head_done and blk: {
            // S3: a fully recorded resume publishes with empty in-memory records — the
            // fragments and the progress file are the truth; only an INCOMPLETE run publishes
            var li2 = self.o.layer_a;
            var all = true;
            while (li2 <= self.o.layer_b) : (li2 += 1) { if (!self.isDone(li2)) all = false; }
            break :blk all;
        })) return error.PublishFailed;
        const io = std.Io.Threaded.global_single_threaded.io();
        const cfgp = try fmtStr("{s}/config.json", .{self.o.src});
        const cfg = try readFile(cfgp);
        defer alloc.free(cfg);
        var items: std.ArrayList(eaq_write.Input) = .empty;
        const dense = try denseRegistry(self.c);
        for (dense) |en| {
            const t = try self.c.get(en.name); // Checkpoint.get sheds the source's language_model hop (dev.zig:54)
            var sh: [4]usize = undefined;
            for (0..@min(t.rank, 4)) |d| sh[d] = t.shape[d];
            const payload: []const u8 = if (en.convert_to_f16) blk: {
                const out = try alloc.alloc(u16, t.bytes.len / 2);
                bf16ToF16(t.bytes, out);
                break :blk std.mem.sliceAsBytes(out);
            } else t.bytes;
            try items.append(alloc, .{
                .name = en.name,
                .dtype = if (en.convert_to_f16) eaq_write.DType.f16 else (eaq_write.DType.parse(en.dtype) orelse return error.UnexpectedDtype), // P1-3: the header declares the payload that was written (load.zig's gate check wants F16)
                .shape = try alloc.dupe(usize, sh[0..t.rank]),
                .payload = payload,
            });
        }
        // the fragments: opened as shards the whole publish lives long (their maps are the
        // payloads the write stages)
        var frags: std.ArrayList(Checkpoint) = .empty;
        var fi: usize = 0;
        while (fi < LAYERS) : (fi += 1) {
            const fp = try fragPathLayer(self.o.work, fi);
            defer alloc.free(fp);
            if (self.prog) |*prg| for (prg.layers.items) |l| if (l.i == fi) { // S5: verify every recorded sha before it ships
                const cur = try sha256File(l.frag);
                defer alloc.free(cur.bytes);
                if (!std.mem.eql(u8, cur.hex[0..], l.sha[0..])) {
                    std.debug.print("quant: publish: {s} hashes {s}, progress recorded {s} — refusing\n", .{ l.frag, cur.hex, l.sha });
                    return error.PublishFailed;
                }
            };
            var ck: Checkpoint = .{};
            const c0 = try alloc.dupeSentinel(u8, fp, 0);
            defer alloc.free(c0);
            try ck.addShard(c0);
            try frags.append(alloc, ck);
        }
        {
            const fp = try fmtStr("{s}/frag_head.safetensors", .{self.o.work});
            defer alloc.free(fp);
            if (self.prog) |*prg| if (prg.head_sha) |hs| { // S5: the head fragment, same check
                const cur = try sha256File(fp);
                defer alloc.free(cur.bytes);
                if (!std.mem.eql(u8, cur.hex[0..], hs[0..])) {
                    std.debug.print("quant: publish: {s} hashes {s}, progress recorded {s} — refusing\n", .{ fp, cur.hex, hs });
                    return error.PublishFailed;
                }
            };
            var ck: Checkpoint = .{};
            const c0 = try alloc.dupeSentinel(u8, fp, 0);
            defer alloc.free(c0);
            try ck.addShard(c0);
            try frags.append(alloc, ck);
        }
        if (frags.items.len != LAYERS + 1) return error.PublishFailed; // a layer never completed: no pack
        for (frags.items) |*ck| for (ck.tensors.items) |t| {
            var sh: [4]usize = undefined;
            for (0..@min(t.rank, 4)) |d| sh[d] = t.shape[d];
            try items.append(alloc, .{
                .name = t.name,
                .dtype = eaq_write.DType.parse(t.dtype) orelse return error.UnexpectedDtype,
                .shape = try alloc.dupe(usize, sh[0..t.rank]),
                .payload = t.bytes,
            });
        };
        // Q3 countersign: the report exists BEFORE the pack — the writer hashes these exact
        // bytes into the staged shards' metadata, and <out>/quant_report.json holds the same bytes.
        const rep = try self.buildReport(frags.items);
        defer alloc.free(rep);
        if (rep.len > 16 * 1024 * 1024) { // the writer's budget, caught where the message can say why
            std.debug.print("quant: the §Q3 report is {d} bytes, past the writer's 16 MiB budget — the itemisation must shrink\n", .{rep.len});
            return error.PublishFailed;
        }
        try mkdirp(self.o.out);
        const rp = try fmtStr("{s}/quant_report.json", .{self.o.out});
        defer alloc.free(rp);
        try writeFile(rp, rep);
        var why: core.eaq_container.Why = .{};
        _ = eaq_write.write(alloc, io, self.o.out, items.items, .{ .config = cfg, .shards = 8, .report = rep }, &why) catch |e| {
            std.debug.print("quant: publish refused ({s}): {s}\n", .{ @errorName(e), why.text() });
            return error.PublishFailed;
        };
        var rd: Checkpoint = .{};
        try rd.openEaq(self.o.out); // the reader's own verdict on the finished pack
        _ = try load.admitExl3(&rd); // P1-3: the full loadExl3 admission — gates dense F16, the slot census — not just the container
        rd.deinit();
        std.debug.print("quant: the reader opens the pack (openEaq + loadExl3 admission clean); report {s} ({d} parts, {d} bytes)\n", .{ rp, self.records.items.len, rep.len });
    }

};

// ------------------------------------------------ device probe and lease (Main 2026-10-26)

extern "c" fn flock(fd: c_int, op: c_int) c_int;
extern "c" fn getpid() c_int;
const LOCK_EX = 2;
const LOCK_NB = 4;
pub const lease_path = "/tmp/xpu-eaq-quant.lease";
var lease_fd: c_int = -1; // held for the life of the process: exit releases the card, close never does

/// The card probe (`--device-check`, the run script's pre-flight): open Level Zero, read
/// the sysman free bytes, compare against this run's estimate; returns the exit code
/// (0 fits, 3 short) and allocates nothing (the fable device-mem ruling's contract).
pub fn deviceCheck(o: Options) !u8 {
    var g = try ze.Gpu.open(0xe223);
    errdefer g.close();
    const need = estimateBytes(o);
    if (g.memory()) |mm| {
        std.debug.print("device: {d} of {d} bytes free; need {d} (+1 GiB)\n", .{ mm.free, mm.size, need });
        return if (mm.free >= need) 0 else 3;
    } else |_| {
        std.debug.print("device: the memory probe failed; need {d} (+1 GiB) — cannot confirm the card fits\n", .{need});
        return 3;
    }
}

/// Takes the card lease: O_CREAT + flock(LOCK_EX|LOCK_NB), then writes pid, start time,
/// work dir and the estimate. Busy flock means one of OUR OWN runs holds the card: the
/// refusal names the lease. The kernel, not this file, is the authority for foreign
/// holders (the ruling's fdinfo scan runs script-side as root).
fn takeLease(o: *const Options) !void {
    const sent = try alloc.dupeSentinel(u8, lease_path, 0);
    defer alloc.free(sent);
    const fd = open(sent.ptr, @as(c_int, 2 | 64), @as(c_int, 0o644)); // O_CREAT | O_RDWR
    if (fd < 0) return; // no lease file is writable: the allocations remain the gate
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        _ = close(fd);
        if (readFile(lease_path)) |lb| {
            defer alloc.free(lb);
            std.debug.print("quant: the card lease is held by our own run: {s}(flock busy; nothing here kills it)\n", .{lb});
        } else |_| {}
        return error.InsufficientDeviceMemory;
    }
    lease_fd = fd;
    const body = try fmtStr("pid={d} started={d} work={s} need={d}\n", .{ getpid(), q.now(), o.work, estimateBytes(o.*) });
    try writeFile(lease_path, body);
}
// ---------------------------------------------------------------- the run (5.5)

/// The planned device footprint (the card gate's estimate, before any allocation): the
/// model, stream, capture, Hessian buffers, slab staging, and the five drivers with the
/// encoder's stat/cost/bp buffers — the ledger from spec 1.5's numbers, scaled by the
/// actual rows/cols. 1 GiB of slack on top.
fn estimateBytes(o: Options) usize {
    var e: usize = 0;
    // buildDenseModel: embed, invf, and every layer's resident caches/small tensors (the
    // per-layer worst case counts the attention and the GDN body alike)
    e += @as(usize, VOCAB) * D * 2;
    e += 128;
    e += @as(usize, LAYERS) * (2 * fwd.kvBytes() + 2 * (48 * 128 * 128 * 4) + 96 * D * 2 +
        (3 * 10240 * 2) + q.STASH + 2 * (D * 2 + FF * 2));
    // fwd.Acts.init at maxRows: wscr (the fused gate|up fp16 staging) dominates, the row
    // buffers (x/h/p/pscr/taps/q/k/v/y and the GDN chunk planes) ride under the term
    e += @as(usize, D) * (2 * FF) * 2 + (768 << 20);
    // calib.Slabs: the four fp16 [K, NT] slots sized to the widest (GDN) pitches
    e += @as(usize, D) * GDN_IN * 2 + 6144 * D * 2 + @as(usize, D) * (2 * FF) * 2 + @as(usize, FF) * D * 2;
    // calib.initCapture: xpad + xt at the row block
    const tp: usize = (o.cols + 63) / 64 * 64;
    e += 2 * tp * FF * 2;
    // cap.h = hbuf[0..4]: the five fp32 [K_s, K_s] Hessian buffers
    for ([_]usize{ D, 6144, D, FF, D }) |kk| e += kk * kk * 4;
    // the quantization stream (rows x cols of fp32 D), its row staging, the ids
    e += @as(usize, o.rows) * o.cols * D * 4;
    e += @as(usize, o.cols) * D * 4;
    e += @as(usize, o.rows) * o.cols * 4;
    // the per-site ldlq Drivers (drvOf: one per slot, sized to that slot's widest part,
    // cached for the whole run — sizes from ldlq.Driver.init / exl3_encode.Encoder.init)
    const sites = [_]struct { k: u32, n: u32, head: bool, mixer: bool }{
        .{ .k = D, .n = 12288, .head = false, .mixer = true }, // the widest mixer_in part (attn q_proj)
        .{ .k = 6144, .n = D, .head = false, .mixer = true },
        .{ .k = D, .n = FF, .head = false, .mixer = true }, // gate and up share the width
        .{ .k = FF, .n = D, .head = false, .mixer = true },
        .{ .k = D, .n = if (o.layer_b == LAYERS - 1) head_window else 0, .head = true, .mixer = false },
    };
    for (sites) |st| {
        if (st.n == 0) continue; // the head's driver is never built unless the head quantizes
        const bits: usize = if (st.head) o.head_bits else o.bits;
        const halves: usize = 2 * bits;
        const k: usize = st.k;
        const n: usize = st.n;
        const kblks: usize = k / 16;
        const nt: usize = n / 16;
        const tiles: usize = kblks * nt;
        e += 128 * 2 * 65536 * 4; // encoder cost plane (count = 128 groups)
        const fresh: usize = if (halves % 2 == 1) (halves - 1) / 2 else halves / 2; // exl3_encode.minFreshBits
        e += 128 * 256 * std.math.pow(usize, 2, @as(u8, @intCast(16 - @as(u8, @intCast(fresh))))); // encoder bp plane
        e += tiles * 32; // encoder stat
        e += k * k * 4; // l32
        e += 2 * k * n * 4; // w32 + t32 at the widest part
        e += 2 * 16 * n * 2 + 16 * n * 4; // s16 + q16 + e32
        e += tiles * halves * 8 * 2; // tr_dev (tileWords int16 per tile)
        e += kblks * ((n + 255) / 256) * 4; // part_scratch
        if (o.proxy and st.mixer) e += k * n * 4; // want_eta's eta_dev strips
    }
    // kernel modules (initKernelTable) and the probe's transient one-part upload
    e += 256 << 20;
    e += 1 << 30;
    return e;
}

/// The calibration ids: decimal tokens under the runner's tokenizer grammar (whitespace
/// and JSON separators), exactly rows x cols of them.
fn loadCalibIds(path: []const u8, want: usize) ![]u32 {
    const txt = try readFile(path);
    defer alloc.free(txt);
    var list: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, txt, " \n\r\t,[]");
    while (it.next()) |tok| try list.append(alloc, try std.fmt.parseInt(u32, tok, 10));
    if (list.items.len != want) {
        std.debug.print("quant: {s} holds {d} ids, the run wants rows x cols = {d}\n", .{ path, list.items.len, want });
        return error.BadCalib;
    }
    return list.items;
}

pub fn openSource(c: *Checkpoint, dir: []const u8) !void {
    // the shard list comes from the source's model.safetensors.index.json
    const idxp = try fmtStr("{s}/model.safetensors.index.json", .{dir});
    const idx = readFile(idxp) catch return error.BadCalib; // read failure => the dir/index is unusable
    defer alloc.free(idx);
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, idx, .{});
    defer parsed.deinit();
    const wm = parsed.value.object.get("weight_map") orelse return error.BadCalib;
    var files: std.ArrayList([]const u8) = .empty;
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(alloc);
    var it = wm.object.iterator();
    while (it.next()) |kv| {
        const shard = switch (kv.value_ptr.*) {
            .string => |nm| nm,
            else => return error.BadCalib,
        };
        if (!std.mem.endsWith(u8, shard, ".safetensors")) continue;
        if (seen.contains(shard)) continue;
        try seen.put(alloc, shard, {});
        try files.append(alloc, try alloc.dupe(u8, shard));
    }
    var si: usize = 1; // deterministic shard order: lexicographic by file name
    while (si < files.items.len) : (si += 1) {
        var sj = si;
        while (sj > 0 and std.mem.order(u8, files.items[sj - 1], files.items[sj]) == .gt) : (sj -= 1) {
            std.mem.swap([]const u8, &files.items[sj - 1], &files.items[sj]);
        }
    }
    for (files.items) |name| {
        const p = try fmtStr("{s}/{s}", .{ dir, name });
        defer alloc.free(p);
        const s0 = try alloc.dupeSentinel(u8, p, 0);
        defer alloc.free(s0);
        try c.addShard(s0);
    }
    var kept: usize = 0;
    for (c.tensors.items) |t| {
        if (isKept(t.name)) kept += 1;
    }
    if (kept != KEPT) {
        std.debug.print("quant: {s} keeps {d} tensors, expected {d} (850 language_model + lm_head after the visual/mtp drop)\n", .{ dir, kept, KEPT });
        return error.Usage;
    }
}

pub fn run(o: Options) !void {
    if (o.src.len == 0 or o.out.len == 0 or o.calib_path.len == 0) return error.Usage;
    if (o.rows == 0 or o.cols <= 16 or o.cols > 4096) return error.Usage;
    if (o.bits < 2 or o.bits > 6 or o.head_bits < 2 or o.head_bits > 6) return error.Usage;
    if (o.layer_a >= LAYERS or o.layer_b >= LAYERS or o.layer_a > o.layer_b) return error.Usage;
    if (!(o.lambda > 0 and o.lambda < 1)) return error.Usage;
    if (!(o.scale > 0)) return error.Usage;
    if (!o.seed_given and !o.resume_run) {
        std.debug.print("quant: --seed N is required (the signs and the per-part draws key on it); a --resume run re-reads it from the progress file\n", .{});
        return error.Usage;
    }
    const sha = try sha256File(o.calib_path);
    defer alloc.free(sha.bytes);
    var o2 = o;
    if (o2.work.len == 0) o2.work = try fmtStr("{s}-work", .{o2.out});
    try mkdirp(o2.work);
    // (S6) no unconditional <work>/h: captured Hessians persist only under --save-h or the
    // red finalizeSite dump, never as ~108 GB of run debris
    const ids = try loadCalibIds(o2.calib_path, @as(usize, o2.rows) * o2.cols);

    std.debug.print("quant: source {s} out {s} rows {d} x cols {d} bits {d} head-bits {d} seed {d} lambda {d} scale {d} layers {d}..{d}\n", .{
        o2.src, o2.out, o2.rows, o2.cols, o2.bits, o2.head_bits, o2.seed, o2.lambda, o2.scale, o2.layer_a, o2.layer_b,
    });
    var c: Checkpoint = .{};
    try openSource(&c, o2.src);

    const d = try alloc.create(Dev);
    d.* = .{ .g = try ze.Gpu.open(0xe223), .list = undefined, .k = undefined };
    errdefer d.g.close(); // before the queue exists, closing the context is the whole teardown
    d.list = try d.g.immediate(0, 0);
    errdefer d.settle(); // the fable ruling: once the queue is real, drain the in-flight work
    // BEFORE destroying anything, on every error path — a context closed under an
    // unsignaled fence is how an exited driver pins gigabytes in xe_file_close.
    const namez: [256]u8 = d.g.props.name;
    var freeb: u64 = 0;
    var allb: u64 = 0;
    const mem_ok = if (d.g.memory()) |mm| blk: {
        freeb = mm.free;
        allb = mm.size;
        break :blk true;
    } else |_| false;
    const need = estimateBytes(o2);
    if (mem_ok) {
        std.debug.print("quant: {s}: {d} of {d} card bytes free; the run estimates {d} (+1 GiB slack)\n", .{
            std.mem.span(@as([*:0]const u8, @ptrCast(&namez))), freeb, allb, need,
        });
        if (freeb < need) {
            std.debug.print("quant: the card cannot fit this run — free the serve holding device bytes (nothing here kills it)\n", .{});
            return error.InsufficientDeviceMemory;
        }
        try takeLease(&o2); // fits: claim the card before the first allocation, lease names us
    } else {
        // S7: a failed probe is a failed PROBE, not an empty card — say so and let the
        // allocations themselves be the gate instead of refusing with a misleading line
        std.debug.print("quant: {s}: the card memory probe failed; proceeding — the run estimates {d} (+1 GiB slack) and the allocations are the gate\n", .{
            std.mem.span(@as([*:0]const u8, @ptrCast(&namez))), need,
        });
    }
    // the lease lives for the whole run (the fd closes and flock releases only at exit);
    // the file goes with a normal return — NEVER a defer inside the admission if-block,
    // whose scope ends immediately and would delete the lease seconds after taking it
    defer if (lease_fd >= 0) removeFile(lease_path);
    try initKernelTable(d);

    const m = try alloc.create(Model);
    m.* = try buildDenseModel(d, &c);
    const a = try alloc.create(fwd.Acts);
    a.* = try fwd.Acts.init(d, true);
    const slabs = try calib.Slabs.init(d);
    var cap = try calib.initCapture(d, @intCast((o2.cols + 63) / 64 * 64));
    errdefer calib.deinitCapture(&cap, d);
    for (&cap.h, 0..) |*h, s| {
        const kk = Quantizer.siteK(s);
        h.* = try d.alloc_(@as(usize, kk) * kk * 4);
    }
    const stream = try d.alloc_(@as(usize, o2.rows) * o2.cols * D * 4);
    const rowbuf = try d.alloc_(@as(usize, o2.cols) * D * 4);
    const ids_dev = try d.upload(std.mem.sliceAsBytes(ids));

    var sq = Quantizer{
        .o = o2, .d = d, .m = m, .a = a, .c = &c, .slabs = slabs, .cap = cap,
        .stream = stream, .rowbuf = rowbuf, .ids_dev = ids_dev, .ids = ids,
    };
    for (&sq.hbuf, 0..) |*h, s| h.* = cap.h[s]; // finalizeSite downloads through the same buffers
    sq.calib_sha = sha.hex;
    try sq.loadPlan(); // --plan, or the degenerate uniform plan synthesized from --bits (spec 3.2)

    // the progress file: resume skips what it records; the parameters must match
    const pp = try Progress.filePath(o2.work);
    defer alloc.free(pp);
    if (o2.resume_run) {
        sq.prog = (try Progress.load(o2.work)) orelse {
            std.debug.print("quant: no progress file in {s} — nothing to resume\n", .{o2.work});
            return error.Usage;
        };
        const pr = &sq.prog.?;
        if (!std.mem.eql(u8, pr.src, o2.src) or !std.mem.eql(u8, pr.calib_sha[0..], sha.hex[0..]) or pr.rows != o2.rows or pr.cols != o2.cols or
            pr.bits != o2.bits or pr.head_bits != o2.head_bits or
            pr.lambda != o2.lambda or pr.scale != o2.scale)
        {
            std.debug.print("quant: {s} was made by different run parameters (rows/cols/bits/calib/lambda/scale) — refusing the resume; use a fresh work dir\n", .{pp});
            return error.Usage;
        }
        if (o.seed_given and pr.seed != o.seed) {
            std.debug.print("quant: --seed {d} disagrees with the recorded seed {d} — refusing the resume\n", .{ o.seed, pr.seed });
            return error.Usage;
        }
        if (!o.seed_given) sq.o.seed = pr.seed;
        for (pr.layers.items) |l| {
            const cur = try sha256File(l.frag); // S5: verify the recorded sha before skipping the layer
            defer alloc.free(cur.bytes);
            if (!std.mem.eql(u8, cur.hex[0..], l.sha[0..])) {
                std.debug.print("quant: {s} hashes {s}, the progress file recorded {s} — refusing the resume\n", .{ l.frag, cur.hex, l.sha });
                return error.Usage;
            }
            if (l.identity_fallback) sq.any_identity_fallback = true; // S3: the fallback verdict survives the resume
            try sq.done.append(alloc, l.i);
            try sq.reloadRecords(l.i); // Q3 binding: the §Q3 report rebuilds from the persisted records
        }
        if (pr.head_sha) |hs| { // P1-4: head_done is earned by re-hashing, like every layer
            const hp = try fmtStr("{s}/frag_head.safetensors", .{o2.work});
            defer alloc.free(hp);
            const cur = try sha256File(hp);
            defer alloc.free(cur.bytes);
            if (std.mem.eql(u8, cur.hex[0..], hs[0..])) {
                sq.head_done = true;
                try sq.reloadRecords(null); // Q3 binding: the head's record must ship too
            } else std.debug.print("quant: {s} hashes {s}, the progress file recorded {s} — the head will be redone\n", .{ hp, cur.hex, hs });
        }
        if (pr.head_idfb) sq.any_identity_fallback = true; // P1-4: a recorded fallback head still closes the pack gate
        const want_ps: ?[64]u8 = if (sq.plan) |p| std.fmt.bytesToHex(p.plan_sha, .lower) else null;
        const same_plan = (pr.plan_sha == null and want_ps == null) or
            (pr.plan_sha != null and want_ps != null and std.mem.eql(u8, &pr.plan_sha.?, &want_ps.?));
        if (!same_plan) {
            var have_buf: [64]u8 = @splat(0);
            var have_txt: []const u8 = "(none)";
            if (pr.plan_sha) |x| {
                have_buf = x;
                have_txt = &have_buf;
            }
            var want_buf: [64]u8 = @splat(0);
            var want_txt: []const u8 = "(none)";
            if (want_ps) |x| {
                want_buf = x;
                want_txt = &want_buf;
            }
            std.debug.print("quant: {s} was made under plan {s}, this run says {s} — refusing the resume; use a fresh work dir\n", .{ pp, have_txt, want_txt });
            return error.Usage;
        }
        std.debug.print("quant: resuming {d} completed layers from {s}\n", .{ sq.done.items.len, pp });
    } else if (fileExists(pp)) {
        std.debug.print("quant: {s} exists — the work dir is not fresh; use --resume or a fresh --work\n", .{pp});
        return error.Usage;
    } else {
        sq.prog = Progress{
            .src = try alloc.dupe(u8, o2.src),
            .calib_sha = sha.hex,
            .rows = o2.rows, .cols = o2.cols, .bits = o2.bits, .head_bits = o2.head_bits,
            .seed = o2.seed, .lambda = o2.lambda, .scale = o2.scale,
            .plan_sha = if (sq.plan) |p| std.fmt.bytesToHex(p.plan_sha, .lower) else null,
        };
    }
    sq.single_layer = o2.proxy and o2.layer_a == o2.layer_b;
    if (o2.layer_b == LAYERS - 1) { // S7: fail the head driver's ~4 GB at minute 1, not hour 23 (drvOf caches — headFlow reuses it)
        _ = try sq.drvOf(4, @max(head_window, sq.siteRunNMax(4)), sq.halvesOf(4, "lm_head"));
    }

    var li: usize = o2.layer_a;
    while (li <= o2.layer_b) : (li += 1) {
        if (sq.isDone(li)) {
            sq.stream_valid = false; // the stream in device memory belongs to an earlier boundary
            continue;
        }
        if (!sq.stream_valid) {
            if (li == 0) {
                try sq.embedStreamPass();
            } else {
                std.debug.print("quant: loading S_{d} from the work dir\n", .{li});
                try sq.loadStream(li);
            }
            sq.stream_valid = true;
        }
        try sq.runLayer(li);
    }
    if (o2.layer_b == LAYERS - 1) {
        if (!sq.isHeadDone()) {
            if (!sq.stream_valid) {
                try sq.loadStream(LAYERS); // S_64: saveStream(i+1) wrote stream_64 after layer 63; LAYERS-1 was layer 63's INPUT (the B4 head-calibration bug)
                sq.stream_valid = true;
            }
            try sq.headFlow();
        }
        if (!o2.dry_run) {
            try sq.publish();
            std.debug.print("quant: pack published to {s}\n", .{o2.out});
        } else {
            std.debug.print("quant: dry run — nothing published, progress untouched\n", .{});
        }
    } else {
        std.debug.print("quant: layer range stops at {d}: no head, no publish; continue with --layers {d}..{d} --resume\n", .{ o2.layer_b, o2.layer_b + 1, LAYERS - 1 });
    }
    std.debug.print("quant: resume contract: xpu-eaq-quant {s} {s} --calib {s} --rows {d} --cols {d} --bits {d} --head-bits {d} --seed {d} --lambda {d} --scale {d} --layers {d}..{d} --work {s} --resume\n", .{
        o2.src, o2.out, o2.calib_path, o2.rows, o2.cols, o2.bits, o2.head_bits, sq.o.seed, o2.lambda, o2.scale, o2.layer_a, o2.layer_b, o2.work,
    });
}

pub const calib_mod = calib; // re-export surface for the CLI's help text
pub const ldlq_mod = ldlq;
