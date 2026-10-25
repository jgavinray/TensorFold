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
//! projectionExl3, calib.dequantSlot into the serving slab, and the slab checked against
//! the exl3_format.dequantize oracle on the PR 3 dequant rows (RMS error 2^-11 and worst
//! element 2^-8 of the RMS weight, qwen27_main.zig:619-645's sampling and bounds).
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
    if (fd < 0) return error.FileNotFound;
    defer _ = close(fd);
    var done: usize = 0;
    while (done < bytes.len) {
        const n = write(fd, bytes.ptr + done, bytes.len - done);
        if (n <= 0) return error.FileNotFound;
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

fn dlslice(d: *Dev, p: *anyopaque, bytes: usize) ![]u8 {
    const b = try alloc.alloc(u8, bytes);
    errdefer alloc.free(b);
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, b.ptr, p, bytes, null, 0, null));
    try d.sync();
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
    m.norm = try load.vecTensor(d, c, "model.language_model.norm.weight", false);
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
            .in_norm = try load.vecTensor(d, c, try fmtStr("{s}input_layernorm.weight", .{p}), false),
            .post_norm = try load.vecTensor(d, c, try fmtStr("{s}post_attention_layernorm.weight", .{p}), false),
            .attn = q.isAttention(i),
            .mixer_in = .{ .w = stage, .s = stage, .bg = stage, .n = if (q.isAttention(i)) ATT_IN else GDN_IN, .k = D, .wide = true, .gs = q.GS },
            .mixer_out = .{ .w = stage, .s = stage, .bg = stage, .n = D, .k = 6144, .wide = false, .gs = q.GS },
            .gu = .{ .w = stage, .s = stage, .bg = stage, .n = 2 * FF, .k = D, .wide = true, .gs = q.GS },
            .down = .{ .w = stage, .s = stage, .bg = stage, .n = D, .k = FF, .wide = false, .gs = q.GS },
        };
        if (l.attn) {
            l.qn = try load.vecTensor(d, c, try fmtStr("{s}self_attn.q_norm.weight", .{p}), false);
            l.kn = try load.vecTensor(d, c, try fmtStr("{s}self_attn.k_norm.weight", .{p}), false);
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

    fn colLoss(j: *ProxyJob, got: []const f16, v: []f64, u: []f64, du: []f64) f64 {
        const k = j.k;
        var loss: f64 = 0;
        for (j.lo..j.hi) |c| {
            for (0..k) |r| v[r] = @as(f64, @floatCast(got[r * j.n + c])) - j.wq[r * j.n + c];
            for (0..k) |p| { // (L^T v)[p]
                var acc: f64 = 0;
                for (0..k) |r| acc += @as(f64, j.l[r * k + p]) * v[r];
                u[p] = acc;
            }
            for (0..k / 16) |bj| for (0..16) |ai| { // (blockdiag D) u
                var acc: f64 = 0;
                for (0..16) |b| acc += j.d[(bj * 16 + ai) * 16 + b] * u[bj * 16 + b];
                du[bj * 16 + ai] = acc;
            };
            for (0..k) |r| { // v . (L du)
                var acc: f64 = 0;
                for (0..k) |p| acc += @as(f64, j.l[r * k + p]) * du[p];
                loss += v[r] * acc;
            }
        }
        return loss;
    }

    fn run(j: *ProxyJob) void {
        const v = alloc.alloc(f64, j.k) catch @panic("oom");
        defer alloc.free(v);
        const u = alloc.alloc(f64, j.k) catch @panic("oom");
        defer alloc.free(u);
        const du = alloc.alloc(f64, j.k) catch @panic("oom");
        defer alloc.free(du);
        j.loss_a = j.colLoss(j.got_a, v, u, du);
        if (j.got_b.len > 0) j.loss_b = j.colLoss(j.got_b, v, u, du);
    }
};

fn proxyJobRun(j: *ProxyJob) void {
    j.run();
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
    const max_threads = 32;
    const nt: usize = n / 16; // whole column tiles per worker
    var nthreads: usize = @min(max_threads, nt);
    nthreads = @max(1, nthreads);
    const tiles_per = (nt + nthreads - 1) / nthreads;
    var jobs: [max_threads]ProxyJob = undefined;
    var threads: [max_threads]std.Thread = undefined;
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
        for (0..k) |kk| {
            const s: f64 = @floatFromInt(su[kk]);
            for (0..n) |j| {
                if (zero[j]) {
                    wq[kk * n + j] = 0;
                    continue;
                }
                const idx = ((n0 + j) * k + kk) * 2;
                const bits: u32 = @as(u32, t[idx]) | (@as(u32, t[idx + 1]) << 8);
                const v: f64 = @as(f64, @floatCast(@as(f32, @bitCast(bits << 16)))); // bf16 = top half of fp32
                wq[kk * n + j] = v * s / @as(f64, @floatCast(svh[j]));
            }
        }
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
        const zc = try alloc.alloc(bool, n);
        defer alloc.free(zc);
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
        const results = try alloc.alloc(ldlq.TileStat, (k / 16) * (n / 16));
        defer alloc.free(results);
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

        if (self.o.proxy and !fin.identity and site != 4) { // the head has no L = I twin and no eta plane (L1)
            const losses = proxyLosses(w, &fin.ldl, tr, &.{}, wq, k, n);
            rec.proxy = losses.a;
            if (self.single_layer) {
                const tr_i = try alloc.alloc(i16, tr.len);
                defer alloc.free(tr_i);
                const res_i = try alloc.alloc(ldlq.TileStat, results.len);
                defer alloc.free(res_i);
                var slot_i = slot;
                slot_i.ldl = &fin.ldl_i.?;
                slot_i.gen = fin.gen + 1; // P0-2: the L = I twin is its own factorization, not the shipped one's
                try dr.beginSlot(d, &slot_i);
                _ = try dr.quantizePart(d, &slot_i, &part, wq, tr_i, res_i);
                try dr.beginSlot(d, &slot); // restore the shipped L for whatever comes next
                const li = proxyLosses(w, &fin.ldl, tr_i, &.{}, wq, k, n);
                rec.proxy_identity = li.a;
                const gain = (li.a - losses.a) / li.a; // the LDLQ gain (spec 4.6's number)
                std.debug.print("quant: {s}: T-LDLQ-4 loss(LDLQ) {e} loss(L=I) {e} gain {d:.3} pct\n", .{ base_name, losses.a, li.a, gain * 100 });
                if (!(losses.a < li.a)) {
                    std.debug.print("quant: {s}: the LDLQ run is NOT below the L = I run — T-LDLQ-4 red\n", .{base_name});
                    return error.ProxyOrdering;
                }
                if (std.mem.eql(i16, tr, tr_i)) {
                    std.debug.print("quant: {s}: the L = I encode produced an identical trellis — L never reached the loop\n", .{base_name});
                    return error.ProxyOrdering;
                }
            }
            const eta = try alloc.alloc(f32, k * n);
            defer alloc.free(eta);
            try dr.readEta(d, eta, @intCast(k), @intCast(n));
            var dsum: f64 = 0;
            for (0..k / 16) |b| {
                for (0..16) |ai| dsum += fin.ldl.d[b * 256 + ai * 16 + ai];
            }
            const md = dsum / @as(f64, @floatFromInt(k));
            rec.dratio = ldlq_ref.etaWeighted(fin.ldl.d, eta, k, n) / (md * rec.eta_sq);
        }

        pw.* = .{ .spec = pp, .win_n = n_win, .tr = tr, .suh = fin.suh, .svh = svh, .halves = w.halves, .rec = rec };
    }

    /// The dequant probe (T-LDLQ-5): the fragment's parts back through the reader's own
    /// admission (exl3_model.projectionExl3) and the serving dequant into the slab
    /// (calib.dequantSlot — the bytes the serving prompt GEMM reads from wscr). The fable
    /// ruling's three accepted metrics (quantfix-gates-answer-fable.md): global relative RMS
    /// of the fp16 slab against the fp64 oracle at most 2^-11; worst-COLUMN relative RMS at
    /// most 2^-8; the single worst element over the global RMS at most 0.02 (the PR 45
    /// dequant gate's enforced bound; the 2^-8 single-element figure is only the
    /// Gaussian-tail reference for a near-Gaussian pack, which this f16 storage cannot meet
    /// once max|w|/rms passes 8-16). The ulp columns name the fp16-rounded-oracle distance.
    /// SAMPLING: the first and last 128-k-row blocks of every part (the exl3OracleCheck
    /// convention), stated here as the ruling requires.
    fn probeSlot(self: *Quantizer, layer_i: usize, site: usize, sp: *const SlotPlan, pws: []const PartWork) !void {
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
            const words = w.tileWords();
            var worst: f64 = 0; // the worst COLUMN RMS ratio (the ruling's metric)
            var err_sumsq: f64 = 0;
            var ref_sumsq: f64 = 0;
            var max_ulp: u32 = 0;
            var ulp0: u64 = 0;
            var ulp1: u64 = 0;
            var elem_worst: f32 = 0; // the single worst element's absolute error
            var elem_worst_ulp: u32 = 0; // its fp16 ulp distance from the fp16-rounded oracle
            const col_err = try alloc.alloc(f64, n);
            defer alloc.free(col_err);
            const col_ref = try alloc.alloc(f64, n);
            defer alloc.free(col_ref);
            @memset(col_err, 0);
            @memset(col_ref, 0);
            for ([_]usize{ 0, k - 128 }) |row0| {
                const rows: usize = 128;
                const wref = try alloc.alloc(u16, rows * n);
                defer alloc.free(wref);
                for (0..rows) |r| {
                    const byte_off = (@as(usize, row0 + r) * sp.nt + pw.spec.n0) * 2;
                    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, wref[r * n ..][0..n].ptr,
                        off(self.slabOf(site), byte_off), n * 2, null, 0, null));
                }
                try d.sync();
                const out64 = try alloc.alloc(f64, rows * n);
                defer alloc.free(out64);
                const scratch64 = try alloc.alloc(f64, 2 * rows * n);
                defer alloc.free(scratch64);
                const trellis = pw.tr[(row0 / 16) * (n / 16) * words ..][0..(rows / 16) * (n / 16) * words];
                efmt.dequantize(w, .mul1, trellis, rows / 16, n / 16, pw.suh[row0..][0..rows], pw.svh, out64, scratch64);
                for (0..rows) |r| for (0..n) |cc| {
                    const ref: f32 = @floatCast(out64[r * n + cc]);
                    const gbits = wref[r * n + cc];
                    const gotv: f32 = @as(f32, @as(f16, @bitCast(gbits)));
                    const dv = @abs(ref - gotv);
                    err_sumsq += @as(f64, dv) * dv;
                    ref_sumsq += @as(f64, ref) * ref;
                    col_err[cc] += @as(f64, dv) * dv;
                    col_ref[cc] += @as(f64, ref) * ref;
                    const rn16: f16 = @floatCast(out64[r * n + cc]); // the fp16-rounded oracle
                    const rbits: u16 = @bitCast(rn16);
                    const gc: i32 = if (gbits & 0x8000 != 0) 0x8000 - @as(i32, gbits) else @intCast(gbits);
                    const rc: i32 = if (rbits & 0x8000 != 0) 0x8000 - @as(i32, rbits) else @intCast(rbits);
                    const u: u32 = @intCast(@abs(gc - rc));
                    if (u > max_ulp) max_ulp = u;
                    if (u == 0) ulp0 += 1;
                    if (u == 1) ulp1 += 1;
                    if (dv > elem_worst) {
                        elem_worst = dv;
                        elem_worst_ulp = u;
                    }
                };
            }
            // the fable ruling (quantfix-gates-answer-fable.md): worst = worst COLUMN RMS ratio;
            // the single element is kept with the PR 45 enforced bound 0.02, the 2^-8 figure
            // beside it as the Gaussian-tail reference only
            var wcol: usize = 0;
            for (0..n) |cc| {
                if (col_ref[cc] == 0) continue;
                const e = @sqrt(col_err[cc] / col_ref[cc]); // the sampler's own metric (fp64_ref.zig:418): pure column ratio
                if (e > worst) {
                    worst = e;
                    wcol = cc;
                }
            }
            const ref_rms: f32 = @floatCast(@sqrt(ref_sumsq / @as(f64, @floatFromInt(2 * 128 * n))));
            const err_rms: f32 = @floatCast(@sqrt(err_sumsq / @as(f64, @floatFromInt(2 * 128 * n))));
            const rel_rms = @as(f64, err_rms) / @max(@as(f64, ref_rms), 1e-30);
            const rel_worst = worst; // already relative to the column's own RMS — do NOT divide by the global RMS again
            const rel_elem = @as(f64, elem_worst) / @max(@as(f64, ref_rms), 1e-30);
            const cmp: f64 = @floatFromInt(2 * 128 * n);
            std.debug.print("quant: T-LDLQ-5 {s}: err RMS {e:.3} of RMS weight (bound 2^-11 = {e:.3}), worst column RMS {e:.3} @ column {d} (bound 2^-8 = {e:.3}; column RMS {e:.3}, column error {e:.3}, svh {e:.3}), worst element {e:.3} (bound 0.02, Gaussian-tail reference 2^-8; {d} ulp from the fp16-rounded oracle), max ulp {d}, ulp0 {d:.1} pct ulp1 {d:.1} pct (sampled: first and last 128-k-row blocks)\n", .{
                pw.spec.base, rel_rms, std.math.pow(f64, 2, -11), rel_worst, wcol, std.math.pow(f64, 2, -8),
                @sqrt(col_ref[wcol] / @as(f64, 2 * 128)), @sqrt(col_err[wcol] / @as(f64, 2 * 128)), @as(f64, pw.svh[wcol]),
                rel_elem, elem_worst_ulp, max_ulp, 100 * @as(f64, @floatFromInt(ulp0)) / cmp, 100 * @as(f64, @floatFromInt(ulp1)) / cmp,
            });
            if (rel_rms > std.math.pow(f64, 2, -11) or rel_worst > std.math.pow(f64, 2, -8) or rel_elem > 0.02) {
                std.debug.print("quant: T-LDLQ-5 {s}: RED\n", .{pw.spec.base});
                return error.ProbeFailed;
            }
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
        try self.capturePass(i, &slots);
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
        // fragments first: the T-LDLQ-5 probe reads the pack's own bytes back through the
        // reader's admission, dequantizes into the serving slab, and checks the oracle
        const sha = try self.writeLayerFragment(i, &slots, &pws);
        for (0..4) |s| try self.probeSlot(i, s, &slots[s], pws[s]);
        // the probe's dequant left the trellis bytes in the slabs — the apply pass runs on them
        try self.applyPass(i);
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
        const fp = try fragPathLayer(self.o.work, i);
        defer alloc.free(fp);
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
        for (chunks.items) |ch| alloc.free(ch.tr);
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
    errdefer d.g.close();
    d.list = try d.g.immediate(0, 0);
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
    } else {
        // S7: a failed probe is a failed PROBE, not an empty card — say so and let the
        // allocations themselves be the gate instead of refusing with a misleading line
        std.debug.print("quant: {s}: the card memory probe failed; proceeding — the run estimates {d} (+1 GiB slack) and the allocations are the gate\n", .{
            std.mem.span(@as([*:0]const u8, @ptrCast(&namez))), need,
        });
    }
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
