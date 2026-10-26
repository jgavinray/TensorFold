//! xpu-eaq-ref — the dense reference logits (eaq-ldlq-spec-fable.md §5.1, §5.5):
//! the bf16 source through the layer-major machinery (calib.embedStream, 64 x
//! calib.layerForward over dense fp16 slabs uploaded from the source, calib.finalNorm),
//! then every position's fp32 head logits from the gate's own fp32 block GEMM
//! (kl.Head / eaq_gemm32) over the dense head slab — one head path for both sides of
//! the KL, so the reference and the candidate differ only in the pack's weights.
//!
//! Not bf16 arithmetic: fp16 between kernels, fp32 inside, as serving (§5.1). The
//! reference's distance from an fp64 forward is the gate's noise floor (T-KL-1), and
//! the whole point of this binary is that the forward is the same layerBody the
//! quantizer captures with — T-CAP-1's machinery, no capture hooks.
//!
//! Fail closed (eaq-review/ref-tool-ruling.md — the position-invariant-input ruling):
//! the tool carries its own ingestion and invariance gates and hard-aborts, writing
//! no byte of the logits file (and no sidecar) until every one of them has passed:
//!
//!  - tokens vary: per row the distinct token count exceeds a floor, row 0's first
//!    sixteen ids match the host file, and the device upload buffer reads back
//!    byte-identical to the host array (the copy-complete assert, before layer 0);
//!  - the card bisect: the --residual-dump run writes the per-position state for
//!    rows 0 and 1 at four points — after the embedding gather, after layer 0, after
//!    the middle layer, and the buffer the head GEMM receives after the final norm —
//!    and prints, per point, the mean cosine of positions against the row mean and
//!    the cross-row cosine; the first collapse names the broken stage;
//!  - head input varies: before every row's head GEMM, the cosine of the norm's
//!    first 64 positions against their mean must be below 0.99;
//!  - the head reads the norm's output by construction: finalNorm writes each row
//!    chunk's fp16 hidden and that same device buffer's handle is passed to the head
//!    GEMM (no device->host->device round trip on the GEMM's A path); a per-row
//!    checksum of that buffer is logged next to the row's achieved TFLOPS;
//!  - self-canary: after the head finishes row 0, teacher-forced top-1 against the
//!    row's own tokens under the shift logits[t] predicts token[t+1]; below 0.30 the
//!    run aborts before the logits file exists;
//!  - the §5.5 gate: canary band 0.45-0.85 on every row and on the aggregate, argmax
//!    mode under 10 percent, more than 2,000 distinct argmax tokens, top-5 not
//!    constant across rows, mean teacher-forced NLL under 4 nats with no row above
//!    6;
//!  - sidecar: <out>.json with the sha256 of the logits and of the prompt file, the
//!    shift convention, and every canary value — written only when every check is
//!    green; xpu-eaq-kl refuses a reference that lacks a passing sidecar.
//!
//! Output: raw little-endian fp32 [positions][VOCAB] in prompt order — exactly what
//! `xpu-eaq-kl --ref` consumes — plus its sha256 and the sidecar.
//!
//! The dense model, the kernel table and the slot plans are quant.zig's own
//! (buildDenseModel / initKernelTable / layerSlots), pub for exactly this reuse:
//! one body, no drift with the capture pass.
//!
//! This file is the RefFix candidate (the 973-line ruling-compliant version at
//! /tmp/reffix-handoff/) folded to its known-open items by KlFixRef: the real insertTop5,
//! head_tflops_row declared, logitsBlockDev called with the norm's device handle
//! (evWait inlined — Head.evWait is private), chunkOf over the live chunk slice,
//! residPoint's runtime label, the headInputCosine row-slice fix, Chunk at file scope,
//! and the prompt read hardened to pread-with-retry (the box's EFAULT/short-read
//! quirk; an early EOF against a known size is the ingestion bug, not a short read).

const std = @import("std");
const q = @import("qwen27");
const ze = q.ze;
const dev = q.dev;
const calib = q.calib;
const quant = @import("quant.zig");
const kl = @import("kl.zig");
const kl_kernels = @import("eaq_kl_kernels");
const Dev = dev.Dev;
const Checkpoint = dev.Checkpoint;
const D = q.D;
const VOCAB = q.VOCAB;
const LAYERS = q.LAYERS;
const alloc = q.alloc;

extern "c" fn open(path: [*:0]const u8, flags: c_int, mode: c_int) c_int;
extern "c" fn close(fd: c_int) c_int;
const CTimespec = extern struct { tv_sec: i64, tv_nsec: i64 };
extern "c" fn nanosleep(req: *const CTimespec, rem: ?*CTimespec) c_int;
extern "c" fn pwrite(fd: c_int, buf: [*]const u8, count: usize, off: i64) isize;
/// unlink with the path moved to a stack sentinel copy (the openOut pattern): the
/// caller holds plain []const u8 slices, the C entry wants [*:0].
fn unlink(path: []const u8) c_int {
    var stack: [4096]u8 = undefined;
    if (path.len == 0 or path.len >= stack.len) return -1;
    @memcpy(stack[0..path.len], path);
    stack[path.len] = 0;
    return std.c.unlink(@ptrCast(&stack));
}
const O_WRONLY_CREAT_TRUNC: c_int = 1 | 0o100 | 0o1000;

const Usage = error{Usage};
/// A gate went red: the tool prints the failing line itself before returning this.
const RefGate = error{RefGate};

/// The shift convention the self-canary, the gate stats and the sidecar all state.
const shift_convention = "logits[t] predicts token[t+1] within each row; a row's last position is dropped";

/// One row chunk's head-input geometry: the norm's device buffer (the GEMM's A by
/// handle) and the single downloaded copy that feeds the checksum and the cosine gate.
const Chunk = struct { at: u32, len: u32, hdev: *anyopaque, hbuf: []u16 };

/// The sidecar the comparator (xpu-eaq-kl) demands beside every reference.
const Sidecar = struct {
    tool: []const u8 = "xpu-eaq-ref",
    src: []const u8 = "",
    prompts: []const u8 = "",
    out: []const u8 = "",
    cols: u32 = 0,
    context: u32 = 0,
    rows: u32 = 0,
    positions: u64 = 0,
    vocab: u32 = 0,
    shift: []const u8 = shift_convention,
    logits_sha256: []const u8 = "",
    prompts_sha256: []const u8 = "",
    ids_dev_match: bool = false,
    ids_fnv_host: u64 = 0,
    ids_fnv_dev: u64 = 0,
    tokens_floor: u32 = 0,
    tokens_distinct_min: u32 = 0,
    head_input_cosine_max: f64 = 0,
    head_input_fnv_row: []const u64 = &.{},
    head_tflops_row: []const f64 = &.{},
    canary_row: []const f64 = &.{},
    canary_agg: f64 = 0,
    nll_row: []const f64 = &.{},
    nll_mean: f64 = 0,
    argmax_mode_token: u32 = 0,
    argmax_mode_frac: f64 = 0,
    distinct_argmax: usize = 0,
    top5_row: []const [5]u32 = &.{},
    top5_constant: bool = true,
    embed_ids: []const u32 = &.{},
    embed_ids_dev: []const u32 = &.{},
    pass: bool = false,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var src: []const u8 = "";
    var prompts_path: []const u8 = "";
    var out_path: []const u8 = "";
    var resid_dir: []const u8 = "";
    var cols: u32 = 2048;
    var ctx: u32 = q.MAX_CTX;
    var ai: usize = 1;
    while (ai < args.len) : (ai += 1) {
        const a = args[ai];
        if (std.mem.eql(u8, a, "--prompts")) {
            ai += 1;
            if (ai >= args.len) return error.Usage;
            prompts_path = args[ai];
        } else if (std.mem.eql(u8, a, "--out")) {
            ai += 1;
            if (ai >= args.len) return error.Usage;
            out_path = args[ai];
        } else if (std.mem.eql(u8, a, "--residual-dump")) {
            ai += 1;
            if (ai >= args.len) return error.Usage;
            resid_dir = args[ai];
        } else if (std.mem.eql(u8, a, "--cols")) {
            ai += 1;
            if (ai >= args.len) return error.Usage;
            cols = try std.fmt.parseInt(u32, args[ai], 10);
        } else if (std.mem.eql(u8, a, "--context")) {
            ai += 1;
            if (ai >= args.len) return error.Usage;
            ctx = try std.fmt.parseInt(u32, args[ai], 10);
        } else if (a.len > 0 and a[0] != '-') {
            if (src.len != 0) return error.Usage;
            src = a;
        } else return error.Usage;
    }
    if (src.len == 0 or prompts_path.len == 0 or out_path.len == 0) {
        std.debug.print(
            \\xpu-eaq-ref SRC_BF16_DIR --prompts eval.i32 --out ref_logits.f32 [--cols 2048] [--context N]
            \\  dense reference logits: layer-major forward over the bf16 source, fp32 head GEMM,
            \\  raw LE fp32 [positions][{d}] in prompt order + sha256 + the gate's <out>.json sidecar.
            \\  --residual-dump DIR adds the card bisect: rows 0 and 1 per-position state after the
            \\  embedding gather, layer 0, the middle layer and the head input, with the row-mean and
            \\  cross-row cosines that name the first collapse. Every invariant fails closed: no byte
            \\  of the logits file is written until all of them pass.
            \\
        , .{VOCAB});
        std.process.exit(2);
    }
    kl.refuseStudyModes(); // the same H7 refusals the gate runs under
    if (cols <= 16 or cols > 4096 or cols % 128 != 0) return error.Usage; // wide-only rows (H8), m tiles
    if (cols > ctx) return error.Usage;
    q.MAX_CTX = ctx;

    const t_start = q.now();
    std.debug.print("xpu-eaq-ref: source {s} prompts {s} cols {d} ctx {d}\n", .{ src, prompts_path, cols, ctx });
    const ids_bytes = try readAll(prompts_path);
    defer alloc.free(ids_bytes);
    if (ids_bytes.len % 4 != 0 or ids_bytes.len / 4 < cols or (ids_bytes.len / 4) % cols != 0) {
        std.debug.print("xpu-eaq-ref: {s} must hold a whole number of {d}-token rows\n", .{ prompts_path, cols });
        return error.Usage;
    }
    const ids_u32 = try alloc.alloc(u32, ids_bytes.len / 4); // aligned copy: bytesAsSlice of a u8 buffer is align(1)
    defer alloc.free(ids_u32);
    @memcpy(std.mem.sliceAsBytes(ids_u32), ids_bytes);
    const ids: []const u32 = ids_u32; // LE int32 as stored
    const rows: u32 = @intCast(ids.len / cols);
    const positions: u64 = @as(u64, rows) * cols;
    std.debug.print("xpu-eaq-ref: {d} rows x {d} = {d} positions -> {d:.2} GB of fp32 logits\n", .{
        rows, cols, positions, @as(f64, @floatFromInt(positions * VOCAB * 4)) / 1e9 });

    // -- host invariants, before the card (the ruling's step one): what the rows really hold.
    //    The device upload buffer is read back below, after the upload, as the copy-complete assert.
    const host_ids0 = ids[0..@min(16, cols)];
    const host_ids1 = if (rows > 1) ids[cols .. cols + @min(16, cols)] else host_ids0;
    std.debug.print("xpu-eaq-ref: host {s} row0 first16 {{", .{prompts_path});
    inlinePrint(host_ids0);
    std.debug.print("}}\n", .{});
    std.debug.print("xpu-eaq-ref: host {s} row1 first16 {{", .{prompts_path});
    inlinePrint(host_ids1);
    std.debug.print("}}\n", .{});
    // the tokens-vary floor: wide rows repeat, but a parsed-and-never-uploaded corpus shows
    // one or two distinct ids per row; anything at or under the floor is the ingestion bug.
    const tokens_floor: u32 = @max(32, cols / 64);
    var tokens_distinct_min: u32 = std.math.maxInt(u32);
    {
        const seen = try alloc.alloc(bool, VOCAB);
        defer alloc.free(seen);
        for (0..rows) |r| {
            @memset(seen, false);
            var distinct: u32 = 0;
            for (ids[@as(usize, r) * cols ..][0..cols]) |t| {
                if (t >= VOCAB) {
                    std.debug.print("xpu-eaq-ref: row {d} holds token {d} >= VOCAB {d}\n", .{ r, t, VOCAB });
                    return error.Usage;
                }
                if (!seen[t]) {
                    seen[t] = true;
                    distinct += 1;
                }
            }
            if (distinct <= tokens_floor) {
                std.debug.print("xpu-eaq-ref: row {d} has only {d} distinct tokens (floor {d}) — the prompt is position-invariant\n", .{ r, distinct, tokens_floor });
                return error.RefGate;
            }
            tokens_distinct_min = @min(tokens_distinct_min, distinct);
        }
    }
    std.debug.print("xpu-eaq-ref: gate tokens-vary PASS (min distinct {d} over floor {d}, all {d} rows)\n", .{ tokens_distinct_min, tokens_floor, rows });

    // -- the card and the dense model (quant.zig's own, pub for this reuse)
    // the serving path's stage dumps ride the same layerBody: XPU_DUMP names the dir
    if (std.c.getenv("XPU_DUMP")) |v| q.dump_dir = std.mem.span(v);
    var c: Checkpoint = .{};
    try quant.openSource(&c, src);
    const d = try alloc.create(Dev);
    d.* = .{ .g = try ze.Gpu.open(0xe223), .list = undefined, .k = undefined };
    errdefer d.g.close();
    d.list = try d.g.immediate(0, 0);
    if (d.g.memory()) |mm| {
        std.debug.print("xpu-eaq-ref: card free {d:.2} GB of {d:.2} GB\n", .{
            @as(f64, @floatFromInt(mm.free)) / 1e9, @as(f64, @floatFromInt(mm.size)) / 1e9 });
    } else |_| {}
    try quant.initKernelTable(d);
    const m = try alloc.create(q.Model);
    m.* = try quant.buildDenseModel(d, &c);
    const a = try alloc.create(q.Acts);
    a.* = try q.Acts.init(d, true);
    const slabs = try calib.Slabs.init(d);

    // -- the stream: every row resident at once (the quantizer's capture geometry)
    const stream = try d.alloc_(@as(usize, rows) * cols * D * 4);
    const ids_dev = try d.upload(std.mem.sliceAsBytes(ids)); // the gather below reads this very handle

    // -- the head: the gate's fp32 GEMM at the dense head slab (fp16 [5120, VOCAB])
    var head = try initDenseHead(d, &c);
    errdefer head.deinit();
    slabProbe(&head); // the KLStatistic probe: the head slab the GEMM reads, as the device holds it
    // XPU_REF_SPOT: bounded device-buffer hashes at the four ruling points plus the
    // slab fills, so one run names the collapse stage even without --residual-dump.
    const spot = std.c.getenv("XPU_REF_SPOT") != null;
    const resid_all = std.c.getenv("XPU_REF_RESID_ALL") != null;
    if (spot) std.debug.print("xpu-eaq-ref: XPU_REF_SPOT on — stream and slab hashes ride the run\n", .{});
    try spotLine(spot, &head, "slab after uploadDense (head lm_head, 4 MB prefix)", head.slab, 1 << 22);

    // -- pass 1: the embedding of every row (stateless per token). The gather indexes the
    //    uploaded ids_dev by this same handle (the ruling's fix): embedStream's ids argument
    //    is ids_dev + the chunk's byte offset, and embed_dense reads IDS[chunk-local t].
    {
        var r: u32 = 0;
        while (r < rows) : (r += 1) {
            var at: u32 = 0;
            while (at < cols) {
                const end: u32 = @intCast(q.chunkEnd(at, cols));
                try calib.embedStream(d, m, a,
                    dev.off(ids_dev, (@as(usize, r) * cols + at) * 4),
                    dev.off(stream, (@as(usize, r) * cols + at) * D * 4), end - at);
                at = end;
            }
        }
        try head.drainRunner(); // the whole embed queue is truly done, not just settle-returned
    }
    std.debug.print("xpu-eaq-ref: embedded {d} rows in {d:.1} s\n", .{ rows, q.now() - t_start });
    try spotLine(spot, &head, "post-embed stream row0 (4 MB prefix)", stream, 1 << 22);

    // -- the copy-complete assert, before layer 0 (the ruling): the upload buffer must hold
    //    the host array byte for byte, and rows 0 and 1 print exactly as they sit on the card.
    const ids_dev_u32 = try alloc.alloc(u32, ids_bytes.len / 4); // u32-aligned: bytesAsSlice would yield align(1)
    defer alloc.free(ids_dev_u32);
    const ids_dev_readback = std.mem.sliceAsBytes(ids_dev_u32);
    try head.download(ids_dev_readback, ids_dev, ids_bytes.len);
    const ids_fnv_host = fnv1a(ids_bytes);
    const ids_fnv_dev = fnv1a(ids_dev_readback);
    const dev_ids: []const u32 = ids_dev_u32;
    std.debug.print("xpu-eaq-ref: upload-buffer row0 first16 {{", .{});
    inlinePrint(dev_ids[0..@min(16, cols)]);
    std.debug.print("}}\n", .{});
    std.debug.print("xpu-eaq-ref: upload-buffer row1 first16 {{", .{});
    inlinePrint(if (rows > 1) dev_ids[cols .. cols + @min(16, cols)] else dev_ids[0..@min(16, cols)]);
    std.debug.print("}}\n", .{});
    std.debug.print("xpu-eaq-ref: ids fnv1a host {x} device {x}\n", .{ ids_fnv_host, ids_fnv_dev });
    if (ids_fnv_host != ids_fnv_dev or !std.mem.eql(u8, ids_dev_readback, ids_bytes)) {
        std.debug.print("xpu-eaq-ref: token upload NOT delivered — the device upload buffer differs from {s}\n", .{prompts_path});
        return error.RefGate;
    }
    if (!std.mem.eql(u32, dev_ids[0..16], host_ids0)) {
        std.debug.print("xpu-eaq-ref: row 0 first sixteen do not match the host file\n", .{});
        return error.RefGate;
    }
    std.debug.print("xpu-eaq-ref: gate copy-complete PASS (upload buffer == {s}, row 0 first sixteen match)\n", .{prompts_path});

    // -- the card bisect (the ruling's decisive diagnostic): rows 0 and 1 per-position state
    // at four points; the first point where both rows and every position collapse to one
    // vector names the broken stage. Cosines always print; the files ride --residual-dump.
    var collapse_stage: []const u8 = "none";
    const n_dump_rows: u32 = @min(rows, 2);
    const rbuf_len: usize = @as(usize, cols) * D; // one row's fp32 state
    const rbuf = try alloc.alloc(f32, 2 * rbuf_len);
    defer alloc.free(rbuf);
    const head_rbuf = try alloc.alloc(f32, 2 * rbuf_len); // the head-input rows, filled as pass 3 streams them
    defer alloc.free(head_rbuf);
    var head_rbuf_filled: u32 = 0;
    var head_cos_max: f64 = 0; // the max over rows of the first-64 cosine against their mean
    var head_fnv = try alloc.alloc(u64, rows);
    defer alloc.free(head_fnv);
    @memset(head_fnv, 0);

    // point 1: after the embedding gather
    try downloadStreamRows(&head, stream, rbuf[0..rbuf_len], rbuf[rbuf_len..], n_dump_rows, cols);
    if (try residPoint(&head, resid_dir, "embed", "prompt ingestion or embedding gather", rbuf[0..rbuf_len], rbuf[rbuf_len..], n_dump_rows, cols)) |stage| collapse_stage = stage;

    // -- pass 2: the layer-major sweep; one slab fill per layer, then every row through it
    for (0..LAYERS) |i| {
        const tl = q.now();
        const sp = try quant.layerSlots(alloc, i);
        for (&sp, 0..) |s, slot| {
            const slab = slabOf(slabs, slot);
            for (s.parts) |pp| {
                const t = try c.get(try std.fmt.allocPrint(alloc, "{s}.weight", .{pp.base}));
                if (!std.mem.eql(u8, t.dtype, "BF16")) {
                    std.debug.print("xpu-eaq-ref: {s}.weight is {s}, expected BF16\n", .{ pp.base, t.dtype });
                    return error.Usage;
                }
                try calib.uploadDense(d, t.bytes, pp.n, s.k, s.nt, pp.n0, slab);
            }
        }
        if (i == 0) for (0..4) |slot| try spotLine(spot, &head,
            try std.fmt.allocPrint(alloc, "slab after uploadDense (layer 0 slot {d}, 4 MB prefix)", .{slot}),
            slabOf(slabs, @intCast(slot)), 1 << 22);
        for (0..rows) |r| {
            const bo: usize = @as(usize, r) * cols * D * 4;
            var at: u32 = 0;
            while (at < cols) {
                const end: u32 = @intCast(q.chunkEnd(at, cols));
                try calib.layerForward(d, m, a, i, slabs,
                    dev.off(stream, bo + @as(usize, at) * D * 4),
                    dev.off(stream, bo + @as(usize, at) * D * 4),
                    end - at, at, at == 0, null);
                at = end;
            }
        }
        // the bisect after layer 0 and after the middle layer (the 32nd of 64, index 31);
        // XPU_REF_RESID_ALL dumps after every layer, so the host ladder can name the
        // first divergent layer without a second run
        if (i == 0 or i == LAYERS / 2 - 1 or resid_all) {
            try head.drainRunner();
            const label = if (i == 0) "after_layer_1" else if (i == LAYERS / 2 - 1) "after_layer_32" else
                try std.fmt.allocPrint(alloc, "after_layer_{d}", .{i + 1});
            const stage = if (i == 0) "layer 0 (its slab fill, GEMMs, mixer or state reset)" else "the layer-major sweep through the middle of the stack";
            try downloadStreamRows(&head, stream, rbuf[0..rbuf_len], rbuf[rbuf_len..], n_dump_rows, cols);
            if (try residPoint(&head, resid_dir, label, stage, rbuf[0..rbuf_len], rbuf[rbuf_len..], n_dump_rows, cols)) |s| {
                if (std.mem.eql(u8, collapse_stage, "none")) collapse_stage = s;
            }
        }
        if (i % 8 == 0 or i == LAYERS - 1)
            std.debug.print("  layer {d}/64 done ({d:.1} s)\n", .{ i, q.now() - tl });
    }
    try spotLine(spot, &head, "post-layer-sweep stream row0 (4 MB prefix)", stream, 1 << 22);

    // -- pass 3: the final norm and the fp32 head per chunk; every position contributes.
    // The norm writes each row chunk's fp16 hidden and that same buffer's handle goes to
    // the head GEMM (by construction, not by a copy). Each row's logits complete in a host
    // buffer; the row's gates run on it; only then do its bytes go to the file — which the
    // tool does not even open until row 0's self-canary has passed.
    const row_buf = try alloc.alloc(f32, @as(usize, cols) * VOCAB); // one row's logits
    defer alloc.free(row_buf);
    const gcounts = try alloc.alloc(u32, VOCAB); // global argmax histogram
    defer alloc.free(gcounts);
    @memset(gcounts, 0);
    const rcounts = try alloc.alloc(u32, VOCAB); // the row's histogram, reused
    defer alloc.free(rcounts);
    const canary_row = try alloc.alloc(f64, rows);
    defer alloc.free(canary_row);
    const nll_row = try alloc.alloc(f64, rows);
    defer alloc.free(nll_row);
    const top5_row = try alloc.alloc([5]u32, rows);
    defer alloc.free(top5_row);
    const head_tflops_row = try alloc.alloc(f64, rows);
    defer alloc.free(head_tflops_row);

    // the row's chunk geometry (identical for every row)
    var chunks: [8]Chunk = undefined;
    var nchunks: usize = 0;
    {
        var at: u32 = 0;
        while (at < cols) {
            const end: u32 = @intCast(q.chunkEnd(at, cols));
            const T = end - at;
            chunks[nchunks] = .{ .at = at, .len = T,
                .hdev = try d.alloc_(@as(usize, T) * D * 2),
                .hbuf = try alloc.alloc(u16, @as(usize, T) * D) };
            nchunks += 1;
            at = end;
        }
    }

    var out_fd: c_int = -1;
    defer {
        if (out_fd >= 0) _ = close(out_fd);
    }
    var write_off: i64 = 0;
    var top1_total: u64 = 0;
    var n_total: u64 = 0;

    const tf = q.now();
    for (0..rows) |r| {
        const bo: usize = @as(usize, r) * cols * D * 4;
        // the norm first: every chunk's fp16 hidden lands on device (that very buffer is the
        // GEMM's A) and downloads once, for the checksum and the head-input cosine gate.
        var row_fnv: u64 = 0xcbf29ce484222325;
        var chunk0_hbuf: []const u16 = &.{};
        for (chunks[0..nchunks]) |*ch| {
            try calib.finalNorm(d, m, a, dev.off(stream, bo + @as(usize, ch.at) * D * 4), ch.hdev, ch.len, null);
            try head.drainRunner(); // the finalNorm launches ride the runner list: make them truly done
            try head.download(std.mem.sliceAsBytes(ch.hbuf), ch.hdev, @as(usize, ch.len) * D * 2);
            row_fnv = fnv1a(std.mem.sliceAsBytes(ch.hbuf));
            if (ch.at == 0) chunk0_hbuf = ch.hbuf;
        }
        if (r == 0) {
            if (std.c.getenv("XPU_DUMP")) |dirz| { // the norm's own input, for the split forensics
                const xn = try alloc.alloc(f32, @as(usize, chunks[0].len) * D);
                defer alloc.free(xn);
                try head.drainRunner();
                try head.download(std.mem.sliceAsBytes(xn), dev.off(stream, bo), @as(usize, chunks[0].len) * D * 4);
                const path = try std.fmt.allocPrintSentinel(alloc, "{s}/xpu_s63.f32", .{std.mem.span(dirz)}, 0);
                const fd = open(path.ptr, O_WRONLY_CREAT_TRUNC, 0o644);
                if (fd >= 0) {
                    var wb: [*]const u8 = @ptrCast(xn.ptr);
                    var left: usize = xn.len * 4;
                    while (left > 0) {
                        const w = std.c.write(fd, wb, left);
                        if (w <= 0) break;
                        wb += @intCast(w);
                        left -= @intCast(w);
                    }
                    _ = close(fd);
                }
            }
            try spotLine(spot, &head, "hbuf after finalNorm (chunk 0, 4 MB prefix)",
                chunks[0].hdev, @min(@as(usize, chunks[0].len) * D * 2, 1 << 22));
        }
        head_fnv[r] = row_fnv;
        // head input varies (the ruling): the norm's first 64 positions against their mean,
        // before this row's head GEMM. A frozen hidden vector across positions is the
        // position-invariant input this ruling was written for.
        const n64: usize = @min(@as(usize, 64), chunk0_hbuf.len / D); // explicit usize: bare-literal @min can infer u7 from the comptime shape
        const cos64 = headInputCosine(chunk0_hbuf[0 .. n64 * D], n64);
        head_cos_max = @max(head_cos_max, cos64);
        if (cos64 >= 0.99) {
            std.debug.print("xpu-eaq-ref: row {d} head input is frozen: cosine(first 64, their mean) {d:.4} >= 0.99 — broken stage: the head input buffer (the norm output the GEMM reads)\n", .{ r, cos64 });
            if (out_fd >= 0) { _ = unlink(out_path); out_fd = -1; }
            return error.RefGate;
        }
        // keep rows 0 and 1's full normed hidden for the bisect's fourth point
        if (r < 2) {
            for (chunks[0..nchunks]) |ch| {
                const base = r * rbuf_len + @as(usize, ch.at) * D;
                for (ch.hbuf, 0..) |hv, i| head_rbuf[base + i] = @floatCast(@as(f16, @bitCast(hv)));
            }
            if (r == 1) head_rbuf_filled = 2;
            // the head-input bisect line prints once both rows are in (or at row 0 if there is only one)
            if ((r == 1 and head_rbuf_filled == 2) or (r == 0 and rows == 1)) {
                if (try residPoint(&head, resid_dir, "head_input", "the final norm's output buffer the head GEMM reads (upstream: the layer sweep or the norm itself)",
                    head_rbuf[0..rbuf_len], head_rbuf[rbuf_len .. 2 * rbuf_len], n_dump_rows, cols)) |stage| {
                    if (std.mem.eql(u8, collapse_stage, "none")) collapse_stage = stage;
                }
            }
        }
        // the head GEMM per 128-row launch over the norm's device buffer itself
        var head_secs: f64 = 0;
        var done: u32 = 0;
        @memset(rcounts[0..VOCAB], 0);
        while (done < @as(u32, @intCast(cols))) {
            const m_blk = @min(@as(u32, 1024), @as(u32, cols) - done);
            const ch = chunkOf(chunks[0..nchunks], done);
            const local = done - ch.at; // the block's first row inside this chunk's hidden
            const t0 = q.now();
            try logitsBlockDev(&head, ch.hdev, std.mem.sliceAsBytes(ch.hbuf[0 .. @as(usize, ch.len) * D]),
                local, m_blk, row_buf[@as(usize, done) * VOCAB ..][0 .. @as(usize, m_blk) * VOCAB]);
            head_secs += q.now() - t0;
            statsBlock(row_buf[@as(usize, done) * VOCAB ..][0 .. @as(usize, m_blk) * VOCAB], m_blk, done, cols,
                ids[@as(usize, r) * cols ..][0..cols], gcounts, rcounts, &top1_total, &n_total);
            done += m_blk;
        }
        const tflops = @as(f64, @floatFromInt(2 * @as(u64, cols) * D * VOCAB)) / 1e12 / @max(head_secs, 1e-6);
        head_tflops_row[r] = tflops;
        std.debug.print("  ref row {d}/{d}: head-input fnv1a {x} at {d:.1} TFLOPS (head-input cos64 {d:.4}) ({d:.1} s)\n", .{
            r + 1, rows, row_fnv, tflops, cos64, q.now() - tf });

        // the row's gates, on its logits, before its bytes go anywhere
        const n_row: u64 = cols - 1; // the shift drops the row's last position
        canary_row[r] = @as(f64, @floatFromInt(top1_row)) / @as(f64, @floatFromInt(n_row));
        nll_row[r] = nll_row_sum / @as(f64, @floatFromInt(n_row));
        // top5 and distinct from this row's histogram
        var top5: [5]u32 = .{ std.math.maxInt(u32), std.math.maxInt(u32), std.math.maxInt(u32), std.math.maxInt(u32), std.math.maxInt(u32) };
        var distinct: usize = 0;
        for (rcounts[0..VOCAB], 0..) |cnt, t| {
            if (cnt == 0) continue;
            distinct += 1;
            insertTop5(&top5, rcounts, @intCast(t));
        }
        for (&top5) |*t| {
            if (t.* == std.math.maxInt(u32)) t.* = 0;
        }
        top5_row[r] = top5;
        top1_row = 0; nll_row_sum = 0; // statsBlock's per-row accumulators reset for the next row

        if (r == 0) {
            // the self-canary: row 0's teacher-forced top-1 against its own tokens; the
            // logits file is not created until this clears 0.30.
            if (canary_row[0] < 0.30) {
                std.debug.print("xpu-eaq-ref: self-canary FAIL: row 0 teacher-forced top-1 {d:.4} < 0.30 under shift {s}\n", .{ canary_row[0], shift_convention });
                return error.RefGate;
            }
            out_fd = openOut(out_path); // the first byte of the logits file exists from here on
            std.debug.print("xpu-eaq-ref: self-canary PASS (row 0 top-1 {d:.4} >= 0.30) — opening {s}\n", .{ canary_row[0], out_path });
        }
        if (canary_row[r] < 0.45 or canary_row[r] > 0.85) {
            std.debug.print("xpu-eaq-ref: gate canary-band FAIL: row {d} teacher-forced top-1 {d:.4} outside 0.45-0.85\n", .{ r, canary_row[r] });
            if (out_fd >= 0) { _ = unlink(out_path); out_fd = -1; }
            return error.RefGate;
        }
        if (nll_row[r] > 6.0) {
            std.debug.print("xpu-eaq-ref: gate loss FAIL: row {d} mean teacher-forced NLL {d:.3} > 6 nats\n", .{ r, nll_row[r] });
            if (out_fd >= 0) { _ = unlink(out_path); out_fd = -1; }
            return error.RefGate;
        }
        if (out_fd >= 0) { // the row passed its gates: append its bytes
            const b = std.mem.sliceAsBytes(row_buf[0 .. @as(usize, cols) * VOCAB]);
            var offb: usize = 0;
            while (offb < b.len) {
                const w = pwrite(out_fd, b.ptr + offb, b.len - offb, write_off + @as(i64, @intCast(offb)));
                if (w <= 0) return error.FileIo;
                offb += @intCast(w);
            }
            write_off += @as(i64, @intCast(b.len));
        }
        if (r % 2 == 1) std.debug.print("  ref rows {d}/{d} ({d:.1} s)\n", .{ r + 1, rows, q.now() - tf });
    }
    if (@as(u64, @bitCast(write_off)) != positions * VOCAB * 4) {
        std.debug.print("xpu-eaq-ref: wrote {d} bytes, expected {d}\n", .{ write_off, positions * VOCAB * 4 });
        return error.FileIo;
    }
    std.debug.print("xpu-eaq-ref: {s} {d} bytes in {d:.1} s (head GEMM {d:.1} s)\n", .{
        out_path, write_off, q.now() - t_start, head.gemm_s });

    // -- the aggregate gate on every row's stats (the ruling's gate, before any KL row is read)
    var agg_top1: f64 = 0;
    {
        var sum: f64 = 0;
        for (canary_row[0..rows]) |t| sum += t;
        agg_top1 = sum / @as(f64, @floatFromInt(rows));
    }
    if (agg_top1 < 0.45 or agg_top1 > 0.85) {
        std.debug.print("xpu-eaq-ref: gate canary-band FAIL: aggregate teacher-forced top-1 {d:.4} outside 0.45-0.85\n", .{agg_top1});
        _ = unlink(out_path);
        return error.RefGate;
    }
    var nll_mean: f64 = 0;
    for (nll_row[0..rows]) |v| nll_mean += v;
    nll_mean /= @as(f64, @floatFromInt(rows));
    if (nll_mean >= 4.0) {
        std.debug.print("xpu-eaq-ref: gate loss FAIL: mean teacher-forced NLL {d:.3} >= 4 nats\n", .{nll_mean});
        _ = unlink(out_path);
        return error.RefGate;
    }
    var gdistinct: usize = 0;
    var gmode: u32 = 0;
    var gmode_tok: u32 = 0;
    for (gcounts[0..VOCAB], 0..) |cnt, t| {
        if (cnt == 0) continue;
        gdistinct += 1;
        if (cnt > gmode) {
            gmode = cnt;
            gmode_tok = @intCast(t);
        }
    }
    const gmode_frac = @as(f64, @floatFromInt(gmode)) / @as(f64, @floatFromInt(positions));
    if (gmode_frac >= 0.10) {
        std.debug.print("xpu-eaq-ref: gate invariance FAIL: argmax mode token {d} at {d:.3} of positions >= 0.10\n", .{ gmode_tok, gmode_frac });
        _ = unlink(out_path);
        return error.RefGate;
    }
    if (gdistinct <= 2000) {
        std.debug.print("xpu-eaq-ref: gate invariance FAIL: {d} distinct argmax tokens <= 2,000\n", .{gdistinct});
        _ = unlink(out_path);
        return error.RefGate;
    }
    var top5_constant = true;
    for (top5_row[1..rows]) |t5| {
        if (!std.mem.eql(u32, &t5, &top5_row[0])) top5_constant = false;
    }
    if (top5_constant) {
        std.debug.print("xpu-eaq-ref: gate invariance FAIL: top-5 argmax set {{", .{});
        inlinePrint(&top5_row[0]);
        std.debug.print("}} constant across all rows\n", .{});
        _ = unlink(out_path);
        return error.RefGate;
    }
    std.debug.print("xpu-eaq-ref: gate no-invariance PASS (mode {d} at {d:.4}, {d} distinct argmax, top-5 varies across {d} rows)\n", .{
        gmode_tok, gmode_frac, gdistinct, rows });
    std.debug.print("xpu-eaq-ref: gate canary PASS (per row {d:.4}-{d:.4}, aggregate {d:.4}; NLL mean {d:.3}, max row {d:.3})\n", .{
        minOf(canary_row[0..rows]), maxOf(canary_row[0..rows]), agg_top1, nll_mean, maxOf(nll_row[0..rows]) });
    if (!std.mem.eql(u8, collapse_stage, "none")) {
        std.debug.print("xpu-eaq-ref: bisect: collapse named at {s}\n", .{collapse_stage});
    } else {
        std.debug.print("xpu-eaq-ref: bisect: no collapse at any of the four points\n", .{});
    }

    const hex = try kl.sha256Hex(out_path);
    std.debug.print("  sha256 {s}\n", .{hex});
    const prompthex = try kl.sha256Hex(prompts_path);
    { // the sidecar the comparator demands and the gate's spot-check workflow picks up
        const sc = Sidecar{
            .src = src,
            .prompts = prompts_path,
            .out = out_path,
            .cols = cols,
            .context = ctx,
            .rows = rows,
            .positions = positions,
            .vocab = VOCAB,
            .logits_sha256 = hex,
            .prompts_sha256 = prompthex,
            .ids_dev_match = true,
            .ids_fnv_host = ids_fnv_host,
            .ids_fnv_dev = ids_fnv_dev,
            .tokens_floor = tokens_floor,
            .tokens_distinct_min = tokens_distinct_min,
            .head_input_cosine_max = head_cos_max,
            .head_input_fnv_row = head_fnv[0..rows],
            .head_tflops_row = head_tflops_row[0..rows],
            .canary_row = canary_row[0..rows],
            .canary_agg = agg_top1,
            .nll_row = nll_row[0..rows],
            .nll_mean = nll_mean,
            .argmax_mode_token = gmode_tok,
            .argmax_mode_frac = gmode_frac,
            .distinct_argmax = gdistinct,
            .top5_row = top5_row[0..rows],
            .top5_constant = top5_constant,
            .embed_ids = host_ids0,
            .embed_ids_dev = dev_ids[0..@min(16, cols)],
            .pass = true,
        };
        const js = try std.json.Stringify.valueAlloc(alloc, sc, .{ .whitespace = .indent_1 });
        const spath = try std.fmt.allocPrintSentinel(alloc, "{s}.json", .{out_path}, 0);
        const sfd = openOut(spath);
        defer _ = close(sfd);
        if (pwrite(sfd, js.ptr, js.len, 0) != @as(isize, @bitCast(js.len))) return error.FileIo;
        std.debug.print("  sidecar {s}.json (sha256 logits {s} prompts {s})\n", .{ out_path, hex, prompthex });
    }
    { // the sidecar the gate's spot-check workflow picks up beside the logits
        var sidecar: [4096]u8 = undefined;
        const sp = std.fmt.bufPrint(&sidecar, "{s}.sha256", .{out_path}) catch return error.FileIo;
        const fd = openOut(sp);
        defer _ = close(fd);
        const line = try std.fmt.allocPrint(alloc, "{s}  {s}\n", .{ hex, out_path });
        defer alloc.free(line);
        if (pwrite(fd, line.ptr, line.len, 0) != @as(isize, @bitCast(line.len))) return error.FileIo;
    }
}

// ---------------------------------------------------------------- the bisect

/// Download rows 0 and 1 (up to n_dump_rows of them) of the fp32 stream, one row per
/// buffer: the per-position state at one of the four bisect points.
fn downloadStreamRows(head: *kl.Head, stream: *anyopaque, row0: []f32, row1: []f32, n_rows: u32, cols: u32) !void {
    const bytes = @as(usize, cols) * D * 4;
    try head.drainRunner(); // the runner list's settle lies on a busy card; the event does not
    try head.download(std.mem.sliceAsBytes(row0), dev.off(stream, 0), if (n_rows > 0) bytes else 0);
    try head.download(std.mem.sliceAsBytes(row1), dev.off(stream, @as(usize, cols) * D * 4), if (n_rows > 1) bytes else 0);
}

/// Mean cosine between every position's vector and the row's mean vector, and the cosine
/// between the two rows at the same positions; prints the point's line (checksums and
/// magnitudes ride with it), writes the fp32 dumps when resid_dir is set, and returns the
/// stage name when both rows and every position have collapsed to one vector.
fn residPoint(head: *kl.Head, resid_dir: []const u8, label: []const u8, stage: []const u8,
              row0: []const f32, row1: []const f32, n_rows: u32, cols: u32) !?[]const u8 {
    _ = head;
    const r0 = row0[0 .. @as(usize, cols) * D];
    const r1 = row1[0 .. @as(usize, cols) * D];
    const mc0 = meanCos(r0, cols);
    const mc1 = if (n_rows > 1) meanCos(r1, cols) else std.math.nan(f32);
    const cross = if (n_rows > 1) crossCos(r0, r1, cols) else std.math.nan(f32);
    const ma0 = meanAbs(r0);
    const ma1 = if (n_rows > 1) meanAbs(r1) else std.math.nan(f32);
    std.debug.print("  bisect {s}: row0 meancos {d:.4} meanabs {d:.4} fnv {x} | row1 meancos {d:.4} meanabs {d:.4} fnv {x} | crosscos {d:.4}\n", .{
        label, mc0, ma0, fnv1a(std.mem.sliceAsBytes(r0)), mc1, ma1, fnv1a(std.mem.sliceAsBytes(r1)), cross });
    if (resid_dir.len != 0) {
        try writeF32(resid_dir, label, 0, r0);
        if (n_rows > 1) try writeF32(resid_dir, label, 1, r1);
    }
    const eps = 0.0001;
    if (n_rows > 1 and mc0 > 0.99 - eps and mc1 > 0.99 - eps and cross > 0.99 - eps and ma0 > eps and ma1 > eps) {
        std.debug.print("  bisect {s}: COLLAPSE — both rows and all positions are one vector; broken stage: {s}\n", .{ label, stage });
        return stage;
    }
    return null;
}

fn writeF32(dir: []const u8, label: []const u8, row: usize, v: []const f32) !void {
    const path = try std.fmt.allocPrintSentinel(alloc, "{s}/xpu_resid_{s}_r{d}.f32", .{ dir, label, row }, 0);
    const fd = openOut(path);
    defer _ = close(fd);
    const b = std.mem.sliceAsBytes(v);
    var off: usize = 0;
    while (off < b.len) {
        const w = pwrite(fd, b.ptr + off, b.len - off, @intCast(off));
        if (w <= 0) return error.FileIo;
        off += @intCast(w);
    }
}

fn meanCos(x: []const f32, cols: u32) f32 {
    if (x.len == 0) return std.math.nan(f32);
    const T: usize = cols;
    var sum = alloc.alloc(f32, D) catch return std.math.nan(f32);
    defer alloc.free(sum);
    @memset(sum, 0);
    for (0..T) |t| {
        for (0..D) |d| {
            sum[d] += x[t * D + d];
        }
    }
    var nrm: f64 = 0;
    for (sum) |s| nrm += @as(f64, s) * s;
    const mu_nrm: f32 = @floatCast(@sqrt(nrm));
    if (mu_nrm == 0) return 0;
    var acc: f64 = 0;
    for (0..T) |t| {
        const row = x[t * D ..][0..D];
        var dotv: f64 = 0;
        var nn: f64 = 0;
        for (row, 0..) |v, d| {
            dotv += @as(f64, v) * sum[d];
            nn += @as(f64, v) * v;
        }
        const n1 = @sqrt(nn);
        if (n1 > 1e-30) acc += dotv / (n1 * @as(f64, mu_nrm));
    }
    return @floatCast(acc / @as(f64, @floatFromInt(T)));
}

fn crossCos(a: []const f32, b: []const f32, cols: u32) f32 {
    const T: usize = cols;
    var acc: f64 = 0;
    for (0..T) |t| {
        const ra = a[t * D ..][0..D];
        const rb = b[t * D ..][0..D];
        var dotv: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        for (ra, rb) |x, y| {
            dotv += @as(f64, x) * y;
            na += @as(f64, x) * x;
            nb += @as(f64, y) * y;
        }
        if (na > 1e-30 and nb > 1e-30) acc += dotv / (@sqrt(na) * @sqrt(nb));
    }
    return @floatCast(acc / @as(f64, @floatFromInt(T)));
}

fn meanAbs(x: []const f32) f32 {
    if (x.len == 0) return std.math.nan(f32);
    var acc: f64 = 0;
    for (x) |v| acc += @abs(v);
    return @floatCast(acc / @as(f64, @floatFromInt(x.len)));
}

/// The head-input gate: cosine of the norm's first n positions against their mean, on the
/// fp16 hidden the head GEMM is about to read (the same rows, the same bytes).
fn headInputCosine(h: []const u16, n: usize) f64 {
    if (n == 0 or h.len != n * D) return 1;
    var sum = alloc.alloc(f64, D) catch return 1;
    defer alloc.free(sum);
    @memset(sum, 0);
    const f = alloc.alloc(f64, h.len) catch return 1;
    defer alloc.free(f);
    for (h, 0..) |hv, i| {
        const v: f64 = @floatCast(@as(f16, @bitCast(hv)));
        f[i] = v;
        sum[i % D] += v;
    }
    var mnm: f64 = 0;
    for (sum) |s| mnm += s * s;
    const mu = @sqrt(mnm);
    if (mu == 0) return 1;
    var acc: f64 = 0;
    for (0..n) |t| {
        const row = f[(t * D)..(t * D + D)];
        var dotv: f64 = 0;
        var nn: f64 = 0;
        for (row, 0..) |v, d| {
            dotv += v * sum[d];
            nn += v * v;
        }
        if (nn > 1e-30) acc += dotv / (@sqrt(nn) * mu);
    }
    return acc / @as(f64, @floatFromInt(n));
}

// ---------------------------------------------------------------- the head, by handle

/// kl.Head's fp32 block GEMM over a device-resident A: the final norm's own buffer is the
/// GEMM's input by construction (logitsBlock's event discipline — 128-row launches, the
/// same-list fence behind the GEMM's event, the slice downloads on the copy list behind
/// it — without the host round trip on the A path). `hbuf` is the chunk's normed fp16
/// bytes, already downloaded once for the checksum and the cosine gate; `local` is the
/// block's first row inside the chunk. kl.Head.evWait is private, so the wait is inlined
/// here exactly the way evWait does it.
fn logitsBlockDev(h: *kl.Head, a_dev: *anyopaque, hbuf: []const u8, local: u32, m: u32, out: []f32) !void {
    std.debug.assert(out.len == @as(usize, m) * h.nt);
    std.debug.assert(hbuf.len >= @as(usize, local + m) * h.k * 2); // the block's rows live in this chunk
    const g = &h.d.g;
    // The norm wrote A on the runner's list; drainRunner proved it done before the
    // checksum download, so the GEMM may read the norm's buffer immediately.
    const gY: u32 = @intCast(h.nt / 128);
    const fence_x: u32 = @min(@divCeil(@as(u32, h.block) * h.k, 1024), 8192);
    var c: u32 = 0;
    while (c < m) : (c += 128) {
        const mc: u32 = @min(128, m - c);
        const a_ptr = @as([*]const u16, @ptrFromInt(@intFromPtr(a_dev) + @as(usize, local + c) * h.k * 2));
        const c_ptr = @as([*]f32, @ptrFromInt(@intFromPtr(h.c) + @as(usize, c) * h.nt * 4));
        inline for (.{
            a_ptr, h.slab, c_ptr,
            @as(i32, @intCast(mc)), @as(i32, @intCast(h.nt)), @as(i32, @intCast(h.k)),
        }, 0..) |a, i| try g.arg(h.gemm, i, a);
        _ = g.api.zeEventHostReset(h.ev[0]);
        try ze.check(g.api.zeCommandListAppendLaunchKernel(h.gl, h.gemm, &ze.GroupCount{ .x = 1, .y = gY }, h.ev[0], 0, null));
        { // the fence copy rides the same list behind the GEMM's event (into the GEMM's
          // A scratch, which this path never reads): its retirement proves the GEMM's
          // stores flushed — the same proof logitsBlock's fence carries.
            inline for (.{ h.a, h.a_stage, @as(u32, @intCast(@divExact(@as(usize, h.block) * h.k * 2, 2))) }, 0..) |aa, i| try g.arg(h.copy16, i, aa);
            _ = g.api.zeEventHostReset(h.ev[6]);
            var w0 = [1]ze.H{h.ev[0]};
            try ze.check(g.api.zeCommandListAppendLaunchKernel(h.gl, h.copy16, &ze.GroupCount{ .x = fence_x, .y = 1 }, h.ev[6], 1, &w0));
            try evWait(h, h.ev[6]);
        }
        var done: usize = @as(usize, c) * h.nt;
        const stop = @as(usize, c + mc) * h.nt;
        while (done < stop) {
            const n = @min(@as(usize, 64 << 20), stop - done);
            _ = g.api.zeEventHostReset(h.ev[1]);
            var w0 = [1]ze.H{h.ev[0]}; // the copy engine waits on the GEMM itself
            try ze.check(g.api.zeCommandListAppendMemoryCopy(h.dl, h.stage,
                @ptrFromInt(@intFromPtr(h.c) + done * 4), n * 4, h.ev[1], 1, &w0));
            try evWait(h, h.ev[1]);
            @memcpy(std.mem.sliceAsBytes(out[done .. done + n]), h.stage[0 .. n * 4]);
            done += n;
        }
    }
}

fn evWait(h: *kl.Head, ev: ze.H) !void {
    const deadline = q.now() + 600; // seconds, like kl.Head.evWait
    while (true) {
        if (h.d.g.api.zeEventHostSynchronize(ev, 300_000_000_000) == 0) return;
        if (q.now() > deadline) return error.DeviceLost;
    }
}

/// XPU_REF_SPOT: a bounded device-buffer prefix hashed and zero-counted on the host,
/// so one run names the collapse stage without the residual files. drainRunner
/// first: the hash must read settled bytes (the runner list's settle lies on a busy card).
/// XPU_REF_SPOT_FILES names a directory: the prefix bytes join the hash there, so a
/// mismatch can be diffed tile by tile instead of guessed at (a hash names the stage,
/// the bytes name the bug).
fn spotLine(spot: bool, head: *kl.Head, name: []const u8, ptr: *anyopaque, nbytes: usize) !void {
    if (!spot) return;
    const buf = try alloc.alloc(u8, nbytes);
    defer alloc.free(buf);
    try head.drainRunner();
    try head.download(buf, ptr, nbytes);
    var zeros: usize = 0;
    for (buf) |b| {
        if (b == 0) zeros += 1;
    }
    std.debug.print("  spot {s}: fnv1a {x} ({d} bytes, {d} zero bytes)\n", .{ name, fnv1a(buf), nbytes, zeros });
    if (std.c.getenv("XPU_REF_SPOT_FILES")) |dirz| {
        const dir = std.mem.span(dirz);
        if (dir.len > 0) {
            const label = try alloc.alloc(u8, name.len);
            defer alloc.free(label);
            for (name, 0..) |ch, k| label[k] = if ((ch >= 'a' and ch <= 'z') or (ch >= 'A' and ch <= 'Z') or
                (ch >= '0' and ch <= '9'))
                    ch
                else
                    '_';
            const path = try std.fmt.allocPrint(alloc, "{s}/spot_{s}.bin", .{ dir, label });
            defer alloc.free(path);
            const fd = openOut(path);
            if (fd >= 0) {
                var off: usize = 0;
                while (off < buf.len) {
                    const w = pwrite(fd, buf.ptr + off, buf.len - off, @intCast(off));
                    if (w <= 0) break;
                    off += @intCast(w);
                }
                _ = close(fd);
            }
        }
    }
}

/// The row chunk whose rows include the block's first row.
fn chunkOf(chunks: []const Chunk, done: u32) Chunk {
    for (chunks) |ch| {
        if (done >= ch.at and done < ch.at + ch.len) return ch;
    }
    unreachable;
}

// ---------------------------------------------------------------- the gate's row stats

var top1_row: u64 = 0;
var nll_row_sum: f64 = 0;

/// One block's logits against the row's own tokens: argmax histogram (row + global) and
/// the teacher-forced top-1 and NLL under the shift logits[t] predicts token[t+1]. The
/// row's last position has no next token and is dropped (the sidecar's shift convention).
fn statsBlock(logits: []const f32, m_blk: u32, done: u32, cols: u32, row_ids: []const u32,
              gcounts: []u32, rcounts: []u32, top1_total: *u64, n_total: *u64) void {
    std.debug.assert(logits.len == @as(usize, m_blk) * VOCAB);
    for (0..m_blk) |i| {
        const p = done + @as(u32, @intCast(i)); // the position in the row
        const lg = logits[i * VOCAB ..][0..VOCAB];
        var best: u32 = 0;
        var bv = lg[0];
        for (lg[1..], 1..) |v, j| {
            if (v > bv) {
                bv = v;
                best = @intCast(j);
            }
        }
        gcounts[best] += 1;
        rcounts[best] += 1;
        if (p + 1 >= cols) continue; // the shift: the row's last position is dropped
        const next = row_ids[p + 1];
        // the softmax's log-partition: a second pass around the running max, stable
        var sum: f64 = 0;
        for (lg) |v| sum += @exp(@as(f64, v) - @as(f64, bv));
        const logz = @as(f64, bv) + @log(sum);
        const nll = logz - @as(f64, lg[next]);
        if (best == next) {
            top1_row += 1;
            top1_total.* += 1;
        }
        nll_row_sum += nll;
        n_total.* += 1;
    }
}

/// The row's top-5 argmax tokens: the caller feeds every token with a nonzero count once;
/// insertion keeps the five highest-count tokens, descending (maxInt = unfilled, mapped to
/// 0 by the caller when the row saw fewer than five argmaxes at all).
fn insertTop5(top5: *[5]u32, counts: []const u32, tok: u32) void {
    const c = counts[tok];
    if (c == 0) return;
    var pos: usize = 5;
    while (pos > 0) {
        const cur = top5[pos - 1];
        const curc: u32 = if (cur == std.math.maxInt(u32)) 0 else counts[cur];
        if (curc >= c) break;
        pos -= 1;
    }
    if (pos == 5) return;
    var i: usize = 4;
    while (i > pos) : (i -= 1) top5[i] = top5[i - 1];
    top5[pos] = tok;
}

fn minOf(v: []const f64) f64 {
    var m: f64 = std.math.inf(f64);
    for (v) |x| m = @min(m, x);
    return m;
}

fn maxOf(v: []const f64) f64 {
    var m: f64 = -std.math.inf(f64);
    for (v) |x| m = @max(m, x);
    return m;
}

/// The first-16 token dump, printed exactly as the ids sit: comma-separated, no
/// container formatting (the ruling's copy-complete print is the ids themselves).
fn inlinePrint(ids: []const u32) void {
    for (ids, 0..) |v, i| {
        if (i != 0) std.debug.print(", ", .{});
        std.debug.print("{d}", .{v});
    }
}
/// The dump flag's and the gates' checksum: FNV-1a 64.
fn fnv1a(bytes: []const u8) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (bytes) |b| {
        h ^= b;
        h *%= 0x100000001b3;
    }
    return h;
}

/// The gate's Head at the dense head slab: same kernel, same event discipline,
/// same [1024][VOCAB] fp32 block — but the slab is the bf16 source's lm_head
/// widened to fp16 (uploadDense), not a dequantized trellis head.
fn initDenseHead(d: *Dev, c: *const Checkpoint) !kl.Head {
    const K: usize = D;
    const nt: usize = VOCAB;
    const block: u32 = 1024;
    const gl = try d.g.immediate(0, 0);
    const dl = d.g.immediate(1, 0) catch gl;
    const stage = @as([*]u8, @ptrCast(try d.g.hostAlloc(256 << 20)));
    const evs = try d.g.events(7);
    const h = kl.Head{
        .d = d, .k = @intCast(K), .nt = @intCast(nt), .vocab = nt,
        .info = .{ .k = @intCast(K), .nt = @intCast(nt), .parts = &.{} },
        .slab = try d.alloc_(@as(usize, K) * nt * 2),
        .gemm = try d.kernel(try d.g.module(&kl_kernels.eaq_kl), "eaq_gemm32", 256),
        .copy16 = try d.kernel(try d.g.module(&kl_kernels.eaq_kl), "eaq_copy16", 256),
        .a = try d.alloc_(@as(usize, block) * K * 2),
        .a_stage = try d.alloc_(@as(usize, block) * K * 2),
        .c = try d.alloc_(@as(usize, block) * nt * 4),
        .block = block,
        .gl = gl, .dl = dl, .stage = stage, .ev = evs,
    };
    try d.fill(h.slab, @as(u16, 0), @as(usize, K) * nt * 2);
    const t = try c.get("lm_head.weight");
    if (!std.mem.eql(u8, t.dtype, "BF16")) {
        std.debug.print("xpu-eaq-ref: lm_head.weight is {s}, expected BF16\n", .{t.dtype});
        return error.Usage;
    }
    try calib.uploadDense(d, t.bytes, VOCAB, D, VOCAB, 0, h.slab);
    try d.sync();
    return h;
}

/// The KLStatistic probe: the head slab as the card really holds it after uploadDense —
/// density and magnitude of the strip the GEMM reads first, beside its checksum.
fn slabProbe(h: *kl.Head) void {
    const probe_rows = 2048; // a [2][probe_rows] fp16 patch of the slab's first columns
    const bytes = @as(usize, probe_rows) * 2 * 2;
    const buf = alloc.alloc(u16, bytes / 2) catch return;
    defer alloc.free(buf);
    h.download(std.mem.sliceAsBytes(buf), h.slab, bytes) catch return;
    var nonzero: usize = 0;
    var mx: f64 = 0;
    for (buf) |v| {
        const f: f64 = @abs(@as(f16, @bitCast(v)));
        if (v != 0) nonzero += 1;
        mx = @max(mx, f);
    }
    std.debug.print("xpu-eaq-ref: head slab probe (first {d} values): nonzero {d} max {d:.5} fnv {x}\n", .{
        buf.len, nonzero, mx, fnv1a(std.mem.sliceAsBytes(buf)) });
}

fn slabOf(s: calib.Slabs, slot: usize) *anyopaque {
    return switch (slot) {
        0 => s.mixer_in,
        1 => s.mixer_out,
        2 => s.gu,
        3 => s.down,
        else => unreachable,
    };
}

extern "c" fn pread(fd: c_int, buf: [*]u8, count: usize, off: i64) isize;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;

/// The prompt read on the box's terms: pread against the lseek'd size, retrying the
/// transient EFAULT/short returns; an early zero against a size we just measured is the
/// ingestion bug the ruling describes, not a short read — it aborts the run, loudly.
fn readAll(path: []const u8) ![]u8 {
    var stack: [4096]u8 = undefined;
    if (path.len == 0 or path.len >= stack.len) return error.FileIo;
    @memcpy(stack[0..path.len], path);
    stack[path.len] = 0;
    const fd = open(@ptrCast(&stack), 0, 0);
    if (fd < 0) return error.FileNotFound;
    defer _ = close(fd);
    const size: usize = @intCast(@max(lseek(fd, 0, 2), 0));
    if (size == 0) return error.FileIo;
    const buf = try alloc.alloc(u8, size);
    errdefer alloc.free(buf);
    var done: usize = 0;
    var tries: u32 = 0;
    while (done < size) {
        const want = @min(@as(usize, 1) << 20, size - done);
        const got = pread(fd, buf.ptr + done, want, @intCast(done));
        if (got > 0) {
            done += @intCast(got);
            tries = 0;
        } else if (got == 0) {
            std.debug.print("xpu-eaq-ref: pread of {s} hit EOF at {d} of {d}\n", .{ path, done, size });
            return error.FileIo;
        } else {
            tries += 1;
            if (tries > 5) {
                std.debug.print("xpu-eaq-ref: pread of {s} keeps failing at {d} of {d} (errno {d})\n", .{ path, done, size, std.c._errno().* });
                return error.FileIo;
            }
            _ = nanosleep(&.{ .tv_sec = 0, .tv_nsec = 1_000_000_000 }, null);
        }
    }
    return buf;
}

fn openOut(path: []const u8) c_int {
    var stack: [4096]u8 = undefined;
    @memcpy(stack[0..path.len], path);
    stack[path.len] = 0;
    return open(@ptrCast(&stack), O_WRONLY_CREAT_TRUNC, 0o644);
}
