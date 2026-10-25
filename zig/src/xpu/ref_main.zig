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
//! Output: raw little-endian fp32 [positions][VOCAB] in prompt order — exactly what
//! `xpu-eaq-kl --ref` consumes — plus its sha256.
//!
//! The dense model, the kernel table and the slot plans are quant.zig's own
//! (buildDenseModel / initKernelTable / layerSlots), pub for exactly this reuse:
//! one body, no drift with the capture pass.

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
extern "c" fn pwrite(fd: c_int, buf: [*]const u8, count: usize, off: i64) isize;
const O_WRONLY_CREAT_TRUNC: c_int = 1 | 0o100 | 0o1000;

const Usage = error{Usage};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var src: []const u8 = "";
    var prompts_path: []const u8 = "";
    var out_path: []const u8 = "";
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
            \\  raw LE fp32 [positions][{d}] in prompt order + sha256
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
    const ids = std.mem.bytesAsSlice(u32, ids_bytes);
    const rows: u32 = @intCast(ids.len / cols);
    const positions: u64 = @as(u64, rows) * cols;
    std.debug.print("xpu-eaq-ref: {d} rows x {d} = {d} positions -> {d:.2} GB of fp32 logits\n", .{
        rows, cols, positions, @as(f64, @floatFromInt(positions * VOCAB * 4)) / 1e9 });

    // -- the card and the dense model (quant.zig's own, pub for this reuse)
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
    const ids_dev = try d.upload(std.mem.sliceAsBytes(ids));
    const out_host = try alloc.alloc(f32, @as(usize, 1024) * VOCAB); // one GEMM block's logits

    // -- the head: the gate's fp32 GEMM at the dense head slab (fp16 [5120, VOCAB])
    var head = try initDenseHead(d, &c);
    errdefer head.deinit();

    const out_fd = openOut(out_path);
    defer _ = close(out_fd);
    var write_off: i64 = 0;

    // -- pass 1: the embedding of every row (stateless per token)
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
        try d.sync();
    }
    std.debug.print("xpu-eaq-ref: embedded {d} rows in {d:.1} s\n", .{ rows, q.now() - t_start });

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
        if (i % 8 == 0 or i == LAYERS - 1)
            std.debug.print("  layer {d}/64 done ({d:.1} s)\n", .{ i, q.now() - tl });
    }

    // -- pass 3: the final norm and the fp32 head per chunk; every position contributes
    const tf = q.now();
    for (0..rows) |r| {
        const bo: usize = @as(usize, r) * cols * D * 4;
        var at: u32 = 0;
        while (at < cols) {
            const end: u32 = @intCast(q.chunkEnd(at, cols));
            const T = end - at;
            const hbuf = try alloc.alloc(u16, @as(usize, T) * D);
            defer alloc.free(hbuf);
            const hdev = try d.alloc_(@as(usize, T) * D * 2);
            defer d.g.free(hdev);
            try calib.finalNorm(d, m, a, dev.off(stream, bo + @as(usize, at) * D * 4), hdev, T, null);
            try head.drainRunner(); // the finalNorm launches ride the runner list: make them truly done
            try head.download(std.mem.sliceAsBytes(hbuf), hdev, @as(usize, T) * D * 2);
            var done: u32 = 0;
            while (done < T) {
                const m_blk = @min(@as(u32, 1024), T - done);
                try head.logitsBlock(std.mem.sliceAsBytes(hbuf[@as(usize, done) * D ..][0 .. @as(usize, m_blk) * D]),
                    m_blk, out_host[0 .. @as(usize, m_blk) * VOCAB], 0);
                const b = std.mem.sliceAsBytes(out_host[0 .. @as(usize, m_blk) * VOCAB]);
                var offb: usize = 0;
                while (offb < b.len) {
                    const w = pwrite(out_fd, b.ptr + offb, b.len - offb, write_off + @as(i64, @intCast(offb)));
                    if (w <= 0) return error.FileIo;
                    offb += @intCast(w);
                }
                write_off += @as(i64, @intCast(b.len));
                done += m_blk;
            }
            at = end;
        }
        if (r % 2 == 1) std.debug.print("  ref rows {d}/{d} ({d:.1} s)\n", .{ r + 1, rows, q.now() - tf });
    }
    if (@as(u64, @bitCast(write_off)) != positions * VOCAB * 4) {
        std.debug.print("xpu-eaq-ref: wrote {d} bytes, expected {d}\n", .{ write_off, positions * VOCAB * 4 });
        return error.FileIo;
    }
    std.debug.print("xpu-eaq-ref: {s} {d} bytes in {d:.1} s (head GEMM {d:.1} s)\n", .{
        out_path, write_off, q.now() - t_start, head.gemm_s });
    const hex = try kl.sha256Hex(out_path);
    std.debug.print("  sha256 {s}\n", .{hex});
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

fn slabOf(s: calib.Slabs, slot: usize) *anyopaque {
    return switch (slot) {
        0 => s.mixer_in,
        1 => s.mixer_out,
        2 => s.gu,
        3 => s.down,
        else => unreachable,
    };
}
extern "c" fn read(fd: c_int, buf: [*]u8, count: usize) isize;

fn readAll(path: []const u8) ![]u8 {
    var stack: [4096]u8 = undefined;
    if (path.len == 0 or path.len >= stack.len) return error.FileIo;
    @memcpy(stack[0..path.len], path);
    stack[path.len] = 0;
    const fd = open(@ptrCast(&stack), 0, 0);
    if (fd < 0) return error.FileNotFound;
    defer _ = close(fd);
    var list: std.ArrayList(u8) = .empty;
    var buf: [65536]u8 = undefined;
    while (true) {
        const r = read(fd, &buf, buf.len);
        if (r <= 0) break;
        try list.appendSlice(alloc, buf[0..@intCast(r)]);
    }
    return list.items;
}

fn openOut(path: []const u8) c_int {
    var stack: [4096]u8 = undefined;
    @memcpy(stack[0..path.len], path);
    stack[path.len] = 0;
    const fd = open(@ptrCast(&stack), O_WRONLY_CREAT_TRUNC, 0o644);
    if (fd < 0) {
        std.debug.print("xpu-eaq-ref: cannot write {s}\n", .{path});
        std.process.exit(1);
    }
    return fd;
}
