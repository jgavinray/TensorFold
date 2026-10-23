//! The EXL3 weight GEMMs of the Qwen3.8-27B XPU forward (PR 4 wiring contract section 5):
//! one dispatch target for every trellis projection, reached from forward.gemmRows's guard
//! when Proj.exl3 is set. The prompt (wide) path dequantizes the fused slot's parts into
//! their column ranges of the shared fp16 scratch with the rotation and the entry scales
//! baked in (the A1 design — exl3_dequant128_c per part, then ONE exl3_prompt16 over the
//! padded NT: no rot_in/rot_out kernels and no per-part GEMM launches, and A is the raw
//! activations), and the decode path (m <= 16, and the head's rows — always <= 16 and
//! never wide, forward.zig's first/rows) runs the fp16-output trellis GEMVs per part.
//! PROMPT/DECODE ASYMMETRY (the PR 3 open operator decision, still standing for PR 5):
//! the prompt dequant uses the exact integer codebook form and the decode GEMVs the fast
//! fp32 form, so prompt-vs-decode logits differ by up to half an fp16 ulp per weight
//! until the variants match (pr3-prompt-kernel-design.md, Measured/Contract notes).
const std = @import("std");
const dev = @import("dev.zig");
const fwd = @import("forward.zig");
const load = @import("load.zig");
const exl3_model = @import("exl3_model.zig");
const off = dev.off;

// ---- XPU_EXL3_TRACE: the fault-localizing trace. dev.launch prints every queued command
// and drains the command list after it (dev.zig), so the last line before a device-lost is
// the command that caused it; the EXL3 launches prepend this line with the args that
// decide their footprint. Unset, the cost is a cached getenv per launch and per copy ----
var trace_on: ?bool = null;
var trace_call: u32 = 0; // gemmRows calls so far: /4 is the layer of a slot call

fn tracing() bool {
    if (trace_on == null) trace_on = std.c.getenv("XPU_EXL3_TRACE") != null;
    return trace_on.?;
}

/// What a fused handle is, from its (K, N) — Proj carries no name (the shapes of
/// qwen27.zig: D 5120, FF 17408, GDN_IN 16640, ATT_IN 14336, VOCAB 248320).
fn slotName(k: u32, n: u32, head: bool) []const u8 {
    if (head) return "head";
    if (k == 5120 and n == 34816) return "gu";
    if (k == 17408 and n == 5120) return "down";
    if (k == 5120 and n == 14336) return "qkv";
    if (k == 5120 and n == 16640) return "mixer_in";
    if (k == 6144 and n == 5120) return "mixer_out";
    return "other";
}

/// One launch's identity, printed before the launch so a faulting command's line is the
/// last one out: the call counter, the slot, the part role, the kernel, the grid and the
/// args that decide its memory footprint (K, the NT pitch, the part's N and column base
/// n0, the fused parts' column total sumn, the row count M, the width), plus the four
/// device pointers it touches. dev.launch's own line and drain follow it.
fn trLaunch(call: u32, slot: []const u8, role: []const u8, part: u32, nparts: u32,
    name: []const u8, gx: u32, gy: u32, K: i32, NT: i32, N: i32, n0: i32, sumn: i32, M: i32, bits: i32,
    p0: *anyopaque, p1: ?*anyopaque, p2: ?*anyopaque, p3: ?*anyopaque) void {
    std.debug.print(
        \\EXL3TR call={d} layer~{d} {s} {s} {d}/{d} {s} gx={d} gy={d} K={d} NT={d} sumn={d} N={d} n0={d} M={d} bits={d} P0={x} P1={x} P2={x} P3={x}
        \\
    , .{ call, @divTrunc(call - 1, 4), slot, role, part, nparts, name, gx, gy, K, NT, sumn, N, n0, M, bits,
        @intFromPtr(p0), ptrBits(p1), ptrBits(p2), ptrBits(p3) });
}

fn ptrBits(p: ?*anyopaque) usize {
    return if (p) |q| @intFromPtr(q) else 0;
}

/// One EXL3 weight GEMM: out fp16 [m, w.n] = h fp16 [m, w.k] · W_effᵀ.
/// wide: per-part exl3_dequant128_c into the wscr column ranges, then ONE exl3_prompt16
///       over the fused NT (the A1 design: rotation baked, no rot_in/rot_out, no
///       per-part GEMM — pr3-prompt-kernel-design.md "Decision: A1"; the brief's
///       "per-part launches are NOT needed" is about the GEMM, not the dequant).
/// !wide: per-part exl3_decode2_h / exl3_decode2_t1_h (fp16 Y) across the part's
///        column range; m <= 16; head with rows > 16 (or the unreachable wide head):
///        error.Exl3HeadRows (contract section 3 — the head keeps its own trellis-decode
///        path, no 2.54 GB slab, and never touches the scratch).
/// The dequant pass runs once per (chunk, projection) — the fused slot's parts share it.
/// The tail mirrors forward.gemmRows's hook (PR 5 fidelity drops in; zero-cost when null).
pub fn gemmRows(d: *dev.Dev, hx: *const exl3_model.Exl3, wscr: ?*anyopaque, h: *anyopaque, ahi: *anyopaque, alo: *anyopaque,
    w: load.Proj, out: *anyopaque, m: u32, wide: bool) !void {
    _ = ahi; // the EXL3 path reads neither plane: no per-group scales exist, the scales and the rotation are baked into the dequant
    _ = alo;
    const K: i32 = @intCast(w.k);
    const NT: i32 = @intCast(hx.nt); // the padded Proj.n: the wscr pitch and the out pitch the consumers already match (a.p, a.r, a.logits)
    const tr = tracing();
    if (tr) trace_call += 1;
    const call = trace_call;
    const slot = slotName(w.k, w.n, hx.head);
    const nparts: u32 = @intCast(hx.parts.len);
    var sumn: i32 = 0; // the fused parts' real columns, against the NT the GEMM walks
    if (tr) {
        for (hx.parts) |p| sumn += @intCast(p.n);
    }
    if (hx.head) {
        // reachable head rows are 1..16 and never wide (forward.zig:433-434 and gemm's m > 16);
        // if it ever arrives, refuse loudly rather than answer wrong (contract section 3).
        if (wide or m > 16) return error.Exl3HeadRows;
        try gemvs(d, hx, h, out, m, K, NT, call, slot, tr, sumn);
    } else if (wide) {
        // the dequant slab exists only on the wide path; the GEMVs stage their own Xt
        const s = wscr orelse return error.Exl3NoScratch;
        for (hx.parts, 0..) |p, pi| { // one dequant pass per (chunk, projection): each part decodes exactly into columns [n0, n0 + p.n) of the slab
            const gx: u32 = @intCast(w.k / 128);
            const gy: u32 = (p.n + 127) / 128;
            if (tr) trLaunch(call, slot, "dequant", @intCast(pi), nparts, "exl3_dequant128_c", gx, gy,
                K, NT, @intCast(p.n), @intCast(p.n0), sumn, @intCast(m), @intCast(p.bits), p.tr, p.suh, p.svh, s);
            try d.launch(d.k.exl3_dequant128_c, .{ .x = gx, .y = gy }, .{
                p.tr,        p.suh,   p.svh,   s,           K,
                @as(i32, @intCast(p.n)), NT, @as(i32, @intCast(p.bits)), @as(i32, @intCast(p.n0)),
            });
        }
        // one fused GEMM over the whole padded NT: A is the raw fp16 activations [m, K],
        // B the [K][NT] slab, C written NT-pitched — the exl3PromptProbe's launches, and
        // no suh/svh here (baked into the dequant, the A1 design).
        const gx: u32 = @intCast((m + 127) / 128);
        const gy: u32 = hx.nt / 128;
        if (tr) trLaunch(call, slot, "gemm", 0, nparts, "exl3_prompt16", gx, gy, K, NT,
            @intCast(hx.nt), 0, sumn, @intCast(m), 0, h, s, out, null);
        try d.launch(d.k.exl3_prompt16, .{ .x = gx, .y = gy },
            .{ h, s, out, @as(i32, @intCast(m)), NT, K });
    } else {
        if (m > 16) return error.Exl3Rows; // the prompt paths pass wide; the GEMVs hold eight rows a launch and sixteen a projection
        try gemvs(d, hx, h, out, m, K, NT, call, slot, tr, sumn);
    }
    if (fwd.gemm_hook) |f| try f(d, h, w, out, m);
}

/// The decode GEMVs' Xt staging is a process-global; a new runner must not reuse the old one's pointer.
pub fn exl3ResetXscr() void {
    xscr = null;
    xscr_elts = 0;
}

/// The fp16-output trellis GEMVs across the fused slot's column ranges, with the format's
/// scales and both Hadamards around them: the raw-word GEMVs compute Z[b] = Σ_a Xt[a] ·
/// W_q[a][b], so each part runs exl3_xform_h first (Xt = H_K (X ⊙ suh), per part because
/// suh is per part) and exl3_yscale_h after (Y = svh ⊙ (H_N Z)) — the two rotations and
/// two scale vectors of the host oracle's diag(suh) H_K W_q H_N diag(svh)
/// (core/exl3_format.zig:200-212), which dequant128_c bakes into the slab for the prompt
/// path. The Xt staging is this file's own grow-only device buffer (the GEMVs' input,
/// [16][K] fp16; parts are serial, so one buffer serves the whole slot and the head).
/// The part's rows land at out's columns [n0, n0 + p.n) at the fused row pitch NT — Y
/// travels pre-shifted to the column base and the kernel's NT argument carries the pitch
/// (the frozen exl3_decode2 couples the two through one N argument, which only fits a
/// single-part projection). kslices = 1: the kernels' slice layout (exl3.cl:44-51,115-119,
/// 171-175) writes slice sums to disjoint rows [ks·T, ks·T+T) and only the probe reduces
/// them host-side (Y sized kslices·T·N, qwen27_main.zig:500), so any kslices > 1 would show
/// the consumers partial sums; the single slice covers all of K/16 (its kt_end rule).
/// The head is unaffected: the probe's divisor rule already yields kslices = 1 at
/// N/16 = 15520 ≥ 2048 (contract section 3), the shape this loop launches for it.
fn gemvs(d: *dev.Dev, hx: *const exl3_model.Exl3, h: *anyopaque, out: *anyopaque, m: u32, K: i32, NT: i32,
    call: u32, slot: []const u8, tr: bool, sumn: i32) !void {
    const nparts: u32 = @intCast(hx.parts.len);
    const xt = try xscratch(d, m, K);
    const gxk: u32 = @intCast(@divExact(K, 128)); // the xform's k blocks; K % 128 == 0 (the loader admits it)
    const mi: i32 = @intCast(m);
    for (hx.parts, 0..) |p, pi| {
        const N: i32 = @intCast(p.n);
        const bits: i32 = @intCast(p.bits); // whole widths only; the loader refuses odd halves (contract section 6)
        const y = off(out, @as(usize, p.n0) * 2);
        const gx: u32 = @intCast(p.n / 16);
        // (a) the part's entry: suh then H_K, into the staging row block the GEMV reads
        if (tr) trLaunch(call, slot, "xform", @intCast(pi), nparts, "exl3_xform_h", gxk, m, K, NT,
            N, @intCast(p.n0), sumn, mi, bits, h, p.suh, xt, null);
        try d.launch(d.k.exl3_xform_h, .{ .x = gxk, .y = m }, .{ h, p.suh, xt, K, mi });
        // (b) the raw trellis words against the transformed row; kslices = 1 (the layout
        // note above), the whole K range in one slice.
        if (m == 1) {
            if (tr) trLaunch(call, slot, "gemv1", @intCast(pi), nparts, "exl3_decode2_t1_h", gx, 1,
                K, NT, N, @intCast(p.n0), sumn, 1, bits, p.tr, xt, y, null);
            try d.launch(d.k.exl3_decode2_t1_h, .{ .x = gx, .y = 1 },
                .{ p.tr, xt, y, K, N, @as(i32, 1), bits, @as(i32, 1), NT });
        } else {
            // rows 2..16: the kernels hold eight rows a launch — the same kernel at row
            // offsets 0 and 8 when m > 8, X advanced by 8 rows of K halves and Y by 8 rows
            // of the pitch NT (the kernel's Y indexing: (ks·T + t)·NT + …, exl3_h.cl:74).
            if (tr) trLaunch(call, slot, "gemv8a", @intCast(pi), nparts, "exl3_decode2_h", gx, 1,
                K, NT, N, @intCast(p.n0), sumn, @intCast(@min(m, 8)), bits, p.tr, xt, y, null);
            try d.launch(d.k.exl3_decode2_h, .{ .x = gx, .y = 1 },
                .{ p.tr, xt, y, K, N, @as(i32, @intCast(@min(m, 8))), bits, @as(i32, 1), NT });
            if (m > 8) {
                const x2 = off(xt, 8 * @as(usize, @intCast(K)) * 2);
                const y2 = off(y, 8 * @as(usize, @intCast(NT)) * 2);
                if (tr) trLaunch(call, slot, "gemv8b", @intCast(pi), nparts, "exl3_decode2_h", gx, 1,
                    K, NT, N, @intCast(p.n0), sumn, @intCast(m - 8), bits, p.tr, x2, y2, null);
                try d.launch(d.k.exl3_decode2_h, .{ .x = gx, .y = 1 }, .{
                    p.tr, x2,   y2,   K, N, @as(i32, @intCast(m - 8)),
                    bits, @as(i32, 1), NT,
                });
            }
        }
        // (c) the part's exit: H_N along its own columns (the admitted parts have N a
        // multiple of 128, so no block straddles), then svh
        const gxn: u32 = @intCast(p.n / 128);
        if (tr) trLaunch(call, slot, "yscale", @intCast(pi), nparts, "exl3_yscale_h", gxn, m, K, NT,
            N, @intCast(p.n0), sumn, mi, bits, y, p.svh, null, null);
        try d.launch(d.k.exl3_yscale_h, .{ .x = gxn, .y = m }, .{ y, p.svh, N, NT, mi });
    }
}

/// The decode GEMVs' Xt staging, fp16 [rows][K]: grow-only, one buffer for the process (a
/// slot's parts and both row halves share it, and one Runner owns the device at a time).
var xscr: ?*anyopaque = null;
var xscr_elts: usize = 0;

fn xscratch(d: *dev.Dev, rows: u32, K: i32) !*anyopaque {
    const n = @as(usize, rows) * @as(usize, @intCast(K));
    if (n > xscr_elts) {
        if (xscr) |old| d.g.free(old);
        xscr = try d.alloc_(n * 2);
        xscr_elts = n;
    }
    return xscr.?;
}
