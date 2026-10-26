//! Block LDLQ on Intel Xe2 (Level Zero): the per-block feedback loop of fable LDLQ spec
//! section 4.3. One part at a time: W_q [K, N] fp32 uploaded, T32 := W32, then blocks
//! j = K/16 - 1 downto 0 - strip_f16 (the fp16 targets the encoder reads, counting
//! |T| > 4 clamps), the step-1 exl3_encode launch over the strip's nt tiles (K = 16,
//! tile0 = the strip start, the block's trellis region TR + j*nt*tileWords, the block's
//! STAT certificate region STAT + j*nt*16, and WHAT = Q16, the encoder's decoded
//! codebook values), strip_diff (E32 against the ORIGINAL W32), and for j > 0 the
//! rank-16 feedback ldlq_update T32[0:16j] -= (L32[16j:16j+16, 0:16j])^T * E32. Blocks
//! run last-to-first (section 4.1); L is the slot's BlockLdl.l uploaded once per slot,
//! D never touches the device. ldlq_eta reduces eta = What - target per block for the
//! PartReport; the driver's host fp64 proxy (section 4.6) comes from core.exl3_ldlq_ref
//! and refuses sizes beyond test scale - the device GEMM form is blocked on
//! exl3_syrk32. The oracle is core/exl3_format.zig; the loop's arbiter is
//! core/exl3_ldlq_ref.zig (T-LDLQ-3). Five launches per block: the spec's four plus the
//! PartReport's eta reduction.
//!
//! WHAT is the tenth argument of exl3_encode (LDLQ spec 4.3): until the exl3_enc.cl
//! addition lands, the launch errors at argument index 9 - fail-closed, no silent skip.

const std = @import("std");
const ze = @import("ze");
const core = @import("core");
const dev = @import("dev.zig");
const kb = @import("xpu_qwen27_kernels");
const exl3_encode = @import("exl3_encode.zig");

pub const Width = core.exl3_format.Width;
pub const TileStat = exl3_encode.TileStat;
const ldlq_ref = core.exl3_ldlq_ref;
const alloc = std.heap.c_allocator; // the runner's allocator (q.alloc, qwen27.zig:39; dev.zig:10 uses q.alloc)

pub const Part = struct {
    n: u32, // the part's N
    n0: u32, // the column base inside the slot (pack-facing; the loop is N-local)
    sv: []const i8, // per-column signs (section 4.2), written to the pack as part of svh
    s: []const f32, // column RMS
    svh: []f16, // fp16(sv * s * g): the pack tensor; the driver itself takes W_q rotated
};

pub const Slot = struct {
    k: u32,
    su: []const i8, // slot-wide row signs (parts share them, so one H', one LDL)
    suh: []f16,
    ldl: *const core.exl3_ldl.BlockLdl,
    gen: u64 = 0, // P0-2: identity is (pointer, gen) — stack-reused BlockLdls carry a fresh gen
    parts: []Part,
};

pub const PartReport = struct {
    tiles: u32,
    optimal: u32,
    closed: u32,
    failed: u32, // from TileStat.status: any nonzero value here aborts trust in the part
    gap_hist: [16]u32, // log2 buckets of cost - open_cost (ldlq_ref.gapBucket)
    clamped: u64, // |target| > 4 count over every block; above 1e-4 of entries, investigate g
    eta_sq: f64, // sum ||eta_k||^2 from the device's deterministic per-block partials
    proxy: ?f64, // tr(deltaW^T H' deltaW) when want_proxy (host fp64, test sizes; 4.6)
};

pub const Driver = struct {
    enc: exl3_encode.Encoder, // the step-1 device encoder, sized for max nt among the slot's parts
    l32: *anyopaque, // [K, K] fp32, one upload per slot
    w32: *anyopaque, // the part's W_q, uploaded fp32
    t32: *anyopaque, // the running feedback targets; doubles as the decodePart staging
    s16: *anyopaque, // the block's fp16 strip, the encoder's WQ
    q16: *anyopaque, // the encoder's WHAT: decoded codebook values of the chosen states
    e32: *anyopaque, // deltaW = fp32(Q16) - W32 against the original
    // loop machinery beyond the spec's field list:
    clamped: *anyopaque, // the strip_f16 counter, zeroed per part
    tr_dev: *anyopaque, // the part trellis [K/16, N/16, 8*halves] for the largest part
    part_scratch: *anyopaque, // ldlq_eta partials [K/16][ceil(n_max/256)] fp32
    eta_dev: ?*anyopaque, // eta strips [K, N] fp32, allocated on first want_eta use
    /// The final-code decode each block feeds its own feedback (fable's probe-down
    /// ruling 1): after encodedWhat, strip_decode regenerates the block's codes from
    /// its own final trellis, so the tail-bitting closure and the clamps reach the
    /// feedback E and the eta strip - the feedback error and eta_sq describe the codes
    /// the pack ships. The what strip is encodeWhat's WHAT as written, kept per block
    /// for the violating-tile dossier; both [k, n_max] fp16 like the eta strip.
    qf_dev: ?*anyopaque = null, // [k, n_max] fp16: decode of each block's final trellis
    what_dev: ?*anyopaque = null, // [k, n_max] fp16: encodeWhat's WHAT, want_what-gated
    k_strip: ze.H,
    k_diff: ze.H,
    k_update: ze.H,
    k_eta: ze.H,
    k_decode: ze.H,
    w: Width,
    kdim: u32, // K: multiples of 16; n_max the widest part any quantizePart will see
    n_max: u32,
    count: u32, // in-flight tiles: a strip of nt tiles is ceil(nt / count) launches
    want_eta: bool = false, // keep the device eta strips for readEta (T-LDLQ-3)
    want_what: bool = false, // keep the encodeWhat WHAT strips for the readback dossier
    want_proxy: bool = false, // --proxy at test sizes: the host fp64 4.6 loss
    dbg_dump: ?[]const u8 = null, // per-block evidence dump for the T-LDLQ-2 bisect:
    // S16/Q16 after the strip and encode, the block's STAT words, and T32 rows [0,16j)
    // after the feedback — the twin of exl3_ldlq_ref's dbg_path, diffed offline
    ldl_uploaded: ?*const core.exl3_ldl.BlockLdl = null,
    ldl_gen: u64 = 0, // P0-2: the gen of the cached upload; pointer equality alone lies when the caller reuses a stack frame
    list: ze.H = null, // the batched async immediate list (Dev.q_list's twin for this driver):
    // the loop's block appends its whole command list here and drains once at the block
    // boundary — on the sync-mode d.list every append paid a submission round trip

    pub fn init(d: *dev.Dev, w: Width, k: u32, n_max: u32, count: u32) !Driver {
        if (k == 0 or k % 16 != 0 or n_max == 0 or n_max % 16 != 0 or count == 0) return error.BadDims;
        if (d.q_list == null) d.q_list = try d.g.immediateAsync(0, 0); // Dev.settle drains this handle
        const kblks = k / 16;
        const nt_max = n_max / 16;
        const ng_max = (n_max + 255) / 256;
        const m = try d.g.module(&kb.ldlq);
        return .{
            .enc = try exl3_encode.Encoder.init(d, w, count, kblks * nt_max),
            .l32 = try d.alloc_(@as(usize, k) * k * 4),
            .w32 = try d.alloc_(@as(usize, k) * n_max * 4),
            .t32 = try d.alloc_(@as(usize, k) * n_max * 4),
            .s16 = try d.alloc_(16 * @as(usize, n_max) * 2),
            .q16 = try d.alloc_(16 * @as(usize, n_max) * 2),
            .e32 = try d.alloc_(16 * @as(usize, n_max) * 4),
            .clamped = try d.alloc_(4),
            .tr_dev = try d.alloc_(kblks * nt_max * w.tileWords() * 2),
            .part_scratch = try d.alloc_(kblks * ng_max * 4),
            .eta_dev = null,
            .k_strip = try d.kernel(m, "strip_f16", 256),
            .k_diff = try d.kernel(m, "strip_diff", 256),
            .k_update = try d.kernel(m, "ldlq_update", 256),
            .k_eta = try d.kernel(m, "ldlq_eta", 256),
            .k_decode = try d.kernel(m, "strip_decode", 256),
            .w = w,
            .kdim = k,
            .n_max = n_max,
            .count = count,
            .list = d.q_list.?,
        };
    }

    /// Uploads the slot's L once (BlockLdl.l, [K, K] fp32: unit-block-lower, diagonal
    /// 16x16 blocks exactly I, everything above the block diagonal exactly zero).
    pub fn beginSlot(dr: *Driver, d: *dev.Dev, slot: *const Slot) !void {
        if (dr.ldl_uploaded == slot.ldl and dr.ldl_gen == slot.gen) return; // same factorization already resident
        if (slot.k != dr.kdim or slot.ldl.k != dr.kdim) return error.BadDims;
        if (slot.ldl.l.len != dr.kdim * dr.kdim or slot.ldl.d.len != (dr.kdim / 16) * 256) return error.BadDims;
        try d.copyIn(dr.l32, std.mem.sliceAsBytes(slot.ldl.l));
        dr.ldl_uploaded = slot.ldl;
        dr.ldl_gen = slot.gen;
    }

    /// The section 4.3 loop for one part. wq64 is the part's W_q [K, n] fp64 from section
    /// 4.2 step 3 (signs flipped, columns scaled, rotated - the driver never re-derives
    /// it); tr receives the part's trellis i16 [K/16, n/16, 8*halves] host-side, results
    /// the per-tile certificates in block order (j descending, tile ascending).
    pub fn quantizePart(
        dr: *Driver,
        d: *dev.Dev,
        slot: *const Slot,
        part: *const Part,
        wq64: []const f64,
        tr: []i16,
        results: []TileStat,
    ) !PartReport {
        if (dr.ldl_uploaded != slot.ldl or dr.ldl_gen != slot.gen) return error.SlotNotStarted;
        const k = dr.kdim;
        const n = part.n;
        if (n == 0 or n % 16 != 0 or n > dr.n_max) return error.BadDims;
        const kblks = k / 16;
        const nt = n / 16;
        const ng = (n + 255) / 256;
        const words = dr.w.tileWords();
        if (wq64.len != k * n or tr.len != kblks * nt * words or results.len != kblks * nt) return error.BadDims;
        if (part.sv.len != n or part.s.len != n or part.svh.len != n) return error.BadDims;
        if (dr.qf_dev == null) dr.qf_dev = try d.alloc_(@as(usize, k) * dr.n_max * 2);
        if (dr.want_what and dr.what_dev == null) dr.what_dev = try d.alloc_(@as(usize, k) * dr.n_max * 2);
        if (dr.want_eta and dr.eta_dev == null) dr.eta_dev = try d.alloc_(@as(usize, k) * dr.n_max * 4);

        // W32 = the one f64 -> f32 rounding of W_q, uploaded; T32 := W32 on the device
        const w32h = try alloc.alloc(f32, k * n);
        defer alloc.free(w32h);
        for (wq64, w32h) |q, *p| {
            if (!std.math.isFinite(q)) return error.BadDims;
            p.* = @floatCast(q);
        }
        const bytes: usize = k * n * 4;
        try d.copyIn(dr.w32, std.mem.sliceAsBytes(w32h));
        try dev.copy(d, dr.t32, dr.w32, bytes);
        try d.copyIn(dr.clamped, &[_]u8{ 0, 0, 0, 0 });
        const t_part = monoNs(); // block cadence for the real-size runs (launch-bound)

        const eta_arg: ?*anyopaque = if (dr.want_eta) dr.eta_dev.? else null;
        var dbg_fd: c_int = -1;
        var dbg_buf: std.ArrayList(u8) = .empty;
        defer dbg_buf.deinit(alloc);
        if (dr.dbg_dump) |path| {
            const sent = try alloc.dupeSentinel(u8, path, 0);
            defer alloc.free(sent);
            dbg_fd = ccreat(sent.ptr, 0o644);
            if (dbg_fd < 0) return error.DumpOpenFailed;
        }
        defer {
            if (dbg_fd >= 0) _ = cclose(dbg_fd);
        }
        var j: u32 = kblks;
        while (j > 0) {
            j -= 1;
            if (j & 127 == 0) std.debug.print("ldlq: block {d} of {d} ({d} ms)\n", .{ j, kblks, (monoNs() -% t_part) / std.time.ns_per_ms });
            const ni: i32 = @intCast(n);
            try d.launchOn(dr.list, dr.k_strip, .{ .x = ng, .y = 16 }, .{
                dr.t32, dr.s16, @as(i32, @intCast(j)), ni, dr.clamped,
            });
            // The strip's tiles, K = 16 so tk = 0: encodeWhat with tile0 in [0, nt) (WQ,
            // WHAT and the STAT window are tile0-local), tr advanced to the block's own
            // trellis region. The block's STAT window is read back before the next block
            // overwrites it; encodeWhat pre-fills it with status 2 so a trapped or lost
            // launch reads back failed, never unspecified allocation.
            const tr_j = dev.off(dr.tr_dev, @as(usize, j) * nt * words * 2);
            var first: u32 = 0;
            while (first < nt) : (first += dr.count) {
                try dr.enc.encodeWhat(d, dr.s16, 16, n, tr_j, first, @min(dr.count, nt - first), dr.q16, dr.list);
            }
            try dr.enc.readStats(d, results[j * nt ..][0..nt], dr.list);
            // fable's probe-down ruling 1: redecode the block from its own final
            // trellis - after encodedWhat's tail-bitting closure and clamps - and
            // advance the feedback and the eta strip from the codes the pack ships.
            // The strip_decode launch is in-order on the same list, and the queued
            // copy freezes encodeWhat's WHAT for the violating-tile dossier.
            const qf_j = dev.off(dr.qf_dev.?, @as(usize, j) * 16 * n * 2);
            try d.launchOn(dr.list, dr.k_decode, .{ .x = nt, .y = 1 }, .{
                tr_j, @as(i32, @intCast(dr.w.halves)), qf_j, ni, @as(i32, @intCast(nt)),
            });
            if (dr.what_dev) |wv| try dev.copyOn(d, dr.list, dev.off(wv, @as(usize, j) * 16 * n * 2), dr.q16, 16 * @as(usize, n) * 2);
            try d.launchOn(dr.list, dr.k_diff, .{ .x = ng, .y = 16 }, .{
                qf_j, dr.w32, dr.e32, @as(i32, @intCast(j)), ni,
            });
            if (j > 0) try d.launchOn(dr.list, dr.k_update, .{ .x = j, .y = ng }, .{
                dr.t32, dr.l32, dr.e32, @as(i32, @intCast(j)), @as(i32, @intCast(16 * j)), ni, @as(i32, @intCast(k)),
            });
            try d.launchOn(dr.list, dr.k_eta, .{ .x = ng }, .{
                qf_j, dr.s16, eta_arg, dev.off(dr.part_scratch, @as(usize, j) * ng * 4), @as(i32, @intCast(j)), ni,
            });
            if (dbg_fd >= 0) try dumpBlock(d, dr, &dbg_buf, dbg_fd, j, results[j * nt ..][0..nt], n);
        }

        // readbacks: the trellis for the pack, the clamp counter and the eta partials
        // for the report - each a queued copy plus the list drain (stats were read per
        // block above)
        try dr.enc.readTrellis(d, dr.tr_dev, tr, dr.list);
        var clamp_bytes: [4]u8 = undefined;
        try readDevBytes(d, dr.list, dr.clamped, &clamp_bytes);
        const partial = try alloc.alloc(f32, kblks * ng);
        defer alloc.free(partial);
        try readDevBytes(d, dr.list, dr.part_scratch, std.mem.sliceAsBytes(partial));

        var rep = PartReport{
            .tiles = kblks * nt,
            .optimal = 0,
            .closed = 0,
            .failed = 0,
            .gap_hist = @splat(0),
            .clamped = std.mem.readInt(u32, &clamp_bytes, .little),
            .eta_sq = 0,
            .proxy = null,
        };
        for (partial) |p| rep.eta_sq += @as(f64, p); // host fp64 over fixed-order partials
        for (results) |st| {
            switch (st.status) {
                0 => rep.optimal += 1,
                1 => rep.closed += 1,
                2 => rep.failed += 1,
                else => return error.CorruptStat, // stat written wrong: fail closed
            }
            rep.gap_hist[ldlq_ref.gapBucket(st.cost -% st.open_cost)] += 1;
        }
        if (dr.want_proxy) {
            // section 4.6 at test sizes; beyond them proxyLoss returns ProxyTooLarge and
            // the run must use the device form (raw strip decoder + exl3_syrk32 GEMMs)
            rep.proxy = try ldlq_ref.proxyLoss(alloc, dr.w, slot.ldl.l, slot.ldl.d, tr, k, n, wq64);
        }
        return rep;
    }

    /// The kept eta strips (want_eta) for the T-LDLQ-3 comparison against the host
    /// reference: the [k, n] fp32 rows the ldlq_eta launches wrote this part.
    pub fn readEta(dr: *Driver, d: *dev.Dev, out: []f32, k: u32, n: u32) !void {
        const eta = dr.eta_dev orelse return error.EtaOff;
        if (out.len != k * n or k != dr.kdim or n > dr.n_max) return error.BadDims;
        try readDevBytes(d, dr.list, eta, std.mem.sliceAsBytes(out));
    }

    /// The final-code decode strips (qf_dev): what the block's feedback actually
    /// consumed, read back for the G1 host-vs-encoder reconstruction check.
    pub fn readQf(dr: *Driver, d: *dev.Dev, out: []f16, k: u32, n: u32) !void {
        if (dr.qf_dev == null) return error.QfOff;
        if (out.len != k * n or k != dr.kdim or n > dr.n_max) return error.BadDims;
        try readDevBytes(d, dr.list, dr.qf_dev.?, std.mem.sliceAsBytes(out));
    }

    /// The encodeWhat WHAT strips as the encoder recorded them (want_what), beside
    /// readQf's final decode: their difference names the violating tiles.
    pub fn readWhat(dr: *Driver, d: *dev.Dev, out: []f16, k: u32, n: u32) !void {
        const wv = dr.what_dev orelse return error.WhatOff;
        if (out.len != k * n or k != dr.kdim or n > dr.n_max) return error.BadDims;
        try readDevBytes(d, dr.list, wv, std.mem.sliceAsBytes(out));
    }

    /// The raw strip decoder over the last quantizePart's device trellis (the WHAT
    /// oracle of T-LDLQ-6 and the no-rotation dequant of 4.6; exl3_dequant128_c rotates
    /// and scales and is unusable there). Stages through t32, so it must run after the
    /// part's loop, never between blocks.
    pub fn decodePart(dr: *Driver, d: *dev.Dev, out: []f16, n: u32) !void {
        if (n == 0 or n % 16 != 0 or n > dr.n_max or out.len != dr.kdim * n) return error.BadDims;
        try d.launchOn(dr.list, dr.k_decode, .{ .x = n / 16, .y = dr.kdim / 16 }, .{
            dr.tr_dev, @as(i32, @intCast(dr.w.halves)), dr.t32, @as(i32, @intCast(n)), @as(i32, @intCast(n / 16)),
        });
        try readDevBytes(d, dr.list, dr.t32, std.mem.sliceAsBytes(out));
    }

    /// The strip-mode timing of measurement 7.4 (and the update share of 7.5): the full
    /// block loop for one part, `reps` times, no readbacks; returns the device-synced
    /// elapsed ns. Tiles per second = (k/16)*(n/16)*reps / seconds, against the
    /// free-running T8 number. skip_update passes ldlq_update a trapping rows value, so
    /// the difference between the two runs isolates the feedback GEMM's traffic cost.
    /// It times the pre-ruling loop shape (strip_f16/encodeWhat/strip_diff/ldlq_update/
    /// ldlq_eta): quantizePart's per-block strip_decode rework is excluded, so 7.4/7.5
    /// stay comparable and the production cadence rides the block print instead.
    pub fn benchPart(dr: *Driver, d: *dev.Dev, wq64: []const f64, n: u32, reps: u32, skip_update: bool) !u128 {
        if (n == 0 or n % 16 != 0 or n > dr.n_max or wq64.len != dr.kdim * n) return error.BadDims;
        const k = dr.kdim;
        const kblks = k / 16;
        const nt = n / 16;
        const ng = (n + 255) / 256;
        const words = dr.w.tileWords();
        const w32h = try alloc.alloc(f32, k * n);
        defer alloc.free(w32h);
        for (wq64, w32h) |q, *p| {
            if (!std.math.isFinite(q)) return error.BadDims;
            p.* = @floatCast(q);
        }
        const bytes: usize = k * n * 4;
        try d.copyIn(dr.w32, std.mem.sliceAsBytes(w32h));
        try dev.copy(d, dr.t32, dr.w32, bytes);

        const t0 = monoNs();
        var rep: u32 = 0;
        while (rep < reps) : (rep += 1) {
            var j: u32 = kblks;
            while (j > 0) {
                j -= 1;
                const ni: i32 = @intCast(n);
                try d.launchOn(dr.list, dr.k_strip, .{ .x = ng, .y = 16 }, .{
                    dr.t32, dr.s16, @as(i32, @intCast(j)), ni, dr.clamped,
                });
                const tr_j = dev.off(dr.tr_dev, @as(usize, j) * nt * words * 2);
                var first: u32 = 0;
                while (first < nt) : (first += dr.count) {
                    try dr.enc.encodeWhat(d, dr.s16, 16, n, tr_j, first, @min(dr.count, nt - first), dr.q16, dr.list);
                }
                try d.launchOn(dr.list, dr.k_diff, .{ .x = ng, .y = 16 }, .{
                    dr.q16, dr.w32, dr.e32, @as(i32, @intCast(j)), ni,
                });
                if (j > 0) try d.launchOn(dr.list, dr.k_update, .{ .x = j, .y = ng }, .{
                    dr.t32,                                              dr.l32, dr.e32,                @as(i32, @intCast(j)),
                    @as(i32, if (skip_update) -1 else @intCast(16 * j)), ni,     @as(i32, @intCast(k)),
                });
                try d.launchOn(dr.list, dr.k_eta, .{ .x = ng }, .{
                    dr.q16, dr.s16, @as(?*anyopaque, null), dr.part_scratch, @as(i32, @intCast(j)), ni,
                });
            }
        }
        try d.sync();
        return @as(u128, monoNs()) -% @as(u128, t0);
    }
};

const ccreat = @extern(*const fn (path: [*:0]const u8, mode: c_uint) callconv(.c) c_int, .{ .name = "creat" });
const cwrite = @extern(*const fn (fd: c_int, buf: [*]const u8, n: usize) callconv(.c) isize, .{ .name = "write" });
const cclose = @extern(*const fn (fd: c_int) callconv(.c) c_int, .{ .name = "close" });

/// T-LDLQ-2 evidence: the block's STAT words, the strip S16 and decoded Q16 as written,
/// and the T32 rows [0,16j) the feedback just produced — read after the block's launches
/// in queue order, so what lands is exactly what the next block will consume. Diffed
/// offline against exl3_ldlq_ref's REF lines, in processing order (j descending).
fn dumpBlock(
    d: *dev.Dev,
    dr: *const Driver,
    buf: *std.ArrayList(u8),
    fd: c_int,
    j: u32,
    stats: []const TileStat,
    n: u32,
) !void {
    for (stats, 0..) |st, tn| try buf.print(alloc, "DEV ST {d} {d} {x} {x} {d} {d}\n", .{ j, tn, st.cost, st.open_cost, st.pinned, st.status });
    const sbuf = try alloc.alloc(u8, 16 * @as(usize, n) * 2);
    defer alloc.free(sbuf);
    try readDevBytes(d, dr.list, dr.s16, sbuf);
    try buf.print(alloc, "DEV S16 {d}\n", .{j});
    for (0..sbuf.len / 2) |i| {
        const v = std.mem.readInt(u16, sbuf[2 * i ..][0..2], .little);
        try buf.print(alloc, "{x:0>4} ", .{v});
    }
    try buf.append(alloc, '\n');
    const qbuf = try alloc.alloc(u8, 16 * @as(usize, n) * 2);
    defer alloc.free(qbuf);
    try readDevBytes(d, dr.list, dr.q16, qbuf);
    try buf.print(alloc, "DEV Q16 {d}\n", .{j});
    for (0..qbuf.len / 2) |i| {
        const v = std.mem.readInt(u16, qbuf[2 * i ..][0..2], .little);
        try buf.print(alloc, "{x:0>4} ", .{v});
    }
    try buf.append(alloc, '\n');
    if (j > 0) {
        const tbuf = try alloc.alloc(u8, @as(usize, 16 * j) * n * 4);
        defer alloc.free(tbuf);
        try readDevBytes(d, dr.list, dr.t32, tbuf);
        try buf.print(alloc, "DEV T32 {d}\n", .{j});
        for (0..tbuf.len / 4) |i| {
            const v = std.mem.readInt(u32, tbuf[4 * i ..][0..4], .little);
            try buf.print(alloc, "{x:0>8} ", .{v});
        }
        try buf.append(alloc, '\n');
    }
    try bwFlush(buf, fd);
}

fn bwFlush(buf: *std.ArrayList(u8), fd: c_int) !void {
    if (buf.items.len == 0) return;
    _ = cwrite(fd, buf.items.ptr, buf.items.len);
    buf.clearRetainingCapacity();
}

fn readDevBytes(d: *dev.Dev, list: ze.H, src: *anyopaque, out: []u8) !void {
    const host = try d.g.hostAlloc(out.len);
    defer d.g.free(host);
    try dev.copyOn(d, list, host, src, out.len); // ordered after the list's own kernels; settle drains it
    try d.sync();
    @memcpy(out, @as([*]const u8, @ptrCast(host))[0..out.len]);
}

fn monoNs() u64 { // the house timer (exl3_ldl.zig:109's pattern; std.time.nanoTimestamp is gone in 0.17)
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

comptime {
    // Zig 0.17 analyzes only reachable declarations: anchor the public surface so the
    // bodies are type-checked whenever this module is, before the CLI probe calls them.
    _ = .{ Driver.init, Driver.beginSlot, Driver.quantizePart, Driver.readEta, Driver.decodePart, Driver.benchPart };
}
