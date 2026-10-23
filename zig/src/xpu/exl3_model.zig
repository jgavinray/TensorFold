//! The EXL3 device layer: the handle a fused projection carries (`load.Proj.exl3`), the
//! stock checkpoint's group admission — the `admit` checks of core/exl3_load.zig mirrored
//! onto dev.Checkpoint tensor metadata, the reader-side twin being `exl3_eaq.Quant` — and
//! the device-buffer builder that mirrors `load.projectionGptq`: per part the trellis
//! words plus fp16 `suh`/`svh` [K]/[N], the packed su/sv sign words unpacked host-side
//! (core/exl3_format.unpackSigns) and the per-16 fp16 style x16-replicated before copyIn.
//! forward_exl3.zig launches against these buffers; nothing here runs per token.
const std = @import("std");
const core = @import("core"); // exl3_format (Codebook, unpackSigns, Mul1Mul) and exl3_load.Summary
const dev = @import("dev.zig");
const load = @import("load.zig");
const Checkpoint = dev.Checkpoint;
const Dev = dev.Dev;
const Proj = load.Proj;
const Tensor = dev.Tensor;
const alloc = std.heap.c_allocator; // the loader's allocator: these buffers live the runner's length

/// One trellis group of a fused projection, uploaded to the device.
pub const Part = struct {
    tr: *anyopaque, // i16 [K/16, N/16, halves*8] trellis words (exl3_load.zig:84 admission shape)
    suh: *anyopaque, // fp16 [K], host-normalized (contract section 2)
    svh: *anyopaque, // fp16 [N], host-normalized
    bias: ?*anyopaque = null, // PR 4 never fills it: any bias part refuses the pack
    n0: u32, // first column of this part inside the fused Proj.n
    n: u32, // the part's real N (before padding)
    halves: u8, // 2 * bits, whole (4..16); the reader-side twin is exl3_eaq.zig:36-44
    bits: u8, // halves / 2, in 1..8 (the kernel's int `bits` argument; the kernels are ring-safe at nw = 8)
    codebook: core.exl3_format.Codebook, // admitted only as .mul1
    dense: bool = false, // eaq.dense/0 part served via the int4/fp16 route instead (PR 4: only lm_head)
};

/// Everything forward_exl3.zig needs to run one fused projection.
pub const Exl3 = struct {
    parts: []const Part, // fused parts stacked on N, in slot order (the exl3_eaq.Mapped order)
    nt: u32, // the Proj.n, i.e. N padded to 256 (contract section 2)
    head: bool = false, // true only for Model.head: forward.gemm reaches it at forward.zig:435
};

/// How a group stored its scales: fp16 per weight (copied as is), fp16 per 16 (x16
/// replicated to [K]/[N] before copyIn), or the packed I16 sign words (unpackSigns to fp16
/// +/-1 before copyIn — the dequant kernel reads fp16 scale buffers only).
pub const Style = enum { f16, f16_16, words };

/// The admitted trellis quantities of one group, derived from the trellis tensor itself
/// exactly as core/exl3_load.admit does: halves = shape[2]/8, k = shape[0]*16, n = shape[1]*16.
pub const Group = struct {
    prefix: []const u8,
    halves: u8,
    bits: u8,
    k: u32,
    n: u32,
    codebook: core.exl3_format.Codebook,
    style: Style,
};

/// The admission table's shape (core/exl3_load.zig:27-49).
pub const Summary = core.exl3_load.Summary;

/// A refusal names the entry and why (the fail(why,...) pattern of exl3_eaq.zig:92-99, printed).
pub fn refuse(comptime fmt: []const u8, args: anytype) void {
    std.debug.print("EXL3: " ++ fmt ++ "\n", args);
}

fn partName(prefix: []const u8, part: []const u8, suffix: []const u8) ![]const u8 {
    return std.fmt.allocPrint(alloc, "{s}{s}{s}", .{ prefix, part, suffix });
}

/// Admits one trellis group by prefix: the `admit` checks one-to-one
/// (core/exl3_load.zig:134-158; EAQ twin exl3_eaq.zig:160-276) plus the PR 4 refusals —
/// whole widths (the kernels take an int bits 2..8; an odd halves would decode to a
/// truncated, wrong answer, exl3.cl:64), the mul1 codebook (its tables are baked into every
/// kernel, exl3.cl:14,33,84,147), and no bias part (no kernel argument consumes one).
pub fn admitGroup(c: *const Checkpoint, prefix: []const u8) !Group {
    var g: Group = .{ .prefix = prefix, .halves = 0, .bits = 0, .k = 0, .n = 0, .codebook = .inst3, .style = .f16 };
    const tr = try c.get(try partName(prefix, "", ".trellis"));
    if (!std.mem.eql(u8, tr.dtype, "I16") or tr.rank != 3) {
        refuse("group {s}: trellis is {s} rank {d}, not I16 rank 3", .{ prefix, tr.dtype, tr.rank });
        return error.BadTrellisPart;
    }
    const words16 = tr.shape[2]; // the trellis's last dim is 16 * bits
    if (words16 % 8 != 0 or words16 / 8 < 2 or words16 / 8 > 16) { // core/exl3_load.zig:136-137
        refuse("group {s}: trellis last dim {d} is not 8 times a halves count in 2..16", .{ prefix, words16 });
        return error.BadTrellisPart;
    }
    g.halves = @intCast(words16 / 8);
    g.k = @intCast(tr.shape[0] * 16);
    g.n = @intCast(tr.shape[1] * 16);
    if (g.k % 128 != 0 or g.n % 128 != 0) { // core/exl3_load.zig:140
        refuse("group {s}: K {d} and N {d} must be multiples of 128", .{ prefix, g.k, g.n });
        return error.BadTrellisPart;
    }
    // The codebook's witness is the group's own marker tensor (core/exl3_load.zig:8-13,141-152).
    // Every kernel hardcodes the mul1 tables, so mcg, inst3 (no marker), or a corrupt marker refuses.
    const mul1 = c.has(try partName(prefix, "", ".mul1"));
    const mcg = c.has(try partName(prefix, "", ".mcg"));
    if (mul1 and mcg) {
        refuse("group {s}: carries both the mul1 and the mcg marker", .{prefix});
        return error.Exl3Codebook;
    }
    if (!mul1) {
        refuse("group {s}: codebook {s}; the XPU kernels serve mul1 only", .{ prefix, if (mcg) "mcg" else "inst3 (no marker)" });
        return error.Exl3Codebook;
    }
    const marker = try c.get(try partName(prefix, "", ".mul1"));
    if (marker.bytes.len != 4 or std.mem.readInt(u32, marker.bytes[0..4], .little) != core.exl3_format.Mul1Mul) {
        refuse("group {s}: the mul1 marker is not the magic word", .{prefix});
        return error.BadTrellisPart;
    }
    if (g.halves % 2 != 0) { // the mul1 half steps have no device path (exl3.cl:64)
        refuse("group {s}: halves {d} is odd (a mul1 half step); no device path exists", .{ prefix, g.halves });
        return error.Exl3HalfWidth;
    }
    g.bits = g.halves / 2;
    if (g.bits < 1 or g.bits > 8) { // the kernel traps (exl3_h.cl:24,166; exl3.cl:215) accept 1..8; bits=1 is oracle-verified on-card
        refuse("group {s}: bits {d} outside the kernels' 1..8", .{ prefix, g.bits });
        return error.Exl3BitsRange;
    }
    // Exactly one scale style, both sides present and consistently shaped (exl3_eaq.zig:240-267).
    const h_suh = c.has(try partName(prefix, "", ".suh"));
    const h_svh = c.has(try partName(prefix, "", ".svh"));
    const h_su = c.has(try partName(prefix, "", ".su"));
    const h_sv = c.has(try partName(prefix, "", ".sv"));
    if ((h_suh != h_svh) or (h_su != h_sv) or (h_suh and h_su)) {
        refuse("group {s}: needs both suh and svh (F16) or both su and sv (I16)", .{prefix});
        return error.BadScalePart;
    }
    if (!h_suh and !h_su) {
        refuse("group {s}: no scales at all (neither suh/svh nor su/sv)", .{prefix});
        return error.BadScalePart;
    }
    if (h_suh) {
        const suh = try c.get(try partName(prefix, "", ".suh"));
        const svh = try c.get(try partName(prefix, "", ".svh"));
        if (!std.mem.eql(u8, suh.dtype, "F16") or !std.mem.eql(u8, svh.dtype, "F16") or suh.rank != 1 or svh.rank != 1) {
            refuse("group {s}: the fp16 scales are not rank-1 F16", .{prefix});
            return error.BadScalePart;
        }
        if (suh.shape[0] == g.k and svh.shape[0] == g.n) {
            g.style = .f16; // per-weight: copied straight through (core/exl3_load.zig:89-92)
        } else if (suh.shape[0] == g.k / 16 and svh.shape[0] == g.n / 16) {
            g.style = .f16_16; // per-16 (the style exl3_eaq.zig:241-247 accepts): replicated on the host
        } else {
            refuse("group {s}: fp16 scales are neither [{d}] and [{d}] nor per-16", .{ prefix, g.k, g.n });
            return error.BadScalePart;
        }
    } else {
        const su = try c.get(try partName(prefix, "", ".su"));
        const sv = try c.get(try partName(prefix, "", ".sv"));
        if (!std.mem.eql(u8, su.dtype, "I16") or !std.mem.eql(u8, sv.dtype, "I16") or su.rank != 1 or sv.rank != 1 or
            su.shape[0] != g.k / 16 or sv.shape[0] != g.n / 16)
        {
            refuse("group {s}: the packed su/sv are not I16 [{d}] and [{d}]", .{ prefix, g.k / 16, g.n / 16 });
            return error.BadScalePart;
        }
        g.style = .words;
    }
    // PR 4 carries no output bias: no kernel argument consumes one (exl3.cl:62,213-214,302-303).
    if (c.has(try partName(prefix, "", ".bias"))) {
        refuse("group {s}: carries a bias part; no EXL3 kernel consumes an output bias", .{prefix});
        return error.Exl3BiasUnsupported;
    }
    return g;
}

/// The packed I16 sign words of one side -> fp16 +/-1 on the host, then one copyIn.
fn uploadSigns(d: *Dev, t: Tensor, words: u32, dst: *anyopaque) !void {
    const plane = try alloc.alloc(i16, words);
    defer alloc.free(plane);
    for (plane, 0..) |*p, w| p.* = @bitCast(std.mem.readInt(u16, t.bytes[@as(usize, w) * 2 ..][0..2], .little));
    const out = try alloc.alloc(f16, @as(usize, words) * 16);
    defer alloc.free(out);
    core.exl3_format.unpackSigns(plane, out); // word bit b set: element 16w+b is -1 (exl3_format.zig:151)
    try d.copyIn(dst, std.mem.sliceAsBytes(out));
}

/// The per-16 fp16 scale style -> each value x16-replicated to [K]/[N] on the host.
fn uploadReplicated(d: *Dev, t: Tensor, words: u32, dst: *anyopaque) !void {
    const out = try alloc.alloc(u16, @as(usize, words) * 16);
    defer alloc.free(out);
    for (0..words) |w| {
        const bits = std.mem.readInt(u16, t.bytes[@as(usize, w) * 2 ..][0..2], .little);
        @memset(out[@as(usize, w) * 16 ..][0..16], bits);
    }
    try d.copyIn(dst, std.mem.sliceAsBytes(out));
}

/// The trellis bytes of one part: (K/16)*(N/16)*(halves*8) i16 (the probe's identical byte
/// count, qwen27_main.zig:471), and its two fp16 scale buffers, K*2 and N*2 bytes each.
pub fn partBytes(g: Group) struct { tr: usize, scales: usize } {
    return .{
        .tr = @as(usize, g.k / 16) * @as(usize, g.n / 16) * (@as(usize, g.halves) * 8) * 2,
        .scales = (@as(usize, g.k) + @as(usize, g.n)) * 2,
    };
}

/// Fused EXL3 projection from the groups `prefix`++`parts`, at least `min_n` columns: the
/// structure of projectionGptq (load.zig:111-143) — size and pad N, allocate, stream each
/// part — without the repack launch: the device holds the raw trellis words and the host
/// normalized fp16 scales, and the EXL3 kernels do the rest. copyIn syncs its own chunks
/// (dev.zig:164-174), so no device staging slab is needed. `head` marks the lm_head
/// handle (the fp16 GEMV path, contract section 3). The Proj's w/s/bg stay inert for an
/// EXL3 handle — every GEMM path guards on it first — and point at part 0's buffers.
pub fn projectionExl3(d: *Dev, c: *const Checkpoint, prefix: []const u8, parts: []const []const u8, wide: bool, min_n: u32, head: bool) !Proj {
    var groups = std.ArrayList(Group).empty;
    defer groups.deinit(alloc);
    var n_total: u32 = 0;
    var k: u32 = 0;
    for (parts) |p| {
        const g = try admitGroup(c, try partName(prefix, p, ""));
        // fused parts of one slot were one projection before packing: they share the input
        // dim K, and every part must be mul1 (admitGroup refuses the rest). Bits may differ
        // per part — the stock packs mix precision inside a fused group (layer 3's q/k) —
        // and each Part carries its own tr/suh/svh/bits, so no agreement is needed there.
        if (groups.items.len > 0 and (g.k != k or g.codebook != groups.items[0].codebook)) {
            refuse("{s}{s}: fused parts disagree with part 0 on K or codebook", .{ prefix, p });
            return error.MismatchedFusedK;
        }
        k = g.k;
        n_total += g.n;
        try groups.append(alloc, g);
    }
    // The GPTQ rule verbatim (load.zig:120): the 256 tile is what the rest of the engine
    // assumes about Proj.n; the kernels themselves need only NT % 128 (exl3.cl:215,304).
    const nt = (@max(n_total, min_n) + 255) / 256 * 256;
    var device_parts = std.ArrayList(Part).empty;
    var n0: u32 = 0;
    for (groups.items) |g| {
        const bytes = partBytes(g);
        const tr = try d.alloc_(bytes.tr);
        try d.copyIn(tr, (try c.get(try partName(g.prefix, "", ".trellis"))).bytes);
        const suh = try d.alloc_(@as(usize, g.k) * 2);
        const svh = try d.alloc_(@as(usize, g.n) * 2);
        switch (g.style) {
            .f16 => {
                try d.copyIn(suh, (try c.get(try partName(g.prefix, "", ".suh"))).bytes);
                try d.copyIn(svh, (try c.get(try partName(g.prefix, "", ".svh"))).bytes);
            },
            .f16_16 => {
                try uploadReplicated(d, try c.get(try partName(g.prefix, "", ".suh")), g.k / 16, suh);
                try uploadReplicated(d, try c.get(try partName(g.prefix, "", ".svh")), g.n / 16, svh);
            },
            .words => {
                try uploadSigns(d, try c.get(try partName(g.prefix, "", ".su")), g.k / 16, suh);
                try uploadSigns(d, try c.get(try partName(g.prefix, "", ".sv")), g.n / 16, svh);
            },
        }
        try device_parts.append(alloc, .{ .tr = tr, .suh = suh, .svh = svh, .n0 = n0, .n = g.n, .halves = g.halves, .bits = g.bits, .codebook = g.codebook });
        n0 += g.n; // fused parts stack on N in slot order; the pad beyond n_total stays in nt
    }
    const hx = try alloc.create(Exl3);
    hx.* = .{ .parts = try device_parts.toOwnedSlice(alloc), .nt = nt, .head = head };
    const p0 = hx.parts[0];
    return .{ .w = p0.tr, .s = p0.suh, .bg = p0.svh, .n = nt, .k = k, .wide = wide, .gs = 64, .ilv = false, .exl3 = hx };
}
