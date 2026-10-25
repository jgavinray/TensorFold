//! EXL3 trellis encoding on Intel Xe2 (Level Zero): the tail-biting Viterbi encoder kernel
//! `exl3_encode` from exl3_enc.cl (fable spec sections 4 and 5, Layout G v1), driven over the
//! dev.Dev conventions (d.kernel, d.alloc_, d.launch, d.sync). One work-group of 256 items per
//! tile; the ping-pong cost arrays (count * 512 KiB), the pass-2 back-pointer bytes
//! (count * 256 * 2^(16 - min b)) and the per-tile 32-byte STAT live in device memory. The oracle is
//! core/exl3_format.zig; widths are the oracle's Width (mul1: halves 2..16, odd halves are the
//! half steps). queryLimits answers section 4's queries — the values the layout decision needs,
//! logged before anything assumes them.

const std = @import("std");
const ze = @import("ze");
const core = @import("core");
const dev = @import("dev.zig");
const kb = @import("xpu_qwen27_kernels");

pub const Width = core.exl3_format.Width;

pub const TileStat = extern struct {
    cost: u64 = 0,
    open_cost: u64 = 0,
    pinned: u32 = 0,
    status: u32 = 0, // Status: 0 optimal, 1 closed, 2 failed (the pre-fill value)
    flags: u32 = 0, // bit0 pruned in pass 1, bit1 pruned in pass 2 (fable rebase ruling)
    reserved: u32 = 0,
};

pub const Status = enum(u8) { optimal = 0, closed = 1, failed = 2 };

/// The section 4 queries; every value is printed by queryLimits, nothing here assumes one.
pub const Limits = struct { slm_bytes: u64 = 0, cache_bytes: u64 = 0, max_group: u32 = 0, xe_cores: u32 = 0 };

/// min over p of b(p): halves/2 at whole widths, (halves-1)/2 at the mul1 half steps.
pub fn minFreshBits(w: Width) u8 {
    return @intCast(if (w.halves % 2 == 1) (w.halves - 1) / 2 else w.halves / 2);
}

pub fn queryLimits(d: *dev.Dev) !Limits {
    const api = d.g.api;
    var cp: ze.ComputeProperties = .{};
    try ze.check(api.zeDeviceGetComputeProperties(d.g.device, &cp));
    var l = Limits{
        .slm_bytes = cp.maxSharedLocalMemory,
        .cache_bytes = 0,
        .max_group = cp.maxTotalGroupSize,
        .xe_cores = d.g.props.numSlices * d.g.props.numSubslicesPerSlice,
    };
    std.debug.print("exl3_enc T6: maxSharedLocalMemory={d} B  maxTotalGroupSize={d}  maxGroupSizeX={d}  subGroupSizes=[", .{ cp.maxSharedLocalMemory, cp.maxTotalGroupSize, cp.maxGroupSizeX });
    for (0..@min(cp.numSubGroupSizes, 8)) |s| std.debug.print("{s}{d}", .{ if (s == 0) "" else " ", cp.subGroupSizes[s] });
    std.debug.print("]  maxGroupCount=[{d},{d},{d}]\n", .{ cp.maxGroupCountX, cp.maxGroupCountY, cp.maxGroupCountZ });
    var nc: u32 = 0;
    try ze.check(api.zeDeviceGetCacheProperties(d.g.device, &nc, null)); // count form, ze_api.h
    if (nc > 0) {
        var caches: [16]ze.CacheProperties = @splat(.{});
        var count: u32 = @min(nc, caches.len);
        try ze.check(api.zeDeviceGetCacheProperties(d.g.device, &count, &caches));
        for (caches[0..count], 0..) |kp, i| {
            std.debug.print("exl3_enc T6: cache[{d}]: {d} bytes (flags 0x{x})\n", .{ i, kp.cacheSize, kp.flags });
            l.cache_bytes = @max(l.cache_bytes, kp.cacheSize);
        }
    }
    std.debug.print("exl3_enc T6: numSlices={d} numSubslicesPerSlice={d} (xe-cores {d}) numEUsPerSubslice={d} numThreadsPerEU={d} coreClockRate={d} Hz\n", .{
        d.g.props.numSlices, d.g.props.numSubslicesPerSlice, l.xe_cores, d.g.props.numEUsPerSubslice, d.g.props.numThreadsPerEU, d.g.props.coreClockRate });
    std.debug.print("exl3_enc T6: one tile's cost array is 256 KiB (the ping-pong pair 512 KiB): {s}\n", .{if (l.slm_bytes >= 262144 + 2048)
        "maxSharedLocalMemory fits a single 256 KiB cost array plus 2 KiB (Layout S candidate); Layout G (v1) runs as specified"
    else
        "maxSharedLocalMemory fits no cost array: Layout G (v1), costs in global memory"});
    return l;
}

/// The built kernel's own properties (spec section 4's zeKernelGetProperties line); the
/// register spill Gpu.kernel already warns about, the local footprint and spills printed here.
pub fn logKernelLimits(d: *dev.Dev, k: ze.H) !void {
    var kp: ze.KernelProperties = .{};
    try ze.check(d.g.api.zeKernelGetProperties(k, &kp));
    std.debug.print("exl3_enc T6: kernel: localMemSize={d} B privateMemSize={d} B spillMemSize={d} B requiredSubgroupSize={d} maxSubgroupSize={d} numKernelArgs={d}\n", .{
        kp.localMemSize, kp.privateMemSize, kp.spillMemSize, kp.requiredSubgroupSize, kp.maxSubgroupSize, kp.numKernelArgs });
}

/// The serving path's dequant kernel (exl3_dequant128_c from the exl3_h image), for T9.
pub fn dequantKernel(d: *dev.Dev) !ze.H {
    return d.kernel(try d.g.module(&kb.exl3_h), "exl3_dequant128_c", 256);
}

pub const Encoder = struct {
    k: ze.H, // exl3_encode from the exl3_enc image, group size 256
    w: Width,
    count: u32, // tiles per launch; the scratch is sized for it
    cost: *anyopaque, // count * 2 * 65536 * 4: the ping-pong cost arrays, 512 KiB per slot
    bp: *anyopaque, // count * 256 * 2^(16 - min b): pass 2's back-pointers, 2 MiB per tile at 3 bits
    stat: *anyopaque, // tiles_total * 32: the 32-byte TileStat per tile (ruling shapes)
    stat_bytes: usize,

    pub fn init(d: *dev.Dev, w: Width, count: u32, tiles_total: u32) !Encoder {
        const k = try d.kernel(try d.g.module(&kb.exl3_enc), "exl3_encode", 256);
        try logKernelLimits(d, k);
        const gmax: usize = @as(usize, 1) << @intCast(16 - minFreshBits(w));
        return .{
            .k = k,
            .w = w,
            .count = count,
            .cost = try d.alloc_(@as(usize, count) * 2 * 65536 * 4),
            .bp = try d.alloc_(@as(usize, count) * 256 * gmax),
            .stat = try d.alloc_(@as(usize, tiles_total) * 32),
            .stat_bytes = @as(usize, tiles_total) * 32,
        };
    }

    /// wq: device fp16 [K, N]; tr: the device int16 trellis for the whole part. Encodes tiles
    /// [first, first + n) of the k-major tile order, n <= count, in one launch of n groups.
    /// WHAT (the kernel's 10th argument) is null: the step-1 tests never ask for it.
    pub fn encode(e: *Encoder, d: *dev.Dev, wq: *anyopaque, k: u32, n_cols: u32,
                  tr: *anyopaque, first: u32, n: u32, on: ze.H) !void {
        try e.launch(d, wq, k, n_cols, tr, first, n, null, on);
    }

    /// encode() with the LDLQ driver's WHAT output (spec 4.3): a device fp16 [K, N] the
    /// kernel fills with codebookValue(.mul1, state[p]) at each placement slot. The caller
    /// owns tr (advanced per block: + j * nt * tileWords * 2 bytes) and passes geometry
    /// for a K=16 strip; the cost/bp arrays stay untouched, indexed by group id.
    pub fn encodeWhat(e: *Encoder, d: *dev.Dev, wq: *anyopaque, k: u32, n_cols: u32,
                      tr: *anyopaque, first: u32, n: u32, what: *anyopaque, on: ze.H) !void {
        try e.launch(d, wq, k, n_cols, tr, first, n, what, on);
    }

    fn launch(e: *Encoder, d: *dev.Dev, wq: *anyopaque, k: u32, n_cols: u32,
              tr: *anyopaque, first: u32, n: u32, what: ?*anyopaque, on: ze.H) !void {
        // the device contract (fable review): the kernel's workgroup-uniform trap returns
        // without writing STAT or TR, and alloc_ content is unspecified - a zero buffer
        // would decode as a perfect cost-0 optimal encode. Refuse on the host, and never
        // hand the kernel a tile range its STAT/TR/COST/BP indexing would run off the end of.
        if (e.w.halves < 2 or e.w.halves > 16) return error.EncodeWidth;
        if (k == 0 or n_cols == 0 or k % 16 != 0 or n_cols % 16 != 0) return error.EncodeShape;
        if (n == 0 or n > e.count) return error.EncodeRange;
        const tiles_total: u64 = @as(u64, k / 16) * @as(u64, n_cols / 16);
        if (@as(u64, first) + n > tiles_total) return error.EncodeRange;
        if ((@as(u64, first) + n) * 32 > e.stat_bytes) return error.EncodeRange;
        // pre-fill the launched STAT window with status 2 (failed): a launch whose trap or a
        // hang still reads back failed on those tiles, never the allocation's unspecified content
        const stat_pat = TileStat{ .status = 2 };
        if (on) |l| {
            try d.fillOn(l, dev.off(e.stat, @as(usize, first) * 32), stat_pat, @as(usize, n) * 32);
        } else {
            try d.fill(dev.off(e.stat, @as(usize, first) * 32), stat_pat, @as(usize, n) * 32);
        }
        const args = .{
            wq,
            @as(i32, @bitCast(k)),
            @as(i32, @bitCast(n_cols)),
            @as(i32, @intCast(e.w.halves)),
            @as(i32, @bitCast(first)),
            tr,
            e.cost,
            e.bp,
            e.stat,
            what,
        };
        if (on) |l| try d.launchOn(l, e.k, .{ .x = n }, args) else try d.launch(e.k, .{ .x = n }, args);
    }

    /// The per-tile certificates, tiles [0, out.len), through one host-visible copy.
    pub fn readStats(e: *Encoder, d: *dev.Dev, out: []TileStat, on: ze.H) !void {
        if (out.len * 32 > e.stat_bytes) return error.StatsRange;
        const host = try d.g.hostAlloc(out.len * 32);
        defer d.g.free(host);
        if (on) |l| try dev.copyOn(d, l, host, e.stat, out.len * 32) else try dev.copy(d, host, e.stat, out.len * 32);
        try d.sync();
        @memcpy(std.mem.sliceAsBytes(out), @as([*]const u8, @ptrCast(host))[0 .. out.len * 32]);
    }

    /// The encoder's trellis words back to host for the round-trip check (device -> USM host).
    pub fn readTrellis(e: *Encoder, d: *dev.Dev, tr: *anyopaque, out: []i16, on: ze.H) !void {
        _ = e;
        const host = try d.g.hostAlloc(out.len * 2);
        defer d.g.free(host);
        if (on) |l| try dev.copyOn(d, l, host, tr, out.len * 2) else try dev.copy(d, host, tr, out.len * 2);
        try d.sync();
        @memcpy(std.mem.sliceAsBytes(out), @as([*]const u8, @ptrCast(host))[0 .. out.len * 2]);
    }
};
