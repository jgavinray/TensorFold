//! Low-level XPU device plumbing for the Qwen3.8-27B runner: safetensors shards, the Level Zero device, profiling.
const std = @import("std");
const ze = @import("ze");
const core = @import("core");
const exl3_eaq = @import("exl3_eaq.zig");
const safetensors = @import("safetensors.zig");
const q = @import("qwen27.zig");
const fwd = @import("forward.zig");
const a8 = fwd.a8;
const alloc = q.alloc;

// ---------------------------------------------------------------- safetensors (mmap through libc)

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;
extern "c" fn munmap(addr: *anyopaque, len: usize) c_int;

pub const Tensor = safetensors.Tensor;

pub const Checkpoint = struct {
    tensors: std.ArrayList(Tensor) = .empty,
    maps: std.ArrayList([]const u8) = .empty, // the shards, mapped as long as the tensors' bytes are read
    headers: std.ArrayList(std.json.Parsed(std.json.Value)) = .empty, // the tensors' names live in them

    pub fn addShard(c: *Checkpoint, path: [:0]const u8) !void {
        const fd = open(path.ptr, 0);
        if (fd < 0) return error.FileNotFound;
        defer _ = close(fd);
        const end = lseek(fd, 0, 2);
        if (end < 8) return error.BadSafetensors;
        const size: usize = @intCast(end);
        const base = mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed; // PROT_READ, MAP_PRIVATE
        if (@intFromPtr(base) == std.math.maxInt(usize)) return error.MmapFailed; // MAP_FAILED
        const bytes = @as([*]const u8, @ptrCast(base))[0..size];
        errdefer _ = munmap(base, size);
        try c.maps.ensureUnusedCapacity(alloc, 1);
        try c.headers.ensureUnusedCapacity(alloc, 1);
        c.headers.appendAssumeCapacity(try safetensors.parse(alloc, bytes, &c.tensors));
        c.maps.appendAssumeCapacity(bytes);
    }

    pub fn deinit(c: *Checkpoint) void {
        for (c.headers.items) |h| h.deinit();
        for (c.maps.items) |m| _ = munmap(@ptrCast(@constCast(m.ptr)), m.len);
        c.tensors.deinit(alloc);
        c.maps.deinit(alloc);
        c.headers.deinit(alloc);
        c.* = .{};
    }

    pub fn get(c: *const Checkpoint, name: []const u8) !Tensor {
        for (c.tensors.items) |t| if (std.mem.endsWith(u8, t.name, name) and
            (t.name.len == name.len or std.mem.startsWith(u8, t.name, "language_model.") or std.mem.startsWith(u8, t.name, "model.language_model."))) return t;
        std.debug.print("missing tensor: {s}\n", .{name});
        return error.MissingTensor;
    }
    pub fn has(c: *const Checkpoint, name: []const u8) bool {
        for (c.tensors.items) |t| if (std.mem.endsWith(u8, t.name, name)) return true;
        return false;
    }

    /// The EAQ-aware open (hasEaq's yes): the reader's verdict first — exl3_eaq.openPack
    /// (the config's eaq block, the manifest against the hash config.json states, the
    /// marker, every shard's size and sha256, family qwen3_5, no fp8 entry) and
    /// exl3_eaq.map (every entry onto the 27B layer table: the complete Table, head and
    /// embedding included, fused parts agreed on K and codebook) — and only then the
    /// pack's shards, which are plain safetensors, through the very addShard the stock
    /// index feeds: the tensors view is the stock one, so every downstream lookup and the
    /// load.zig dispatch (isExl3 → loadExl3) see the pack as a set of admitted shards.
    /// The manifest and marker gate the door; they form no second load path. Any refusal
    /// names itself on the Why and opens nothing.
    pub fn openEaq(c: *Checkpoint, dir: []const u8) !void {
        var why: core.eaq_container.Why = .{};
        const io = std.Io.Threaded.global_single_threaded.io(); // pack verification is sync reads only
        var pack = exl3_eaq.openPack(alloc, io, dir, &why) catch |err| {
            std.debug.print("EAQ: {s} refused: {s} ('{s}')\n", .{ dir, @errorName(err), why.text() });
            return err;
        };
        defer pack.deinit(); // the Table borrows its arena: freed after the shard maps below
        why = .{};
        var table = exl3_eaq.map(alloc, &pack, &why) catch |err| {
            std.debug.print("EAQ: {s} maps to no Table: {s} ('{s}')\n", .{ dir, @errorName(err), why.text() });
            return err;
        };
        defer table.deinit(alloc);
        if (!table.complete) {
            std.debug.print("EAQ: {s}: the Table is incomplete: a layer slot, the head or the embedding has no entry\n", .{dir});
            return error.EaqIncomplete;
        }
        var n_trellis: usize = 0;
        for (pack.entries) |e| n_trellis += @intFromBool(e.format == .trellis);
        const hex = std.fmt.bytesToHex(pack.manifest_sha256, .lower);
        std.debug.print("  eaq: pack verified — family {s}, layout {s}, {s}, manifest {s}\n", .{ pack.family, core.eaq_container.layout, pack.version, &hex });
        std.debug.print("       {d} shards hashed, {d} entries ({d} trellis, {d} dense), Table: {d} projections, {d} others, head and embedding present\n", .{
            pack.shards.len, pack.entries.len, n_trellis, pack.entries.len - n_trellis, table.projections.len, table.others.len });
        // the same view the stock index path builds: every manifest shard, mmap for the
        // tensors' bytes, header parsed into c.headers; names keep the HF wrapper, which
        // Checkpoint.get and the loader's canonExl3 already strip
        for (pack.shards) |s| {
            const path = try std.fmt.allocPrintSentinel(alloc, "{s}", .{s.path}, 0);
            errdefer alloc.free(path);
            try c.addShard(path);
            alloc.free(path);
        }
    }
};

/// The dispatch's probe (qwen27.zig Runner.init): does the model dir hold the EAQ
/// manifest (core/eaq_container.zig manifest_name)? A stock pack answers no and takes the
/// index path untouched.
pub fn hasEaq(dir: []const u8) bool {
    const path = std.fmt.allocPrintSentinel(alloc, "{s}/{s}", .{ dir, core.eaq_container.manifest_name }, 0) catch return false;
    defer alloc.free(path);
    const fd = open(path.ptr, 0);
    if (fd < 0) return false;
    _ = close(fd);
    return true;
}

// ---------------------------------------------------------------- device

/// A list that will not drain may still read or write host memory its callers free: the process ends (tests swap it).
pub var lost: *const fn (rc: u32) void = exitLost;

fn exitLost(rc: u32) void {
    std.log.err("XPU: a command list did not drain (Level Zero 0x{x}): its copies may still use host memory, so the process ends", .{rc});
    std.process.exit(1);
}

pub const Dev = struct {
    g: ze.Gpu,
    list: ze.H,
    /// The quantize loop's batched list (see ze.Gpu.immediateAsync): async-mode appends,
    /// drained by settle. Null until a Driver takes it; launchOn/fillOn/copyOn name the
    /// list they append to, and every host read that follows a drained loop settles both.
    q_list: ze.H = null,
    staging: []u8 = &.{}, // USM host memory: the driver cannot copy from file-backed mmap pages directly
    allocated: usize = 0, // device bytes allocated (frees not counted: buffers live as long as the runner)
    k: struct {
        embed: ze.H,
        add_rmsnorm_dec: ze.H,
        add_rmsnorm: ze.H,
        gdn_pre: ze.H,
        gdn_pre_rows: ze.H,
        gated_norm: ze.H,
        attn_prep: ze.H,
        attn_gate: ze.H,
        swiglu: ze.H,
        argmax: ze.H,
        argmax_part: ze.H,
        argmax_part_masked: ze.H,
        tap: ze.H,
        conv_commit: ze.H,
        repack: ze.H,
        gdn_step: ze.H,
        attn_decode: ze.H,
        attn_merge: ze.H,
        kv_move: ze.H,
        exl3_gemv: ze.H,
        exl3_decode2: ze.H,
        exl3_decode2_t1: ze.H,
        exl3_dequant128: ze.H,
        exl3_prompt16: ze.H,
        exl3_decode2_h: ze.H,
        exl3_decode2_t1_h: ze.H,
        exl3_dequant128_c: ze.H,
        exl3_xform_h: ze.H,
        exl3_yscale_h: ze.H,
        rot_heads: ze.H,
        attn_prompt: ze.H,
        attn_prompt8: ze.H,
        embed_dense: ze.H,
        dense_rows: ze.H,
        repack_gptq: ze.H,
        quantize_mlx: ze.H,
        dec_wide_g128: ze.H,
        dec_wide64_g128: ze.H,
        dec_narrow_g128: ze.H,
        ngen_wide_g128: ze.H,
        ngen_narrow_g128: ze.H,
        ngen_swiglu: ze.H,
        ngen_swiglu_g128: ze.H,
        rowsums_ff: ze.H,
        fakequant_rows: ze.H,
        a8: ze.H,
        a8_swiglu: ze.H,
        quant_a8: ze.H,
        a4: ze.H,
        quant_a4: ze.H,
        dec_wide: ze.H,
        dec_wide64: ze.H,
        dec_narrow: ze.H,
        gdn_prep: ze.H,
        gdn_state: ze.H,
        gdn_out: ze.H,
        ngen_wide: ze.H,
        ngen_narrow: ze.H,
        // calibration capture (calib.cl): dense bf16 -> slab, X^T staging, H += X^T X
        bf16_to_slab: ze.H,
        transpose16: ze.H,
        exl3_syrk32: ze.H,
    },

    /// Waits for the queue, then gives the card back everything made on it (Gpu.close).
    pub fn deinit(d: *Dev) void {
        d.settle(); // the runner's host tables and staging go next: they must be idle
        d.g.close();
    }

    /// Waits for every command on the list; callers settle before host memory a queued copy reads or writes goes.
    pub fn settle(d: *Dev) void {
        if (d.q_list) |l| {
            const rcq = d.g.api.zeCommandListHostSynchronize(l, std.math.maxInt(u64));
            if (rcq != 0) lost(rcq);
        }
        const rc = d.g.api.zeCommandListHostSynchronize(d.list, std.math.maxInt(u64));
        if (rc != 0) lost(rc);
    }

    pub fn kernel(d: *Dev, m: ze.H, name: [*:0]const u8, local: u32) !ze.H {
        const k = try d.g.kernel(m, name);
        try ze.check(d.g.api.zeKernelSetGroupSize(k, local, 1, 1));
        return k;
    }

    pub fn alloc_(d: *Dev, bytes: usize) !*anyopaque {
        d.allocated += @max(bytes, 64);
        return d.g.deviceAlloc(@max(bytes, 64));
    }

    pub fn upload(d: *Dev, bytes: []const u8) !*anyopaque {
        const p = try d.alloc_(bytes.len);
        try d.copyIn(p, bytes);
        return p;
    }

    /// Host bytes (any memory, mmap included) to the device, through the USM staging buffer.
    /// Event-signaled: the list's settle (`d.sync`) returns early on a busy card, and the
    /// host had been reusing the staging (and launching behind the copy) while the DMA was
    /// still in flight — the fable 1b ruling's copy-delivery loss. Each chunk rides with a
    /// host-visible event and the caller blocks until the card really signals.
    var copy_ev: ze.H = null;
    pub fn copyIn(d: *Dev, dst: *anyopaque, bytes: []const u8) !void {
        if (d.staging.len == 0) d.staging = @as([*]u8, @ptrCast(try d.g.hostAlloc(256 << 20)))[0 .. 256 << 20];
        if (copy_ev == null) copy_ev = (try d.g.events(1))[0];
        var done: usize = 0;
        while (done < bytes.len) {
            const n = @min(d.staging.len, bytes.len - done);
            @memcpy(d.staging[0..n], bytes[done .. done + n]);
            _ = d.g.api.zeEventHostReset(copy_ev);
            try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, off(dst, done), d.staging.ptr, n, copy_ev, 0, null));
            const deadline = q.now() + 600;
            while (d.g.api.zeEventHostSynchronize(copy_ev, 300_000_000_000) != 0) {
                if (q.now() > deadline) return error.DeviceLost;
            }
            try d.sync();
            done += n;
        }
    }

    /// XPU_EXL3_TRACE: the kernel's own name, from the handle's field in d.k (the twin of
    /// the profiler's table, qwen27.zig:322), so a trace line names any launch, not only
    /// the EXL3 ones.
    pub fn kernelName(d: *Dev, k: ze.H) []const u8 {
        inline for (@typeInfo(@TypeOf(d.k)).@"struct".field_names) |name| {
            if (@field(d.k, name) == k) return name;
        }
        return "?";
    }

    pub fn launch(d: *Dev, k: ze.H, groups: ze.GroupCount, args: anytype) !void {
        try d.launchOn(d.list, k, groups, args);
    }

    /// launch on an explicit list — the ldlq driver's quantize loop batches its block's
    /// commands onto Dev.q_list (the sync-mode d.list pays a host round trip per append).
    pub fn launchOn(d: *Dev, list: ze.H, k: ze.H, groups: ze.GroupCount, args: anytype) !void {
        inline for (args, 0..) |a, i| try d.g.arg(k, i, a);
        var ev: ze.H = null;
        if (q.prof.on) {
            if (q.prof.used == q.prof.events.len) try d.sync();
            ev = q.prof.events[q.prof.used];
            q.prof.kernel[q.prof.used] = k;
            q.prof.used += 1;
        }
        try ze.check(d.g.api.zeCommandListAppendLaunchKernel(list, k, &groups, ev, 0, null));
        if (launchTracing()) { // the drain that pins a device-lost to the command that caused it
            launch_seq += 1;
            std.debug.print("EXL3TR launch #{d} {s} gx={d} gy={d} gz={d}\n", .{
                launch_seq, d.kernelName(k), groups.x, groups.y, groups.z });
            try d.sync();
        }
    }

    pub fn sync(d: *Dev) !void {
        d.settle();
        if (q.prof.on) {
            for (q.prof.events[0..q.prof.used], q.prof.kernel[0..q.prof.used]) |e, k| {
                const t = try d.g.seconds(e);
                try ze.check(d.g.api.zeEventHostReset(e));
                for (&q.prof.names, &q.prof.total) |n, *tot| {
                    if (n.k == k) {
                        tot.* += t;
                        break;
                    }
                }
            }
            q.prof.used = 0;
        }
    }

    pub fn fill(d: *Dev, p: *anyopaque, pattern: anytype, bytes: usize) !void {
        const T = @TypeOf(pattern);
        const host = try alloc.alloc(T, bytes / @sizeOf(T));
        defer alloc.free(host);
        @memset(host, pattern);
        try d.copyIn(p, std.mem.sliceAsBytes(host));
    }

    /// Device-side pattern fill on `list`: the batched loop's STAT pre-fill rides the
    /// queue with the launches instead of blocking on the copyIn event per call.
    pub fn fillOn(d: *Dev, list: ze.H, p: *anyopaque, pattern: anytype, bytes: usize) !void {
        const T = @TypeOf(pattern);
        if (bytes % @sizeOf(T) != 0 or @sizeOf(T) > 256) return error.BadDims;
        const pat: T = pattern;
        try ze.check(d.g.api.zeCommandListAppendMemoryFill(list, p, &pat, @sizeOf(T), bytes, null, 0, null));
    }
};

/// XPU_EXL3_TRACE: every queued command prints and the command list drains, so the last
/// line before a device-lost (Level Zero 0x70000001) is the command that caused it. The
/// EXL3 launches add their args to it (forward_exl3.trLaunch). Unset, one cached getenv
/// per launch and per copy.
var launch_trace: ?bool = null;
var launch_seq: u64 = 0;

fn launchTracing() bool {
    if (launch_trace == null) launch_trace = std.c.getenv("XPU_EXL3_TRACE") != null;
    return launch_trace.?;
}

pub fn copy(d: *Dev, dst: *anyopaque, src: *anyopaque, n: usize) !void {
    try copyOn(d, d.list, dst, src, n);
}

/// copy on an explicit list: a readback ordered after a batched loop's kernels rides
/// the same list (the list's own drain then covers the copy too).
pub fn copyOn(d: *Dev, list: ze.H, dst: *anyopaque, src: *anyopaque, n: usize) !void {
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(list, dst, src, n, null, 0, null));
    if (launchTracing()) std.debug.print("EXL3TR copy #{d} dst={x} src={x} n={d}\n", .{
        launch_seq, @intFromPtr(dst), @intFromPtr(src), n });
}

pub fn off(p: *anyopaque, bytes: usize) *anyopaque {
    return @ptrFromInt(@intFromPtr(p) + bytes);
}

/// XPU_PROFILE=1: GPU time per kernel, summed over the run.
pub const Prof = struct {
    on: bool = false,
    events: [4096]ze.H = @splat(null),
    kernel: [4096]ze.H = @splat(null),
    used: usize = 0,
    names: [64]struct { k: ze.H = null, name: []const u8 = "" } = @splat(.{}),
    total: [64]f64 = @splat(0),
};

pub fn profReport(label: []const u8) void {
    if (!q.prof.on) return;
    var sum: f64 = 0;
    for (q.prof.total) |t| sum += t;
    std.debug.print("GPU time, {s}: {d:.1} ms\n", .{ label, sum * 1e3 });
    for (q.prof.names, &q.prof.total) |n, *t| {
        if (t.* > 0) std.debug.print("  {s:<14} {d:8.2} ms {d:5.1} %\n", .{ n.name, t.* * 1e3, t.* / sum * 100 });
        t.* = 0;
    }
}
