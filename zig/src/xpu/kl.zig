//! The KL gate (eaq-ldlq-spec-fable.md §5) as corrected by kl-order-ruling.md:
//! the candidate side of the comparison.
//!
//! `Runner.prompt` over the evaluation rows with `XPU_DUMP_H` (qwen27.dump_h set after
//! `Runner.init`, so every chunk's final normed hidden rows fp16 [T, D] land in a memfd),
//! the pack's head dequantized once into a resident fp16 slab [K][NT] with the serving
//! kernel `exl3_dequant128_c` (the 2.54 GB the KL tool is allowed and serving refuses,
//! §5.1), and the logits as ONE fp32 device GEMM per row block — `eaq_gemm32`
//! (eaq_kl.cl): the exl3_syrk32 form of §2.1 at accumulate = 0 and the triangle skip off.
//! One head path for both sides keeps the fp16 GEMV rounding out of the comparison; the
//! runner's own fp16 head logits of the last row are kept as the §5.1 spot check.
//!
//! The statistic streams in fp64 over the vocab rows in 1024-row blocks, host, threaded
//! (`kl_stats.zig`): forward KL with the reference weighting the sum, both sides
//! max-subtracted log-sum-exp from fp32 logits, temperature 1, reported as
//! KL(ref||cand); the headline is the natural rows only, the random rows separate
//! (§5.2). A reference logits file (--ref, raw fp32 [positions][VOCAB] in prompts row
//! order — the xpu-eaq-ref contract), or `--self` (the candidate's own block as its
//! reference). `--cand-logits FILE` is the diagnostic path: the candidate comes from a
//! logits file instead of the runner and the head — with `--ref` naming the same file
//! that is the ref-vs-ref identity run (mean/median/p99 exactly 0, top-1 exactly 1,
//! enforced), and `--dump-hist` writes the per-position KL histogram that separates a
//! two-valued statistic kernel from an alignment or head-path defect. `--dump-kl`
//! writes the per-position rows for `--pair A.kl.bin B.kl.bin`, the paired-difference
//! block (mean/se of per-position differences, McNemar top-1 discordance).
const std = @import("std");
const q = @import("qwen27");
const ze = q.ze;
const kl_kernels = @import("eaq_kl_kernels");
pub const stats = @import("kl_stats.zig");

/// Source fingerprint of the gate itself (kl-order-ruling.md invariant 6, binary
/// provenance): a 64-bit FNV-1a over the two sources that carry the statistic and the
/// report, printed as 16 hex digits. Two builds with different numbers cannot share
/// this id; two builds with the same id compute the same statistic. (A comptime hash
/// std's crypto/Wyhash loops cannot survive: the sha2 rounds over a 110 KB embedded
/// file exhaust even a raised comptime branch budget.)
fn fnv1a(comptime data: []const u8) u64 {
    @setEvalBranchQuota(1000000);
    var h: u64 = 0xcbf2_9ce4_8422_2325;
    for (data) |b| {
        h ^= b;
        h *%= 0x0000_0100_0000_01b3;
    }
    return h;
}

pub const buildID: []const u8 = blk: {
    const h = fnv1a(@embedFile("kl.zig") ++ @embedFile("kl_stats.zig"));
    const hexdig = "0123456789abcdef";
    const out: [16]u8 = blk2: {
        var s: [16]u8 = undefined;
        for (0..16) |j| {
            const nib: u4 = @truncate(@as(u64, h >> @intCast(60 - 4 * j)));
            s[j] = hexdig[@as(usize, @intCast(nib))];
        }
        break :blk2 s;
    };
    break :blk out[0..16];
};

const devmod = q.dev; // the same dev.zig instance the runner runs on (qwen27 re-exports it)
extern "c" fn memfd_create(name: [*:0]const u8, flags: c_uint) c_int;
extern "c" fn pread(fd: c_int, buf: [*]u8, count: usize, off: i64) isize;
extern "c" fn pwrite(fd: c_int, buf: [*]const u8, count: usize, off: i64) isize;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) callconv(.c) *anyopaque;
extern "c" fn munmap(addr: *anyopaque, len: usize) c_int;
const PROT_RW: c_int = 0x1 | 0x2; // PROT_READ|PROT_WRITE
const MAP_PRIVATE_ANON_POP: c_int = 0x02 | 0x20 | 0x8000; // PRIVATE|ANONYMOUS|POPULATE
extern "c" fn close(fd: c_int) c_int;
const CTimespec = extern struct { tv_sec: i64, tv_nsec: i64 };
extern "c" fn nanosleep(req: *const CTimespec, rem: ?*CTimespec) c_int;

const alloc = q.alloc;

/// GB-scale preads on this box intermittently come back EFAULT/short even though
/// the buffer mapping is intact (reproduced while python reads the same bytes), so
/// read in 8 MiB chunks and retry each chunk instead of trusting one syscall.
fn preadAll(fd: c_int, buf: []u8, off: i64) !void {
    const chunk: usize = 8 << 20;
    var sent: usize = 0;
    while (sent < buf.len) {
        const want = @min(chunk, buf.len - sent);
        var tries: u32 = 0;
        var got: isize = pread(fd, buf.ptr + sent, want, off + @as(i64, @intCast(sent)));
        while (got < 0 or got == 0) : (tries += 1) {
            if (tries >= 64) {
                std.debug.print("  preadAll: off={d} want={d} got={d} errno={d}\n", .{ off + @as(i64, @intCast(sent)), want, got, std.c._errno().* });
                return error.PreadShort;
            }
            const ts = CTimespec{ .tv_sec = 0, .tv_nsec = 20 * std.time.ns_per_ms };
            _ = nanosleep(&ts, null);
            got = pread(fd, buf.ptr + sent, want, off + @as(i64, @intCast(sent)));
        }
        sent += @intCast(got);
    }
}

/// The driver's blocking host sync can report a list not yet drained (Level Zero
/// 0x70000001) on the gate's long commands — a multi-second GEMM or the ~1 GB logits
/// download — where the serving path's shorter commands never reach the wait cap. The
/// drain contract is kept: the hook returns only once the list is truly drained, and an
/// endless one still ends the process through the original handler.
var drain_dev: ?*q.Dev = null;
var drain_deadline: f64 = 0;
var drain_orig: *const fn (u32) void = undefined;

var drain_reports: usize = 0;

fn drainHook(rc: u32) void {
    if (rc == 0) return;
    const d = drain_dev orelse { drain_orig(rc); return; };
    drain_reports += 1;
    if (drain_reports == 1) std.debug.print("  drain: Level Zero reported the list not ready (0x{x}); waiting it out\n", .{rc});
    while (d.g.api.zeCommandListHostSynchronize(d.list, std.math.maxInt(u64)) != 0) {
        if (q.now() > drain_deadline) {
            devmod.lost = drain_orig;
            drain_orig(rc);
            return;
        }
        _ = nanosleep(&.{ .tv_sec = 0, .tv_nsec = 20_000_000 }, null);
    }
}

pub const Error = error{
    NoTrellisHead,
    HeadWidth,
    DenseHeadPart,
    BadHeadShape,
    MissingMode,
    StudyMode,
    RefTooSmall,
    PromptsHash,
    ManifestHash,
    ManifestRows,
    ColsMismatch,
    BadPrompts,
    HiddenShort,
    SelfCheck,
    FileIo,
    DumpShort,
    CandTooSmall, // the --cand-logits file shorter than positions·VOCAB·4
    CandShort, // a short pread of the candidate logits file mid-run
    PairMismatch, // --pair inputs disagree on prompts/reference/count, or the ref argmaxes differ
    CandidateDegenerate, // the device candidate fails the ruling's no-invariance bands
};

extern "c" fn open(path: [*:0]const u8, flags: c_int, mode: c_int) c_int;
const O_WRONLY_CREAT_TRUNC: c_int = 1 | 0o100 | 0o1000; // linux O_* agree on x86_64 and aarch64

/// Raw fd opens — the hot reads/pwrites below are pread/pwrite by fd, and 0.17's File
/// layer wants an Io context threaded through for what is a syscall with a sentinel.
fn openFd(path: []const u8, flags: c_int) !c_int {
    var stack: [4096]u8 = undefined;
    if (path.len == 0 or path.len >= stack.len) return Error.FileIo;
    @memcpy(stack[0..path.len], path);
    stack[path.len] = 0;
    const fd = open(stack[0..path.len :0], flags, 0o644);
    if (fd < 0) return Error.FileIo;
    return fd;
}

/// The study options that change the layer math must be off (spec H7): refuse loudly, do not silently measure a different model.
pub fn refuseStudyModes() void {
    for ([_][:0]const u8{ "XPU_A8", "XPU_A8_SITES", "XPU_A8_SITES4", "XPU_PREFILL_INT8", "XPU_PREFILL_INT4", "XPU_KV8", "XPU_KV_INT8", "XPU_DUMP_H" }) |v|
        if (std.c.getenv(v) != null) {
            std.debug.print("xpu-eaq-kl: {s} is set — the gate runs the serving math with every study option off\n", .{v});
            std.process.exit(2);
        };
}

pub const PartInfo = struct { n: u32, n0: u32, halves: u8 };
pub const HeadInfo = struct { k: u32, nt: u32, parts: []const PartInfo };

/// The pack's head as a resident fp16 slab plus the fp32 block GEMM, sized for one KL block.
pub const Head = struct {
    d: *q.Dev,
    k: u32,
    nt: u32, // the GEMM's pitch and column width (the Proj's padded n; VOCAB for an n-exact head)
    vocab: usize, // the KL width actually read: the parts' summed n
    info: HeadInfo,
    slab: *anyopaque, // fp16 [k][nt], zero-filled, one exl3_dequant128_c launch per part
    gemm: ze.H,
    copy16: ze.H, // eaq_copy16: same-list A delivery (fable 1b ruling)
    a: *anyopaque, // device fp16 [block][k]
    c: *anyopaque, // device fp32 [block][nt]
    block: u32,
    gl: ze.H, // private compute list: the gate's GEMM launches
    dl: ze.H, // private copy-group list: the logits downloads (host-synchronizable copies)
    stage: [*]u8, // USM host staging the downloads land in before the caller sees them
    ev: [7]ze.H, // host-visible events: [0] the GEMM, [1..] the slice copies, [4] the runner drain, [5] the A copy kernel, [6] the post-GEMM fence copy
    a_stage: *anyopaque, // device staging: H2D ends here on the copy list
    gemm_s: f64 = 0, // summed logitsBlock seconds, for the report

    pub fn init(r: *q.Runner, block: u32) !Head {
        const d = &r.d;
        const hx = r.model.head.exl3 orelse return Error.NoTrellisHead;
        const K = r.model.head.k;
        const nt = hx.nt;
        if (K % 128 != 0 or nt % 128 != 0 or block == 0 or block % 128 != 0) return Error.BadHeadShape;
        var sumn: u32 = 0;
        var n0: u32 = 0;
        for (hx.parts) |p| {
            if (p.dense) return Error.DenseHeadPart;
            if (p.n0 != n0) return Error.HeadWidth; // fused stacking: the reference's columns must be contiguous from 0
            sumn += p.n;
            n0 += p.n;
        }
        if (sumn != q.VOCAB) return Error.HeadWidth; // the reference logits are [positions][VOCAB]
        var parts = try alloc.alloc(PartInfo, hx.parts.len); // var: 0.17 const propagation makes const bindings read-only
        for (hx.parts, 0..) |p, j| parts[j] = .{ .n = p.n, .n0 = p.n0, .halves = p.halves };
        const info = HeadInfo{
            .k = K, .nt = nt,
            .parts = parts,
        };
        // the gate owns its command path: GEMM launches ride a private compute list,
        // the logits downloads a private copy-group list, and completion is taken from
        // host-visible events — not the runner's list (whose appends after the first
        // host sync never submit) nor CommandListHostSynchronize on a busy list (which
        // the driver answers early, delivering slices before they land)
        const gl = try d.g.immediate(0, 0);
        const dl = d.g.immediate(1, 0) catch gl; // group 1 is the copy engine on the B70
        const stage = @as([*]u8, @ptrCast(try d.g.hostAlloc(256 << 20)));
        const evs = try d.g.events(7);
        const self = Head{
            .d = d, .k = K, .nt = nt, .vocab = sumn, .info = info,
            .slab = try d.alloc_(@as(usize, K) * nt * 2),
            .gemm = try d.kernel(try d.g.module(&kl_kernels.eaq_kl), "eaq_gemm32", 256),
            .copy16 = try d.kernel(try d.g.module(&kl_kernels.eaq_kl), "eaq_copy16", 256),
            .a = try d.alloc_(@as(usize, block) * K * 2),
            .a_stage = try d.alloc_(@as(usize, block) * K * 2),
            .c = try d.alloc_(@as(usize, block) * nt * 4),
            .block = block,
            .gl = gl, .dl = dl, .stage = stage, .ev = evs,
        };
        errdefer {
            d.g.free(self.slab);
            d.g.free(self.a);
            d.g.free(self.c);
            d.g.free(self.a_stage);
        }
        try d.fill(self.slab, @as(u16, 0), @as(usize, K) * nt * 2); // the pad columns decode as zero once, as Acts.init does
        for (hx.parts) |p| { // the serving dequant: rotations and entry scales baked in (forward_exl3.gemmRows's wide launch, n0 the column base)
            try d.launch(d.k.exl3_dequant128_c, .{ .x = @intCast(K / 128), .y = (p.n + 127) / 128 }, .{
                p.tr,                  p.suh,   p.svh,   self.slab,       @as(i32, @intCast(K)),
                @as(i32, @intCast(p.n)), @as(i32, @intCast(nt)), @as(i32, @intCast(p.halves)), @as(i32, @intCast(p.n0)),
            });
        }
        try d.sync();
        return self;
    }

    pub fn deinit(h: *Head) void {
        h.d.g.free(h.slab);
        h.d.g.free(h.a);
        h.d.g.free(h.a_stage);
        h.d.g.free(h.stage);
    }

    /// One block: hidden fp16 [m][k] (host bytes) up, eaq_gemm32 over the slab, the fp32
    /// logits [m][nt] downloaded through USM staging in 256 MB event-signaled slices (a
    /// single ~1 GB copy truncates on this driver with no error). Returns with the host
    /// copy complete.
    pub fn logitsBlock(h: *Head, hidden: []const u8, m: u32, out: []f32, gy: u32) !void {
        std.debug.assert(out.len == @as(usize, m) * h.nt);
        const t0 = q.now();
        const g = &h.d.g;
        try h.drainRunner(); // dequant/other runner-list work must be truly done, not just settle-returned
        // A delivery per the fable 1b ruling (copy-delivery loss at the upload, not
        // the block-IO contract): the H2D lands on the copy list into device staging,
        // then eaq_copy16 runs on the GEMM's own compute list — in-order ahead of the
        // GEMM, on the engine that reads A — so no promotion or reordering can leave
        // DPAS reading rows the copy engine had not made visible.
        var up: usize = 0;
        while (up < hidden.len) {
            const n = @min(@as(usize, 256 << 20), hidden.len - up);
            @memcpy(h.stage[0 .. n], hidden[up .. up + n]);
            _ = g.api.zeEventHostReset(h.ev[1]);
            try ze.check(g.api.zeCommandListAppendMemoryCopy(h.dl, @ptrFromInt(@intFromPtr(h.a_stage) + up),
                h.stage, n, h.ev[1], 0, null));
            try h.evWait(h.ev[1]);
            up += n;
        }
        { // same-list kernel copy a_stage -> a, device-chained behind the last upload chunk
            const nu: u32 = @intCast(@divExact(hidden.len, 2));
            const gx: u32 = @min(@divCeil(nu, 1024), 8192);
            inline for (.{ h.a, h.a_stage, nu }, 0..) |aa, i| try g.arg(h.copy16, i, aa);
            _ = g.api.zeEventHostReset(h.ev[5]);
            var w1 = [1]ze.H{h.ev[1]};
            try ze.check(g.api.zeCommandListAppendLaunchKernel(h.gl, h.copy16, &ze.GroupCount{ .x = gx, .y = 1 }, h.ev[5], 1, &w1));
            try h.evWait(h.ev[5]);
        }
        // The single m=1024 launch loses ~87% of its workgroups' stores on this
        // driver and nothing on the submission side cures it: the battery shows the
        // identical dead_beef 87% / zero-on-device 0 / feed_face 0 signature under
        // same-list fences, device-side event chaining, and a second launch (the
        // m=1024 fp64 oracle: |delta|max 6.26e18, 234.9M of 254.3M entries outside
        // the band), while m=128 launches at the same y-grid are proven exact at
        // complete coverage. So the block streams as proven-green 128-row launches:
        // each chunk's GEMM is chained behind the copy event, fenced by a same-list
        // copy on the GEMM's own event, and its rows download only after the fence
        // host-side and on the GEMM's event device-side.
        const gY: u32 = if (gy > 0) @intCast(gy) else @intCast(h.nt / 128);
        const nu: u32 = @intCast(@divExact(hidden.len, 2));
        const fence_x: u32 = @min(@divCeil(nu, 1024), 8192);
        var c: u32 = 0;
        while (c < m) : (c += 128) {
            const mc: u32 = @min(128, m - c);
            const a_ptr = @as([*]const u16, @ptrFromInt(@intFromPtr(h.a) + @as(usize, c) * h.k * 2));
            const c_ptr = @as([*]f32, @ptrFromInt(@intFromPtr(h.c) + @as(usize, c) * h.nt * 4));
            inline for (.{
                a_ptr, h.slab, c_ptr,
                @as(i32, @intCast(mc)), @as(i32, @intCast(h.nt)), @as(i32, @intCast(h.k)),
            }, 0..) |a, i| try g.arg(h.gemm, i, a);
            _ = g.api.zeEventHostReset(h.ev[0]);
            var w5 = [1]ze.H{h.ev[5]};
            try ze.check(g.api.zeCommandListAppendLaunchKernel(h.gl, h.gemm, &ze.GroupCount{ .x = 1, .y = gY }, h.ev[0], 1, &w5));
            { // the fence copy rides the same list behind the GEMM's event; the
                // chunk's rows are only safe to download once it has retired
                inline for (.{ h.a, h.a_stage, nu }, 0..) |aa, i| try g.arg(h.copy16, i, aa);
                _ = g.api.zeEventHostReset(h.ev[6]);
                var w0 = [1]ze.H{h.ev[0]};
                try ze.check(g.api.zeCommandListAppendLaunchKernel(h.gl, h.copy16, &ze.GroupCount{ .x = fence_x, .y = 1 }, h.ev[6], 1, &w0));
                try h.evWait(h.ev[6]); // transitively: GEMM chunk done, stores flushed
            }
            var done: usize = @as(usize, c) * h.nt;
            const stop = @as(usize, c + mc) * h.nt;
            while (done < stop) {
                const n = @min(@as(usize, 64 << 20), stop - done);
                _ = g.api.zeEventHostReset(h.ev[1]);
                var w0 = [1]ze.H{h.ev[0]}; // the copy engine waits on the GEMM itself
                try ze.check(g.api.zeCommandListAppendMemoryCopy(h.dl, h.stage,
                    @ptrFromInt(@intFromPtr(h.c) + done * 4), n * 4, h.ev[1], 1, &w0));
                try h.evWait(h.ev[1]);
                @memcpy(std.mem.sliceAsBytes(out[done .. done + n]), h.stage[0 .. n * 4]);
                done += n;
            }
        }
        h.gemm_s += q.now() - t0;
    }

    /// Blocks on an event with a real per-call timeout and an overall deadline; past
    /// the deadline the card is treated as lost, like settle()'s path.
    fn evWait(h: *Head, ev: ze.H) !void {
        const deadline = q.now() + 600; // seconds
        while (true) {
            if (h.d.g.api.zeEventHostSynchronize(ev, 300_000_000_000) == 0) return;
            if (q.now() > deadline) return error.DeviceLost;
        }
    }

    /// Event-signaled device-to-host of any gate buffer through the USM staging (the
    /// probe forensic reads: slab, C, c16, DA, DB — the runner list's settle cannot be
    /// trusted to have landed the copy when it returns success).
    pub fn download(h: *Head, dst: []u8, src: *anyopaque, bytes: usize) !void {
        std.debug.assert(dst.len >= bytes);
        const g = &h.d.g;
        var done: usize = 0;
        while (done < bytes) {
            const n = @min(@as(usize, 64 << 20), bytes - done);
            _ = g.api.zeEventHostReset(h.ev[1]);
            try ze.check(g.api.zeCommandListAppendMemoryCopy(h.dl, h.stage,
                @ptrFromInt(@intFromPtr(src) + done), n, h.ev[1], 0, null));
            try h.evWait(h.ev[1]);
            @memcpy(dst[done ..][0..n], h.stage[0..n]);
            done += n;
        }
    }

    /// Event-signaled host-to-device on the GEMM's own in-order list (the probe's A and
    /// sample-index uploads: never copyIn, whose settle returns early on a busy list).
    pub fn upload(h: *Head, dst: *anyopaque, bytes: []const u8) !void {
        const g = &h.d.g;
        var up: usize = 0;
        while (up < bytes.len) {
            const n = @min(@as(usize, 256 << 20), bytes.len - up);
            @memcpy(h.stage[0..n], bytes[up .. up + n]);
            _ = g.api.zeEventHostReset(h.ev[1]);
            try ze.check(g.api.zeCommandListAppendMemoryCopy(h.gl, @ptrFromInt(@intFromPtr(dst) + up),
                h.stage, n, h.ev[1], 0, null));
            try h.evWait(h.ev[1]);
            up += n;
        }
    }

    /// Launch any kernel of the gate's module on the in-order compute list and block on
    /// its event (probe only: eaq_dump_areg / eaq_ab_store16).
    pub fn launchProbe(h: *Head, k: ze.H, gx: u32, gy: u32, args: anytype) !void {
        const g = &h.d.g;
        inline for (args, 0..) |a, i| try g.arg(k, i, a);
        const groups = ze.GroupCount{ .x = gx, .y = gy };
        _ = g.api.zeEventHostReset(h.ev[0]);
        try ze.check(g.api.zeCommandListAppendLaunchKernel(h.gl, k, &groups, h.ev[0], 0, null));
        try h.evWait(h.ev[0]);
    }

    /// Event-signaled fill of the block's C: d.fill shares the premature-sync path the
    /// gate cannot trust when the next command rides another list (probes only; the
    /// init-time slab fill stays on d.fill because its dequant follows on the same list).
    pub fn fillC(h: *Head, bytes: usize, pattern: u32) !void {
        const g = &h.d.g;
        std.debug.assert(bytes % 4 == 0);
        var off: usize = 0;
        while (off < bytes) {
            const n = @min(@as(usize, 256 << 20), bytes - off);
            @memset(std.mem.bytesAsSlice(u32, h.stage[0 .. n]), pattern);
            _ = g.api.zeEventHostReset(h.ev[2]);
            try ze.check(g.api.zeCommandListAppendMemoryCopy(h.gl, @ptrFromInt(@intFromPtr(h.c) + off),
                h.stage, n, h.ev[2], 0, null));
            try h.evWait(h.ev[2]);
            off += n;
        }
    }

    /// Event-signaled readback of A bytes (probe forensics: what the device really holds).
    pub fn readA(h: *Head, byte_off: usize, dst: []u8) !void {
        _ = h.d.g.api.zeEventHostReset(h.ev[3]);
        try ze.check(h.d.g.api.zeCommandListAppendMemoryCopy(h.dl, dst.ptr,
            @ptrFromInt(@intFromPtr(h.a) + byte_off), dst.len, h.ev[3], 0, null));
        try h.evWait(h.ev[3]);
    }

    /// Truly drain the runner list before a gate command rides another list. A memory
    /// copy CANNOT be the marker: the driver dispatches copies to the copy engine and
    /// kernels to the compute engine with no cross-engine order inside one immediate
    /// list, so a 4-byte self-copy completes while the queued kernels are still
    /// in flight — every download that follows then reads the stream mid-sweep (the
    /// resid/s63 forensic reads, the head-input checksums). The marker is a compute
    /// kernel instead — eaq_copy16 of one ushort at C, in-order on the compute engine
    /// behind every queued kernel — and the caller's download can only be enqueued
    /// after this returns, which is what orders the copy engine behind the kernels.
    pub fn drainRunner(h: *Head) !void {
        const g = &h.d.g;
        const one: u32 = 1;
        inline for (.{ h.c, h.c, one }, 0..) |arg, i| try g.arg(h.copy16, i, arg);
        _ = g.api.zeEventHostReset(h.ev[4]);
        try ze.check(g.api.zeCommandListAppendLaunchKernel(h.d.list, h.copy16,
            &ze.GroupCount{ .x = 1, .y = 1, .z = 1 }, h.ev[4], 0, null));
        try h.evWait(h.ev[4]);
    }
};

/// Arm the process on the drain hook (the driver's not-yet-drained reports on the gate's
/// long commands); disarm restores the runner's own loss handler.
pub fn armDrain(d: *q.Dev) void {
    drain_dev = d;
    drain_deadline = q.now() + 7200;
    drain_orig = devmod.lost;
    devmod.lost = drainHook;
}

pub fn disarmDrain() void {
    devmod.lost = drain_orig;
    drain_dev = null;
}

pub const Probe = struct {
    slab_nonzero: usize = 0,
    slab_total: usize = 0,
    slab_max: f64 = 0,
    oracle_max_abs: f64 = 0,
    oracle_rms: f64 = 0,
    gemm_max_abs: f64 = 0,
    gemm_max_ref: f64 = 0,
    dequant_ok: bool = false,
    gemm_ok: bool = false,
};

/// On-card probe (measure before fixing): the slab copied back for density, part 0's
/// leading 128-column strip checked against the core/exl3_format oracle with the
/// fidelity site's strip gather (fp64_ref.zig:555), and an all-ones block through the
/// fp32 GEMM against the slab's host column sums. The verdict fields must be true
/// before any measured KL is trusted.
pub fn probeSlab(r: *q.Runner) !Probe {
    const core = @import("core");
    var head = try Head.init(r, 128);
    defer head.deinit();
    const d = &r.d;
    const K: usize = head.k;
    const nt: usize = head.nt;
    var pr = Probe{ .slab_total = K * head.vocab };
    const slab = try alloc.alloc(u16, K * nt);
    defer alloc.free(slab);
    try devmod.copy(d, slab.ptr, head.slab, slab.len * 2);
    try d.sync();
    for (slab) |hw| {
        const v: f64 = @abs(@as(f16, @bitCast(hw)));
        if (v != 0) pr.slab_nonzero += 1;
        if (v > pr.slab_max) pr.slab_max = v;
    }
    const hx = r.model.head.exl3 orelse return Error.NoTrellisHead;
    const p0 = hx.parts[0];
    const kt = K / 16;
    const words: usize = @as(usize, p0.halves) * 8;
    const strip = try alloc.alloc(i16, kt * 8 * words);
    defer alloc.free(strip);
    for (0..kt) |tk| { // the checkpoint stores tiles K-major in N: gather the strip's tile rows contiguous
        const src = @as([*]u8, @ptrFromInt(@intFromPtr(p0.tr) + tk * (p0.n / 16) * words * 2));
        try devmod.copy(d, strip[tk * 8 * words ..][0 .. 8 * words].ptr, src, 8 * words * 2);
    }
    try d.sync();
    const suh = try alloc.alloc(f16, K);
    defer alloc.free(suh);
    const svh = try alloc.alloc(f16, 128);
    defer alloc.free(svh);
    try devmod.copy(d, suh.ptr, p0.suh, K * 2);
    try devmod.copy(d, svh.ptr, p0.svh, 128 * 2);
    try d.sync();
    const w64 = try alloc.alloc(f64, K * 128);
    defer alloc.free(w64);
    const scratch = try alloc.alloc(f64, 2 * K * 128);
    defer alloc.free(scratch);
    core.exl3_format.dequantize(.{ .halves = p0.halves }, .mul1, strip[0 .. kt * 8 * words], kt, 8, suh, svh, w64, scratch);
    var ssq: f64 = 0;
    for (0..K) |kk| for (0..128) |cn| {
        const got: f64 = @floatCast(@as(f16, @bitCast(slab[kk * nt + cn])));
        const want = w64[kk * 128 + cn];
        const dd = @abs(got - want);
        if (dd > pr.oracle_max_abs) pr.oracle_max_abs = dd;
        ssq += want * want;
    };
    pr.oracle_rms = @sqrt(ssq / @as(f64, @floatFromInt(K * 128)));
    pr.dequant_ok = pr.oracle_rms > 0 and pr.oracle_max_abs < pr.oracle_rms * 0.004; // the fidelity bound, 2^-8 of the oracle's RMS
    // the GEMM against the all-ones block at one tile and at a full block: rows past the
    // first 128 exercise grid-x tiles the probe's 128-row launch never reached
    var head_big = try Head.init(r, 1024);
    defer head_big.deinit();
    const ones = try alloc.alloc(u16, 1024 * K);
    defer alloc.free(ones);
    @memset(ones, @as(u16, @bitCast(@as(f16, 1))));
    const out = try alloc.alloc(f32, 1024 * nt);
    defer alloc.free(out);
    try head_big.fillC(1024 * nt * 4, 0xdead_beef); // event-drained: d.fill's settle returns early
    const host_sent = @as(u32, 0xfeed_face);
    { // host-side sentinel too: untouched host slots mean the download dropped them, not the kernel
        const ob = std.mem.sliceAsBytes(out);
        var bi: usize = 0;
        while (bi < ob.len) : (bi += 4) @memcpy(ob[bi .. bi + 4], std.mem.asBytes(&host_sent));
    }
    const probe_y: u32 = if (std.c.getenv("XPU_KL_PROBE_Y")) |cs| std.fmt.parseInt(u32, std.mem.span(cs), 10) catch 0 else 0;
    try head_big.logitsBlock(std.mem.sliceAsBytes(ones), 1024, out, probe_y);
    { // the A buffer as the device holds it now, at rows that came back zeroed
        const arow = try alloc.alloc(u16, K);
        defer alloc.free(arow);
        for ([_]usize{ 5, 900, 1023 }) |row| {
            try head_big.readA(row * K * 2, std.mem.sliceAsBytes(arow));
            var ones_n: usize = 0;
            for (arow) |vv| { if (vv == 0x3c00) ones_n += 1; }
            std.debug.print("  A row {d}: {d}/{d} ones\n", .{ row, ones_n, K });
        }
    }
    const Bad = struct { row: usize, n: usize, got: f64, want: f64 };
    var bads: [8]Bad = @splat(.{ .row = 0, .n = 0, .got = 0, .want = 0 });
    var nbad: usize = 0;
    var n_sent: usize = 0;
    var n_zero: usize = 0;
    var row_bad: [1024]u16 = @splat(0);
    for (0..1024) |row| {
        var n: usize = 0;
        while (n < head_big.vocab) : (n += 1571) {
            var want: f64 = 0;
            for (0..K) |kk| want += @as(f64, @floatCast(@as(f16, @bitCast(slab[kk * nt + n]))));
            const gotv: f64 = out[row * nt + n];
            const dd = @abs(gotv - want);
            if (dd > pr.gemm_max_abs) pr.gemm_max_abs = dd;
            if (@abs(want) > pr.gemm_max_ref) pr.gemm_max_ref = @abs(want);
            if (dd > 0.25) {
                if (nbad < bads.len) bads[nbad] = .{ .row = row, .n = n, .got = gotv, .want = want };
                nbad += 1;
                if (row_bad[row] < 400) row_bad[row] += 1;
                const bu: u32 = @bitCast(out[row * nt + n]);
                if (bu == 0xfeed_face) n_sent += 1 else if (out[row * nt + n] == 0) n_zero += 1;
            }
        }
    }
    var rr: usize = 0;
    while (rr < 1024) : (rr += 1) {
        if (rr == 0 or rr == 1023 or (row_bad[rr] != 0) != (row_bad[rr - 1] != 0))
            std.debug.print("  row {d}: {d} bad\n", .{ rr, row_bad[rr] });
    }
    std.debug.print("  gemm classes: untouched-device {d}, zero-on-device {d}\n", .{ n_sent, n_zero });
    // the stride-1571 sample is blind to tile-aligned column patterns; sample every
    // 8th column tile with 16 interior points to see whether zeros follow the
    // workgroup-tile schedule or are scattered below tile level
    var tzero: usize = 0;
    var tsamp: usize = 0;
    var c_zero: usize = 0; // real zeros in C: the kernel computed zero
    var c_face: usize = 0; // host-side poison: the download never delivered the slice
    var c_beef: usize = 0; // device fill sentinel: the kernel never wrote the tile
    var per_mt: [8]usize = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
    for (0..8) |mt| {
        var ntile: usize = 0;
        while (ntile * 128 < head_big.vocab) : (ntile += 8) {
            tsamp += 1;
            var z: usize = 0;
            for ([4]usize{ 5, 37, 69, 101 }) |moff| {
                for ([4]usize{ 3, 35, 67, 99 }) |noff| {
                    const n = ntile * 128 + noff;
                    if (n >= head_big.vocab) continue;
                    const bu: u32 = @bitCast(out[(mt * 128 + moff) * nt + n]);
                    if (bu == 0 or bu == 0xfeed_face or bu == 0xdead_beef) {
                        z += 1;
                        if (bu == 0) c_zero += 1 else if (bu == 0xfeed_face) c_face += 1 else c_beef += 1;
                    }
                }
            }
            if (z != 0) { tzero += 1; per_mt[mt] += 1; }
        }
    }
    std.debug.print("  tiles: {d}/{d} sampled have zeros, per-mt {d} {d} {d} {d} {d} {d} {d} {d}\n", .{ tzero, tsamp, per_mt[0], per_mt[1], per_mt[2], per_mt[3], per_mt[4], per_mt[5], per_mt[6], per_mt[7] });
    std.debug.print("  tile classes split: real-zero {d}, host-poison feed_face {d}, device-fill dead_beef {d} (of {d} probes)\n", .{ c_zero, c_face, c_beef, tsamp * 16 });
    if (probe_y > 0) { // second pass: one column tile only
        try head_big.fillC(1024 * nt * 4, 0xdead_beef);
            try head_big.logitsBlock(std.mem.sliceAsBytes(ones), 1024, out, 1);
            var z1: usize = 0;
            var r1: usize = 0;
            for (0..8) |mt| for ([4]usize{ 5, 37, 69, 101 }) |moff| for ([4]usize{ 3, 35, 67, 99 }) |noff| {
                const bu: u32 = @bitCast(out[(mt * 128 + moff) * nt + noff]);
                if (bu == 0 or bu == 0xfeed_face or bu == 0xdead_beef) z1 += 1 else r1 += 1;
            };
            std.debug.print("  y=1 tile0: {d} zero/sentinel, {d} real\n", .{ z1, r1 });
    }
    pr.gemm_ok = pr.gemm_max_abs < @max(0.25, pr.gemm_max_ref * 0.002); // fp16 DPAS across K=5120
    return pr;
}

/// The probe as the `--probe` CLI runs it: the pack's runner, the drain hook armed for
/// the slab copy-back and the block GEMM, the measurements of `probeSlab`, and the
/// A/B head-GEMM battery of `probeGemm`. The printed lines are the A/B dump evidence;
/// XPU_KL_SPOT_DUMP=PATH additionally writes the raw forensic bytes (DA, DB, the
/// m=1024/m=128 fp32 logits, the fp16-store C).
pub fn probePack(pack_dir: []const u8, context: u32) !struct { Probe, GemmAB } {
    refuseStudyModes();
    const r = try q.Runner.init(pack_dir, context);
    defer r.deinit();
    armDrain(&r.d);
    defer disarmDrain();
    const pr = try probeSlab(r);
    const ab = try probeGemm(r);
    return .{ pr, ab };
}

// ---------------------------------------------------------------- A/B head-GEMM probe
//
// The defect this battery isolates: with an m=1024 block the gate's fp32 GEMM writes
// garbage (KlGate's probeC: |delta| up to 6.26e18) at ~84% of columns on every row
// outside a ~77-row band, while the m=128 launch is fully green against the host fp64
// oracle — and the band moves whenever only the host-side copy/sync stack changes.
// A and the slab are device-verified (probeC: A rows read back 5120/5120 ones), so the
// loss is inside the kernel's read/compute/store path. Three hypotheses under test:
// the intel_sub_group_2d_block_read_* y-coordinate/height contract for A rows past the
// first tile row, the 32b block-store path, and per-workgroup behavior in x. The
// battery runs the same buffers through:
//  * random integer A (row-distinguishable): fp64 oracle over every entry, m=1024 and
//    m=128, the |delta| band 4.6e-5;
//  * the in-kernel register dumps (eaq_dump_areg/eaq_dump_breg: eaq_gemm32's body
//    verbatim plus per-lane stores of the exact prompt16 IO call results), multiset-
//    checked per block against the host A and slab — this is the y/height verdict;
//  * the exl3_prompt16-verbatim 16b-store twin (eaq_ab_store16) at the same buffers —
//    the store-path A/B;
//  * the k-select pattern (A[r][k] = 1 iff k/16 == r % 320): row r's logits must be
//    exactly the slab's rows 16*(r%320)..+16 summed — zeros mean the read got zeros,
//    a match for another slab row names the miscoordinate directly.
pub const GemmAB = struct {
    band: f64 = 4.6e-5,
    oracle_max_abs_m1024: f64 = 0,
    oracle_max_abs_m128: f64 = 0,
    viol_m1024: usize = 0, // entries outside the 4.6e-5 band, all 1024 x 248320
    viol_m128: usize = 0,
    zero_rows_m1024: usize = 0, // rows whose first 256 entries are all exact zeros
    ks_zero_rows: usize = 0, // k-select rows that read zeros
    ks_mix_rows: usize = 0, // k-select rows that read another row's block
    ks_first_mix: []const u8 = "", // "row r saw slab block b" for the first mix
    a_read_ok: bool = false, // every dumped A register is the expected multiset
    b_read_ok: bool = false,
    a_dump_blocks: usize = 0,
    a_dump_bad: usize = 0,
    b_dump_blocks: usize = 0,
    b_dump_bad: usize = 0,
    store16_max_abs: f64 = 0, // |got - fp16(oracle)| over sampled entries
    store16_zero_rows: usize = 0,
    // the single x=8 eaq_gemm32 launch (the geometry the shipped 128-row chunk stream
    // sidesteps): rows the 32b block store never landed, rows it zeroed, rows wrong
    raw1024_dropped_rows: usize = 0, // sampled cols all still 0xdeadbeef
    raw1024_zero_rows: usize = 0, // sampled cols all exact zero against a nonzero oracle
    raw1024_viol: usize = 0, // sampled entries outside the band, sentinels excluded
    raw1024_bad_groups: usize = 0, // row groups (of 8) carrying any defect
    ok: bool = false,
};

fn abSort(comptime T: type, x: []T) void {
    std.sort.pdq(T, x, {}, struct {
        fn lt(_: void, a: T, b: T) bool {
            return a < b;
        }
    }.lt);
}

/// Random row-distinguishable A: integer halves in {-3..3}, wyhash-deterministic.
fn abPattern(a: std.mem.Allocator, rows: usize, K: usize, seed: u64) ![]u16 {
    const A = try a.alloc(u16, rows * K);
    for (0..rows) |r| {
        for (0..K) |k| {
            const key: u64 = (@as(u64, @intCast(r)) *% 0x9E37_79B9_7F4A_7C15) ^ (@as(u64, @intCast(k)) *% 0xD1B5_4A32_D192_ED03);
            const v: u8 = @truncate(std.hash.Wyhash.hash(seed, std.mem.asBytes(&key)));
            const iv: i8 = @intCast(v % 7);
            A[r * K + k] = @bitCast(@as(f16, @floatFromInt(iv - 3)));
        }
    }
    return A;
}

const AbOracle = struct {
    A: []const u16, slab: []const u16, out: []const f32,
    K: usize, nt: usize, band: f64,
    row_max: []f64, row_viol: []u32, // per row: max |delta|, entries outside band
};

fn abOracleRows(o: *AbOracle, at: usize, to: usize) void {
    for (at..to) |row| {
        var mx: f64 = 0;
        var nv: u32 = 0;
        for (0..o.nt) |n| {
            var acc: f64 = 0;
            for (0..o.K) |kk| acc += @as(f64, @floatCast(@as(f16, @bitCast(o.A[row * o.K + kk])))) *
                @as(f64, @floatCast(@as(f16, @bitCast(o.slab[kk * o.nt + n]))));
            const d: f64 = @abs(@as(f64, o.out[row * o.nt + n]) - acc);
            if (d > mx) mx = d;
            if (d > o.band) nv += 1;
        }
        o.row_max[row] = mx;
        o.row_viol[row] = nv;
    }
}

fn abOracleJob(o: *AbOracle, at: usize, to: usize) void {
    abOracleRows(o, at, to);
}

/// One fp64 oracle over the A rows in parallel across threads.
fn abOracle(o: *AbOracle, rows: usize, threads: usize) !void {
    const Seg = struct { o: *AbOracle, at: usize, to: usize };
    var jobs: [64]Seg = undefined;
    var list: [64]?std.Thread = @splat(null);
    const per = (rows + threads - 1) / @max(threads, 1);
    var n: usize = 0;
    var at: usize = 0;
    while (at < rows) : (n += 1) {
        const to = @min(rows, at + per);
        jobs[n] = .{ .o = o, .at = at, .to = to };
        list[n] = std.Thread.spawn(.{}, abOracleJob, .{ jobs[n].o, at, to }) catch null;
        at = to;
    }
    for (list[0..n]) |t| if (t) |th| th.join();
    for (jobs[0..n], 0..) |j, i| if (list[i] == null) abOracleJob(j.o, j.at, j.to);
}

/// The sample columns for the row classification passes: the first 64, the last, and
/// every 1571st in between.
fn abSampleCols(a: std.mem.Allocator, nt: usize) ![]usize {
    var list: std.ArrayList(usize) = .empty;
    for (0..@min(nt, 64)) |n| try list.append(a, n);
    var n: usize = 64;
    while (n < nt - 1) : (n += 1571) try list.append(a, n);
    try list.append(a, nt - 1);
    return list.items;
}

pub fn probeGemm(r: *q.Runner) !GemmAB {
    var ab = GemmAB{};
    var head = try Head.init(r, 1024);
    defer head.deinit();
    const d = &r.d;
    const K: usize = head.k; // 5120
    const nt: usize = head.nt; // 248320
    const threads = @min(std.Thread.getCpuCount() catch 4, 64);

    // the slab the GEMM multiplies, as the host can hold it: every oracle number below
    // keys off this copy, event-drained through the gate's own download path
    const slab = try alloc.alloc(u16, K * nt);
    defer alloc.free(slab);
    try head.drainRunner(); // Head.init's slab dequant rode d.list on d.sync, the premature-sync path
    try head.download(std.mem.sliceAsBytes(slab), head.slab, K * nt * 2);

    // -- 1) random integer A through the clean GEMM, full-precision oracle, both sizes
    const A = try abPattern(alloc, 1024, K, 0xEA0C_0DDE);
    defer alloc.free(A);
    const out = try alloc.alloc(f32, 1024 * nt);
    defer alloc.free(out);
    try head.fillC(1024 * nt * 4, 0xdead_beef);
    try head.logitsBlock(std.mem.sliceAsBytes(A), 1024, out, 0);
    var oracle = AbOracle{
        .A = A, .slab = slab, .out = out, .K = K, .nt = nt, .band = ab.band,
        .row_max = try alloc.alloc(f64, 1024), .row_viol = try alloc.alloc(u32, 1024),
    };
    defer {
        alloc.free(oracle.row_max);
        alloc.free(oracle.row_viol);
    }
    std.debug.print("  AB: fp64 oracle over all 1024 x {d} x {d} entries ({d} threads)\n", .{ nt, K, threads });
    try abOracle(&oracle, 1024, threads);
    for (0..1024) |row| {
        if (oracle.row_max[row] > ab.oracle_max_abs_m1024) ab.oracle_max_abs_m1024 = oracle.row_max[row];
        ab.viol_m1024 += oracle.row_viol[row];
        if (oracle.row_viol[row] > (nt * 95) / 100) { // the zero-row signature: nearly everything wrong
            var allzero: usize = 0;
            for (0..@min(nt, 256)) |n| {
                if (out[row * nt + n] == 0) allzero += 1;
            }
            if (allzero == @min(nt, 256)) ab.zero_rows_m1024 += 1;
        }
    }
    std.debug.print("  AB m=1024: oracle |delta|max {d:.9} over all {d} entries, band {d} violations {d}, zeroed rows {d}\n", .{
        ab.oracle_max_abs_m1024, 1024 * nt, ab.band, ab.viol_m1024, ab.zero_rows_m1024 });

    // the m=128 block (the historically green launch, checked the same way)
    const out128 = try alloc.alloc(f32, 128 * nt);
    defer alloc.free(out128);
    try head.fillC(128 * nt * 4, 0xdead_beef);
    try head.logitsBlock(std.mem.sliceAsBytes(A[0 .. 128 * K]), 128, out128, 0);
    {
        var o128 = AbOracle{ .A = A, .slab = slab, .out = out128, .K = K, .nt = nt, .band = ab.band,
            .row_max = oracle.row_max[0..128], .row_viol = oracle.row_viol[0..128] };
        try abOracle(&o128, 128, threads);
        for (0..128) |row| {
            if (o128.row_max[row] > ab.oracle_max_abs_m128) ab.oracle_max_abs_m128 = o128.row_max[row];
            ab.viol_m128 += o128.row_viol[row];
        }
    }
    std.debug.print("  AB m=128:  oracle |delta|max {d:.9} over all {d} entries, band violations {d}\n", .{
        ab.oracle_max_abs_m128, 128 * nt, ab.viol_m128 });

    // -- 2) register dumps: the exact prompt16 IO calls at these very buffers, with the
    //    block reads' raw bits stored through plain per-lane writes (kk samples 0, 1,
    //    mid, last). A is still the random pattern the m=128 launch uploaded.
    const nk: usize = 4;
    const kks = [_]i32{ 0, 1, @intCast(K / 320), @intCast(K / 16 - 1) };
    const kkk = try d.alloc_(@sizeOf([nk]i32));
    defer d.g.free(kkk);
    try head.upload(kkk, std.mem.sliceAsBytes(&kks));
    const da_n = 8 * 16 * 4 * nk * 128; // [tileRow][sg][i][sample][lane*8]
    const db_n = (nt / 128) * 16 * 2 * nk * 128; // [tileCol][sg][j][sample][lane*8]
    const da = try d.alloc_(da_n * 2);
    defer d.g.free(da);
    const db = try d.alloc_(db_n * 4);
    defer d.g.free(db);
    const dumpk = try d.kernel(try d.g.module(&kl_kernels.eaq_kl), "eaq_dump_areg", 256);
    const st16k = try d.kernel(try d.g.module(&kl_kernels.eaq_kl), "eaq_ab_store16", 256);
    try head.launchProbe(dumpk, 8, @intCast(nt / 128), .{
        head.a, head.slab, head.c, da, db,
        @as(i32, 1024), @as(i32, @intCast(nt)), @as(i32, @intCast(K)), @as(i32, @intCast(nk)), kkk, kkk,
    });
    const DA = try alloc.alloc(u16, da_n);
    defer alloc.free(DA);
    const DB = try alloc.alloc(u32, db_n);
    defer alloc.free(DB);
    try head.download(std.mem.sliceAsBytes(DA), da, da_n * 2);
    try head.download(std.mem.sliceAsBytes(DB), db, db_n * 4);
    // multiset check per (tileRow, sg, i, sample): the 128 dumped bits are the 8x16 block
    // of A at the read's exact origin — lane-layout agnostic, so the dump proves what the
    // hardware delivered regardless of how the matrix maps to lanes
    for (0..8) |tr| {
        for (0..16) |sg| {
            for (0..4) |i| {
        const m0: usize = tr * 128 + (sg % 4) * 32 + 8 * i;
        for (0..nk) |x| {
            const base = ((tr * 16 + sg) * 4 + i) * nk + x;
            const got = DA[base * 128 ..][0 .. 128];
            var want: [128]u16 = undefined;
            for (0..8) |rr| {
                for (0..16) |cc| {
                    want[rr * 16 + cc] = A[(m0 + rr) * K + @as(usize, @intCast(kks[@intCast(x)])) * 16 + cc];
                }
            }
            abSort(u16, got);
            abSort(u16, &want);
            ab.a_dump_blocks += 1;
            if (!std.mem.eql(u16, got[0..128], &want)) ab.a_dump_bad += 1;
        }
            }
        }
    }
    // the transform read unpacked to halves on both sides: the multiset of the 16x16
    // half2 block of the slab at the origin, packing direction immaterial
    for (0..nt / 128) |tc| {
        for (0..16) |sg| {
            for (0..2) |j| {
        const n0: usize = tc * 128 + (sg / 4) * 32 + 16 * j;
        for (0..nk) |x| {
            const base = ((tc * 16 + sg) * 2 + j) * nk + x;
            const got = DB[base * 128 ..][0 .. 128];
            var got16: [256]u16 = undefined;
            for (0..128) |u| {
                got16[2 * u] = @truncate(got[u]);
                got16[2 * u + 1] = @truncate(got[u] >> 16);
            }
            var want: [256]u16 = undefined;
            for (0..16) |rr| {
                for (0..16) |cc| {
                    want[rr * 16 + cc] = slab[(@as(usize, @intCast(kks[@intCast(x)])) * 16 + rr) * nt + n0 + cc];
                }
            }
            abSort(u16, &got16);
            abSort(u16, &want);
            ab.b_dump_blocks += 1;
            if (!std.mem.eql(u16, &got16, &want)) ab.b_dump_bad += 1;
        }
            }
        }
    }
    ab.a_read_ok = ab.a_dump_bad == 0;
    ab.b_read_ok = ab.b_dump_bad == 0;
    std.debug.print("  AB dumps: A registers {d}/{d} blocks exact, B registers {d}/{d}\n", .{
        ab.a_dump_blocks - ab.a_dump_bad, ab.a_dump_blocks, ab.b_dump_blocks - ab.b_dump_bad, ab.b_dump_blocks });

    // -- 3) the 16b-store twin: exl3_prompt16 verbatim at the same buffers, same A
    const c16 = try d.alloc_(1024 * nt * 2);
    defer d.g.free(c16);
    try head.launchProbe(st16k, 8, @intCast(nt / 128), .{
        head.a, head.slab, c16, @as(i32, 1024), @as(i32, @intCast(nt)), @as(i32, @intCast(K)),
    });
    const H16 = try alloc.alloc(u16, 1024 * nt);
    defer alloc.free(H16);
    try head.download(std.mem.sliceAsBytes(H16), c16, 1024 * nt * 2);
    {
        const cols = try abSampleCols(alloc, nt);
        defer alloc.free(cols);
        var zr: usize = 0;
        for (0..1024) |row| {
            var allzero: usize = 0;
            for (cols) |n| {
                var acc: f64 = 0;
                for (0..K) |kk| acc += @as(f64, @floatCast(@as(f16, @bitCast(A[row * K + kk])))) *
                    @as(f64, @floatCast(@as(f16, @bitCast(slab[kk * nt + n]))));
                const want16: u16 = @bitCast(@as(f16, @floatCast(acc)));
                const gotv: f64 = @floatCast(@as(f16, @bitCast(H16[row * nt + n])));
                const dabs = @abs(gotv - @as(f64, @floatCast(@as(f16, @bitCast(want16)))));
                if (dabs > ab.store16_max_abs) ab.store16_max_abs = dabs;
                if (H16[row * nt + n] == 0 and want16 != 0) allzero += 1;
            }
            if (allzero == cols.len) zr += 1;
        }
        ab.store16_zero_rows = zr;
        std.debug.print("  AB store16 (prompt16 twin): |delta vs fp16(oracle)|max {d:.9}, zeroed rows {d}\n", .{
            ab.store16_max_abs, ab.store16_zero_rows });
    }

    // -- 3b) the raw single launch: eaq_gemm32 itself, x=8 groups at once, into a
    //    sentinel-filled C. The shipped stream chunks m down to 128-row launches; the
    //    acceptance for the kernel is this launch too: the same random A, the same
    //    slab, the fp64 oracle on sampled columns, and the sentinel tell for stores
    //    the hardware never landed (dead_beef), zeroed, or wrong.
    {
        try head.fillC(1024 * nt * 4, 0xdead_beef);
        try head.launchProbe(head.gemm, 8, @intCast(nt / 128), .{
            head.a, head.slab, head.c, @as(i32, 1024), @as(i32, @intCast(nt)), @as(i32, @intCast(K)),
        });
        const RC = try alloc.alloc(f32, 1024 * nt);
        defer alloc.free(RC);
        try head.download(std.mem.sliceAsBytes(RC), head.c, 1024 * nt * 4);
        const cols = try abSampleCols(alloc, nt);
        defer alloc.free(cols);
        var bad_groups = [8]bool{ false, false, false, false, false, false, false, false };
        for (0..1024) |row| {
            var sent: usize = 0;
            var zer: usize = 0;
            for (cols) |n| {
                const gotu = @as(u32, @bitCast(RC[row * nt + n]));
                if (gotu == 0xdead_beef) {
                    sent += 1;
                    continue;
                }
                if (gotu == 0) zer += 1;
                var acc: f64 = 0;
                for (0..K) |kk| acc += @as(f64, @floatCast(@as(f16, @bitCast(A[row * K + kk])))) *
                    @as(f64, @floatCast(@as(f16, @bitCast(slab[kk * nt + n]))));
                if (@abs(@as(f64, RC[row * nt + n]) - acc) > ab.band) ab.raw1024_viol += 1;
            }
            if (sent == cols.len) {
                ab.raw1024_dropped_rows += 1;
                bad_groups[row / 128] = true;
            } else if (zer == cols.len) {
                var want_nonzero = false;
                for (cols) |n| {
                    var acc: f64 = 0;
                    for (0..K) |kk| acc += @as(f64, @floatCast(@as(f16, @bitCast(A[row * K + kk])))) *
                        @as(f64, @floatCast(@as(f16, @bitCast(slab[kk * nt + n]))));
                    if (acc != 0) {
                        want_nonzero = true;
                        break;
                    }
                }
                if (want_nonzero) {
                    ab.raw1024_zero_rows += 1;
                    bad_groups[row / 128] = true;
                }
            } else if (sent > 0 or zer > 0) {
                bad_groups[row / 128] = true;
            }
        }
        for (bad_groups) |b| {
            if (b) ab.raw1024_bad_groups += 1;
        }
        std.debug.print("  AB raw x=8 single launch: dropped rows {d}, zeroed rows {d}, band violations {d}, bad row-groups {d}/8\n", .{
            ab.raw1024_dropped_rows, ab.raw1024_zero_rows, ab.raw1024_viol, ab.raw1024_bad_groups });
    }

    // -- 4) the k-select pattern: row r's logits are the slab rows 16*(r%320)..+16 summed
    const A2 = try alloc.alloc(u16, 1024 * K);
    defer alloc.free(A2);
    for (0..1024) |row| {
        for (0..K) |k| {
            A2[row * K + k] = if (k / 16 == row % (K / 16)) @as(u16, 0x3c00) else 0;
        }
    }
    try head.fillC(1024 * nt * 4, 0xdead_beef);
    try head.logitsBlock(std.mem.sliceAsBytes(A2), 1024, out, 0);
    {
        const cols = try abSampleCols(alloc, nt);
        defer alloc.free(cols);
        for (0..1024) |row| {
            var okn: usize = 0;
            var zer: usize = 0;
            for (cols) |n| {
                const gotv: f64 = out[row * nt + n];
                if (gotv == 0) zer += 1;
                var want: f64 = 0;
                for (0..16) |i| want += @as(f64, @floatCast(@as(f16, @bitCast(slab[((row % (K / 16)) * 16 + i) * nt + n]))));
                if (@abs(gotv - want) <= 0.02) okn += 1;
            }
            if (okn == cols.len) continue;
            if (zer == cols.len) {
                ab.ks_zero_rows += 1;
                continue;
            }
            // not zeros: does this row's values match some other k-block's slab rows?
            var bestb: usize = std.math.maxInt(usize);
            outer: for (0..K / 16) |b| {
                var hit: usize = 0;
                for (cols[0..@min(cols.len, 32)]) |n| {
                    var w: f64 = 0;
                    for (0..16) |i| w += @as(f64, @floatCast(@as(f16, @bitCast(slab[(b * 16 + i) * nt + n]))));
                    if (@abs(out[row * nt + n] - w) <= 0.02) {
                        hit += 1;
                    } else {
                        break;
                    }
                }
                if (hit >= 30) {
                    bestb = b;
                    break :outer;
                }
            }
            if (bestb != std.math.maxInt(usize)) {
                ab.ks_mix_rows += 1;
                if (ab.ks_first_mix.len == 0)
                    ab.ks_first_mix = try std.fmt.allocPrint(alloc, "row {d} saw slab block {d}", .{ row, bestb });
            }
        }
        std.debug.print("  AB k-select: zero-read rows {d}, row-mix rows {d} ({s})\n", .{
            ab.ks_zero_rows, ab.ks_mix_rows, if (ab.ks_first_mix.len > 0) ab.ks_first_mix else "none" });
    }

    // A's product-side residency is proven by the trio that READS A THE WAY THE
    // GEMM DOES: the fp64 oracle over every entry of both C blocks (chunked
    // launches, viol 0), eaq_store16 (same 2d block reads, per-lane stores, zero
    // zeroed rows, exact against fp16(oracle)), and the k-select (zero-read and
    // row-mix rows 0). The A-dump kernel is a diagnostic on its own: it is a
    // FOURTH reader, outside the shipping path, whose block read self-mismatches
    // ~19% of dumped blocks (1664/2048) while every product metric is exact —
    // the dump's register-level read is the flaky party, not A. It reports
    // (the raw DA bytes go to the spot dump for forensics) but it cannot fail
    // the product: the gate is the product's own evidence.
    ab.ok = ab.viol_m1024 == 0 and ab.viol_m128 == 0 and ab.b_read_ok and
        ab.ks_zero_rows == 0 and ab.ks_mix_rows == 0 and ab.store16_zero_rows == 0 and
        ab.raw1024_dropped_rows == 0 and ab.raw1024_zero_rows == 0 and ab.raw1024_viol == 0;

    if (std.c.getenv("XPU_KL_SPOT_DUMP")) |cs| { // the raw forensic bytes, not just the summary
        const path = std.mem.span(cs);
        const f = openFd(path, O_WRONLY_CREAT_TRUNC) catch {
            std.debug.print("  AB: cannot write {s}\n", .{path});
            return ab;
        };
        defer _ = close(f);
        const parts = [_][]const u8{
            std.mem.sliceAsBytes(DA), std.mem.sliceAsBytes(DB),
            std.mem.sliceAsBytes(out), std.mem.sliceAsBytes(out128), std.mem.sliceAsBytes(H16),
        };
        var base: i64 = 0;
        for (parts) |p| {
            var offb: usize = 0;
            while (offb < p.len) {
                const w = pwrite(f, p.ptr + offb, p.len - offb, base + @as(i64, @intCast(offb)));
                if (w <= 0) break;
                offb += @intCast(w);
            }
            base += @intCast(p.len);
        }
        std.debug.print("  AB: raw dumps (DA, DB, C m=1024, C m=128, C16) written to {s}\n", .{path});
    }
    return ab;
}


/// The eval rows through the serving prompt path, hidden states into a memfd: every row
/// resets (GDN/conv states, position), `prompt` chunks it, and each chunk's final normed
/// rows append in order (§5.1, forward.zig's XPU_DUMP_H block). The last sequence keeps
/// its runner logits for the spot check. Returns the fd (position-read afterwards by the
/// block loop) and the byte count, which must be positions·D·2.
pub fn captureHidden(r: *q.Runner, rows: []const []const u32, last_row: usize) !struct { fd: c_int, bytes: u64 } {
    const fd = memfd_create("xpu_kl_h", 0);
    if (fd < 0) return error.OutOfMemory;
    q.dump_h = fd;
    defer q.dump_h = null;
    for (rows, 0..) |row, i| {
        try r.reset();
        try r.prompt(row, i == last_row); // the last row only: the runner's own fp16 head pass, for the spot check
    }
    const bytes: u64 = @intCast(lseek(fd, 0, 2));
    _ = lseek(fd, 0, 0);
    return .{ .fd = fd, .bytes = bytes };
}

pub const SpotCheck = struct { // §5.1: the runner's last-row logits against the fp32 head GEMM of that row
    position: usize,
    agree: bool,
    max_abs_delta: f64,
    runner_top1: u32,
    head_top1: u32,
};

pub const Times = struct { prompts_s: f64, dequant_s: f64, gemm_s: f64, stats_s: f64, total_s: f64 };

pub const Domain = struct { name: []const u8, seq_rows: usize, s: stats.Summary };
pub const SelfCheck = struct { pass: bool, rows_checked: usize, max_abs_kl: f64, top1_frac: f64 };

/// The per-row spread (kl-order-ruling.md invariant 4): every row's mean over its
/// positions, the kind separating the headline, and the 3x-the-headline flag on the
/// natural rows.
pub const RowAgg = struct {
    row: usize,
    domain: []const u8,
    kind: []const u8, // "natural" (headline rows) or "random" (§5.2, separate)
    mean: f64,
    median: f64, // the row's own median: the shape invariant (mean at or above median) per row
    flagged: bool, // natural row above 3x the headline mean
};

/// The alignment canary (invariant 5): teacher-forced next-token accuracy against the
/// prompt tokens — argmax(logits at position t) equals token t+1, every position but
/// a row's last. A 27B at 2048 context sits in the 50-70 % band on the natural rows
/// and near 1/V on the random rows; a reference outside the band means the reference
/// file is misaligned, and a candidate far below the reference's own accuracy means
/// the candidate path is.
pub const CanaryBand = struct { n: usize, ref_acc: f64, cand_acc: f64 };
pub const Canary = struct { natural: CanaryBand, random: CanaryBand, inv_vocab: f64 };

pub const Report = struct {
    tool: []const u8 = "xpu-eaq-kl",
    build_id: []const u8 = buildID, // invariant 6: the statistic's own fingerprint
    direction: []const u8 = "KL(ref||cand)", // forward KL, the reference weights the sum
    mode: []const u8 = "pack", // "pack" (runner + head) or "logits" (diagnostic candidate)
    pack: []const u8,
    cand: []const u8 = "", // the --cand-logits path in the diagnostic mode
    cand_sha256: ?[]const u8 = null,
    layout: []const u8 = "unknown", // eaq-manifest[:sha] | stock-exl3 | logits-file | none
    prompts: []const u8,
    prompts_sha256: []const u8,
    ref: []const u8, // the ref path, or "self"
    ref_sha256: ?[]const u8 = null,
    dump_sha256: ?[]const u8 = null, // set in --dump-ref mode: the produced file's hash
    seq_rows: usize,
    cols: usize,
    positions: usize,
    vocab: usize,
    block: usize,
    threads: usize,
    head: ?HeadInfo = null, // null in the diagnostic mode (no head was built)
    seconds: Times,
    headline: stats.Summary,
    random: ?stats.Summary = null,
    per_domain: []const Domain,
    per_row: []const RowAgg = &.{}, // invariant 4
    canary: ?Canary = null, // invariant 5
    hist_edges: []const f64 = &stats.hist_edges,
    histogram: ?[]const u64 = null, // per-position KL bins, headline (natural) rows
    histogram_random: ?[]const u64 = null,
    spot_check: ?SpotCheck = null,
    self_check: ?SelfCheck = null,
};

/// sha256 of a file, streamed in 64 MiB chunks — the prompts content hash and the
/// reference's provenance (the gate's "same reference logits for every candidate" is
/// what this hash has to prove). Lowercase hex, allocator-owned.
pub fn sha256Hex(path: []const u8) ![]const u8 {
    const f = openFd(path, 0) catch |e| {
        std.debug.print("xpu-eaq-kl: cannot open {s}: {s}\n", .{ path, @errorName(e) });
        return e;
    };
    defer _ = close(f);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    const buf = try alloc.alloc(u8, 64 << 20);
    defer alloc.free(buf);
    var off: u64 = 0;
    while (true) {
        const n = pread(f, buf.ptr, buf.len, @as(i64, @intCast(off)));
        if (n < 0) return error.FileBusy;
        if (n == 0) break;
        h.update(buf[0..@intCast(n)]);
        off += @intCast(n);
    }
    const out = try alloc.alloc(u8, 64);
    _ = std.fmt.bufPrint(out, "{x}", .{h.finalResult()}) catch unreachable;
    return out;
}

/// Whole file into memory — for the small sidecars (prompts, manifest), not the 32 GB ref.
fn readFileAll(path: []const u8, limit: usize) ![]u8 {
    const f = openFd(path, 0) catch |e| {
        std.debug.print("xpu-eaq-kl: cannot open {s}: {s}\n", .{ path, @errorName(e) });
        return e;
    };
    defer _ = close(f);
    const size: usize = @intCast(lseek(f, 0, 2));
    if (size == 0 or size > limit) return Error.FileIo;
    const buf = try alloc.alloc(u8, size);
    errdefer alloc.free(buf);
    var got: usize = 0;
    while (got < size) {
        const r = pread(f, buf.ptr + got, size - got, @as(i64, @intCast(got)));
        if (r <= 0) return Error.FileIo;
        got += @intCast(r);
    }
    return buf;
}

const ManifestRow = struct {
    row: usize,
    kind: ?[]const u8 = null, // "random": reported separately, out of the headline (§5.2)
    domain: ?[]const u8 = null,
};
const Manifest = struct {
    cols: ?usize = null,
    content_sha256: ?[]const u8 = null, // sha256 of the raw little-endian i32 prompts file
    rows: ?[]const ManifestRow = null,
};

pub const Options = struct {
    pack: []const u8,
    prompts: []const u8,
    ref: ?[]const u8 = null, // raw fp32 [positions][VOCAB], prompts row order (the xpu-eaq-ref contract)
    manifest: ?[]const u8 = null, // the prompts manifest: per-row domain and content_sha256
    report: ?[]const u8 = null, // kl.json
    dump_logits: ?[]const u8 = null, // write the candidate logits in the reference's own format
    dump_ref: ?[]const u8 = null, // produce the reference file itself: this pack, this path, raw fp32 [positions][VOCAB]
    cand_logits: ?[]const u8 = null, // diagnostic candidate (kl-order-ruling.md): fp32 logits file in the reference's own format — no runner, no head
    dump_kl: ?[]const u8 = null, // write the raw per-position KL rows (KlDumpHeader + KlDumpRow records) for --pair
    dump_hist: ?[]const u8 = null, // write the per-position KL histogram (text) for the stock 2.0 run
    cols: usize = 2048,
    block: usize = 1024, // §5.3: 1024-row streaming blocks (~1 GB of fp32 logits a side)
    threads: usize = 0, // 0: one per core
    context: u32 = 4096, // the runner's KV rows; a 2048-token row chunks under it
    expect_sha256: ?[]const u8 = null, // refuse a prompts file that is not this content
    self_mode: bool = false, // candidate against itself: KL exactly 0, top-1 100%
};

/// One block's rows, handed to the threads as contiguous ranges; each row is independent
/// and every aggregate reduces afterwards in index order, so the thread schedule cannot
/// move a number.
const BlockCtx = struct {
    cand: []const f32, // [m][nt]
    ref: []const f32, // [m][VOCAB]; the cand buffer itself in --self mode
    self_ref: bool,
    nt: usize,
    vocab: usize,
    out: []stats.Row,
};

const RowJob = struct { ctx: *BlockCtx, at: usize, to: usize };

fn rowJob(job: *RowJob) void {
    for (job.at..job.to) |i| {
        // i indexes the block (and the block-sliced out): the jobs partition [0..m), so
        // every row lands in its own slot and the thread schedule cannot move a number
        const cand = job.ctx.cand[i * job.ctx.nt ..][0 .. job.ctx.vocab];
        const ref = if (job.ctx.self_ref) cand else job.ctx.ref[i * job.ctx.vocab ..][0 .. job.ctx.vocab];
        job.ctx.out[i] = stats.rowKl(ref, cand);
    }
}

fn parallelRows(ctx: *BlockCtx, m: usize, threads: usize) !void {
    const workers = @min(threads, @max(1, @divTrunc(m, 8))); // never schedule below eight rows a thread
    const per = (m + workers - 1) / workers;
    var jobs: [256]RowJob = undefined;
    var threads_list: [256]?std.Thread = @splat(null);
    var spawned: usize = 0;
    var at: usize = 0;
    while (at < m) : (spawned += 1) {
        const to = @min(m, at + per);
        jobs[spawned] = .{ .ctx = ctx, .at = at, .to = to };
        threads_list[spawned] = std.Thread.spawn(.{}, rowJob, .{&jobs[spawned]}) catch null;
        at = to;
    }
    for (threads_list[0..spawned]) |t| if (t) |th| th.join();
    // a spawn refusal ran nothing: do those ranges inline (same math, same order of results)
    for (threads_list[0..spawned], 0..) |t, i| if (t == null) rowJob(&jobs[i]);
}

fn argmaxMax(z: []const f32) struct { at: u32, max: f32 } {
    var at: u32 = 0;
    var mx: f32 = -std.math.inf(f32);
    for (z, 0..) |v, i| if (v > mx) {
        mx = v;
        at = @intCast(i);
    };
    return .{ .at = at, .max = mx };
}

/// The whole gate run. Prints a progress line per block plus the invariant lines,
/// writes --report (kl.json), --dump-logits, --dump-kl and --dump-hist when asked,
/// and returns the report. In --self mode and in the ref-vs-ref identity run the
/// caller checks report.self_check.pass (the tool exits nonzero when it fails).
/// `--cand-logits` is the diagnostic candidate path of kl-order-ruling.md decisive
/// check 1: the candidate logits come from a file in the reference's own format, the
/// runner and the head are never built, and the identity run needs no card.
pub fn run(req: Options) !Report {
    refuseStudyModes();
    var o = req;
    if (req.dump_ref) |dr| { // ref production (the xpu-eaq-ref contract as a mode): the same serving path, logits streamed to disk
        if (req.ref != null or req.self_mode) return Error.MissingMode; // --dump-ref stands alone
        o.self_mode = true; // the self check rides along: the dump must be GEMM-deterministic too
        o.dump_logits = dr;
    }
    if ((o.ref != null) == o.self_mode) return Error.MissingMode; // exactly one of --ref / --self
    const logits_mode = o.cand_logits != null;
    if (logits_mode) {
        // the identity path: the reference as its own candidate with the runner and the
        // head out of the picture — no pack, no device, and the zeros become provable
        if (o.ref == null or o.self_mode or req.dump_ref != null) return Error.MissingMode;
        if (o.pack.len != 0) return Error.MissingMode; // the pack side is the thing under diagnosis
    }
    comptime { // the i32 prompts and f32 logits files are little-endian; refuse a big-endian build
        const probe: u16 = 0x0102;
        const bytes: [2]u8 = @bitCast(probe);
        std.debug.assert(bytes[0] == 0x02);
    }
    if (o.block == 0 or o.block % 128 != 0) return Error.BadHeadShape;

    const prompts_sha = try sha256Hex(o.prompts);
    if (o.expect_sha256) |e| if (!std.ascii.eqlIgnoreCase(e, prompts_sha)) {
        std.debug.print("xpu-eaq-kl: {s} hashes to {s}, not the expected {s}\n", .{ o.prompts, prompts_sha, e });
        return Error.PromptsHash;
    };

    const raw = try readFileAll(o.prompts, 1 << 28);
    defer alloc.free(raw);
    if (raw.len == 0 or raw.len % (4 * o.cols) != 0) return Error.BadPrompts;
    const ids = std.mem.bytesAsSlice(i32, raw);
    for (ids) |t| if (t < 0 or @as(u32, @bitCast(t)) >= q.VOCAB) return Error.BadPrompts;
    const seq_rows: usize = ids.len / o.cols;
    const positions = seq_rows * o.cols;

    // per-row provenance: the manifest when there is one, else the last four rows are the random ones (§5.2)
    var domains = try alloc.alloc([]const u8, seq_rows);
    var random_row = try alloc.alloc(bool, seq_rows);
    if (o.manifest) |mpath| {
        const mraw = try readFileAll(mpath, 1 << 26);
        defer alloc.free(mraw);
        const m = try std.json.parseFromSlice(Manifest, alloc, mraw, .{ .ignore_unknown_fields = true });
        defer m.deinit();
        if (m.value.content_sha256) |want| if (!std.ascii.eqlIgnoreCase(want, prompts_sha)) {
            std.debug.print("xpu-eaq-kl: {s} says {s} for the prompts hash, computed {s}\n", .{ mpath, want, prompts_sha });
            return Error.ManifestHash;
        };
        if (m.value.cols) |mc| if (mc != o.cols) return Error.ColsMismatch;
        const mrows = m.value.rows orelse return Error.ManifestRows;
        if (mrows.len != seq_rows) return Error.ManifestRows;
        for (mrows, 0..) |mr, i| {
            if (mr.row != i) return Error.ManifestRows;
            const rnd = mr.kind != null and std.mem.eql(u8, mr.kind.?, "random");
            random_row[i] = rnd;
            // the parser and mraw die at the block's end; the report prints the domains
            // long after, so they live in their own copy
            domains[i] = try alloc.dupe(u8, mr.domain orelse (if (rnd) "random" else "heldout"));
        }
    } else {
        for (0..seq_rows) |i| {
            random_row[i] = i >= seq_rows - @min(seq_rows, 4);
            domains[i] = if (random_row[i]) "random" else "heldout";
        }
    }
    // candidate degeneracy guard — the ruling's gate-before-any-KL-row bands on the
    // candidate straight off the device: a frozen hidden state passes --self at KL 0
    // against itself, so the self-check means nothing without these enforced. The
    // per-position argmaxes and max logits come from the block loop's cbuf for free.
    var cand_amap = std.AutoHashMap(u32, u32).init(alloc); // argmax -> count
    defer cand_amap.deinit();
    var cand_maxes = try alloc.alloc(f32, positions); // the max logit of every position
    defer alloc.free(cand_maxes);

    const threads = @min(if (o.threads != 0) o.threads else std.Thread.getCpuCount() catch 1, @max(1, @divTrunc(positions, 8)));
    std.debug.print("xpu-eaq-kl: pack {s}\n  prompts {s} sha256 {s}\n  {d} rows x {d} = {d} positions, ref {s}, block {d}, {d} threads\n", .{
        o.pack, o.prompts, prompts_sha, seq_rows, o.cols, positions, o.ref orelse "self", o.block, threads });

    const t_total = q.now();
    var rp: ?*q.Runner = null;
    var head_mem: Head = undefined;
    var hd: ?*Head = null;
    var head_nt: usize = q.VOCAB; // the head's slab pitch, bound at init so no buffer width can read hd before it exists
    var armed = false;
    // (no errdefer twins of the defers below: an error return that fires them must not
    // deinit the runner or head a second time — Dev.deinit twice calls settle on the
    // closed Gpu, whose api pointers are gone; the defers register the instant the
    // resources exist, which retires the errdefer pattern for rp, hd and armed.)
    var cap_fd: c_int = -1;
    var runner_logits: []const f32 = &.{};
    var prompts_s: f64 = 0;
    var dequant_s: f64 = 0;
    var hbuf: []u16 = &.{};
    if (!logits_mode) {
        // 1) the serving prompt path with the hidden dump: every row reset + prompted;
        //    the last row's chunk keeps the runner's own logits for the spot check
        const r = try q.Runner.init(o.pack, o.context);
        rp = r;
        armed = true;
        armDrain(&r.d);
        defer if (rp) |rr| rr.deinit(); // registered the moment the runner exists
        defer if (armed) disarmDrain(); // disarms ahead of the runner's settle, as the old order did
        const rows = try alloc.alloc([]const u32, seq_rows);
        for (0..seq_rows) |i| rows[i] = @alignCast(std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(ids[i * o.cols ..][0 .. o.cols])));
        const t0 = q.now();
        const cap = try captureHidden(r, rows, seq_rows - 1);
        cap_fd = cap.fd;
        prompts_s = q.now() - t0;
        if (cap.bytes != @as(u64, positions) * q.D * 2) return Error.HiddenShort;
        runner_logits = try alloc.dupe(f32, try r.logits()); // the fp16 head GEMV's kept row, read before anything else touches the device
        std.debug.print("  prompts: {d:.1} s, hidden {d} B\n", .{ prompts_s, cap.bytes });
        hbuf = try alloc.alloc(u16, o.block * q.D);
    }

    defer if (hd) |h| h.deinit();

    defer if (hbuf.len != 0) alloc.free(hbuf);
    defer if (cap_fd >= 0) { _ = close(cap_fd); };

    // the streaming buffers: GB-scale preads want raw private mappings, MAP_POPULATE,
    // self-touched up front (GB-scale malloc regions fault EFAULT inside the runner on
    // this box while the identical read works outside it — Level Zero co-tenant). The
    // GEMM's cbuf keeps its alloc.alloc: it is filled by memcpy from the gate's
    // staging, never by a file pread; the candidate's file buffer is a mapping too.
    var rbuf: []f32 = &.{};
    var rmap: ?*anyopaque = null;
    var rmap_len: usize = 0;
    if (!o.self_mode) {
        const m = try mapF32(o.block * q.VOCAB);
        rmap = m.p;
        rmap_len = o.block * q.VOCAB * @sizeOf(f32);
        rbuf = m.buf;
    }
    defer if (rmap) |p| { _ = munmap(p, rmap_len); };
    var cbuf: []f32 = &.{};
    var cmap: ?*anyopaque = null;
    var cmap_len: usize = 0;
    // 2) the head slab: dequantized once with the serving kernel (suh/svh baked exactly
    //    as forward_exl3.gemmRows bakes them), the block GEMM reuses it for every block.
    //    Before the streaming buffers: cbuf's width is the head's own nt (the null deref
    //    this ordering fixes: cbuf was sized here from hd, which step 2 only set below).
    if (!logits_mode) {
        const t1 = q.now();
        head_mem = try Head.init(rp.?, @intCast(o.block));
        hd = &head_mem;
        head_nt = head_mem.nt;
        dequant_s = q.now() - t1;
    }
    if (logits_mode) {
        const m = try mapF32(o.block * q.VOCAB);
        cmap = m.p;
        cmap_len = o.block * q.VOCAB * @sizeOf(f32);
        cbuf = m.buf;
    } else {
        cbuf = try alloc.alloc(f32, o.block * head_nt);
    }
    defer if (cmap) |p| { _ = munmap(p, cmap_len); };
    defer if (!logits_mode) alloc.free(cbuf);

    // 3) the reference — and the diagnostic candidate — streamed per block from files;
    //    the candidate's own block is the reference in --self mode
    var ref_fd: c_int = -1;
    if (o.ref) |rp0| {
        ref_fd = try openFd(rp0, 0);
        const size: u64 = @intCast(lseek(ref_fd, 0, 2));
        if (size < @as(u64, positions) * q.VOCAB * 4) return Error.RefTooSmall;
        std.debug.print("  ref {s} ({d} B)\n", .{ rp0, size });
    }
    defer if (ref_fd >= 0) { _ = close(ref_fd); }; // NOT inside the if: Zig defers run at block end, and the capture loop outlives it
    var cand_fd: c_int = -1;
    if (o.cand_logits) |cl| {
        cand_fd = try openFd(cl, 0);
        const size: u64 = @intCast(lseek(cand_fd, 0, 2));
        if (size < @as(u64, positions) * q.VOCAB * 4) return Error.CandTooSmall;
        std.debug.print("  cand {s} ({d} B)\n", .{ cl, size });
    }
    defer if (cand_fd >= 0) { _ = close(cand_fd); };

    const rows_stats = try alloc.alloc(stats.Row, positions);
    defer alloc.free(rows_stats);
    var dump_fd: c_int = -1;
    if (o.dump_logits) |dp| {
        const f = openFd(dp, O_WRONLY_CREAT_TRUNC) catch |e| {
            std.debug.print("xpu-eaq-kl: cannot write {s}: {s}\n", .{ dp, @errorName(e) });
            return e;
        };
        dump_fd = f;
    }
    defer if (dump_fd >= 0) { _ = close(dump_fd); };
    var dump_off: u64 = 0;

    // 4) the streaming blocks (§5.3): fp32 logits out of the GEMM (or out of the
    //    candidate file in the diagnostic mode), fp64 softmax/KL over the vocab rows,
    //    threaded; the last block carries the §5.1 spot check (it needs the runner)
    var stats_s: f64 = 0;
    var spot: ?SpotCheck = null;
    const nb = (positions + o.block - 1) / o.block;
    for (0..nb) |bi| {
        const base = bi * o.block;
        const m = @min(o.block, positions - base);
        if (!logits_mode) {
            const hb = std.mem.sliceAsBytes(hbuf[0 .. m * q.D]);
            preadAll(cap_fd, hb, @as(i64, @intCast(base * q.D * 2))) catch return Error.HiddenShort;
        }
        if (!o.self_mode) {
            const rb = std.mem.sliceAsBytes(rbuf[0 .. m * q.VOCAB]);
            const ref_off: i64 = @intCast(base * @as(u64, q.VOCAB) * 4);
            preadAll(ref_fd, rb, ref_off) catch |e| {
                std.debug.print("  ref block bi={d} off={d}: {s}\n", .{ bi, ref_off, @errorName(e) });
                return Error.RefTooSmall;
            };
        }
        if (logits_mode) {
            const cb = std.mem.sliceAsBytes(cbuf[0 .. m * q.VOCAB]);
            const cand_off: i64 = @intCast(base * @as(u64, q.VOCAB) * 4);
            preadAll(cand_fd, cb, cand_off) catch |e| {
                std.debug.print("  cand block bi={d} off={d}: {s}\n", .{ bi, cand_off, @errorName(e) });
                return Error.CandShort;
            };
        } else {
            try hd.?.logitsBlock(std.mem.sliceAsBytes(hbuf[0 .. m * q.D]), @intCast(m), cbuf[0 .. m * head_nt], 0);
        }
        const t2 = q.now();
        var ctx = BlockCtx{
            .cand = cbuf, .ref = if (o.self_mode) cbuf else rbuf,
            .self_ref = o.self_mode,
            .nt = if (logits_mode) q.VOCAB else head_nt,
            .vocab = if (logits_mode) q.VOCAB else hd.?.vocab,
            .out = rows_stats[base .. base + m],
        };
        try parallelRows(&ctx, m, threads);
        stats_s += q.now() - t2;
        if (!logits_mode) for (0..m) |i| { // the guard's raw material, free off the stats pass
            const am = rows_stats[base + i].a_cand;
            const gop = try cand_amap.getOrPut(am);
            if (!gop.found_existing) gop.value_ptr.* = 1 else gop.value_ptr.* += 1;
            cand_maxes[base + i] = cbuf[i * ctx.nt + am];
        };
        if (dump_fd >= 0) for (0..m) |i| {
            const b = std.mem.sliceAsBytes(cbuf[i * ctx.nt ..][0 .. ctx.vocab]);
            if (pwrite(dump_fd, b.ptr, b.len, @as(i64, @intCast(dump_off))) != @as(isize, @intCast(b.len))) return Error.DumpShort;
            dump_off += b.len;
        };
        if (bi == nb - 1 and !logits_mode) { // the runner kept exactly this row: its fp16 GEMV logits vs our fp32 GEMM logits
            const cand_row = cbuf[(m - 1) * ctx.nt ..][0 .. ctx.vocab];
            const a1 = argmaxMax(cand_row);
            const a2 = argmaxMax(runner_logits);
            var dmax: f64 = 0;
            for (cand_row, runner_logits) |cv, rv| dmax = @max(dmax, @abs(@as(f64, cv) - @as(f64, rv)));
            spot = .{ .position = positions - 1, .agree = a1.at == a2.at, .max_abs_delta = dmax, .runner_top1 = a2.at, .head_top1 = a1.at };
            if (std.c.getenv("XPU_KL_SPOT_DUMP") != null) { // the two sides' raw rows, for the spot divergence post-mortem
                const path = std.c.getenv("XPU_KL_SPOT_DUMP").?;
                const f = openFd(std.mem.sliceTo(path, 0), O_WRONLY_CREAT_TRUNC) catch return Error.FileIo;
                defer _ = close(f);
                const rb = std.mem.sliceAsBytes(runner_logits);
                const cb2 = std.mem.sliceAsBytes(cand_row);
                if (pwrite(f, rb.ptr, rb.len, 0) != @as(isize, @intCast(rb.len))) return Error.DumpShort;
                if (pwrite(f, cb2.ptr, cb2.len, @as(i64, @intCast(rb.len))) != @as(isize, @intCast(cb2.len))) return Error.DumpShort;
                const nz_r = blk: { var c: usize = 0; for (runner_logits) |v| { if (v != 0) c += 1; } break :blk c; };
                const nz_h = blk: { var c: usize = 0; for (cand_row) |v| { if (v != 0) c += 1; } break :blk c; };
                std.debug.print("  spot dump: runner nonzero {d}/{d} (top5 {d:.3} {d:.3} {d:.3} {d:.3} {d:.3}), head nonzero {d}/{d} (top5 {d:.3} {d:.3} {d:.3} {d:.3} {d:.3})\n", .{
                    nz_r, runner_logits.len,
                    runner_logits[0], runner_logits[1], runner_logits[2], runner_logits[3], runner_logits[4],
                    nz_h, cand_row.len,
                    cand_row[0], cand_row[1], cand_row[2], cand_row[3], cand_row[4] });
            }
        }
        std.debug.print("  block {d}/{d}: gemm {d:.1} s, stats {d:.1} s\n", .{ bi + 1, nb, if (hd) |h| h.gemm_s else @as(f64, 0), stats_s });
    }

    if (!logits_mode) { // the degeneracy guard, fail closed before any aggregate or file
        var mode_n: u32 = 0;
        var it = cand_amap.valueIterator();
        while (it.next()) |v| mode_n = @max(mode_n, v.*);
        const distinct = cand_amap.count();
        const sorted = try alloc.dupe(f32, cand_maxes);
        defer alloc.free(sorted);
        std.mem.sort(f32, sorted, {}, std.sort.asc(f32));
        const med_max = sorted[@divFloor(sorted.len, 2)];
        const pf: f64 = @floatFromInt(positions);
        std.debug.print("  candidate degeneracy: distinct argmax {d}/{d} ({d:.3}), argmax mode share {d:.3}, median max-logit {d:.3}\n", .{
            distinct, positions, @as(f64, @floatFromInt(distinct)) / pf, @as(f64, @floatFromInt(mode_n)) / pf, med_max });
        const min_distinct: usize = @min(2000, @divFloor(positions, 8)); // the full-dump 2000 band, scaled for canaries
        if (@as(f64, @floatFromInt(mode_n)) > 0.10 * pf or distinct < min_distinct or med_max < 10.0) {
            std.debug.print("xpu-eaq-kl: the candidate is degenerate (bands: argmax mode <10%, distinct >= {d}, median max-logit >= 10) — failing closed\n", .{min_distinct});
            return Error.CandidateDegenerate;
        }
    }

    // provenance hashes (invariant 6). The reference is hashed exactly once per run and
    // shared by the report and the --dump-kl header; the candidate file, when it is a
    // file, confirms the identity run beyond the path string.
    const ref_sha = if (o.ref) |rp0| try sha256Hex(rp0) else null;
    var cand_sha: ?[]const u8 = null;
    if (o.cand_logits) |cl| cand_sha = try sha256Hex(cl);
    const identity = logits_mode and (std.mem.eql(u8, o.cand_logits.?, o.ref.?) or
        (ref_sha != null and cand_sha != null and std.mem.eql(u8, ref_sha.?, cand_sha.?)));
    if (logits_mode) std.debug.print("  cand sha256 {s}{s}\n", .{
        cand_sha orelse "?", if (identity) "  (IDENTITY RUN: cand == ref; the zeros are the requirement)" else "" });

    // 5) aggregates (§5.3): headline over the non-random positions, random separately, the same per domain
    var headline: std.ArrayList(stats.Row) = .empty;
    var rand_rows: std.ArrayList(stats.Row) = .empty;
    var dnames: std.ArrayList([]const u8) = .empty;
    var dbuckets: std.ArrayList(std.ArrayList(stats.Row)) = .empty;
    defer {
        headline.deinit(alloc);
        rand_rows.deinit(alloc);
        dnames.deinit(alloc);
        for (dbuckets.items) |*b| b.deinit(alloc);
        dbuckets.deinit(alloc);
    }
    for (0..seq_rows) |i| {
        const slice = rows_stats[i * o.cols ..][0 .. o.cols];
        if (random_row[i]) {
            try rand_rows.appendSlice(alloc, slice);
            continue;
        }
        try headline.appendSlice(alloc, slice);
        var di: ?usize = null;
        for (dnames.items, 0..) |dn, j| if (std.mem.eql(u8, dn, domains[i])) {
            di = j;
            break;
        };
        if (di) |j| {
            try dbuckets.items[j].appendSlice(alloc, slice);
        } else {
            var b: std.ArrayList(stats.Row) = .empty;
            try b.appendSlice(alloc, slice);
            try dnames.append(alloc, domains[i]);
            try dbuckets.append(alloc, b);
        }
    }
    const per_domain = try alloc.alloc(Domain, dnames.items.len);
    for (dnames.items, dbuckets.items, 0..) |dn, *b, i| per_domain[i] = .{
        .name = dn, .seq_rows = @divTrunc(b.items.len, o.cols), .s = try stats.Summary.init(b.items),
    };
    const headline_summary = try stats.Summary.init(headline.items);

    // invariant 4: the per-row spread — every row's mean and median, natural rows
    // above 3x the headline mean flagged
    var per_row = try alloc.alloc(RowAgg, seq_rows);
    var rows_ge_median: usize = 0;
    var n_flagged: usize = 0;
    for (0..seq_rows) |i| {
        const rs = try stats.Summary.init(rows_stats[i * o.cols ..][0 .. o.cols]);
        per_row[i] = .{
            .row = i, .domain = domains[i], .kind = if (random_row[i]) "random" else "natural",
            .mean = rs.mean, .median = rs.median,
            .flagged = !random_row[i] and rs.mean > 3 * headline_summary.mean,
        };
        rows_ge_median += @intFromBool(rs.mean >= rs.median);
        n_flagged += @intFromBool(per_row[i].flagged);
    }

    // invariant 5: the alignment canary — teacher-forced next-token accuracy of both
    // sides against the prompt tokens (position t predicts token t+1; a row's last
    // position has no next token and is not scored). Random rows near 1/V proves the
    // prompt alignment; a reference outside the 0.5-0.7 band is a misaligned file.
    var cn: usize = 0;
    var cn_ref: usize = 0;
    var cn_cand: usize = 0;
    var cr: usize = 0;
    var cr_ref: usize = 0;
    var cr_cand: usize = 0;
    for (0..seq_rows) |i| {
        const basep = i * o.cols;
        for (0 .. o.cols - 1) |t| {
            const next_tok: u32 = @intCast(@as(u32, @bitCast(ids[basep + t + 1])));
            const r0 = rows_stats[basep + t];
            if (random_row[i]) {
                cr += 1;
                cr_ref += @intFromBool(r0.a_ref == next_tok);
                cr_cand += @intFromBool(r0.a_cand == next_tok);
            } else {
                cn += 1;
                cn_ref += @intFromBool(r0.a_ref == next_tok);
                cn_cand += @intFromBool(r0.a_cand == next_tok);
            }
        }
    }
    const cnf: f64 = @floatFromInt(@max(cn, 1));
    const crf: f64 = @floatFromInt(@max(cr, 1));
    const canary = Canary{
        .natural = .{
            .n = cn,
            .ref_acc = @as(f64, @floatFromInt(cn_ref)) / cnf,
            .cand_acc = @as(f64, @floatFromInt(cn_cand)) / cnf,
        },
        .random = .{
            .n = cr,
            .ref_acc = @as(f64, @floatFromInt(cr_ref)) / crf,
            .cand_acc = @as(f64, @floatFromInt(cr_cand)) / crf,
        },
        .inv_vocab = 1.0 / @as(f64, @floatFromInt(q.VOCAB)),
    };

    // the per-position KL histogram: mass only in the zero bin plus one interior bin is
    // the two-valued signature of a broken statistic kernel; a sane run is right-tailed
    const hist_nat = try alloc.alloc(u64, stats.hist_bins);
    stats.histCounts(headline.items, hist_nat);
    var hist_ran: ?[]const u64 = null;
    if (rand_rows.items.len > 0) {
        const c = try alloc.alloc(u64, stats.hist_bins);
        stats.histCounts(rand_rows.items, c);
        hist_ran = c;
    }
    const bins_line = try histBinsLine(alloc);
    std.debug.print("  histogram bins: {s}\n", .{bins_line});
    std.debug.print("  histogram natural n={d}: {s}\n", .{ headline.items.len, try histCountsLine(alloc, hist_nat) });
    if (hist_ran) |c| std.debug.print("  histogram random  n={d}: {s}\n", .{ rand_rows.items.len, try histCountsLine(alloc, c) });
    if (o.dump_hist) |hp2| {
        var list: std.ArrayList(u8) = .empty;
        defer list.deinit(alloc);
        try list.print(alloc, "xpu-eaq-kl per-position KL histogram — KL(ref||cand) — build {s}\n", .{buildID});
        try list.print(alloc, "mode {s} pack '{s}' cand '{s}' layout {s}\n", .{
            if (logits_mode) "logits" else "pack", o.pack, o.cand_logits orelse "", try layoutTag(o.pack, logits_mode) });
        try list.print(alloc, "prompts {s} sha256 {s}\nref {s} sha256 {s}\n", .{
            o.prompts, prompts_sha, o.ref orelse "self", ref_sha orelse "self" });
        try list.print(alloc, "bins: {s}\n", .{bins_line});
        try list.print(alloc, "natural n={d}: {s}\n", .{ headline.items.len, try histCountsLine(alloc, hist_nat) });
        if (hist_ran) |c| try list.print(alloc, "random  n={d}: {s}\n", .{ rand_rows.items.len, try histCountsLine(alloc, c) });
        const f = openFd(hp2, O_WRONLY_CREAT_TRUNC) catch |e| {
            std.debug.print("xpu-eaq-kl: cannot write {s}\n", .{hp2});
            return e;
        };
        defer _ = close(f);
        var off: usize = 0;
        while (off < list.items.len) {
            const w2 = pwrite(f, list.items.ptr + off, list.items.len - off, @intCast(off));
            if (w2 <= 0) return Error.FileIo;
            off += @intCast(w2);
        }
        std.debug.print("  wrote {s}\n", .{hp2});
    }
    if (o.dump_kl) |kp| try writeKlDump(kp, .{
        .cols = o.cols, .seq_rows = seq_rows, .vocab = q.VOCAB,
    }, prompts_sha, ref_sha, if (logits_mode) "logits" else "pack", rows_stats);

    var self_check: ?SelfCheck = null;
    if (o.self_mode or identity) {
        var max_abs: f64 = 0;
        var top1: usize = 0;
        for (rows_stats) |rs| {
            top1 += @intFromBool(rs.top1);
            max_abs = @max(max_abs, @max(@abs(rs.kl), @max(@abs(rs.kl_rev), @abs(rs.shift))));
        }
        self_check = .{
            .pass = max_abs == 0 and top1 == rows_stats.len,
            .rows_checked = rows_stats.len,
            .max_abs_kl = max_abs,
            .top1_frac = @as(f64, @floatFromInt(top1)) / @as(f64, @floatFromInt(rows_stats.len)),
        };
    }

    const layout = try layoutTag(o.pack, logits_mode);
    const report = Report{
        .pack = o.pack,
        .mode = if (logits_mode) "logits" else "pack",
        .cand = o.cand_logits orelse "",
        .cand_sha256 = cand_sha,
        .layout = layout,
        .prompts = o.prompts, .prompts_sha256 = prompts_sha,
        .ref = o.ref orelse (if (req.dump_ref) |dr| dr else "self"),
        .ref_sha256 = ref_sha,
        .dump_sha256 = if (req.dump_ref) |dr| try sha256Hex(dr) else null,
        .seq_rows = seq_rows, .cols = o.cols, .positions = positions, .vocab = q.VOCAB,
        .block = o.block, .threads = threads,
        .head = if (hd) |h| h.info else null,
        .seconds = .{ .prompts_s = prompts_s, .dequant_s = dequant_s, .gemm_s = if (hd) |h| h.gemm_s else 0, .stats_s = stats_s, .total_s = q.now() - t_total },
        .headline = headline_summary,
        .random = if (rand_rows.items.len > 0) try stats.Summary.init(rand_rows.items) else null,
        .per_domain = per_domain,
        .per_row = per_row,
        .canary = canary,
        .histogram = hist_nat,
        .histogram_random = hist_ran,
        .spot_check = spot,
        .self_check = self_check,
    };

    std.debug.print("  KL(ref||cand) mean {d:.6} ± {d:.6} nats, median {d:.6}, p99 {d:.6}, max {d:.6}, top-1 {d:.3}, shift {d:.4}, reverse {d:.6} (n={d})\n", .{
        report.headline.mean, report.headline.se, report.headline.median, report.headline.p99, report.headline.max,
        report.headline.top1_frac, report.headline.shift_mean, report.headline.reverse_mean, report.headline.n });
    if (report.random) |*rm| std.debug.print("  random rows: mean {d:.6} nats (n={d})\n", .{ rm.mean, rm.n });
    for (per_domain) |dm| std.debug.print("  domain {s} ({d} rows): mean {d:.6} se {d:.6} median {d:.6} p99 {d:.6} top-1 {d:.3}\n", .{
        dm.name, dm.seq_rows, dm.s.mean, dm.s.se, dm.s.median, dm.s.p99, dm.s.top1_frac });
    std.debug.print("  per-row means:", .{});
    for (per_row) |pr0| std.debug.print(" {d}:{d:.6}{s}", .{ pr0.row, pr0.mean, if (pr0.flagged) "*" else "" });
    std.debug.print("\n", .{});
    std.debug.print("  canary next-token acc: natural ref {d:.4} cand {d:.4} (band ~0.5-0.7, n={d}) | random ref {d:.6} cand {d:.6} (1/V {d:.6})\n", .{
        canary.natural.ref_acc, canary.natural.cand_acc, canary.natural.n,
        canary.random.ref_acc, canary.random.cand_acc, canary.inv_vocab });
    var dom_ge: usize = 0;
    for (per_domain) |dm| {
        if (dm.s.mean >= dm.s.median) dom_ge += 1;
    }
    const viol = (n_flagged != 0) or headline_summary.mean < headline_summary.median or
        dom_ge != per_domain.len or rows_ge_median != seq_rows;
    std.debug.print("  invariants: mean>=median headline {s}, domains {d}/{d}, rows {d}/{d}; rows over 3x headline: {d}{s}\n", .{
        if (headline_summary.mean >= headline_summary.median) "ok" else "FAIL", dom_ge, per_domain.len,
        rows_ge_median, seq_rows, n_flagged, if (viol) "  <- INVARIANT VIOLATION" else "" });
    if (spot) |sp| std.debug.print("  spot check @pos {d}: agree {s}, max |Δ| {d:.6}, runner top1 {d}, head top1 {d}\n", .{
        sp.position, if (sp.agree) "yes" else "NO", sp.max_abs_delta, sp.runner_top1, sp.head_top1 });
    if (self_check) |sc| std.debug.print("  self check: {s} ({d} rows, max |KL| {d}, top-1 {d:.4})\n", .{
        if (sc.pass) "PASS" else "FAIL", sc.rows_checked, sc.max_abs_kl, sc.top1_frac });

    if (o.report) |rpp| {
        const js = try std.json.Stringify.valueAlloc(alloc, report, .{ .whitespace = .indent_1 });
        defer alloc.free(js);
        const f = openFd(rpp, O_WRONLY_CREAT_TRUNC) catch |e| {
            std.debug.print("xpu-eaq-kl: cannot write {s}: {s}\n", .{ rpp, @errorName(e) });
            return e;
        };
        defer _ = close(f);
        if (pwrite(f, js.ptr, js.len, 0) != @as(isize, @intCast(js.len))) return Error.FileIo;
        std.debug.print("  wrote {s}\n", .{rpp});
    }
    if (self_check) |sc| {
        if (!sc.pass) return Error.SelfCheck;
    }
    return report;
}

/// A GB-scale anonymous f32 buffer for the streamed logits files: raw private mapping,
/// MAP_POPULATE, self-touched before any syscall sees it (see the note above rbuf).
fn mapF32(n: usize) !struct { p: *anyopaque, buf: []f32 } {
    const bytes = n * @sizeOf(f32);
    const p = mmap(null, bytes, PROT_RW, MAP_PRIVATE_ANON_POP, -1, 0);
    if (@intFromPtr(p) == @as(usize, @bitCast(@as(isize, -1)))) return error.OutOfMemory;
    const fp: [*]f32 = @ptrCast(@alignCast(p));
    const buf = fp[0..n];
    @memset(buf, 0);
    return .{ .p = p, .buf = buf };
}

/// The provenance flag (kl-order-ruling.md invariant 6, pack manifest or stock-layout
/// flag): an EAQ pack carries eaq_manifest.json (the runner's dispatch key), a stock
/// EXL3 pack does not; the diagnostic mode has no pack at all.
fn layoutTag(pack_dir: []const u8, logits_mode: bool) ![]const u8 {
    if (logits_mode) return "logits-file";
    if (pack_dir.len == 0) return "none";
    const mpath = try std.fmt.allocPrint(alloc, "{s}/eaq_manifest.json", .{pack_dir});
    if (openFd(mpath, 0)) |fd| {
        _ = close(fd);
        if (sha256Hex(mpath)) |sha| {
            return std.fmt.allocPrint(alloc, "eaq-manifest:{s}", .{sha[0..16]});
        } else |_| {
            return "eaq-manifest";
        }
    } else |_| {
        return "stock-exl3";
    }
}

fn histCountsLine(a: std.mem.Allocator, counts: []const u64) ![]u8 {
    var l: std.ArrayList(u8) = .empty;
    for (counts, 0..) |c, i| {
        if (i != 0) try l.print(a, " ", .{});
        try l.print(a, "{d}", .{c});
    }
    return l.toOwnedSlice(a);
}

fn histBinsLine(a: std.mem.Allocator) ![]u8 {
    var l: std.ArrayList(u8) = .empty;
    try l.print(a, "zero", .{});
    for (stats.hist_edge_names, 0..) |name, i| {
        if (i == 0) try l.print(a, " | (0,{s}]", .{name}) else try l.print(a, " | ({s},{s}]", .{ stats.hist_edge_names[i - 1], name });
    }
    try l.print(a, " | ({s},inf)", .{stats.hist_edge_names[stats.hist_edge_names.len - 1]});
    return l.toOwnedSlice(a);
}

// ---- the per-position dump and the paired-difference block (invariant 3, G1 form) ----

pub const kl_dump_magic: u32 = 0x4B4C_4431; // "KLD1"

pub const KlDumpHeader = extern struct {
    magic: u32 = kl_dump_magic,
    version: u32 = 1,
    count: u64 = 0,
    cols: u64 = 0,
    seq_rows: u64 = 0,
    vocab: u64 = 0,
    prompts_sha: [64]u8 = std.mem.zeroes([64]u8), // lowercase hex, zero-padded
    ref_sha: [64]u8 = std.mem.zeroes([64]u8),
    mode: [8]u8 = std.mem.zeroes([8]u8), // "pack" | "logits"
};

pub const KlDumpRow = extern struct {
    kl: f64 = 0,
    a_ref: u32 = 0,
    a_cand: u32 = 0,
    top1: u8 = 0,
    pad: [3]u8 = .{ 0, 0, 0 },
};

fn setPadded(dst: []u8, src: ?[]const u8) void {
    @memset(dst, 0);
    if (src) |s| {
        const n = @min(s.len, dst.len);
        @memcpy(dst[0..n], s[0..n]);
    }
}

fn writeKlDump(
    path: []const u8,
    hdr0: KlDumpHeader,
    prompts_sha: []const u8,
    ref_sha: ?[]const u8,
    mode_str: []const u8,
    rows: []const stats.Row,
) !void {
    var hdr = hdr0;
    hdr.count = rows.len;
    setPadded(hdr.prompts_sha[0..], prompts_sha);
    setPadded(hdr.ref_sha[0..], ref_sha);
    setPadded(hdr.mode[0..], mode_str);
    const recs = try alloc.alloc(KlDumpRow, rows.len);
    defer alloc.free(recs);
    for (rows, recs) |r, *rec| rec.* = .{ .kl = r.kl, .a_ref = r.a_ref, .a_cand = r.a_cand, .top1 = @intFromBool(r.top1) };
    const fd = openFd(path, O_WRONLY_CREAT_TRUNC) catch |e| {
        std.debug.print("xpu-eaq-kl: cannot write {s}\n", .{path});
        return e;
    };
    defer _ = close(fd);
    var pos: usize = 0;
    for ([_][]const u8{ std.mem.asBytes(&hdr), std.mem.sliceAsBytes(recs) }) |bytes| {
        var off: usize = 0;
        while (off < bytes.len) {
            const w = pwrite(fd, bytes.ptr + off, bytes.len - off, @intCast(pos));
            if (w <= 0) return Error.DumpShort;
            off += @intCast(w);
            pos += @intCast(w);
        }
    }
    std.debug.print("  wrote {s} ({d} positions)\n", .{ path, rows.len });
}

fn readKlDump(path: []const u8) !struct { hdr: KlDumpHeader, rows: []stats.Row } {
    const raw = try readFileAll(path, 1 << 28); // the 40960-position dump is ~1 MB
    defer alloc.free(raw);
    if (raw.len < @sizeOf(KlDumpHeader)) return Error.PairMismatch;
    const hdr = std.mem.bytesAsValue(KlDumpHeader, raw[0..@sizeOf(KlDumpHeader)]).*;
    if (hdr.magic != kl_dump_magic or hdr.version != 1) return Error.PairMismatch;
    const want = @sizeOf(KlDumpHeader) + @as(usize, @intCast(hdr.count)) * @sizeOf(KlDumpRow);
    if (raw.len != want) return Error.PairMismatch;
    const recs = std.mem.bytesAsSlice(KlDumpRow, raw[@sizeOf(KlDumpHeader)..]);
    const rows = try alloc.alloc(stats.Row, @intCast(hdr.count));
    for (recs, rows) |rec, *r| r.* = .{
        .kl = rec.kl, .kl_rev = 0, .shift = 0, .top1 = rec.top1 != 0,
        .a_ref = rec.a_ref, .a_cand = rec.a_cand,
    };
    return .{ .hdr = hdr, .rows = rows };
}

pub const PairReport = struct {
    tool: []const u8 = "xpu-eaq-kl-pair",
    build_id: []const u8 = buildID,
    direction: []const u8 = "KL(ref||cand)",
    a: []const u8,
    b: []const u8,
    prompts_sha256: []const u8,
    ref_sha256: []const u8,
    positions: usize,
    natural_positions: usize,
    all: stats.Paired,
    natural: stats.Paired,
};

fn printPair(name: []const u8, p: stats.Paired) void {
    const favours = if (p.mean_diff < 0) "A" else if (p.mean_diff > 0) "B" else "neither";
    const top_favours = if (p.n10 > p.n01) "B" else if (p.n01 > p.n10) "A" else "neither";
    std.debug.print("  {s} (n={d}): KL mean A {d:.6} B {d:.6} | paired mean(A-B) {d:.6} se {d:.6} — favours {s} by {d:.1} se\n", .{
        name, p.n, p.mean_a, p.mean_b, p.mean_diff, p.se_diff, favours, if (p.se_diff > 0) @abs(p.mean_diff) / p.se_diff else @as(f64, 0) });
    std.debug.print("  {s} top-1: A {d} B {d} | discordance A-only {d} B-only {d} — favours {s}, McNemar chi2 {d:.2}\n", .{
        name, p.top1_a, p.top1_b, p.n01, p.n10, top_favours, p.mcnemar_chi2 });
}

/// The paired-difference block for two candidates measured against the same reference
/// logits (kl-order-ruling.md invariant 3): the mean and se of the per-position KL
/// differences and the top-1 discordance counts, headline over the natural rows with
/// all positions beside. The dumps must share prompts and reference — a mismatch is
/// refused, and so is a reference argmax that moves (the runs would not be paired on
/// the same distributions). Ordering claims cite these numbers, never quadrature of
/// the marginal se.
pub fn pairRun(pa: []const u8, pb: []const u8, mpath: ?[]const u8, report_path: ?[]const u8) !PairReport {
    const A = try readKlDump(pa);
    const B = try readKlDump(pb);
    if (A.hdr.count != B.hdr.count or A.hdr.cols != B.hdr.cols or A.hdr.seq_rows != B.hdr.seq_rows or A.hdr.vocab != B.hdr.vocab) {
        std.debug.print("xpu-eaq-kl --pair: {s} is {d} positions ({d} rows x {d}) but {s} is {d} ({d} x {d}) — the block needs the same positions\n", .{
            pa, A.hdr.count, A.hdr.seq_rows, A.hdr.cols, pb, B.hdr.count, B.hdr.seq_rows, B.hdr.cols });
        return Error.PairMismatch;
    }
    if (!std.mem.eql(u8, A.hdr.prompts_sha[0..], B.hdr.prompts_sha[0..])) {
        std.debug.print("xpu-eaq-kl --pair: prompts differ ({s} vs {s}) — not paired\n", .{
            std.mem.sliceTo(&A.hdr.prompts_sha, 0), std.mem.sliceTo(&B.hdr.prompts_sha, 0) });
        return Error.PairMismatch;
    }
    if (!std.mem.eql(u8, A.hdr.ref_sha[0..], B.hdr.ref_sha[0..])) {
        std.debug.print("xpu-eaq-kl --pair: reference logits differ ({s} vs {s}) — not paired\n", .{
            std.mem.sliceTo(&A.hdr.ref_sha, 0), std.mem.sliceTo(&B.hdr.ref_sha, 0) });
        return Error.PairMismatch;
    }
    var ref_mismatch: usize = 0;
    for (A.rows, B.rows) |ra, rb| {
        if (ra.a_ref != rb.a_ref) ref_mismatch += 1;
    }
    if (ref_mismatch != 0) {
        std.debug.print("xpu-eaq-kl --pair: the reference argmax moves on {d} positions — the dumps were not taken against one reference\n", .{ref_mismatch});
        return Error.PairMismatch;
    }
    std.debug.print("pair KL(ref||cand): A {s}\n     pair             : B {s}\n", .{ pa, pb });
    std.debug.print("  prompts sha256 {s} ref sha256 {s} (shared, {d} positions)\n", .{
        std.mem.sliceTo(&A.hdr.prompts_sha, 0), std.mem.sliceTo(&A.hdr.ref_sha, 0), A.rows.len });

    const seq_rows: usize = @intCast(A.hdr.seq_rows);
    const cols: usize = @intCast(A.hdr.cols);
    std.debug.assert(A.rows.len == seq_rows * cols);
    var random_row = try alloc.alloc(bool, seq_rows);
    if (mpath) |mp| {
        const mraw = try readFileAll(mp, 1 << 26);
        defer alloc.free(mraw);
        const m = try std.json.parseFromSlice(Manifest, alloc, mraw, .{ .ignore_unknown_fields = true });
        defer m.deinit();
        const mrows = m.value.rows orelse return Error.ManifestRows;
        if (mrows.len != seq_rows) return Error.ManifestRows;
        for (mrows, 0..) |mr, i| {
            if (mr.row != i) return Error.ManifestRows;
            random_row[i] = mr.kind != null and std.mem.eql(u8, mr.kind.?, "random");
        }
    } else {
        for (0..seq_rows) |i| random_row[i] = i >= seq_rows - @min(seq_rows, 4);
    }
    var anat: std.ArrayList(stats.Row) = .empty;
    var bnat: std.ArrayList(stats.Row) = .empty;
    for (A.rows, 0..) |ra, i| {
        if (!random_row[i / cols]) {
            try anat.append(alloc, ra);
            try bnat.append(alloc, B.rows[i]);
        }
    }
    const allp = stats.Paired.init(A.rows, B.rows);
    const natp = stats.Paired.init(anat.items, bnat.items);
    printPair("all", allp);
    printPair("natural", natp);

    const pr = PairReport{
        .a = pa, .b = pb,
        .prompts_sha256 = std.mem.sliceTo(&A.hdr.prompts_sha, 0),
        .ref_sha256 = std.mem.sliceTo(&A.hdr.ref_sha, 0),
        .positions = A.rows.len,
        .natural_positions = natp.n,
        .all = allp,
        .natural = natp,
    };
    if (report_path) |rp| {
        const js = try std.json.Stringify.valueAlloc(alloc, pr, .{ .whitespace = .indent_1 });
        defer alloc.free(js);
        const f = openFd(rp, O_WRONLY_CREAT_TRUNC) catch |e| {
            std.debug.print("xpu-eaq-kl: cannot write {s}: {s}\n", .{ rp, @errorName(e) });
            return e;
        };
        defer _ = close(f);
        if (pwrite(f, js.ptr, js.len, 0) != @as(isize, @intCast(js.len))) return Error.FileIo;
        std.debug.print("  wrote {s}\n", .{rp});
    }
    return pr;
}
