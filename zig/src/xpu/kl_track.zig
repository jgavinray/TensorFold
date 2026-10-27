//! Allocator-instance instrumentation for the KL gate's host path (kl.zig ticket:
//! the 20-row `xpu-eaq-kl --self` nondeterministic segfault — gdb shows a
//! malloc_usable_size SEGV inside ze.Gpu.keep / ArrayList append during Head.init's
//! module load: heap metadata corrupted upstream, surfacing at the next malloc).
//!
//! The module compiles into every xpu-eaq-kl, but every entry point below is a
//! no-op unless the BUILD sets tracking=true in its kl_build options module
//! (zig/build/xpu_kl.zig: -Dkl-track=true, or TF_KL_ALLOC_TRACK=1 in the env): the
//! default build swaps no allocator and calls nothing here, so the shipped gate is
//! behaviourally untouched. With the flag set, qwen27.zig's `pub const alloc`
//! becomes `kl_track.allocator` and the checkpoints in kl.zig light up.
//! Run the instrumented binary normally — tracking is baked in at compile
//! time; it prints `kl_track: ON` on the first allocation so a stale uninstrumented
//! binary can never be mistaken for one.
//!
//! What it does:
//!  * every host block carries a header (magic, size, map length), head and tail
//!    redzones, and is registered with the libc backtrace taken at allocation —
//!    and again at free — the per-instance tracking the ticket asks for;
//!  * free poison-fills the payload and QUARANTINES the mapping (munmap would hide
//!    the exact suspect: a device DMA landing in a freed block after a premature
//!    d.sync — e.g. the XPU_DUMP_H host copy in forward.zig — scribbling a recycled
//!    chunk's malloc bookkeeping);
//!  * verify(label) sweeps every live block's redzones and every quarantined block's
//!    poison and reports offenders with their alloc/free backtraces — the first
//!    moved poison names the victim and, through its alloc trace, the code that
//!    owned the memory when the write landed;
//!  * phase() + the checkpoints in kl.run bind the corruption to a phase (the
//!    captureHidden memfd pass, Head.init's module load, each streaming block...);
//!  * finish() dumps the live-allocation instance audit (who holds what when the
//!    teardown-only exit segfault would fire — a clean audit plus a crash afterwards
//!    localizes that crash to the Level Zero teardown rather than Zig's heap) and a
//!    condensed /proc/self/maps so the memfd-vs-streaming-mmap-vs-heap layout the
//!    interplay hypothesis needs is on record.
//!
//! Address symbolication: frames print as `exe+0x...` once dladdr resolves the main
//! image base; feed those offsets to
//!   addr2line -f -C -i -e <exe> <off>   (PIE: raw offsets; non-PIE: base+off)
//! to name them. Nothing here allocates: the registry is fixed statics and reports
//! go straight to fd 2, so the tracker can never nest into the allocator it watches.
//! The ze.Gpu keep-lists deliberately stay on the libc allocator: those lists are
//! the victim of this bug, not the culprit, and the tracker's redzones around the
//! runner/gate host blocks are where a wild write lands first.
const std = @import("std");
pub const tracking = @import("kl_build").tracking;
const Alignment = std.mem.Alignment; // the vtable's alignment type in this Zig (Allocator.Alignment is not pub)

const red: u32 = 0xDEADB33F;
const poison: u8 = 0xAA;
const magic: u64 = 0x5A17_AC1E_9D3F_0C71;
const frames_max: usize = 12;
const slots_max: usize = 65536; // live-block registry (freed blocks leave it immediately)
const quarantine_max: usize = 32768; // freed blocks kept mapped (bounded in count and bytes)
const quarantine_bytes_max: usize = 6 << 30;
const report_max: usize = 10; // offenders printed per verify pass
const redwords: u32 = 16; // 64B head + 64B tail

const Header = extern struct {
    magic: u64,
    size: u32, // requested bytes
    redz: u32, // redzone words on each side
    map_len: usize, // the exact mmap length (page-rounded)
    slot: u32, // live-registry slot, maxInt(u32) when unregistered
    ret_addr: usize, // first return address the caller volunteered
    freed: bool, // quarantined: payload poisoned, mapping kept for the sweep
    pad: u8,
    quar_next: ?*Header, // quarantine-list linkage
};

const Rec = struct {
    used: bool = false,
    ptr: usize = 0,
    size: usize = 0,
    site: []const u8 = "",
    epoch: u32 = 0,
    alloc_bt: [frames_max]usize = undefined,
    n_alloc: u8 = 0,
};

const Region = struct { base: usize, len: usize, kind: []const u8 };
const Expect = struct { name: []const u8, bytes: usize };

/// Names a buffer family the corruption hunt cares about (the captureHidden memfd,
/// the per-block logits D2H, the slab H2D...). finish() tags every /proc/self/maps
/// entry whose size lands within 2x of one of these, so the adjacency question —
/// did the rotted chunk sit next to the memfd or next to a GB-scale D2H mapping —
/// is answered on the printout, without anyone diffing maps tables by hand.
pub fn expect(name: []const u8, bytes: usize) void {
    if (!tracking) return;
    mutex.lock();
    defer mutex.unlock();
    for (expects[0..n_expect]) |e| if (e.bytes == bytes) return; // one line per size
    if (n_expect < expects.len) {
        expects[n_expect] = .{ .name = name, .bytes = bytes };
        n_expect += 1;
    }
}

const Spin = struct {
    word: u8 = 0,
    fn lock(self: *Spin) void {
        while (@atomicRmw(u8, &self.word, .Xchg, 1, .acq_rel) != 0) {}
    }
    fn unlock(self: *Spin) void {
        @atomicStore(u8, &self.word, 0, .release);
    }
};
var mutex: Spin = .{};
var recs: [slots_max]Rec = undefined;
var live_blocks: usize = 0;
var live_bytes: usize = 0;
var quar_head: ?*Header = null;
var quar_blocks: usize = 0;
var quar_bytes: usize = 0;
var capacity_warned = false;
var epoch: u32 = 0;
var phase_label: []const u8 = "init";
var regions: [96]Region = undefined;
var n_region: usize = 0;
var expects: [16]Expect = undefined;
var n_expect: usize = 0;
var banner_done = false;
var exe_base: usize = 0;

const Dl_info = extern struct {
    dli_fname: ?[*:0]const u8,
    dli_fbase: ?*anyopaque,
    dli_sname: ?[*:0]const u8,
    dli_saddr: ?*anyopaque,
};
extern "c" fn backtrace(buffer: [*]*anyopaque, size: c_int) c_int;
extern "c" fn dladdr(addr: *const anyopaque, info: *Dl_info) c_int;
extern "c" fn write(fd: c_int, buf: [*]const u8, count: usize) isize;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) callconv(.c) *anyopaque;
extern "c" fn munmap(addr: *anyopaque, len: usize) c_int;
extern "c" fn mprotect(addr: *anyopaque, len: usize, prot: c_int) c_int;
extern "c" fn open(path: [*:0]const u8, flags: c_int, mode: c_int) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, count: usize) isize;
extern "c" fn close(fd: c_int) c_int;

const PROT_NONE: c_int = 0;
const PROT_RW: c_int = 0x1 | 0x2;
const MAP_PRIVATE_ANON: c_int = 0x02 | 0x20;
const O_RDONLY: c_int = 0;

fn puts(s: []const u8) void {
    _ = write(2, s.ptr, s.len);
}

fn hexBuf(buf: []u8, x: u64) void {
    const d = "0123456789abcdef";
    var j: usize = buf.len;
    var v = x;
    while (j > 0) {
        j -= 1;
        buf[j] = d[@as(usize, @intCast(v & 0xf))];
        v >>= 4;
    }
}

fn fmtU(s: []u8, x: usize) []const u8 {
    if (x == 0) return "0";
    var d: [24]u8 = undefined;
    var j: usize = d.len;
    var v = x;
    while (v > 0) {
        j -= 1;
        d[j] = '0' + @as(u8, @intCast(v % 10));
        v /= 10;
    }
    const out = s[0 .. d.len - j];
    @memcpy(out, d[j..]);
    return out;
}

fn log(comptime f: []const u8, args: anytype) void {
    var buf: [2048]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, f, args) catch return;
    puts(s);
}

fn frameName(addr: usize, out: []u8) []const u8 {
    var hx: [16]u8 = undefined;
    if (exe_base != 0 and addr >= exe_base) {
        hexBuf(&hx, addr - exe_base);
        return std.fmt.bufPrint(out, "exe+0x{s}", .{hx[0..12]}) catch "?";
    }
    hexBuf(&hx, addr);
    return std.fmt.bufPrint(out, "0x{s}", .{hx[0..16]}) catch "?";
}

fn resolveExeBase() void {
    var di: Dl_info = undefined;
    if (dladdr(&resolveExeBase, &di) != 0) {
        if (di.dli_fbase) |b| exe_base = @intFromPtr(b);
    }
}

/// Skips backtrace() and this wrapper; stores raw return addresses.
fn captureBt(bt: *[frames_max]usize) u8 {
    var raw: [frames_max]*anyopaque = undefined;
    const n = backtrace(&raw, @intCast(frames_max));
    if (n <= 2) return 0;
    var k: usize = 0;
    var i: usize = 2;
    while (i < @as(usize, @intCast(n)) and k < frames_max) : (i += 1) {
        bt[k] = @intFromPtr(raw[i]);
        k += 1;
    }
    return @intCast(k);
}

fn dumpBt(tag: []const u8, bt: []const usize, nf: u8) void {
    var name: [64]u8 = undefined;
    for (bt[0..nf], 0..) |a, i| log("    {s}[{d}] {s}\n", .{ tag, i, frameName(a, &name) });
}

fn hdrOf(user: *anyopaque) *Header {
    return @ptrFromInt(@intFromPtr(user) - @sizeOf(Header) - redwords * 4);
}

inline fn redHeadMut(h: *const Header) [*]u32 {
    return @ptrFromInt(@intFromPtr(h) + @sizeOf(Header));
}
inline fn redTailMut(h: *const Header) [*]u32 {
    return @ptrFromInt(@intFromPtr(h) + @sizeOf(Header) + h.redz * 4 + h.size);
}
inline fn redHead(h: *const Header) [*]const u32 {
    return redHeadMut(h);
}
inline fn redTail(h: *const Header) [*]const u32 {
    return redTailMut(h);
}

fn banner() void {
    if (banner_done) return;
    banner_done = true;
    resolveExeBase();
    puts("kl_track: ON (TF_KL_ALLOC_TRACK build; quarantine + redzone sweep active)\n");
}

/// A Level Zero host wait that returned SUCCESS while its event was still
/// unsignaled: the premature-d.sync hypothesis caught red-handed, with the phase
/// label attached. kl.zig's evWait re-checks every event immediately after the
/// synchronize returns, so the lie and the phase that trusted it print together —
/// the quarantine then names whichever freed block the flying DMA actually ate.
pub fn lie(what: []const u8, handle: usize) void {
    if (!tracking) return;
    banner();
    var hx: [16]u8 = undefined;
    hexBuf(&hx, handle);
    log("!! kl_track WAIT-LIE [{s}]: {s} (event 0x{s}) — driver answered early, a downstream DMA is still flying\n", .{ phase_label, what, hx[0..16] });
}

fn report(at: ?usize, why: []const u8, rec_in: ?*const Rec, h: ?*const Header, diverged: ?usize) void {
    var hx: [16]u8 = undefined;
    if (at) |p| hexBuf(&hx, p) else hexBuf(&hx, 0);
    var s1: [24]u8 = undefined;
    var s2: [24]u8 = undefined;
    var s3: [24]u8 = undefined;
    log("!! kl_track CORRUPTION [{s}] at 0x{s}: {s} (live {s}/0x{s}B, quarantined {s})\n", .{
        phase_label, hx[0..16], why, fmtU(&s1, live_blocks), fmtU(&s2, live_bytes), fmtU(&s3, quar_blocks),
    });
    if (diverged) |d| log("   first diverged byte offset: {d}\n", .{d});
    if (rec_in) |rec| {
        log("   block {d}B, alloc phase {s}, alloc epoch {d}\n", .{ rec.size, rec.site, rec.epoch });
        dumpBt("alloc", &rec.alloc_bt, rec.n_alloc);
    } else if (h) |hh| {
        log("   quarantined block {d}B\n", .{hh.size});
        if (hh.ret_addr != 0) {
            var name: [64]u8 = undefined;
            log("    alloc ret {s}\n", .{frameName(hh.ret_addr, &name)});
        }
    }
}

fn allocImpl(_: *anyopaque, len: usize, alignment: Alignment, ret_addr: usize) ?[*]u8 {
    if (len == 0) return null;
    banner();
    const ab = @max(alignment.toByteUnits(), @alignOf(Header));
    const raw_off = @sizeOf(Header) + redwords * 4;
    const payload_off = raw_off + ((ab - raw_off % ab) % ab);
    const total = (payload_off + len + redwords * 4 + 4095) & ~@as(usize, 4095);
    const mem = mmap(null, total, PROT_RW, MAP_PRIVATE_ANON, -1, 0);
    if (@intFromPtr(mem) == @as(usize, @bitCast(@as(isize, -1)))) return null;
    const base: usize = @intFromPtr(mem);
    const h: *Header = @ptrFromInt(base);
    h.* = .{
        .magic = magic, .size = @intCast(len), .redz = redwords, .map_len = total,
        .slot = std.math.maxInt(u32), .ret_addr = ret_addr, .freed = false, .pad = 0, .quar_next = null,
    };
    const rh = redHeadMut(h);
    for (0..h.redz) |i| rh[i] = red;
    const rt = redTailMut(h);
    for (0..h.redz) |i| rt[i] = red;
    const user: [*]u8 = @ptrFromInt(base + payload_off);
    @memset(user[0..len], poison); // alloc poison-fills too: an unread-then-read byte tells
    mutex.lock();
    defer mutex.unlock();
    var probe: usize = (base / 16) % slots_max;
    var i: usize = 0;
    while (i < slots_max) : (i += 1) {
        if (!recs[probe].used) {
            recs[probe] = .{ .used = true, .ptr = base + payload_off, .size = len, .site = phase_label, .epoch = epoch };
            recs[probe].n_alloc = captureBt(&recs[probe].alloc_bt);
            h.slot = @intCast(probe);
            live_blocks += 1;
            live_bytes += len;
            return user;
        }
        probe = (probe + 1) % slots_max;
    }
    capacity_warned = true; // registry full: the block still works, just untracked
    return user;
}

fn resizeImpl(_: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) bool {
    _ = .{ &memory, &alignment, &new_len, &ret_addr };
    return false; // always relocate: alloc+copy+free through the instrumented path
}

fn remapImpl(_: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
    _ = .{ &memory, &alignment, &new_len, &ret_addr };
    return null; // caller falls back to alloc+copy+free
}

fn freeImpl(_: *anyopaque, memory: []const u8, alignment: Alignment, ret_addr: usize) void {
    _ = .{ &alignment, &ret_addr };
    if (memory.len == 0) return;
    const user: [*]u8 = @constCast(memory.ptr);
    const h = hdrOf(@ptrCast(user));
    if (h.magic != magic) {
        report(@intFromPtr(user), "free of a block whose header is gone — external scribble or a foreign free", null, null, null);
        return;
    }
    var broke: ?usize = null;
    for (0..h.redz) |i| if (redHead(h)[i] != red) {
        broke = i;
        break;
    };
    if (broke == null) for (0..h.redz) |i| if (redTail(h)[i] != red) {
        broke = 1000 + i;
        break;
    };
    if (broke != null) report(@intFromPtr(user), "free with redzones already written — the corruption predates this free", null, h, broke);
    mutex.lock();
    defer mutex.unlock();
    if (h.slot != std.math.maxInt(u32)) {
        const rec = &recs[h.slot];
        if (rec.used and rec.ptr == @intFromPtr(user)) {
            live_blocks -= 1;
            live_bytes -= rec.size;
            rec.* = .{}; // the live registry keeps only live blocks
        }
        h.slot = std.math.maxInt(u32);
    }
    if (h.freed) {
        report(@intFromPtr(user), "double free — this block is already in quarantine", null, h, null);
        return;
    }
    h.freed = true;
    if (quar_blocks < quarantine_max and quar_bytes + h.size <= quarantine_bytes_max) {
        @memset(user[0..h.size], poison); // quarantine: keep the mapping so a late DMA is visible
        h.quar_next = quar_head; // the sweep walks this list; an unlinked block is an unseen block
        quar_head = h;
        quar_blocks += 1;
        quar_bytes += h.size;
        return;
    }
    capacity_warned = true;
    _ = munmap(@ptrFromInt(@intFromPtr(h)), h.map_len); // header is the map base
}

/// Sets the label every subsequent allocation and verify line carries.
pub fn phase(label: []const u8) void {
    if (!tracking) return;
    mutex.lock();
    defer mutex.unlock();
    epoch += 1;
    phase_label = label;
    banner();
    var s1: [24]u8 = undefined;
    var s2: [24]u8 = undefined;
    log("  kl_track: phase {s} (live {s} blocks {s} B, quarantined {d})\n", .{ label, fmtU(&s1, live_blocks), fmtU(&s2, live_bytes), quar_blocks });
}

fn sweepBlock(h: *const Header, rec_in: ?*const Rec, printed: *usize) bool {
    if (h.magic != magic) {
        if (printed.* < report_max) {
            report(@intFromPtr(h), "header magic gone", rec_in, h, null);
            printed.* += 1;
        }
        return true;
    }
    var broke: ?usize = null;
    for (0..h.redz) |i| if (redHead(h)[i] != red) {
        broke = i;
        break;
    };
    if (broke == null) for (0..h.redz) |i| if (redTail(h)[i] != red) {
        broke = 1000 + i;
        break;
    };
    if (broke != null) {
        if (printed.* < report_max) {
            const base_addr = @intFromPtr(h) + @sizeOf(Header) + redwords * 4;
            report(base_addr, if (broke.? < 1000) "head redzone written" else "tail redzone written (buffer overrun)", rec_in, h, broke);
            printed.* += 1;
        }
        return true;
    }
    if (h.freed) {
        const user: [*]const u8 = @ptrFromInt(@intFromPtr(h) + @sizeOf(Header) + redwords * 4);
        var off: ?usize = null;
        for (user[0..h.size], 0..) |b, k| if (b != poison) {
            off = k;
            break;
        };
        if (off) |k| {
            if (printed.* < report_max) {
                report(@intFromPtr(user), "QUARANTINED (freed) memory was WRITTEN after free — the DMA-into-freed-block signature", rec_in, h, k);
                printed.* += 1;
            }
            return true;
        }
    }
    return false;
}

/// Sweeps live redzones + quarantined poison; prints up to report_max offenders.
/// The phase-boundary checkpoint: the first verify that reports a moved poison or
/// a dead magic names the phase that produced the corrupter.
pub fn verify(label: []const u8) void {
    if (!tracking) return;
    banner();
    mutex.lock();
    defer mutex.unlock();
    var printed: usize = 0;
    var bad: usize = 0;
    for (&recs) |*rec| {
        if (!rec.used) continue;
        const h = hdrOf(@ptrFromInt(rec.ptr));
        if (sweepBlock(h, rec, &printed)) bad += 1;
    }
    var it = quar_head;
    while (it) |h| : (it = h.quar_next) {
        if (sweepBlock(h, null, &printed)) bad += 1;
    }
    var s1: [24]u8 = undefined;
    var s2: [24]u8 = undefined;
    if (bad == 0) {
        log("  kl_track: verify [{s}] CLEAN — {d} live + {d} quarantined checked, {s} B live, {s} B quarantined\n", .{
            label, live_blocks, quar_blocks, fmtU(&s1, live_bytes), fmtU(&s2, quar_bytes),
        });
    } else {
        log("  kl_track: verify [{s}] FOUND {d} corrupted blocks\n", .{ label, bad });
    }
    if (capacity_warned) puts("  kl_track: NOTE registry/quarantine capacity reached — later blocks untracked\n");
}

/// Leak/instance audit + final sweep; call on the clean exit path. A clean audit
/// followed by a segfault outside these lines localizes the exit crash to the
/// Level Zero teardown, not to Zig-allocated memory.
pub fn finish() void {
    if (!tracking) return;
    verify("finish");
    mutex.lock();
    defer mutex.unlock();
    log("  kl_track: instance audit — {d} live blocks / {d} B still held at exit (leaks), {d} blocks / {d} B quarantined\n", .{
        live_blocks, live_bytes, quar_blocks, quar_bytes,
    });
    for (&recs) |*rec| {
        if (!rec.used) continue;
        var name: [64]u8 = undefined;
        log("    LEAK {d}B [{s}] first frame {s}\n", .{ rec.size, rec.site, frameName(rec.alloc_bt[0], &name) });
        dumpBt("alloc", &rec.alloc_bt, @min(4, rec.n_alloc));
    }
    for (regions[0..n_region]) |r| {
        var hx: [16]u8 = undefined;
        hexBuf(&hx, r.base);
        var s1: [24]u8 = undefined;
        log("    region {s} base 0x{s} len {s}B\n", .{ r.kind, hx[0..16], fmtU(&s1, r.len) });
    }
    dumpMaps();
}

var mapbuf: [1 << 19]u8 = undefined; // /proc/self/maps of an L0 process runs to hundreds of KB

fn dumpMaps() void {
    const maps_path = "/proc/self/maps";
    const fd = open(maps_path, O_RDONLY, 0);
    if (fd < 0) return;
    defer _ = close(fd);
    var filled: usize = 0;
    while (filled < mapbuf.len) {
        const n = read(fd, mapbuf[filled..].ptr, mapbuf.len - filled);
        if (n <= 0) break;
        filled += @intCast(n);
    }
    var shown: usize = 0;
    var lines = std.mem.splitScalar(u8, mapbuf[0..filled], '\n');
    while (lines.next()) |line| {
        var size: usize = 0;
        const sized = blk: {
            const dash = std.mem.indexOfScalar(u8, line, '-') orelse break :blk false;
            const lo = std.fmt.parseInt(usize, line[0..dash], 16) catch break :blk false;
            const rest = line[dash + 1 ..];
            const sp = std.mem.indexOfScalar(u8, rest, ' ') orelse break :blk false;
            const hi = std.fmt.parseInt(usize, rest[0..sp], 16) catch break :blk false;
            if (hi <= lo) break :blk false;
            size = hi - lo;
            break :blk true;
        };
        var tag: ?[]const u8 = null;
        if (sized) for (expects[0..n_expect]) |e| {
            if (size * 2 >= e.bytes and size <= e.bytes * 2) {
                tag = e.name;
                break;
            }
        };
        const interesting = sized and ((size >= (64 << 20)) or tag != null) or
            std.mem.indexOf(u8, line, "memfd") != null or
            std.mem.indexOf(u8, line, "/dev/shm") != null;
        if (interesting) {
            if (tag) |t| log("    maps: {s}   <== suspect family: {s}\n", .{ line, t })
            else log("    maps: {s}\n", .{line});
            shown += 1;
        }
    }
    log("    maps: {d} large/suspect/memfd/shm entries shown of {d} B read\n", .{ shown, filled });
}

/// Guard-paged anonymous mapping for the streaming buffers: PROT_NONE pages seal
/// both ends, so a pread or a memcpy that runs off the end of rbuf/cbuf faults at
/// the offender instead of eating whichever malloc chunk the kernel placed next
/// door. Returns the usable payload window.
pub fn mapGuarded(bytes: usize) !struct { p: *anyopaque, base: usize, total: usize } {
    if (!tracking) @compileError("mapGuarded requires TF_KL_ALLOC_TRACK");
    const body = (bytes + 4095) & ~@as(usize, 4095);
    const total = 4096 + body + 4096;
    const base = mmap(null, total, PROT_RW, MAP_PRIVATE_ANON, -1, 0);
    if (@intFromPtr(base) == @as(usize, @bitCast(@as(isize, -1)))) return error.OutOfMemory;
    _ = mprotect(base, 4096, PROT_NONE);
    _ = mprotect(@ptrFromInt(@intFromPtr(base) + 4096 + body), 4096, PROT_NONE);
    mutex.lock();
    defer mutex.unlock();
    if (n_region < regions.len) {
        regions[n_region] = .{ .base = @intFromPtr(base), .len = total, .kind = "streaming buffer (guarded)" };
        n_region += 1;
    }
    return .{ .p = @ptrFromInt(@intFromPtr(base) + 4096), .base = @intFromPtr(base), .total = total };
}

pub fn noteRegion(kind: []const u8, base: usize, len: usize) void {
    if (!tracking) return;
    mutex.lock();
    defer mutex.unlock();
    if (n_region < regions.len) {
        regions[n_region] = .{ .base = base, .len = len, .kind = kind };
        n_region += 1;
    }
}

pub const allocator: std.mem.Allocator = .{
    .ptr = undefined,
    .vtable = &.{
        .alloc = allocImpl,
        .resize = resizeImpl,
        .remap = remapImpl,
        .free = freeImpl,
    },
};
