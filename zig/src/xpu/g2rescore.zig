//! g2rescore: re-score every trellis part of a published EAQ pack under the FINAL
//! part-relative G2 rule — fable's (c) of the down-band ruling (eaq-review/g2-downband-
//! ruling-fable.md, 2026-10-28; the binding code is xpu/quant.zig probeSlot :2488-2510,
//! binary lineage a302b7ee). Host-only: the card is never opened.
//!
//! For each trellis part (400 body + lm_head = 401) the pack's own .trellis/.suh/.svh
//! bytes decode through core/exl3_format.dequantize in 128-row strips — the host oracle
//! of the serving kernel exl3_dequant128_c (bit-identical per enc T9 / T-HS-7), the same
//! path T-LDLQ-5's DqCtx/dqChunk and the fragcheck referee use — and score against the
//! dense source exactly as ProbeJob.run does (quant.zig:894-977): column-major source
//! read, E = |decode - source bf16| / |svh[col]|, dead columns (colRmsWin predicate,
//! quant.zig:1761/1830) skipped from both the sums and the denominators. Then the final
//! rule's quantities: j0 = the LAST 16-row block (kblks-1), the whole-part RMS, the
//! rank from the top, and top_allow = max(2, ceil(0.005*kblks)).
//!
//! The quant-time record — j0 err RMS, corr, verdict and the gate version in force at
//! the scoring — is parsed from the run log (the per-part T-LDLQ-5 prints). lm_head was
//! never probed (quant.zig publish: "the head is not probed"), so it carries no record;
//! the re-score gives it its first G2 verdict under the final rule.
//!
//! Usage: xpu-eaq-rescore-g2 PACK_DIR MODEL_DIR RECORDS_LOG OUT_JSON [THREADS]

const std = @import("std");
const core = @import("core");
const q = @import("qwen27");
const devm = q.dev;
const Checkpoint = devm.Checkpoint;
const efmt = core.exl3_format;

const alloc = std.heap.c_allocator;

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;
extern "c" fn munmap(addr: *anyopaque, len: usize) c_int;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;
extern "c" fn mkdir(path: [*:0]const u8, mode: c_uint) c_int;
extern "c" fn time(?*i64) i64;
const PROT_READ = 1;
const MAP_PRIVATE = 2;

const MappedFile = struct {
    bytes: []const u8,
    pub fn openFile(path: []const u8) !MappedFile {
        const sent = try alloc.dupeSentinel(u8, path, 0);
        defer alloc.free(sent);
        const fd = open(sent.ptr, 0);
        if (fd < 0) return error.FileNotFound;
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
    var mf = try MappedFile.openFile(path);
    defer mf.unmap();
    return alloc.dupe(u8, mf.bytes);
}

fn writeFile(path: []const u8, bytes: []const u8) !void {
    const sent = try alloc.dupeSentinel(u8, path, 0);
    defer alloc.free(sent);
    const fd = open(sent.ptr, @as(c_int, 65 | 512), @as(c_int, 0o644)); // O_CREAT | O_WRONLY | O_TRUNC
    if (fd < 0) return error.FileNotFound;
    defer _ = close(fd);
    var done: usize = 0;
    while (done < bytes.len) {
        const n = write(fd, bytes.ptr + done, bytes.len - done);
        if (n <= 0) return error.FileNotFound;
        done += @intCast(n);
    }
}

// ---------------------------------------------------------------- names

/// Source prefix -> pack base (quant.zig packName: the pack ships the stock names).
fn packBase(prefix: []const u8) ![]u8 {
    const src = "model.language_model.layers.";
    if (std.mem.startsWith(u8, prefix, src)) return std.fmt.allocPrint(alloc, "model.layers.{s}", .{prefix[src.len..]});
    return alloc.dupe(u8, prefix); // lm_head keeps its name
}

fn findTensor(ck: *const Checkpoint, want: []const u8) !devm.Tensor {
    for (ck.tensors.items) |t| if (std.mem.eql(u8, t.name, want)) return t;
    std.debug.print("rescore: no tensor '{s}'\n", .{want});
    return error.MissingTensor;
}

// ---------------------------------------------------------------- the run-log records

const Rec = struct {
    scored_runs: usize = 0,
    ts: []const u8 = "",
    verdict: []const u8 = "",
    j0: usize = 0,
    j0_rms: f64 = 0,
    corr: f64 = 0,
    bounds: []const u8 = "",
    version: []const u8 = "unknown",
    joint_amber: bool = false,
    has_rel: bool = false,
    rel_part_rms: f64 = 0,
    rel_ratio: f64 = 0,
    rel_rank: usize = 0,
    rel_kblks: usize = 0,
    rel_top_allow: usize = 0,
};

fn tokAt(line: []const u8, at: usize) []const u8 {
    var j = at;
    while (j < line.len and line[j] != ' ' and line[j] != ';' and line[j] != ',' and line[j] != ')') : (j += 1) {}
    return line[at..j];
}

fn numAfter(line: []const u8, anchor: []const u8) ?f64 {
    const i = std.mem.indexOf(u8, line, anchor) orelse return null;
    return std.fmt.parseFloat(f64, tokAt(line, i + anchor.len)) catch null;
}

fn intAfter(line: []const u8, anchor: []const u8) ?usize {
    const i = std.mem.indexOf(u8, line, anchor) orelse return null;
    return std.fmt.parseUnsigned(usize, tokAt(line, i + anchor.len), 10) catch null;
}

/// The gate version in force at this print, from the line's own markers (the ruling
/// lineage: absolute 0.16 -> head-strip 0.20 (e874289d) -> column gate (ceb7db93) ->
/// part-relative final (a302b7ee), g2-*-ruling-fable.md).
fn classify(line: []const u8, base: []const u8) []const u8 {
    if (std.mem.indexOf(u8, line, "part-relatively") != null) return "g2-part-relative-final-2026-10-28";
    if (std.mem.indexOf(u8, line, "window-gated") != null) return "g2-column-gate-ceb7db93-2026-10-27";
    const strip = std.mem.endsWith(u8, base, "o_proj") or std.mem.endsWith(u8, base, "out_proj");
    if (strip and std.mem.indexOf(u8, line, "bounds [0.10,0.20]") != null) return "g2-head-strip-0.20-e874289d-2026-10-27";
    if (std.mem.indexOf(u8, line, "bounds [0.10,0.16]") != null) return "g2-absolute-0.16";
    return "unknown";
}

fn parseLog(text: []const u8) !std.StringHashMap(Rec) {
    var map = std.StringHashMap(Rec).init(alloc);
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        const p = std.mem.indexOf(u8, line, "T-LDLQ-5 ") orelse continue;
        const after = line[p + "T-LDLQ-5 ".len ..];
        const c = std.mem.indexOf(u8, after, ": ") orelse continue;
        const base = after[0..c];
        const body = after[c + 2 ..];
        var ts: []const u8 = "";
        if (line.len > 1 and line[0] == '[') {
            if (std.mem.indexOfScalar(u8, line, ']')) |e| ts = line[1..e];
        }
        var rec = map.get(base) orelse Rec{};
        if (std.mem.startsWith(u8, body, "G2 feedback-free block")) {
            rec.scored_runs += 1;
            rec.j0 = intAfter(body, "block ") orelse 0;
            rec.j0_rms = numAfter(body, "err RMS ") orelse 0;
            rec.corr = numAfter(body, "corr(pack, source) ") orelse 0;
            if (std.mem.indexOf(u8, body, "part-relatively") != null) {
                rec.bounds = "part-relative";
            } else if (std.mem.indexOf(u8, body, "bounds [")) |b| {
                const s = b + "bounds ".len;
                const e = std.mem.indexOfScalar(u8, body[s..], ']') orelse 0;
                rec.bounds = body[s .. s + e];
            } else rec.bounds = "";
            rec.version = classify(body, base);
            if (ts.len != 0) rec.ts = ts;
        } else if (std.mem.startsWith(u8, body, "G2 relative:")) {
            rec.has_rel = true;
            rec.rel_part_rms = numAfter(body, "whole-part err RMS ") orelse 0;
            rec.rel_ratio = numAfter(body, "ratio ") orelse 0;
            rec.rel_rank = intAfter(body, "rank from top ") orelse 0;
            rec.rel_kblks = intAfter(body, " of ") orelse 0;
            rec.rel_top_allow = intAfter(body, "worst ") orelse 0;
            if (ts.len != 0) rec.ts = ts;
        } else if (std.mem.startsWith(u8, body, "pack ")) { // a fresh attempt: the G2 relative line prints BEFORE the headline
            rec.has_rel = false;
        } else if (std.mem.startsWith(u8, body, "GREEN")) {
            rec.verdict = "GREEN";
        } else if (std.mem.startsWith(u8, body, "RED")) {
            rec.verdict = "RED";
        } else if (std.mem.startsWith(u8, body, "G2 JOINT AMBER")) {
            rec.joint_amber = true;
        } else continue;
        try map.put(base, rec);
    }
    return map;
}

// ---------------------------------------------------------------- the re-score

/// One part's fold: the per-16-row-block error energy (live columns only) plus the
/// weight-space correlation partials — the exact quantities ProbeJob.run accumulates.
const Sum = struct {
    st: f64 = 0,
    sq: f64 = 0,
    stq: f64 = 0,
    stt: f64 = 0,
    sqq: f64 = 0,
    blk: []f64 = &.{},
};

const Job = struct {
    tr: []const i16,
    tnn: usize,
    words: usize,
    suh: []const f16,
    svh: []const f16,
    zc: []const bool,
    src: []const u8,
    k: usize,
    n: usize,
    kblks: usize,
    counter: *std.atomic.Value(usize),
    sums: []Sum,
    w: efmt.Width,
};

fn worker(job: *const Job, tid: usize) void {
    const s = &job.sums[tid];
    const out = alloc.alloc(f64, 128 * job.n) catch return;
    defer alloc.free(out);
    const scratch = alloc.alloc(f64, 2 * 128 * job.n) catch return;
    defer alloc.free(scratch);
    while (true) {
        const strip = job.counter.fetchAdd(1, .monotonic);
        if (strip >= job.k / 128) break;
        const row0 = strip * 128;
        const t = job.tr[(row0 / 16) * job.tnn * job.words ..][0 .. 8 * job.tnn * job.words];
        efmt.dequantize(job.w, .mul1, t, 8, job.tnn, job.suh[row0..][0..128], job.svh, out, scratch);
        var r: usize = 0;
        while (r < 128) : (r += 1) {
            const rr = row0 + r;
            const jb = rr / 16;
            const rowoff = r * job.n;
            var cc: usize = 0;
            while (cc < job.n) : (cc += 1) {
                if (job.zc[cc]) continue;
                const qv: f32 = @floatCast(out[rowoff + cc]);
                const idxb = (cc * job.k + rr) * 2;
                const sbits: u32 = @as(u32, job.src[idxb]) | (@as(u32, job.src[idxb + 1]) << 8);
                const sv_src: f32 = @as(f32, @bitCast(sbits << 16));
                const ef: f64 = @abs(@as(f64, qv) - @as(f64, sv_src)) / @max(@as(f64, @abs(job.svh[cc])), 1e-30);
                s.blk[jb] += ef * ef;
                s.st += sv_src;
                s.sq += qv;
                s.stq += @as(f64, sv_src) * qv;
                s.stt += @as(f64, sv_src) * sv_src;
                s.sqq += @as(f64, qv) * qv;
            }
        }
    }
}

const PartRec = struct { prefix: []const u8, k: u32, n: u32, halves: u16 };
const ReportFile = struct { parts: []const PartRec };

const Recomputed = struct {
    j0: usize,
    j0_rms: f64,
    part_rms: f64,
    ratio: f64,
    rank_from_top: usize,
    kblks: usize,
    top_allow: usize,
    n_live: usize,
    dead_cols: usize,
    corr: f64,
};

const RecordedOut = struct {
    ts: []const u8,
    verdict: []const u8,
    scored_runs: usize,
    gate_version: []const u8,
    bounds: []const u8,
    j0: usize,
    j0_rms: f64,
    corr: f64,
    joint_amber: bool,
    part_rms: ?f64,
    ratio: ?f64,
    rank_from_top: ?usize,
    kblks: ?usize,
    top_allow: ?usize,
};

const PartOut = struct {
    prefix: []const u8,
    pack_base: []const u8,
    k: u32,
    n: u32,
    halves: u16,
    gate_version_recorded: ?[]const u8,
    recorded: ?RecordedOut,
    recomputed: Recomputed,
    verdict_final: []const u8,
    reasons: []const []const u8,
    joint_amber_final: bool,
    delta_j0_rms: ?f64,
    delta_corr: ?f64,
    flip: bool,
    flip_kind: []const u8,
};

const LayerOut = struct {
    layer: []const u8,
    parts: usize,
    green: usize,
    red: usize,
    never_scored: usize,
    flips: usize,
    gate_versions: []const []const u8,
    flip_parts: []const []const u8,
};

const RuleMeta = struct {
    id: []const u8,
    ruling: []const u8,
    implementation: []const u8,
    conditions: []const u8,
    dequant_path: []const u8,
    element_form: []const u8,
    dead_cols: []const u8,
    lineage: []const u8,
};

const VersionCount = struct { version: []const u8, parts: usize };


const LayerAcc = struct {
    parts: usize = 0,
    green: usize = 0,
    red: usize = 0,
    never_scored: usize = 0,
    flips: usize = 0,
    gate_versions: std.ArrayList([]const u8) = .empty,
    flip_parts: std.ArrayList([]const u8) = .empty,
};

const Root = struct {
    tool: []const u8 = "xpu-eaq-rescore-g2",
    date: []const u8,
    pack_dir: []const u8,
    model_dir: []const u8,
    records_log: []const u8,
    gate_rule_final: RuleMeta,
    pack_manifest_sha256: []const u8,
    parts_total: usize,
    trellis_tensors_in_pack: usize,
    parts: []const PartOut,
    layers: []const LayerOut,
    summary: Summary,
};

const Summary = struct {
    parts: usize,
    final_green: usize,
    final_red: usize,
    never_scored_at_quant: usize, // the head: published without a G2 verdict
    head_verdict: []const u8,
    head_flip: bool,
    flips: usize, // parts whose final-rule verdict differs from the recorded one
    flip_parts: []const []const u8,
    layers_with_flips: usize,
    recorded_gate_versions: []const VersionCount,
    rescored_under_final_rule: usize, // parts whose accepted print used the final rule
    max_abs_delta_j0_rms: ?f64, // |recorded - recomputed| over parts with a record
    max_abs_delta_corr: ?f64,
};

fn rescorePart(pack: *const Checkpoint, src: *const Checkpoint, pr: PartRec, threads: usize) !PartOut {
    const base = try packBase(pr.prefix);
    const tr_t = try findTensor(pack, try std.fmt.allocPrint(alloc, "{s}.trellis", .{base}));
    const suh_t = try findTensor(pack, try std.fmt.allocPrint(alloc, "{s}.suh", .{base}));
    const svh_t = try findTensor(pack, try std.fmt.allocPrint(alloc, "{s}.svh", .{base}));
    const wsrc = try findTensor(src, try std.fmt.allocPrint(alloc, "{s}.weight", .{pr.prefix}));
    if (!std.mem.eql(u8, tr_t.dtype, "I16") or !std.mem.eql(u8, suh_t.dtype, "F16") or !std.mem.eql(u8, svh_t.dtype, "F16")) return error.BadDtype;
    if (!std.mem.eql(u8, wsrc.dtype, "BF16")) return error.BadSourceDtype;
    const tnk = tr_t.shape[0]; // k/16
    const tnn = tr_t.shape[1]; // n/16
    const words = tr_t.shape[2];
    const halves: u8 = @intCast(words / 8);
    const k = tnk * 16;
    const n = tnn * 16;
    if (k != pr.k or n != pr.n or halves != pr.halves) return error.ShapeMismatch;
    const w: efmt.Width = .{ .halves = halves };
    if (words != w.tileWords()) return error.BadDims;
    if (tr_t.bytes.len != tnk * tnn * words * 2) return error.BadBytes;
    if (suh_t.bytes.len != k * 2 or svh_t.bytes.len != n * 2) return error.BadBytes;
    if (wsrc.bytes.len != k * n * 2) return error.BadSource;
    if (k % 128 != 0) return error.BadDims;
    const tr: []const i16 = @alignCast(std.mem.bytesAsSlice(i16, tr_t.bytes));
    const suh: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, suh_t.bytes));
    const svh: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, svh_t.bytes));

    // the capture's zero-column mask, recomputed from the source (colRmsWin + the S1
    // dead-column predicate, quant.zig:1761/1830)
    const zc = try alloc.alloc(bool, n);
    defer alloc.free(zc);
    const src_bytes = wsrc.bytes;
    var dead: usize = 0;
    for (0..n) |cc| {
        var sum: f64 = 0;
        for (0..k) |kk| {
            const idx = (cc * k + kk) * 2;
            const bits: u32 = @as(u32, src_bytes[idx]) | (@as(u32, src_bytes[idx + 1]) << 8);
            const v: f64 = @as(f64, @floatCast(@as(f32, @bitCast(bits << 16))));
            sum += v * v;
        }
        const sj: f64 = @as(f64, @as(f32, @floatCast(@sqrt(sum / @as(f64, @floatFromInt(k))))));
        zc[cc] = !std.math.isFinite(sj) or !(sj > 1e-30);
        if (zc[cc]) dead += 1;
    }
    const n_live = n - dead;
    if (n_live == 0) return error.AllColumnsDead;

    const kblks = k / 16;
    const sums = try alloc.alloc(Sum, threads);
    defer alloc.free(sums);
    for (sums) |*sm| {
        sm.* = .{ .blk = try alloc.alloc(f64, kblks) };
        @memset(sm.blk, 0);
    }
    defer for (sums) |sm| alloc.free(sm.blk);

    var counter = std.atomic.Value(usize).init(0);
    const job = Job{
        .tr = tr, .tnn = tnn, .words = words, .suh = suh, .svh = svh, .zc = zc,
        .src = src_bytes, .k = k, .n = n, .kblks = kblks,
        .counter = &counter, .sums = sums, .w = w,
    };
    var used: usize = 1;
    if (threads > 1) {
        var tids = try alloc.alloc(std.Thread, threads);
        defer alloc.free(tids);
        for (0..threads) |t| tids[t] = try std.Thread.spawn(.{}, worker, .{ &job, t });
        for (tids) |th| th.join();
        used = threads;
    } else worker(&job, 0);

    var st: f64 = 0;
    var sq: f64 = 0;
    var stq: f64 = 0;
    var stt: f64 = 0;
    var sqq: f64 = 0;
    const blk = try alloc.alloc(f64, kblks);
    defer alloc.free(blk);
    @memset(blk, 0);
    for (sums[0..used]) |sm| {
        st += sm.st;
        sq += sm.sq;
        stq += sm.stq;
        stt += sm.stt;
        sqq += sm.sqq;
        for (0..kblks) |b| blk[b] += sm.blk[b];
    }

    // the final part-relative rule (quant.zig:2488-2510)
    const j0 = kblks - 1;
    const cnt_blk: f64 = @floatFromInt(16 * n_live);
    const j0_rms = @sqrt(blk[j0] / cnt_blk);
    var part_sq: f64 = 0;
    var bl_above: usize = 0;
    for (0..kblks) |b| {
        part_sq += blk[b];
        if (blk[b] > blk[j0]) bl_above += 1;
    }
    const part_rms = @sqrt(part_sq / (@as(f64, @floatFromInt(k)) * @as(f64, @floatFromInt(n_live))));
    const top_allow = @max(@as(usize, 2), @as(usize, @intFromFloat(@ceil(0.005 * @as(f64, @floatFromInt(kblks))))));
    const nn: f64 = @floatFromInt(k * n_live);
    const cov = nn * stq - st * sq;
    const den = @sqrt(@max(nn * stt - st * st, 0.0) * @max(nn * sqq - sq * sq, 0.0));
    const corr = if (den > 1e-30) cov / den else 0.0;

    var reasons = std.ArrayList([]const u8).empty;
    if (j0_rms < 0.10) try reasons.append(alloc, "collapse-absolute(j0<0.10)");
    if (j0_rms > 0.30) try reasons.append(alloc, "blowup-absolute(j0>0.30)");
    if (part_rms > 0.30) try reasons.append(alloc, "part-collapse(part>0.30)");
    if (j0_rms < 0.50 * part_rms) try reasons.append(alloc, "collapse-relative(j0<0.50x-part)");
    if (j0_rms > 1.80 * part_rms) try reasons.append(alloc, "blowup-relative(j0>1.80x-part)");
    if (bl_above < top_allow) try reasons.append(alloc, "rank-top(inside-worst-tail)");
    const verdict: []const u8 = if (reasons.items.len != 0) "RED" else "GREEN";
    const strip = std.mem.endsWith(u8, base, "o_proj") or std.mem.endsWith(u8, base, "out_proj");

    return PartOut{
        .prefix = pr.prefix,
        .pack_base = base,
        .k = pr.k,
        .n = pr.n,
        .halves = pr.halves,
        .gate_version_recorded = null,
        .recorded = null,
        .recomputed = .{
            .j0 = j0, .j0_rms = j0_rms, .part_rms = part_rms, .ratio = j0_rms / @max(part_rms, 1e-30),
            .rank_from_top = bl_above + 1, .kblks = kblks, .top_allow = top_allow,
            .n_live = n_live, .dead_cols = dead, .corr = corr,
        },
        .verdict_final = verdict,
        .reasons = try reasons.toOwnedSlice(alloc),
        .joint_amber_final = strip and j0_rms > 0.18 and corr < 0.99,
        .delta_j0_rms = null,
        .delta_corr = null,
        .flip = false,
        .flip_kind = "never-scored-at-quant",
    };
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(alloc);
    if (args.len < 5) {
        std.debug.print("usage: xpu-eaq-rescore-g2 PACK_DIR MODEL_DIR RECORDS_LOG OUT_JSON [THREADS]\n", .{});
        return error.Usage;
    }
    const pack_dir = args[1];
    const model_dir = args[2];
    const log_path = args[3];
    const out_path = args[4];
    const nproc = std.Thread.getCpuCount() catch 4;
    var threads: usize = if (args.len > 5) try std.fmt.parseInt(usize, args[5], 10) else @min(nproc, 32);
    threads = @max(@as(usize, 1), @min(threads, 64));

    // the §Q3 report inside the pack is the part list (401: 400 body + lm_head)
    const rep_text = try readFile(try std.fmt.allocPrint(alloc, "{s}/quant_report.json", .{pack_dir}));
    const rep = try std.json.parseFromSlice(ReportFile, alloc, rep_text, .{ .ignore_unknown_fields = true });

    const log_text = try readFile(log_path);
    var records = try parseLog(log_text);
    defer records.deinit();

    var pack: Checkpoint = .{};
    errdefer pack.deinit();
    try pack.openEaq(pack_dir); // the reader's verdict first: manifest, shard hashes, the Table

    // pack manifest sha for the stamp
    var man_hex: [64]u8 = undefined;
    {
        const man = try readFile(try std.fmt.allocPrint(alloc, "{s}/eaq_manifest.json", .{pack_dir}));
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update(man);
        man_hex = std.fmt.bytesToHex(h.finalResult(), .lower);
    }

    // the dense source, shard by shard: every part scores against its own shard's bytes
    const idx_text = try readFile(try std.fmt.allocPrint(alloc, "{s}/model.safetensors.index.json", .{model_dir}));
    const idx = try std.json.parseFromSlice(std.json.Value, alloc, idx_text, .{});
    const wm = idx.value.object.get("weight_map") orelse return error.BadIndex;

    var shard_names = std.StringHashMap(void).init(alloc);
    defer shard_names.deinit();
    var wmit = wm.object.iterator();
    while (wmit.next()) |e| {
        try shard_names.put(e.value_ptr.string, {});
    }

    const parts_out = try alloc.alloc(PartOut, rep.value.parts.len);
    var layer_acc = std.StringHashMap(LayerAcc).init(alloc);
    var seen_versions = std.StringHashMap(usize).init(alloc);
    var final_rescored: usize = 0;
    var max_dj: f64 = 0;
    var max_dc: f64 = 0;
    var have_dj = false;
    var head_verdict: []const u8 = "none";
    var head_flip = false;
    var never_scored: usize = 0;
    var green: usize = 0;
    var red: usize = 0;
    var flips_list = std.ArrayList([]const u8).empty;

    var srit = shard_names.keyIterator();
    while (srit.next()) |shard| {
        const sp = try std.fmt.allocPrintSentinel(alloc, "{s}/{s}", .{ model_dir, shard.* }, 0);
        defer alloc.free(sp);
        var src: Checkpoint = .{};
        defer src.deinit();
        try src.addShard(sp);
        for (rep.value.parts, 0..) |pr, pi| {
            const wname = try std.fmt.allocPrint(alloc, "{s}.weight", .{pr.prefix});
            defer alloc.free(wname);
            const ent = wm.object.get(wname) orelse continue;
            if (!std.mem.eql(u8, ent.string, shard.*)) continue;
            var po = try rescorePart(&pack, &src, pr, @min(threads, @max(pr.k / 128, 1)));
            if (std.mem.eql(u8, pr.prefix, "lm_head")) { // the head: never probed at quant
                head_verdict = po.verdict_final;
                head_flip = std.mem.eql(u8, po.verdict_final, "RED");
                never_scored = 1;
            } else if (records.get(pr.prefix)) |rec| {
                po.gate_version_recorded = rec.version;
                po.recorded = .{
                    .ts = rec.ts, .verdict = rec.verdict, .scored_runs = rec.scored_runs,
                    .gate_version = rec.version, .bounds = rec.bounds, .j0 = rec.j0,
                    .j0_rms = rec.j0_rms, .corr = rec.corr, .joint_amber = rec.joint_amber,
                    .part_rms = if (rec.has_rel) rec.rel_part_rms else null,
                    .ratio = if (rec.has_rel) rec.rel_ratio else null,
                    .rank_from_top = if (rec.has_rel) rec.rel_rank else null,
                    .kblks = if (rec.has_rel) rec.rel_kblks else null,
                    .top_allow = if (rec.has_rel) rec.rel_top_allow else null,
                };
                po.delta_j0_rms = po.recomputed.j0_rms - rec.j0_rms;
                po.delta_corr = po.recomputed.corr - rec.corr;
                if (!have_dj or @abs(po.delta_j0_rms.?) > @abs(max_dj)) { max_dj = po.delta_j0_rms.?; have_dj = true; }
                if (!have_dj or @abs(po.delta_corr.?) > @abs(max_dc)) max_dc = po.delta_corr.?;
                po.flip = !std.mem.eql(u8, rec.verdict, po.verdict_final);
                po.flip_kind = if (po.flip) "verdict-flip" else "same";
                if (std.mem.eql(u8, rec.version, "g2-part-relative-final-2026-10-28")) final_rescored += 1;
                const vc = try seen_versions.getOrPut(rec.version);
                if (!vc.found_existing) vc.value_ptr.* = 0;
                vc.value_ptr.* += 1;
            }
            if (std.mem.eql(u8, po.verdict_final, "RED")) red += 1 else green += 1;
            if (po.flip) try flips_list.append(alloc, po.prefix);
            // the layer accumulator: 49 / 63 / head
            var lkey: []const u8 = "head";
            if (std.mem.indexOf(u8, pr.prefix, ".layers.")) |lp| {
                const digits = pr.prefix[lp + ".layers.".len ..];
                const e = std.mem.indexOfNone(u8, digits, "0123456789") orelse digits.len;
                lkey = try alloc.dupe(u8, digits[0..e]);
            }
            const gop = try layer_acc.getOrPut(lkey);
            if (!gop.found_existing) gop.value_ptr.* = .{};
            const la = gop.value_ptr;
            la.parts += 1;
            if (std.mem.eql(u8, po.verdict_final, "RED")) la.red += 1 else la.green += 1;
            if (po.recorded == null) la.never_scored += 1;
            if (po.flip) {
                la.flips += 1;
                try la.flip_parts.append(alloc, po.prefix);
            }
            if (po.gate_version_recorded) |gv| {
                var have = false;
                for (la.gate_versions.items) |v| {
                    if (std.mem.eql(u8, v, gv)) {
                        have = true;
                    }
                }
                if (!have) try la.gate_versions.append(alloc, gv);
            }
            parts_out[pi] = po;
            std.debug.print("rescore: {s} k {d} n {d} halves {d} | recorded {s} {s} {d} | final {s} (j0 {d:.6} part {d:.6} ratio {d:.3} rank {d}/{d} top {d} corr {d:.4})\n", .{
                pr.prefix, pr.k, pr.n, pr.halves,
                if (po.gate_version_recorded) |v| v else "none",
                if (po.recorded) |r| r.verdict else "-",
                if (po.recorded) |r| @as(usize, @intFromFloat(r.j0_rms * 1e6)) else 0,
                po.verdict_final, po.recomputed.j0_rms, po.recomputed.part_rms, po.recomputed.ratio,
                po.recomputed.rank_from_top, po.recomputed.kblks, po.recomputed.top_allow, po.recomputed.corr,
            });
        }
    }

    var layers = std.ArrayList(LayerOut).empty;
    var lrit = layer_acc.iterator();
    while (lrit.next()) |e| {
        const a = e.value_ptr;
        try layers.append(alloc, .{
            .layer = e.key_ptr.*, .parts = a.parts, .green = a.green, .red = a.red,
            .never_scored = a.never_scored, .flips = a.flips,
            .gate_versions = a.gate_versions.items, .flip_parts = a.flip_parts.items,
        });
    }
    std.mem.sort(LayerOut, layers.items, {}, struct {
        fn lt(_: void, a: LayerOut, b: LayerOut) bool {
            const ai = std.fmt.parseInt(usize, a.layer, 10) catch std.math.maxInt(usize);
            const bi = std.fmt.parseInt(usize, b.layer, 10) catch std.math.maxInt(usize);
            return ai < bi;
        }
    }.lt);
    var layers_with_flips: usize = 0;
    for (layers.items) |l| {
        if (l.flips != 0) layers_with_flips += 1;
    }

    var vcs = std.ArrayList(VersionCount).empty;
    var vrit = seen_versions.iterator();
    while (vrit.next()) |e| try vcs.append(alloc, .{ .version = e.key_ptr.*, .parts = e.value_ptr.* });

    const date = try std.fmt.allocPrint(alloc, "{d}", .{time(null)});

    var trellis_count: usize = 0;
    for (pack.tensors.items) |t| {
        if (std.mem.endsWith(u8, t.name, ".trellis")) trellis_count += 1;
    }

    const root = Root{
        .date = date,
        .pack_dir = pack_dir,
        .model_dir = model_dir,
        .records_log = log_path,
        .gate_rule_final = .{
            .id = "g2-part-relative-final-2026-10-28",
            .ruling = "g2-downband-ruling-fable.md (c) — the absolute 0.16 (0.20 head-strip) ceilings false-gated ~35-50 pct of down parts",
            .implementation = "zig/src/xpu/quant.zig:2488-2510 (probeSlot: j0_rms, part_rms, bl_above, top_allow, g2_rel_red)",
            .conditions = "RED if j0_rms < 0.10 or j0_rms > 0.30 or part_rms > 0.30 or j0_rms < 0.50*part_rms or j0_rms > 1.80*part_rms or (rank from top of j0 among the part's own blocks) <= top_allow, top_allow = max(2, ceil(0.005*kblks)); j0 = block kblks-1 (rows k-16..k-1, the feedback-free block)",
            .dequant_path = "zig/src/core/exl3_format.zig:221 dequantize (128-row strips) — host oracle of the serving kernel exl3_dequant128_c (zig/kernels/xpu/qwen27/exl3_h.cl:175; T9 and T-HS-7 bit-identical)",
            .element_form = "zig/src/xpu/quant.zig:894-977 ProbeJob.run: E = |decode - source bf16| / |svh[col]|, column-major source read, dead columns skipped",
            .dead_cols = "zig/src/xpu/quant.zig:1761 colRmsWin + :1830 (dead when the source column RMS is not finite or <= 1e-30)",
            .lineage = "g2-absolute-0.16 -> g2-head-strip-0.20 (e874289d) -> g2-column-gate (ceb7db93) -> g2-part-relative-final (a302b7ee, binding)",
        },
        .pack_manifest_sha256 = man_hex[0..],
        .parts_total = rep.value.parts.len,
        .trellis_tensors_in_pack = trellis_count,
        .parts = parts_out,
        .layers = layers.items,
        .summary = .{
            .parts = rep.value.parts.len,
            .final_green = green,
            .final_red = red,
            .never_scored_at_quant = never_scored,
            .head_verdict = head_verdict,
            .head_flip = head_flip,
            .flips = flips_list.items.len,
            .flip_parts = flips_list.items,
            .layers_with_flips = layers_with_flips,
            .recorded_gate_versions = vcs.items,
            .rescored_under_final_rule = final_rescored,
            .max_abs_delta_j0_rms = if (have_dj) max_dj else null,
            .max_abs_delta_corr = if (have_dj) max_dc else null,
        },
    };
    const json = try std.json.Stringify.valueAlloc(alloc, root, .{});
    try writeFile(out_path, json);
    std.debug.print("rescore: wrote {s} ({d} bytes); {d} parts, {d} green / {d} red final; {d} flips over {d} layers; head (never scored): {s}{s}\n", .{
        out_path, json.len, rep.value.parts.len, green, red, flips_list.items.len, layers_with_flips,
        head_verdict, if (head_flip) " — WOULD FLIP (no quant-time gate)" else "",
    });
}
