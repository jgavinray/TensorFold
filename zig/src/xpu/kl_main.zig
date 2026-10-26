//! xpu-eaq-kl — the KL gate (eaq-ldlq-spec-fable.md §5.5, kl-order-ruling.md):
//!
//!   xpu-eaq-kl PACK_DIR --prompts eval.i32 (--ref ref_logits.f32 | --self | --dump-ref out.f32) --report kl.json
//!            [--manifest eval.manifest.json] [--dump-logits cand.f32] [--dump-kl rows.kld]
//!            [--dump-hist hist.txt] [--cols 2048] [--block 1024] [--threads N] [--context 4096]
//!            [--expect-sha256 HEX] [--stats-selftest] [--sidecar ref.json]
//!
//!   xpu-eaq-kl --prompts eval.i32 --ref ref_logits.f32 --cand-logits cand_logits.f32 --report kl.json
//!            [--dump-kl rows.kld] [--dump-hist hist.txt]
//!
//!   xpu-eaq-kl --pair A.kld B.kld [--manifest eval.manifest.json] [--report pair.json]
//!
//! The candidate side end to end: Runner.prompt over the evaluation rows with the hidden
//! dump, the trellis head dequantized into the resident slab, the fp32 head GEMM block by
//! block, and the §5.3 statistic (mean/median/p99/max KL, standard error, top-1
//! agreement, per-domain, reverse KL, per-row spread, alignment canary, per-position
//! histogram) into kl.json. `--self` runs the whole pipeline against itself — the gate's
//! own plumbing check: every KL exactly 0, top-1 100% — and exits nonzero if it is not.
//! `--cand-logits` is the diagnostic candidate path of kl-order-ruling.md: the candidate
//! logits come from a file, the runner and head are never built, and passing the
//! reference file itself makes the ref-vs-ref identity run need no card. `--pair` reads
//! two --dump-kl dumps taken against one reference and prints the paired-difference
//! block (invariant 3). `--stats-selftest` exercises the fp64 statistic against closed
//! forms without a device.
//!
//! The reference sidecar (eaq-review/ref-tool-ruling.md): a reference the comparator reads
//! with --ref is trusted only with the sidecar xpu-eaq-ref writes beside it — JSON at
//! <ref>.json (or --sidecar PATH) carrying the sha256 of the logits and of eval.i32, the
//! shift convention and every canary value (token upload read-back, head-input cosine,
//! teacher-forced top-1 per row and aggregate, argmax mode, distinct argmax tokens,
//! top-5 across rows, NLL). The comparator re-hashes both files, re-checks every value
//! against the gate and refuses the reference — nonzero exit, no KL row read — when the
//! sidecar is missing, certifies different bytes, or records any red check.
//! Until a reference carries a passing sidecar, every KL table stays void.
const std = @import("std");
const kl = @import("kl.zig");

const usage =
    \\usage: xpu-eaq-kl PACK_DIR --prompts eval.i32 (--ref ref_logits.f32 | --self | --dump-ref out.f32)
    \\          [--report kl.json] [--manifest eval.manifest.json]
    \\          [--dump-logits cand.f32] [--dump-kl rows.kld] [--dump-hist hist.txt]
    \\          [--cols N] [--block N] [--threads N] [--context N] [--expect-sha256 HEX]
    \\          [--stats-selftest] [--probe] [--sidecar PATH]
    \\  or:   xpu-eaq-kl --prompts eval.i32 --ref ref.f32 --cand-logits cand.f32
    \\          [--report kl.json] [--dump-kl rows.kld] [--dump-hist hist.txt]
    \\  or:   xpu-eaq-kl --pair A.kld B.kld [--manifest eval.manifest.json] [--report pair.json]
    \\  every --ref needs the passing sidecar xpu-eaq-ref writes beside it (<ref>.json,
    \\  or PATH via --sidecar); a reference without it is refused before any KL row is read
    \\
;

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len < 2) {
        std.debug.print(usage, .{});
        std.process.exit(2);
    }
    var o = kl.Options{ .pack = "", .prompts = "" };
    var i: usize = 1;
    var stats_test = false;
    var probe = false;
    var pair_a: []const u8 = "";
    var pair_b: []const u8 = "";
    var sidecar_path: []const u8 = "";
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--prompts")) {
            i += 1;
            o.prompts = arg(args, i, a);
        } else if (std.mem.eql(u8, a, "--ref")) {
            i += 1;
            o.ref = arg(args, i, a);
        } else if (std.mem.eql(u8, a, "--self")) {
            o.self_mode = true;
        } else if (std.mem.eql(u8, a, "--report")) {
            i += 1;
            o.report = arg(args, i, a);
        } else if (std.mem.eql(u8, a, "--manifest")) {
            i += 1;
            o.manifest = arg(args, i, a);
        } else if (std.mem.eql(u8, a, "--dump-logits")) {
            i += 1;
            o.dump_logits = arg(args, i, a);
        } else if (std.mem.eql(u8, a, "--dump-ref")) {
            i += 1;
            o.dump_ref = arg(args, i, a);
        } else if (std.mem.eql(u8, a, "--cand-logits")) {
            i += 1;
            o.cand_logits = arg(args, i, a);
        } else if (std.mem.eql(u8, a, "--dump-kl")) {
            i += 1;
            o.dump_kl = arg(args, i, a);
        } else if (std.mem.eql(u8, a, "--dump-hist")) {
            i += 1;
            o.dump_hist = arg(args, i, a);
        } else if (std.mem.eql(u8, a, "--pair")) {
            pair_a = arg(args, i + 1, a);
            pair_b = arg(args, i + 2, a);
            i += 2;
        } else if (std.mem.eql(u8, a, "--cols")) {
            i += 1;
            o.cols = std.fmt.parseInt(usize, arg(args, i, a), 10) catch die("bad --cols");
        } else if (std.mem.eql(u8, a, "--block")) {
            i += 1;
            o.block = std.fmt.parseInt(usize, arg(args, i, a), 10) catch die("bad --block");
        } else if (std.mem.eql(u8, a, "--threads")) {
            i += 1;
            o.threads = std.fmt.parseInt(usize, arg(args, i, a), 10) catch die("bad --threads");
        } else if (std.mem.eql(u8, a, "--context")) {
            i += 1;
            o.context = std.fmt.parseInt(u32, arg(args, i, a), 10) catch die("bad --context");
        } else if (std.mem.eql(u8, a, "--expect-sha256")) {
            i += 1;
            o.expect_sha256 = arg(args, i, a);
        } else if (std.mem.eql(u8, a, "--stats-selftest")) {
            stats_test = true;
        } else if (std.mem.eql(u8, a, "--probe")) {
            probe = true;
        } else if (std.mem.eql(u8, a, "--sidecar")) {
            i += 1;
            sidecar_path = arg(args, i, a);
        } else if (a.len > 0 and a[0] != '-') {
            if (o.pack.len != 0) die("two pack directories");
            o.pack = a;
        } else die("unknown argument");
    }
    if (stats_test) {
        kl.stats.selfCheck() catch |e| {
            std.debug.print("stats self-check: FAIL ({s})\n", .{@errorName(e)});
            std.process.exit(1);
        };
        std.debug.print("stats self-check: PASS (closed forms, identity, aggregates)\n", .{});
        return;
    }
    if (probe) {
        if (o.pack.len == 0) die("--probe needs a pack directory");
        const res = kl.probePack(o.pack, o.context) catch |e| {
            std.debug.print("xpu-eaq-kl: probe failed: {s}\n", .{@errorName(e)});
            std.process.exit(1);
        };
        const pr = res[0];
        const ab = res[1];
        std.debug.print(
            \\probe: slab {d}/{d} nonzero max {d:.6} | oracle |Δ|max {d:.6} rms {d:.6} ({s}) | gemm |Δ|max {d:.6} on |ref| max {d:.6} ({s})
            \\
        , .{ pr.slab_nonzero, pr.slab_total, pr.slab_max, pr.oracle_max_abs, pr.oracle_rms,
            if (pr.dequant_ok) "ok" else "BAD", pr.gemm_max_abs, pr.gemm_max_ref,
            if (pr.gemm_ok) "ok" else "BAD" });
        std.debug.print(
            \\probe AB: m1024 |Δ|max {d:.6} viol {d} zerorows {d} | m128 |Δ|max {d:.6} viol {d} | dumps A {d}/{d} B {d}/{d} | store16 |Δ|max {d:.6} zerorows {d} | kselect zero {d} mix {d} ({s})
            \\probe AB raw: x=8 dropped {d} zeroed {d} viol {d} bad-groups {d}/8
            \\
        , .{ ab.oracle_max_abs_m1024, ab.viol_m1024, ab.zero_rows_m1024,
            ab.oracle_max_abs_m128, ab.viol_m128,
            ab.a_dump_blocks - ab.a_dump_bad, ab.a_dump_blocks, ab.b_dump_blocks - ab.b_dump_bad, ab.b_dump_blocks,
            ab.store16_max_abs, ab.store16_zero_rows, ab.ks_zero_rows, ab.ks_mix_rows,
            if (ab.ks_first_mix.len > 0) ab.ks_first_mix else "none",
            ab.raw1024_dropped_rows, ab.raw1024_zero_rows, ab.raw1024_viol, ab.raw1024_bad_groups });
        std.process.exit(if (pr.dequant_ok and pr.gemm_ok and ab.ok) 0 else 1);
    }
    if (pair_a.len > 0) {
        _ = kl.pairRun(pair_a, pair_b, o.manifest, o.report) catch |e| {
            std.debug.print("xpu-eaq-kl: pair failed: {s}\n", .{@errorName(e)});
            std.process.exit(1);
        };
        return;
    }
    if (o.prompts.len == 0) die("needs --prompts");
    if (o.pack.len == 0 and o.cand_logits == null) die("needs a pack directory or --cand-logits");
    // The reference sidecar (eaq-review/ref-tool-ruling.md): no sidecar, no KL table.
    // --self and --dump-ref read no external reference; every --ref path pays the check.
    if (o.ref != null and !o.self_mode) requireRefSidecar(gpa, o.ref.?, o.prompts, sidecar_path);
    _ = kl.run(o) catch |e| {
        std.debug.print("xpu-eaq-kl: {s}\n", .{@errorName(e)});
        std.process.exit(if (e == kl.Error.SelfCheck) 1 else 1);
    };
}

fn arg(args: []const [:0]const u8, i: usize, opt: []const u8) []const u8 {
    if (i >= args.len) {
        std.debug.print("xpu-eaq-kl: {s} needs a value\n", .{opt});
        std.process.exit(2);
    }
    return args[i];
}

fn die(why: []const u8) noreturn {
    std.debug.print("xpu-eaq-kl: {s}\n", .{why});
    std.process.exit(2);
}

// ---------------------------------------------------------------- the reference sidecar
//
// eaq-review/ref-tool-ruling.md: the comparator refuses a reference that lacks a passing
// sidecar. xpu-eaq-ref writes <out>.json beside the logits: the sha256 of the logits and
// of the prompt file, the shift convention, and every canary value — the token-upload
// read-back, the head-input cosine, the teacher-forced top-1 per row and aggregate, the
// argmax mode, the distinct argmax token count, the top-5 across rows and the NLL. This
// re-hashes both files against the sidecar, re-checks every recorded value against the
// gate thresholds and names the first red line; a reference is accepted only when all of
// them are green. The fields default to values that fail their own check, so a missing
// field is a refusal, not a pass.

extern "c" fn open(path: [*:0]const u8, flags: c_int, mode: c_int) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn pread(fd: c_int, buf: [*]u8, count: usize, off: i64) isize;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;

/// The [positions][VOCAB] width the gate's logits are defined on.
const RefVocab: u32 = 248320;

/// The sidecar's gate-critical view; unknown fields (identity, per-row TFLOPS, top-5
/// sets, dump hashes) are ignored here and enforced by whoever reads them.
const RefSidecar = struct {
    tool: []const u8 = "",
    shift: []const u8 = "",
    logits_sha256: []const u8 = "",
    prompts_sha256: []const u8 = "",
    rows: u32 = 0,
    cols: u32 = 0,
    positions: u64 = 0,
    vocab: u32 = 0,
    ids_dev_match: bool = false,
    tokens_floor: u32 = 0,
    tokens_distinct_min: u32 = 0,
    head_input_cosine_max: f64 = 1,
    canary_row: []const f64 = &.{},
    canary_agg: f64 = 0,
    nll_row: []const f64 = &.{},
    nll_mean: f64 = 1e9,
    argmax_mode_frac: f64 = 1,
    distinct_argmax: usize = 0,
    top5_constant: bool = true,
    pass: bool = false,
};

/// Load and enforce the sidecar beside `ref_path`; refuse (exit 1) on the first red check.
fn requireRefSidecar(gpa: std.mem.Allocator, ref_path: []const u8, prompts_path: []const u8, sidecar_override: []const u8) void {
    const spath = if (sidecar_override.len != 0) sidecar_override else
        std.fmt.allocPrint(gpa, "{s}.json", .{ref_path}) catch refuse(ref_path, "out of memory naming the sidecar path", .{});
    const raw = readSmall(gpa, spath, 4 << 20) orelse refuse(ref_path,
        "no sidecar {s}: a reference is trusted only with the sidecar xpu-eaq-ref writes (sha256 of the logits and of the prompt file, the shift convention, every canary) — regenerate the reference or pass --sidecar PATH", .{spath});
    const parsed = std.json.parseFromSlice(RefSidecar, gpa, raw, .{ .ignore_unknown_fields = true }) catch
        refuse(ref_path, "sidecar {s} is not valid JSON", .{spath});
    const sc = parsed.value;

    if (!sc.pass) refuse(ref_path, "sidecar {s} does not certify the reference (pass is false or missing)", .{spath});
    if (sc.vocab != RefVocab or sc.positions == 0 or sc.positions != @as(u64, sc.rows) * sc.cols)
        refuse(ref_path, "sidecar {s} is self-inconsistent: rows {d} cols {d} positions {d} vocab {d} (expected positions = rows x cols, vocab {d})", .{
            spath, sc.rows, sc.cols, sc.positions, sc.vocab, RefVocab });
    const size = fileSize(ref_path) orelse refuse(ref_path, "cannot stat the reference file", .{});
    if (size != sc.positions * sc.vocab * 4)
        refuse(ref_path, "sidecar {s} certifies {d} bytes of fp32 logits; {s} holds {d} bytes", .{ spath, sc.positions * sc.vocab * 4, ref_path, size });
    const refhex = kl.sha256Hex(ref_path) catch refuse(ref_path, "cannot hash {s}", .{ref_path});
    if (sc.logits_sha256.len == 0 or !std.mem.eql(u8, sc.logits_sha256, refhex))
        refuse(ref_path, "logits sha256 mismatch: sidecar {s} says {s}, the file on disk hashes to {s}", .{ spath, sc.logits_sha256, refhex });
    if (prompts_path.len != 0) {
        const prompthex = kl.sha256Hex(prompts_path) catch refuse(ref_path, "cannot hash the prompt file {s}", .{prompts_path});
        if (sc.prompts_sha256.len == 0 or !std.mem.eql(u8, sc.prompts_sha256, prompthex))
            refuse(ref_path, "prompt sha256 mismatch: sidecar {s} says {s}, {s} hashes to {s}", .{ spath, sc.prompts_sha256, prompts_path, prompthex });
    }
    if (std.mem.indexOf(u8, sc.shift, "logits[t] predicts token[t+1]") == null)
        refuse(ref_path, "sidecar {s} does not state the shift convention (expected logits[t] predicts token[t+1]; recorded '{s}')", .{ spath, sc.shift });
    if (!sc.ids_dev_match)
        refuse(ref_path, "sidecar {s} does not certify the token upload (ids_dev_match is false or missing) — the copy-complete assert did not pass when the reference ran", .{spath});
    if (sc.tokens_floor == 0 or sc.tokens_distinct_min <= sc.tokens_floor)
        refuse(ref_path, "sidecar {s} fails the tokens-vary floor: min distinct {d} over floor {d}", .{ spath, sc.tokens_distinct_min, sc.tokens_floor });
    if (sc.head_input_cosine_max >= 0.99)
        refuse(ref_path, "sidecar {s}: head input is frozen — max cosine(first 64, their mean) {d:.4} >= 0.99", .{ spath, sc.head_input_cosine_max });
    if (sc.canary_row.len != sc.rows)
        refuse(ref_path, "sidecar {s}: canary_row holds {d} values, the dump has {d} rows", .{ spath, sc.canary_row.len, sc.rows });
    if (sc.canary_agg < 0.45 or sc.canary_agg > 0.85)
        refuse(ref_path, "sidecar {s}: aggregate teacher-forced top-1 {d:.4} outside the 0.45-0.85 canary band", .{ spath, sc.canary_agg });
    for (sc.canary_row, 0..) |v, r| {
        if (v < 0.45 or v > 0.85) refuse(ref_path, "sidecar {s}: row {d} teacher-forced top-1 {d:.4} outside the 0.45-0.85 canary band", .{ spath, r, v });
    }
    if (sc.nll_row.len != sc.rows)
        refuse(ref_path, "sidecar {s}: nll_row holds {d} values, the dump has {d} rows", .{ spath, sc.nll_row.len, sc.rows });
    if (sc.nll_mean >= 4.0)
        refuse(ref_path, "sidecar {s}: mean teacher-forced NLL {d:.3} >= 4 nats", .{ spath, sc.nll_mean });
    for (sc.nll_row, 0..) |v, r| {
        if (v > 6.0) refuse(ref_path, "sidecar {s}: row {d} mean teacher-forced NLL {d:.3} > 6 nats", .{ spath, r, v });
    }
    if (sc.argmax_mode_frac >= 0.10)
        refuse(ref_path, "sidecar {s}: argmax mode covers {d:.4} of positions >= 0.10 (no-invariance)", .{ spath, sc.argmax_mode_frac });
    if (sc.distinct_argmax <= 2000)
        refuse(ref_path, "sidecar {s}: only {d} distinct argmax tokens (need more than 2,000)", .{ spath, sc.distinct_argmax });
    if (sc.top5_constant)
        refuse(ref_path, "sidecar {s}: the top-5 argmax set is constant across rows", .{spath});
    std.debug.print(
        "xpu-eaq-kl: reference {s} accepted — sidecar {s}: canary agg {d:.4} (rows {d}), argmax mode {d:.4}, distinct argmax {d}, NLL mean {d:.3}, head-input cos {d:.4}, shift [{s}]\n",
        .{ ref_path, spath, sc.canary_agg, sc.rows, sc.argmax_mode_frac, sc.distinct_argmax, sc.nll_mean, sc.head_input_cosine_max, sc.shift });
}

/// Whole file into memory for the small sidecar JSON (never the 40 GB logits).
fn readSmall(gpa: std.mem.Allocator, path: []const u8, limit: usize) ?[]u8 {
    var stack: [4096]u8 = undefined;
    if (path.len == 0 or path.len >= stack.len) return null;
    @memcpy(stack[0..path.len], path);
    stack[path.len] = 0;
    const fd = open(@ptrCast(&stack), 0, 0);
    if (fd < 0) return null;
    defer _ = close(fd);
    const size: usize = @intCast(lseek(fd, 0, 2));
    if (size == 0 or size > limit) return null;
    const buf = gpa.alloc(u8, size) catch return null;
    var got: usize = 0;
    while (got < size) {
        const r = pread(fd, buf.ptr + got, size - got, @as(i64, @intCast(got)));
        if (r <= 0) return null;
        got += @intCast(r);
    }
    return buf;
}

fn fileSize(path: []const u8) ?u64 {
    var stack: [4096]u8 = undefined;
    if (path.len == 0 or path.len >= stack.len) return null;
    @memcpy(stack[0..path.len], path);
    stack[path.len] = 0;
    const fd = open(@ptrCast(&stack), 0, 0);
    if (fd < 0) return null;
    defer _ = close(fd);
    const sz = lseek(fd, 0, 2);
    if (sz < 0) return null;
    return @intCast(sz);
}

/// The gate's refusal: name the red check, exit nonzero, read no KL row.
fn refuse(ref_path: []const u8, comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("xpu-eaq-kl: refusing reference {s}: " ++ fmt ++ "\n", .{ref_path} ++ args);
    std.process.exit(1);
}
