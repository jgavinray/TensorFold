//! xpu-eaq-kl — the KL gate (eaq-ldlq-spec-fable.md §5.5):
//!
//!   xpu-eaq-kl PACK_DIR --prompts eval.i32 (--ref ref_logits.f32 | --self | --dump-ref out.f32) --report kl.json
//!            [--manifest eval.manifest.json] [--dump-logits cand.f32]
//!            [--cols 2048] [--block 1024] [--threads N] [--context 4096] [--expect-sha256 HEX]
//!            [--stats-selftest]
//!
//! The candidate side end to end: Runner.prompt over the evaluation rows with the hidden
//! dump, the trellis head dequantized into the resident slab, the fp32 head GEMM block by
//! block, and the §5.3 statistic (mean/median/p99/max KL, standard error, top-1
//! agreement, per-domain, reverse KL) into kl.json. `--self` runs the whole pipeline
//! against itself — the gate's own plumbing check: every KL exactly 0, top-1 100% — and
//! exits nonzero if it is not. `--stats-selftest` exercises the fp64 statistic against
//! closed forms without a device.
const std = @import("std");
const kl = @import("kl.zig");

const usage =
    \\usage: xpu-eaq-kl PACK_DIR --prompts eval.i32 (--ref ref_logits.f32 | --self | --dump-ref out.f32)
    \\          [--report kl.json] [--manifest eval.manifest.json] [--dump-logits cand.f32]
    \\          [--cols N] [--block N] [--threads N] [--context N] [--expect-sha256 HEX]
    \\          [--stats-selftest] [--probe]
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
            \\
        , .{ ab.oracle_max_abs_m1024, ab.viol_m1024, ab.zero_rows_m1024,
            ab.oracle_max_abs_m128, ab.viol_m128,
            ab.a_dump_blocks - ab.a_dump_bad, ab.a_dump_blocks, ab.b_dump_blocks - ab.b_dump_bad, ab.b_dump_blocks,
            ab.store16_max_abs, ab.store16_zero_rows, ab.ks_zero_rows, ab.ks_mix_rows,
            if (ab.ks_first_mix.len > 0) ab.ks_first_mix else "none" });
        std.process.exit(if (pr.dequant_ok and pr.gemm_ok and ab.ok) 0 else 1);
    }
    if (o.pack.len == 0 or o.prompts.len == 0) {
        std.debug.print(usage, .{});
        std.process.exit(2);
    }
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
