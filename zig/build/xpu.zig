//! Level Zero (Intel Arc Pro B70): mock-backed admission tests join `zig build test`; on-card runs are opt-in steps
//! (`xpu-test`, `xpu-kernel-exact`). Kernels are compiled with ocloc for device bmg; set -Docloc / -Dxpu-device to
//! override, or -Dqwen-kernels=prebuilt:<dir> to embed modules from an earlier build on a host without ocloc.

const std = @import("std");
const exlr8_build = @import("exlr8.zig");
/// The pinned Qwen 3.8 27B geometry, the source of the -D flags the 27B kernels are compiled with. Reading the
/// module here (not a copy) keeps one definition; it has no imports beyond std, so the build can host it.
const q27 = @import("../src/families/qwen27/config.zig");

/// Whether the named ocloc binary is on PATH, so the default kernel source can follow it.
fn oclocAvailable(b: *std.Build, name: []const u8) bool {
    const path = b.graph.environ_map.get("PATH") orelse return false;
    var it = std.mem.tokenizeScalar(u8, path, ':');
    while (it.next()) |dir| {
        if (dir.len == 0) continue;
        const full = b.pathJoin(&.{ dir, name });
        std.Io.Dir.accessAbsolute(std.Io.Threaded.global_single_threaded.io(), full, .{}) catch continue;
        return true;
    }
    return false;
}

/// xpu-host-test (part of `test`), xpu-test (build + run on the card), xpu-gpu-build, xpu-kernel-exact (CLI install).
/// The Qwen kernel sources under zig/kernels/xpu/shared/ carry the 2B shape values as preprocessor defaults; a 27B
/// build overrides them by passing `-Dqwen-shape-defines="-DQW_HIDDEN=5120 -D..."` (see the defines at the top of each
/// shared/<key>/<key>.cl and `zig/src/families/qwen27/config.zig`).
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step, build_options: *std.Build.Step.Options) void {
    const fixtures = b.addOptions();
    for ([_][]const u8{ "success", "failed", "missing" }) |kind| {
        const m = b.createModule(.{ .target = b.graph.host, .link_libc = true });
        m.addCSourceFile(.{
            .file = b.path("zig/tests/xpu_mock.c"),
            .flags = if (std.mem.eql(u8, kind, "missing")) &.{ "-DOMIT_DRIVERGET", "-DINIT_RESULT=0" } else if (std.mem.eql(u8, kind, "failed")) &.{"-DINIT_RESULT=1"} else &.{"-DINIT_RESULT=0"},
        });
        const lib = b.addLibrary(.{ .name = b.fmt("xpu-mock-{s}", .{kind}), .linkage = .dynamic, .root_module = m });
        fixtures.addOptionPath(kind, lib.getEmittedBin());
    }
    const host_module = b.createModule(.{
        .root_source_file = b.path("zig/src/xpu/admission_tests.zig"),
        .target = b.graph.host,
        .link_libc = true,
    });
    host_module.addOptions("xpu_fixtures", fixtures);
    const host_tests = b.addTest(.{ .root_module = host_module });
    const run_host = b.addRunArtifact(host_tests);
    test_step.dependOn(&run_host.step);
    b.step("xpu-host-test", "Level Zero admission tests without GPU work").dependOn(&run_host.step);

    const ocloc = b.option([]const u8, "ocloc", "Intel OpenCL offline compiler for the XPU test kernels") orelse "ocloc";
    const xpu_device = b.option([]const u8, "xpu-device", "ocloc device name for the XPU test kernels") orelse "bmg";
    // Extra -D flags for the shared Qwen kernels (27B: the model-shape overrides; empty = 2B defaults).
    const shape_defines = b.option([]const u8, "qwen-shape-defines", "space-separated -D flags for the shared Qwen kernels (empty = 2B defaults)") orelse "";

    // Test kernels, compiled to native binaries: <name>_<device>.bin. The path is passed to the tests as a build option.
    // `-Dtest-kernels=prebuilt[:<dir>]` embeds modules built earlier on a host without ocloc (same fallback the Qwen
    // kernels use), so the on-card runtime tests stay runnable while intel-ocloc is absent.
    const test_kernels = b.option([]const u8, "test-kernels", "ocloc | prebuilt[:<dir>]: where the runtime/dpas test kernel modules come from") orelse if (oclocAvailable(b, ocloc)) "ocloc" else "prebuilt";
    const test_prebuilt: ?[]const u8 = if (std.mem.eql(u8, test_kernels, "prebuilt"))
        "zig/kernels/xpu/prebuilt-test"
    else if (std.mem.startsWith(u8, test_kernels, "prebuilt:"))
        test_kernels["prebuilt:".len..]
    else
        null;
    if (test_prebuilt == null and !std.mem.eql(u8, test_kernels, "ocloc")) {
        std.debug.panic("test-kernels must be `ocloc`, `prebuilt` or `prebuilt:<dir>`, got {s}", .{test_kernels});
    }
    const kernels = b.addOptions();
    inline for (.{ "runtime_tests", "dpas" }) |name| {
        const file = b.fmt("{s}_{s}.bin", .{ name, xpu_device });
        if (test_prebuilt) |dir| {
            const src = b.pathJoin(&.{ dir, file });
            const wf = b.addWriteFiles();
            const copied = wf.addCopyFile(if (std.fs.path.isAbsolute(src)) b.graph.cwdRelativePath(src) else b.path(src), file);
            kernels.addOptionPath(name, copied);
        } else {
            const c = b.addSystemCommand(&.{ ocloc, "compile", "-device", xpu_device, "-options", "-cl-std=CL3.0", "-file" });
            c.addFileArg(b.path("zig/kernels/xpu/" ++ name ++ ".cl"));
            c.addArg("-out_dir");
            const out = c.addOutputDirectoryArg(name ++ "-out");
            kernels.addOptionPath(name, out.path(b, file));
        }
    }

    const gpu_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/runtime_tests.zig"), .target = target, .link_libc = true });
    gpu_module.addOptions("xpu_kernels", kernels);
    const gpu_test = b.addTest(.{ .name = "tf-xpu-test", .root_module = gpu_module });
    const install_gpu = b.addInstallArtifact(gpu_test, .{});
    b.step("xpu-gpu-build", "Build zig-out/bin/tf-xpu-test (the on-card tests) without running them").dependOn(&install_gpu.step);
    const run_gpu = b.addRunArtifact(gpu_test);
    run_gpu.step.dependOn(&install_gpu.step);
    b.step("xpu-test", "Level Zero runtime on the B70s: copies, in-order regression, DPAS probe (opt-in, needs the cards)").dependOn(&run_gpu.step);

    const cli_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/kernel_exact.zig"), .target = target, .link_libc = true });
    const cli = b.addExecutable(.{ .name = "tf-xpu-kernel-exact", .root_module = cli_module });
    const install = b.addInstallArtifact(cli, .{});
    b.step("xpu-kernel-exact", "Build tf-xpu-kernel-exact, the generic on-card kernel gate runner").dependOn(&install.step);

    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    // Phase 4.1-4.2: Qwen3.5-2B weights and state on the card (opt-in, needs a card and the checkpoint).
    const q_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_qwen35_test.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    q_module.addImport("core", core);
    const q_exe = b.addExecutable(.{ .name = "tf-xpu-qwen35-test", .root_module = q_module });
    const q_install = b.addInstallArtifact(q_exe, .{});
    const q_run = b.addRunArtifact(q_exe);
    q_run.step.dependOn(&q_install.step);
    b.step("xpu-qwen35-test", "Build and run tf-xpu-qwen35-test: Qwen3.5-2B weights + state gates on a B70 (opt-in)").dependOn(&q_run.step);

    // Phase 4.3: Qwen3.5-2B forward pass (shared OpenCL kernels, ocloc-compiled) with gates A-C (opt-in, needs a card).
    // The binaries are @embedFile'd into the module (like upstream's metallibs and hip.zig's probe.hsaco), so the
    // executable carries its own kernels and works from any working directory - no cwd-relative .zig-cache paths.
    // One source per key (zig/kernels/xpu/shared/<key>/<key>.cl) with the 2B shape values as preprocessor defaults;
    // the 27B build overrides them with -D flags (see zig/build/xpu.zig `sharedDefines`).
    //
    // The kernel modules come from one of two places:
    //   * `ocloc` (default when the binary is on PATH): compile zig/kernels/xpu/shared/<key>/<key>.cl here.
    //   * `prebuilt` (default when ocloc is absent, or `-Dqwen-kernels=prebuilt:<dir>`): @embedFile the
    //     <dir>/<key>_<device>.bin committed under zig/kernels/xpu/prebuilt/, so a host without ocloc still builds
    //     the on-card tests from modules that were compiled from the same sources.
    // Both paths produce module bytes that load on the card; nothing else in the graph changes.
    const qwen_kernels = b.option([]const u8, "qwen-kernels", "ocloc | prebuilt[:<dir>]: where the Qwen kernel modules come from") orelse if (oclocAvailable(b, ocloc)) "ocloc" else "prebuilt";
    const prebuilt: ?[]const u8 = if (std.mem.eql(u8, qwen_kernels, "prebuilt"))
        "zig/kernels/xpu/prebuilt"
    else if (std.mem.startsWith(u8, qwen_kernels, "prebuilt:"))
        qwen_kernels["prebuilt:".len..]
    else
        null;
    if (prebuilt == null and !std.mem.eql(u8, qwen_kernels, "ocloc")) {
        std.debug.panic("qwen-kernels must be `ocloc`, `prebuilt` or `prebuilt:<dir>`, got {s}", .{qwen_kernels});
    }
    if (prebuilt == null and shape_defines.len != 0) {
        std.debug.panic("qwen-kernels=ocloc cannot apply -Dqwen-shape-defines here: the 27B shapes need their own build host", .{});
    }
    const qk_files = b.addWriteFiles();
    var qk_src: []const u8 = "//! Compiled Qwen kernel binaries, embedded (see zig/build/xpu.zig).\n";
    // A 27B build supplies the model-shape -D overrides (space-separated); empty keeps the 2B defaults.
    // ocloc takes compile options as one string after -options, so the -D flags belong there, not as bare args.
    const opt_string = b.fmt("-cl-std=CL3.0 {s}", .{shape_defines});
    inline for (.{ "layout", "norm", "norm_nores", "qmm", "qmm_pair", "norm_qmm", "mlp_act", "mlp_act_down", "gdn_pre", "gdn_chain", "gdn_post", "attn_partial", "attn_merge" }) |name| {
        const file = b.fmt("{s}_{s}.bin", .{ name, xpu_device });
        if (prebuilt) |dir| {
            const src = b.pathJoin(&.{ dir, file });
            _ = qk_files.addCopyFile(if (std.fs.path.isAbsolute(src)) b.graph.cwdRelativePath(src) else b.path(src), file);
        } else {
            const c = b.addSystemCommand(&.{ ocloc, "compile", "-device", xpu_device, "-options", opt_string, "-file" });
            c.addFileArg(b.path("zig/kernels/xpu/shared/" ++ name ++ "/" ++ name ++ ".cl"));
            c.addArg("-out_dir");
            const out = c.addOutputDirectoryArg("qwen35-" ++ name ++ "-out");
            _ = qk_files.addCopyFile(out.path(b, file), file);
        }
        qk_src = b.fmt("{s}pub const {s} align(64) = @embedFile(\"{s}\").*;\n", .{ qk_src, name, file });
    }
    const qk = b.createModule(.{ .root_source_file = qk_files.add("xpu_qwen35_kernels.zig", qk_src) });
    // Phase 6: Qwen 3.8 27B. One more compilation of the same shared sources with the 27B shape defines, embedded
    // under the same module name (`xpu_qwen35_kernels`), so `families/qwen27/xpu_forward.zig` reads them by key.
    // The defines are the pinned geometry of families/qwen27/config.zig; the 2B build's empty default keeps its own.
    // QW_ZS is gdn_post.cl's in_proj_z row stride, i.e. the value projection's width QW_NV*QW_DV = v_heads*dv
    // (6144 at 27B), NOT `hidden`. The two coincide on the 2B (16*128 = 2048 = hidden), which is why the kernel's
    // default works there; at 27B the z row is 6144 wide against hidden 5120, and passing `hidden` made gdn_post
    // read each row 1024 elements early - heads 40..47 of every GDN layer consumed the next row's z (the last row
    // read past the buffer), which is exactly the row-to-row corruption the smoke gate reported.
    const q27_defines = b.fmt("-DQW_HIDDEN={d} -DQW_INTERMEDIATE={d} -DQW_HEADS={d} -DQW_Q_HEADS={d} -DQW_KV_HEADS={d} -DQW_NV={d} -DQW_HV={d} -DQW_ZS={d} -DQW_QROWS={d} -DQW_OUT_WIDTH={d}", .{
        q27.Pinned.hidden,
        q27.Pinned.intermediate,
        q27.Pinned.heads,
        q27.Pinned.heads,
        q27.Pinned.kv_heads,
        q27.Pinned.v_heads,
        q27.Pinned.v_heads,
        q27.Pinned.gdn_value_dim,
        // QW_QROWS: the `queries` buffer's row stride. The 27B forward allocates it for the whole round
        // (rows * heads * head_dim) and passes the round's row count here, so layout27.cl strides by the round
        // and dims[2] carries the segment's base. QW_OUT_WIDTH: attn_merge27.cl's OUT stride, heads*head_dim
        // (6144), against hidden 5120 - the 2B defaults (rows, 2048) coincide with their own geometry, which is
        // exactly how the shared sources' defaults would silently mis-stride the 27B.
        q27.Pinned.hidden, // rows are runtime; this value is only the compile-time fallback for a 0 argument
        2 * q27.Pinned.heads * q27.Pinned.head_dim,
    });
    // Where the 27B modules come from, same two-source rule as the 2B path above:
    //   * `ocloc` (default when the binary is on PATH): compile the shared sources with the 27B -D set here.
    //   * `prebuilt` (default when ocloc is absent, or `-Dqwen27-kernels=prebuilt:<dir>`): @embedFile the
    //     <dir>/<key>_bmg.bin committed under zig/kernels/xpu/prebuilt-27b/, so `zig build xpu-qwen27-tools`
    //     works on a host without intel-ocloc. Those modules were compiled from the same sources with these
    //     defines; the 2B-default modules under zig/kernels/xpu/prebuilt/ are NOT a substitute (they carry the
    //     -cl-std=CL3.0 defaults only, so a 27B forward linked against them would run 2B shapes).
    const q27_kernels = b.option([]const u8, "qwen27-kernels", "ocloc | prebuilt[:<dir>]: where the Qwen 3.8 27B kernel modules come from") orelse if (oclocAvailable(b, ocloc)) "ocloc" else "prebuilt";
    const q27_prebuilt: ?[]const u8 = if (std.mem.eql(u8, q27_kernels, "prebuilt"))
        "zig/kernels/xpu/prebuilt-27b"
    else if (std.mem.startsWith(u8, q27_kernels, "prebuilt:"))
        q27_kernels["prebuilt:".len..]
    else
        null;
    if (q27_prebuilt == null and !std.mem.eql(u8, q27_kernels, "ocloc")) {
        std.debug.panic("qwen27-kernels must be `ocloc`, `prebuilt` or `prebuilt:<dir>`, got {s}", .{q27_kernels});
    }
    const q27_files = b.addWriteFiles();
    var q27_src: []const u8 = "//! Compiled Qwen kernel binaries for Qwen 3.8 27B, embedded (see zig/build/xpu.zig).\n";
    const q27_opt = b.fmt("-cl-std=CL3.0 {s}", .{q27_defines});
    inline for (.{ "layout", "layout27", "norm", "norm_nores", "norm_qmm", "qmm", "qmm_pair", "mlp_act", "mlp_act_down", "gdn_pre", "gdn_chain", "gdn_post", "attn_partial", "attn_merge", "attn_merge27" }) |name| {
        const file = b.fmt("{s}_{s}.bin", .{ name, xpu_device });
        if (q27_prebuilt) |dir| {
            const src = b.pathJoin(&.{ dir, file });
            _ = q27_files.addCopyFile(if (std.fs.path.isAbsolute(src)) b.graph.cwdRelativePath(src) else b.path(src), file);
        } else {
            const cc = b.addSystemCommand(&.{ ocloc, "compile", "-device", xpu_device, "-options", q27_opt, "-file" });
            cc.addFileArg(b.path("zig/kernels/xpu/shared/" ++ name ++ "/" ++ name ++ ".cl"));
            cc.addArg("-out_dir");
            const out = cc.addOutputDirectoryArg("qwen27-" ++ name ++ "-out");
            _ = q27_files.addCopyFile(out.path(b, file), file);
        }
        q27_src = b.fmt("{s}pub const {s} align(64) = @embedFile(\"{s}\").*;\n", .{ q27_src, name, file });
    }
    const q27k = b.createModule(.{ .root_source_file = q27_files.add("xpu_qwen27_kernels.zig", q27_src) });
    const q27_mod = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_qwen27_forward.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    q27_mod.addImport("xpu_qwen27_kernels", q27k);
    // The weights module reads safetensors through the `core` module (its file lives outside this module's root).
    q27_mod.addImport("core", core);
    // Phase 6.3: the two 27B tools. tf-xpu-qwen27-logits writes len(tokens) x vocab bf16 logits; tf-xpu-qwen27-greedy
    // prints n greedy token ids. Both share the forward and the 27B kernels above.
    const q27_logits = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_qwen27_logits.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    q27_logits.addImport("xpu_qwen35_kernels", q27k);
    q27_logits.addImport("qwen27_forward", q27_mod);
    const logits_exe = b.addExecutable(.{ .name = "tf-xpu-qwen27-logits", .root_module = q27_logits });
    const logits_install = b.addInstallArtifact(logits_exe, .{});
    const greedy = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_qwen27_greedy.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    greedy.addImport("xpu_qwen35_kernels", q27k);
    greedy.addImport("qwen27_forward", q27_mod);
    const greedy_exe = b.addExecutable(.{ .name = "tf-xpu-qwen27-greedy", .root_module = greedy });
    const greedy_install = b.addInstallArtifact(greedy_exe, .{});
    const tools = b.step("xpu-qwen27-tools", "Build tf-xpu-qwen27-logits and tf-xpu-qwen27-greedy into zig-out/bin");
    tools.dependOn(&logits_install.step);
    tools.dependOn(&greedy_install.step);
    // Phase 6.3/6.4: the 27B forward gate binary (gate C greedy vs the fixture, plus the decode-step measurement).
    // Same forward module and 27B kernels as the tools; the run needs a card and the checkpoint.
    const q27_fwd_test = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_qwen27_forward_test.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    q27_fwd_test.addImport("xpu_qwen27_kernels", q27k);
    q27_fwd_test.addImport("qwen27_forward", q27_mod);
    const q27_fwd_exe = b.addExecutable(.{ .name = "tf-xpu-qwen27-forward-test", .root_module = q27_fwd_test });
    const q27_fwd_install = b.addInstallArtifact(q27_fwd_exe, .{});
    const q27_fwd_run = b.addRunArtifact(q27_fwd_exe);
    q27_fwd_run.step.dependOn(&q27_fwd_install.step);
    b.step("xpu-qwen27-forward-test", "Build and run tf-xpu-qwen27-forward-test: Qwen 3.8 27B forward gate C + decode step on a B70 (opt-in)").dependOn(&q27_fwd_run.step);
    // S04: the 27B decode-speed tool (recorded-step replay, fused variants, per-kernel GB/s). Build-only by
    // default: it runs for minutes on a card, so the S04 gate invokes the binary directly.
    const q27_speed = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_qwen27_speed.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    q27_speed.addImport("xpu_qwen27_kernels", q27k);
    q27_speed.addImport("qwen27_forward", q27_mod);
    const q27_speed_exe = b.addExecutable(.{ .name = "tf-xpu-qwen27-speed", .root_module = q27_speed });
    const q27_speed_install = b.addInstallArtifact(q27_speed_exe, .{});
    b.step("xpu-qwen27-speed", "Build tf-xpu-qwen27-speed into zig-out/bin (S04 measurement tool)").dependOn(&q27_speed_install.step);
    const geo_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/xpu_geometry.zig"), .target = b.graph.host });
    const geo_run = b.addRunArtifact(b.addTest(.{ .name = "xpu-qwen35-geometry", .root_module = geo_module }));
    test_step.dependOn(&geo_run.step);
    b.step("xpu-qwen35-geometry-test", "Launch geometry table vs launch.py values (no GPU)").dependOn(&geo_run.step);
    const f_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_qwen35_forward_test.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    f_module.addImport("xpu_qwen35_kernels", qk);
    f_module.addImport("core", core);
    const f_exe = b.addExecutable(.{ .name = "tf-xpu-qwen35-forward-test", .root_module = f_module });
    const f_install = b.addInstallArtifact(f_exe, .{});
    const f_run = b.addRunArtifact(f_exe);
    f_run.step.dependOn(&f_install.step);
    b.step("xpu-qwen35-forward-test", "Build and run tf-xpu-qwen35-forward-test: Qwen3.5-2B forward gates A-C on a B70 (opt-in)").dependOn(&f_run.step);

    // TEMP (S02 investigation): per-shape qmm bench probe; delete with zig/src/xpu_bench_probe.zig.
    const bp_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_bench_probe.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    bp_module.addImport("xpu_qwen35_kernels", qk);
    bp_module.addImport("core", core);
    const bp_exe = b.addExecutable(.{ .name = "tf-xpu-bench-probe", .root_module = bp_module });
    const bp_install = b.addInstallArtifact(bp_exe, .{});
    const bp_run = b.addRunArtifact(bp_exe);
    bp_run.step.dependOn(&bp_install.step);
    b.step("xpu-bench-probe", "TEMP: per-shape qmm timing probe").dependOn(&bp_install.step);

    // TEMP (S02 investigation): fused norm_qmm vs norm+qmm correctness probe.
    const nq_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_normqmm_probe.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    nq_module.addImport("xpu_qwen35_kernels", qk);
    nq_module.addImport("core", core);
    const nq_exe = b.addExecutable(.{ .name = "tf-xpu-normqmm-probe", .root_module = nq_module });
    const nq_install = b.addInstallArtifact(nq_exe, .{});
    b.step("xpu-normqmm-probe", "TEMP: fused norm_qmm vs norm+qmm bit-exactness probe").dependOn(&nq_install.step);

    // TEMP (S02 investigation): fused mlp_act_down vs mlp_act+qmm correctness probe.
    const ad_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_actdown_probe.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    ad_module.addImport("xpu_qwen35_kernels", qk);
    ad_module.addImport("core", core);
    const ad_exe = b.addExecutable(.{ .name = "tf-xpu-actdown-probe", .root_module = ad_module });
    const ad_install = b.addInstallArtifact(ad_exe, .{});
    b.step("xpu-actdown-probe", "TEMP: fused mlp_act_down vs mlp_act+qmm bit-exactness probe").dependOn(&ad_install.step);

    // TEMP (S02 investigation): device-time comparison of fused vs unfused pairs, batched (no per-launch sync).
    const fp_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_fusion_probe.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    fp_module.addImport("xpu_qwen35_kernels", qk);
    fp_module.addImport("core", core);
    const fp_exe = b.addExecutable(.{ .name = "tf-xpu-fusion-probe", .root_module = fp_module });
    const fp_install = b.addInstallArtifact(fp_exe, .{});
    b.step("xpu-fusion-probe", "TEMP: fused vs unfused device-time comparison (batched)").dependOn(&fp_install.step);

    // TEMP (S02 investigation): qmm_pair vs two qmm calls on the real a/b weights, bit-exactness + device time.
    const pairprobe_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_qmmpair_probe.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    pairprobe_module.addImport("xpu_qwen35_kernels", qk);
    pairprobe_module.addImport("core", core);
    const pairprobe_exe = b.addExecutable(.{ .name = "tf-xpu-qmmpair-probe", .root_module = pairprobe_module });
    const pairprobe_install = b.addInstallArtifact(pairprobe_exe, .{});
    b.step("xpu-qmmpair-probe", "TEMP: qmm_pair vs two qmm calls (bit-exact + device time)").dependOn(&pairprobe_install.step);

    // Phase 4.4-4.5: `Xpu` behind the lane engine's backend interface + qualify, on card (opt-in, needs a card).
    const lanes_mod = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    const bk_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_qwen35_backend_test.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    bk_module.addImport("xpu_qwen35_kernels", qk);
    bk_module.addImport("lanes", lanes_mod);
    bk_module.addImport("core", core);
    const bk_exe = b.addExecutable(.{ .name = "tf-xpu-qwen35-backend-test", .root_module = bk_module });
    const bk_install = b.addInstallArtifact(bk_exe, .{});
    const bk_run = b.addRunArtifact(bk_exe);
    bk_run.step.dependOn(&bk_install.step);
    b.step("xpu-qwen35-backend-test", "Build and run tf-xpu-qwen35-backend-test: Xpu through the lane engine + xpu_qualify on a B70 (opt-in)").dependOn(&bk_run.step);
    b.step("xpu-qwen35-backend-build", "Build tf-xpu-qwen35-backend-test without running it").dependOn(&bk_install.step);

    // TEMP probe: the engine's queue path (prefill/first/queue) vs the reference, to isolate the serve divergence.
    const qp_module = b.createModule(.{ .root_source_file = b.path("zig/src/qprobe.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    qp_module.addImport("xpu_qwen35_kernels", qk);
    qp_module.addImport("lanes", lanes_mod);
    qp_module.addImport("core", core);
    const qp_exe = b.addExecutable(.{ .name = "tf-qprobe", .root_module = qp_module });
    const qp_install = b.addInstallArtifact(qp_exe, .{});
    const qp_run = b.addRunArtifact(qp_exe);
    qp_run.step.dependOn(&qp_install.step);
    b.step("qprobe", "TEMP: engine queue path vs reference").dependOn(&qp_run.step);

    const vp_module = b.createModule(.{ .root_source_file = b.path("zig/src/vprobe.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    vp_module.addImport("xpu_qwen35_kernels", qk);
    vp_module.addImport("lanes", lanes_mod);
    vp_module.addImport("core", core);
    const vp_exe = b.addExecutable(.{ .name = "tf-vprobe", .root_module = vp_module });
    const vp_install = b.addInstallArtifact(vp_exe, .{});
    const vp_run = b.addRunArtifact(vp_exe);
    vp_run.step.dependOn(&vp_install.step);
    b.step("vprobe", "TEMP: 1-row verify path vs reference").dependOn(&vp_run.step);

    // Phase 5: `zig build xpu-native` -> zig-out/xpu/bin/tensorfold-native: the HTTP server over `Xpu` (serve --backend xpu).
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes_mod }} });
    const engines = b.createModule(.{ .root_source_file = b.path("zig/src/native_xpu.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes_mod }, .{ .name = "core", .module = core } } });
    engines.addImport("xpu_qwen35_kernels", qk);
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const checkpoint = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "native_engines", .module = engines }, exlr8_build.cliImport(b, target, .ReleaseSafe, core) } });
    const server = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = engines }, .{ .name = "checkpoint_cli", .module = checkpoint } },
    }) });
    server.root_module.addOptions("build_options", build_options);
    const server_install = b.addInstallArtifact(server, .{ .dest_dir = .{ .override = .{ .custom = "xpu/bin" } } });
    b.step("xpu-native", "tensorfold-native with the Intel XPU engine into zig-out/xpu/bin (serve --backend xpu)").dependOn(&server_install.step);
    // Phase 6.4: Qwen 3.8 27B weights and state on the card (opt-in, needs a card and the multi-shard checkpoint).
    const q27_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_qwen27_test.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    const q27_exe = b.addExecutable(.{ .name = "tf-xpu-qwen27-test", .root_module = q27_module });
    const q27_install = b.addInstallArtifact(q27_exe, .{});
    const q27_run = b.addRunArtifact(q27_exe);
    q27_run.step.dependOn(&q27_install.step);
    b.step("xpu-qwen27-test", "Build and run tf-xpu-qwen27-test: Qwen 3.8 27B weights + state gates on a B70 (opt-in)").dependOn(&q27_run.step);
    b.step("xpu-qwen27-build", "Build zig-out/bin/tf-xpu-qwen27-test without running it").dependOn(&q27_install.step);

    // The 27B pieces' host-side tests (arena packing, state sizes): no card, part of `test`.
    const q27_host = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_qwen27_tests.zig"), .target = b.graph.host, .link_libc = true });
    const q27_tests = b.addTest(.{ .root_module = q27_host });
    const q27_host_run = b.addRunArtifact(q27_tests);
    test_step.dependOn(&q27_host_run.step);
    b.step("xpu-qwen27-host-test", "Host-side tests for the 27B XPU weights and state modules").dependOn(&q27_host_run.step);
}
