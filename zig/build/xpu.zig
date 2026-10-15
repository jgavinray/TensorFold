//! Level Zero (Intel Arc Pro B70): mock-backed admission tests join `zig build test`; on-card runs are opt-in steps
//! (`xpu-test`, `xpu-kernel-exact`). Kernels are compiled with ocloc for device bmg; set -Docloc / -Dxpu-device to override.

const std = @import("std");
const exlr8_build = @import("exlr8.zig");
/// The pinned Qwen 3.8 27B geometry, the source of the -D flags the 27B kernels are compiled with. Reading the
/// module here (not a copy) keeps one definition; it has no imports beyond std, so the build can host it.
const q27 = @import("../src/families/qwen27/config.zig");

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
    const kernels = b.addOptions();
    inline for (.{ "runtime_tests", "dpas" }) |name| {
        const c = b.addSystemCommand(&.{ ocloc, "compile", "-device", xpu_device, "-options", "-cl-std=CL3.0", "-file" });
        c.addFileArg(b.path("zig/kernels/xpu/" ++ name ++ ".cl"));
        c.addArg("-out_dir");
        const out = c.addOutputDirectoryArg(name ++ "-out");
        kernels.addOptionPath(name, out.path(b, b.fmt(name ++ "_{s}.bin", .{xpu_device})));
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
    const qk_files = b.addWriteFiles();
    var qk_src: []const u8 = "//! Compiled Qwen kernel binaries, embedded (see zig/build/xpu.zig).\n";
    // A 27B build supplies the model-shape -D overrides (space-separated); empty keeps the 2B defaults.
    // ocloc takes compile options as one string after -options, so the -D flags belong there, not as bare args.
    const opt_string = b.fmt("-cl-std=CL3.0 {s}", .{shape_defines});
    inline for (.{ "layout", "norm", "norm_nores", "qmm", "mlp_act", "gdn_pre", "gdn_chain", "gdn_post", "attn_partial", "attn_merge" }) |name| {
        const c = b.addSystemCommand(&.{ ocloc, "compile", "-device", xpu_device, "-options", opt_string, "-file" });
        c.addFileArg(b.path("zig/kernels/xpu/shared/" ++ name ++ "/" ++ name ++ ".cl"));
        c.addArg("-out_dir");
        const out = c.addOutputDirectoryArg("qwen35-" ++ name ++ "-out");
        const file = b.fmt("{s}_{s}.bin", .{ name, xpu_device });
        _ = qk_files.addCopyFile(out.path(b, file), file);
        qk_src = b.fmt("{s}pub const {s} align(64) = @embedFile(\"{s}\").*;\n", .{ qk_src, name, file });
    }
    const qk = b.createModule(.{ .root_source_file = qk_files.add("xpu_qwen35_kernels.zig", qk_src) });
    // Phase 6: Qwen 3.8 27B. One more compilation of the same shared sources with the 27B shape defines, embedded
    // under the same module name (`xpu_qwen35_kernels`), so `families/qwen27/xpu_forward.zig` reads them by key.
    // The defines are the pinned geometry of families/qwen27/config.zig; the 2B build's empty default keeps its own.
    const q27_defines = b.fmt("-DQW_HIDDEN={d} -DQW_INTERMEDIATE={d} -DQW_HEADS={d} -DQW_Q_HEADS={d} -DQW_KV_HEADS={d} -DQW_NV={d} -DQW_HV={d} -DQW_ZS={d}", .{
        q27.Pinned.hidden, q27.Pinned.intermediate, q27.Pinned.heads, q27.Pinned.heads, q27.Pinned.kv_heads, q27.Pinned.v_heads, q27.Pinned.v_heads, q27.Pinned.hidden,
    });
    const q27_files = b.addWriteFiles();
    var q27_src: []const u8 = "//! Compiled Qwen kernel binaries for Qwen 3.8 27B, embedded (see zig/build/xpu.zig).\n";
    const q27_opt = b.fmt("-cl-std=CL3.0 {s}", .{q27_defines});
    inline for (.{ "layout", "norm", "norm_nores", "qmm", "mlp_act", "gdn_pre", "gdn_chain", "gdn_post", "attn_partial", "attn_merge" }) |name| {
        const cc = b.addSystemCommand(&.{ ocloc, "compile", "-device", xpu_device, "-options", q27_opt, "-file" });
        cc.addFileArg(b.path("zig/kernels/xpu/shared/" ++ name ++ "/" ++ name ++ ".cl"));
        cc.addArg("-out_dir");
        const out = cc.addOutputDirectoryArg("qwen27-" ++ name ++ "-out");
        const file = b.fmt("{s}_{s}.bin", .{ name, xpu_device });
        _ = q27_files.addCopyFile(out.path(b, file), file);
        q27_src = b.fmt("{s}pub const {s} align(64) = @embedFile(\"{s}\").*;\n", .{ q27_src, name, file });
    }
    const q27k = b.createModule(.{ .root_source_file = q27_files.add("xpu_qwen27_kernels.zig", q27_src) });
    const q27_mod = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_qwen27_forward.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    q27_mod.addImport("xpu_qwen35_kernels", q27k);
    // Phase 6.3: the two 27B tools. tf-xpu-qwen27-logits writes len(tokens) x vocab bf16 logits; tf-xpu-qwen27-greedy
    // prints n greedy token ids. Both share the forward and the 27B kernels above.
    const q27_logits = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_qwen27_logits.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
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
