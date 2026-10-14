//! Level Zero (Intel Arc Pro B70): mock-backed admission tests join `zig build test`; on-card runs are opt-in steps
//! (`xpu-test`, `xpu-kernel-exact`). Kernels are compiled with ocloc for device bmg; set -Docloc / -Dxpu-device to override.

const std = @import("std");

/// Modules with an accepted port under zig/kernels/xpu/qwen3_5/ports/.
const port_keys = [_][]const u8{ "layout", "norm_nores", "qmm", "mlp_act", "gdn_pre", "gdn_post", "attn_merge" };

/// xpu-host-test (part of `test`), xpu-test (build + run on the card), xpu-gpu-build, xpu-kernel-exact (CLI install).
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
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

    // Phase 4.1-4.2: Qwen3.5-2B weights and state on the card (opt-in, needs a card and the checkpoint).
    const q_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_qwen35_test.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    const q_exe = b.addExecutable(.{ .name = "tf-xpu-qwen35-test", .root_module = q_module });
    const q_install = b.addInstallArtifact(q_exe, .{});
    const q_run = b.addRunArtifact(q_exe);
    q_run.step.dependOn(&q_install.step);
    b.step("xpu-qwen35-test", "Build and run tf-xpu-qwen35-test: Qwen3.5-2B weights + state gates on a B70 (opt-in)").dependOn(&q_run.step);

    // Phase 4.3: Qwen3.5-2B forward pass (reference OpenCL kernels, ocloc-compiled) with gates A-C (opt-in, needs a card).
    const qk = b.addOptions();
    inline for (.{ "layout", "norm", "norm_nores", "qmm", "mlp_act", "gdn_pre", "gdn_chain", "gdn_post", "attn_partial", "attn_merge" }) |name| {
        const c = b.addSystemCommand(&.{ ocloc, "compile", "-device", xpu_device, "-options", "-cl-std=CL3.0", "-file" });
        c.addFileArg(b.path("zig/kernels/xpu/qwen3_5/reference/" ++ name ++ "/" ++ name ++ ".cl"));
        c.addArg("-out_dir");
        const out = c.addOutputDirectoryArg("qwen35-" ++ name ++ "-out");
        qk.addOptionPath(name, out.path(b, b.fmt(name ++ "_{s}.bin", .{xpu_device})));
    }
    // Accepted optimized ports (ports/<key>/<key>.cl): option "<key>_port". Adding one = drop the files, add the key here
    // and the .port arm in zig/src/families/qwen3_5/xpu_geometry.zig.
    inline for (port_keys) |name| {
        const c = b.addSystemCommand(&.{ ocloc, "compile", "-device", xpu_device, "-options", "-cl-std=CL3.0", "-file" });
        c.addFileArg(b.path("zig/kernels/xpu/qwen3_5/ports/" ++ name ++ "/" ++ name ++ ".cl"));
        c.addArg("-out_dir");
        const out = c.addOutputDirectoryArg("qwen35-" ++ name ++ "-port-out");
        qk.addOptionPath(name ++ "_port", out.path(b, b.fmt(name ++ "_{s}.bin", .{xpu_device})));
    }
    const geo_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/xpu_geometry.zig"), .target = b.graph.host });
    const geo_run = b.addRunArtifact(b.addTest(.{ .name = "xpu-qwen35-geometry", .root_module = geo_module }));
    test_step.dependOn(&geo_run.step);
    b.step("xpu-qwen35-geometry-test", "Launch geometry table vs launch.py values (no GPU)").dependOn(&geo_run.step);
    const f_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_qwen35_forward_test.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    f_module.addOptions("xpu_qwen35_kernels", qk);
    const f_exe = b.addExecutable(.{ .name = "tf-xpu-qwen35-forward-test", .root_module = f_module });
    const f_install = b.addInstallArtifact(f_exe, .{});
    const f_run = b.addRunArtifact(f_exe);
    f_run.step.dependOn(&f_install.step);
    b.step("xpu-qwen35-forward-test", "Build and run tf-xpu-qwen35-forward-test: Qwen3.5-2B forward gates A-C on a B70 (opt-in)").dependOn(&f_run.step);

    // Phase 4.4-4.5: `Xpu` behind the lane engine's backend interface + qualify, on card (opt-in, needs a card).
    const lanes_mod = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    const bk_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu_qwen35_backend_test.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    bk_module.addOptions("xpu_qwen35_kernels", qk);
    bk_module.addImport("lanes", lanes_mod);
    const bk_exe = b.addExecutable(.{ .name = "tf-xpu-qwen35-backend-test", .root_module = bk_module });
    const bk_install = b.addInstallArtifact(bk_exe, .{});
    const bk_run = b.addRunArtifact(bk_exe);
    bk_run.step.dependOn(&bk_install.step);
    b.step("xpu-qwen35-backend-test", "Build and run tf-xpu-qwen35-backend-test: Xpu through the lane engine + xpu_qualify on a B70 (opt-in)").dependOn(&bk_run.step);
    b.step("xpu-qwen35-backend-build", "Build tf-xpu-qwen35-backend-test without running it").dependOn(&bk_install.step);
}
