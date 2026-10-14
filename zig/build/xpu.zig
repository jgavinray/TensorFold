//! Level Zero (Intel Arc Pro B70): mock-backed admission tests join `zig build test`; on-card runs are opt-in steps
//! (`xpu-test`, `xpu-kernel-exact`). Kernels are compiled with ocloc for device bmg; set -Docloc / -Dxpu-device to override.

const std = @import("std");

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
}
