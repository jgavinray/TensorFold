//! The XPU EAQ reader's host tests: synthetic packs, no device and no model download.
const std = @import("std");

pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    const target = b.graph.host;
    const step = b.step("test-xpu-eaq", "Host tests of the XPU EAQ reader on synthetic packs");
    all.dependOn(step);
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "tokenizer", .module = tokenizer }} });
    const reader = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/exl3_eaq.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "core", .module = core }} });
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/exl3_eaq_test.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "exl3_eaq", .module = reader } } });
    const host = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    host.setCwd(b.path("."));
    step.dependOn(&host.step);
}

/// The EXL3 trellis encoder's host gate: the T0-T5 reference tests of the
/// tail-biting Viterbi encoder, scalar only, no device and no model download.
pub fn encodeTests(b: *std.Build, all: *std.Build.Step) void {
    const target = b.graph.host;
    const step = b.step("test-exl3-encode", "Host tests of the EXL3 tail-biting Viterbi encoder (T0-T5)");
    all.dependOn(step);
    const encode = b.createModule(.{ .root_source_file = b.path("zig/src/core/exl3_encode.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const host = b.addRunArtifact(b.addTest(.{ .root_module = encode }));
    host.setCwd(b.path("."));
    step.dependOn(&host.step);
}

/// The block-LDLQ host reference gate: T-LDLQ-1 (identity L reproduces the step-1 pack
/// bit for bit; the real L lowers the proxy on >= 9/10 seeds), scalar only, no device.
/// The closure also runs the encoder's T0-T5 and the exl3_format oracle tests, as
/// encodeTests does: the reference shares encodeTile/pack with them by import.
pub fn ldlqTests(b: *std.Build, all: *std.Build.Step) void {
    const target = b.graph.host;
    const step = b.step("test-exl3-ldlq", "Host tests of the block-LDLQ fp64 reference (T-LDLQ-1)");
    all.dependOn(step);
    const ref = b.createModule(.{ .root_source_file = b.path("zig/src/core/exl3_ldlq_ref.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const host = b.addRunArtifact(b.addTest(.{ .root_module = ref }));
    host.setCwd(b.path("."));
    step.dependOn(&host.step);
}

/// The step-5 half-step host gate: the T-HS-0..3 oracle-parity tests of the odd-halves
/// decode (the unified E(p), both kernel window forms, the dequant phase-A and the two
/// GEMV transcriptions against exl3_format's oracle), scalar only, no device.
pub fn halfstepTests(b: *std.Build, all: *std.Build.Step) void {
    const target = b.graph.host;
    const step = b.step("test-exl3-halfstep", "Host tests of the odd-halves decode kernels (T-HS-0..3) and the reader admission refusals (T-HS-8)");
    all.dependOn(step);
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const reader = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/exl3_eaq.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "core", .module = core }} });
    const gate = b.createModule(.{ .root_source_file = b.path("zig/src/core/exl3_halfstep_test.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "exl3_eaq", .module = reader } } });
    const host = b.addRunArtifact(b.addTest(.{ .root_module = gate }));
    host.setCwd(b.path("."));
    step.dependOn(&host.step);
}

/// The repacker CLI: `zig build eaq-repack` installs `zig-out/bin/tf-eaq-repack`.
pub fn repack(b: *std.Build) void {
    const target = b.graph.host;
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "tokenizer", .module = tokenizer }} });
    const reader = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/exl3_eaq.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "core", .module = core }} });
    const exe = b.addExecutable(.{ .name = "tf-eaq-repack", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/core/eaq_repack_cli.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "exl3_eaq", .module = reader } },
    }) });
    b.step("eaq-repack", "Build tf-eaq-repack: rewrite a stock EXL3 directory into an EAQ pack")
        .dependOn(&b.addInstallArtifact(exe, .{}).step);
}
