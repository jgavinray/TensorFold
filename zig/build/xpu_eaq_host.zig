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
