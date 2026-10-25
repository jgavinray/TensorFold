//! The KL gate binary (`xpu-eaq-kl`, eaq-ldlq-spec-fable.md §5.5), built like `xpu-qwen27`.
//! Its own module file so the gate slice never edits zig/build/xpu.zig (its kernel images
//! are compiled here into this step's embed), with one addition of its own:
//! zig/kernels/xpu/qwen27/eaq_kl.cl (the fp32-output head GEMM).
//!
//! The image list mirrors zig/build/xpu.zig steps(): Runner.init loads every one of them,
//! so the list must stay a superset of what qwen27.zig's kernel table asks for.

const std = @import("std");

fn userOption(b: *std.Build, name: []const u8, default: []const u8) []const u8 {
    if (b.user_input_options.get(name)) |up| {
        return switch (up) { .scalar => |v| v, else => default };
    }
    return default;
}

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    // xpu.zig's steps() already declared these options; re-declaring panics the
    // configurer, so read the user-set values straight off the map and keep the
    // same defaults (bmg / ocloc / ngen) as zig/build/xpu.zig does.
    const ocloc = userOption(b, "ocloc", "ocloc");
    const device = userOption(b, "xpu-device", "bmg");
    const ngen_include = userOption(b, "ngen-include", "ngen");
    const files = b.addWriteFiles();
    const opts = "-cl-std=CL3.0 -cl-intel-256-GRF-per-thread";
    for ([_]struct { []const u8, []const u8, []const u8 }{
        .{ "glue", "zig/kernels/xpu/qwen27/glue.cl", "" },
        .{ "decode", "zig/kernels/xpu/qwen27/decode.cl", "" },
        .{ "exl3", "zig/kernels/xpu/qwen27/exl3.cl", "" },
        .{ "exl3_h", "zig/kernels/xpu/qwen27/exl3_h.cl", "" },
        .{ "exl3_enc", "zig/kernels/xpu/qwen27/exl3_enc.cl", "" },
        .{ "calib", "zig/kernels/xpu/qwen27/calib.cl", "" },
        .{ "ldlq", "zig/kernels/xpu/qwen27/ldlq.cl", "" },
        .{ "gemm", "zig/kernels/xpu/qwen27/gemm.cl", " -DGSZ=64" },
        .{ "gemm128", "zig/kernels/xpu/qwen27/gemm.cl", " -DGSZ=128" },
        .{ "gdn", "zig/kernels/xpu/qwen27/gdn_chunk.cl", "" },
        .{ "drafter", "zig/kernels/xpu/qwen27/drafter.cl", "" },
        .{ "eaq_kl", "zig/kernels/xpu/qwen27/eaq_kl.cl", "" }, // the gate's fp32-C GEMM: memory-streaming like the exl3 family
    }) |k| {
        const c = b.addSystemCommand(&.{ ocloc, "compile", "-q", "-file" });
        c.addFileArg(b.path(k[1]));
        // the EXL3 kernels stream memory: the 256-GRF flag would only halve their occupancy
        const exl3_family = std.mem.endsWith(u8, k[1], "exl3.cl") or std.mem.endsWith(u8, k[1], "exl3_h.cl") or
            std.mem.endsWith(u8, k[1], "exl3_enc.cl") or std.mem.endsWith(u8, k[1], "calib.cl") or
            std.mem.endsWith(u8, k[1], "ldlq.cl") or std.mem.endsWith(u8, k[1], "eaq_kl.cl");
        const base_opts = if (exl3_family) opts[0 .. opts.len - " -cl-intel-256-GRF-per-thread".len] else opts;
        const inc = if (exl3_family) "" else " -I zig/kernels/xpu/qwen27";
        c.addArgs(&.{ "-device", device, "-options", b.fmt("{s}{s}{s}", .{ base_opts, k[2], inc }), "-output_no_suffix", "-output", k[0], "-out_dir" });
        c.setCwd(b.path("."));
        if (!exl3_family) { // the exl3 files include nothing; skip the extra rebuild triggers
            for ([_][]const u8{ "glue_attn.cl", "glue_quant.cl" }) |part| c.addFileInput(b.path(b.fmt("zig/kernels/xpu/qwen27/{s}", .{part})));
        }
        const dir = c.addOutputDirectoryArg(b.fmt("ocloc-kl-{s}", .{k[0]}));
        _ = files.addCopyFile(dir.path(b, b.fmt("{s}.bin", .{k[0]})), b.fmt("{s}.bin", .{k[0]}));
    }
    // nGEN generator, built for the host and run at build time for the two prompt GEMM families (groups of 64 and 128) — as zig/build/xpu.zig
    const gen = b.addSystemCommand(&.{ "c++", "-std=c++17", "-O1", "-fno-operator-names", b.fmt("-I{s}", .{ngen_include}), "-o" });
    const gen_bin = gen.addOutputFileArg("w4_ngen");
    gen.addFileArg(b.path("zig/tools/xpu/w4_ngen.cpp"));
    for ([_][]const u8{ "w4_prompt.hpp", "w4a8_prompt.hpp" }) |h| gen.addFileInput(b.path(b.fmt("zig/tools/xpu/{s}", .{h})));
    for ([_]struct { []const u8, []const u8, []const []const u8 }{
        .{ "ngen_wide", "64", &.{"b3"} },
        .{ "ngen_narrow", "64", &.{ "s4", "asplit" } },
        .{ "ngen_wide128", "128", &.{"b3"} },
        .{ "ngen_narrow128", "128", &.{ "s4", "asplit" } },
        .{ "ngen_swiglu", "64", &.{ "b3", "swiglu" } },
        .{ "ngen_swiglu128", "128", &.{ "b3", "swiglu" } },
        .{ "ngen_a8", "128", &.{"a8"} },
        .{ "ngen_a8_swiglu", "128", &.{ "a8", "swiglu" } },
        .{ "ngen_a4", "128", &.{"a4"} },
    }) |z| {
        const run = std.Build.Step.Run.create(b, b.fmt("kl nGEN {s}", .{z[0]}));
        run.addFileArg(gen_bin);
        run.addArgs(&.{ "xe2", z[1] });
        const out = run.addOutputFileArg(b.fmt("{s}.zebin", .{z[0]}));
        run.addArgs(z[2]);
        _ = files.addCopyFile(out, b.fmt("{s}.zebin", .{z[0]}));
    }
    const embedded = b.createModule(.{ .root_source_file = files.add("xpu_qwen27_kernels.zig",
        \\pub const glue align(8) = @embedFile("glue.bin").*;
        \\pub const decode align(8) = @embedFile("decode.bin").*;
        \\pub const exl3 align(8) = @embedFile("exl3.bin").*;
        \\pub const exl3_h align(8) = @embedFile("exl3_h.bin").*;
        \\pub const exl3_enc align(8) = @embedFile("exl3_enc.bin").*;
        \\pub const calib align(8) = @embedFile("calib.bin").*;
        \\pub const ldlq align(8) = @embedFile("ldlq.bin").*;
        \\pub const gemm align(8) = @embedFile("gemm.bin").*;
        \\pub const gemm128 align(8) = @embedFile("gemm128.bin").*;
        \\pub const gdn align(8) = @embedFile("gdn.bin").*;
        \\pub const drafter align(8) = @embedFile("drafter.bin").*;
        \\pub const ngen_wide align(8) = @embedFile("ngen_wide.zebin").*;
        \\pub const ngen_narrow align(8) = @embedFile("ngen_narrow.zebin").*;
        \\pub const ngen_wide128 align(8) = @embedFile("ngen_wide128.zebin").*;
        \\pub const ngen_narrow128 align(8) = @embedFile("ngen_narrow128.zebin").*;
        \\pub const ngen_swiglu align(8) = @embedFile("ngen_swiglu.zebin").*;
        \\pub const ngen_swiglu128 align(8) = @embedFile("ngen_swiglu128.zebin").*;
        \\pub const ngen_a8 align(8) = @embedFile("ngen_a8.zebin").*;
        \\pub const ngen_a8_swiglu align(8) = @embedFile("ngen_a8_swiglu.zebin").*;
        \\pub const ngen_a4 align(8) = @embedFile("ngen_a4.zebin").*;
        \\
    ) });
    const kl_kernels = b.createModule(.{ .root_source_file = files.add("xpu_kl_kernels.zig",
        \\pub const eaq_kl align(8) = @embedFile("eaq_kl.bin").*;
        \\
    ) });
    const ze_mod = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/ze.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core_mod = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const qwen27 = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/qwen27.zig"), .target = target, .optimize = optimize, .link_libc = true });
    qwen27.addImport("xpu_qwen27_kernels", embedded);
    qwen27.addImport("ze", ze_mod);
    qwen27.addImport("core", core_mod);
    const module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/kl_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    module.addImport("xpu_qwen27_kernels", embedded); // the runner's images (kl.zig only needs its own, qwen27.zig needs all)
    module.addImport("eaq_kl_kernels", kl_kernels);
    module.addImport("ze", ze_mod);
    module.addImport("core", core_mod);
    module.addImport("qwen27", qwen27);
    const exe = b.addExecutable(.{ .name = "xpu-eaq-kl", .root_module = module });
    const install = b.addInstallArtifact(exe, .{});
    b.step("xpu-eaq-kl", "Build the KL gate (pack side: Runner.prompt + head GEMM + the §5.3 statistic; needs ocloc, a C++17 compiler and nGEN headers)").dependOn(&install.step);
}
