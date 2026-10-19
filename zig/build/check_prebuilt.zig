//! Prebuilt XPU kernel modules must come from the current sources. Each committed prebuilt dir carries SOURCES, one
//! line `<key> <sha256 hex>` per module, the hash taken over the ocloc options, a newline, then the bytes of
//! zig/kernels/xpu/shared/<key>/<key>.cl. A build that embeds the dir checks it first, so a kernel source edited
//! without refreshing its module fails the build instead of shipping the old kernel (2026-10-19: 9 of 10 27B modules
//! were stale - an ocloc-less build would have served a qmm without q8/q8x and attention without the GQA kernel).
//!
//! Usage (cwd = the build root): check-prebuilt check|write <dir> <options> <key>...
//!   check  every key's SOURCES line matches its current source and options, else list the stale ones and exit 1
//!   write  rewrite <dir>/SOURCES from the current sources (after `zig build xpu-prebuilt-refresh` copied the modules)
const std = @import("std");

pub const Hex = [64]u8;

/// The hash a module's SOURCES line records: options, a newline, the kernel source.
pub fn digest(options: []const u8, source: []const u8) Hex {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(options);
    h.update("\n");
    h.update(source);
    var out: [32]u8 = undefined;
    h.final(&out);
    return std.fmt.bytesToHex(out, .lower);
}

/// The recorded hash of `key` in a SOURCES manifest, or null when the key has no line.
pub fn lookup(manifest: []const u8, key: []const u8) ?[]const u8 {
    var lines = std.mem.tokenizeScalar(u8, manifest, '\n');
    while (lines.next()) |line| {
        var fields = std.mem.tokenizeScalar(u8, line, ' ');
        const k = fields.next() orelse continue;
        const v = fields.next() orelse continue;
        if (std.mem.eql(u8, k, key)) return v;
    }
    return null;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 5) return error.Usage;
    const mode, const dir, const options = .{ args[1], args[2], args[3] };
    const keys = args[4..];
    const cwd = std.Io.Dir.cwd();
    const manifest_path = try std.fs.path.join(a, &.{ dir, "SOURCES" });

    var lines: std.ArrayList(u8) = .empty;
    defer lines.deinit(gpa);
    const manifest: []const u8 = if (std.mem.eql(u8, mode, "check"))
        cwd.readFileAlloc(io, manifest_path, a, .limited(1 << 20)) catch |e| {
            std.debug.print("{s}: no SOURCES manifest ({t}); refresh the prebuilt modules: zig build xpu-prebuilt-refresh (needs ocloc)\n", .{ dir, e });
            std.process.exit(1);
        }
    else if (std.mem.eql(u8, mode, "write")) "" else return error.Usage;

    var stale: usize = 0;
    for (keys) |key| {
        const path = try std.fmt.allocPrint(a, "zig/kernels/xpu/shared/{s}/{s}.cl", .{ key, key });
        const source = try cwd.readFileAlloc(io, path, gpa, .limited(1 << 24));
        defer gpa.free(source);
        const want = digest(options, source);
        if (manifest.len == 0) {
            try lines.print(gpa, "{s} {s}\n", .{ key, want });
        } else if (!std.mem.eql(u8, lookup(manifest, key) orelse "", &want)) {
            std.debug.print("{s}/{s}_*.bin is stale: {s} or the options changed since the module was built\n", .{ dir, key, path });
            stale += 1;
        }
    }
    if (manifest.len == 0) {
        try cwd.writeFile(io, .{ .sub_path = manifest_path, .data = lines.items });
        return;
    }
    if (stale != 0) {
        std.debug.print("{d} prebuilt module(s) in {s} do not match their sources; refresh: zig build xpu-prebuilt-refresh (needs ocloc)\n", .{ stale, dir });
        std.process.exit(1);
    }
}

test "the digest follows the source and the options, and the manifest lookup finds each key" {
    const opt = "-cl-std=CL3.0 -DQW_HIDDEN=5120";
    const d = digest(opt, "kernel void k() {}");
    try std.testing.expectEqualStrings(&d, &digest(opt, "kernel void k() {}"));
    // bad controls: an edited source, or the same source compiled with other defines, no longer match
    try std.testing.expect(!std.mem.eql(u8, &d, &digest(opt, "kernel void k() { }")));
    try std.testing.expect(!std.mem.eql(u8, &d, &digest("-cl-std=CL3.0", "kernel void k() {}")));
    const manifest = "qmm " ++ d ++ "\nnorm 00\n";
    try std.testing.expectEqualStrings(&d, lookup(manifest, "qmm").?);
    try std.testing.expectEqualStrings("00", lookup(manifest, "norm").?);
    try std.testing.expect(lookup(manifest, "qm") == null);
    try std.testing.expect(lookup(manifest, "layout") == null);
}
