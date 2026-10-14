//! Read-only preflight before any device work (same rules as exp/harness/xpu_harness.py preflight): both B70 power caps
//! read exactly 100 W, MemAvailable >= 24 GiB, and the card is bound to a driver. Never writes sysfs.
const std = @import("std");
const device = @import("device.zig");
const sys = @import("sys.zig");

pub const cap_uw: u64 = 100_000_000;
pub const min_available_kib: u64 = 24 * 1024 * 1024;

pub const Refusal = error{ NoPowerCap, PowerCapNotHundredWatts, CardNotBound, MemoryLow, Unreadable };

/// Human-readable reason for the last refusal.
pub const Diag = struct {
    buf: [256]u8 = undefined,
    len: usize = 0,
    pub fn message(self: *const Diag) []const u8 {
        return self.buf[0..self.len];
    }
    fn set(self: *Diag, comptime fmt: []const u8, args: anytype) void {
        self.len = if (std.fmt.bufPrint(&self.buf, fmt, args)) |t| t.len else |_| self.buf.len;
    }
};

pub const Paths = struct {
    /// Production values; tests point these at temp trees.
    pci_root: []const u8 = "/sys/bus/pci/devices",
    meminfo: []const u8 = "/proc/meminfo",
};

pub fn check(p: Paths, bdf: ?[]const u8, diag: *Diag) Refusal!void {
    for (device.cards) |card| {
        const cap = try readCap(p, card, diag);
        if (cap != cap_uw) {
            diag.set("{s}: power1_cap is {d} uW, not 100000000 (100 W) - ask the human (do not re-set it)", .{ card, cap });
            return error.PowerCapNotHundredWatts;
        }
    }
    if (bdf) |b| {
        var path: [160]u8 = undefined;
        const driver = sys.zfmt(&path, "{s}/{s}/driver", .{ p.pci_root, b }) orelse return error.Unreadable;
        if (!sys.exists(driver)) {
            diag.set("{s}: not bound to a driver - card may be wedged; stop and tell the human", .{b});
            return error.CardNotBound;
        }
    }
    const avail = memAvailableKiB(p.meminfo) orelse {
        diag.set("cannot read MemAvailable from {s}", .{p.meminfo});
        return error.Unreadable;
    };
    if (avail < min_available_kib) {
        diag.set("MemAvailable {d} KiB < 24 GiB", .{avail});
        return error.MemoryLow;
    }
}

fn readCap(p: Paths, card: []const u8, diag: *Diag) Refusal!u64 {
    var i: u32 = 0;
    while (i < 64) : (i += 1) {
        var path: [192]u8 = undefined;
        const z = sys.zfmt(&path, "{s}/{s}/hwmon/hwmon{d}/power1_cap", .{ p.pci_root, card, i }) orelse return error.Unreadable;
        var buf: [32]u8 = undefined;
        const text = sys.readSmall(z, &buf) orelse continue;
        return std.fmt.parseInt(u64, std.mem.trim(u8, text, " \n\r\t"), 10) catch {
            diag.set("{s}: power1_cap unparsable", .{card});
            return error.Unreadable;
        };
    }
    diag.set("{s}: no power1_cap (driver not bound?)", .{card});
    return error.NoPowerCap;
}

pub fn memAvailableKiB(meminfo_path: []const u8) ?u64 {
    var pathz: [128:0]u8 = undefined;
    if (meminfo_path.len >= pathz.len) return null;
    @memcpy(pathz[0..meminfo_path.len], meminfo_path);
    pathz[meminfo_path.len] = 0;
    var buf: [8192]u8 = undefined;
    const text = sys.readSmall(&pathz, &buf) orelse return null;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "MemAvailable:")) continue;
        var it = std.mem.tokenizeAny(u8, line["MemAvailable:".len..], " \tkB");
        return std.fmt.parseInt(u64, it.next() orelse return null, 10) catch null;
    }
    return null;
}

const Fixture = struct { root: [200]u8, rl: usize, mem: [200]u8, ml: usize };

fn fixture(tmp: *std.testing.TmpDir, caps: [2][]const u8, bound: bool, mem_kib: u64) !Fixture {
    const io = std.testing.io;
    var out: Fixture = undefined;
    for (device.cards, caps) |card, cap| {
        var d: [96]u8 = undefined;
        try tmp.dir.createDirPath(io, try std.fmt.bufPrint(&d, "dev/{s}/hwmon/hwmon3", .{card}));
        try tmp.dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&d, "dev/{s}/hwmon/hwmon3/power1_cap", .{card}), .data = cap });
        if (bound) try tmp.dir.createDirPath(io, try std.fmt.bufPrint(&d, "dev/{s}/driver", .{card}));
    }
    var m: [96]u8 = undefined;
    try tmp.dir.writeFile(io, .{ .sub_path = "meminfo", .data = try std.fmt.bufPrint(&m, "MemTotal: 1 kB\nMemAvailable:   {d} kB\n", .{mem_kib}) });
    const real = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(real);
    out.rl = (try std.fmt.bufPrint(&out.root, "{s}/dev", .{real})).len;
    out.ml = (try std.fmt.bufPrint(&out.mem, "{s}/meminfo", .{real})).len;
    return out;
}

test "preflight admits 100 W caps, bound cards and 24 GiB" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const f = try fixture(&tmp, .{ "100000000\n", "100000000\n" }, true, 30 * 1024 * 1024);
    var d: Diag = .{};
    try check(.{ .pci_root = f.root[0..f.rl], .meminfo = f.mem[0..f.ml] }, "0000:08:00.0", &d);
}

test "preflight refuses a changed cap, an unbound card, low memory and a missing cap" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var d: Diag = .{};
    const f = try fixture(&tmp, .{ "100000000\n", "120000000\n" }, true, 30 * 1024 * 1024);
    try std.testing.expectError(error.PowerCapNotHundredWatts, check(.{ .pci_root = f.root[0..f.rl], .meminfo = f.mem[0..f.ml] }, null, &d));
    try std.testing.expect(std.mem.indexOf(u8, d.message(), "0000:44:00.0") != null);

    var tmp2 = std.testing.tmpDir(.{});
    defer tmp2.cleanup();
    const g = try fixture(&tmp2, .{ "100000000", "100000000" }, false, 30 * 1024 * 1024);
    try std.testing.expectError(error.CardNotBound, check(.{ .pci_root = g.root[0..g.rl], .meminfo = g.mem[0..g.ml] }, "0000:44:00.0", &d));

    var tmp3 = std.testing.tmpDir(.{});
    defer tmp3.cleanup();
    const h = try fixture(&tmp3, .{ "100000000", "100000000" }, true, 23 * 1024 * 1024);
    try std.testing.expectError(error.MemoryLow, check(.{ .pci_root = h.root[0..h.rl], .meminfo = h.mem[0..h.ml] }, "0000:08:00.0", &d));
    try std.testing.expectError(error.NoPowerCap, check(.{ .pci_root = "/nonexistent", .meminfo = "/proc/meminfo" }, null, &d));
}
