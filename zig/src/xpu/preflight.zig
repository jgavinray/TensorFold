//! Read-only preflight before any device work: the held card is bound to a driver and MemAvailable >= 24 GiB.
//! Never writes sysfs.
const std = @import("std");
const device = @import("device.zig");
const sys = @import("sys.zig");

pub const min_available_kib: u64 = 24 * 1024 * 1024;

pub const Refusal = error{ CardNotBound, MemoryLow, Unreadable };

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
    if (bdf) |b| {
        if (!isBound(p, b)) {
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

/// Whether the slot holds a usable card: an Intel display device (sysfs `vendor` 0x8086, `class` 0x03xxxx) with a
/// driver bound. The lock and the pool sweep both use this, so neither a parked card nor a slot whose address now
/// belongs to another device is handed out or swept (2026-10-19: one B70 was removed and 0000:44:00.0 became an NVMe
/// drive - bound, but to `nvme`).
pub fn isUsable(p: Paths, card: []const u8) bool {
    return isIntelDisplay(p, card) and isBound(p, card);
}

/// Whether the device at `card` is an Intel display-class device, from its sysfs `vendor` and `class` files.
pub fn isIntelDisplay(p: Paths, card: []const u8) bool {
    var path: [160]u8 = undefined;
    var buf: [32]u8 = undefined;
    const vendor = sys.zfmt(&path, "{s}/{s}/vendor", .{ p.pci_root, card }) orelse return false;
    const v = sys.readSmall(vendor, &buf) orelse return false;
    if (!std.mem.eql(u8, std.mem.trim(u8, v, " \n\r\t"), "0x8086")) return false;
    const class = sys.zfmt(&path, "{s}/{s}/class", .{ p.pci_root, card }) orelse return false;
    const k = sys.readSmall(class, &buf) orelse return false;
    return std.mem.startsWith(u8, std.mem.trim(u8, k, " \n\r\t"), "0x03");
}

/// Whether the card has a driver bound: a card with no `driver` symlink cannot be opened, so it is not usable.
pub fn isBound(p: Paths, card: []const u8) bool {
    var path: [160]u8 = undefined;
    const driver = sys.zfmt(&path, "{s}/{s}/driver", .{ p.pci_root, card }) orelse return false;
    return sys.exists(driver);
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

/// Test fixture: writes the sysfs `vendor`/`class` of an Intel display device for `card` under `root`.
pub fn markIntelDisplay(tmp: *std.testing.TmpDir, root: []const u8, card: []const u8) !void {
    const io = std.testing.io;
    var d: [128]u8 = undefined;
    try tmp.dir.createDirPath(io, try std.fmt.bufPrint(&d, "{s}/{s}", .{ root, card }));
    try tmp.dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&d, "{s}/{s}/vendor", .{ root, card }), .data = "0x8086\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&d, "{s}/{s}/class", .{ root, card }), .data = "0x030000\n" });
}

const Fixture = struct { root: [200]u8, rl: usize, mem: [200]u8, ml: usize };

fn fixture(tmp: *std.testing.TmpDir, bound: bool, mem_kib: u64) !Fixture {
    const io = std.testing.io;
    var out: Fixture = undefined;
    for (device.cards) |card| {
        var d: [96]u8 = undefined;
        try tmp.dir.createDirPath(io, try std.fmt.bufPrint(&d, "dev/{s}", .{card}));
        if (bound) try tmp.dir.createDirPath(io, try std.fmt.bufPrint(&d, "dev/{s}/driver", .{card}));
        try markIntelDisplay(tmp, "dev", card);
    }
    var m: [96]u8 = undefined;
    try tmp.dir.writeFile(io, .{ .sub_path = "meminfo", .data = try std.fmt.bufPrint(&m, "MemTotal: 1 kB\nMemAvailable:   {d} kB\n", .{mem_kib}) });
    const real = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(real);
    out.rl = (try std.fmt.bufPrint(&out.root, "{s}/dev", .{real})).len;
    out.ml = (try std.fmt.bufPrint(&out.mem, "{s}/meminfo", .{real})).len;
    return out;
}

test "preflight admits a bound card and 24 GiB" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const f = try fixture(&tmp, true, 30 * 1024 * 1024);
    var d: Diag = .{};
    try check(.{ .pci_root = f.root[0..f.rl], .meminfo = f.mem[0..f.ml] }, "0000:08:00.0", &d);
}

test "preflight refuses an unbound card and low memory" {
    var d: Diag = .{};
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const g = try fixture(&tmp, false, 30 * 1024 * 1024);
    try std.testing.expectError(error.CardNotBound, check(.{ .pci_root = g.root[0..g.rl], .meminfo = g.mem[0..g.ml] }, "0000:44:00.0", &d));

    var tmp2 = std.testing.tmpDir(.{});
    defer tmp2.cleanup();
    const h = try fixture(&tmp2, true, 23 * 1024 * 1024);
    try std.testing.expectError(error.MemoryLow, check(.{ .pci_root = h.root[0..h.rl], .meminfo = h.mem[0..h.ml] }, "0000:08:00.0", &d));
}

test "a slot whose address now belongs to another device is not a card (2026-10-19: 0000:44:00.0 became an NVMe)" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    // 08: a bound B70. 44: bound to a driver, but an NVMe controller (vendor 0x144d, class 0x010802).
    try tmp.dir.createDirPath(io, "dev/0000:08:00.0/driver");
    try markIntelDisplay(&tmp, "dev", "0000:08:00.0");
    try tmp.dir.createDirPath(io, "dev/0000:44:00.0/driver");
    try tmp.dir.writeFile(io, .{ .sub_path = "dev/0000:44:00.0/vendor", .data = "0x144d\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "dev/0000:44:00.0/class", .data = "0x010802\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "meminfo", .data = "MemAvailable:   31457280 kB\n" });
    const real = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(real);
    var root: [200]u8 = undefined;
    var mem: [200]u8 = undefined;
    const p: Paths = .{ .pci_root = (try std.fmt.bufPrint(&root, "{s}/dev", .{real})), .meminfo = (try std.fmt.bufPrint(&mem, "{s}/meminfo", .{real})) };
    try std.testing.expect(isUsable(p, "0000:08:00.0"));
    try std.testing.expect(!isUsable(p, "0000:44:00.0"));
    var d: Diag = .{};
    try check(p, "0000:08:00.0", &d);
}
