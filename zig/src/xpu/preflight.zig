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
        // A card with no driver bound cannot be opened, so it is not part of the usable pool: the harness mirrors this
        // by calling `preflight` per card inside its lock loop and moving on when one refuses (see xpu_harness.py
        // `card()`). Refusing the whole pool for a card the session will never touch would block a job on a parked
        // card; instead skip it here, and require the card actually held (`bdf`) to be bound below.
        if (!isUsable(p, card)) continue;
        const cap = try readCap(p, card, diag);
        if (cap != cap_uw) {
            diag.set("{s}: power1_cap is {d} uW, not 100000000 (100 W) - ask the human (do not re-set it)", .{ card, cap });
            return error.PowerCapNotHundredWatts;
        }
    }
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
/// drive - bound, but to `nvme`; preflight then refused the whole pool with NoPowerCap).
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

/// Test fixture: writes the sysfs `vendor`/`class` of an Intel display device for `card` under `root`.
pub fn markIntelDisplay(tmp: *std.testing.TmpDir, root: []const u8, card: []const u8) !void {
    const io = std.testing.io;
    var d: [128]u8 = undefined;
    try tmp.dir.createDirPath(io, try std.fmt.bufPrint(&d, "{s}/{s}", .{ root, card }));
    try tmp.dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&d, "{s}/{s}/vendor", .{ root, card }), .data = "0x8086\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&d, "{s}/{s}/class", .{ root, card }), .data = "0x030000\n" });
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
    // A bound card with no readable `power1_cap` is refused; a card with no driver is skipped, not refused (the pool
    // may legitimately hold a parked card, as xpu_harness.py's per-card `preflight` loop does).
    var tmp4 = std.testing.tmpDir(.{});
    defer tmp4.cleanup();
    const io = std.testing.io;
    try tmp4.dir.createDirPath(io, "dev/0000:08:00.0/driver");
    try markIntelDisplay(&tmp4, "dev", "0000:08:00.0");
    const real = try tmp4.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(real);
    var root: [200]u8 = undefined;
    const rl = (try std.fmt.bufPrint(&root, "{s}/dev", .{real})).len;
    try std.testing.expectError(error.NoPowerCap, check(.{ .pci_root = root[0..rl], .meminfo = "/proc/meminfo" }, null, &d));
}

test "a slot whose address now belongs to another device is not a card (2026-10-19: 0000:44:00.0 became an NVMe)" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    // 08: a bound B70 at 100 W. 44: bound to a driver, but an NVMe controller (vendor 0x144d, class 0x010802), no cap.
    try tmp.dir.createDirPath(io, "dev/0000:08:00.0/hwmon/hwmon3");
    try tmp.dir.writeFile(io, .{ .sub_path = "dev/0000:08:00.0/hwmon/hwmon3/power1_cap", .data = "100000000\n" });
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
    try check(p, "0000:08:00.0", &d); // the NVMe is skipped, not refused for having no power cap
}
