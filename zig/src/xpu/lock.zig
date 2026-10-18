//! Card lock shared with the Python harness: flock on <lock_dir>/card-<bdf>.lock; take a free card of the pool or wait.
const std = @import("std");
const device = @import("device.zig");
const preflight = @import("preflight.zig");
const sys = @import("sys.zig");

pub const default_dir = "/storage01/qwen-flash-on-intel/run";

pub const Error = error{ Timeout, Invalid, CannotOpenLock };

pub const Held = struct {
    file: sys.File,
    /// Index into device.cards.
    index: usize,

    pub fn bdf(self: Held) []const u8 {
        return device.cards[self.index];
    }
    pub fn release(self: *Held) void {
        self.file.unlockClose();
        self.* = undefined;
    }
};

fn lockPath(buf: []u8, dir: []const u8, bdf: []const u8) ?[:0]u8 {
    return sys.zfmt(buf, "{s}/card-{s}.lock", .{ dir, bdf });
}

/// Tries to lock one specific card without waiting. A card with no driver bound cannot be opened, so it is skipped
/// (`null`): the lock pool is the usable cards, matching xpu_harness.py, which moves on when a card's preflight fails.
pub fn tryCard(dir: []const u8, index: usize, paths: preflight.Paths) Error!?Held {
    if (!preflight.isBound(paths, device.cards[index])) return null;
    var buf: [256]u8 = undefined;
    const path = lockPath(&buf, dir, device.cards[index]) orelse return error.Invalid;
    const f = sys.File.openLock(path) orelse return error.CannotOpenLock;
    if (!f.tryLock()) {
        f.unlockClose();
        return null;
    }
    return .{ .file = f, .index = index };
}

/// Takes the first free card (the preferred one first when given), polling every 200 ms until `wait_ms` elapses.
pub fn acquire(dir: []const u8, prefer: ?usize, wait_ms: u64) Error!Held {
    return acquireOf(dir, prefer, null, wait_ms);
}

/// As `acquire`, but when `only` is set that one card is the whole pool.
pub fn acquireOf(dir: []const u8, prefer: ?usize, only: ?usize, wait_ms: u64) Error!Held {
    return acquireOfPaths(dir, prefer, only, wait_ms, .{});
}

/// As `acquireOf`, but with the sysfs root `paths` (tests point it at a fixture tree).
pub fn acquireOfPaths(dir: []const u8, prefer: ?usize, only: ?usize, wait_ms: u64, paths: preflight.Paths) Error!Held {
    var dirz: [256:0]u8 = undefined;
    if (dir.len >= dirz.len) return error.Invalid;
    @memcpy(dirz[0..dir.len], dir);
    dirz[dir.len] = 0;
    sys.makeDir(@ptrCast(&dirz));
    var waited: u64 = 0;
    while (true) {
        var order: [device.cards.len]usize = undefined;
        for (&order, 0..) |*o, i| o.* = i;
        if (prefer) |p| std.mem.swap(usize, &order[0], &order[p]);
        for (order) |i| {
            if (only) |o| if (i != o) continue;
            if (try tryCard(dir, i, paths)) |held| {
                var buf: [32]u8 = undefined;
                held.file.setContents(std.fmt.bufPrint(&buf, "{d} zig-xpu\n", .{sys.pid()}) catch "");
                return held;
            }
        }
        if (waited >= wait_ms) return error.Timeout;
        sys.sleepMs(200);
        waited += 200;
    }
}

test "locks exclude one another, free on release, and a pool of two serves two holders" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    const real = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(real);
    // The lock pool is the driver-bound cards, so the fixture marks both bound under a fake sysfs root.
    var d: [96]u8 = undefined;
    for (device.cards) |card| try tmp.dir.createDirPath(io, try std.fmt.bufPrint(&d, "pci/{s}/driver", .{card}));
    var root: [200]u8 = undefined;
    const rl = (try std.fmt.bufPrint(&root, "{s}/pci", .{real})).len;
    const paths: preflight.Paths = .{ .pci_root = root[0..rl] };
    var a = try acquireOfPaths(real, null, null, 0, paths);
    var b = try acquireOfPaths(real, null, null, 0, paths);
    try std.testing.expect(a.index != b.index);
    try std.testing.expectError(error.Timeout, acquireOfPaths(real, null, null, 200, paths));
    const first = a.index;
    a.release();
    var c = try acquireOfPaths(real, null, null, 0, paths);
    try std.testing.expectEqual(first, c.index);
    c.release();
    b.release();
}

test "the lock pool skips a card with no driver bound" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    const real = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(real);
    // Only the second card is bound: both `acquire`s must land on it (the first is skipped, never handed out).
    var d: [96]u8 = undefined;
    try tmp.dir.createDirPath(io, try std.fmt.bufPrint(&d, "pci/{s}/driver", .{device.cards[1]}));
    var root: [200]u8 = undefined;
    const rl = (try std.fmt.bufPrint(&root, "{s}/pci", .{real})).len;
    const paths: preflight.Paths = .{ .pci_root = root[0..rl] };
    var a = try acquireOfPaths(real, null, null, 0, paths);
    try std.testing.expectEqual(@as(usize, 1), a.index);
    try std.testing.expectError(error.Timeout, acquireOfPaths(real, null, null, 200, paths)); // the one card is held
    a.release();
}
