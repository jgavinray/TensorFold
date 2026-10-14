//! Card lock shared with the Python harness: flock on <lock_dir>/card-<bdf>.lock; take a free card of the pool or wait.
const std = @import("std");
const device = @import("device.zig");
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

/// Tries to lock one specific card without waiting.
pub fn tryCard(dir: []const u8, index: usize) Error!?Held {
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
            if (try tryCard(dir, i)) |held| {
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
    const real = try tmp.dir.realPathFileAlloc(std.testing.io, ".", std.testing.allocator);
    defer std.testing.allocator.free(real);
    var a = try acquire(real, null, 0);
    var b = try acquire(real, null, 0);
    try std.testing.expect(a.index != b.index);
    try std.testing.expectError(error.Timeout, acquire(real, null, 200));
    const first = a.index;
    a.release();
    var c = try acquire(real, null, 0);
    try std.testing.expectEqual(first, c.index);
    c.release();
    b.release();
}
