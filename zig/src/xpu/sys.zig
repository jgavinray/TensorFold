//! Thin libc file access (read-only /sys and /proc reads, flock): no allocator, no Io instance needed.
const std = @import("std");

const O_RDONLY = 0;
const O_WRONLY = 1;
const O_CREAT = 0o100;
const O_CLOEXEC = 0o2000000;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;
extern "c" fn ftruncate(fd: c_int, len: c_long) c_int;
extern "c" fn flock(fd: c_int, op: c_int) c_int;
extern "c" fn mkdir(path: [*:0]const u8, mode: c_uint) c_int;
extern "c" fn usleep(us: c_uint) c_int;
extern "c" fn getpid() c_int;
extern "c" fn access(path: [*:0]const u8, mode: c_int) c_int;

pub fn exists(path: [*:0]const u8) bool {
    return access(path, 0) == 0;
}

/// Reads up to buf.len bytes of a small file; null when it cannot be opened.
pub fn readSmall(path: [*:0]const u8, buf: []u8) ?[]const u8 {
    const fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return null;
    defer _ = close(fd);
    var n: usize = 0;
    while (n < buf.len) {
        const got = read(fd, buf.ptr + n, buf.len - n);
        if (got <= 0) break;
        n += @intCast(got);
    }
    return buf[0..n];
}

pub const File = struct {
    fd: c_int,

    /// Opens (creating, never truncating) for locking.
    pub fn openLock(path: [*:0]const u8) ?File {
        const fd = open(path, O_WRONLY | O_CREAT | O_CLOEXEC, @as(c_uint, 0o644));
        return if (fd < 0) null else .{ .fd = fd };
    }
    /// Non-blocking exclusive flock; false when another holder has it.
    pub fn tryLock(self: File) bool {
        return flock(self.fd, 2 | 4) == 0; // LOCK_EX | LOCK_NB
    }
    pub fn unlockClose(self: File) void {
        _ = flock(self.fd, 8); // LOCK_UN
        _ = close(self.fd);
    }
    pub fn setContents(self: File, text: []const u8) void {
        _ = ftruncate(self.fd, 0);
        _ = write(self.fd, text.ptr, text.len);
    }
};

pub fn makeDir(path: [*:0]const u8) void {
    _ = mkdir(path, 0o755);
}
pub fn sleepMs(ms: u32) void {
    _ = usleep(ms * 1000);
}
pub fn pid() c_int {
    return getpid();
}

/// Formats into `buf` with a trailing NUL; null when it does not fit.
pub fn zfmt(buf: []u8, comptime fmt: []const u8, args: anytype) ?[:0]u8 {
    const s = std.fmt.bufPrint(buf[0 .. buf.len - 1], fmt, args) catch return null;
    buf[s.len] = 0;
    return buf[0..s.len :0];
}
