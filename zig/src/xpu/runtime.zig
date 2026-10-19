//! Complete model-free Level Zero runtime admission: the loader is dlopen'ed, every symbol the runtime uses must
//! resolve and zeInit must succeed before any value escapes. Public surface of zig/src/xpu/.
const std = @import("std");
pub const abi = @import("abi.zig");
pub const device = @import("device.zig");
pub const preflight = @import("preflight.zig");
pub const lock = @import("lock.zig");
pub const context = @import("context.zig");
pub const memory = @import("memory.zig");
pub const queue = @import("queue.zig");
pub const cmdlist = @import("cmdlist.zig");
pub const module = @import("module.zig");
pub const launch = @import("launch.zig");
pub const event = @import("event.zig");
pub const session = @import("session.zig");

pub const Error = error{ DriverUnavailable, MissingSymbol, ZeFailed, Invalid, ForbiddenCard, NoSuchDevice, OutOfDeviceMemory, DeviceLost };
pub const loader_name = "libze_loader.so.1";

/// ze_result_t of the most recent failed call on this thread, for logs.
pub threadlocal var last_result: abi.Result = 0;

pub const Runtime = struct {
    lib: std.DynLib,
    api: abi.Api,

    pub fn open() Error!Runtime {
        return openPath(loader_name);
    }

    pub fn openPath(path: []const u8) Error!Runtime {
        var lib = std.DynLib.open(path) catch return error.DriverUnavailable;
        errdefer lib.close();
        const api = try @import("symbols.zig").resolve(abi.Api, &lib);
        // ZE_INIT_FLAG_GPU_ONLY: no VPU or other driver families are initialised.
        try check(api.zeInit(1));
        return .{ .lib = lib, .api = api };
    }

    pub fn close(self: *Runtime) void {
        self.lib.close();
        self.* = undefined;
    }
};

pub fn check(result: abi.Result) Error!void {
    if (result != abi.success) {
        last_result = result;
        return error.ZeFailed;
    }
}

/// Set by the first failed call that puts work on the card or waits for it. After it the process's GPU context may be
/// faulted and its queues banned (xe resets a faulting job's exec queues), and a process that keeps submitting re-arms
/// the driver's reset path, which can wedge the card: 2026-10-19 19:36 a test whose kernel read out of bounds went on
/// to its next measurement - 42 more faults and 7 exec queue resets. From then on every submission or wait in this
/// process fails at once with error.DeviceLost, without reaching the driver.
pub var poisoned = std.atomic.Value(bool).init(false);

/// Call before a submission or wait: error.DeviceLost once the process is poisoned.
pub fn live() Error!void {
    if (poisoned.load(.acquire)) return error.DeviceLost;
}

/// `check` for a submission or wait: a failure poisons the process (see `poisoned`).
pub fn submit(result: abi.Result) Error!void {
    if (result != abi.success) {
        last_result = result;
        poisoned.store(true, .release);
        std.log.warn("xpu: a GPU submission or wait failed (ze result 0x{x}): no further GPU work in this process", .{result});
        return error.DeviceLost;
    }
}

test "complete runtime refuses an absent library" {
    try std.testing.expectError(error.DriverUnavailable, Runtime.openPath("/nonexistent/tensorfold-test/libze_loader.so"));
}

test "a failed submission poisons the process: every later submission fails without reaching the driver" {
    defer poisoned.store(false, .release);
    try live();
    try submit(0);
    try live();
    try std.testing.expectError(error.DeviceLost, submit(0x70000001));
    try std.testing.expectEqual(@as(abi.Result, 0x70000001), last_result);
    try std.testing.expectError(error.DeviceLost, live());
    try std.testing.expectError(error.DeviceLost, live());
}

test "a failed call cannot become success and remembers its code" {
    try check(0);
    try std.testing.expectError(error.ZeFailed, check(0x78000001));
    try std.testing.expectEqual(@as(abi.Result, 0x78000001), last_result);
}
