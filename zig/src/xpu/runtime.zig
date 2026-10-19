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

pub const Error = error{ DriverUnavailable, MissingSymbol, ZeFailed, Invalid, ForbiddenCard, NoSuchDevice, OutOfDeviceMemory };
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

test "complete runtime refuses an absent library" {
    try std.testing.expectError(error.DriverUnavailable, Runtime.openPath("/nonexistent/tensorfold-test/libze_loader.so"));
}

test "a failed call cannot become success and remembers its code" {
    try check(0);
    try std.testing.expectError(error.ZeFailed, check(0x78000001));
    try std.testing.expectEqual(@as(abi.Result, 0x78000001), last_result);
}
