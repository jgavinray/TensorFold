//! Mock-backed Level Zero admission tests: no GPU, no real driver. The mock (zig/tests/xpu_mock.c) exposes two Intel
//! GPUs at the B70 addresses and one foreign-vendor GPU at 0000:c1:00.0 that must never be opened.
const std = @import("std");
const rt = @import("runtime.zig");
const fixtures = @import("xpu_fixtures");

test {
    _ = @import("abi.zig");
    _ = @import("runtime.zig");
    _ = @import("device.zig");
    _ = @import("preflight.zig");
    _ = @import("lock.zig");
    _ = @import("event.zig");
    _ = @import("launch.zig");
    _ = @import("session.zig");
    _ = @import("memory.zig").Buffer.alloc;
    _ = @import("module.zig").Module.load;
    _ = @import("cmdlist.zig").Recorder.init;
}

test "a library missing a Level Zero symbol is refused whole" {
    try std.testing.expectError(error.MissingSymbol, rt.Runtime.openPath(fixtures.missing));
}

test "zeInit failure is refused and init asks for GPU drivers only" {
    try std.testing.expectError(error.ZeFailed, rt.Runtime.openPath(fixtures.failed));
    var r = try rt.Runtime.openPath(fixtures.success);
    defer r.close();
    const flags = r.lib.lookup(*const fn () callconv(.c) u32, "tf_mock_init_flags").?;
    try std.testing.expectEqual(@as(u32, 1), flags());
}

test "device selection by PCI address finds exactly the two B70s and refuses everything else" {
    var r = try rt.Runtime.openPath(fixtures.success);
    defer r.close();
    const foreign = r.lib.lookup(*const fn () callconv(.c) c_int, "tf_mock_foreign_pci_queries").?;
    for (rt.device.cards) |card| {
        const d = try rt.device.select(&r, card);
        var buf: [12]u8 = undefined;
        try std.testing.expectEqualStrings(card, d.bdf.format(&buf));
        try std.testing.expectEqual(@as(u32, 0x8086), d.props.vendorId);
    }
    for ([_][]const u8{ "0000:c1:00.0", "0000:01:00.0", "0000:08:00.1", "garbage" }) |bad|
        try std.testing.expectError(error.ForbiddenCard, rt.device.select(&r, bad));
    // The foreign-vendor device was skipped by vendor id before its PCI address was ever read.
    try std.testing.expectEqual(@as(c_int, 0), foreign());
}

test "the command list is created in-order on the compute group" {
    var r = try rt.Runtime.openPath(fixtures.success);
    defer r.close();
    const d = try rt.device.select(&r, "0000:44:00.0");
    var ctx = try rt.context.Context.init(d);
    defer ctx.deinit();
    try std.testing.expectEqual(@as(u32, 1), ctx.compute_ordinal);
    var q = try rt.queue.Queue.init(&ctx);
    defer q.deinit();
    const flags = r.lib.lookup(*const fn () callconv(.c) u32, "tf_mock_last_queue_flags").?;
    try std.testing.expectEqual(rt.abi.queue_flag_in_order, flags() & rt.abi.queue_flag_in_order);
}

test "allocation failure and oversize allocation are errors, not buffers" {
    var r = try rt.Runtime.openPath(fixtures.success);
    defer r.close();
    const d = try rt.device.select(&r, "0000:08:00.0");
    var ctx = try rt.context.Context.init(d);
    defer ctx.deinit();
    try std.testing.expectError(error.ZeFailed, rt.memory.Buffer.alloc(&ctx, .device, 4096));
    try std.testing.expectError(error.Invalid, rt.memory.Buffer.alloc(&ctx, .device, 0));
    try std.testing.expectError(error.Invalid, rt.memory.Buffer.alloc(&ctx, .device, 1 << 40));
}

test "a regular list is created on the compute group and a replay submits exactly one list" {
    var r = try rt.Runtime.openPath(fixtures.success);
    defer r.close();
    const d = try rt.device.select(&r, "0000:44:00.0");
    var ctx = try rt.context.Context.init(d);
    defer ctx.deinit();
    var rec = try rt.cmdlist.Recorder.init(&ctx);
    const group = r.lib.lookup(*const fn () callconv(.c) u32, "tf_mock_last_list_group").?;
    try std.testing.expectEqual(ctx.compute_ordinal, group());

    var q = try rt.cmdlist.Queue.init(&ctx);
    defer q.deinit();
    const qgroup = r.lib.lookup(*const fn () callconv(.c) u32, "tf_mock_last_queue_group").?;
    try std.testing.expectEqual(ctx.compute_ordinal, qgroup());

    var list = try rt.cmdlist.Recorded.take(&rec);
    defer list.deinit();
    const executes = r.lib.lookup(*const fn () callconv(.c) c_int, "tf_mock_executes").?;
    const submitted = r.lib.lookup(*const fn () callconv(.c) u32, "tf_mock_last_execute_lists").?;
    try std.testing.expectEqual(@as(c_int, 0), executes());
    try list.replaySync(&q);
    try std.testing.expectEqual(@as(c_int, 1), executes());
    try std.testing.expectEqual(@as(u32, 1), submitted());
    // A second replay re-submits the same closed list: the recorded step is immutable.
    try list.replaySync(&q);
    try std.testing.expectEqual(@as(c_int, 2), executes());
    try std.testing.expectEqual(@as(u32, 1), submitted());
}
