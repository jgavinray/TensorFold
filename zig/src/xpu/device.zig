//! Device selection by PCI address: exactly the project's two Intel Arc Pro B70s, nothing else.
const std = @import("std");
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");

pub const Bdf = struct {
    domain: u32,
    bus: u32,
    device: u32,
    function: u32,

    pub fn parse(text: []const u8) error{Invalid}!Bdf {
        // dddd:bb:dd.f
        if (text.len != 12 or text[4] != ':' or text[7] != ':' or text[10] != '.') return error.Invalid;
        const p = std.fmt.parseInt;
        return .{
            .domain = p(u32, text[0..4], 16) catch return error.Invalid,
            .bus = p(u32, text[5..7], 16) catch return error.Invalid,
            .device = p(u32, text[8..10], 16) catch return error.Invalid,
            .function = p(u32, text[11..12], 16) catch return error.Invalid,
        };
    }

    pub fn eql(a: Bdf, b: Bdf) bool {
        return a.domain == b.domain and a.bus == b.bus and a.device == b.device and a.function == b.function;
    }

    pub fn fromAddress(a: abi.PciAddress) Bdf {
        return .{ .domain = a.domain, .bus = a.bus, .device = a.device, .function = a.function };
    }

    pub fn format(self: Bdf, buf: *[12]u8) []const u8 {
        return std.fmt.bufPrint(buf, "{x:0>4}:{x:0>2}:{x:0>2}.{x}", .{ self.domain, self.bus, self.device, self.function }) catch unreachable;
    }
};

/// The only cards this runtime will open (same pool as exp/harness/xpu_harness.py CARDS).
pub const cards = [_][]const u8{ "0000:08:00.0", "0000:44:00.0" };

pub fn isAllowed(bdf: []const u8) bool {
    for (cards) |c| if (std.mem.eql(u8, c, bdf)) return true;
    return false;
}

pub const Device = struct {
    r: *const runtime.Runtime,
    driver: abi.Driver,
    handle: abi.Device,
    bdf: Bdf,
    props: abi.DeviceProperties,

    pub fn name(self: *const Device) []const u8 {
        return std.mem.sliceTo(&self.props.name, 0);
    }
};

/// Finds the Intel GPU at `bdf_text`. The address is checked against the allow-list before the driver is touched;
/// devices of any other vendor are skipped before their PCI address is even read.
pub fn select(r: *const runtime.Runtime, bdf_text: []const u8) runtime.Error!Device {
    if (!isAllowed(bdf_text)) return error.ForbiddenCard;
    const want = Bdf.parse(bdf_text) catch return error.ForbiddenCard;
    var n_drivers: u32 = 0;
    try runtime.check(r.api.zeDriverGet(&n_drivers, null));
    if (n_drivers == 0 or n_drivers > 8) return error.NoSuchDevice;
    var drivers: [8]abi.Driver = undefined;
    try runtime.check(r.api.zeDriverGet(&n_drivers, &drivers));
    for (drivers[0..n_drivers]) |drv| {
        var n_dev: u32 = 0;
        try runtime.check(r.api.zeDeviceGet(drv, &n_dev, null));
        if (n_dev == 0 or n_dev > 16) continue;
        var devs: [16]abi.Device = undefined;
        try runtime.check(r.api.zeDeviceGet(drv, &n_dev, &devs));
        for (devs[0..n_dev]) |dev| {
            var props: abi.DeviceProperties = .{};
            try runtime.check(r.api.zeDeviceGetProperties(dev, &props));
            if (props.vendorId != abi.intel_vendor_id or props.type != abi.device_type_gpu) continue;
            var pci: abi.PciExtProperties = .{};
            try runtime.check(r.api.zeDevicePciGetPropertiesExt(dev, &pci));
            const got = Bdf.fromAddress(pci.address);
            if (got.eql(want)) return .{ .r = r, .driver = drv, .handle = dev, .bdf = got, .props = props };
        }
    }
    return error.NoSuchDevice;
}

test "BDF parsing and the allow-list accept exactly the two B70s" {
    try std.testing.expect(isAllowed("0000:08:00.0"));
    try std.testing.expect(isAllowed("0000:44:00.0"));
    for ([_][]const u8{ "0000:01:00.0", "0000:c1:00.0", "0000:08:00.1", "08:00.0", "", "0000:44:00.0 " }) |bad|
        try std.testing.expect(!isAllowed(bad));
    const b = try Bdf.parse("0000:44:00.0");
    try std.testing.expectEqual(@as(u32, 0x44), b.bus);
    var buf: [12]u8 = undefined;
    try std.testing.expectEqualStrings("0000:44:00.0", b.format(&buf));
    try std.testing.expectError(error.Invalid, Bdf.parse("0000:4g:00.0"));
}
