//! Launch geometry of the Qwen3.5 XPU kernels per variant. A kernel's geometry comes with its variant: the reference
//! kernels (zig/kernels/xpu/qwen3_5/reference/<key>/) and the accepted ports (.../ports/<key>/) are launched differently.
//! Each function mirrors the launch.py next to the kernel; the tests below pin the values launch.py returns.
//! Adding a port: drop ports/<key>/<key>.cl + launch.py, add <key> to `port_keys` in zig/build/xpu.zig, and give the
//! matching function here a `.port` arm (a function with no arm for a variant means both variants launch alike).
const std = @import("std");

pub const Variant = enum { reference, port };
/// Global (work-items, a multiple of local) and local size, OpenCL style.
pub const Geo = struct { gx: u32, gy: u32 = 1, gz: u32 = 1, lx: u32, ly: u32 = 1, lz: u32 = 1 };

fn ceilDiv(a: u32, b: u32) u32 {
    return (a + b - 1) / b;
}

/// qwen35_qmm: R rows, N outputs.
pub fn qmm(v: Variant, rows: u32, outputs: u32) Geo {
    return switch (v) {
        .reference => .{ .gx = ceilDiv(outputs, 64) * 64, .gy = rows, .lx = 64 },
        .port => .{ .gx = @max(1, ceilDiv(outputs, 32)) * 256, .gy = @max(1, ceilDiv(rows, 64)), .lx = 256 },
    };
}
pub fn normNores(_: Variant, rows: u32) Geo {
    return .{ .gx = 128, .gy = rows, .lx = 128 };
}
pub fn norm(_: Variant, rows: u32) Geo {
    return .{ .gx = 128, .gy = rows, .lx = 128 };
}
pub fn mlpAct(v: Variant, rows: u32, intermediate: u32) Geo {
    return switch (v) {
        .reference => .{ .gx = intermediate, .gy = rows, .lx = 256 },
        .port => .{ .gx = intermediate / 8, .gy = rows, .lx = 256 },
    };
}
pub fn gdnPre(_: Variant, rows: u32, heads3: u32) Geo {
    return .{ .gx = 128, .gy = heads3, .gz = rows, .lx = 128 };
}
pub fn gdnPost(v: Variant, rows: u32, heads: u32) Geo {
    return switch (v) {
        .reference => .{ .gx = 128, .gy = heads, .gz = rows, .lx = 128 },
        .port => .{ .gx = 32, .gy = heads, .gz = rows, .lx = 32 },
    };
}
pub fn gdnChain(_: Variant, linear_dim: u32, heads: u32) Geo {
    return .{ .gx = linear_dim, .gy = heads, .lx = 64 };
}
pub fn attnMerge(v: Variant, rows: u32, qheads: u32) Geo {
    return switch (v) {
        .reference => .{ .gx = 256, .gy = qheads, .gz = rows, .lx = 256 },
        .port => .{ .gx = 32, .gy = qheads, .gz = rows, .lx = 32 },
    };
}
/// layout.cl kernels have the same geometry in both variants.
pub const Layout = enum { embed, head_norm, queries, keys, attention_gate };
pub fn layout(_: Variant, k: Layout, n: u32) Geo {
    return switch (k) {
        .embed => .{ .gx = 1024, .gy = n, .lx = 64 },
        .head_norm => .{ .gx = 64 * n, .lx = 64 },
        .queries => .{ .gx = 256, .gy = 8, .gz = n, .lx = 256 },
        .keys => .{ .gx = 256, .gy = 2, .gz = n, .lx = 256 },
        .attention_gate => .{ .gx = 2048, .gy = n, .lx = 256 },
    };
}

fn expect(g: Geo, gx: u32, gy: u32, gz: u32, lx: u32) !void {
    try std.testing.expectEqual(Geo{ .gx = gx, .gy = gy, .gz = gz, .lx = lx }, g);
}

test "geometry matches launch.py (reference and ports)" {
    // qmm: reference ((N+63)//64*64, R)/(64,1); port (ceil(N/32)*256, ceil(R/64))/(256,1)
    try expect(qmm(.reference, 1, 2048), 2048, 1, 1, 64);
    try expect(qmm(.port, 1, 2048), 64 * 256, 1, 1, 256);
    try expect(qmm(.port, 176, 6144), 192 * 256, 3, 1, 256);
    try expect(qmm(.reference, 176, 248320), 248320, 176, 1, 64);
    try expect(qmm(.port, 5, 17), 256, 1, 1, 256);
    // mlp_act: (6144, M)/256 vs (768, M)/256
    try expect(mlpAct(.reference, 3, 6144), 6144, 3, 1, 256);
    try expect(mlpAct(.port, 3, 6144), 768, 3, 1, 256);
    try expect(gdnPost(.reference, 4, 16), 128, 16, 4, 128);
    try expect(gdnPost(.port, 4, 16), 32, 16, 4, 32);
    try expect(attnMerge(.reference, 4, 8), 256, 8, 4, 256);
    try expect(attnMerge(.port, 4, 8), 32, 8, 4, 32);
    try expect(gdnPre(.port, 4, 48), 128, 48, 4, 128);
    try expect(normNores(.port, 7), 128, 7, 1, 128);
    try expect(layout(.port, .head_norm, 16), 1024, 1, 1, 64);
    try expect(layout(.port, .keys, 3), 256, 2, 3, 256);
}
