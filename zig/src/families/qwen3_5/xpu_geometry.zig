//! Launch geometry of the Qwen3.5 XPU kernels. Each function mirrors the launch.py next to
//! zig/kernels/xpu/shared/<key>/<key>.cl, and the tests below pin the values launch.py returns.
//! One kernel source serves both Qwen3.5-2B and Qwen 3.8 27B: the model-shape constants are the
//! kernel's preprocessor defaults (2B) or -D overrides (27B), and the grid dimensions here take the
//! runtime shape values (rows, outputs, heads) so the same geometry formula serves both.
const std = @import("std");

/// Global (work-items, a multiple of local) and local size, OpenCL style.
pub const Geo = struct { gx: u32, gy: u32 = 1, gz: u32 = 1, lx: u32, ly: u32 = 1, lz: u32 = 1 };

fn ceilDiv(a: u32, b: u32) u32 {
    return (a + b - 1) / b;
}

/// qwen35_qmm: R rows, N outputs. One work-group of 256 covers QW_TILE = 64 output columns of a row.
pub fn qmm(rows: u32, outputs: u32) Geo {
    return .{ .gx = @max(1, ceilDiv(outputs, 64)) * 256, .gy = rows, .lx = 256 };
}
/// qwen35_norm / qwen35_norm_nores: one work-group of QW_TPG = hidden/QW_ELEMS per row.
pub fn norm(rows: u32, hidden: u32) Geo {
    const tpg = hidden / 16; // QW_ELEMS = 16
    return .{ .gx = tpg, .gy = rows, .lx = tpg };
}
pub fn normNores(rows: u32, hidden: u32) Geo {
    return norm(rows, hidden);
}
/// qwen35_mlp_act: each work-item covers QW_VEC = 8 columns, so ceil(intermediate/8) work-items per row.
/// The grid is rounded up to a whole number of 256-wide work-groups because the runtime rejects a global size that
/// is not a multiple of the local size; the kernel's own `c >= QW_COLS_PER_ROW` guard discards the extra work-items.
/// At the 2B intermediate (6144) the rounded value is the exact one (768); at the 27B 17408 it is 2304 (vs 2176).
pub fn mlpAct(rows: u32, intermediate: u32) Geo {
    const cols = ceilDiv(intermediate, 8); // QW_VEC = 8
    return .{ .gx = ceilDiv(cols, 256) * 256, .gy = rows, .lx = 256 };
}
/// qwen35_gdn_pre: QW_DK = 128 work-items per (head, row); heads3 = 2*QW_NK + QW_NV.
pub fn gdnPre(rows: u32, heads3: u32) Geo {
    return .{ .gx = 128, .gy = heads3, .gz = rows, .lx = 128 };
}
/// qwen35_gdn_post: one 32-lane work-group per (head, row).
pub fn gdnPost(rows: u32, heads: u32) Geo {
    return .{ .gx = 32, .gy = heads, .gz = rows, .lx = 32 };
}
/// qwen35_gdn_chain: one 16-lane sub-group per state row; HV*DV state rows.
pub fn gdnChain(linear_dim: u32, heads: u32) Geo {
    return .{ .gx = 16 * heads * linear_dim, .lx = 128 };
}
/// qwen35_attn_partial: NCH key chunks over W window rows, query heads folded into y as qheads*W.
pub fn attnPartial(nch: u32, w: u32, qheads: u32) Geo {
    return .{ .gx = 16, .gy = qheads * w, .gz = nch, .lx = 16 };
}
/// qwen35_attn_merge: 32 dim slices over qheads x W.
pub fn attnMerge(rows: u32, qheads: u32) Geo {
    return .{ .gx = 32, .gy = qheads, .gz = rows, .lx = 32 };
}
/// layout.cl kernels.
pub const Layout = enum { embed, head_norm, queries, keys, attention_gate };
pub fn layout(k: Layout, n: u32, hidden: u32, head_dim: u32, qheads: u32, kv_heads: u32) Geo {
    return switch (k) {
        .embed => .{ .gx = hidden / 2, .gy = n, .lx = 64 },
        .head_norm => .{ .gx = 64 * n, .lx = 64 },
        .queries => .{ .gx = head_dim, .gy = qheads, .gz = n, .lx = head_dim },
        .keys => .{ .gx = head_dim, .gy = kv_heads, .gz = n, .lx = head_dim },
        .attention_gate => .{ .gx = hidden, .gy = n, .lx = 256 },
    };
}

fn expect(g: Geo, gx: u32, gy: u32, gz: u32, lx: u32) !void {
    try std.testing.expectEqual(Geo{ .gx = gx, .gy = gy, .gz = gz, .lx = lx }, g);
}

test "geometry matches launch.py (2B defaults)" {
    // qmm: (ceil(N/64)*256, R)/(256,1)
    try expect(qmm(1, 2048), 32 * 256, 1, 1, 256);
    try expect(qmm(176, 6144), 96 * 256, 176, 1, 256);
    try expect(qmm(5, 17), 256, 5, 1, 256);
    // norm/norm_nores: hidden/16 work-items per row
    try expect(normNores(7, 2048), 128, 7, 1, 128);
    try expect(norm(4, 2048), 128, 4, 1, 128);
    // mlp_act: intermediate/8 per row
    try expect(mlpAct(3, 6144), 768, 3, 1, 256);
    // gdn
    try expect(gdnPre(4, 48), 128, 48, 4, 128);
    try expect(gdnPost(4, 16), 32, 16, 4, 32);
    try expect(gdnChain(128, 16), 16 * 16 * 128, 1, 1, 128);
    // attention
    try expect(attnPartial(2, 16, 8), 16, 128, 2, 16);
    try expect(attnPartial(1, 1, 8), 16, 8, 1, 16);
    try expect(attnMerge(4, 8), 32, 8, 4, 32);
    // layout (2B: hidden 2048, head_dim 256, qheads 8, kv_heads 2)
    try expect(layout(.head_norm, 16, 2048, 256, 8, 2), 1024, 1, 1, 64);
    try expect(layout(.keys, 3, 2048, 256, 8, 2), 256, 2, 3, 256);
    try expect(layout(.embed, 4, 2048, 256, 8, 2), 1024, 4, 1, 64);
}

test "geometry scales to the 27B shapes" {
    // 27B: hidden 5120, intermediate 17408, head_dim 256, qheads 24, kv_heads 4, k/v heads 16/48.
    try expect(norm(4, 5120), 320, 4, 1, 320);
    try expect(mlpAct(3, 17408), 2304, 3, 1, 256);
    try expect(layout(.attention_gate, 4, 5120, 256, 24, 4), 5120, 4, 1, 256);
    try expect(layout(.queries, 4, 5120, 256, 24, 4), 256, 24, 4, 256);
    try expect(layout(.keys, 4, 5120, 256, 24, 4), 256, 4, 4, 256);
    try expect(gdnPre(4, 2 * 16 + 48), 128, 80, 4, 128);
    try expect(gdnPost(4, 48), 32, 48, 4, 32);
    try expect(gdnChain(128, 48), 16 * 48 * 128, 1, 1, 128);
    try expect(attnPartial(2, 16, 24), 16, 384, 2, 16);
    try expect(attnMerge(4, 24), 32, 24, 4, 32);
}
