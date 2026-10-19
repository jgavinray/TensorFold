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
/// qwen35_qmm_q8 (decode matmul): up to four projections in one launch, each its own run of QW_TILE = 64-column work-groups.
pub fn qmmQ8(rows: u32, outputs: [4]u32) Geo {
    var tiles: u32 = 0;
    for (outputs) |o| tiles += ceilDiv(o, 64);
    return .{ .gx = @max(1, tiles) * 256, .gy = rows, .lx = 256 };
}
/// qwen35_qmm_q8x (prefill matmul, int8 XMX): 256-item work-groups = 16 sub-groups (Q8X_SG) down the rows, each 8 rows
/// (Q8X_RB = 1 row block) x the work-group's 32 columns (Q8X_NT = 2 tiles of 16): 32 columns x 128 rows per work-group.
/// Must match the kernel's Q8X_NT / Q8X_RB / Q8X_SG defaults.
pub fn qmmQ8x(rows: u32, outputs: u32) Geo {
    return .{ .gx = ceilDiv(outputs, 32) * 256, .gy = ceilDiv(rows, 128), .lx = 256 };
}
/// qwen35_qmm_q8xs (decode/verify rows, int8 XMX, byte-identical to q8): 256-item work-groups = 16 sub-groups that
/// split the 32 lane classes of the work-group's 32 columns (Q8XS_NT = 2 tiles of 16) x 8 rows. Must match the kernel's
/// Q8XS_NT / Q8XS_SG.
pub fn qmmQ8xs(rows: u32, outputs: u32) Geo {
    return .{ .gx = ceilDiv(outputs, 32) * 256, .gy = ceilDiv(rows, 8), .lx = 256 };
}
/// qwen35_qmm_split: one 64-item work-group per (quantization group, row).
pub fn qmmSplit(rows: u32, groups: u32) Geo {
    return .{ .gx = groups * 64, .gy = rows, .lx = 64 };
}
/// qwen35_qmm_pair: two same-shape N-column projections of one row, walked in lockstep. A work-group covers
/// QW_HALF_TILE = 32 columns of each projection, so the grid is ceil(N/32) work-groups of 256 per row.
pub fn qmmPair(rows: u32, outputs_per_projection: u32) Geo {
    return .{ .gx = @max(1, ceilDiv(outputs_per_projection, 32)) * 256, .gy = rows, .lx = 256 };
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
/// qwen35_attn_partial_gqa: one work-group per (kv head, window row, chunk) = q_per_kv query heads x 4 key slices,
/// 16 items each (QA_SPLIT = 4 in attn_partial.cl).
pub fn attnPartialGqa(nch: u32, w: u32, kv_heads: u32, q_per_kv: u32) Geo {
    return .{ .gx = 16 * 4 * q_per_kv, .gy = kv_heads * w, .gz = nch, .lx = 16 * 4 * q_per_kv };
}
/// qwen35_attn_prefill_xmx: one 256-item work-group (4 row groups of 8 rows x 4 dim slices) per query head and 32 window
/// rows.
pub fn attnPrefillXmx(w: u32, qheads: u32) Geo {
    return .{ .gx = 256, .gy = qheads * ceilDiv(w, 32), .lx = 256 };
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
        // The kernel covers `qheads * head_dim` columns (one per gate element of a row), not `hidden`: at the 2B
        // geometry heads*head_dim == hidden (8*256 = 2048), so `hidden` happened to be the right extent; at 27B
        // heads*head_dim = 6144 > hidden 5120, and a `hidden`-wide grid left the last 4 heads' gates unwritten.
        .attention_gate => .{ .gx = qheads * head_dim, .gy = n, .lx = 256 },
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
    // qmm_q8: the segments' 64-column tiles back to back (27B GDN a, b, qkv, z: 1 + 1 + 160 + 96)
    try expect(qmmQ8(1, .{ 48, 48, 10240, 6144 }), 258 * 256, 1, 1, 256);
    try expect(qmmQ8(3, .{ 1024, 1024, 12288, 0 }), 224 * 256, 3, 1, 256);
    // qmm_pair: 32 columns of each projection per 256-work-item group
    try expect(qmmPair(1, 16), 256, 1, 1, 256);
    try expect(qmmPair(176, 16), 256, 176, 1, 256);
    try expect(qmmPair(1, 512), 16 * 256, 1, 1, 256);
    try expect(qmmPair(1, 2048), 64 * 256, 1, 1, 256);
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
    // 2B gate width: qheads*head_dim = 8*256 = 2048, which equals hidden here.
    try expect(layout(.attention_gate, 4, 2048, 256, 8, 2), 2048, 4, 1, 256);
}

test "geometry scales to the 27B shapes" {
    // 27B: hidden 5120, intermediate 17408, head_dim 256, qheads 24, kv_heads 4, k/v heads 16/48.
    try expect(norm(4, 5120), 320, 4, 1, 320);
    try expect(mlpAct(3, 17408), 2304, 3, 1, 256);
    // 27B gate width: qheads*head_dim = 24*256 = 6144 > hidden 5120, so `hidden` was the wrong extent.
    try expect(layout(.attention_gate, 4, 5120, 256, 24, 4), 6144, 4, 1, 256);
    try expect(layout(.queries, 4, 5120, 256, 24, 4), 256, 24, 4, 256);
    try expect(layout(.keys, 4, 5120, 256, 24, 4), 256, 4, 4, 256);
    try expect(gdnPre(4, 2 * 16 + 48), 128, 80, 4, 128);
    try expect(gdnPost(4, 48), 32, 48, 4, 32);
    try expect(gdnChain(128, 48), 16 * 48 * 128, 1, 1, 128);
    try expect(attnPartial(2, 16, 24), 16, 384, 2, 16);
    // 27B: 4 kv heads x 6 query heads each, 4 key slices
    try expect(attnPartialGqa(4, 1, 4, 6), 384, 4, 4, 384);
    try expect(qmmSplit(3, 80), 80 * 64, 3, 1, 64);
    // q8x: 32 columns x 128 rows per 256-item work-group (17408 outputs, a 128-row chunk; a 37-row tail)
    try expect(qmmQ8x(128, 17408), 544 * 256, 1, 1, 256);
    try expect(qmmQ8x(37, 48), 2 * 256, 1, 1, 256);
    // q8xs: 32 columns x 8 rows per 256-item work-group (the lm_head, a 2-row window; 8 streams x 2 rows)
    try expect(qmmQ8xs(2, 248320), 7760 * 256, 1, 1, 256);
    try expect(qmmQ8xs(16, 5120), 160 * 256, 2, 1, 256);
    try expect(attnMerge(4, 24), 32, 24, 4, 32);
    // attn_prefill_xmx: 24 heads x 4 blocks of 32 rows (a 128-row chunk); a 37-row tail needs 2 blocks per head
    try expect(attnPrefillXmx(128, 24), 256, 96, 1, 256);
    try expect(attnPrefillXmx(37, 24), 256, 48, 1, 256);
}
