"""Launch geometry for qwen35_qmm_pair.

Two same-shape N-column int4 projections of one activation row are walked in lockstep: one work-group
of QW_WG work-items covers QW_HALF_TILE = 32 columns of each projection (QW_SETS_PER_PROJ = 4 column
sets of 8 columns per projection; each work-item owns one set), so the grid is
(ceil(N/QW_HALF_TILE), R) work-groups of (QW_WG, 1). The kernel derives the projection from the set
index (sets below QW_SETS_PER_PROJ belong to A, the rest to B) and column base =
group_id(0)*QW_HALF_TILE + (set % QW_SETS_PER_PROJ)*8.

N, K and R are runtime arguments, so this geometry serves the 2B and 27B shape sets alike; only the
tile constants (QW_SETS = 8, QW_LANES = 32, QW_HALF_TILE = 32) are fixed and match the kernel defaults.
The pairs in use are the GatedDeltaNet a/b (N = 16) and the attention k/v (N = 512).
"""

SETS = 8
LANES = 32
HALF_TILE = SETS * 8 // 2   # 32 columns of each projection per work-group
WG = SETS * LANES           # 256 work-items


def launch(R, N, K):
    return ((WG * ((N + HALF_TILE - 1) // HALF_TILE), R), (WG, 1))
