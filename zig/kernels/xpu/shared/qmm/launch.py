"""Launch geometry for qwen35_qmm.

One work-group of QW_WG work-items covers QW_TILE output columns of one activation row (QW_SETS column
sets of 8 columns; each work-item owns one set): the grid is (ceil(N/QW_TILE), R) work-groups of
(QW_WG, 1). The kernel derives column base = group_id(0)*QW_TILE + set*8, row = group_id(1).

N, K and R are runtime arguments, so this geometry serves the 2B and 27B shape sets alike; only the
tile constants (QW_SETS = 8, QW_LANES = 32, QW_TILE = 64) are fixed and match the kernel defaults.
"""

SETS = 8
LANES = 32
TILE = SETS * 8   # 64 columns per work-group
WG = SETS * LANES  # 256 work-items


def launch(R, N, K):
    return ((WG * ((N + TILE - 1) // TILE), R), (WG, 1))
