"""Launch geometry for qwen35_qmm.

One work-group of 256 work-items covers 64 output columns of one activation row (8 column sets of
8 columns; each work-item owns one set): the grid is (ceil(N/64), R) work-groups of (256, 1). The
kernel derives column base = group_id(0)*64 + set*8, row = group_id(1).
"""

def launch(R, N, K):
    return ((256 * ((N + 63) // 64), R), (256, 1))
