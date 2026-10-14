"""Launch geometry for qwen35_mlp_act (mlp_act.cl): each work-item covers 8 columns.

global = (6144 / 8, M) = (768, M), local = (256, 1): 768 % 256 == 0, any M is valid
because the y local size is 1.
"""

COLS_PER_ROW = 6144 // 8  # 768 vectorized work-items per row


def launch(M):
    return (COLS_PER_ROW, M), (256, 1)
