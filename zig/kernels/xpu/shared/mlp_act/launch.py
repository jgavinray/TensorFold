"""Launch geometry for qwen35_mlp_act (mlp_act.cl): each work-item covers QW_VEC columns.

Defaults are the Qwen3.5-2B geometry: N = 6144, VEC = 8, so global = (768, M), local = (256, 1);
768 % 256 == 0, and any M is valid because the y local size is 1. A 27B build that overrides
QW_INTERMEDIATE must use the matching COLS_PER_ROW (see the defines at the top of mlp_act.cl).
"""

VEC = 8
INTERMEDIATE = 6144
COLS_PER_ROW = INTERMEDIATE // VEC  # 768 vectorized work-items per row


def launch(M):
    return (COLS_PER_ROW, M), (256, 1)
