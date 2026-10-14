"""Launch geometry for qwen35_norm_nores: one work-group of 128 work-items per row (16 elements each)."""

K = 2048
E = 16
TPG = K // E  # 128


def launch(M):
    return ((TPG, M), (TPG, 1))
