"""Launch geometry for qwen35_attn_merge (attn_merge.cl).

Mirrors the Metal grid: global (32, 8, W) = (8-dim slice, query head, window row),
local (32, 1, 1). Each work-item covers 8 contiguous dims (32 slices x 8 = 256);
32 % 32 == 0 and W % 1 == 0, so any W >= 1 is valid.
"""


def launch(W):
    return (32, 8, W), (32, 1, 1)
