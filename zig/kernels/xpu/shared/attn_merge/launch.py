"""Launch geometry for qwen35_attn_merge (attn_merge.cl).

Mirrors the Metal grid: global (32, QW_Q_HEADS, W) = (DPL-dim slice, query head, window row),
local (32, 1, 1). Each work-item covers QW_HEAD_DIM/32 = 8 contiguous dims (32 slices x 8 = 256);
32 % 32 == 0 and W % 1 == 0, so any W >= 1 is valid.

Defaults are the Qwen3.5-2B geometry (QW_Q_HEADS = 8); the 27B build passes QW_Q_HEADS = 24.
"""

Q_HEADS = 8


def launch(W):
    return (32, Q_HEADS, W), (32, 1, 1)
