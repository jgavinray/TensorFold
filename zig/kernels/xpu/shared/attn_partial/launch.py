"""Launch geometry for qwen35_attn_partial (attn_partial.cl).

One work-group per (query head, window row, QW_CHUNK-key chunk): global (S, QW_Q_HEADS*W, NCH),
local (S, 1, 1) with S = 16 dim slices. Dimension 0 is an explicit slice index, not a lane id, so no
sub-group width is assumed; 16 items fill one 16-wide Intel sub-group exactly. Every (y, z) cell owns
its whole work-group, so any W >= 1, NCH >= 1 is valid.

Defaults are the Qwen3.5-2B geometry (QW_Q_HEADS = 8); the 27B build passes QW_Q_HEADS = 24.
"""

Q_HEADS = 8
S = 16  # dim slices per work-group


def launch(NCH, W):
    return (S, Q_HEADS * W, NCH), (S, 1, 1)
