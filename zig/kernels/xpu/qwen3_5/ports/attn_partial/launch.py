"""Launch geometry for qwen35_attn_partial (attn_partial.cl).

One work-group per (query head, window row, 128-key chunk): global (S, 8*W, NCH),
local (S, 1, 1) with S = 16 dim slices. Dimension 0 is an explicit slice index, not a
lane id, so no sub-group width is assumed; 16 items fill one 16-wide Intel sub-group
exactly. Every (y, z) cell owns its whole work-group, so any W >= 1, NCH >= 1 is valid.
"""


def launch(NCH, W):
    return (16, 8 * W, NCH), (16, 1, 1)
