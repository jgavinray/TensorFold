"""Launch geometry for qwen35_attn_partial (attn_partial.cl).

One work-group per (query head qh, window row w, 128-key chunk c): global (S, 8*W, NCH),
local (S, 1, 1) with S = 16 dim slices. Dimension 0 is an explicit dim-slice index, not a lane id,
so no sub-group width is assumed: 16 work-items fill exactly one 16-wide Intel sub-group. Each
(y, z) cell owns its whole work-group, so any W >= 1, NCH >= 1 is valid.
"""


def launch(NCH, W):
    return (16, 8 * W, NCH), (16, 1, 1)
