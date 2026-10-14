"""Launch geometry for layout.cl - the five Qwen3.5 layout kernels.

Metal's grids are reproduced exactly; only the local sizes are chosen to be valid on Xe
(a work-group size must divide the grid extent of dim 0, and 256/32/64 are all multiples of 32,
so any sub-group width of 16 or 32 is fine).

  embed          (1024, R)   local (64, 1, 1)   64 bytes per work-item = one 64-element dequant group
  head_norm      (64*rows*heads) local (64, 1, 1)  one work-group per (token, head) row, 64 items x 4 dims = 256
  queries/keys   (256, H, R) local (256, 1, 1)  dim 0 is one 256-wide head
  attention_gate (2048, rows) local (256, 1, 1) dim 0 covers the 8 heads x 256 dims of a row
"""


def launch(kernel, n):
    if kernel == "embed":                      # n = rows R
        return (1024, n, 1), (64, 1, 1)
    if kernel == "head_norm":                  # n = rows * heads (one group per row, 64 work-items)
        return (64 * n, 1, 1), (64, 1, 1)
    if kernel == "queries":                    # n = R, 8 heads, dims[1] = cache capacity
        return (256, 8, n), (256, 1, 1)
    if kernel == "keys":                       # n = R, 2 heads, dims[1] = cache capacity
        return (256, 2, n), (256, 1, 1)
    if kernel == "attention_gate":             # n = rows
        return (2048, n, 1), (256, 1, 1)
    raise KeyError(kernel)
