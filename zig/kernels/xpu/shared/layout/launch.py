"""Launch geometry for layout.cl - the five Qwen3.5 layout kernels.

Metal's grids are reproduced exactly; only the local sizes are chosen to be valid on Xe
(a work-group size must divide the grid extent of dim 0, and QW_HEAD_DIM/QW_HIDDEN are multiples of 32,
so any sub-group width of 16 or 32 is fine). Defaults are the Qwen3.5-2B geometry.

  embed          (QW_HIDDEN/2, R) local (64, 1, 1)   64 bytes per work-item = one 64-element dequant group
  head_norm      (64*rows*heads) local (64, 1, 1)  one work-group per (token, head) row, 64 items x 4 dims = QW_HEAD_DIM
  queries/keys   (QW_HEAD_DIM, H, R) local (QW_HEAD_DIM, 1, 1)  dim 0 is one head
  attention_gate (QW_HIDDEN, rows) local (256, 1, 1) dim 0 covers the QW_HEADS heads x QW_HEAD_DIM dims of a row

The 27B build overrides QW_HIDDEN (5120), QW_HEADS (24) and QW_KV_HEADS (4); the head_norm grid is passed
in by the caller as rows*heads, so only embed and attention_gate read the hidden size here.
"""

HEAD_DIM = 256
HIDDEN = 2048
EMBED_BYTES = HIDDEN // 2  # 1024 bytes per embedding row


def launch(kernel, n):
    if kernel == "embed":                      # n = rows R
        return (EMBED_BYTES, n, 1), (64, 1, 1)
    if kernel == "head_norm":                  # n = rows * heads (one group per row, 64 work-items)
        return (64 * n, 1, 1), (64, 1, 1)
    if kernel == "queries":                    # n = R, QW_HEADS heads, dims[1] = cache capacity
        return (HEAD_DIM, 8, n), (HEAD_DIM, 1, 1)
    if kernel == "keys":                       # n = R, QW_KV_HEADS heads, dims[1] = cache capacity
        return (HEAD_DIM, 2, n), (HEAD_DIM, 1, 1)
    if kernel == "attention_gate":             # n = rows
        return (HIDDEN, n, 1), (256, 1, 1)
    raise KeyError(kernel)
