"""Flash Next's XPU key/value cache: bf16 or ExLlamaV3-style quantized KV (cq8 / cq4: fp8-abscissae
midpoint grids), with the row-packing layout the engine's attention reads.

The cache itself (``KVCache``) and every reference kernel here are plain torch on the tensors'
device, so the CUDA engine's implementations are shared, not restated: the row-packing sizes, the
group quantizer and its inverse are one source of truth that both engines' tests pin.
``tests/cuda/test_qwen4_exp_kvquant.py`` checks the same bits on both sides."""

from __future__ import annotations

from ..cuda.kvcache import (GROUP, SCALE_DTYPE, DTYPES, BITS_OF, KVCache, check, row_bytes,
                            h32_ref, pack_nibbles, unpack_nibbles, quantize_ref, dequant_ref)

__all__ = ["GROUP", "SCALE_DTYPE", "DTYPES", "BITS_OF", "KVCache", "check", "row_bytes",
           "h32_ref", "pack_nibbles", "unpack_nibbles", "quantize_ref", "dequant_ref"]
