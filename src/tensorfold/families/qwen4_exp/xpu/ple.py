"""The PLE n-gram rows on Intel GPUs: NGram's host ids staged over pinned memory and dequantized on the card.

The tables stay memory-mapped on the host (``host_table``); a gather slices only the requested rows, parks
them in a pinned staging buffer and copies them to the device on the current stream (``cuda/direct_read``'s
pattern). The row unpack runs on the device, per shard layout:

- ``HostTable``/``SSDTable`` (MLX 4-bit): raw words, scales and biases go over; nibble x scale + bias in
  fp32, rounded once to bf16, then the table scale (``cuda/glue._ple_embed``'s math in torch ops).
- ``BF16Table``: the bf16 bits as stored, times the table scale when there is one.
- ``FP8Table``: raw e4m3 codes; the dequant is code x table scale in fp32 rounded once to bf16 (e4m3's two
  NaN codes pinned to the LUT's canonical value). The table keeps no fp32 scale - its host dequant is a
  256-entry bf16 LUT - so the scale is read back from the LUT's code 0x38 (e4m3 1.0) and proved against all
  256 codes on the card once per table; the arithmetic dequant runs only where it reproduces the LUT bit
  for bit, the LUT gather otherwise.
- ``NVFP4Table``: raw e2m1 codes and e4m3 block scales; code x block x table scale in fp32, one bf16 rounding.
"""

from __future__ import annotations

from weakref import WeakKeyDictionary

import numpy as np
import torch

from tensorfold.cuda.direct_read import _current_stream, _event

from ..host_table import BF16Table, FP8Table, HostTable, NVFP4Table, _copy_rows
from ..ssd_table import SSDTable
from .weight_types import PLEW

_STAGING: dict[torch.device, list] = {}     # device -> [pinned staging of copies in flight, and their events]
_KEEP = 8                                   # staging buffers held before the oldest is waited on and dropped
_DTYPES = {np.dtype(np.uint8): torch.uint8, np.dtype(np.int16): torch.int16, np.dtype(np.int32): torch.int32}
_FP8: WeakKeyDictionary = WeakKeyDictionary()     # FP8Table -> {device: ("scale", tensor) or ("lut", tensor)}


def ids(ple: PLEW, history, tokens) -> torch.Tensor:
    """The lookup's row ids [L, heads] int64: ``NGram.ids`` on the host, the oracle the tests check."""

    return torch.from_numpy(np.ascontiguousarray(ple.ngram.ids(history, tokens)))


def gather(ple: PLEW, ids, device) -> torch.Tensor:
    """Row ids [L, heads] -> the table's rows [L, heads, dims] bf16 on ``device`` (host mmap slice, pinned staging,
    one async copy per component, dequant on the card). ``ids`` comes from ``NGram.ids`` (or :func:`ids`)."""

    device = torch.device(device)
    table, ng = ple.table, ple.ngram
    flat = np.asarray(ids.detach().cpu() if isinstance(ids, torch.Tensor) else ids, dtype=np.int64).reshape(-1)
    if isinstance(table, (HostTable, SSDTable)):
        words, scales, biases = table.gather(flat)            # the raw 4-bit rows: the unpack runs on the card
        rows = _unpack4(_to_device(words.view(np.int32), device),
                        _to_device(scales.view(np.int16), device).view(torch.bfloat16),
                        _to_device(biases.view(np.int16), device).view(torch.bfloat16), ng.dims)
        scale = float(getattr(table, "weight_scale", 1.0))
        if scale != 1.0:
            rows = (rows.to(torch.float32) * scale).to(torch.bfloat16)
    elif isinstance(table, FP8Table):
        rows = _fp8(table, _to_device(_u8_rows(table, flat, table.values), device), device)
    elif isinstance(table, NVFP4Table):
        rows = _nvfp4(table, _to_device(_u8_rows(table, flat, table.values), device),
                      _to_device(_u8_rows(table, flat, table.scales), device), device)
    elif isinstance(table, BF16Table):
        rows = _to_device(table.gather(flat).view(np.int16), device).view(torch.bfloat16)
        scale = float(getattr(table, "weight_scale", 1.0))
        if scale != 1.0:
            rows = (rows.to(torch.float32) * scale).to(torch.bfloat16)
    else:
        raise TypeError(f"{type(table).__name__}: no XPU gather for this n-gram table")
    return rows.reshape(-1, ng.heads, ng.dims)


def embed(ple: PLEW, ids, device) -> torch.Tensor:
    """The gathered rows flat [L, heads * dims] bf16 - ``cuda/forward.py``'s ``ple_emb`` feeding key/value_proj."""

    return gather(ple, ids, device).reshape(len(ids), ple.ngram.heads * ple.ngram.dims)


def _u8_rows(table, flat: np.ndarray, maps: list) -> np.ndarray:
    """``BF16Table``'s shard split over uint8 mmaps (fp8 codes, NVFP4 codes and block scales): raw bytes only."""

    shard, local = table._where(flat)
    out = np.empty((shard.size, maps[0].shape[1]), dtype=np.uint8)
    _copy_rows(table._pool, shard, lambda f, at: out.__setitem__(at, maps[f][local[at]]))
    return out


def _to_device(src: np.ndarray, device: torch.device) -> torch.Tensor:
    """``src``'s rows on ``device``: a pinned host staging copy, then one async H2D on the current stream."""

    if device.type not in ("xpu", "cuda"):
        return torch.from_numpy(src).to(device)
    host = torch.empty(src.shape, dtype=_DTYPES[src.dtype], pin_memory=True)
    host.numpy()[:] = src
    out = torch.empty(src.shape, dtype=_DTYPES[src.dtype], device=device)
    out.copy_(host, non_blocking=True)
    event = _event(device.type)()
    event.record(_current_stream(device))
    slots = _STAGING.setdefault(device, [])
    slots.append((host, event))
    while len(slots) > _KEEP:
        _, old = slots.pop(0)
        old.synchronize()
    return out


def _unpack4(words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, dims: int) -> torch.Tensor:
    """``cuda/glue._ple_embed``'s unpack in torch ops: nibbles x bf16 scale + bias in fp32, one bf16 rounding."""

    if words.shape[1] * 8 != dims or scales.shape[1] * 32 != dims:
        raise ValueError(f"the gathered 4-bit rows hold {words.shape[1] * 8} values, expected {dims}")
    shifts = torch.arange(0, 32, 4, device=words.device, dtype=torch.int32)
    q = ((words.unsqueeze(-1) >> shifts) & 0xF).to(torch.float32).reshape(words.shape[0], scales.shape[1], 32)
    v = q * scales.to(torch.float32).unsqueeze(-1) + biases.to(torch.float32).unsqueeze(-1)
    return v.to(torch.bfloat16).reshape(words.shape[0], dims)


def _fp8(table: FP8Table, codes: torch.Tensor, device: torch.device) -> torch.Tensor:
    """The fp8 shards' dequant on the card: code x table scale in fp32, one bf16 rounding (or the LUT itself)."""

    cache = _FP8.setdefault(table, {})
    if device not in cache:
        lut = torch.from_numpy(table.lut.view(np.int16)).to(device).view(torch.bfloat16)
        scale = lut[0x38].to(torch.float32)               # e4m3's 1.0 times the table scale: the scale's bf16 bits
        probe = torch.arange(256, device=device, dtype=torch.uint8)
        got = (probe.view(torch.float8_e4m3fn).to(torch.float32) * scale).to(torch.bfloat16)
        nan = (probe & 0x7F) == 0x7F
        got[nan] = lut[nan]                               # the LUT pins e4m3's two NaN codes to one canonical value
        cache[device] = ("scale", scale, lut) if torch.equal(got.view(torch.int16), lut.view(torch.int16)) \
            else ("lut", None, lut)
    kind, scale, lut = cache[device]
    if kind == "scale":
        rows = (codes.view(torch.float8_e4m3fn).to(torch.float32) * scale).to(torch.bfloat16)
        rows[(codes & 0x7F) == 0x7F] = lut[0x7F]
        return rows
    return lut[codes.to(torch.int64)]


def _nvfp4(table: NVFP4Table, codes: torch.Tensor, blocks: torch.Tensor, device: torch.device) -> torch.Tensor:
    """``NVFP4Table.gather``'s dequant on the card: e2m1 nibble x e4m3 block scale x table scale, one rounding."""

    n, half = codes.shape
    nib = torch.stack([codes & 0xF, codes >> 4], -1).reshape(n, half * 2)
    e2m1 = torch.from_numpy(table.e2m1).to(device=device, dtype=torch.float32)
    e4m3 = torch.from_numpy(table.e4m3).to(device=device, dtype=torch.float32)
    v = e2m1[nib.to(torch.int64)] * e4m3[blocks.to(torch.int64)].repeat_interleave(16, -1) * float(table.g)
    return v.to(torch.bfloat16)
