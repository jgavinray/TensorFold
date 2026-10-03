"""Quantized-KV tree attention for XPU: committed keys and values read as int8/int4 codes plus fp16
scales and dequantized in registers, then the same tile math as ``kernels/attention.py``.

The caches stay exactly as ``kvcache.KVCache`` stores them: codes H32-rotated in g32 groups, ExLlamaV3's
midpoint grid.  The kernel undoes the rotation in registers -- H32 is its own inverse (the butterfly
squares to 32 I, and ``h32`` scales by ``1/sqrt(32)``, so ``h32(h32(x)) == x``).  A row's window keys are
the bf16 node tensors, as in the bf16 kernel.  ``kernels/attention.py`` is not modified: this module
reuses its ``Plan`` and mirrors its output contract (attend (W, H, D) queries to committed keys and own
paths -> (W, H, D) bf16), with a serial-equivalent kernel (one program a row and key head; the chunked
merge scheme can be mirrored later if decode throughput needs it).  Prefill reads through a dequantized
materialization instead (a cold path), reusing ``prefill_attention`` unchanged.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import torch
import triton
import triton.language as tl

from tensorfold.cuda.kernels.attention import CHUNK, MAX_NODES, QUERY_TILE, Plan

from . import kvcache

KEY_TILE = tl.constexpr(16)   # keys a dequant tile holds (dot's minimum; keeps dequant registers small)
_MAX_NODES = tl.constexpr(MAX_NODES)   # triton-xpu 3.8: jit bodies need tl.constexpr globals


@lru_cache(maxsize=None)
def _base_u8(index: int) -> torch.Tensor:
    """The byte origin code offsets are measured from."""
    return torch.zeros(64, dtype=torch.uint8, device=torch.device("xpu", index))


@lru_cache(maxsize=None)
def _base_f16(index: int) -> torch.Tensor:
    """The fp16-element origin scale offsets are measured from."""
    return torch.zeros(64, dtype=torch.float16, device=torch.device("xpu", index))


def base(device) -> torch.Tensor:
    """The code origin for ``device`` (an XPU tensor; quantized caches are an XPU feature)."""
    device = torch.device(device)
    return _base_u8(device.index if device.index is not None else torch.xpu.current_device())


def offsets(caches: Sequence[kvcache.KVCache], device) -> list[int]:
    """Per cache [k byte offset, v byte offset, k scale offset, v scale offset], scales in fp16 elements."""

    origin = base(device).data_ptr()
    scales_origin = _base_f16(device.index if device.index is not None else torch.xpu.current_device()).data_ptr()
    out = []
    for cache in caches:
        for codes, scales in ((cache.k, cache.ks), (cache.v, cache.vs)):
            if codes.dtype not in (torch.uint8, torch.int8) or not codes.is_contiguous() \
                    or scales.dtype != kvcache.SCALE_DTYPE or not scales.is_contiguous():
                raise ValueError("quantized caches: contiguous uint8/int8 codes and fp16 scales")
            delta = codes.data_ptr() - origin
            delta = ((delta + 2**63) % 2**64) - 2**63   # xpu addresses use the full 64-bit space; wrap to signed
            if delta % 16:
                raise ValueError("caches must be 16-byte aligned")
            sdelta = scales.data_ptr() - scales_origin
            sdelta = ((sdelta + 2**63) % 2**64) - 2**63
            if sdelta % 16:
                raise ValueError("scale tensors must be 16-byte aligned")
            out += [delta, sdelta // 2]
    return out


@triton.jit
def _tile(q, k, v, m, l, o, valid, scale: tl.constexpr):
    """``kernels/attention.py``'s online-softmax tile, verbatim (one code path with the bf16 kernel)."""
    scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
    scores = tl.where(valid[None, :], scores, float("-inf"))
    tile_m = tl.max(scores, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    l = l * alpha + tl.sum(p, 1)
    return next_m, l, o


@triton.jit
def _h32(x):
    """H32 over the last axis (32) in registers, ``kvcache.h32_ref``'s operations -- its own inverse."""

    for lo in (1, 2, 4, 8, 16):
        hi = 32 // (2 * lo)
        t = tl.reshape(x, (x.shape[0], x.shape[1], hi, 2, lo))
        a, b = tl.split(tl.permute(t, (0, 1, 2, 4, 3)))
        x = tl.reshape(tl.permute(tl.join(a + b, a - b), (0, 1, 2, 4, 3)), (x.shape[0], x.shape[1], 32))
    return x * 0.17677669529663687            # R32


@triton.jit
def _dequant_tile(KC, KS, koff, ksoff, rows, hk, d, committed, D: tl.constexpr, BITS: tl.constexpr):
    """One (KEY_TILE, D) key tile from the caches: dequantized to bf16 and unrotated, garbage rows zeroed."""

    if BITS == 8:
        c = tl.load(KC + koff + (rows[:, None] * HK + hk) * D + d[None, :], mask=committed[:, None], other=0)
        c = c.to(tl.int32)
        c = tl.where(c >= 128, c - 256, c)    # the int8 codes are signed; bytes load unsigned
        s = tl.load(KS + ksoff + (rows[:, None] * HK + hk) * (D // 32) + (d // 32)[None, :],
                    mask=committed[:, None], other=0).to(tl.float32)
        x = ((c.to(tl.float32) + 0.5) * s * 0.0078125).to(tl.bfloat16)
    else:
        packed = tl.load(KC + koff + (rows[:, None] * HK + hk) * (D // 2) + (d // 2)[None, :],
                         mask=committed[:, None], other=0).to(tl.int32)
        lo, hi = packed & 15, (packed >> 4) & 15
        c = tl.where((d % 2)[None, :] == 0, lo, hi)          # low nibble = even column
        s = tl.load(KS + ksoff + (rows[:, None] * HK + hk) * (D // 32) + (d // 32)[None, :],
                    mask=committed[:, None], other=0).to(tl.float32)
        x = ((c.to(tl.float32) - 7.5) * s * 0.125).to(tl.bfloat16)
    g = tl.reshape(x.to(tl.float32), (KEY_TILE, D // 32, 32))
    g = _h32(g)                               # the codes stay rotated; the dot needs the stored value
    return tl.reshape(g, (KEY_TILE, D)).to(tl.bfloat16)


@triton.jit
def _attend(Q, KC, VC, KS, VS, KN, VN, OFF, ROWS, STREAMS, PATHS, DEPTHS, OUT,
            H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, BITS: tl.constexpr,
            SCALE: tl.constexpr):
    """One (row, key head) program: the row's committed keys from the quantized caches, its path from the
    bf16 node tensors; the same online softmax as the bf16 kernel."""

    node = tl.program_id(0)
    hk = tl.program_id(1)
    s = tl.load(ROWS + node)
    p = tl.load(STREAMS + s * 4 + 2)
    depth = tl.load(DEPTHS + node)
    n = p + depth
    koff = tl.load(OFF + s * 4)
    voff = tl.load(OFF + s * 4 + 1)
    ksoff = tl.load(OFF + s * 4 + 2)
    vsoff = tl.load(OFF + s * 4 + 3)
    gg = tl.arange(0, 16)
    d = tl.arange(0, D)
    q = tl.load(Q + (node * H + hk * G + gg[:, None]) * D + d[None, :], mask=gg[:, None] < G,
                other=0).to(tl.bfloat16)
    m = tl.full((16,), float("-inf"), tl.float32)
    l = tl.zeros((16,), tl.float32)
    o = tl.zeros((16, D), tl.float32)
    for at in range(0, tl.cdiv(n, KEY_TILE)):
        ki = at * KEY_TILE + tl.arange(0, KEY_TILE)
        valid = ki < n
        committed = ki < p
        path_slot = ki - p
        on_path = (path_slot >= 0) & (path_slot < depth)
        path_node = tl.load(PATHS + node * _MAX_NODES + path_slot, mask=on_path, other=0)
        kk = _dequant_tile(KC, KS, koff, ksoff, ki, hk, d, committed, D, BITS)
        vv = _dequant_tile(VC, VS, voff, vsoff, ki, hk, d, committed, D, BITS)
        kn = tl.load(KN + (path_node[:, None] * HK + hk) * D + d[None, :], mask=on_path[:, None], other=0)
        vn = tl.load(VN + (path_node[:, None] * HK + hk) * D + d[None, :], mask=on_path[:, None], other=0)
        kk = tl.where(committed[:, None], kk, kn)
        vv = tl.where(committed[:, None], vv, vn)
        m, l, o = _tile(q, kk, vv, m, l, o, valid, SCALE)
    tl.store(OUT + (node * H + hk * G + gg[:, None]) * D + d[None, :], (o / l[:, None]).to(tl.bfloat16),
             mask=gg[:, None] < G)


def _device_index(device) -> int:
    device = torch.device(device)
    return device.index if device.index is not None else torch.xpu.current_device()


def attention(q: torch.Tensor, k_nodes: torch.Tensor, v_nodes: torch.Tensor, offs: torch.Tensor, p: Plan, *,
              scale: float) -> torch.Tensor:
    """Attend (W, H, D) queries to committed keys (quantized caches via ``offs``) and own paths; (W, H, D) bf16.
    ``offs`` is (streams, 5) int64: code byte offsets, scale fp16-element offsets, and the codes' bits."""

    w, h, d = q.shape
    hk = k_nodes.shape[1]
    if not (w == p.width and d in (128, 256) and k_nodes.shape == (w, hk, d) and v_nodes.shape == k_nodes.shape
            and h % hk == 0 and h // hk <= QUERY_TILE):
        raise ValueError("unsupported attention shape")
    if any(x.dtype != torch.bfloat16 or x.device.type != "xpu" or not x.is_contiguous()
           for x in (q, k_nodes, v_nodes)):
        raise ValueError("q and node keys and values must be contiguous XPU bf16 tensors")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    if offs.shape != (p.streams.shape[0], 5) or int(offs[0, 4]) not in (4, 8) \
            or any(int(offs[s, 4]) != int(offs[0, 4]) for s in range(offs.shape[0])):
        raise ValueError("offs: (streams, 5) int64 with one bits value")
    bits = int(offs[0, 4])
    out = torch.empty_like(q)
    _attend[(w, hk)](q, base(q.device), base(q.device), _base_f16(_device_index(q.device)),
                     _base_f16(_device_index(q.device)), k_nodes, v_nodes, offs, p.rows, p.streams,
                     p.paths, p.depths, out, H=h, HK=hk, D=d, G=h // hk, BITS=bits, SCALE=scale,
                     num_warps=4)
    return out


def dequant_rows(cache: kvcache.KVCache, stop: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The cache's first ``stop`` rows as bf16 keys and values (the serial reference's caches): dequantize,
    then undo the H32 rotation -- the kernel's in-register pipeline in torch form."""

    def one(codes, scales):
        d = kvcache.dequant_ref(codes[:stop], scales[:stop], cache.bits)
        return kvcache.h32_ref(d.float()).reshape(d.shape).to(torch.bfloat16)

    return one(cache.k, cache.ks), one(cache.v, cache.vs)


def prefill_attention(q: torch.Tensor, cache: kvcache.KVCache, p0: int, *, scale: float) -> torch.Tensor:
    """Prefill read: dequantize [0, p0 + W) once and run the bf16 prefill kernel unchanged."""

    from tensorfold.cuda.kernels import prefill_attention as bf16_prefill

    k, v = dequant_rows(cache, p0 + q.shape[0])
    return bf16_prefill.attention(q, k, v, p0, scale=scale)
