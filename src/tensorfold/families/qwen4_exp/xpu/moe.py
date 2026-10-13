"""Softmax top-k MoE on Intel GPUs, the ``cuda/moe.py`` semantics on device-independent torch ops.

The CUDA-side Triton kernels compile and run as-is on the B70 (triton 3.8, Intel driver — verified),
but their ``tl.exp`` and ``tl.dot`` bits are the device's own, and the grouped-expert launches behind
``select`` are a CUDA extension. Routing therefore runs as IEEE-rounded elementwise ATen kernels —
the same sequence of ops on either side — so the card's picks and weights are ``torch.equal`` to a
CPU run, a row's bits never depend on the row count, and ``xpu/nvfp4.exp`` is the exp.

Semantics kept from the CUDA kernels: logits are fp32 dot products of the bf16 rows; each row's
``top_k`` experts come off largest-logit-first with ties to the lower id; the weights are
``exp(l_k - l_0) / sum`` in fp32 rounded to bf16; the shared expert (id ``experts``) takes slot
``top_k`` with weight ``bf16(sigmoid(bf16(shared gate logit)))``; ``combine`` scales the fp32 slots
by their weights, sums them in slot order and rounds to bf16 once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from . import nvfp4

NEG = float("-inf")


class MoEBuffers:
    """Static scratch for up to ``rows`` rows (graph safe: every shape fixed at construction)."""

    def __init__(self, rows: int, cfg, device: torch.device | str, *, prefill: bool = False) -> None:
        slots = cfg.num_experts_per_tok + 1
        self.rows, self.slots = rows, slots
        self.logits = torch.empty((rows, cfg.num_experts + 1), dtype=torch.float32, device=device)
        self.pick = torch.empty((rows, slots), dtype=torch.int32, device=device)
        self.wts = torch.empty((rows, slots), dtype=torch.float32, device=device)
        self.y = torch.empty((rows, slots, cfg.hidden_size), dtype=torch.bfloat16 if prefill else torch.float32,
                             device=device)


def router(x: torch.Tensor, rows: torch.Tensor, out: torch.Tensor | None = None, *,
           fused: bool | None = None) -> torch.Tensor:
    """x [R, D] bf16, rows [E + 1, D] bf16 (router rows, then the shared expert's gate row) -> [R, E + 1] fp32.

    ``fused=False`` keeps the rank-one oracle: the fp32 dot products accumulate k in order
    (``nvfp4.mm_ordered``), so a row's logits are its own bits at any row count — the CUDA kernel's
    ``tl.dot`` gives the same values to fp32 rounding. ``fused=True`` runs the one card matmul
    (``nvfp4.mm_fused``), whose fp32 reduction order is the card's: the logits sit within fp32
    reduction noise of the oracle, which is the ``tests/cuda`` NVFP4 tolerance contract, not bit
    equality. ``fused=None`` (the default) consults ``TENSORFOLD_XPU_MOE_ORDERED``.

    The router is the decode step's dominant term — D is the full hidden width on every row of every
    layer — so the oracle's K launches per call are what the fused lane removes."""

    if fused is None:
        fused = os.environ.get(nvfp4._ORDERED_ENV) != "1"
    w_t = rows.to(x.dtype).t().contiguous()
    # ``mm_fused`` takes an fp32 right operand (the dequantized expert tables are already fp32); the
    # router's rows arrive bf16, so widen them here. bf16 -> fp32 is lossless, and ``mm_ordered``
    # widens both sides itself, so the two lanes see identical products and differ only in reduction
    # order — which is the tolerance contract.
    got = nvfp4.mm_fused(x, w_t.to(torch.float32)) if fused else nvfp4.mm_ordered(x, w_t)
    return got if out is None else out.copy_(got)


def select_rows(logits: torch.Tensor, buf: MoEBuffers, top_k: int, experts: int) -> None:
    """Each row's experts and weights, rows independent (buf.pick, buf.wts), ties to the lower id."""

    device, rows = logits.device, logits.shape[0]
    ar = torch.arange(experts, dtype=torch.int32, device=device)
    v = logits[:, :experts]
    top = v.max(dim=1, keepdim=True).values
    picks = torch.zeros((rows, buf.slots), dtype=torch.int32, device=device)
    exs = torch.zeros((rows, buf.slots), dtype=torch.float32, device=device)
    total = torch.zeros((rows,), dtype=torch.float32, device=device)
    for k in range(top_k):
        m = v.max(dim=1, keepdim=True).values
        idx = torch.where(v == m, ar.expand_as(v), experts).min(dim=1).values
        ex = nvfp4.exp(m.squeeze(1) - top.squeeze(1))
        picks[:, k] = idx.to(torch.int32)
        exs[:, k] = ex
        total = total + ex                                        # slot order, like the kernel's loop
        v = torch.where(ar.expand_as(v) == idx[:, None], NEG, v)
    renorm = torch.div(exs[:, :top_k], total[:, None]).to(torch.bfloat16).to(torch.float32)
    w = torch.zeros((rows, buf.slots), dtype=torch.float32, device=device)
    w[:, :top_k] = renorm
    sg = logits[:, experts].to(torch.bfloat16).to(torch.float32)
    picks[:, top_k] = experts
    w[:, top_k] = nvfp4.sigmoid(sg).to(torch.bfloat16).to(torch.float32)
    buf.pick[:rows].copy_(picks)
    buf.wts[:rows].copy_(w)


def select(logits: torch.Tensor, buf: MoEBuffers, top_k: int, experts: int) -> None:
    """Each row's experts and weights; the grouping is the ``grouped_moe`` sort, not a CUDA-ext plan."""

    select_rows(logits, buf, top_k, experts)


def combine(y: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
    """y [R, S, D] (fp32, or bf16 in prefill) and wts [R, S] fp32 -> [R, D] bf16, slots summed in order."""

    rows, slots, dims = y.shape
    acc = torch.zeros((rows, dims), dtype=torch.float32, device=y.device)
    for s in range(slots):
        acc = acc + wts[:, s:s + 1] * y[:, s, :].to(torch.float32)
    return acc.to(torch.bfloat16)


def moe(x: torch.Tensor, router_rows: torch.Tensor, ex, buf: MoEBuffers, top_k: int, experts: int, *,
        fused: bool | None = None) -> MoEBuffers:
    """Route x [R, D] bf16 and run its experts into buf.y [R, k + 1, D]; slot k is the shared expert.
    ``fused=False`` runs the rank-one oracle faces (phase 2's regression reference); ``None`` consults
    ``TENSORFOLD_XPU_MOE_ORDERED``."""

    from . import nvfp4 as xpu_nvfp4

    rows = x.shape[0]
    router(x, router_rows, buf.logits[:rows])
    select_rows(buf.logits[:rows], buf, top_k, experts)
    buf.y[:rows] = xpu_nvfp4.grouped_moe(x, ex, buf.pick[:rows], prefill=buf.y.dtype is torch.bfloat16,
                                         fused=fused)
    return buf


@dataclass
class Routed:
    """A layer's router rows [E + 1, D] bf16 (the shared expert's gate row last) and its experts table."""

    router: torch.Tensor
    experts: object                                  # Stacked4
    top_k: int

    @property
    def count(self) -> int:
        return self.router.shape[0] - 1


class _Shape:
    def __init__(self, m: Routed) -> None:
        self.num_experts_per_tok, self.num_experts = m.top_k, m.count
        self.moe_intermediate_size, self.hidden_size = m.experts.width, m.experts.dims


_scratch: dict[tuple, MoEBuffers] = {}


def run(x: torch.Tensor, m: Routed, *, prefill: bool = False, fused: bool | None = None) -> torch.Tensor:
    """x [R, D] bf16 -> [R, D] bf16: its top-k experts by renormalized softmax weight plus the shared expert."""

    rows = x.shape[0]
    size = 1 << max(4, (rows - 1).bit_length())                 # scratch per power of two, reused by every layer
    key = (size, m.count, m.top_k, m.experts.width, m.experts.dims, prefill, x.device)
    buf = _scratch.get(key)
    if buf is None:
        buf = _scratch[key] = MoEBuffers(size, _Shape(m), x.device, prefill=prefill)
    moe(x.contiguous(), m.router, m.experts, buf, m.top_k, m.count, fused=fused)
    return combine(buf.y[:rows], buf.wts[:rows])
