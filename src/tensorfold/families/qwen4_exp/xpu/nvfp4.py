"""NVFP4 on Intel GPUs: the device dequant bit-matching the CUDA reference, and the grouped expert matmul.

Correctness-first, so every output is the CPU reference's bits rather than an approximation of them:

* the dequant is ``cuda/nvfp4.py``'s own decode (``e2m1_bits``/``e4m3_bits`` are pure torch), run with the
  checkpoint's arrays on card, so the card's fp32 weight is ``torch.equal`` to the CPU reference's;
* matmuls accumulate as ``acc += x[:, k] * w[k]`` for ``k = 0..K-1`` — every op an IEEE-rounded elementwise
  ATen kernel — so a row's bits are its own (no reduction whose order the launch or batch shape picks,
  unlike ``torch.sum``/``bmm``) and the same code on CPU tensors is the reference;
* the grouped expert matmul's hot faces run ``mm_fused`` — one fp32 card matmul per face instead of ``K``
  rank-one launches — judged against the rank-one oracle within the ``tests/cuda`` NVFP4 tolerances
  (phase 2), with ``mm_ordered`` kept in-tree as that oracle and as ``grouped_moe(fused=False)``.
* ``exp`` is an fp32 Horner polynomial on the exp2 range, built from the same ATen kernels on either side
  (``torch.exp`` differs in ULPs between the CPU and the card, and softmax weights gate bit-equal).

The MoE entry mirrors ``cuda/nvfp4_moe.py``'s ``moe(x, xs, router_rows, ex, buf, cfg)`` so the forward
port can call it with ``MoEW.experts`` as loaded (slice 1): the FP4 routed stacks dequantize on card and
the bf16 shared expert rides the same matmul lane.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass

import torch

os.environ.setdefault("TRITON_DEFAULT_BACKEND", "intel")   # the venv may carry both wheels; the card is intel

from ..cuda import nvfp4 as _ref                           # noqa: E402  (device-free reference decode)

GS = _ref.GS

# the scaled-exponential split: exp(x) = 2**r * e**f with r = round(x * log2 e), f = x - r * ln2 (Cody-Waite),
# e**f on |f| <= ln2/2 by fp32 Horner on the 1/n! series (truncation ~1 fp32 ULP there).
_LOG2E = 1.4426950408889634
_LN2_HI = 0.6931457519531250                                # fp32(x): the high half of ln2, for Cody-Waite
_LN2_LO = 1.4286066322321901e-06                             # ln2 - _LN2_HI as an fp32-representable residual
_COEF = (1.0, 1.0, 0.5, 0.16666667, 0.041666668, 0.008333333, 0.0013888889)


# -- the dequant ----------------------------------------------------------------------------------------------

def exp(x: torch.Tensor) -> torch.Tensor:
    """fp32 exp by ``k``-scaled exp2: every step an IEEE-rounded elementwise op, bit-identical cpu and card.
    Overflow returns +inf (so ``1 / (1 + exp(-x))`` reaches 0 on both sides), underflow 0."""

    x = x.to(torch.float32)
    r = torch.round(x * _LOG2E)
    f = torch.sub(torch.sub(x, r * _LN2_HI), r * _LN2_LO)
    p = torch.full_like(f, _COEF[6])
    for c in _COEF[5::-1]:
        p = torch.add(torch.mul(p, f), c)
    n = r.to(torch.int32) + 127
    scale = (n.clamp(1, 254) << 23).view(torch.float32)     # 2**(n - 127)
    y = torch.mul(torch.where(n <= 0, torch.zeros_like(p), p), scale)
    y = torch.where(n >= 255, torch.full_like(p, float("inf")), y)
    y = torch.where(x == float("-inf"), torch.zeros_like(p), y)
    return torch.where(torch.isnan(x) | (x == float("inf")), torch.where(torch.isnan(x), torch.full_like(p, float("nan")), torch.full_like(p, float("inf"))), y)

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """1 / (1 + exp(-x)) with an IEEE-rounded divide, the CUDA kernels' ``div_rn`` bits."""

    return torch.div(1.0, 1.0 + exp(-x.to(torch.float32)))


def dequantize(words: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2) -> torch.Tensor:
    """Packed codes and e4m3 block scales -> the exact fp32 weight, on the tensors' device.

    A (N, K/2) matrix goes through ``cuda/nvfp4.dequantize`` itself; a stacked (E, N, K/2) table decodes
    each expert with its own ``scale_2`` row, the reference's arithmetic with the block interleave an
    ``dim=-1`` away."""

    if words.dim() == 2:
        return _ref.dequantize(words, weight_scale, weight_scale_2)
    e = words.shape[0]
    s2 = weight_scale_2 if isinstance(weight_scale_2, torch.Tensor) else torch.tensor(float(weight_scale_2))
    if s2.numel() == 1:
        factors = s2.reshape(1, 1, 1).to(torch.float32)
    else:
        factors = s2.reshape(e, 1, 1).to(torch.float32)
    scales = _ref._bits_to_f32(_ref.e4m3_bits(weight_scale)) * factors
    return _ref._bits_to_f32(_ref.e2m1_bits(words)) * scales.repeat_interleave(GS, dim=-1)


# -- the row-invariant matmul -----------------------------------------------------------------------------------

def mm_ordered(x: torch.Tensor, w_t: torch.Tensor, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """x [M, K] (bf16 or fp32) @ ``w_t`` [K, N] -> fp32 [M, N]: rank-one updates added in ``k`` order.

    Each step multiplies two exact fp32 values and adds once, so the result never depends on the row
    count or the device's matmul schedule — and the same call on CPU tensors is the gate's reference."""

    m, k = x.shape
    n = w_t.shape[1]
    xf = x.to(torch.float32)
    acc = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    for i in range(k):
        acc = torch.add(acc, xf[:, i:i + 1] * w_t[i:i + 1, :])
    return acc if out_dtype is None else acc.to(out_dtype)


@contextmanager
def _ieee_matmul():
    """True fp32 matmuls for the block: the down face's CUDA gate is a 1e-4 relative error, which a
    tf32/bf16 input path (10-bit/8-bit mantissa) cannot hold; restore the lane's own settings after."""

    mm = torch.backends.cuda.matmul
    tf32, prec = mm.allow_tf32, getattr(mm, "fp32_precision", None)
    mm.allow_tf32, mm.fp32_precision = False, "ieee"
    try:
        yield
    finally:
        mm.allow_tf32, mm.fp32_precision = tf32, prec


def mm_fused(x: torch.Tensor, w_t: torch.Tensor, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """x [M, K] (bf16 or fp32) @ ``w_t`` [K, N] fp32 -> fp32 [M, N]: one card matmul, fp32 accumulate.

    The phase-2 lane of the expert faces: ``K`` rank-one launches collapse into one, and the fp32
    reduction's order is the card's, so the bits are within fp32 reduction noise of ``mm_ordered`` —
    the CUDA NVFP4 gates (``tests/cuda/test_nvfp4_experts.py``: relative max 2e-2 gate/up, 1e-4 down),
    not the oracle's bits. Inputs widen to fp32 exactly (bf16 -> fp32 is lossless, the tables are
    already fp32), so with tf32 held off every product is an exact fp32 multiply and only the
    accumulation order differs from the oracle's."""

    with _ieee_matmul():
        acc = torch.matmul(x.to(torch.float32), w_t)
    return acc if out_dtype is None else acc.to(out_dtype)


def swiglu(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """The experts' activation with ``nvfp4_moe.MoE4.shared_act``'s bits: bf16(bf16(silu(g)) * u) on bf16 g and u."""

    gate, up = g.to(torch.bfloat16).to(torch.float32), u.to(torch.bfloat16).to(torch.float32)
    return ((gate / (1.0 + exp(-gate))).to(torch.bfloat16).to(torch.float32) * up).to(torch.bfloat16)


# -- one layer's experts on card -----------------------------------------------------------------------------------

def _stack_t(triple: torch.Tensor | tuple, device, name: str) -> torch.Tensor:
    """A stacked expert table -> its fp32 transpose [E, in, out]: an FP4 (words, scales, scale_2) triple
    dequantizes on card, bf16 rows (the MTP layer) just widen."""

    w = triple[0] if isinstance(triple, tuple) else triple
    moved = [t.to(device) for t in (triple if isinstance(triple, tuple) else (triple,))]
    if isinstance(triple, tuple):
        rows = dequantize(moved[0], moved[1], moved[2])
    else:
        rows = moved[0].to(torch.float32)
    if rows.dim() != 3:
        raise ValueError(f"{name}: expected stacked experts, got {tuple(rows.shape)}")
    return rows.transpose(1, 2).contiguous()


@dataclass
class Stacked4:
    """One layer's experts as fp32 transposes on card: gate/up ``[E, D, NI]``, down ``[E, NI, D]`` (so a
    pair's matmul is ``mm_fused(pairs, table[e])``, or ``mm_ordered`` for the oracle), plus the shared
    expert's three faces."""

    gate_t: torch.Tensor
    up_t: torch.Tensor
    down_t: torch.Tensor
    shared_gate_t: torch.Tensor       # [D, NS]
    shared_up_t: torch.Tensor         # [D, NS]
    shared_down_t: torch.Tensor       # [NS, D]

    @property
    def routed(self) -> int:
        return int(self.gate_t.shape[0])

    @property
    def dims(self) -> int:
        return int(self.gate_t.shape[1])                       # D: x's and down's outputs' width

    @property
    def width(self) -> int:
        return int(self.gate_t.shape[2])                       # NI: a gate row's inputs, down's outputs

    @property
    def shared_width(self) -> int:
        return int(self.shared_gate_t.shape[1])

    @property
    def device(self):
        return self.gate_t.device


def stacked4_from_experts(experts: dict, device) -> Stacked4:
    """The raw ``MoEW.experts`` dict (slice 1) -> the layer's fp32 tables on ``device``."""

    shared = experts["shared"]
    if any(getattr(t, "dtype", None) not in (torch.bfloat16, torch.float32) for t in shared):
        raise NotImplementedError(f"xpu experts: the shared expert's FP8 faces have no lane matmul here yet "
                                  f"({[getattr(t, 'dtype', None) for t in shared]})")
    return Stacked4(_stack_t(experts["gate"], device, "gate"),
                    _stack_t(experts["up"], device, "up"),
                    _stack_t(experts["down"], device, "down"),
                    shared[0].to(device).to(torch.float32).t().contiguous(),
                    shared[1].to(device).to(torch.float32).t().contiguous(),
                    shared[2].to(device).to(torch.float32).t().contiguous())


# -- the grouped expert matmul --------------------------------------------------------------------------------

def grouped_moe(x: torch.Tensor, ex: Stacked4, pick: torch.Tensor, *, prefill: bool = False,
                fused: bool = True) -> torch.Tensor:
    """Run the router's pairs: x [R, D] bf16, pick [R, S] int32 (shared expert id ``routed`` last slot) ->
    y [R, S, D] (fp32, or bf16 for a prompt as the CUDA prefill keeps its slots).

    Pairs sort stably by expert — each segment then runs its expert's table on the whole segment (one
    ``mm_fused`` per face by default) and the activation's ``swiglu`` bits, so pairs of different experts
    never share an accumulator and a pair's output is its row's and its expert's alone. ``fused=False``
    reruns the faces with ``mm_ordered`` — the rank-one oracle, whose bits the fused path must match
    within the ``tests/cuda`` NVFP4 tolerances."""

    face = mm_fused if fused else mm_ordered

    rows, slots = pick.shape
    dims = ex.dims
    out = torch.empty((rows * slots, dims), dtype=torch.bfloat16 if prefill else torch.float32, device=x.device)
    flat = pick.reshape(-1).to(torch.int64)
    order = torch.argsort(flat * flat.numel() + torch.arange(flat.numel(), device=flat.device))
    counts = torch.bincount(flat, minlength=ex.routed + 1).tolist()
    x_rows = x.index_select(0, (order // slots))
    at = 0
    for e in range(ex.routed + 1):
        take = counts[e]
        if not take:
            continue
        seg = x_rows[at:at + take]
        if e < ex.routed:
            act = swiglu(face(seg, ex.gate_t[e]), face(seg, ex.up_t[e]))
            y = face(act, ex.down_t[e])
        else:
            act = swiglu(face(seg, ex.shared_gate_t), face(seg, ex.shared_up_t))
            y = face(act, ex.shared_down_t)
        out.index_copy_(0, order[at:at + take], y if out.dtype == torch.float32 else y.to(torch.bfloat16))
        at += take
    return out.view(rows, slots, dims)


def moe(x: torch.Tensor, xs: torch.Tensor, router_rows: torch.Tensor, ex: Stacked4, buf, cfg, *,
        fused: bool = True) -> None:
    """Route ``x`` [R, D] bf16 and run its experts into ``buf`` — the ``cuda/nvfp4_moe.moe`` contract:
    ``buf.y`` [R, k + 1, D] and ``buf.wts``; ``xs`` is unused (the FP4 experts have no biases).
    ``fused=False`` selects the rank-one oracle faces (phase 2's regression reference)."""

    from . import moe as xpu_moe

    rows, top_k = x.shape[0], int(cfg.num_experts_per_tok)
    xpu_moe.router(x, router_rows, buf.logits[:rows])
    xpu_moe.select_rows(buf.logits[:rows], buf, top_k, ex.routed)
    buf.y[:rows] = grouped_moe(x, ex, buf.pick[:rows], prefill=buf.y.dtype is torch.bfloat16, fused=fused)
    return buf
