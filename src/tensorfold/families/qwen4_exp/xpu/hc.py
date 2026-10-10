"""The hyper-connections' branch/merge on Intel GPUs: the CUDA-family glue kernels' arithmetic in torch ops.

Each op mirrors its ``cuda/glue.py`` triton kernel (``hc_writeback``, ``hc_normed``, ``hc_act``,
``hc_reduce_act``, ``hc_mix``) and the mix half of ``cuda/hc_upmix.py``: fp32 math with bf16 rounding
at every stored model tensor, fp32 sums added in the kernels' fixed orders, and one bf16 rounding on
each store. The matmuls the read-out sandwiches (the down/up Q4 projections) live in the matmul
slice; these ops take their outputs exactly as ``cuda/forward.py`` hands them to the glue kernels.

Two rules keep the card's bits equal to the host's, run through the same code:

* fp32 ``add``/``sub``/``mul`` and the bf16 casts are IEEE-correctly-rounded on both backends, so
  every arithmetic step here is one of those, or goes through ``exp``/``div``/``sqrt`` below;
* torch's fp32 ``exp``/``sqrt``/scalar-``div`` are NOT bit-equal across CPU and XPU (and ``.sum``
  picks its tree per launch shape, the peer finding), so reductions are elementwise fixed-order
  folds and the transcendentals are computed from fp64 tensor-by-tensor operations, which are
  (probed on the B70 #2: fp64 add/sub/mul/div and the f32<->f64 casts all agree bit-for-bit).
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------- portable primitives
# fp32-by-value results computed from IEEE-correctly-rounded fp64 tensor ops only: the same
# program on any backend returns the same bits (torch's native fp32 exp/sqrt/div do not).

_COEF = (1.0, 1.0, 0.5, 0.16666666666666666, 0.041666666666666664, 0.008333333333333333,
         0.001388888888888889, 0.00019841269841269841, 2.7557319223985893e-05,
         3.5433905869363475e-06, 4.189140014642468e-07, 4.6393073436483364e-08,
         4.7852238760752596e-09, 4.681021964016418e-10, 4.258703798322955e-11, 3.5998551424640045e-12)
_EXP_CLAMP = 84.0                      # exp(x) stays fp32-normal after the reciprocal, exp never overflows
_CONST: dict[torch.device, dict] = {}


def _const(device: torch.device, key: str, make) -> torch.Tensor:
    got = _CONST.setdefault(device, {})
    if key not in got:
        got[key] = make().to(device)
    return got[key]


def _fp64_const(device: torch.device, key: str, value: float) -> torch.Tensor:
    return _const(device, key, lambda: torch.tensor(value, dtype=torch.float64))


def _exp64(xd: torch.Tensor) -> torch.Tensor:
    """exp of an fp64 tensor: 2^n * poly(r) in fp64 (|r| <= ln2/2), the power of two from its bits.

    The argument clamps at +-84: sig/silu are fp32-saturated beyond it, and the clamp keeps every
    intermediate fp32-normal (no denormals or infinities to differ over)."""

    hi, lo = (_fp64_const(xd.device, k, v) for k, v in
             (("ln2hi", 0.693145751953125), ("ln2lo", 1.4286068203094172e-06)))
    u = torch.clamp(xd, -_EXP_CLAMP, _EXP_CLAMP)
    n = torch.round(u * _fp64_const(xd.device, "invln2", 1.0 / math.log(2.0)))
    r = u - n * hi
    r = r - n * lo
    acc = _fp64_const(xd.device, f"c{len(_COEF) - 1}", _COEF[-1]).expand_as(r).clone()
    for i in range(len(_COEF) - 2, -1, -1):
        acc = acc * r + _fp64_const(xd.device, f"c{i}", _COEF[i])
    return acc * ((n.to(torch.int64) + 1023) << 52).view(torch.float64)


def exp(x: torch.Tensor) -> torch.Tensor:
    """fp32 ``exp`` at fp64 precision, rounded back once."""

    return _exp64(x.double()).to(torch.float32)


def div(a: torch.Tensor, b: float | torch.Tensor) -> torch.Tensor:
    """fp32 quotient with correctly-rounded division: fp64 tensor-by-tensor, one fp32 rounding.

    A Python-scalar or 0-dim operand is expanded to the full shape first: torch's CPU kernels fold
    scalar division into a reciprocal multiply the XPU backend does not (probed on B70 #2)."""

    ad = a.double()
    if isinstance(b, (int, float)):
        bd = _fp64_const(a.device, f"k{float(b)!r}", float(b)).expand(ad.shape)
    else:
        bd = b.double()
    return (ad / bd).to(torch.float32)


def sqrt(t: torch.Tensor) -> torch.Tensor:
    """fp32 square root at fp64 precision: the exponent-halved seed through eight Newton rounds."""

    td = t.double()
    bits = td.view(torch.int64)
    se = (((bits >> 52) & 0x7FF) + 1023) >> 1
    s = ((se << 52) | (bits & ((1 << 52) - 1))).view(torch.float64)
    for _ in range(8):
        s = 0.5 * (s + td / s)
    return s.to(torch.float32)


def _one(device: torch.device) -> torch.Tensor:
    return _const(device, "one", lambda: torch.ones((), dtype=torch.float64))


def bsig(x: torch.Tensor) -> torch.Tensor:
    """glue's ``_bsig``: fp32 1 / (1 + exp(-x)), rounded once to bf16 and back."""

    e = _exp64(-x.double())
    return (torch.ones_like(e) / (_one(x.device) + e)).to(torch.float32).to(torch.bfloat16).to(torch.float32)


def bsilu(x: torch.Tensor) -> torch.Tensor:
    """glue's ``_bsilu``: fp32 x / (1 + exp(-x)) (the division, not a multiply by the sigmoid),
    rounded once to bf16 and back."""

    return div(x, 1.0 + exp(-x)).to(torch.bfloat16).to(torch.float32)


# ---------------------------------------------------------------- ordered reductions

def fold(x: torch.Tensor) -> torch.Tensor:
    """The last dimension summed by xor-paired halving, elementwise adds only: a fixed order that
    no launch or batch shape can rechoose (the peer finding's fixed-order channel reduction)."""

    while x.shape[-1] > 1:
        x = x.reshape(*x.shape[:-1], 2, x.shape[-1] // 2)
        x = x[..., 0, :] + x[..., 1, :]
    return x[..., 0]


def group_sums(x: torch.Tensor, groups: int, width: int) -> torch.Tensor:
    """``x`` [rows, groups * width] fp32 to the fixed-order sums of each ``width``-wide run."""

    return fold(x.reshape(*x.shape[:-1], groups, width))


def _seq_sum(parts: list[torch.Tensor]) -> torch.Tensor:
    """The fp32 slices added left to right, in order (the kernels' static_range accumulators)."""

    acc = parts[0]
    for p in parts[1:]:
        acc = acc + p
    return acc


def _bf(x: torch.Tensor) -> torch.Tensor:
    """One bf16 rounding, kept as fp32 (the triton kernels' ``.to(bf16).to(f32)``)."""

    return x.to(torch.bfloat16).to(torch.float32)


# ---------------------------------------------------------------- the glue ops, same signatures

def hc_writeback(h: torch.Tensor, hout: torch.Tensor, pss: torch.Tensor, streams: int, mode: int,
                 branch: torch.Tensor | None = None, inject: torch.Tensor | None = None,
                 y: torch.Tensor | None = None, wts: torch.Tensor | None = None,
                 block: int = 256) -> None:
    """``glue.hc_writeback``: the pending branch into the streams h (in place, hout is h), then each
    stream's 256-column sums of squares into pss. mode 0 none, 1 branch, 2 MoE slots, 3 rank
    partials, 4 K slices; the fp32 sums in fixed order, rounded once. ``branch`` is [R, D] bf16
    (mode 1) or the unreduced [W, R, D] fp32 (modes 3/4); ``y`` [R, slots, D] bf16 with ``wts``
    [R, slots] fp32 (mode 2); ``inject`` [R, streams] bf16."""

    if mode not in (0, 1, 2, 3, 4):
        raise ValueError(f"unknown hc_writeback mode {mode}")
    rows, wide = h.shape
    d = wide // streams
    if mode == 1:
        br = branch.float()
    elif mode in (3, 4):
        br = _bf(_seq_sum([branch[k] for k in range(branch.shape[0])]))     # fp32 slices, in order
    elif mode == 2:
        slots = wts.shape[1]
        yf = y.reshape(rows * slots, d)
        acc = torch.zeros((rows, d), dtype=torch.float32, device=h.device)
        for k in range(slots):
            acc = acc + yf[k::slots].float() * wts[:, k][:, None]
        br = _bf(acc)
    else:
        br = None
    hs = h.view(rows, streams, d)
    for s in range(streams):
        hv = hs[:, s].float()
        if mode != 0:
            inj = inject.float()
            hv = _bf(hv + br * inj[:, s:s + 1])
            hout.view(rows, streams, d)[:, s].copy_(hv.to(torch.bfloat16))
        pss[:, :, s].copy_(group_sums(hv * hv, d // block, block))


def hc_normed(h: torch.Tensor, pss: torch.Tensor, scale: torch.Tensor, normed: torch.Tensor,
              xs: torch.Tensor, streams: int, eps: float, block: int = 512) -> None:
    """``glue.hc_normed``: per-stream RMSNorm of h from its 256-column pss partials summed in order,
    times ``scale`` [S*D] fp32, stored bf16, and the stored rows' 32-group sums into xs."""

    rows, wide = h.shape
    d = wide // streams
    total = _seq_sum([pss[:, c, :] for c in range(pss.shape[1])])            # fp32, chunk order
    rinv = div(torch.ones_like(total), sqrt(div(total, d) + eps))            # 1 / sqrt(total / D + eps)
    hs = h.view(rows, streams, d)
    ns = normed.view(rows, streams, d)
    ws = scale.view(streams, d)
    for s in range(streams):
        y = ((hs[:, s].float() * rinv[:, s][:, None]) * ws[s][None, :]).to(torch.bfloat16)
        ns[:, s].copy_(y)
        xs.view(rows, streams, d // 32)[:, s].copy_(group_sums(y.float(), d // 32, 32))


def hc_act(dn: torch.Tensor, act: torch.Tensor, xs: torch.Tensor, inj: torch.Tensor | None,
           streams: int, low: int) -> None:
    """``glue.hc_act``: the down projection's row [low | inject] to act = bf16(silu(bf16(v / S))) and
    its 32-group sums, and the inject gates bf16(2 bf16(sigmoid(bf16(v / S))))."""

    rows = dn.shape[0]
    v = _bf(div(dn[:, :low], streams))
    a = bsilu(v)
    act[:, :low].copy_(a.to(torch.bfloat16))
    xs.view(rows, low // 32).copy_(group_sums(a, low // 32, 32))
    if inj is not None:
        iv = _bf(div(dn[:, low:low + streams], streams))
        inj.copy_((2.0 * bsig(iv)).to(torch.bfloat16))


def hc_reduce_act(part: torch.Tensor, act: torch.Tensor, xs: torch.Tensor, inj: torch.Tensor | None,
                  streams: int, low: int) -> None:
    """``glue.hc_reduce_act``: the split-K slices summed in order with one bf16 rounding, fused with
    ``hc_act``: the same bits as the reduce, then the act. ``part`` is [SK, R, N] fp32."""

    sk, rows, ndn = part.shape
    v = _bf(_seq_sum([part[k][:, :low] for k in range(sk)]))
    v = _bf(div(v, streams))
    a = bsilu(v)
    act[:, :low].copy_(a.to(torch.bfloat16))
    xs.view(rows, low // 32).copy_(group_sums(a, low // 32, 32))
    if inj is not None:
        iv = _bf(_seq_sum([part[k][:, low:low + streams] for k in range(sk)]))
        iv = _bf(div(iv, streams))
        inj.copy_((2.0 * bsig(iv)).to(torch.bfloat16))


def hc_mix(up: torch.Tensor, normed: torch.Tensor, mixed: torch.Tensor, xs: torch.Tensor,
           streams: int, block: int = 256) -> None:
    """``glue.hc_mix`` and the mix half of ``cuda/hc_upmix.py``: mixed[d] =
    bf16((sum over streams, in order, of bf16(bf16(sigmoid(up)) * normed)) / S) and its 32-group sums."""

    rows, wide = up.shape
    d = wide // streams
    ups = up.view(rows, streams, d)
    ns = normed.view(rows, streams, d)
    total = torch.zeros((rows, d), dtype=torch.float32, device=up.device)
    for s in range(streams):
        total = total + _bf(bsig(ups[:, s].float()) * ns[:, s].float())
    m = div(total, streams).to(torch.bfloat16)
    mixed.copy_(m)
    xs.view(rows, d // 32).copy_(group_sums(m.float(), d // 32, 32))


def write_norm(h: torch.Tensor, pss: torch.Tensor, scale: torch.Tensor, normed: torch.Tensor,
               xs: torch.Tensor, streams: int, eps: float, mode: int, branch=None, inject=None,
               y=None, wts=None) -> None:
    """``cuda/hc_fused.write_norm``'s call shape: the pending branch into h (in place) then the
    normalized read-out rows, in the released two-pass bits (what ``hc_check._check`` compares a
    fusion against). No gluon fusion ships for XPU: the Intel ``gl.sum`` reduction order is
    backend-private (it disagreed with the fixed-order folds by one bf16 ulp on the gate fixture)
    and triton's fp32 rsqrt/div are not bit-portable, so this torch composition is the path."""

    hc_writeback(h, h, pss, streams, mode, branch=branch, inject=inject, y=y, wts=wts)
    hc_normed(h, pss, scale, normed, xs, streams, eps)
