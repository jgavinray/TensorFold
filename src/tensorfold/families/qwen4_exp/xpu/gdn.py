"""Flash Next's Gated DeltaNet on XPU: gdn.cu's per-row bits in plain torch, with an XPUGraph chunk capture.

One row: the depthwise causal conv over [the stream's conv state; the window's projection rows] (4 taps,
fp32 in tap order, SiLU with one bf16 rounding), the fp32 L2 norms of q and k per key head (eps inside the
sum, q times DK^-0.5), v bf16, g = exp(-exp(A_log) * softplus(a + dt_bias)) fp32, beta = bf16(sigmoid(b)),
the delta-rule update and read-out in fp32, and the sigmoid-gated RMSNorm of the read-out into bf16 ``out``
and its 32-channel group sums ``xs`` (gdn_io.cu's front and back kernels).

The serial step is qwen3_5/xpu/gdn_fallback's (the proven row-exact reproduction of the CUDA ext), copied
rather than imported because that module is shared and its capture keeps q/k in bf16 — qwen3_5's front
rounds them, while this family's front leaves them fp32 like gdn.cu (whose ``k_save`` is fp32); bf16
buffers would silently round fp32 q/k and break replay's bit-equality with the chain.  Everything is
derived from the tensor shapes, so the layer's (16, 48) and a rank share's (8, 24) both run.
"""

from __future__ import annotations

import math

import torch

NK, NV, DK, DV = 16, 48, 128, 128
TAPS = 4
CONV = 2 * NK * DK + NV * DV
PW = CONV + NV * DV + 2 * NV

__all__ = ["NK", "NV", "DK", "DV", "TAPS", "CONV", "PW", "widths", "GDNScratch", "chain", "replay"]


def widths(nk: int, nv: int) -> tuple[int, int]:
    """(conv channels, projection row width) for nk key heads and nv value heads."""

    conv = 2 * nk * DK + nv * DV
    return conv, conv + nv * DV + 2 * nv


class GDNScratch:
    """A sequence's replay inputs for its last window of up to ``rows`` rows."""

    def __init__(self, rows: int, device, nk: int = NK, nv: int = NV) -> None:
        self.k = torch.empty((rows, nk, DK), dtype=torch.float32, device=device)
        self.v = torch.empty((rows, nv, DV), dtype=torch.bfloat16, device=device)
        self.g = torch.empty((rows, nv), dtype=torch.float32, device=device)
        self.b = torch.empty((rows, nv), dtype=torch.float32, device=device)


_GRAPH_CHUNKS: dict = {}


class _ChunkGraph:
    """A captured 64-row chain for one (device, nv, dv, dk, nk, rep) shape key — qwen3_5's
    gdn_fallback._ChunkGraph with fp32 q/k buffers (see the module docstring).

    The captured region is the serial step loop itself — identical ops and order as ``_step`` per
    row — so replay is bit-exact with the eager chain.  Inputs are copied into the static tensors
    per call; the state and outputs are read back out.
    """

    CT = 64

    def __init__(self, nv: int, dv: int, dk: int, nk: int, rep: int, device):
        self.nv, self.dv, self.dk, self.nk, self.rep = nv, dv, dk, nk, rep
        self.device = device
        self.qC = torch.zeros(self.CT, nk, dk, device=device)
        self.kC = torch.zeros(self.CT, nk, dk, device=device)
        self.vC = torch.zeros(self.CT, nv, dv, device=device, dtype=torch.bfloat16)
        self.gC = torch.zeros(self.CT, nv, device=device)
        self.bC = torch.zeros(self.CT, nv, device=device)
        self.S = torch.zeros(nv, dv, dk, device=device)
        self.Ssnap = torch.zeros(self.CT, nv, dv, dk, device=device)
        self.Y = torch.zeros(self.CT, nv, dv, device=device)
        self.yrow = torch.zeros(nv, dv, device=device)
        chunk = self._chunk
        for _ in range(2):
            chunk()
        torch.xpu.synchronize()
        self.graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(self.graph):
            chunk()

    def _chunk(self) -> None:
        S, qC, kC, vC, gC, bC, Y, yrow = self.S, self.qC, self.kC, self.vC, self.gC, self.bC, self.Y, self.yrow
        for t in range(self.CT):
            qf = qC[t].float().repeat_interleave(self.rep, 0)
            kf = kC[t].float().repeat_interleave(self.rep, 0)
            s = S * gC[t].float()[:, None, None]
            kv_mem = (s * kf[:, None, :]).sum(-1)
            delta = (vC[t].float() - kv_mem) * bC[t].float()[:, None]
            S.copy_(s + kf[:, None, :] * delta[:, :, None])
            self.Ssnap[t].copy_(S)
            yrow.copy_((S * qf[:, None, :]).sum(-1))
            Y[t].copy_(yrow)

    def run(self, q, k, v, g, beta, state) -> tuple[torch.Tensor, torch.Tensor]:
        """(q, k, v, g, beta, state) -> (Y[:T] fp32, state fp32).  T may be < CT: the tail rows keep
        the capture's stale inputs but are never read.  Stream-ordered: no internal sync."""

        T = q.shape[0]
        if T > self.CT:
            return None, None
        self.S.copy_(state)
        self.qC[:T].copy_(q)
        self.kC[:T].copy_(k)
        self.vC[:T].copy_(v)
        self.gC[:T].copy_(g)
        self.bC[:T].copy_(beta)
        self.graph.replay()
        return self.Y[:T], self.Ssnap[T - 1]


def _step(S: torch.Tensor, qt: torch.Tensor, kt: torch.Tensor, vt: torch.Tensor,
          gt: torch.Tensor, betat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One row's step (qwen3_5/xpu/gdn_fallback._step, copied — a shared file): the row's recurrence in
    fp32, S <- S*g + k ((v - k.S) beta), y <- q.S."""

    S = S * gt[:, None, None]
    kv_mem = (S * kt[:, None, :]).sum(-1)
    delta = (vt - kv_mem) * betat[:, None]
    S = S + kt[:, None, :] * delta[:, :, None]
    return S, (S * qt[:, None, :]).sum(-1)


def _expanded(rows: torch.Tensor, rep: int) -> torch.Tensor:
    """(W, nk, dk) rows (or one (nk, dk) row) to fp32 with each key head shared by its ``rep`` value heads."""

    return rows.float().repeat_interleave(rep, -2)


def _butterfly(x: torch.Tensor) -> torch.Tensor:
    """gdn.cu's warp_sum: the xor-paired halving of the last dim in fp32.  Elementwise adds only, so a
    row's bits never depend on the batch (torch's .sum picks its tree per launch shape)."""

    while x.shape[-1] > 1:
        x = x.reshape(*x.shape[:-1], 2, x.shape[-1] // 2)
        x = x[..., 0, :] + x[..., 1, :]
    return x[..., 0]


def _ss(x: torch.Tensor) -> torch.Tensor:
    """The kernels' block sum of squares: four consecutive channels per lane, then the 32-lane butterfly.
    ``x``'s last dim is 128; returns the sums with a trailing singleton for broadcasting."""

    lane = (x * x).reshape(*x.shape[:-1], x.shape[-1] // 4, 4)
    lane = lane[..., 0] + lane[..., 1] + lane[..., 2] + lane[..., 3]
    return _butterfly(lane).unsqueeze(-1)


def _front(p: torch.Tensor, conv_state: torch.Tensor, conv_w: torch.Tensor, a_log: torch.Tensor,
           dt_bias: torch.Tensor, rows: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                       torch.Tensor, torch.Tensor]:
    """gdn_io.cu's front_kernel in torch: the conv + SiLU over the projection's q | k | v channels (fp32
    taps in tap order, one bf16 rounding), q and k L2-normalized fp32 per key head (eps inside the sum, q
    times DK^-0.5), v bf16, g = exp(-exp(A_log) softplus(a + dt_bias)) fp32 and beta = bf16(sigmoid(b))
    as fp32.  ``conv_state`` [TAPS - 1, C] is read only."""

    nv = a_log.numel()
    c = conv_w.shape[0]
    nk = (c - nv * DV) // (2 * DK)
    assert conv_state.shape == (TAPS - 1, c) and p.shape[1] == c + nv * (DV + 2)
    x = torch.empty((rows + TAPS - 1, c), dtype=torch.float32, device=p.device)
    x[:TAPS - 1].copy_(conv_state)
    x[TAPS - 1:].copy_(p[:rows, :c])
    w = conv_w.float()
    acc = w[:, 0] * x[:-3] + w[:, 1] * x[1:-2] + w[:, 2] * x[2:-1] + w[:, 3] * x[3:]
    act = (acc / (1.0 + torch.exp(-acc))).to(torch.bfloat16)
    q = act[:, :nk * DK].reshape(rows, nk, DK).float()
    k = act[:, nk * DK:2 * nk * DK].reshape(rows, nk, DK).float()
    v = act[:, 2 * nk * DK:].reshape(rows, nv, DV)
    q = q * ((1.0 / torch.sqrt(_ss(q) + 1e-6)) * (1.0 / math.sqrt(DK)))
    k = k * (1.0 / torch.sqrt(_ss(k) + 1e-6))
    b = p[:rows, c + nv * DV:c + nv * DV + nv].float()
    a = p[:rows, c + nv * DV + nv:c + nv * DV + 2 * nv].float()
    g = torch.exp(-torch.exp(a_log) * torch.nn.functional.softplus(a + dt_bias))
    return q, k, v, g, torch.sigmoid(b).to(torch.bfloat16).float()


def _delta(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
           state: torch.Tensor, final: torch.Tensor) -> torch.Tensor:
    """The serial step a row at a time from ``state`` (read only) to ``final``: the rows' y in ``v``'s
    dtype.  Graph-captured for 2..64 rows on xpu, eager otherwise (a row's bits never depend on its
    window either way)."""

    rows = q.shape[0]
    if q.device.type == "xpu" and hasattr(torch.xpu, "XPUGraph") and 1 < rows <= 64 \
            and _GRAPH_CHUNKS is not None:
        rep = v.shape[1] // q.shape[1]
        key = (q.device, v.shape[1], v.shape[2], q.shape[2], q.shape[1], rep)
        cg = _GRAPH_CHUNKS.get(key)
        if cg is None:
            cg = _GRAPH_CHUNKS[key] = _ChunkGraph(v.shape[1], v.shape[2], q.shape[2], q.shape[1],
                                                  rep, q.device)
        y, s = cg.run(q, k, v, g, beta, state)
        final.copy_(s)
        return y.to(v.dtype)
    rep = v.shape[1] // q.shape[1]
    qf, kf = _expanded(q, rep), _expanded(k, rep)
    vf = v.float()
    S = state
    ys = []
    for t in range(rows):
        S, yr = _step(S, qf[t], kf[t], vf[t], g[t].float(), beta[t].float())
        ys.append(yr)
    final.copy_(S)
    return torch.stack(ys).to(v.dtype)


def _back(y: torch.Tensor, p: torch.Tensor, norm_w: torch.Tensor, eps: float, rows: int,
          out: torch.Tensor, xs: torch.Tensor) -> None:
    """gdn_io.cu's back_kernel in torch: per value head, the RMSNorm of the read-out (eps inside the sqrt
    over DV), one bf16 rounding after the scale and after the weight, the sigmoid gate from the
    projection's z channels, and ``xs`` — the fp32 sums of each 32-channel group of the final bf16 output."""

    nv = y.shape[1]
    yf = y.float()
    rinv = 1.0 / torch.sqrt(_ss(yf) / DV + eps)
    yn = (yf * rinv).to(torch.bfloat16).float() * norm_w.float()
    z = p[:rows, p.shape[1] - nv * (DV + 2):p.shape[1] - nv * (DV + 2) + nv * DV]
    o = (yn.to(torch.bfloat16).float() * torch.sigmoid(z.float()).reshape(rows, nv, DV)).to(torch.bfloat16)
    out.copy_(o.reshape(rows, -1))
    xs.copy_(_butterfly(o.float().reshape(rows, nv, DV // 32, 32)).reshape(rows, -1))


def chain(p: torch.Tensor, conv_state: torch.Tensor, conv_w: torch.Tensor, state_in: torch.Tensor,
          a_log: torch.Tensor, dt_bias: torch.Tensor, norm_w: torch.Tensor, eps: float, rows: int,
          scratch: GDNScratch, state_out: torch.Tensor, out: torch.Tensor, xs: torch.Tensor) -> None:
    """gdn.cu's chain in torch: the conv front, the serial delta-rule step over ``rows`` (graph-captured
    for 2..64 rows on xpu) and the gated-RMSNorm back.  ``out`` [rows, NV*DV] bf16 and ``xs`` (its
    32-group sums) may be row views of a larger buffer; the conv state is read only (forward.py's
    shift_windows updates it) and ``state_in`` is not written."""

    q, k, v, g, beta = _front(p, conv_state, conv_w, a_log, dt_bias, rows)
    scratch.k[:rows].copy_(k)
    scratch.v[:rows].copy_(v)
    scratch.g[:rows].copy_(g)
    scratch.b[:rows].copy_(beta)
    y = _delta(q, k, v, g, beta, state_in, state_out)
    _back(y, p, norm_w, eps, rows, out, xs)


def replay(state_in: torch.Tensor, scratch: GDNScratch, rows: int, state_out: torch.Tensor) -> None:
    """gdn.cu's replay: the same serial step over the retained k, v, g and beta, so an accepted prefix's
    state has the chain's bits."""

    rep = state_in.shape[0] // scratch.k.shape[1]
    kf = _expanded(scratch.k[:rows], rep)
    vf = scratch.v[:rows].float()
    g, b = scratch.g[:rows], scratch.b[:rows]
    S = state_in
    for t in range(rows):
        S, _ = _step(S, kf[t], kf[t], vf[t], g[t].float(), b[t].float())
    state_out.copy_(S)
