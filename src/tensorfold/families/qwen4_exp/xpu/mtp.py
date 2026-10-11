"""The MTP head on Intel GPUs: the ``cuda/mtp.py`` port on ``xpu/forward.py`` and ``xpu/state.py``.

Same signatures, same buffer slices, same bookkeeping: the head reads final residual streams and
next-token embeddings, appending its own attention entries that the next absorb trims after chained
drafts. ``glue.rmsnorm`` and ``glue.add_streams`` become ``hc``'s fixed-order sums and portable
primitives (the ``_ple_gate`` pattern), so a row's bits are its own on any backend; the draft-only
requantized head (``bf16.quantize4``'s group-32 words) runs the dequantized fp32 matmul, the same
program ``_mm`` runs for CPU tensors, because Intel's lane matmul admits only gs=64 packed weights.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from tensorfold.engine.exact_sampling import MARGIN, choose_rows

from ..cuda import bf16 as bf16_ops
from ..cuda import qmm
from ..cuda.decode import sample_mapped
from tensorfold.cuda.sampling import sample_rows
from . import hc as hc_ops
from .forward import _chunk_sums, _embed, _mm, candidates, finish, layer_forward
from .state import Buffers, State
from .weight_types import Weights


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float, group: int | None = None,
            out: torch.Tensor | None = None, xs: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``glue.rmsnorm``: each row (or each ``group``-wide run of one) RMS-normalized, times ``w``,
    stored bf16, and the stored rows' 32-group sums into ``xs``.

    The squares sum in the ``_ple_gate`` fixed order (512-wide pieces added left to right), the
    reciprocal through ``hc``'s fp64-precision sqrt and correctly-rounded division, so card and CPU
    return the same bits: ``x * rinv * w`` rounds to bf16 once, the same fp32 left-to-right order."""

    rows, d = x.shape
    g = group or d
    if d % g:
        raise ValueError(f"rmsnorm: {d} columns do not split into groups of {g}")
    if out is None:
        out = torch.empty((rows, d), dtype=torch.bfloat16, device=x.device)
    if xs is None:
        xs = torch.empty((rows, d // 32), dtype=torch.float32, device=x.device)
    xf = x.float()
    for gi in range(d // g):
        sl = slice(gi * g, (gi + 1) * g)
        xv = xf[:, sl]
        rinv = hc_ops.div(torch.ones_like(t := _chunk_sums(xv * xv, 512)),
                          hc_ops.sqrt(hc_ops.div(t, g) + eps))                    # 1 / sqrt(mean + eps)
        y = ((xv * rinv[:, None]) * w.float()[sl][None, :]).to(torch.bfloat16)
        out[:, sl].copy_(y)
        xs.view(rows, d // 32)[:, gi * (g // 32):(gi + 1) * (g // 32)].copy_(hc_ops.group_sums(y.float(), g // 32, 32))
    return out, xs


def add_streams(e: torch.Tensor, hs: torch.Tensor, out: torch.Tensor, streams: int) -> None:
    """``glue.add_streams``: ``out[r, s, :] = bf16(e[r] + hs[r, s])`` (the MTP input: the embedding
    branch added to each stream). Elementwise fp32 adds with one bf16 rounding: portable as written."""

    rows, d = e.shape
    out.view(rows, streams, d).copy_(
        (e.float()[:, None, :] + hs.view(rows, streams, d).float()).to(torch.bfloat16))


def _draft_mm(x: torch.Tensor, head, xs: torch.Tensor, out: torch.Tensor, b: Buffers) -> torch.Tensor:
    """The draft head's lane matmul. The draft-only requantization (``bf16.quantize4``) ships group-32
    words that Intel's lane matmul rejects: ``tensorfold/cuda/kernels/qmm.py``'s ``_xpu_lane_matmul``
    admits only gs=64 packed weights (a known kernel gap, documented rather than patched: the CUDA
    ``qmm.py`` and ``bf16.py`` stay frozen). Correctness first: dequantize once to bf16, wrap the
    table as the ``b16`` face and run ``_mm`` unchanged, so the card takes the same BF16 kernel every
    other NVFP4-checkpoint linear uses and the CPU reference takes ``_cpu_mm``'s dequantized fp32
    math. The ``xs`` group sums go unused by that face; the table stays attached to the head."""

    if isinstance(head, qmm.Q4):
        face = getattr(head, "_xpu_b16_face", None)
        if face is None or face.weight.device != x.device:
            face = bf16_ops.b16_from_rows(qmm.dequantize_q4(head).to(x.device, torch.bfloat16).contiguous())
            head._xpu_b16_face = face
        return _mm(x, face, xs, out, b)
    return _mm(x, head, xs, out, b)


def mtp_stage(w: Weights, b: Buffers, windows: Sequence[tuple[State, Sequence[int], torch.Tensor]]) -> list:
    """Host work before an MTP step (next tokens, input streams); returns each stream's segment."""

    segs: list = []
    for st, next_tokens, _ in windows:
        a0 = segs[-1][2] if segs else 0
        if st.mtp_len + len(next_tokens) > st.capacity:
            raise ValueError("MTP context past the cache capacity")
        segs.append((st, a0, a0 + len(next_tokens)))
    n = segs[-1][2]
    b.staged.synchronize()
    b.ids_host[:n].numpy()[:] = [int(t) for _, next_tokens, _ in windows for t in next_tokens]
    b.ids[:n].copy_(b.ids_host[:n], non_blocking=True)
    b.last_host[:len(segs)].numpy()[:] = [a1 - 1 for _, _, a1 in segs]
    b.last[:len(segs)].copy_(b.last_host[:len(segs)], non_blocking=True)
    for (_, _, streams), (_, a0, a1) in zip(windows, segs):
        if streams.data_ptr() != b.mtp_in[a0:a1].data_ptr():
            b.mtp_in[a0:a1].copy_(streams, non_blocking=True)
    b.staged.record()
    return segs


def mtp_compute(w: Weights, segs: Sequence, b: Buffers, *, last_only: bool = True,
                context: int | None = None) -> torch.Tensor:
    """The MTP head's GPU work (capturable); ``last_only``: the draft head's logits of each stream's last row."""

    c = w.cfg
    m = w.mtp
    n = segs[-1][2]
    _embed(w, b.ids[:n], 1, b.mtp_e[:n])
    en, xe = rmsnorm(b.mtp_e[:n], m.norm_e, c.eps, out=b.mixed[:n], xs=b.xs_mixed[:n])
    _mm(en, m.fc_e, xe, b.mtp_eo[:n], b)
    hn, xh = rmsnorm(b.mtp_in[:n], m.norm_h, c.eps, out=b.mtp_hn[:n], xs=b.mtp_xh[:n])
    _mm(hn.view(n * c.streams, c.hidden), m.fc_h, xh.view(n * c.streams, c.hidden // 32), b.mtp_hs[:n * c.streams], b)
    add_streams(b.mtp_eo[:n], b.mtp_hs[:n * c.streams], b.h[:n], c.streams)
    pending = layer_forward(m.layer, w, segs, b, n, None, mtp=True, context=context)
    finish(w, m.mixer, b, n, pending, logits=False)
    if not last_only:
        return _mm(b.mixed[:n], w.head, b.xs_mixed[:n], b.logits[:n], b)
    head = w.head if w.draft_head is None else w.draft_head
    k = len(segs)
    if b.prefill:                                # prefill buffers mix the last row into row 0: its logits only
        rows, xs, k = b.mixed[:1], b.xs_mixed[:1], 1
    elif k == 1:
        rows, xs = b.mixed[n - 1:n], b.xs_mixed[n - 1:n]
    else:
        rows = b.mixed.index_select(0, b.last[:k])
        xs = b.xs_mixed.index_select(0, b.last[:k])
    out = _draft_mm(rows, head, xs, b.logits.view(-1)[:k * head.n].view(k, head.n), b)     # contiguous at any k
    if w.comm is not None:
        candidates(w, b, out, k, id_map=w.draft_ids, offset=int(w.meta["vocab_offset"]))
    return out


@torch.no_grad()
def mtp_forward(w: Weights, st: State, b: Buffers, next_tokens: Sequence[int], streams: torch.Tensor,
                *, last_only: bool = True) -> torch.Tensor:
    """Process bf16 main-model streams and next tokens into last-row or all-row logits plus b.streams[:n], appending n cache entries while the caller advances ``st.mtp_len``."""

    return mtp_compute(w, mtp_stage(w, b, [(st, next_tokens, streams)]), b, last_only=last_only)


# -- the draft head's consumption side ----------------------------------------------------------------------------
# What ``cuda/decode.py``'s Engine calls for the drafts, in free functions over (w, st, mbuf): the
# engine wiring ports the Engine, these keep the draft rules and confidence next to the head.

def sample_draft(logits: torch.Tensor, position: int, sampling, draft_ids: torch.Tensor | None = None,
                 draft_host: np.ndarray | None = None) -> tuple[int, float]:
    """``Engine.sample_draft``'s single-card rule on one row: the keyed draft token and its
    temperature-1 probability; this confidence ends draft chains early without changing output."""

    host = draft_host if draft_host is not None else (
        draft_ids.detach().cpu().numpy() if draft_ids is not None else None)
    row = logits[:1].float()
    lse = torch.logsumexp(row, dim=-1, keepdim=True)
    if sampling is None or sampling.temperature <= 0:
        top, col = row.max(dim=-1, keepdim=True)             # the first maximum: argmax's (and serial's) choice
        got = torch.cat([top, lse, col.float()], dim=1).cpu().numpy()[0]      # one sync
        c = int(got[2])
        tok = int(host[c]) if host is not None else c
        return tok, float(np.exp(float(got[0]) - float(got[1])))
    k = min(row.shape[1], int(sampling.top_k) + MARGIN) if sampling.top_k else row.shape[1]
    vals, idx = torch.topk(row, k, dim=-1, sorted=False)
    got = torch.cat([vals, lse, idx.float()], dim=1).cpu().numpy()[0]         # one sync
    cols = got[k + 1:].astype(np.int64)
    ids = host[cols] if host is not None else cols
    tok = choose_rows(got[None, :k].astype(np.float32), ids[None, :], [position], sampling)[0]
    hit = np.nonzero(ids == tok)[0]
    return int(tok), float(np.exp(float(got[hit[0]]) - float(got[k]))) if len(hit) else 0.0


def _sample_row(w: Weights, logits: torch.Tensor, position: int, sampling) -> int:
    """``Engine.sample``'s single-card ``draft=True`` branch: the draft head's rows over the draft
    vocabulary through ``sample_mapped`` (shared with the CUDA engine bit-for-bit), the whole
    vocabulary through the serial keyed rule."""

    if w.draft_ids is not None:
        return sample_mapped(logits, [position], sampling, w.draft_ids)[0]
    return sample_rows(logits, [position], sampling)[0]


def absorb(w: Weights, st: State, b: Buffers, streams: torch.Tensor, next_tokens: Sequence[int]) -> torch.Tensor:
    """``cuda/decode.absorb``: the MTP cache takes positions with main-model streams [n, S*D] and next
    tokens; logits of the last. Chained drafts, not yet verified, leave the cache first."""

    if st.mtp_drafted:
        st.set_mtp_len(st.mtp_len - st.mtp_drafted)
        st.mtp_drafted = 0
    logits = mtp_forward(w, st, b, next_tokens, streams)
    st.set_mtp_len(st.mtp_len + len(next_tokens))
    return logits


def draft(w: Weights, st: State, b: Buffers, streams: torch.Tensor, next_tokens: Sequence[int], position: int,
          count: int, sampling, confidence: float = 0.0) -> list[int]:
    """``cuda/decode.draft``: absorb kept rows and chain drafts, always retaining the first even below
    ``confidence``, then stopping before later drafts below it or after a low-confidence first draft."""

    logits = absorb(w, st, b, streams, next_tokens)
    drafts: list[int] = []
    for j in range(count):
        low = False
        if confidence > 0:
            d, p = sample_draft(logits, position + j, sampling, w.draft_ids)
            low = p < confidence
            if low and j > 0:
                break
        else:
            d = _sample_row(w, logits[:1], position + j, sampling)
        drafts.append(d)
        if low:
            break
        if j + 1 < count:
            prev = b.streams[len(next_tokens) - 1:len(next_tokens)] if j == 0 else b.streams[:1]
            logits = mtp_forward(w, st, b, [d], prev)
            st.set_mtp_len(st.mtp_len + 1)
            st.mtp_drafted += 1
            next_tokens = [d]
    return drafts
