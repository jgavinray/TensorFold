"""Flash Next forward on Intel GPUs: every op treats each row alone, so a window row equals the serial step.

The ``cuda/forward.py`` port on the XPU slice modules: the GDN chain (``xpu/gdn.py``), the
hyper-connection kernels (``xpu/hc.py``), the n-gram rows (``xpu/ple.py``), the FP4 experts and
their ordered router (``xpu/nvfp4.py``, ``xpu/moe.py``), and the attention (``xpu/attention.py``).
Cross-row channel sums are elementwise-add fixed orders (the ``_butterfly``/``fold`` pattern), the
transcendentals and quotients route through ``hc``'s fp64 primitives, and the lane matmuls run on
the card through ``cuda/qmm.py`` and ``cuda/bf16.py`` unmodified; on CPU tensors the same calls
run the dequantized fp32 matmul, which makes the forward its own reference for the gate tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ..cuda import bf16, qmm
from ..cuda.hc_check import fuser as _hc_fuser
from . import attention as attn_mod
from . import gdn as gdn_mod
from . import hc as hc_ops
from . import nvfp4
from . import ple as ple_mod
from .state import ATT_ROWS, CAND, Buffers, State, _MoECfg
from .weights import HC, LayerW, Weights


def _gather(w: Weights, b: Buffers, part: torch.Tensor, flat: torch.Tensor, R: int) -> torch.Tensor:
    """All ranks' fp32 partials [R, D] in rank order: [world, R, D] (summed rank 0 first by the consumer)."""

    d = part.shape[1]
    out = flat[:b.world * R * d]
    w.comm.all_gather(part[:R], out)
    return out.view(b.world, R, d)


def _cpu_mm(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """A lane matmul on the CPU reference side: fp32 math on the dequantized rows, rounded once."""

    out.copy_((x.float() @ weight.float().t()).to(out.dtype))
    return out


def _mm(x: torch.Tensor, q: qmm.Q4, xs: torch.Tensor, out: torch.Tensor, b: Buffers, **kw) -> torch.Tensor:
    if getattr(q, "kernel", "qmm") == "b16":      # an NVFP4 checkpoint's BF16 linear (non-experts)
        if x.device.type == "cpu":
            return _cpu_mm(x, q.weight, out)
        return bf16.matmul(x, q, out=out)
    if not isinstance(q, qmm.Q4):                 # an EXL3 pack's matrix (``exl3_mm``): prompts on its prompt path
        if x.device.type == "cpu":
            raise NotImplementedError("the CPU reference runs the NVFP4 checkpoint's bf16 and FP4 faces")
        return q.prefill(x, out) if b.prefill else q(x, out)
    if x.device.type == "cpu":                    # the FP4 face dequantized; the ordered lane kernel needs the card
        return _cpu_mm(x, qmm.dequantize_q4(q), out)
    mm = qmm.prefill_matmul if b.prefill else qmm.matmul
    return mm(x, q, xs, out=out, part=b.part, **kw)


def _embed(w: Weights, ids: torch.Tensor, copies: int, out: torch.Tensor) -> torch.Tensor:
    if len(w.embed) == 1:     # an unquantized embedding: an NVFP4 checkpoint's BF16 table
        rows = w.embed[0].index_select(0, ids.long())
        out.view(rows.shape[0], copies, -1).copy_(rows[:, None, :])
        return out
    raise NotImplementedError("quantized embedding tables land with their own XPU lane")


def _image_rows(segs: Sequence[Seg], b: Buffers, streams: int) -> None:
    """``image_rows.embed``: image prompt chunks land with the vision port; plain chunks do nothing here."""

    if any(st.image_rows for st, _, _ in segs):
        raise NotImplementedError("image prompts land with the vision port")


def hc_block(hc: HC, b: Buffers, R: int, eps: float, streams: int, low: int, mode: int, inject_prev,
             inject_out, h: torch.Tensor, branch=None, y=None, wts=None) -> None:
    """Write the pending branch back into the streams h (in place), then the hyper-connection's read-out: b.mixed [R, D] (+ group sums), and its inject gates into ``inject_out``."""

    if b.prefill and R > FUSED_ROWS and isinstance(hc.down, qmm.Q4):
        fused = _hc_fuser(h.device)
        if fused is not None:
            fused(h[:R], b.pss[:R], hc.scale, b.normed[:R], b.xs_normed[:R], streams, eps, mode,
                  branch=branch, inject=inject_prev, y=y, wts=wts)
            _readout_plain(hc, b, h, R, eps, streams, low, inject_out[:R] if hc.inject else None, normed=True)
            return
    hc_ops.hc_writeback(h[:R], h[:R], b.pss[:R], streams, mode, branch=branch, inject=inject_prev, y=y, wts=wts)
    _readout(hc, b, h, R, eps, streams, low, inject_out[:R] if hc.inject else None)


FUSED_ROWS = 16      # decode windows: the read-out in 3 kernels; wider windows (prefill) in 5, the same bits


def _readout(hc: HC, b: Buffers, h: torch.Tensor, R: int, eps: float, streams: int, low: int, inject) -> None:
    """normed streams -> down -> SiLU / inject -> up -> mix: b.mixed [R, D] and its group sums."""

    if getattr(hc.down, "kernel", "qmm") == "b16":     # an NVFP4 checkpoint: the same steps, bf16 kernels
        _readout_b16(hc, b, h, R, eps, streams, low, inject)
    elif R <= FUSED_ROWS and not b.prefill and h.device.type != "cpu" and isinstance(hc.down, qmm.Q4):
        _readout_fused(hc, b, h, R, eps, streams, low, inject)
    else:
        _readout_plain(hc, b, h, R, eps, streams, low, inject)


def _readout_b16(hc: HC, b: Buffers, h: torch.Tensor, R: int, eps: float, streams: int, low: int, inject) -> None:
    """The read-out on the bf16 kernels: norm, down, activation and inject gates, up, the mix; same bits per row.

    Deviation from CUDA: the down projection stores bf16 (``bf16.matmul``'s ``f32=True`` is a CUDA
    lane); the activation therefore sees the one extra bf16 rounding, inside the gate tolerances."""

    hc_ops.hc_normed(h[:R], b.pss[:R], hc.scale, b.normed[:R], b.xs_normed[:R], streams, eps)
    got = _mm(b.normed[:R], hc.down, b.xs_normed[:R],
              torch.empty((R, hc.down.n), dtype=torch.bfloat16, device=h.device), b)
    hc_ops.hc_act(got, b.act[:R], b.xs_act[:R], inject, streams, low)
    _mm(b.act[:R], hc.up, b.xs_act[:R], b.up[:R], b)
    hc_ops.hc_mix(b.up[:R], b.normed[:R], b.mixed[:R], b.xs_mixed[:R], streams)


def _readout_fused(hc: HC, b: Buffers, h: torch.Tensor, R: int, eps: float, streams: int, low: int, inject) -> None:
    """The norm inside the down projection, the mix inside the up projection."""

    out = b.dn[:R] if hc.down.n == b.dn.shape[1] else b.dn_mix[:R]
    got = qmm.hc_down(h[:R], b.pss[:R], hc.scale, b.normed[:R], hc.down, eps, streams, out=out, part=b.part)
    if got.dim() == 3:
        hc_ops.hc_reduce_act(got, b.act[:R], b.xs_act[:R], inject, streams, low)
    else:
        hc_ops.hc_act(got, b.act[:R], b.xs_act[:R], inject, streams, low)
    qmm.hc_upmix(b.act[:R], b.xs_act[:R], hc.up, b.normed[:R], b.mixed[:R], b.xs_mixed[:R], streams)


def _readout_plain(hc: HC, b: Buffers, h: torch.Tensor, R: int, eps: float, streams: int, low: int, inject,
                   normed: bool = False) -> None:
    """The norm, the down projection with SiLU and the inject gates, the up projection, the mix: separate kernels."""

    if not normed:
        hc_ops.hc_normed(h[:R], b.pss[:R], hc.scale, b.normed[:R], b.xs_normed[:R], streams, eps)
    _down_act(hc, b, R, streams, low, inject)
    if b.prefill and R >= 512 and streams == 4 and low == 320 and isinstance(hc.up, qmm.Q4) and hc.up.n == 10240:
        upmix = _hc_fuser(h.device, upmix=True)
        if upmix is not None:
            upmix(b.act[:R], hc.up, b.normed[:R], b.mixed[:R], b.xs_mixed[:R], streams)
            return
    _mm(b.act[:R], hc.prefill_up if b.prefill else hc.up, b.xs_act[:R], b.up[:R], b)
    hc_ops.hc_mix(b.up[:R], b.normed[:R], b.mixed[:R], b.xs_mixed[:R], streams)


def _down_act(hc: HC, b: Buffers, R: int, streams: int, low: int, inject) -> None:
    """A hyper-connection's down projection, then SiLU and the inject gates: b.act, b.xs_act (and ``inject``). With a split K the slice sum is fused into the activation kernel (the same bits as reduce, then act)."""

    out = b.dn[:R] if hc.down.n == b.dn.shape[1] else b.dn_mix[:R]
    if isinstance(hc.down, qmm.Q4):
        got = _mm(b.normed[:R], hc.prefill_down if b.prefill else hc.down, b.xs_normed[:R], out, b, reduce=False)
    elif b.prefill:                                   # an EXL3 pack's fp16 matrix: summed slices, any row count
        got = hc.down(b.normed[:R], out)
    else:
        got = hc.down.partials(b.normed[:R])          # fp32 slices [SK, R, N] that the activation sums in order
    if got.dim() == 3:
        hc_ops.hc_reduce_act(got, b.act[:R], b.xs_act[:R], inject, streams, low)
    else:
        hc_ops.hc_act(got, b.act[:R], b.xs_act[:R], inject, streams, low)


Seg = tuple[State, int, int]     # a stream's committed state and its rows [a0, a1) of the window


@dataclass
class Cut:
    """A kept point ``row`` rows into the prompt piece at buffer row ``at``: its DeltaNet states and conv windows."""

    row: int
    at: int = 0
    rec: torch.Tensor | None = None          # [linear layers, heads, dv, dk], filled layer by layer
    conv: torch.Tensor | None = None         # [linear layers, taps, channels]


def gdn_block(layer: LayerW, w: Weights, segs: Sequence[Seg], b: Buffers, R: int, cuts: Sequence[Cut] = ()) -> None:
    c = w.cfg
    g = layer.gdn
    li = segs[0][0].lin_index[layer.index]
    if b.prefill:
        _mm(b.mixed[:R], g.proj, b.xs_mixed[:R], b.proj[0, :R], b)
        at = {cut.at: cut for cut in cuts}
        for st, a0, a1 in segs:
            _prefill_chain(g, st, li, b, a0, a1, c, at.get(a0))
        return _out_proj(w, b, b.gout[:R], g.out, b.gxs[:R], R)
    _mm(b.mixed[:R], g.proj, b.xs_mixed[:R], b.proj[li, :R], b)
    if getattr(b, "gdn_tables", None) is not None:
        raise NotImplementedError("concurrent rounds land with the engine port")
    for st, a0, a1 in segs:
        cur = st.cur[li]
        gdn_mod.chain(b.proj[li, a0:a1], st.conv[li], g.conv, st.rec[cur, li], g.a_log, g.dt_bias, g.norm, c.eps,
                      a1 - a0, st.scratch[li], st.rec[1 - cur, li], b.gout[a0:a1], b.gxs[a0:a1])
    return _out_proj(w, b, b.gout[:R], g.out, b.gxs[:R], R)


def _prefill_chain(g, st: State, li: int, b: Buffers, a0: int, a1: int, c, cut: Cut | None = None) -> None:
    """A prompt chunk's DeltaNet; the layer commits at once (a chunk keeps every row).

    CUDA runs the conv front over the whole chunk at once (``gdn_io.front``) and chains slices of
    it; the chain here builds its own [conv state | rows] taps, so the rows after a kept point
    re-run the front from the window shifted by the rows before it - the same taps, the same bits."""

    n, p, cur = a1 - a0, b.proj[0, a0:a1], st.cur[li]
    if cut is None:
        gdn_mod.chain(p, st.conv[li], g.conv, st.rec[cur, li], g.a_log, g.dt_bias, g.norm, c.eps, n,
                      st.scratch[li], st.rec[1 - cur, li], b.gout[a0:a1], b.gxs[a0:a1])
    else:                                        # the rows before the kept point, its state, then the rest from it
        m, mid = cut.row, st.rec[1 - cur, li]
        gdn_mod.chain(p[:m], st.conv[li], g.conv, st.rec[cur, li], g.a_log, g.dt_bias, g.norm, c.eps, m,
                      st.scratch[li], mid, b.gout[a0:a0 + m], b.gxs[a0:a0 + m])
        if cut.rec is None:
            cut.rec, cut.conv = torch.empty_like(st.rec[0]), torch.empty_like(st.conv)
        cut.rec[li].copy_(mid)
        cut.conv[li].copy_(st.conv[li])
        shift_windows(cut.conv[li:li + 1], b.proj[0:1, a0:a0 + m], m, c.conv_dim)
        gdn_mod.chain(p[m:], cut.conv[li], g.conv, mid, g.a_log, g.dt_bias, g.norm, c.eps, n - m,
                      st.scratch[li], st.rec[cur, li], b.gout[a0 + m:a1], b.gxs[a0 + m:a1])
        cur = 1 - cur                            # the second launch wrote the state back where the first read it
    st.cur[li] = 1 - cur
    shift_windows(st.conv[li:li + 1], b.proj[0:1, a0:a1], n, c.conv_dim)


def _out_proj(w: Weights, b: Buffers, x: torch.Tensor, q: qmm.Q4, xs: torch.Tensor, R: int):
    """A block's output projection: (1, bf16 branch) on one GPU; (3, gathered fp32 partials) across ranks."""

    if not isinstance(q, qmm.Q4):      # an NVFP4 checkpoint's BF16 face, or an EXL3 pack (one GPU): the bf16 branch
        return 1, _mm(x, q, xs, b.branch[:R], b)
    if w.comm is None:
        got = _mm(x, q, xs, b.branch[:R], b, reduce=False)
        if got.dim() == 3:
            return 4, got            # K slices: the write-back sums them in order (the bits of reduce, then round)
        return 1, got
    _mm(x, q, xs, b.part_branch[:R], b, f32=True)
    return 3, _gather(w, b, b.part_branch, b.g_branch, R)


def _caches(layer: LayerW, st: State, mtp: bool) -> tuple:
    """The layer's caches and committed length (on the device and on the host)."""

    if mtp:
        return st.mtp_kc, st.mtp_ikc, st.mtp_pooled, st.mtp_pos, st.mtp_len
    ai = st.att_index[layer.index]
    return st.kc[ai], st.ikc[ai], st.pooled[ai], st.pos_dev, st.pos


def attn_block(layer: LayerW, w: Weights, segs: Sequence[Seg], b: Buffers, R: int, mtp: bool,
               context: int | None = None):
    """Attention over each stream's own caches; ``context`` bounds the launches (a graph's bucket)."""

    c = w.cfg
    a = layer.attn
    _mm(b.mixed[:R], a.proj, b.xs_mixed[:R], b.pa[:R], b)
    scale = c.head_dim ** -0.5
    sections = getattr(c, "mrope_section", (11, 11, 10))
    if getattr(b, "attn_step", None) is not None:
        raise NotImplementedError("concurrent rounds land with the engine port")
    for st, a0, a1 in segs:
        cache, ikc, pooled, pos, host_pos = _caches(layer, st, mtp)
        bits = 0 if not cache.quantized else cache.bits
        keys = context if context is not None else host_pos + a1 - a0
        rope = st.image_positions
        length = 0 if rope is None else rope.shape[0]
        delta = st.rope_delta_dev if rope is not None or st.rope_delta else None
        attn_mod.attn_prep(b.pa[a0:a1], pos, a.q_scale, a.k_scale, a.iq_scale, w.inv_freq, b.q[a0:], cache.k, cache.v,
                           b.iq[a0:], ikc, c.eps, q_heads=c.heads, kv_heads=c.kv_heads, head_dim=c.head_dim,
                           index_heads=c.index_heads, index_dim=c.index_dim, ks=cache.ks, vs=cache.vs, bits=bits,
                           rope=rope, delta=delta, length=length, sections=sections)
        if b.prefill:
            if b.attn.qsa:
                attn_mod.qsa_pool(ikc, pooled, pos, a.ik_scale, w.inv_freq, c.eps, b.attn, a1 - a0, rope=rope,
                                  delta=delta, length=length, sections=sections)
            for r0 in range(a0, a1, ATT_ROWS):
                n = min(ATT_ROWS, a1 - r0)
                b.pos_blk.fill_(host_pos + r0 - a0)
                ends = host_pos + r0 - a0 + n
                if b.attn.qsa:
                    attn_mod.qsa_rows(b.iq[r0:r0 + n], pooled, b.pos_blk, b.attn, n, context=ends)
                attn_mod.attention(b.q[r0:r0 + n], cache.k, cache.v, b.pos_blk, b.attn, n, scale,
                                   out=b.attn_o[r0:r0 + n], context=ends, ks=cache.ks, vs=cache.vs, bits=bits)
            continue
        if b.attn.qsa:
            attn_mod.qsa_select(b.iq[a0:a1], ikc, pooled, pos, a.ik_scale, w.inv_freq, c.eps, b.attn, a1 - a0,
                                context=keys, rope=rope, delta=delta, length=length, sections=sections)
        o = attn_mod.attention(b.q[a0:a1], cache.k, cache.v, pos, b.attn, a1 - a0, scale, context=keys,
                               ks=cache.ks, vs=cache.vs, bits=bits)
        if len(segs) > 1:                       # the scratch output is the next stream's too
            b.attn_o[a0:a1].copy_(o[:a1 - a0])
    o = b.attn_o if b.prefill or len(segs) > 1 else o
    attn_mod.attn_gate(o[:R], b.pa[:R], b.gated[:R], b.xs_gated[:R], q_heads=c.heads, head_dim=c.head_dim)
    return _out_proj(w, b, b.gated[:R], a.o, b.xs_gated[:R], R)


def _chunk_sums(x: torch.Tensor, chunk: int) -> torch.Tensor:
    """``x`` [rows, D] fp32: the sum of each row, chunk-wide pieces added left to right (the kernels' c-loop)."""

    d = x.shape[1]
    if d % chunk:
        chunk = d
    parts = hc_ops.group_sums(x, d // chunk, chunk).unbind(1)
    acc = parts[0]
    for p in parts[1:]:
        acc = acc + p
    return acc


def _ple_gate(keys, vals, h, norm_key, norm_query, gated, pss, eps: float, streams: int) -> None:
    """``glue.ple_gate``: normalize key and query streams to bf16, signed sqrt and sigmoid on their
    scaled fp32 dot, and the bf16 gated values with the squared sums for ``_ple_conv``."""

    rows, wide = h.shape
    d = wide // streams
    kf, hf = keys.float(), h.float()
    nk, nq = norm_key.float(), norm_query.float()
    for s in range(streams):
        kv, hv = kf[:, s * d:(s + 1) * d], hf[:, s * d:(s + 1) * d]
        kss = _chunk_sums(kv * kv, 512)
        qss = _chunk_sums(hv * hv, 512)
        krinv = hc_ops.div(torch.ones_like(kss), hc_ops.sqrt(hc_ops.div(kss, d) + eps))
        qrinv = hc_ops.div(torch.ones_like(qss), hc_ops.sqrt(hc_ops.div(qss, d) + eps))
        kn = hc_ops._bf(kv * krinv[:, None] * nk[s * d:(s + 1) * d][None, :])
        qn = hc_ops._bf(hv * qrinv[:, None] * nq[s * d:(s + 1) * d][None, :])
        dot = _chunk_sums(hc_ops._bf(kn * qn), 512)
        gate = hc_ops._bf(hc_ops.div(dot, hc_ops.sqrt(torch.full_like(dot, d))))
        mag = hc_ops.sqrt(torch.maximum(torch.abs(gate), torch.full_like(gate, 1e-6)))
        gate = hc_ops._bf(torch.where(gate > 0, mag, torch.where(gate < 0, -mag, torch.zeros_like(gate))))
        sg = hc_ops.bsig(gate)
        gv = (sg[:, None] * vals.float()).to(torch.bfloat16)          # one value row serves every stream
        gated[:, s * d:(s + 1) * d].copy_(gv)
        gf = gv.float()
        pss[:, s].copy_(_chunk_sums(gf * gf, 512))


def _ple_conv(gated, pss, norm_conv, tail, conv_w, h, hout, nrow, eps: float, streams: int, dilation: int) -> None:
    """``glue.ple_conv``: convolve bf16 normed rows over [TAIL; current rows] in fp32 with taps
    ``dilation`` apart, bf16 SiLU and the residual adds, and retain the normed rows in ``nrow``."""

    rows, wide = h.shape
    d = wide // streams
    taps = conv_w.shape[1]
    nt = (taps - 1) * dilation
    gf = gated.float()
    rinv = hc_ops.div(torch.ones_like(pss), hc_ops.sqrt(hc_ops.div(pss, d) + eps))       # [rows, streams]
    stream_of = torch.arange(wide, device=gated.device) // d
    nc = norm_conv.float()[None, :]
    normed = hc_ops._bf(gf * rinv[:, stream_of] * nc)                                   # each row's own rinv
    acc = torch.zeros((rows, wide), dtype=torch.float32, device=gated.device)
    cw = conv_w.float()
    for t in range(taps):
        at = torch.arange(rows, device=gated.device) + t * dilation                     # [TAIL (NT rows); rows]
        from_tail = (at < nt)[:, None]
        xv = torch.where(from_tail, tail[at.clamp(max=nt - 1)].float(),
                         normed[(at - nt).clamp(min=0)])
        acc = acc + cw[:, t][None, :] * xv                                             # taps in order, per channel
    conv = hc_ops._bf(acc)
    act = hc_ops.bsilu(conv)
    ple = hc_ops._bf(gf + act)
    hout.copy_((h.float() + ple).to(torch.bfloat16))
    nrow.copy_(normed.to(torch.bfloat16))


def ple_block(layer: LayerW, w: Weights, segs: Sequence[Seg], b: Buffers, R: int) -> None:
    """h += the n-gram embedding branch, each stream through its own conv tail (rows staged by ``stage``)."""

    c = w.cfg
    p = layer.ple
    assert p is not None                             # ple_block only runs on a layer that carries one
    heads, dh = p.ngram.heads, p.ngram.dims          # rows staged by ``stage_ple_rows``, table scale applied
    emb = b.ple_v[:R * heads].view(R, heads * dh)
    b.ple_emb[:R].copy_(emb)
    b.xs_ple[:R].copy_(hc_ops.group_sums(emb.float(), c.ple_dim // 32, 32))
    _mm(b.ple_emb[:R], p.key, b.xs_ple[:R], b.ple_keys[:R], b)
    _mm(b.ple_emb[:R], p.value, b.xs_ple[:R], b.ple_vals[:R], b)
    _ple_gate(b.ple_keys[:R], b.ple_vals[:R], b.h[:R], p.norm_key, p.norm_query, b.ple_gated[:R],
              b.ple_pss[:R], c.eps, c.streams)
    for st, a0, a1 in segs:
        _ple_conv(b.ple_gated[a0:a1], b.ple_pss[a0:a1], p.norm_conv, st.ple_tail, p.conv, b.h[a0:a1],
                  b.h[a0:a1], b.ple_nrow[a0:a1], c.eps, c.streams, c.ngram_size)


def stage_ple_rows(p, b: Buffers, ids, at: int = 0) -> None:
    """Copy the rows' n-gram table entries (host memory map, dequantized on the card) to the buffer, from row ``at``."""

    rows = ple_mod.gather(p, ids, b.ple_v.device)                    # [L, heads, dims] bf16 on the card
    n = rows.shape[0] * rows.shape[1]
    b.ple_v[at:at + n].copy_(rows.reshape(-1, rows.shape[-1]))


def moe_block(layer: LayerW, w: Weights, b: Buffers, R: int) -> tuple:
    """Routed experts + the shared expert. Returns the pending write-back: (2, slots y, weights) on one GPU, (3, gathered fp32 partials, None) across ranks."""

    m = layer.moe
    if not isinstance(m.experts, dict) and getattr(m.experts, "kernel", "nvfp4") != "nvfp4":
        raise NotImplementedError("the XPU engine reads the NVFP4 checkpoint's expert stacks")
    buf = b.moe
    ex = m.experts
    if isinstance(ex, dict):                                         # the loader's raw stacks -> the card's transposes
        ex = getattr(m, "_xpu_stacked", None)
        if ex is None:
            ex = nvfp4.stacked4_from_experts(m.experts, b.mixed.device)
            m._xpu_stacked = ex
    nvfp4.moe(b.mixed[:R], b.xs_mixed[:R], m.router, ex, buf, _MoECfg(w.cfg))
    if w.comm is None:
        return 2, buf.y[:R], buf.wts[:R]
    _moe_partial(buf.y[:R], buf.wts[:R], b.part_moe, R)
    return 3, _gather(w, b, b.part_moe, b.g_moe, R), None


def _moe_partial(y: torch.Tensor, wts: torch.Tensor, out: torch.Tensor, rows: int) -> None:
    """``glue.moe_partial``: each row's slot outputs weighted and summed in slot order into the fp32 partial."""

    acc = torch.zeros((rows, y.shape[-1]), dtype=torch.float32, device=y.device)
    for k in range(y.shape[1]):
        acc = acc + y[:, k].float() * wts[:, k][:, None]
    out[:rows].copy_(acc)


def _writeback(h: torch.Tensor, b: Buffers, R: int, c, pending) -> None:
    """Apply a pending branch to the streams (in place), no read-out."""

    mode, a, wts, inj = pending
    if mode == 2:
        hc_ops.hc_writeback(h[:R], h[:R], b.pss[:R], c.streams, 2, inject=inj[:R], y=a, wts=wts)
    else:
        hc_ops.hc_writeback(h[:R], h[:R], b.pss[:R], c.streams, mode, branch=a, inject=inj[:R])


def _pre_moe(layer: LayerW, w: Weights, segs: Sequence[Seg], b: Buffers, R: int, pending, *, mtp: bool = False,
            context: int | None = None, cuts: Sequence[Cut] = ()) -> None:
    """A decoder layer up to its experts' input b.mixed[:R]: the n-gram branch, the mixer and both hyper-connections."""

    c = w.cfg
    h = b.h
    if layer.ple is not None:
        if pending is not None:
            _writeback(h, b, R, c, pending)
            pending = None
        ple_block(layer, w, segs, b, R)
    if pending is None:
        hc_block(layer.attn_hc, b, R, c.eps, c.streams, c.low, 0, None, b.inj_a, h)
    else:
        mode, a, wts, inj = pending
        if mode == 2:
            hc_block(layer.attn_hc, b, R, c.eps, c.streams, c.low, 2, inj[:R], b.inj_a, h, y=a, wts=wts)
        else:
            hc_block(layer.attn_hc, b, R, c.eps, c.streams, c.low, mode, inj[:R], b.inj_a, h, branch=a)
    if layer.linear:
        mode, branch = gdn_block(layer, w, segs, b, R, cuts)
    else:
        mode, branch = attn_block(layer, w, segs, b, R, mtp, context)
    hc_block(layer.mlp_hc, b, R, c.eps, c.streams, c.low, mode, b.inj_a[:R], b.inj_m, h, branch=branch)


def layer_forward(layer: LayerW, w: Weights, segs: Sequence[Seg], b: Buffers, R: int, pending, *,
                  mtp: bool = False, context: int | None = None, cuts: Sequence[Cut] = ()):
    """One decoder layer on b.h[:R]; ``pending`` = the previous MoE's (mode, branch, weights, inject) or None. Returns the new pending write-back."""

    _pre_moe(layer, w, segs, b, R, pending, mtp=mtp, context=context, cuts=cuts)
    moe_mode, a, wts = moe_block(layer, w, b, R)
    return (moe_mode, a, wts, b.inj_m)


def finish(w: Weights, mixer: HC, b: Buffers, R: int, pending, logits: bool = True,
           ends: Sequence[int] = ()) -> torch.Tensor | None:
    """The last write-back, mixer and head; a prompt pass heads only its ``ends`` rows."""

    c = w.cfg
    b.streams[:R].copy_(b.h[:R])
    _writeback(b.streams, b, R, c, pending)
    if b.prefill:
        if len(ends) > 1:
            at = torch.tensor(list(ends), dtype=torch.long, device=b.pss.device)
            b.pss[:len(ends)].copy_(b.pss.index_select(0, at))
            rows = b.streams.index_select(0, at)
        else:
            last = ends[0] if ends else R - 1
            b.pss[0].copy_(b.pss[last])
            rows = b.streams[last:last + 1]
        R = max(1, len(ends))
        _readout(mixer, b, rows, R, c.eps, c.streams, c.low, None)
    else:
        _readout(mixer, b, b.streams, R, c.eps, c.streams, c.low, None)
    if not logits:
        return None
    out = _mm(b.mixed[:R], w.head, b.xs_mixed[:R], b.logits[:R], b)
    if w.comm is not None:
        candidates(w, b, out, R, offset=int(w.meta["vocab_offset"]))
    return out


def candidates(w: Weights, b: Buffers, logits: torch.Tensor, R: int, *, id_map: torch.Tensor | None = None,
               offset: int = 0) -> None:
    """Gather each rank's top CAND values, global ids as int32 bits and log-sum-exp into b.cand_all [world, R, 2 CAND + 1] inside the step graph, avoiding a sampling collective."""

    lf = logits.float()
    vals, idx = torch.topk(lf, CAND, dim=-1, sorted=False)
    ids = (id_map[idx] if id_map is not None else idx + offset).to(torch.int32)
    c = b.cand[:R]
    c[:, :CAND] = vals
    c[:, CAND:2 * CAND] = ids.view(torch.float32)
    c[:, 2 * CAND:] = torch.logsumexp(lf, dim=-1, keepdim=True)
    w.comm.all_gather(c, b.cand_all[:b.world * R * (2 * CAND + 1)])


def stage(w: Weights, b: Buffers, windows: Sequence[tuple[State, Sequence[int]]]) -> list[Seg]:
    """Host work before a forward (token ids, n-gram rows into static buffers); returns each stream's segment."""

    segs: list[Seg] = []
    for st, tokens in windows:
        a0 = segs[-1][2] if segs else 0
        if st.pos + len(tokens) > st.capacity:
            raise ValueError("context past the cache capacity")
        segs.append((st, a0, a0 + len(tokens)))
    R = segs[-1][2]
    if R > b.rows:
        raise ValueError(f"window of {R} rows, buffers hold {b.rows}")
    b.staged.synchronize()               # the previous step's copies out of the pinned buffers are done
    b.ids_host[:R].numpy()[:] = np.asarray([t for _, tokens in windows for t in tokens], dtype=np.int32)
    b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
    for layer in w.layers:
        if layer.ple is not None:
            p = layer.ple
            for (st, tokens), (_, a0, _) in zip(windows, segs):
                toks = np.asarray(tokens, dtype=np.int64)
                ids = p.ngram.ids(st.ple_history, toks)
                st.ple_last = (st.ple_history, toks)
                stage_ple_rows(p, b, ids, at=a0 * (ids.size // len(toks)))       # ids [rows, heads]
    b.staged.record()
    return segs


def compute(w: Weights, segs: Sequence[Seg], b: Buffers, *, logits: bool = True, context: int | None = None,
            ends: Sequence[int] = (), cuts: Sequence[Cut] = (), features=None):
    """The forward's GPU work on staged rows (capturable); ``context`` bounds attention, ``ends`` get the head, ``cuts`` keep states."""

    c = w.cfg
    R = segs[-1][2]
    _embed(w, b.ids[:R], c.streams, b.h[:R])
    if b.prefill:
        _image_rows(segs, b, c.streams)
    if features is not None:
        target, source = features
        b.h.index_copy_(0, target, source.to(b.h.dtype).repeat(1, c.streams))
    pending = None
    for layer in w.layers:
        pending = layer_forward(layer, w, segs, b, R, pending, context=context, cuts=cuts)
    return finish(w, w.mixer, b, R, pending, logits=logits, ends=ends)


def converges(w: Weights) -> bool:
    """Whether a decode window and a prompt pass can share each layer's expert launch (the grouped FP4 experts, one GPU)."""

    return w.comm is None and getattr(w, "x3", None) is None and all(
        isinstance(layer.moe.experts, dict) or getattr(layer.moe.experts, "kernel", "nvfp4") == "nvfp4"
        for layer in w.layers)


def compute_mixed(w: Weights, dsegs: Sequence[Seg], db: Buffers, psegs: Sequence[Seg], pb: Buffers, *,
                  ends: Sequence[int] = (), cuts: Sequence[Cut] = ()) -> tuple:
    """A decode window and a prompt pass in one forward, each on its own kernels and bits, experts read once."""

    c = w.cfg
    Rd, Rp = dsegs[-1][2], psegs[-1][2]
    if Rp + Rd > pb.rows:
        raise ValueError(f"a pass of {Rp} rows and a window of {Rd} exceed the prompt buffers' {pb.rows}")
    _embed(w, db.ids[:Rd], c.streams, db.h[:Rd])
    _embed(w, pb.ids[:Rp], c.streams, pb.h[:Rp])
    _image_rows(psegs, pb, c.streams)
    dp = pp = None
    for layer in w.layers:
        _pre_moe(layer, w, dsegs, db, Rd, dp)
        _pre_moe(layer, w, psegs, pb, Rp, pp, cuts=cuts)
        pb.mixed[Rp:Rp + Rd].copy_(db.mixed[:Rd])
        mode, y, wts = moe_block(layer, w, pb, Rp + Rd)
        dp, pp = (mode, y[Rp:], wts[Rp:], db.inj_m), (mode, y[:Rp], wts[:Rp], pb.inj_m)
    return finish(w, w.mixer, db, Rd, dp), finish(w, w.mixer, pb, Rp, pp, logits=bool(ends), ends=ends)


@torch.no_grad()
def forward(w: Weights, st: State, b: Buffers, tokens: Sequence[int], *, logits: bool = True,
            cut: Cut | None = None, features=None):
    """Rows for ``tokens`` at positions st.pos .. st.pos + R - 1: logits [R, V] bf16 (a view of b.logits) and the residual streams b.streams[:R]. The committed state is unchanged until ``commit``; ``cut`` (a prompt chunk): keeps each DeltaNet layer's state at its row."""

    if cut is not None and not (b.prefill and cut.at == 0 and 0 < cut.row < len(tokens)):
        raise ValueError(f"a prompt chunk of {len(tokens)} rows has no kept point at row {cut.row}")
    return compute(w, stage(w, b, [(st, tokens)]), b, logits=logits,
                   cuts=() if cut is None else (cut,), features=features)


def shift_windows(old: torch.Tensor, new: torch.Tensor, keep: int, channels: int) -> None:
    """old [L, T, C] (in place), new [L, R, W >= C] (the first C columns of each row are the window's).

    ``cuda/forward.py``'s triton kernel in torch: window rows j < T become rows keep + j of
    [old (T rows); new rows]; every element reads its source before any store, so one pass copies."""

    layers, taps, _ = old.shape
    j = torch.arange(taps, device=old.device)
    src = keep + j
    from_old = (src < taps)[:, None]
    old_rows = old[:, src.clamp(max=taps - 1)]
    new_rows = new[:, (src - taps).clamp(min=0)][..., :channels]
    rows = torch.where(from_old, old_rows, new_rows.to(old.dtype))
    old.copy_(rows)


@torch.no_grad()
def commit(w: Weights, st: State, b: Buffers, R: int, keep: int, at: int = 0, states: bool = True) -> None:
    """Keep the first ``keep`` of R rows ``st`` ran from row ``at``; ``states=False``: the concurrent round committed them."""

    c = w.cfg
    if not 1 <= keep <= R or (b.prefill and keep != R):
        raise ValueError("keep must be in 1..R, and all of a prompt chunk")
    n = 0 if b.prefill else len(st.cur)          # a prompt chunk's DeltaNet layers committed during the forward
    if n:
        for li in range(n if states else 0):
            cur = st.cur[li]
            if keep < R:
                gdn_mod.replay(st.rec[cur, li], st.scratch[li], keep, st.rec[1 - cur, li])
            st.cur[li] = 1 - cur
        shift_windows(st.conv, b.proj[:, at:at + R], keep, c.conv_dim)
    if st.ple_last is not None:
        history, tokens = st.ple_last
        st.ple_history = np.concatenate([history, tokens[:keep]])[-(c.ngram_size - 1):]
        st.ple_last = None
        tail = st.ple_tail
        shift_windows(tail[None], b.ple_nrow[None, at:at + R], keep, tail.shape[1])
    st.set_pos(st.pos + keep)


def cut_snapshot(w: Weights, st: State, b: Buffers, cut: Cut, mtp_len: int) -> dict:
    """``State.snapshot`` at a kept point inside a prompt piece, before ``commit``: the n-gram windows after its rows."""

    c = w.cfg
    tail, history = st.ple_tail.clone(), st.ple_history
    if st.ple_last is not None:
        before, tokens = st.ple_last
        history = np.concatenate([before, tokens[:cut.row]])[-(c.ngram_size - 1):]
        shift_windows(tail[None], b.ple_nrow[None, cut.at:cut.at + cut.row], cut.row, tail.shape[1])
    return {"pos": st.pos + cut.row, "rec": cut.rec, "conv": cut.conv, "ple_tail": tail,
            "ple_history": None if history is None else history.copy(), "mtp_len": mtp_len}
