"""Flash Next attention and QSA on Intel GPUs: the CUDA kernels' row-at-a-time semantics in torch ops.

Every op here is elementwise or a fixed-order reduction, so the same call on CPU tensors is the
gate's reference: scores and the p-dot are ``nvfp4.mm_ordered`` (rank-one updates in ``k`` order),
softmax ``exp`` is ``nvfp4.exp``, sums are ``hc.fold`` halfings, roots and quotients are the ``hc``
fp64 primitives, and cos/sin are the fp64 reductions below.  A row reads only its own keys, tiles
in chunk order and merges partials in chunk order, exactly like ``_chunks``/``_chunk``/
``_merge_row``, so a window row equals the serial step."""

from __future__ import annotations

import torch

from . import hc, nvfp4

CHUNK = 512
TILE = 64

# -- portable rotary: cos and sin at fp64 precision, each rounded once ----------------------------------------

_CW = (1.5707963267948966, 6.123233995736766e-17, 6.123233995663300e-34)     # Cody-Waite fp64 pi/2 terms
_INV_HALF_PI = 0.6366197723675814                                             # 2 / pi, fp64
_SIN_FACT = (1.0, 6.0, 120.0, 5040.0, 362880.0, 39916800.0, 6227020800.0)     # (2k + 1)!, k = 0..6
_COS_FACT = (1.0, 2.0, 24.0, 720.0, 40320.0, 3628800.0, 479001600.0, 87178291200.0)   # (2k)!, k = 0..7


def _cos_sin(ang: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of an fp32 tensor: fp64 quadrant reduction and Maclaurin sums, one fp32 rounding each."""

    x = ang.double()
    n = torch.round(x * _INV_HALF_PI).to(torch.int64)
    r = x
    for t in _CW:
        r = r - torch.mul(n.to(torch.float64), t)          # |r| <= pi/4
    r2 = r * r
    sinr = torch.zeros_like(r)
    pw = r
    for k, f in enumerate(_SIN_FACT):
        sinr = sinr - pw / f if k % 2 else sinr + pw / f
        pw = pw * r2
    cosr = torch.zeros_like(r)
    pw = torch.ones_like(r)
    for k, f in enumerate(_COS_FACT):
        cosr = cosr - pw / f if k % 2 else cosr + pw / f
        pw = pw * r2
    q = n % 4
    odd = (q == 1) | (q == 2)
    sneg = (q == 2) | (q == 3)
    cosneg = (q == 1) | (q == 2)
    s = torch.where(odd, cosr, sinr)
    cn = torch.where(odd, sinr, cosr)
    s = torch.where(sneg, -s, s)
    cn = torch.where(cosneg, -cn, cn)
    return cn.to(torch.float32), s.to(torch.float32)


# -- the quantized cache's transform (kvquant's kernels, same bits) ---------------------------------------------

_C32 = 0.17677669529663688110      # 1 / sqrt(32): the stored bits are ExLlamaV3's


def _h32(x: torch.Tensor) -> torch.Tensor:
    """H32 over the last axis of an (..., 32) fp32 block: five add/sub butterfly stages, then 1/sqrt(32)."""

    m = x.shape[0]
    x = x.reshape(m, 32)
    for hi, lo in ((16, 1), (8, 2), (4, 4), (2, 8), (1, 16)):
        t = x.reshape(m, hi, 2, lo)
        a, b = t[:, :, 0, :], t[:, :, 1, :]
        x = torch.stack((a + b, a - b), dim=2).reshape(m, 32)
    return x * _C32


def _quant_scale(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = _h32(x)
    s = torch.max(x.abs(), dim=-1).values + 1e-10
    return x, s, hc.div(torch.ones_like(s), s)


def quant_groups_8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(M, 32) fp32 -> int8 codes (M, 32), ``q - 128``, and fp16 scales."""

    x, s, inv = _quant_scale(x)
    q = torch.floor(x * inv[..., None] * 128.0) + 128.0
    code = torch.minimum(torch.maximum(q, 0.0), 255.0) - 128.0
    return code.to(torch.int8), s.to(torch.float16)


def quant_groups_4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(M, 32) fp32 -> uint8 (M, 16), two unsigned codes per byte, low nibble = even index."""

    x, s, inv = _quant_scale(x)
    q = torch.floor(x * inv[..., None] * 8.0) + 8.0
    q = torch.minimum(torch.maximum(q, 0.0), 15.0).to(torch.int32)
    lo, hi = q[..., 0::2], q[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8), s.to(torch.float16)


def dequant_group_8(code: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """int8 codes (..., W) -> bf16, still rotated: ``(code + 0.5) * s / 128``."""

    w = code.shape[-1]
    s = scale.to(torch.float32).unsqueeze(-1)
    c = code.to(torch.float32)
    return ((c + 0.5) * s * 0.0078125).to(torch.bfloat16)


def dequant_group_4(code: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """uint8 (..., W/2), low nibble first -> bf16 (..., W), still rotated: ``(q - 7.5) * s / 8``."""

    raw = code.to(torch.int32)
    lo, hi = raw & 15, (raw >> 4) & 15
    q = torch.stack((lo, hi), dim=-1).reshape(*code.shape[:-1], code.shape[-1] * 2).to(torch.float32)
    s = scale.to(torch.float32).unsqueeze(-1)
    return ((q - 7.5) * s * 0.125).to(torch.bfloat16)


# -- the scratch (cuda/attention.py's, same fields) -------------------------------------------------------------

class AttnScratch:
    def __init__(self, rows: int, heads: int, head_dim: int, capacity: int, device, *, budget: int = 2048,
                 ratio: int = 4) -> None:
        # a row reads at most budget + ratio - 1 keys (dense below the budget, its selected blocks and tail past it)
        self.nch = -(-min(capacity, budget + ratio - 1) // CHUNK)
        self.po = torch.zeros((rows, self.nch, heads, head_dim), dtype=torch.float32, device=device)
        self.pm = torch.zeros((rows, self.nch, heads), dtype=torch.float32, device=device)
        self.pl = torch.zeros((rows, self.nch, heads), dtype=torch.float32, device=device)
        self.out = torch.empty((rows, heads, head_dim), dtype=torch.bfloat16, device=device)
        # sparse attention (QSA): each row's key list, its length, whether it is sparse, block scores
        self.budget, self.ratio = budget, ratio
        self.idw = budget + ratio
        self.qsa = capacity > budget
        self.nb = -(-capacity // ratio)
        self.ids = torch.zeros((rows, self.idw), dtype=torch.int32, device=device)
        self.nk = torch.zeros((rows,), dtype=torch.int32, device=device)
        self.sparse = torch.zeros((rows,), dtype=torch.int32, device=device)
        self.scores = torch.zeros((rows, self.nb), dtype=torch.float32, device=device) if self.qsa else None


# -- the rope axis (image_rows.rope_axis for one row) -----------------------------------------------------------

def _rope_axis(pos: int, mode: int, *, rope: torch.Tensor | None, delta: torch.Tensor | None, length: int,
               index: torch.Tensor, s1: int, s2: int, device) -> torch.Tensor:
    """``rope_axis`` for position ``pos``: the axis per rotary half (mrope picks t/h/w, texts past images shift)."""

    if mode == 0:
        return torch.full_like(index, float(pos))
    if mode == 1:
        return torch.full_like(index, float(pos)) + delta.to(torch.float32)
    live = pos < length
    base = torch.zeros_like(index)
    if live:
        pt, ph, pw = (rope[pos, c].to(torch.float32) for c in range(3))
        axis = torch.where((index % 3 == 1) & (index < 3 * s1), ph,
                           torch.where((index % 3 == 2) & (index < 3 * s2), pw, pt))
        return axis
    return base + float(pos) + delta.to(torch.float32)


# -- attn_prep / attn_gate (glue's kernels, same signatures) -----------------------------------------------------

def attn_prep(p: torch.Tensor, pos0: torch.Tensor, q_scale, k_scale, i_scale, inv_freq, q, kc, vc, iq, ikc,
              eps: float, *, q_heads: int, kv_heads: int, head_dim: int, index_heads: int, index_dim: int,
              ks: torch.Tensor | None = None, vs: torch.Tensor | None = None, bits: int = 0,
              rope: torch.Tensor | None = None, delta: torch.Tensor | None = None, length: int = 0,
              sections: tuple[int, int, int] = (11, 11, 10)) -> None:
    """Write normalized queries and cache rows using text positions or the full image prompt's rotary positions."""

    rows, pw = p.shape
    if bits and (ks is None or vs is None):
        raise ValueError("a quantized KV cache needs its scale tensors")
    if ks is None:
        ks = vs = kc
    if kc.dim() == 3:                                                  # slots address the flattened cache
        kc, vc = kc.reshape(-1, kc.shape[-1]), vc.reshape(-1, vc.shape[-1])
        if bits:
            ks, vs = ks.reshape(-1, ks.shape[-1]), vs.reshape(-1, vs.shape[-1])
    mode = 2 if rope is not None else 1 if delta is not None else 0
    p0 = int(pos0.item())
    half = inv_freq.numel()
    dev = p.device
    f32 = torch.float32

    def head_norm(src: torch.Tensor, w: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = src.to(f32)                                       # [n, width] per head, its own norm
        den = hc.sqrt(hc.fold(x * x) / width + eps)
        rinv = hc.div(torch.ones_like(den), den)
        xn = (x * rinv[..., None] * w.to(f32)).to(torch.bfloat16).to(f32)
        partner = torch.arange(width, device=dev)
        partner = torch.where(partner < half, partner + half,
                              torch.where(partner < 2 * half, partner - half, partner)).clamp(max=width - 1)
        xp = src[:, partner].to(f32)
        xpn = (xp * rinv[..., None] * w.to(f32)[partner]).to(torch.bfloat16).to(f32)
        return x, xn, xpn

    d = torch.arange(head_dim, device=dev)
    i = torch.where(d < half, d, torch.where(d < 2 * half, d - half, torch.zeros_like(d)))
    qs, ks_w, is_w = q_scale.to(f32), k_scale.to(f32), i_scale.to(f32)

    for r in range(rows):
        pos = p0 + r
        axis = _rope_axis(pos, mode, rope=rope, delta=delta, length=length, index=i, s1=sections[1],
                          s2=sections[2], device=dev)
        ang = axis * inv_freq[i].to(f32)
        cos, sin = _cos_sin(ang)

        def rot(xn: torch.Tensor, xpn: torch.Tensor, width: int) -> torch.Tensor:
            lo = xn[:, :half] * cos[:half] - xpn[:, :half] * sin[:half]
            hi = xpn[:, half:2 * half] * cos[half:2 * half] + xn[:, half:2 * half] * sin[half:2 * half]
            tail = xn[:, 2 * half:]
            return torch.cat((lo, hi, tail), dim=-1).to(torch.bfloat16)

        for h in range(q_heads):
            src = p[r:r + 1, h * 2 * head_dim:h * 2 * head_dim + head_dim]
            _, xn, xpn = head_norm(src, qs, head_dim)
            out = rot(xn, xpn, head_dim)
            if bits:                                            # the query rides the cache's rotation
                out = _h32(out.to(f32).reshape(head_dim // 32, 32)).reshape(1, head_dim).to(torch.bfloat16)
            q[r, h].copy_(out[0])
        for h in range(kv_heads):
            src = p[r:r + 1, q_heads * 2 * head_dim + h * head_dim:q_heads * 2 * head_dim + (h + 1) * head_dim]
            _, xn, xpn = head_norm(src, ks_w, head_dim)
            out = rot(xn, xpn, head_dim)
            slot = pos * kv_heads + h
            v = p[r, q_heads * 2 * head_dim + kv_heads * head_dim + h * head_dim:
                  q_heads * 2 * head_dim + (kv_heads + 1) * head_dim + h * head_dim]
            if bits:
                block = out.to(f32).reshape(head_dim // 32, 32)
                vblock = v.to(f32).reshape(head_dim // 32, 32)
                if bits == 4:
                    kcode, ksc = quant_groups_4(block)
                    vcode, vsc = quant_groups_4(vblock)
                    kc[slot].copy_(kcode.reshape(-1).view(torch.uint8).to(kc.dtype) if kc.dtype != torch.uint8
                                   else kcode.reshape(-1))
                    vc[slot].copy_(vcode.reshape(-1).view(torch.uint8).to(vc.dtype) if vc.dtype != torch.uint8
                                   else vcode.reshape(-1))
                else:
                    kcode, ksc = quant_groups_8(block)
                    vcode, vsc = quant_groups_8(vblock)
                    kc[slot].copy_(kcode.reshape(-1))
                    vc[slot].copy_(vcode.reshape(-1))
                ks[slot].copy_(ksc)
                vs[slot].copy_(vsc)
            else:
                kc[slot].copy_(out[0])
                vc[slot].copy_(v)
        di = torch.arange(index_dim, device=dev)
        ii = torch.where(di < half, di, torch.where(di < 2 * half, di - half, torch.zeros_like(di)))
        for h in range(index_heads):
            base = q_heads * 2 * head_dim + 2 * kv_heads * head_dim + h * index_dim
            src = p[r:r + 1, base:base + index_dim]
            w = is_w[:index_dim]
            _, xn, xpn = head_norm(src, w, index_dim)
            axis = _rope_axis(pos, mode, rope=rope, delta=delta, length=length, index=ii, s1=sections[1],
                               s2=sections[2], device=dev)
            cosv, sinv = _cos_sin(axis * inv_freq[ii].to(f32))
            lo_w = min(half, index_dim)
            hi_w = min(2 * half, index_dim)
            lo = xn[:, :lo_w] * cosv[:lo_w] - xpn[:, :lo_w] * sinv[:lo_w]
            parts = [lo]
            if index_dim > half:
                parts.append(xpn[:, half:hi_w] * cosv[half:hi_w] + xn[:, half:hi_w] * sinv[half:hi_w])
            if index_dim > hi_w:
                parts.append(xn[:, hi_w:])
            out = torch.cat(parts, dim=-1).to(torch.bfloat16)
            iq[r, h, :index_dim].copy_(out[0])
        base = q_heads * 2 * head_dim + 2 * kv_heads * head_dim + index_heads * index_dim
        ikc[p0 + r].copy_(p[r, base:base + index_dim])


def attn_gate(o: torch.Tensor, p: torch.Tensor, out: torch.Tensor, xs: torch.Tensor, *, q_heads: int,
              head_dim: int) -> None:
    """glue.attn_gate: bf16(o * sigmoid(gate)) (fp32 math), the gate from the [q | gate] pairs, group sums."""

    rows = o.shape[0]
    of = o.float()
    g = p[:, :q_heads * 2 * head_dim].view(rows, q_heads, 2, head_dim)[:, :, 1, :].float()
    g = hc.div(g, 1.0 + nvfp4.exp(-g)).to(torch.bfloat16).to(torch.float32)                    # _bsig's bits
    v = (of.view(rows, q_heads, head_dim) * g).to(torch.bfloat16)
    out.view(rows, q_heads * head_dim).copy_(v.reshape(rows, -1))
    xs.view(rows, q_heads * head_dim // 32).copy_(hc.group_sums(v.float().reshape(rows, -1),
                                                                 q_heads * head_dim // 32, 32))


# -- the attention itself ---------------------------------------------------------------------------------------

def attention(q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, pos0: torch.Tensor, scratch: AttnScratch,
              rows: int, scale: float, out: torch.Tensor | None = None, *,
              context: int | None = None, ks: torch.Tensor | None = None, vs: torch.Tensor | None = None,
              bits: int = 0) -> torch.Tensor:
    """q [R, H, D] against caches holding [0, P0 + R) -> [R, H, D] bf16; sparse rows read ``scratch.ids``;
    ``bits`` 8 or 4: codes with ``ks``/``vs`` scales, q rotated in, the output equal to the rotated dot."""

    if bits and (ks is None or vs is None):
        raise ValueError("a quantized KV cache needs its scale tensors")
    if ks is None:
        ks = vs = kc
    _, h, d = q.shape
    hk = kc.shape[1]
    if kc.dim() == 3:
        kc, vc = kc.reshape(-1, kc.shape[-1]), vc.reshape(-1, vc.shape[-1])
        if bits:
            ks, vs = ks.reshape(-1, ks.shape[-1]), vs.reshape(-1, vs.shape[-1])
    g = h // hk
    if g > 16:
        raise ValueError(
            f"this attention kernel tiles 16 query heads per KV head (G={g} does not fit); "
            "a G under 16 is masked off after the rotation, not rotated differently")
    nch = scratch.nch
    keys = nch * CHUNK if context is None else context
    if scratch.qsa:
        keys = min(keys, (scratch.budget // scratch.ratio + 1) * scratch.ratio - 1)
    chunks = min(nch, -(-keys // CHUNK))
    out = scratch.out if out is None else out
    p0 = int(pos0.item())
    neg = float("-inf")
    for r in range(rows):
        n = p0 + r + 1
        sparse = False
        if scratch.qsa:
            sparse = bool(int(scratch.sparse[r]) != 0)
            if sparse:
                n = int(scratch.nk[r])
        bound = min(n, chunks * CHUNK)                        # chunks past a row's keys run nothing
        for kv in range(hk):
            heads = slice(kv * g, (kv + 1) * g)
            qr = q[r, heads]                                  # [g, D] bf16
            m = torch.full((g,), neg, dtype=torch.float32, device=q.device)
            l = torch.zeros((g,), dtype=torch.float32, device=q.device)
            o = torch.zeros((g, d), dtype=torch.float32, device=q.device)
            for start in range(0, bound, CHUNK):
                for t0 in range(0, min(bound - start, CHUNK), TILE):
                    at = t0 + torch.arange(TILE, device=q.device)
                    ki = start + at
                    valid = ki < n
                    idx = ki
                    if sparse:
                        idx = scratch.ids[r, ki.clamp(max=scratch.idw - 1)].to(torch.int64)
                    slot = (idx * hk + kv).clamp(max=kc.shape[0] - 1)
                    if bits == 8:
                        kk = dequant_group_8(kc[slot], ks[slot]).to(torch.float32)
                        vv = dequant_group_8(vc[slot], vs[slot]).to(torch.float32)
                    elif bits == 4:
                        kk = dequant_group_4(kc[slot], ks[slot]).to(torch.float32)
                        vv = dequant_group_4(vc[slot], vs[slot]).to(torch.float32)
                    else:
                        kk = kc[slot].to(torch.float32)
                        vv = vc[slot].to(torch.float32)
                    scores = nvfp4.mm_ordered(qr, kk.t()) * scale
                    scores = torch.where(valid[None, :], scores, torch.full_like(scores, neg))
                    tile_m = scores.max(dim=1).values
                    active = tile_m != neg
                    next_m = torch.where(active, torch.maximum(m, tile_m), m)
                    alpha = torch.where(active, torch.where(m == neg, torch.zeros_like(m),
                                                            nvfp4.exp(m - next_m)), torch.ones_like(m))
                    p = torch.where(valid[None, :] & active[:, None],
                                    nvfp4.exp(scores - next_m[:, None]),
                                    torch.zeros_like(scores))
                    o = o * alpha[:, None] + nvfp4.mm_ordered(p.to(torch.bfloat16), vv)
                    l = l * alpha + hc.fold(p)
                    m = next_m
            out[r, heads].copy_(hc.div(o, l[:, None]).to(torch.bfloat16))
    return out


# -- QSA: pool, score and select (the CUDA kernels' fixed orders) -------------------------------------------------

def qsa_pool(ikc: torch.Tensor, pooled: torch.Tensor, pos0: torch.Tensor, ik_scale: torch.Tensor,
             inv_freq: torch.Tensor, eps: float, scratch: AttnScratch, rows: int, *, rope=None, delta=None,
             length=0, sections=(11, 11, 10)) -> None:
    """Pool completed blocks with full prompt rotary positions, including a block spanning prompt pieces."""

    mode = 2 if rope is not None else 1 if delta is not None else 0
    p0 = int(pos0.item())
    ratio, di = scratch.ratio, ikc.shape[1]
    half = inv_freq.numel()
    dev = ikc.device
    d = torch.arange(di, device=dev)
    j = torch.where(d < half, d, torch.where(d < 2 * half, d - half, torch.zeros_like(d)))
    partner = torch.where(d < half, d + half, torch.where(d < 2 * half, d - half, d))
    w = ik_scale.to(torch.float32)
    for i in range(rows // ratio + 2):
        b = p0 // ratio + i
        if ratio * b + ratio > p0 + rows:
            continue
        acc = ikc[ratio * b].to(torch.float32)
        for k in range(1, ratio):
            acc = acc + ikc[ratio * b + k].to(torch.float32)
        x = (acc / ratio).to(torch.bfloat16).to(torch.float32)
        den = hc.sqrt(hc.fold(x * x) / di + eps)
        rinv = hc.div(torch.ones_like(den), den)
        xn = (x * rinv * w).to(torch.bfloat16).to(torch.float32)
        xp = ikc[ratio * b, partner].to(torch.float32)
        for k in range(1, ratio):
            xp = xp + ikc[ratio * b + k, partner].to(torch.float32)
        xp = (xp / ratio).to(torch.bfloat16).to(torch.float32)
        xpn = (xp * rinv * w[partner]).to(torch.bfloat16).to(torch.float32)
        axis = _rope_axis(ratio * b, mode, rope=rope, delta=delta, length=length, index=j, s1=sections[1],
                          s2=sections[2], device=dev)
        cos, sin = _cos_sin(axis * inv_freq)
        lo = xn[:half] * cos[:half] - xpn[:half] * sin[:half]
        hi = xpn[half:2 * half] * cos[half:2 * half] + xn[half:2 * half] * sin[half:2 * half]
        rot = torch.cat((lo, hi, xn[2 * half:]), dim=-1)
        pooled[b].copy_(rot.to(torch.bfloat16))


def qsa_rows(iq: torch.Tensor, pooled: torch.Tensor, pos0: torch.Tensor, scratch: AttnScratch, rows: int, *,
             context: int | None = None) -> None:
    """Score and select the blocks of rows P0 .. P0 + rows - 1; ``context`` bounds the blocks scored."""

    di = pooled.shape[1]
    ratio, top = scratch.ratio, scratch.budget // scratch.ratio
    p0 = int(pos0.item())
    blocks = scratch.nb if context is None else min(scratch.nb, max(1, -(-context // ratio)))
    f32 = torch.float32
    for r in range(rows):
        end = p0 + r + 1
        complete = end // ratio
        if complete <= top:
            scratch.nk[r] = end
            scratch.sparse[r] = 0
            continue
        scored = min(complete, blocks)
        for j0 in range(0, scored, 64):
            b = j0 + torch.arange(min(64, scored - j0), device=pooled.device)
            k = pooled[b].to(f32)                                    # [B, DI]
            total = torch.zeros((b.numel(),), dtype=f32, device=pooled.device)
            for h in range(iq.shape[1]):
                total = total + torch.maximum(hc.fold(k * iq[r, h].to(f32)), torch.zeros_like(total))
            scratch.scores[r, b] = hc.div(total, float(di) ** 0.5)
        _select_row(scratch, r, end, complete, ratio, top)


def _select_row(scratch: AttnScratch, r: int, end: int, complete: int, ratio: int, top: int) -> None:
    """``_select`` for one row: the top blocks by the order-preserving key, ties to lower block ids, then the tail."""

    dev = scratch.scores.device
    b = torch.arange(complete, dtype=torch.int32, device=dev)
    v = scratch.scores[r, :complete]
    bits = v.view(torch.int32)
    key = torch.where((bits < 0), ~bits, bits | (-2147483648))            # signed mirror of the uint32 keys
    key = key.to(torch.int64) & 0xFFFFFFFF                                  # uint32 order
    cut = torch.zeros((), dtype=torch.int64, device=dev)
    lo, hi = torch.zeros((), dtype=torch.int64, device=dev), torch.full((), 0xFFFFFFFF, dtype=torch.int64,
                                                                        device=dev)
    for _ in range(33):                                                     # the largest t with >= top keys >= t
        mid = (lo + hi + 1) // 2
        take = (key >= mid).sum() >= top
        lo = torch.where(take, mid, lo)
        hi = torch.where(take, hi, mid - 1)
    cut = lo
    above = key > cut
    equal = key == cut
    need = top - above.to(torch.int32).sum()
    rank = torch.cumsum(equal.to(torch.int32), dim=0)
    chosen = above | (equal & (rank <= need))
    place = torch.cumsum(chosen.to(torch.int32), dim=0) - 1
    for k in range(ratio):
        scratch.ids[r].index_put_((place.long() * ratio + k,), (b * ratio + k)[chosen], accumulate=False)
    t = torch.arange(ratio, dtype=torch.int32, device=dev)
    tail = ratio * complete + t
    keep = tail < end
    scratch.ids[r].index_put_((top * ratio + t[keep],), tail[keep], accumulate=False)
    scratch.nk[r] = top * ratio + end - ratio * complete
    scratch.sparse[r] = 1


def qsa_select(iq: torch.Tensor, ikc: torch.Tensor, pooled: torch.Tensor, pos0: torch.Tensor, ik_scale: torch.Tensor,
               inv_freq: torch.Tensor, eps: float, scratch: AttnScratch, rows: int, *,
               context: int | None = None, rope=None, delta=None, length=0, sections=(11, 11, 10)) -> None:
    """Pool the blocks the window completes, score and select each sparse row's blocks (scratch.ids/nk/sparse)."""

    qsa_pool(ikc, pooled, pos0, ik_scale, inv_freq, eps, scratch, rows,
             rope=rope, delta=delta, length=length, sections=sections)
    qsa_rows(iq, pooled, pos0, scratch, rows, context=context)
