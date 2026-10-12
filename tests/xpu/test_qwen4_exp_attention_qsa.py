"""The QSA kernels on CPU: the pooled block rotation and the row selection, against the
``cuda/attention.py`` triton kernels translated line by line (``_pool_block`` and ``_select``),
plus ``attn_prep``'s rope against ``cuda/glue.py``'s kernel and ``_cos_sin`` against torch.

These are the port's SPEC pins. The suite's usual oracle — the XPU CPU walk — calls these
same functions, so card-vs-CPU bit equality is structurally blind to a fault in shared code:
the pool's unindexed frequency table, the quadrant fold and the rotate-half form all passed
every gate until the first multi-chunk prefill executed them (the QSA scratch is dense while
``capacity <= budget``). A ported kernel needs at least one test against the CUDA source it
mirrors, never only against the port's own CPU path.
"""

import math

import torch

from tensorfold.families.qwen4_exp.xpu import attention as attn


def _pool_reference(ikc, w, inv, eps, ratio, b_index):
    """``_pool_block`` for block ``b_index``, translated from the triton source."""
    di, half = ikc.shape[1], inv.numel()
    d = torch.arange(di)
    partner = torch.where(d < half, d + half, torch.where(d < 2 * half, d - half, d))
    j = torch.where(d < half, d, torch.where(d < 2 * half, d - half, torch.zeros_like(d)))
    acc = ikc[ratio * b_index].to(torch.float32)
    for k in range(1, ratio):
        acc = acc + ikc[ratio * b_index + k].to(torch.float32)
    x = (acc / ratio).to(torch.bfloat16).to(torch.float32)
    rinv = 1.0 / torch.sqrt(torch.sum(x * x) / di + eps)
    xn = (x * rinv * w).to(torch.bfloat16).to(torch.float32)
    xp = ikc[ratio * b_index, partner].to(torch.float32)
    for k in range(1, ratio):
        xp = xp + ikc[ratio * b_index + k, partner].to(torch.float32)
    xp = (xp / ratio).to(torch.bfloat16).to(torch.float32)
    xpn = (xp * rinv * w[partner]).to(torch.bfloat16).to(torch.float32)
    ang = torch.full((di,), float(ratio * b_index)) * inv.to(torch.float32)[j]
    cos, sin = torch.cos(ang), torch.sin(ang)
    rot = torch.where(d < half, xn * cos - xpn * sin,
                      torch.where(d < 2 * half, xpn * sin + xn * cos, xn))
    return rot.to(torch.bfloat16)


def test_qsa_pool_rotates_each_completed_block():
    torch.manual_seed(7)
    di, ratio, rows, p0, eps = 8, 4, 12, 5, 1e-6  # the window straddles partial blocks at both ends
    half = di // 4
    ikc = (torch.randn(64, di) * 0.3).to(torch.bfloat16)
    w = torch.rand(di) + 0.5
    inv = torch.rand(half) + 0.25
    scratch = attn.AttnScratch(rows, 1, 4, 4096, torch.device("cpu"), budget=8, ratio=ratio)
    pooled = torch.zeros(4096 // ratio, di, dtype=torch.bfloat16)
    attn.qsa_pool(ikc, pooled, torch.tensor([p0]), w, inv, eps, scratch, rows)
    checked = 0
    for i in range(rows // ratio + 2):
        b = p0 // ratio + i
        if ratio * b + ratio > p0 + rows:
            continue
        want = _pool_reference(ikc, w, inv, eps, ratio, b)
        got = pooled[b].float()
        assert torch.allclose(got, want.float(), rtol=1e-2, atol=1e-2), (b, got, want)
        checked += 1
    assert checked >= 3
    # blocks the window does not complete keep their bits
    assert torch.equal(pooled[p0 // ratio + checked].float(), torch.zeros(di))
def test_cos_sin_quadrant_fold_matches_reference():
    ang = torch.tensor([0.0, 0.3, 1.6, 2.0, 3.0, 4.0, 4.4094, 5.5, 8.9, 10.7, -1.2, -3.9, 100.5])
    cos, sin = attn._cos_sin(ang.clone())
    assert torch.allclose(cos, torch.cos(ang), rtol=0, atol=1e-6), (ang, cos, torch.cos(ang))
    assert torch.allclose(sin, torch.sin(ang), rtol=0, atol=1e-6), (ang, sin, torch.sin(ang))

def _glue_rot(x_src, w, width, half, pos, inv, eps):
    """glue.py's per-head RMSNorm + rotate-half for one row, from the triton source."""
    d = torch.arange(width)
    partner = torch.where(d < half, d + half, torch.where(d < 2 * half, d - half, d))
    j = torch.where(d < half, d, torch.where(d < 2 * half, d - half, torch.zeros_like(d)))
    x = x_src.to(torch.float32)
    rinv = 1.0 / torch.sqrt(torch.sum(x * x) / width + eps)
    xn = (x * rinv * w.to(torch.float32)).to(torch.bfloat16).to(torch.float32)
    xp = x_src[partner].to(torch.float32)
    xpn = (xp * rinv * w.to(torch.float32)[partner]).to(torch.bfloat16).to(torch.float32)
    ang = torch.full((width,), float(pos)) * inv.to(torch.float32)[j]
    cos, sin = torch.cos(ang), torch.sin(ang)  # the kernel's tl.cos/tl.sin
    return torch.where(d < half, xn * cos - xpn * sin,
                       torch.where(d < 2 * half, xpn * sin + xn * cos, xn)).to(torch.bfloat16)


def test_attn_prep_matches_the_glue_kernel_reference():
    # head_dim 8 with half = 3: rotary over dims 0..5 (pos >= 3 drives the quadrants 2/3 the
    # port once got wrong), dims 6..7 unrotated; index_dim 8 folds the same head-width shape.
    torch.manual_seed(3)
    q_heads, kv_heads, head_dim, index_heads, index_dim, rows, p0, eps = 2, 1, 8, 1, 8, 6, 0, 1e-6
    inv = torch.tensor([1.0, 0.7, 0.4])
    half = inv.numel()
    p = (torch.randn(rows, q_heads * 2 * head_dim + kv_heads * 2 * head_dim
                     + index_heads * 2 * index_dim) * 0.5).to(torch.bfloat16)
    qs = torch.rand(head_dim) + 0.5
    ks_w = torch.rand(head_dim) + 0.5
    is_w = torch.rand(index_dim) + 0.5
    q = torch.zeros(rows, q_heads, head_dim, dtype=torch.bfloat16)
    kc = torch.zeros(rows * kv_heads, head_dim, dtype=torch.bfloat16)
    vc = torch.zeros_like(kc)
    iq = torch.zeros(rows, index_heads, index_dim, dtype=torch.bfloat16)
    ikc = torch.zeros(rows * 2, index_dim, dtype=torch.bfloat16)
    attn.attn_prep(p, torch.tensor([p0]), qs, ks_w, is_w, inv, q, kc, vc, iq, ikc, eps,
                   q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim,
                   index_heads=index_heads, index_dim=index_dim)
    for r in range(rows):
        pos = p0 + r
        for h in range(q_heads):
            src = p[r, h * 2 * head_dim:h * 2 * head_dim + head_dim]
            want = _glue_rot(src, qs, head_dim, half, pos, inv, eps)
            assert torch.allclose(q[r, h].float(), want.float(), rtol=1e-2, atol=1e-2), ("q", r, h)
        for h in range(kv_heads):
            off = q_heads * 2 * head_dim
            src = p[r, off + h * head_dim:off + (h + 1) * head_dim]
            want = _glue_rot(src, ks_w, head_dim, half, pos, inv, eps)
            slot = pos * kv_heads + h
            assert torch.allclose(kc[slot].float(), want.float(), rtol=1e-2, atol=1e-2), ("k", r, h)
            v = p[r, off + kv_heads * head_dim + h * head_dim:off + (kv_heads + 1) * head_dim]
            assert torch.equal(vc[slot], v), r
        base = q_heads * 2 * head_dim + 2 * kv_heads * head_dim
        for h in range(index_heads):
            src = p[r, base + h * index_dim:base + (h + 1) * index_dim]
            want = _glue_rot(src, is_w, index_dim, half, pos, inv, eps)
            assert torch.allclose(iq[r, h].float(), want.float(), rtol=1e-2, atol=1e-2), ("ik", r, h)
        assert torch.equal(ikc[p0 + r], p[r, base + index_heads * index_dim
                                          :base + 2 * index_heads * index_dim])





def test_qsa_rows_dense_below_the_budget_and_topk_above():
    torch.manual_seed(11)
    di, ratio = 8, 4
    budget, rows, p0 = 16, 3, 58  # ends 59, 60, 61: complete 14, 15, 15 blocks - all sparse at top=4
    top = budget // ratio
    scratch = attn.AttnScratch(rows, 1, 4, 64, torch.device("cpu"), budget=budget, ratio=ratio)
    assert scratch.qsa and scratch.nb == 64 // ratio
    iq = (torch.randn(rows, 1, di) * 0.2).to(torch.bfloat16)
    pooled = (torch.randn(scratch.nb, di) * 0.3).to(torch.bfloat16)
    pooled[3] = pooled[9]  # an exact tie between blocks 3 and 9
    attn.qsa_rows(iq, pooled, torch.tensor([p0]), scratch, rows)
    assert torch.equal(scratch.sparse, torch.ones(rows, dtype=torch.int32))
    f32 = torch.float32
    for r in range(rows):
        end = p0 + r + 1
        complete = end // ratio
        scores = []
        for b in range(complete):
            k = pooled[b].to(f32)
            scores.append(max(float((k * iq[r, 0].to(f32)).sum()), 0.0) / math.sqrt(di))
        order = sorted(range(complete), key=lambda b: (-scores[b], b))[:top]
        want_ids = [b * ratio + k for b in sorted(order) for k in range(ratio)]
        tail = [ratio * complete + t for t in range(ratio) if ratio * complete + t < end]
        assert scratch.ids[r, :len(want_ids)].tolist() == want_ids, (r, scratch.ids[r], order)
        assert scratch.ids[r, top * ratio:top * ratio + len(tail)].tolist() == tail
        assert int(scratch.nk[r]) == top * ratio + end - ratio * complete
    # dense rows: fewer complete blocks than the budget allows
    dense = attn.AttnScratch(2, 1, 4, 64, torch.device("cpu"), budget=1024, ratio=ratio)
    attn.qsa_rows(iq[:2], pooled, torch.tensor([p0 - 48]), dense, 2)  # ends 11, 12: complete 2, 3 <= top
    assert torch.equal(dense.sparse, torch.zeros(2, dtype=torch.int32))
    assert dense.nk.tolist() == [11, 12]
