"""Slice 10's CPU lane: the fused grouped GEMM against the rank-one oracle without a card.

Same judgement as the card gates — ``mm_fused``/``grouped_moe(fused=True)`` inside the ``tests/cuda``
NVFP4 tolerances of ``mm_ordered``/``grouped_moe(fused=False)`` (pair relative max 2e-2, fp32 product
1e-4) — run entirely on CPU tensors, so the semantics of the fused lane are checkable while the card
is busy. The card run adds the fp32-matmul-precision confirmation and the wall-time gates.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

os.environ.setdefault("TRITON_DEFAULT_BACKEND", "intel")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cuda"))

from tensorfold.families.qwen4_exp.xpu import moe as xpu_moe  # noqa: E402
from tensorfold.families.qwen4_exp.xpu import nvfp4 as xpu  # noqa: E402

REL_DOWN = 1e-4                      # tests/cuda/test_nvfp4_experts.py: the fp32 down product's gate
REL_FACE = 2e-2                      # the same file's gate/up (swiglu'd) pair gate
D, NI, SW, E, TOPK = 2560, 640, 640, 8, 4


def _tables(seed: int) -> xpu.Stacked4:
    g = torch.Generator().manual_seed(seed)

    def t(*shape):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.02).to(torch.bfloat16).to(torch.float32)
    return xpu.Stacked4(t(E, D, NI), t(E, D, NI), t(E, NI, D), t(D, SW), t(D, SW), t(SW, D))


def _picks(rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, D, generator=g, dtype=torch.float32).to(torch.bfloat16)
    pick = torch.zeros((rows, TOPK + 1), dtype=torch.int32)
    for r in range(rows):
        pick[r, :TOPK] = torch.randperm(E, generator=g)[:TOPK].to(torch.int32)
        pick[r, TOPK] = E                                    # the shared expert's slot
    return x, pick


# -- the faces: one fp32 matmul inside 1e-4 of the k-ordered sum ------------------------------------------------


@pytest.mark.parametrize("k,n", [(D, NI), (NI, D), (D, SW), (SW, D)])   # gate/up, down, shared faces
def test_cpu_face_matches_the_rank_one_oracle(k, n):
    g = torch.Generator().manual_seed(5)
    x = torch.randn(7, k, generator=g, dtype=torch.float32).to(torch.bfloat16)
    w = (torch.randn(k, n, generator=g, dtype=torch.float32) * 0.02)
    got, oracle = xpu.mm_fused(x, w), xpu.mm_ordered(x, w)
    rel = float((got - oracle).abs().max() / oracle.abs().max())
    assert rel < REL_DOWN, f"[{k}, {n}] fused vs rank-one: relative max {rel}"
    assert torch.allclose(got, oracle, rtol=REL_FACE, atol=REL_FACE)


# -- the grouped path: every pair at the CUDA pair gate, faces under equal inputs at 1e-4 ---------------------


def test_cpu_grouped_fused_within_the_cuda_gates():
    ex = _tables(13)
    # The fp32 product's 1e-4 gate belongs to the faces under equal inputs (test_cpu_face above: a
    # crossing of a bf16 rounding boundary in gate/up can flip act by one ulp, which the CUDA down
    # test excludes by feeding both sides the same act). The composed path rides the pair gate.
    for rows, prefill in ((1, False), (5, False), (5, True)):
        x, pick = _picks(rows, 17)
        oracle = xpu.grouped_moe(x, ex, pick, fused=False)
        got = xpu.grouped_moe(x, ex, pick, prefill=prefill)
        assert got.dtype is (torch.bfloat16 if prefill else torch.float32)
        rel = max(float((got[r, s].float() - oracle[r, s]).abs().max() / oracle[r, s].abs().max())
                  for r in range(rows) for s in range(TOPK + 1))
        assert rel < REL_FACE, f"rows={rows} prefill={prefill}: fused vs oracle relative max {rel}"
        assert torch.allclose(got.float(), oracle, rtol=REL_FACE, atol=REL_FACE)


def test_cpu_layer_argmax_stable_and_within_the_pair_gate():
    ex = _tables(23)
    g = torch.Generator().manual_seed(23)
    router_rows = (torch.randn(E + 1, D, generator=g, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    x, _ = _picks(4, 29)
    m = xpu_moe.Routed(router=router_rows, experts=ex, top_k=TOPK)
    fused, oracle = xpu_moe.run(x, m, fused=True), xpu_moe.run(x, m, fused=False)
    assert torch.equal(fused.argmax(dim=-1), oracle.argmax(dim=-1)), "a row's argmax moved"
    rel = float((fused.float() - oracle.float()).abs().max() / oracle.float().abs().max())
    assert rel < REL_FACE, f"layer output relative max {rel}"


def test_cpu_oracle_is_the_reference_it_claims_to_be():
    """The oracle's own invariants, cheap enough to pin here: rows and row-count never move its bits."""

    ex = _tables(29)
    x, pick = _picks(3, 41)
    batched = xpu.grouped_moe(x, ex, pick, fused=False)
    for r in range(3):
        one = xpu.grouped_moe(x[r:r + 1], ex, pick[r:r + 1], fused=False)
        assert torch.equal(batched[r], one[0]), f"oracle row {r} moved with the batch"
