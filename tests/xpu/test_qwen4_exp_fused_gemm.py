"""Slice 10's gate: the fused grouped NVFP4 GEMM (phase 2, throughput) on Intel GPUs (B70 #2).

``nvfp4.mm_fused`` replaces the expert faces' rank-one ``k``-loop with one fp32 card matmul per face, so
the bits are the card's reduction order, not the oracle's: the fused path is judged against the rank-one
oracle (``mm_ordered``, ``grouped_moe(fused=False)``) with the tolerances ``tests/cuda`` holds an NVFP4
GEMM — ``test_nvfp4_experts.py``'s per-pair relative max 2e-2 for the swiglu'd gate/up faces and 1e-4 for
the down face's fp32 product. The two-layer decode-step wall-time gate (>= 10x) runs under
``TENSORFOLD_XPU_SPEED_GATE=1``; the rest is cheap enough to always run on the card.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
import torch

os.environ.setdefault("TRITON_CACHE_DIR", "/storage01/.triton-cache-fused")   # the root fs is full
os.environ.setdefault("TRITON_DEFAULT_BACKEND", "intel")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cuda"))

from tensorfold.families.qwen4_exp.xpu import moe as xpu_moe  # noqa: E402
from tensorfold.families.qwen4_exp.xpu import nvfp4 as xpu  # noqa: E402

REL_DOWN = 1e-4                      # tests/cuda/test_nvfp4_experts.py: the fp32 down product's gate
REL_FACE = 2e-2                      # the same file's gate/up (swiglu'd) pair gate; allclose's atol too
D, NI, SW, E, TOPK = 2560, 640, 640, 8, 4     # the checkpoint's face widths, a fixture-sized expert count
CFG = type("C", (), {"num_experts": E, "num_experts_per_tok": TOPK, "hidden_size": D,
                     "moe_intermediate_size": NI, "shared_expert_intermediate_size": SW})
REAL = dict(E=512, D=2560, NI=640, SW=640, TOPK=10, LAYERS=2)   # the 2-layer decode-step gate's shape


@pytest.fixture(scope="module")
def dev() -> torch.device:
    """B70 #2: the usable device whose uuid hex contains 4400 (the selector leaves it at index 0)."""

    for i in range(torch.xpu.device_count()):
        try:
            uuid = bytes(torch.xpu.get_device_properties(i).uuid.bytes).hex()
        except Exception:  # noqa: BLE001
            continue
        if "4400" in uuid:
            return torch.device("xpu", i)
    pytest.skip("no usable XPU device with uuid containing 4400")


def _mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def _tables(e: int, d: int, ni: int, sw: int, g: torch.Generator) -> xpu.Stacked4:
    """A synthetic layer's fp32 tables at the checkpoint's face widths, the dequant's magnitudes."""

    def t(*shape):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.02).to(torch.bfloat16).to(torch.float32)
    return xpu.Stacked4(t(e, d, ni), t(e, d, ni), t(e, ni, d), t(d, sw), t(d, sw), t(sw, d))


def _moved_tables(e, d, ni, sw, g, device) -> xpu.Stacked4:
    ex = _tables(e, d, ni, sw, g)
    return xpu.Stacked4(*[w.to(device) for w in (ex.gate_t, ex.up_t, ex.down_t,
                                                 ex.shared_gate_t, ex.shared_up_t, ex.shared_down_t)])


def _picks(rows: int, e: int, topk: int, seed: int, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, D, generator=g, dtype=torch.float32).to(torch.bfloat16)
    pick = torch.zeros((rows, topk + 1), dtype=torch.int32)
    for r in range(rows):
        ids = torch.randperm(e, generator=g)[:topk]
        pick[r, :topk] = ids.to(torch.int32)
        pick[r, topk] = e                                    # the shared expert's slot
    return x.to(device), pick.to(device)


def _worst_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """The CUDA pair gate's own metric: worst over the pairs of max|a-b| / max|b|."""

    worst = 0.0
    for r in range(a.shape[0]):
        for s in range(a.shape[1]):
            worst = max(worst, float((a[r, s] - b[r, s]).abs().max() / b[r, s].abs().max().clamp(min=1e-12)))
    return worst


# -- gate 1: the fused face against the rank-one oracle ---------------------------------------------------------


@pytest.mark.parametrize("k,n", [(D, NI), (NI, D), (D, SW), (SW, D)])   # gate/up, down, shared faces
def test_mm_fused_is_fp32_reduction_noise_around_the_oracle(dev, k, n):
    """One fp32 matmul's bits sit inside ``REL_DOWN`` of the k-ordered sum — the down face's CUDA gate,
    which a tf32/bf16 input lane could not hold — and ``torch.allclose`` holds both absolutely."""

    g = torch.Generator().manual_seed(5)
    x = torch.randn(7, k, generator=g, dtype=torch.float32).to(torch.bfloat16).to(dev)
    w = (torch.randn(k, n, generator=g, dtype=torch.float32) * 0.02).to(dev)
    got = xpu.mm_fused(x, w).cpu()
    oracle = xpu.mm_ordered(x.to("cpu"), w.to("cpu"))
    rel = float((got - oracle).abs().max() / oracle.abs().max())
    assert rel < REL_DOWN, f"[{k}, {n}] fused vs rank-one: relative max {rel}"
    assert torch.allclose(got, oracle, rtol=REL_FACE, atol=REL_FACE)


def test_grouped_moe_fused_pair_gate_decode_and_prefill(dev):
    """Every pair's fused slot — fp32 decode and bf16 prefill alike — is within the CUDA pair gate of
    the rank-one oracle's slot, and segments of different experts never share an accumulator."""

    g = torch.Generator().manual_seed(13)
    ex = _moved_tables(E, D, NI, SW, g, dev)
    for rows, prefill in ((1, False), (5, False), (5, True)):
        x, pick = _picks(rows, E, TOPK, 17)
        oracle = xpu.grouped_moe(x.to(dev), ex, pick.to(dev), fused=False).cpu()
        got = xpu.grouped_moe(x.to(dev), ex, pick.to(dev), prefill=prefill).cpu()
        assert got.dtype is (torch.bfloat16 if prefill else torch.float32)
        rel = _worst_rel(got.float(), oracle)
        assert rel < REL_FACE, f"rows={rows} prefill={prefill}: fused vs oracle relative max {rel}"
        assert torch.allclose(got.float(), oracle, rtol=REL_FACE, atol=REL_FACE)


# -- gate 2: argmax stability on a fixture decode step ----------------------------------------------------------


def test_argmax_stable_through_the_moe_layer(dev):
    """A whole MoE layer (router, select, fused experts, combine): every row's output argmax — the
    feature the next op reads — is the oracle's argmax, and the rows' bits stay inside the pair gate."""

    g = torch.Generator().manual_seed(23)
    ex = _moved_tables(E, D, NI, SW, g, dev)
    router_rows = (torch.randn(E + 1, D, generator=g, dtype=torch.float32) * 0.02).to(torch.bfloat16).to(dev)
    x, _ = _picks(8, E, TOPK, 29)
    m = xpu_moe.Routed(router=router_rows, experts=ex, top_k=TOPK)
    fused = xpu_moe.run(x.to(dev), m, fused=True).cpu()
    oracle = xpu_moe.run(x.to(dev), m, fused=False).cpu()
    assert torch.equal(fused.argmax(dim=-1), oracle.argmax(dim=-1)), "a row's argmax moved"
    assert float((fused.float() - oracle.float()).abs().max() / oracle.float().abs().max()) < REL_FACE


# -- gate 3: the fused path collapses the launch storm ----------------------------------------------------------


def _timed(fn, dev, *, repeats: int = 3) -> float:
    fn()                                                    # warm the kernels and the allocator
    torch.xpu.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.xpu.synchronize(dev)
    return (time.perf_counter() - t0) / repeats


def test_fused_decode_step_beats_the_k_loop(dev):
    """Fixture-shape decode row: the oracle pays ``K`` launches per face, the fused lane one — the wall
    ratio is launch-dominated, and a regression to the k-loop (or a fused lane that lost its way) trips."""

    g = torch.Generator().manual_seed(31)
    d, ni, e, topk = 512, 128, 8, 3
    ex = _moved_tables(e, d, ni, ni, g, dev)
    x = torch.randn(1, d, generator=g, dtype=torch.float32).to(torch.bfloat16).to(dev)
    pick = torch.zeros((1, topk + 1), dtype=torch.int32)
    pick[0, :topk] = torch.tensor([0, 1, 2], dtype=torch.int32)
    pick[0, topk] = e
    slow = _timed(lambda: xpu.grouped_moe(x, ex, pick.to(dev), fused=False), dev, repeats=1)
    fast = _timed(lambda: xpu.grouped_moe(x, ex, pick.to(dev)), dev)
    assert slow / fast >= 6.0, f"fused {fast:.4f} s vs rank-one {slow:.4f} s: launch storm still standing"


def test_two_layer_decode_step_speed_gate(dev):
    """The slice's gate: a 2-layer decode step (the checkpoint's E/topk/widths, ``nvfp4.moe`` per layer)
    drops >= 10x fused vs rank-one. Runs only under ``TENSORFOLD_XPU_SPEED_GATE=1`` — it holds ~20 GB of
    tables; the orchestrator runs it alone, on an idle card, and greps the printed metric lines."""

    if os.environ.get("TENSORFOLD_XPU_SPEED_GATE") != "1":
        pytest.skip("set TENSORFOLD_XPU_SPEED_GATE=1 to run the 2-layer wall-time gate")
    if _mem_available_gb() < 24.0:
        pytest.skip(f"MemAvailable {_mem_available_gb():.1f} GB < 24 GB")
    e, d, ni, sw, topk, layers = REAL["E"], REAL["D"], REAL["NI"], REAL["SW"], REAL["TOPK"], REAL["LAYERS"]
    g = torch.Generator().manual_seed(31)
    ex = _moved_tables(e, d, ni, sw, g, dev)
    router_rows = (torch.randn(e + 1, d, generator=g, dtype=torch.float32) * 0.02).to(torch.bfloat16).to(dev)
    cfg = type("C", (), {"num_experts_per_tok": topk, "num_experts": e,
                         "moe_intermediate_size": ni, "hidden_size": d})
    buf = xpu_moe.MoEBuffers(1, cfg, dev)
    x = torch.randn(1, d, generator=g, dtype=torch.float32).to(torch.bfloat16).to(dev)

    def step(fused: bool, reps: int = 3) -> float:
        def fn():
            for _ in range(layers):
                xpu.moe(x, None, router_rows, ex, buf, cfg, fused=fused)
        return _timed(fn, dev, repeats=reps)

    fast = step(True)
    slow = step(False, reps=2)                                # the oracle is seconds per step
    out = xpu.grouped_moe(x, ex, buf.pick[:1]).cpu()          # buf.pick: both lanes route the same row
    oracle = xpu.grouped_moe(x, ex, buf.pick[:1], fused=False).cpu()
    rel = _worst_rel(out, oracle)
    am = bool(torch.equal(out.argmax(-1), oracle.argmax(-1)))
    peak = torch.xpu.max_memory_allocated(dev) / 2**30
    print(f"FUSED_MOE seconds_per_step oracle={slow / layers:.3f} fused={fast / layers:.4f} "
          f"speedup={slow / fast:.1f} two_layer_decode oracle={slow:.3f}s fused={fast:.4f}s")
    print(f"FUSED_MOE quality max_rel={rel:.2e} argmax_match={am} peak_vram_gb={peak:.2f}")
    assert rel < REL_FACE
    assert am, "the real-width decode step's argmax moved"
    assert slow / fast >= 10.0, f"2-layer decode: fused {fast:.3f} s vs oracle {slow:.3f} s, want >= 10x"
