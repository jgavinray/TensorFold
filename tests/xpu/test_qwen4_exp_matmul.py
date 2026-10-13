"""Slice 4's gate: the NVFP4 dequant, MoE routing and the grouped expert matmul on Intel GPUs.

The reference is the CUDA side's own bits: ``cuda/nvfp4.dequantize`` is the decoder, and every xpu
function is its op-for-op twin runnable on either side — so the device results must be
``torch.equal`` to a CPU run of the same code, and within the tests/cuda fp16-vs-fp64 relative-norm
tolerance of an independent naive fp32 matmul (whose accumulation order the ordered path cannot match).
The grouped expert matmul's fused lane (slice 10) is judged against the rank-one oracle instead:
its bits are fp32 reduction noise around the oracle's, held to the tests/cuda NVFP4 pair gate.
Fixture dims, plus one real-checkpoint expert stack (layer 0, experts 0-7) moved to the card."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cuda"))

from tensorfold.families.qwen4_exp.cuda import nvfp4 as ref  # noqa: E402
from tensorfold.families.qwen4_exp.xpu import moe as xpu_moe  # noqa: E402
from tensorfold.families.qwen4_exp.xpu import nvfp4 as xpu  # noqa: E402

FIXTURE = dict(layers=2, experts=4, vocab=64, hidden=128, heads=2, kv_heads=2, hd=32, nk=2, nv=2,
               dk=32, dv=32, moe_width=64, shared_width=96, streams=1, low=128, ple=False, mtp=False)
E, TOPK, HID, NI, SW = 4, 2, 128, 64, 96
CFG = type("C", (), {"num_experts": E, "num_experts_per_tok": TOPK, "hidden_size": HID})
CKPT = "/storage01/models/Qwen3.8-Flash-Next-NVFP4"
REL = 1e-2                       # the tests/cuda fp16-vs-fp64 gate
BUFS = torch.bfloat16            # the CUDA prefill's slot dtype, beside the decode fp32
REL_FACE = 2e-2                  # tests/cuda/test_nvfp4_experts.py: pair gate, the fused lane's tolerance


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


@pytest.fixture(scope="module")
def ck(tmp_path_factory) -> dict:
    """One tiny modelopt checkpoint's layer 0, as the loader's raw expert dict + the router rows."""

    import nvfp4_tiny
    import safetensors.torch

    root = nvfp4_tiny.write(tmp_path_factory.mktemp("matmul") / "tiny", **FIXTURE)
    st = safetensors.torch.load_file(str(root / "model-00001-of-00001.safetensors"))
    b = "model.layers.0"

    def triple(proj: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = f"{b}.mlp.experts.{{e}}.{proj}"
        return (torch.stack([st[p.format(e=e) + ".weight"] for e in range(E)]),
                torch.stack([st[p.format(e=e) + ".weight_scale"] for e in range(E)]),
                torch.tensor([float(st[p.format(e=e) + ".weight_scale_2"]) for e in range(E)]))

    shared = tuple(st[f"{b}.mlp.shared_expert.{p}.weight"] for p in ("gate_proj", "up_proj", "down_proj"))
    assert all(t.dtype is torch.bfloat16 for t in shared)
    router = torch.cat([st[f"{b}.mlp.gate.weight"].to(torch.bfloat16),
                        st[f"{b}.mlp.shared_expert_gate.weight"].to(torch.bfloat16)])
    return dict(gate=triple("gate_proj"), up=triple("up_proj"), down=triple("down_proj"),
                shared=shared, router=router,
                config=dict(num_experts=E, num_experts_per_tok=TOPK, hidden_size=HID, moe_intermediate_size=NI,
                            shared_expert_intermediate_size=SW))


def _rows(g: torch.Generator, *shape, dtype=torch.float32) -> torch.Tensor:
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(dtype)


def _naive_row(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """One row's expert fp32 bits, the naive (matmul-order) way: the tolerance reference."""

    g = (x @ gate.T).to(torch.bfloat16).float()
    u = (x @ up.T).to(torch.bfloat16).float()
    act = (g * torch.sigmoid(g)).to(torch.bfloat16).float() * u
    return act.to(torch.bfloat16).float() @ down.T


# -- gate 1: the device dequant --------------------------------------------------------------------------------


def test_dequantize_matches_cpu_reference(dev, ck):
    for name in ("gate", "up", "down"):
        words, scales, s2 = ck[name]
        assert words.dtype is torch.uint8 and scales.dtype is torch.float8_e4m3fn
        for e in range(E):                                          # the reference 2D path, on card
            cpu = ref.dequantize(words[e], scales[e], float(s2[e]))
            got = xpu.dequantize(words[e].to(dev), scales[e].to(dev), float(s2[e])).cpu()
            assert torch.equal(cpu, got), f"{name} expert {e}: card decode != cuda/nvfp4.dequantize"
        cpu = xpu.dequantize(words, scales, s2)                     # the stacked table, both sides
        got = xpu.dequantize(words.to(dev), scales.to(dev), s2.to(dev)).cpu()
        assert torch.equal(cpu, got), f"{name}: stacked card decode != the CPU decode"
        assert torch.equal(cpu, torch.stack([ref.dequantize(words[e], scales[e], float(s2[e]))
                                             for e in range(E)])), f"{name}: stacked != per-expert reference"


# -- gate 2: the router ---------------------------------------------------------------------------------------


def _logits() -> torch.Tensor:
    g = torch.Generator().manual_seed(3)
    lg = _rows(g, 17, E + 1) * 3.0
    lg[:, 0] = lg[:, 1]                        # duplicate columns: every row's max ties between 0 and 1
    lg[9, 0] = lg[9, 1] + 4.0                  # a row whose max is unique
    lg[3, 4] = float("-inf")
    lg[7, :] = 0.0                             # an all-equal row: lowest ids must win
    return lg


def test_router_matches_cpu_reference(dev):
    lg = _logits()
    bc, bd = xpu_moe.MoEBuffers(17, CFG, "cpu"), xpu_moe.MoEBuffers(17, CFG, dev)
    xpu_moe.select_rows(lg.clone(), bc, TOPK, E)
    xpu_moe.select_rows(lg.clone().to(dev), bd, TOPK, E)
    assert torch.equal(bc.pick, bd.pick.cpu()), "expert ids differ card vs cpu"
    assert torch.equal(bc.wts, bd.wts.cpu()), "expert weights differ card vs cpu"
    for r in range(17):                                          # an independent CPU reference
        want = []
        v = lg[r, :E].clone()
        base = v.max()
        total = 0.0
        for k in range(TOPK):
            m = v.max()
            i = int((v == m).nonzero()[0])                       # ties to the lower id
            want.append(i)
            total += float(torch.exp(m - base))
            v[i] = float("-inf")
        assert bc.pick[r, :TOPK].tolist() == want, f"row {r}: ids {bc.pick[r, :TOPK].tolist()} != {want}"
        for k, i in enumerate(want):
            w = float(torch.exp(lg[r, i] - base)) / total
            assert abs(float(bc.wts[r, k]) - w) <= 8e-3 * max(abs(w), 1e-6), f"row {r} slot {k}: weight"
        sg = float(lg[r, E].to(torch.bfloat16))
        assert abs(float(bc.wts[r, TOPK]) - float(torch.sigmoid(torch.tensor(sg)).to(torch.bfloat16))) <= 8e-3


# -- gate 3: the grouped expert matmul --------------------------------------------------------------------------


def _picked(rows: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = _rows(g, rows, HID, dtype=torch.bfloat16)
    pick = torch.zeros((rows, TOPK + 1), dtype=torch.int32)
    for r in range(rows):
        pick[r, 0], pick[r, 1] = r % E, (r * 3 + 1) % E
        pick[r, TOPK] = E                                        # the shared expert's slot
    return x, pick


def _check_grouped(dev, ck, x, pick):
    ex_c = xpu.stacked4_from_experts(ck, "cpu")
    ex_d = xpu.stacked4_from_experts(ck, dev)
    yo = xpu.grouped_moe(x, ex_c, pick, fused=False)                      # the rank-one oracle (phase 2's reference)
    assert torch.equal(xpu.grouped_moe(x.to(dev), ex_d, pick.to(dev), fused=False).cpu(), yo), \
        "rank-one grouped slots differ card vs cpu"
    yd = xpu.grouped_moe(x.to(dev), ex_d, pick.to(dev)).cpu()              # the fused lane: the engine's default
    bc = xpu.grouped_moe(x, ex_c, pick, prefill=True, fused=False)
    bd = xpu.grouped_moe(x.to(dev), ex_d, pick.to(dev), prefill=True).cpu()
    assert torch.allclose(bd.float(), bc.float(), rtol=REL_FACE, atol=REL_FACE), "prefill fused slots off the oracle"
    for r in range(x.shape[0]):
        for s in range(TOPK + 1):                                          # the CUDA pair gate, per tensor
            rel = float((yd[r, s] - yo[r, s]).abs().max() / yo[r, s].abs().max())
            assert rel < REL_FACE and torch.allclose(yd[r, s], yo[r, s], rtol=REL_FACE, atol=REL_FACE), (r, s, rel)
    assert torch.equal(bd.float().to(torch.bfloat16).float(), bd), "prefill slots are not bf16 values"
    gf, uf, df = ck["gate"], ck["up"], ck["down"]
    g32, u32, d32 = (xpu.dequantize(*gf), xpu.dequantize(*uf), xpu.dequantize(*df))
    xs = x.float()
    worst = 0.0
    for r in range(x.shape[0]):
        for s in range(TOPK + 1):
            e = int(pick[r, s])
            if e < E:
                y = _naive_row(xs[r], g32[e], u32[e], d32[e])
            else:
                sh = [t.float() for t in ck["shared"]]
                y = _naive_row(xs[r], sh[0], sh[1], sh[2])
            worst = max(worst, float((yd[r, s] - y).norm() / y.norm()))
    assert worst < REL, f"grouped y vs the naive fp32 reference: worst relative norm {worst}"
    out = xpu_moe.combine(yd, bc_wts := _wts(rows=x.shape[0], pick=pick))
    assert out.dtype is torch.bfloat16
    return yd


def _wts(rows: int, pick: torch.Tensor) -> torch.Tensor:     # any fixed weights: combine's bits gate on them
    w = torch.full((rows, TOPK + 1), 1.0 / (TOPK + 1), dtype=torch.float32)
    return w


def test_grouped_matmul_fixture(dev, ck):
    _check_grouped(dev, ck, *_picked(5, 2))


# -- gate 3b: one real expert stack (layer 0, experts 0-7, 128 rows) --------------------------------------------


def _real_layer0():
    import json

    from safetensors import safe_open

    free = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) // 1024
    if free < 24:
        pytest.skip(f"only {free} GB available")
    wm = json.load(open(CKPT + "/model.safetensors.index.json"))["weight_map"]
    base = "model.language_model.layers.0.mlp."
    out = {}
    for proj in ("gate", "up", "down"):
        wds, scls, s2 = [], [], []
        for e in range(8):
            key = f"{base}experts.{e}.{proj}_proj"
            with safe_open(CKPT + "/" + wm[key + ".weight"], framework="torch") as sf:
                wds.append(sf.get_tensor(key + ".weight"))
                scls.append(sf.get_tensor(key + ".weight_scale"))
                s2.append(float(sf.get_tensor(key + ".weight_scale_2")))
        out[proj] = (torch.stack(wds), torch.stack(scls), torch.tensor(s2, dtype=torch.float32))
    sh = []
    for proj in ("gate", "up", "down"):
        with safe_open(CKPT + "/" + wm[base + f"shared_expert.{proj}_proj.weight"], framework="torch") as sf:
            sh.append(sf.get_tensor(base + f"shared_expert.{proj}_proj.weight"))
    out["shared"] = tuple(sh)
    return out


def test_grouped_matmul_real_slice(dev):
    real = _real_layer0()
    dims = real["gate"][0].shape[-1] * 2
    g = torch.Generator().manual_seed(11)
    x = _rows(g, 128, dims, dtype=torch.bfloat16)
    pick = torch.zeros((128, TOPK + 1), dtype=torch.int32)
    for r in range(128):
        pick[r, 0], pick[r, 1], pick[r, TOPK] = r % 8, (r * 3 + 5) % 8, 8
    ex_c = xpu.stacked4_from_experts(real, "cpu")
    ex_d = xpu.stacked4_from_experts(real, dev)
    got = xpu.dequantize(*real["gate"]).cpu()
    assert torch.equal(got, xpu.dequantize(real["gate"][0].to(dev), real["gate"][1].to(dev),
                                           real["gate"][2].to(dev)).cpu()), "real table: card decode != cpu"
    yo = xpu.grouped_moe(x, ex_c, pick, fused=False)                      # rank-one oracle, CPU
    assert torch.equal(xpu.grouped_moe(x.to(dev), ex_d, pick.to(dev), fused=False).cpu(), yo), \
        "real slots: rank-one oracle differs card vs cpu"
    yd = xpu.grouped_moe(x.to(dev), ex_d, pick.to(dev)).cpu()              # the fused lane
    rel = max(float((yd[r, s] - yo[r, s]).abs().max() / yo[r, s].abs().max())
              for r in range(0, 128, 16) for s in range(TOPK + 1))
    assert rel < REL_FACE, f"real expert stack fused vs oracle: worst relative max {rel}"
    g32, u32, d32 = (xpu.dequantize(*real["gate"]), xpu.dequantize(*real["up"]), xpu.dequantize(*real["down"]))
    xs = x.float()
    worst = 0.0
    for r in range(0, 128, 8):                                    # sixteen rows against the naive reference
        for s in range(TOPK + 1):
            e = int(pick[r, s])
            sh = [t.float() for t in real["shared"]]
            y = _naive_row(xs[r], *[t[e] if e < 8 else t for t in (g32, u32, d32)] if e < 8 else sh)
            worst = max(worst, float((yd[r, s] - y).norm() / y.norm()))
    assert worst < REL, f"real expert stack vs the naive fp32 reference: worst relative norm {worst}"


# -- gate 4: the Q4 and BF16 Triton kernels on the card ----------------------------------------------------------


def _quant_group64(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MLX affine 4-bit words for 64-value groups: (n, k/8) int32 words, (n, k/64) bf16 scales/biases."""

    n, k = w.shape
    g = w.float().reshape(n, k // 64, 64)
    mn, mx = g.min(-1, keepdim=True).values, g.max(-1, keepdim=True).values
    s = ((mx - mn) / 15.0).clamp(min=1e-8)
    q = ((g - mn) / s).round().clamp(0, 15).to(torch.uint8).reshape(n, k)
    return (q[:, 1::2] << 4 | q[:, 0::2]).view(torch.uint32), s.reshape(n, k // 64).to(torch.bfloat16), \
        mn.reshape(n, k // 64).to(torch.bfloat16)


def test_q4_b16_smoke(dev):
    from tensorfold.cuda.kernels import qmm as lane
    from tensorfold.families.qwen3_5.xpu.qmm import dequantize as lane_dequantize
    from tensorfold.families.qwen4_exp.cuda import bf16 as bf16_ref

    n, k = 128, 1024
    w = _rows(torch.Generator().manual_seed(5), n, k).to(torch.bfloat16) * 0.05
    words, scales, biases = _quant_group64(w)
    kq = _rows(torch.Generator().manual_seed(6), 3, k, dtype=torch.bfloat16)
    qd = lane.pack(words.to(dev), scales.to(dev), biases.to(dev), 64)
    got = lane.matmul(kq.to(dev), qd, lane.group_sums(kq.to(dev), 64), sk=1, f32=False).cpu().float()
    # the lane matmul is bf16-out; f32 is CUDA-only
    want = (kq.float() @ lane_dequantize(words, scales, biases, 4, 64).float().T).to(torch.bfloat16).float()
    rel = float((got - want).norm() / want.norm())
    assert rel < REL, f"Q4 lane kernel vs the CPU dequantize reference: relative norm {rel}"
    onb = bf16_ref.matmul(kq.to(dev), bf16_ref.B16(w.to(dev), n, k), f32=True).cpu()
    nb = kq.float() @ w.float().T
    relb = float((onb - nb).norm() / nb.norm())
    assert relb < REL, f"BF16 kernel vs the CPU reference: relative norm {relb}"
    print(f"q4/b16 card smoke: Q4 rel {rel:.3e}, B16 rel {relb:.3e}")


# -- gate 5: rows one at a time match the batched run -------------------------------------------------------------


def test_row_invariance(dev, ck):
    g = torch.Generator().manual_seed(8)
    x = _rows(g, 9, HID, dtype=torch.bfloat16)
    rows = ck["router"]
    # row invariance is the oracle lane's property (fused=False): ``mm_ordered`` accumulates k in order,
    # so a row's bits are its own at any row count. The fused lane's reduction order is the card's and
    # is launch-shape dependent by design — its contract is the tolerance test below, not bit equality.
    lit = xpu_moe.router(x.to(dev), rows.to(dev), fused=False).cpu()
    for r in range(9):
        assert torch.equal(lit[r], xpu_moe.router(x[r:r + 1].to(dev), rows.to(dev), fused=False).cpu()[0]), f"row {r}"
    assert torch.equal(lit, xpu_moe.router(x, rows, fused=False)), "router logits differ card vs cpu"
    ex_d = xpu.stacked4_from_experts(ck, dev)
    _, pick = _picked(9, 4)
    y = xpu.grouped_moe(x.to(dev), ex_d, pick.to(dev), fused=False).cpu()  # oracle: bits independent of the batch
    for r in range(9):
        one = xpu.grouped_moe(x[r:r + 1].to(dev), ex_d, pick[r:r + 1].to(dev), fused=False).cpu()
        assert torch.equal(y[r], one[0]), f"rank-one row {r} changed with the batch size"
    yf = xpu.grouped_moe(x.to(dev), ex_d, pick.to(dev)).cpu()              # fused: the same, to the pair gate
    for r in range(9):
        one = xpu.grouped_moe(x[r:r + 1].to(dev), ex_d, pick[r:r + 1].to(dev)).cpu()
        assert float((yf[r] - one[0]).abs().max() / yf[r].abs().max()) < REL_FACE, f"fused row {r} off the oracle"
    b1, b2 = (xpu_moe.MoEBuffers(9, CFG, dev), xpu_moe.MoEBuffers(1, CFG, dev))
    xpu_moe.select_rows(lit.to(dev), b1, TOPK, E)
    for r in range(9):
        xpu_moe.select_rows(lit[r:r + 1].to(dev), b2, TOPK, E)
        assert torch.equal(b1.pick[r:r + 1], b2.pick), f"row {r}: picks changed with the batch size"
        assert torch.equal(b1.wts[r:r + 1], b2.wts), f"row {r}: weights changed with the batch size"


def test_router_fused_lane_matches_the_oracle(dev, ck):
    """The fused router (one card matmul) against the rank-one oracle, on the pair-gate tolerance.

    The router is the decode step's dominant term — D is the full hidden width on every row of every
    layer — so the default lane runs ``mm_fused``: one matmul per call instead of ``K`` rank-one
    launches. Its fp32 reduction order is the card's, so the logits are the oracle's within fp32
    reduction noise (the ``tests/cuda`` NVFP4 pair gate), and the selected experts must not move."""

    g = torch.Generator().manual_seed(8)
    x = _rows(g, 9, HID, dtype=torch.bfloat16)
    rows = ck["router"]
    oracle = xpu_moe.router(x.to(dev), rows.to(dev), fused=False).cpu()
    fused = xpu_moe.router(x.to(dev), rows.to(dev), fused=True).cpu()
    assert torch.equal(oracle.argmax(-1), fused.argmax(-1)), "the fused router moved a row's top expert"
    denom = oracle.abs().max().clamp_min(1e-9)
    assert float((fused - oracle).abs().max() / denom) < REL_FACE, "fused router off the oracle past the pair gate"
    for r in range(9):
        one = xpu_moe.router(x[r:r + 1].to(dev), rows.to(dev), fused=True).cpu()
        assert float((fused[r] - one[0]).abs().max() / denom) < REL_FACE, f"fused router row {r} off by the batch"
