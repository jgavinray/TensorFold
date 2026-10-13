"""Slice 6's gate: the qwen4_exp forward on Intel GPUs (B70 #2).

The reference is the same source run on CPU tensors: every op outside the two lane matmuls
(``cuda/qmm.py``, ``cuda/bf16.py``) is a fixed-order reduction or an ``hc``/``nvfp4`` portable
primitive, so card and CPU agree bit-for-bit where no lane kernel touches, and within the
tests/cuda fp16-vs-fp64 relative-norm tolerance (and argmax) where they do.

Gate 1 block-level torch.equal (hc_block; moe_block on the rank-one lane — its shipped fused lane is
judged on the tests/cuda NVFP4 pair gates against that lane's bits), tolerance (gdn, attn, ple
blocks); gate 2 end-to-end tiny-fixture logits + prefill; gate 3 decode row invariance (bits on the
``TENSORFOLD_XPU_MOE_ORDERED`` lane, tolerance and argmax on the shipped fused lane);
gate 4 a real-checkpoint one-layer smoke with peak RSS and VRAM; gate 5 the KV cache refs."""

from __future__ import annotations

import copy
import dataclasses
import os
import resource
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("TRITON_CACHE_DIR", "/storage01/.triton-cache-forward")   # the root fs is full
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cuda"))

import tensorfold.families.qwen4_exp.xpu.weights as xpu_weights  # noqa: E402  (import xpu.nvfp4 first pins the backend)
from tensorfold.families.qwen4_exp.xpu import forward as fw  # noqa: E402
from tensorfold.families.qwen4_exp.xpu import kvcache  # noqa: E402
from tensorfold.families.qwen4_exp.xpu.state import Buffers, State  # noqa: E402

FIXTURE = dict(layers=2, experts=4, vocab=200, hidden=2560, heads=24, kv_heads=2, hd=256,
               nk=16, nv=48, dk=128, dv=128, moe_width=640, shared_width=640, streams=4, low=320,
               ple=True, mtp=False, seed=7)
CKPT = "/storage01/models/Qwen3.8-Flash-Next-NVFP4"
REL = 1e-2                       # the tests/cuda fp16-vs-fp64 gate
REL_FACE = 2e-2                  # tests/cuda/test_nvfp4_experts.py:86 — the grouped-expert pair gate
ROWS, PREFILL_ROWS, CAPACITY = 8, 12, 32
TOKENS = [7, 199, 42, 88, 130, 5, 61, 155]


def _mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return 0.0


@pytest.fixture(scope="session", autouse=True)
def _ram_guard():
    """The orchestrator's guard: this gate's real-checkpoint load must never run under 24 GB."""

    if _mem_available_gb() < 24.0:
        raise RuntimeError(f"MemAvailable {_mem_available_gb():.1f} GB < 24 GB: aborting the XPU forward gate")


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
def pair(tmp_path_factory) -> dict:
    """The tiny checkpoint loaded once per side: card weights and the CPU reference's weights."""

    import nvfp4_tiny

    root = nvfp4_tiny.write(tmp_path_factory.mktemp("forward") / "tiny", **FIXTURE)
    return dict(cpu=xpu_weights.load(root, "cpu"),
                xpu=xpu_weights.load(root, f"xpu:{dev_index()}"))


_DEV_INDEX = 0


def dev_index() -> int:
    return _DEV_INDEX


def bufs(w, prefill: bool = False) -> Buffers:
    return Buffers(w, PREFILL_ROWS if prefill else ROWS, CAPACITY, prefill=prefill)


def _rand(g: torch.Generator, *shape, dtype=torch.float32) -> torch.Tensor:
    return torch.randn(*shape, generator=g, dtype=torch.float32).copy_().to(dtype)


def _seed_states(stx: State, sc: State) -> None:
    """Both sequences start from the same random conv windows, recurrent states and committed length."""

    g = torch.Generator().manual_seed(11)
    c = torch.empty(tuple(stx.conv.shape)).normal_(generator=g)
    r = torch.empty(tuple(stx.rec.shape)).normal_(generator=g)
    stx.conv.copy_(c.to(stx.conv.device))
    sc.conv.copy_(c)
    stx.rec.copy_(r.to(stx.rec.device))
    sc.rec.copy_(r)
    stx.set_pos(5)
    sc.set_pos(5)
    stx.cur = sc.cur = [0] * len(sc.cur)


def _mixed(bx: Buffers, bc: Buffers, gseed: int = 3) -> None:
    g = torch.Generator().manual_seed(gseed)
    c = torch.randn(bc.h.shape, generator=g)
    bx.h.copy_(c.to(bx.h.device, torch.bfloat16))
    bc.h.copy_(c.to(torch.bfloat16))


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().cpu(), b.float().cpu()
    return float(np.linalg.norm((a - b).numpy()) / max(np.linalg.norm(b.numpy()), 1e-9))


# -- gate 1: block level ------------------------------------------------------------------------------------------


def test_hc_block_is_bit_exact_on_card(dev, pair):
    wx, wc = pair["xpu"], pair["cpu"]
    bx, bc = bufs(wx), bufs(wc)
    _seed_states(State(wx, CAPACITY, ROWS), State(wc, CAPACITY, ROWS))
    _mixed(bx, bc)
    c = wx.cfg
    hc = wx.layers[0].attn_hc
    fw.hc_block(hc, bx, ROWS, c.eps, c.streams, c.low, 0, None, bx.inj_a, bx.h)
    fw.hc_block(wc.layers[0].attn_hc, bc, ROWS, c.eps, c.streams, c.low, 0, None, bc.inj_a, bc.h)
    for name in ("pss", "normed", "xs_normed"):    # everything before the lane matmul: the same bits
        assert torch.equal(getattr(bx, name)[:ROWS].cpu(), getattr(bc, name)[:ROWS]), name
    assert torch.equal(bx.h[:ROWS].cpu(), bc.h[:ROWS])          # write-back in place, same bits
    for name in ("act", "xs_act", "mixed", "xs_mixed", "inj_a"):  # downstream of the card's lane kernel
        assert rel(getattr(bx, name)[:ROWS].cpu(), getattr(bc, name)[:ROWS]) < REL, name


def test_moe_block_matches_the_rank_one_oracle(dev, pair, monkeypatch):
    """The rank-one lane (``TENSORFOLD_XPU_MOE_ORDERED=1``) keeps gate 1's card-vs-CPU bit equality;
    the shipped fused lane replaces K launches with one card matmul per face and must sit within the
    ``tests/cuda`` NVFP4 pair gates of that oracle's bits."""

    wx, wc = pair["xpu"], pair["cpu"]
    bx, bc = bufs(wx), bufs(wc)
    _mixed(bx, bc)
    g = torch.Generator().manual_seed(5)
    m = torch.randn((ROWS, wx.cfg.hidden), generator=g).to(torch.bfloat16)
    bx.mixed.copy_(m.to(bx.mixed.device))
    bc.mixed.copy_(m)
    monkeypatch.setenv("TENSORFOLD_XPU_MOE_ORDERED", "1")        # the ordered faces: layout-independent bits
    mode_x, y_x, wts_x = fw.moe_block(wx.layers[0], wx, bx, ROWS)
    mode_c, y_c, wts_c = fw.moe_block(wc.layers[0], wc, bc, ROWS)
    assert (mode_x, mode_c) == (2, 2)
    assert torch.equal(wts_x.cpu(), wts_c.cpu())               # the ordered router's gates
    assert torch.equal(y_x.float().cpu(), y_c.float().cpu())   # the ordered grouped matmul
    oracle = y_c.float().cpu()
    monkeypatch.delenv("TENSORFOLD_XPU_MOE_ORDERED")            # the shipped lane: one matmul per face
    mode_x, y_f, wts_f = fw.moe_block(wx.layers[0], wx, bx, ROWS)
    assert mode_x == 2
    # the shipped lane fuses the router too, so its gates carry the card's fp32 reduction order: they
    # sit within the pair gate of the oracle's rather than equalling them bit for bit. The picks are the
    # contract that must not move — a changed top expert would change which experts run at all.
    assert rel(wts_f.cpu().float(), wts_c.cpu().float()) < REL_FACE, "fused routing weights off the oracle"
    assert torch.equal(bx.moe.pick[:ROWS].cpu(), bc.moe.pick[:ROWS]), "the fused router moved the picks"
    got = y_f.float().cpu()
    assert rel(got, oracle) < REL, f"fused moe_block: rel {rel(got, oracle):.2e}"
    assert torch.allclose(got, oracle, rtol=REL_FACE, atol=REL_FACE)


@pytest.mark.parametrize("which", ("gdn", "attn", "ple"))
def test_blocks_match_the_cpu_reference_within_tolerance(dev, pair, which):
    wx, wc = pair["xpu"], pair["cpu"]
    bx, bc = bufs(wx), bufs(wc)
    stx, sc = State(wx, CAPACITY, ROWS), State(wc, CAPACITY, ROWS)
    _seed_states(stx, sc)
    _mixed(bx, bc)
    gm = torch.Generator().manual_seed(9)
    m = torch.randn((ROWS, wx.cfg.hidden), generator=gm).to(torch.bfloat16)
    bx.mixed.copy_(m.to(bx.mixed.device))
    bc.mixed.copy_(m)                              # the blocks' input: the same bits on both sides
    c = wx.cfg
    layer_x = next(l for l in wx.layers if (l.linear if which == "gdn" else not l.linear))
    layer_c = next(l for l in wc.layers if (l.linear if which == "gdn" else not l.linear))
    if which == "ple":
        layer_x, layer_c = wx.layers[1], wc.layers[1]     # the n-gram branch rides the attention layer
        fw.stage(wx, bx, [(stx, TOKENS)])
        fw.stage(wc, bc, [(sc, TOKENS)])
    segs_x, segs_c = [(stx, 0, ROWS)], [(sc, 0, ROWS)]
    if which == "gdn":
        fw.gdn_block(layer_x, wx, segs_x, bx, ROWS)
        fw.gdn_block(layer_c, wc, segs_c, bc, ROWS)
        outs = ("gout", "gxs")
    elif which == "attn":
        fw.attn_block(layer_x, wx, segs_x, bx, ROWS, mtp=False)
        fw.attn_block(layer_c, wc, segs_c, bc, ROWS, mtp=False)
        outs = ("gated",)              # single-segment decode keeps the attention out in the scratch
    else:
        fw.ple_block(layer_x, wx, segs_x, bx, ROWS)
        fw.ple_block(layer_c, wc, segs_c, bc, ROWS)
        outs = ("ple_emb", "ple_keys", "ple_gated", "ple_pss", "ple_nrow")
    for name in outs:
        x, y = getattr(bx, name)[:ROWS].cpu(), getattr(bc, name)[:ROWS].cpu()
        assert rel(x, y) < REL, f"{which}.{name}: rel {rel(x, y):.2e}"


# -- gate 2: end to end -------------------------------------------------------------------------------------------


def _logits(w, st_factory):
    st, b = st_factory(w, CAPACITY, ROWS), bufs(w)
    out = fw.forward(w, st, b, TOKENS)
    return out[:len(TOKENS)].clone(), st, b


def test_forward_logits_match_cpu(dev, pair):
    lx, stx, _ = _logits(pair["xpu"], State)
    lc, sc, _ = _logits(pair["cpu"], State)
    assert rel(lx, lc) < REL
    assert torch.equal(lx.argmax(-1).cpu(), lc.argmax(-1))


def test_prefill_matches_cpu(dev, pair):
    out = {}
    for side, w, device in (("xpu", pair["xpu"], dev), ("cpu", pair["cpu"], None)):
        st, b = State(w, CAPACITY, PREFILL_ROWS), bufs(w, prefill=True)
        prompt = [3 + 11 * i % 190 for i in range(PREFILL_ROWS)]
        segs = fw.stage(w, b, [(st, prompt)])
        logits = fw.compute(w, segs, b)
        fw.commit(w, st, b, PREFILL_ROWS, PREFILL_ROWS)
        # the kept-point branch: a cut at row 5, its snapshot, and a decode step restored to it
        st2, b2 = State(w, CAPACITY, PREFILL_ROWS), bufs(w, prefill=True)
        cut = fw.Cut(row=5)
        logits2 = fw.forward(w, st2, b2, prompt, cut=cut)
        snap = fw.cut_snapshot(w, st2, b2, cut, mtp_len=0)
        fw.commit(w, st2, b2, PREFILL_ROWS, PREFILL_ROWS)
        st2.restore(snap)
        after = fw.forward(w, st2, b2, [prompt[5], prompt[9]])[:2].clone()
        out[side] = (logits[:1].clone(), logits2[:1].clone(), after, (st.pos, st2.pos))
    assert rel(out["xpu"][0], out["cpu"][0]) < REL
    assert torch.equal(out["xpu"][0].argmax(-1).cpu(), out["cpu"][0].argmax(-1).cpu())
    assert rel(out["xpu"][1], out["cpu"][1]) < REL
    assert torch.equal(out["xpu"][1].argmax(-1).cpu(), out["cpu"][1].argmax(-1).cpu())
    assert rel(out["xpu"][2], out["cpu"][2]) < REL                    # the cut's kept state decodes alike
    assert out["xpu"][3] == out["cpu"][3]


# -- gate 3: decode row invariance --------------------------------------------------------------------------------


def test_decode_rows_one_at_a_time_equal_batched(dev, pair, monkeypatch):
    """A row's decode bits never depend on the other rows: on the rank-one lane they match exactly;
    on the shipped fused lane the fp32 reduction schedule follows the segment's row count, so there
    a row's greedy decision — its argmax — must not move, with a gross-damage norm canary."""

    monkeypatch.setenv("TENSORFOLD_XPU_MOE_ORDERED", "1")
    w = pair["xpu"]
    st_b, b_b = State(w, CAPACITY, ROWS), bufs(w)
    batched = fw.forward(w, st_b, b_b, TOKENS)[:ROWS].clone()
    fw.commit(w, st_b, b_b, ROWS, ROWS)
    st_s, b_s = State(w, CAPACITY, ROWS), bufs(w)
    singles = []
    for t in TOKENS:
        lg = fw.forward(w, st_s, b_s, [t])[:1].clone()
        fw.commit(w, st_s, b_s, 1, 1)
        singles.append(lg)
    single = torch.cat(singles)
    assert torch.equal(batched.cpu(), single.cpu())
    assert torch.equal(st_b.pos_dev.cpu(), st_s.pos_dev.cpu())
    assert torch.equal(st_b.conv.cpu(), st_s.conv.cpu())
    for li in range(len(st_b.cur)):
        assert torch.equal(st_b.rec[st_b.cur[li], li].cpu(), st_s.rec[st_s.cur[li], li].cpu())
    assert np.array_equal(st_b.ple_history, st_s.ple_history)
    monkeypatch.delenv("TENSORFOLD_XPU_MOE_ORDERED")            # the schedule follows the segment's rows
    st_b2, b_b2 = State(w, CAPACITY, ROWS), bufs(w)
    batched2 = fw.forward(w, st_b2, b_b2, TOKENS)[:ROWS].clone()
    fw.commit(w, st_b2, b_b2, ROWS, ROWS)
    st_s2, b_s2 = State(w, CAPACITY, ROWS), bufs(w)
    singles2 = []
    for t in TOKENS:
        lg = fw.forward(w, st_s2, b_s2, [t])[:1].clone()
        fw.commit(w, st_s2, b_s2, 1, 1)
        singles2.append(lg)
    single2 = torch.cat(singles2)
    assert torch.equal(batched2.argmax(-1).cpu(), single2.argmax(-1).cpu()), \
        "a row's argmax moved on the fused lane"
    # fp32 matmul noise (~1e-7 pre-activation) flips a logit's bf16 rounding, and an untrained fixture
    # amplifies the flip: one row's raw logits moved 45% while its argmax held. The fp32 rel is gated
    # at REL by the fused lane's own gates; row invariance here IS the argmax above — this norm is the
    # gross-damage canary (a wrong expert or table would move every row).
    assert rel(batched2.float(), single2.float()) < 20 * REL


# -- gate 4: the real checkpoint, one layer -----------------------------------------------------------------------


def _moved(obj, device):
    """Every tensor under ``obj`` (tensors, dataclasses, containers) copied to ``device``."""

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        other = copy.copy(obj)
        for f in dataclasses.fields(obj):
            setattr(other, f.name, _moved(getattr(obj, f.name), device))
        return other
    if isinstance(obj, dict):
        return {k: _moved(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_moved(v, device) for v in obj)
    if hasattr(obj, "__dict__") and not isinstance(obj, type):   # a plain class's tensors (the B16 face)
        other = copy.copy(obj)
        for key, value in vars(obj).items():
            setattr(other, key, _moved(value, device))
        return other
    return obj


def test_real_checkpoint_one_layer_smoke(dev):
    from dataclasses import replace

    real_read = xpu_weights.Config.read
    xpu_weights.Config.read = staticmethod(lambda d: replace(real_read(d), layers=1))
    try:
        w = xpu_weights.load(CKPT, "cpu")
    finally:
        xpu_weights.Config.read = real_read
    assert [l.index for l in w.layers] == [0]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0
    torch.xpu.reset_peak_memory_stats(dev)
    wx = _moved(w, dev)
    st, b = State(w, CAPACITY, ROWS), bufs(w)
    stx, bx = State(wx, CAPACITY, ROWS), bufs(wx)
    assert wx.cfg.heads == 24 and wx.cfg.streams == 4 and wx.cfg.experts >= 10      # the real shapes
    tokens = [10 + i for i in range(ROWS)]
    lx = fw.forward(wx, stx, bx, tokens)[:ROWS].clone()
    lc = fw.forward(w, st, b, tokens)[:ROWS].clone()
    r = rel(lx, lc)
    vram = torch.xpu.max_memory_allocated(dev) / 1048576.0
    print(f"real-ckpt one-layer smoke: rel {r:.2e} argmax equal: "
          f"{torch.equal(lx.argmax(-1).cpu(), lc.argmax(-1).cpu())} peak RSS {rss:.1f} GiB peak VRAM {vram:.0f} MiB")
    assert r < REL
    assert torch.equal(lx.argmax(-1).cpu(), lc.argmax(-1).cpu())
    del wx, bx, stx, w, st, b
    import gc
    gc.collect()
    torch.xpu.empty_cache()


# -- gate 5: the KV cache references ------------------------------------------------------------------------------


def test_kv_row_bytes_and_quantize_ref_match_cpu(dev):
    for dtype in ("bf16", "int8", "int4"):
        assert kvcache.row_bytes(2, 256, dtype) == kvcache.row_bytes(2, 256, dtype)
    cx = kvcache.KVCache(CAPACITY, 2, 256, dev, "int8")
    cc = kvcache.KVCache(CAPACITY, 2, 256, "cpu", "int8")
    assert cx.k.shape == cc.k.shape and cx.ks.dtype == cc.ks.dtype and cx.nbytes == cc.nbytes
    g = torch.Generator().manual_seed(17)
    k = torch.randn((CAPACITY, 2, 256), generator=g, dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn((CAPACITY, 2, 256), generator=g, dtype=torch.float32).to(torch.bfloat16)
    for bits in (8, 4):
        want = kvcache.quantize_ref(k, v, bits)
        got = kvcache.quantize_ref(k.to(dev), v.to(dev), bits)
        for a, b in zip(got, want):
            assert torch.equal(a.cpu(), b), f"bits {bits}"
        back = kvcache.dequant_ref(*got[:2], bits)
