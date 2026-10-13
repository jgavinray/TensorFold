"""Wave 4a's gate: the qwen4_exp MTP head and draft head on Intel GPUs (B70 #2).

The reference is the same source run on CPU tensors (the oracle the tests/cuda gates use): every op
outside the lane matmuls is a fixed-order reduction or an ``hc`` portable primitive, so card and CPU
agree within the tests/cuda fp16-vs-fp64 relative-norm tolerance (and argmax) where a lane or
b16 kernel touches (a bf16 logit's 1 ulp rides on the draft confidence), and bit-for-bit
elsewhere: the dequantized tables, ``hc``'s reductions, the cache writes.

Gate 1 tiny-fixture mtp logits + the MTP cache's write/read under absorb semantics;
gate 2 the draft head: id_map mapping, greedy draft parity at temperature 0, confidence prob;
gate 3 the quantized draft head on the dequant route (the gs=32 lane gap documented);
gate 4 MTP rows one at a time vs batched (torch.equal);
gate 5 the real checkpoint's mtp window, card vs CPU, with peak RSS and VRAM."""

from __future__ import annotations

import copy
import dataclasses
import gc
import os
import resource
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("TRITON_CACHE_DIR", "/storage01/.triton-cache-forward")   # the root fs is full
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cuda"))

import tensorfold.families.qwen4_exp.xpu.moe  # noqa: E402,F401  (the card backend pin lands before any launch)
import tensorfold.families.qwen4_exp.xpu.nvfp4  # noqa: E402,F401
import tensorfold.families.qwen4_exp.xpu.weights as xpu_weights  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import qmm  # noqa: E402
from tensorfold.families.qwen4_exp.xpu import mtp  # noqa: E402
from tensorfold.families.qwen4_exp.xpu.state import Buffers, State  # noqa: E402

FIXTURE = dict(layers=2, experts=4, vocab=200, hidden=2560, heads=24, kv_heads=2, hd=256,
               nk=16, nv=48, dk=128, dv=128, moe_width=640, shared_width=640, streams=4, low=320,
               ple=True, mtp=True, seed=7)
CKPT = "/storage01/models/Qwen3.8-Flash-Next-NVFP4"
REL = 1e-2                       # the tests/cuda fp16-vs-fp64 gate
ROWS, CAPACITY = 8, 32
NEXT = [7, 199, 42, 88]          # the MTP head's next tokens
DRAFT_IDS = [5, 7, 42, 61, 88, 130, 155, 199]


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
        raise RuntimeError(f"MemAvailable {_mem_available_gb():.1f} GB < 24 GB: aborting the XPU MTP gate")


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


_DEV_INDEX = 0


def dev_index() -> int:
    return _DEV_INDEX


@pytest.fixture(scope="module")
def pair(tmp_path_factory) -> dict:
    """The tiny checkpoint (MTP head included) loaded once per side; the draft head scores DRAFT_IDS.

    The card side is the CPU load carried over (gate 5's real-checkpoint flow), not a direct load
    to the device: ``bf16.quantize4``'s ``(mx - mn) / 15.0`` is a tensor/scalar divide and this
    XPU backend lowers tensor/scalar divides to a reciprocal multiply (6e5 of 1e6 quotients
    differ from the correctly-rounded ones), so quantizing the draft rows on-card lands a few
    ``.5``-boundary nibbles off the reference's table. The card must score the table the CPU
    reference scored; the loader's on-device quantize gap is reported as a shared-file bug."""

    import nvfp4_tiny

    root = tmp_path_factory.mktemp("mtp")
    tiny = nvfp4_tiny.write(root / "tiny", **FIXTURE)
    vocab = root / "draft_vocab.txt"
    vocab.write_text("\n".join(str(i) for i in DRAFT_IDS))
    wc = xpu_weights.load(tiny, "cpu", mtp=True, draft_vocab=str(vocab))
    return dict(cpu=wc, xpu=_moved(wc, torch.device("xpu", dev_index())))


def bufs(w) -> Buffers:
    return Buffers(w, ROWS, CAPACITY)


def rand_streams(rows: int, seed: int, to=None) -> torch.Tensor:
    """The main model's bf16 streams [rows, streams * hidden], the same bits on every side."""

    g = torch.Generator().manual_seed(seed)
    wide = FIXTURE["streams"] * FIXTURE["hidden"]
    x = torch.randn((rows, wide), generator=g, dtype=torch.float32).to(torch.bfloat16)
    return x if to is None else x.to(to)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().cpu(), b.float().cpu()
    return float(np.linalg.norm((a - b).numpy()) / max(np.linalg.norm(b.numpy()), 1e-9))


# -- gate 1: the MTP head against the CPU reference ----------------------------------------------------------------


def test_mtp_forward_matches_cpu_reference(dev, pair):
    wx, wc = pair["xpu"], pair["cpu"]
    stx, bx, stc, bc = State(wx, CAPACITY, ROWS), bufs(wx), State(wc, CAPACITY, ROWS), bufs(wc)
    streams = rand_streams(4, seed=21)
    lx = mtp.mtp_forward(wx, stx, bx, NEXT, streams.to(wx.device), last_only=False)[:len(NEXT)].clone()
    lc = mtp.mtp_forward(wc, stc, bc, NEXT, streams, last_only=False)[:len(NEXT)].clone()
    assert rel(lx, lc) < REL
    assert torch.equal(lx.argmax(-1).cpu(), lc.argmax(-1).cpu())


def _absorb_run(w, streams):
    """One side's lockstep: 4 rows in, the entries read back by a chained row, absorb trims the drafts."""

    st, b = State(w, CAPACITY, ROWS), bufs(w)
    lg = mtp.mtp_forward(w, st, b, NEXT, streams[:4], last_only=False).clone()
    assert st.mtp_len == 0                                        # the caller advances the length
    st.set_mtp_len(st.mtp_len + 4)
    read = mtp.mtp_forward(w, st, b, [55], streams[3:4], last_only=False).clone()   # reads the 4 entries back
    st.set_mtp_len(st.mtp_len + 1)
    st.mtp_drafted = 2                                            # two chained drafts, never verified
    back = mtp.absorb(w, st, b, streams[4:5], [13]).clone()       # trims to 3 rows, then takes the kept one
    return st, b, lg, read, back


def test_mtp_cache_write_read_and_absorb_match_reference(dev, pair):
    wx, wc = pair["xpu"], pair["cpu"]
    streams = rand_streams(5, seed=23)
    stx, bx, lgx, rd_x, bk_x = _absorb_run(wx, streams.to(wx.device))
    stc, bc, lgc, rd_c, bk_c = _absorb_run(wc, streams)
    for a, b_, tag in ((lgx, lgc, "window"), (rd_x, rd_c, "read"), (bk_x, bk_c, "absorb")):
        a, b_ = a[:, :8], b_[:, :8]                               # one row of draft-head logits each
        assert rel(a, b_) < REL, tag
    assert torch.equal(lgx[:len(NEXT)].argmax(-1).cpu(), lgc[:len(NEXT)].argmax(-1).cpu())
    # the cache state: the same committed rows, positions and draft accounting
    assert (stx.mtp_len, stx.mtp_drafted) == (stc.mtp_len, stc.mtp_drafted) == (4, 0)
    assert torch.equal(stx.mtp_pos.cpu(), stc.mtp_pos.cpu())
    for name in ("k", "v"):
        assert rel(getattr(stx.mtp_kc, name)[:5], getattr(stc.mtp_kc, name)[:5]) < REL, name
    assert rel(stx.mtp_ikc[:5], stc.mtp_ikc[:5]) < REL


# -- gate 2: the draft head ----------------------------------------------------------------------------------------


def test_draft_id_map_matches_the_reference_vocab(dev, pair):
    for side in ("xpu", "cpu"):
        w = pair[side]
        assert np.array_equal(w.draft_ids.cpu().numpy(), np.array(sorted(set(DRAFT_IDS))))
        assert w.draft_head.n == len(DRAFT_IDS)


def _draft_row(w, seed=31):
    """One window's draft-head logits (the last row, ``last_only``'s shape) on one side."""

    st, b = State(w, CAPACITY, ROWS), bufs(w)
    streams = rand_streams(1, seed=seed, to=w.device)
    return mtp.mtp_forward(w, st, b, [42], streams, last_only=True).clone(), b


def _confidence(logits: torch.Tensor) -> float:
    """``sample_draft``'s fp32 confidence rule recomputed on the row (the check's own oracle)."""

    row = logits.float()
    return float(torch.exp(row.max(-1, keepdim=True).values
                           - torch.logsumexp(row, -1, keepdim=True)))


def test_draft_head_greedy_parity_and_confidence(dev, pair):
    wx, wc = pair["xpu"], pair["cpu"]
    lx, _ = _draft_row(wx)
    lc, _ = _draft_row(wc)
    assert lx.shape == lc.shape == (1, len(DRAFT_IDS))
    assert rel(lx, lc) < REL
    assert torch.equal(lx.argmax(-1).cpu(), lc.argmax(-1).cpu())  # the same draft column both sides
    col = int(lx.argmax(-1)[0])
    assert rel(lx[0, col], lc[0, int(lc.argmax(-1)[0])]) < REL
    tok_x, p_x = mtp.sample_draft(lx, 9, None, wx.draft_ids)
    tok_c, p_c = mtp.sample_draft(lc, 9, None, wc.draft_ids)
    assert tok_x == tok_c                                         # the reference's greedy path
    assert tok_x == int(wx.draft_ids[col].cpu())                  # the id_map maps the column to its real id
    assert abs(p_x - _confidence(lx)) < 1e-6 and abs(p_c - _confidence(lc)) < 1e-6   # fp32-exact per side
    assert 0.0 <= p_x <= 1.0 and 0.0 <= p_c <= 1.0
    # the b16 draft matmul and ``_cpu_mm`` are different programs: a 1-ulp bf16 logit moves
    # exp(top - lse) by up to ~1%, so the cross-side bound on the ulp-moved confidence is ~5e-3.
    assert abs(p_x - p_c) < 5e-3
    zero = Sampling(seed=3, temperature=0.0)
    assert mtp.sample_draft(lx, 9, zero, wx.draft_ids)[0] == tok_x
    assert mtp.sample_mapped(lx, [9], zero, wx.draft_ids) == [tok_x]      # decode.sample's draft branch


def test_draft_chain_matches_the_reference(dev, pair):
    """``draft``'s chain (the engine wiring's consumption): the same drafts and cache accounting as CPU."""

    wx, wc = pair["xpu"], pair["cpu"]
    streams = rand_streams(4, seed=41)
    stx, bx = State(wx, CAPACITY, ROWS), bufs(wx)
    stc, bc = State(wc, CAPACITY, ROWS), bufs(wc)
    zero = Sampling(seed=5, temperature=0.0)
    dx = mtp.draft(wx, stx, bx, streams.to(wx.device), [3, 61, 88, 5], position=12, count=3, sampling=zero)
    dc = mtp.draft(wc, stc, bc, streams, [3, 61, 88, 5], position=12, count=3, sampling=zero)
    assert dx == dc and all(d in set(int(i) for i in DRAFT_IDS) for d in dx)
    assert (stx.mtp_len, stx.mtp_drafted) == (stc.mtp_len, stc.mtp_drafted)


# -- gate 3: the quantized draft head on the dequant route -----------------------------------------------------------


def test_draft_head_lane_gap_and_dequant_route(dev, pair):
    wx, wc = pair["xpu"], pair["cpu"]
    assert isinstance(wx.draft_head, qmm.Q4) and wx.draft_head.gs == 32    # bf16.quantize4's pack
    x = rand_streams(1, seed=51, to=wx.device)[:, :FIXTURE["hidden"]].contiguous()
    xs = torch.zeros((1, FIXTURE["hidden"] // 32), dtype=torch.float32, device=wx.device)
    with pytest.raises(NotImplementedError, match="gs=64"):        # the known kernel gap, documented here
        qmm.matmul(x, wx.draft_head, xs)
    # the dequant table: elementwise fp32 arithmetic, the same bits on card and reference
    assert torch.equal(qmm.dequantize_q4(wx.draft_head).cpu(), qmm.dequantize_q4(wc.draft_head))
    b = bufs(wx)
    out = torch.empty((1, wx.draft_head.n), dtype=torch.bfloat16, device=wx.device)
    got = mtp._draft_mm(x, wx.draft_head, xs, out, b).cpu()
    want = (x.float().cpu() @ qmm.dequantize_q4(wc.draft_head).float().cpu().t()).to(torch.bfloat16).float()
    assert rel(got, want) < REL
    assert torch.equal(got.argmax(-1), want.to(torch.bfloat16).argmax(-1).cpu().reshape(1))


# -- gate 4: MTP row invariance --------------------------------------------------------------------------------------


def test_mtp_rows_one_at_a_time_equal_batched(dev, pair):
    wx = pair["xpu"]
    streams = rand_streams(4, seed=61, to=wx.device)
    st_b, b_b = State(wx, CAPACITY, ROWS), bufs(wx)
    batched = mtp.mtp_forward(wx, st_b, b_b, NEXT, streams, last_only=True).clone()
    st_b.set_mtp_len(len(NEXT))
    st_s, b_s = State(wx, CAPACITY, ROWS), bufs(wx)
    last = None
    for j, t in enumerate(NEXT):
        last = mtp.mtp_forward(wx, st_s, b_s, [t], streams[j:j + 1], last_only=True).clone()
        st_s.set_mtp_len(st_s.mtp_len + 1)
    assert torch.equal(batched.cpu(), last.cpu())                 # the draft head's last-row logits
    assert st_b.mtp_len == st_s.mtp_len == len(NEXT)
    assert torch.equal(st_b.mtp_kc.k[:4].cpu(), st_s.mtp_kc.k[:4].cpu())   # one window wrote what four steps did
    assert torch.equal(st_b.mtp_kc.v[:4].cpu(), st_s.mtp_kc.v[:4].cpu())
    assert torch.equal(st_b.mtp_ikc[:4].cpu(), st_s.mtp_ikc[:4].cpu())


def test_mtp_multi_window_equals_separate_streams(dev, pair):
    """Two windows in one pass (the ``b.last`` gather, k > 1) equal each stream's own run."""

    wx = pair["xpu"]
    streams = rand_streams(4, seed=71, to=wx.device)
    st1, st2 = State(wx, CAPACITY, ROWS), State(wx, CAPACITY, ROWS)
    b = bufs(wx)
    outs = mtp.mtp_compute(wx, mtp.mtp_stage(wx, b, [(st1, NEXT[:2], streams[:2]),
                                                     (st2, NEXT[2:], streams[2:])]), b, last_only=True)
    alone = []
    for st, part, toks in ((st1, streams[:2], NEXT[:2]), (st2, streams[2:], NEXT[2:])):
        st_a, b_a = State(wx, CAPACITY, ROWS), bufs(wx)
        alone.append(mtp.mtp_forward(wx, st_a, b_a, toks, part, last_only=True).clone())
    assert torch.equal(outs[0:1].cpu(), alone[0].cpu())
    assert torch.equal(outs[1:2].cpu(), alone[1].cpu())


# -- gate 5: the real checkpoint -------------------------------------------------------------------------------------


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


def test_real_checkpoint_mtp_window_smoke(dev):
    """The real checkpoint's MTP head, one layer, both sides — and the card memory this test borrows is
    handed back before it returns.

    The card is a shared 32 GB resource, and this test is the file's only real-checkpoint leg. Returning
    the borrow explicitly — the moved weights, the buffers, and the cached blocks — keeps the file
    order-independent and leaves the next file a clean L0 context.

    (History: this test once failed with ``UR_RESULT_ERROR_OUT_OF_RESOURCES`` on an 84 MiB allocation
    while the card reported 32 GB free. That was not allocator pressure from this file — the device was
    failing underneath the run, and ``devcoredump`` later recorded it. The explicit release below is
    still correct hygiene; it is not a fix for a memory bug that never existed.)"""

    from dataclasses import replace

    real_read = xpu_weights.Config.read
    xpu_weights.Config.read = staticmethod(lambda d: replace(real_read(d), layers=1))
    try:
        w = xpu_weights.load(CKPT, "cpu", mtp=True, draft_vocab="default")
    finally:
        xpu_weights.Config.read = real_read
    wx = None
    try:
        assert w.mtp is not None and w.draft_head is not None
        assert isinstance(w.draft_head, qmm.Q4) and w.draft_head.gs == 32
        assert w.draft_head.n == int(w.draft_ids.shape[0]) > 0
        assert [l.index for l in w.layers] == [0]
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0
        torch.xpu.reset_peak_memory_stats(dev)
        wx = _moved(w, dev)
        st, b = State(w, CAPACITY, ROWS), bufs(w)
        stx, bx = State(wx, CAPACITY, ROWS), bufs(wx)
        tokens = [3 + 977 * i % int(w.meta["full"].vocab) for i in range(4)]
        streams = rand_streams(4, seed=91)
        lx = mtp.mtp_forward(wx, stx, bx, tokens, streams.to(wx.device), last_only=False)[:4].clone()
        lc = mtp.mtp_forward(w, st, b, tokens, streams, last_only=False)[:4].clone()
        r = rel(lx, lc)
        dx = mtp.mtp_forward(wx, stx, bx, tokens[:1], streams[:1].to(wx.device), last_only=True).clone()
        dc = mtp.mtp_forward(w, st, b, tokens[:1], streams[:1], last_only=True).clone()
        rd = rel(dx, dc)
        tok_x = mtp.sample_draft(dx, 4, None, wx.draft_ids)[0]
        tok_c = mtp.sample_draft(dc, 4, None, w.draft_ids)[0]
        vram = torch.xpu.max_memory_allocated(dev) / 1048576.0
        print(f"real-ckpt mtp smoke: full-head rel {r:.2e} draft rel {rd:.2e} draft token equal: {tok_x == tok_c} "
              f"peak RSS {rss:.1f} GiB peak VRAM {vram:.0f} MiB")
        assert r < REL and torch.equal(lx.argmax(-1).cpu(), lc.argmax(-1).cpu())
        assert rd < REL and tok_x == tok_c
    finally:
        # release the card borrow before the module teardown runs its own empty_cache
        wx = None
        gc.collect()
        if torch.xpu.is_initialized():
            torch.xpu.empty_cache()
            torch.xpu.reset_peak_memory_stats(dev)
