"""Wave 4b's gate: the qwen4_exp engine on Intel GPUs (B70 #2).

The reference is the same source run on CPU tensors (the tests/cuda oracle rule): greedy decode
through the engine must walk the CPU forward's argmax sequence token for token. Gate 1 is that
end-to-end run (prefill 64 + decode 8); gate 2 checks the engine mechanics (twin, reset, and the
KV cache's admission/eviction at capacity with the 57eac59 row_bytes price semantics); gate 3 puts
the fixture's routed experts through the slot pool — the rank-one lane must ``torch.equal`` the
all-resident grouped path, and the shipped fused lane sits within the tests/cuda NVFP4 pair gates
of it — plus exact hit/miss bookkeeping; gate 4 smokes the real checkpoint streamed on the card (or
a reduced layer count: the rank-one lane walks the CPU reference's tokens token for token, the
shipped lane is judged on card determinism and shared-prefix tolerance) with peak VRAM and RSS;
gate 5 keeps graphs off and the GDN chunk captures exact.
"""

from __future__ import annotations

import contextlib
import gc
import os
import sys
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import pytest
import torch

os.environ.setdefault("TRITON_CACHE_DIR", "/storage01/.triton-cache-engine")   # the root fs is full
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cuda"))

import tensorfold.families.qwen4_exp.xpu.weights as xpu_weights  # noqa: E402  (import xpu.nvfp4 first pins the backend)
from tensorfold.families.qwen4_exp.xpu import forward as fw  # noqa: E402
from tensorfold.families.qwen4_exp.xpu import kvcache, moe as xpu_moe, nvfp4, stream  # noqa: E402
from tensorfold.families.qwen4_exp.xpu import decode as dec  # noqa: E402
from tensorfold.families.qwen4_exp.xpu.engine import Engine  # noqa: E402
from tensorfold.families.qwen4_exp.xpu.state import Buffers, State  # noqa: E402

FIXTURE = dict(layers=2, experts=4, vocab=200, hidden=2560, heads=24, kv_heads=2, hd=256,
               nk=16, nv=48, dk=128, dv=128, moe_width=640, shared_width=640, streams=4, low=320,
               ple=True, mtp=False, seed=7)
CKPT = "/storage01/models/Qwen3.8-Flash-Next-NVFP4"
REL = 2e-2                       # both walks land in bf16 matmuls: measured worst step 1.3e-2 with the
                                 # argmax sequence exact, so the fp16-vs-fp64 1e-2 band is halved in precision
PROMPT = [3 + 11 * i % 190 for i in range(64)]
DECODED = 8
SLOTS = 2 * FIXTURE["experts"]       # a layer's window plus as many for decode: evictions exercise the LRU


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
        raise RuntimeError(f"MemAvailable {_mem_available_gb():.1f} GB < 24 GB: aborting the XPU engine gate")


_DEV_INDEX = 0


@pytest.fixture(scope="module")
def dev() -> torch.device:
    """B70 #2: the usable device whose uuid hex contains 4400 (the selector leaves it at index 0)."""

    global _DEV_INDEX
    for i in range(torch.xpu.device_count()):
        try:
            uuid = bytes(torch.xpu.get_device_properties(i).uuid.bytes).hex()
        except Exception:  # noqa: BLE001
            continue
        if "4400" in uuid:
            _DEV_INDEX = i
            return torch.device("xpu", i)
    pytest.skip("no usable XPU device with uuid containing 4400")


def dev_index() -> int:
    return _DEV_INDEX


@pytest.fixture(scope="module")
def box(dev, tmp_path_factory) -> dict:
    """The tiny checkpoint once per side: resident weights (card and CPU) and the streamed card weights."""

    import nvfp4_tiny

    root = nvfp4_tiny.write(tmp_path_factory.mktemp("engine") / "tiny", **FIXTURE)
    d = f"xpu:{dev_index()}"
    sx, streamer = stream.load(root, d, mtp=False, slots=SLOTS, workers=4)
    return dict(root=root, cpu=xpu_weights.load(root, "cpu"), xpu=xpu_weights.load(root, d),
                sx=sx, streamer=streamer)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float().cpu() - b.detach().float().cpu()).norm()
                 / b.detach().float().cpu().norm().clamp(min=1e-9))


@contextlib.contextmanager
def _layers(n: int):
    """Patch the loader to read the first ``n`` decoder layers only (the reduced-real-checkpoint trick)."""

    real = xpu_weights.Config.read
    xpu_weights.Config.read = staticmethod(lambda d: replace(real(d), layers=n))
    try:
        yield
    finally:
        xpu_weights.Config.read = real


def _reference_walk(w, prompt, count: int):
    """The CPU oracle: chunked prefill and one-token steps, each row sampled as the engine's greedy would.

    Returns the tokens (prefill's first included) and one cloned logit row per decode step."""

    capacity = len(prompt) + count + 16
    st = State(w, capacity, max(8, len(prompt)), "bf16")        # scratch spans the oracle's full-prompt forward
    b = Buffers(w, max(16, len(prompt)), capacity, prefill=True)
    bd = Buffers(w, 8, capacity)
    logits = fw.forward(w, st, b, list(prompt), logits=True)
    out = [int(logits[-1:].argmax(dim=-1)[0])]                # the last row's logits
    fw.commit(w, st, b, len(prompt), len(prompt))
    lgs = []
    for _ in range(count):
        lg = fw.forward(w, st, bd, [out[-1]])[:1].clone()
        out.append(int(lg.argmax(dim=-1)[0]))
        fw.commit(w, st, bd, 1, 1)
        lgs.append(lg)
    return out, lgs


def _reference_tokens(w, prompt, count: int) -> list[int]:
    return _reference_walk(w, prompt, count)[0]


def _card_tokens(e: Engine, prompt, count: int):
    """The same walk on the card through the engine; returns the tokens and the step logits."""

    first = dec.prefill(e, list(prompt), None)
    out, lgs = [first], []
    for _ in range(count):
        lg = e.forward([out[-1]])[:1].clone()
        tok = e.sample(lg, [e.st.pos + 1], None)[0]
        fw.commit(e.w, e.st, e.buf, 1, 1)
        out.append(tok)
        lgs.append(lg)
    return out, lgs


# -- gate 1: engine end to end against the CPU oracle -------------------------------------------------------------


def test_engine_greedy_matches_cpu_oracle(box):
    want = _reference_tokens(box["cpu"], PROMPT, DECODED)
    e = Engine(box["xpu"], capacity=128, max_rows=8, prefill_rows=64)
    assert e.mbuf is None and e.w.mtp is None and e.streamer is None      # the fixture has no MTP head
    assert dec.serial_decode(e, dec.prefill(e, list(PROMPT), None), DECODED + 1, None).tokens == want
    # the streamed engine walks the same tokens
    es = Engine(box["sx"], capacity=128, max_rows=8, prefill_rows=64)
    assert es.streamer is box["streamer"] and box["streamer"]._buffers["rows"] is es.buf.moe
    assert dec.serial_decode(es, dec.prefill(es, list(PROMPT), None), DECODED + 1, None).tokens == want


def test_sampling_is_position_keyed(box):
    """A keyed draw (temperature with top-k) is deterministic per sequence and stays in the vocabulary."""

    from tensorfold.engine.exact_sampling import Sampling

    s = Sampling(temperature=0.8, top_k=20, seed=5)
    for side in ("xpu", "sx"):
        w = box[side]
        a = dec.serial_decode(Engine(w, capacity=128, max_rows=8, prefill_rows=64),
                              dec.prefill(Engine(w, capacity=128, max_rows=8, prefill_rows=64),
                                          list(PROMPT), s), 4, s).tokens
        b = dec.serial_decode(Engine(w, capacity=128, max_rows=8, prefill_rows=64),
                              dec.prefill(Engine(w, capacity=128, max_rows=8, prefill_rows=64),
                                          list(PROMPT), s), 4, s).tokens
        assert a == b
        assert all(0 <= t < FIXTURE["vocab"] for t in a)


# -- gate 2: engine mechanics --------------------------------------------------------------------------------------


def test_twin_state_is_independent(box):
    want = _reference_tokens(box["cpu"], PROMPT, DECODED)
    e = Engine(box["xpu"], capacity=128, max_rows=8, prefill_rows=64)
    kept = dec.serial_decode(e, dec.prefill(e, list(PROMPT), None), 5, None).tokens
    assert kept == want[:5]
    t = e.twin()
    assert t.st is not e.st and t.buf is e.buf and t.pbuf is e.pbuf and t.mbuf is None and t.graphs is None
    assert dec.serial_decode(t, dec.prefill(t, list(PROMPT), None), 5, None).tokens == want[:5]
    # the main engine's committed state survived the twin's run: it continues exactly where it was
    assert dec.serial_decode(e, want[4], DECODED - 3, None).tokens == want[4:]


def test_reset_empties_the_sequence(box):
    e = Engine(box["xpu"], capacity=128, max_rows=8, prefill_rows=64)
    dec.serial_decode(e, dec.prefill(e, list(PROMPT), None), 5, None)
    assert e.st.pos > 0
    e.reset()
    assert e.st.pos == 0 and e.st.mtp_len == 0
    assert not e.st.conv.any() and not e.st.rec.any()
    want = _reference_tokens(box["cpu"], PROMPT, DECODED)
    assert dec.serial_decode(e, dec.prefill(e, list(PROMPT), None), DECODED + 1, None).tokens == want


def test_kv_admission_and_eviction_at_capacity(box):
    """row_bytes carries the real dtype price (57eac59); ensure grows by the step to the limit; resize keeps committed rows."""

    c = box["cpu"].cfg
    index_bytes = c.index_dim * 2
    for dtype in ("bf16", "int8"):
        st = State(box["cpu"], 64, 8, dtype, limit=256)
        assert st.row_bytes == kvcache.row_bytes(c.kv_heads, c.head_dim, dtype) + index_bytes
        assert st.layer_bytes(64) == 64 * st.row_bytes + -(-64 // st.ratio) * index_bytes
        assert st.cache_bytes() == st.layers * st.layer_bytes()
    st = State(box["cpu"], 64, 8, "bf16", limit=256)
    got = st.ensure(72, step=16)
    assert st.capacity == 80 and got == st.layers * (st.layer_bytes(80) - st.layer_bytes(64)) > 0
    with pytest.raises(ValueError, match="past this sequence's"):
        st.ensure(300)
    st.set_pos(50)
    with pytest.raises(ValueError, match="can't keep"):
        st.resize(49)
    added = st.resize(128)
    assert st.capacity == 128 and added == st.layers * (st.layer_bytes(128) - st.layer_bytes(80)) > 0
    e = Engine(box["xpu"], capacity=64, max_rows=8, prefill_rows=64)
    e.st.set_pos(60)
    with pytest.raises(ValueError, match="past this sequence's"):
        e.st.ensure(1024)


def test_engine_refuses_multi_rank(box):
    import copy as _copy

    w = _copy.copy(box["cpu"])
    w.comm = object()
    with pytest.raises(ValueError, match="tensor parallelism of 1"):
        Engine(w)
    from tensorfold.families.qwen4_exp.xpu import xpu_engine

    with pytest.raises(ValueError, match="one GPU"):
        xpu_engine(box["root"], tp=2)
    with pytest.raises(ValueError, match="kv-dtype"):
        xpu_engine(box["root"], kv_dtype="fp8")


# -- gate 3: the expert slot pool ------------------------------------------------------------------------------------


def test_streamed_experts_equal_the_resident_grouped_path(box, dev):
    c = box["sx"].cfg
    x = torch.randn((8, c.hidden), generator=torch.Generator().manual_seed(23)).to(dev).bfloat16()
    b = Buffers(box["sx"], 8, 64)
    box["streamer"].attach_buffers(b.moe, b.moe)        # an Engine would register the pick scratch here
    m = box["sx"].layers[0].moe
    xpu_moe.router(x, m.router, b.moe.logits[:8])
    xpu_moe.select_rows(b.moe.logits[:8], b.moe, c.top_k, c.experts)
    pick = b.moe.pick[:8].clone()
    resident = nvfp4.stacked4_from_experts(box["xpu"].layers[0].moe.experts, dev)
    box["streamer"].begin("rows", 8)
    streamed = nvfp4.grouped_moe(x, m.experts, pick)
    y = nvfp4.grouped_moe(x, resident, pick)
    # the rank-one oracle: elementwise bits, independent of the table's memory — streaming answers exactly
    assert torch.equal(nvfp4.grouped_moe(x, m.experts, pick, fused=False).cpu(),
                       nvfp4.grouped_moe(x, resident, pick, fused=False).cpu())
    # the shipped fused lane: the fp32 matmul schedule follows the input's memory (pool slot vs
    # resident stack), so this lane rides the tests/cuda NVFP4 pair gates, not bits
    assert rel(streamed, y) < REL, f"streamed vs resident fused: rel {rel(streamed, y):.1e}"
    assert torch.equal(streamed.argmax(-1).cpu(), y.argmax(-1).cpu())
    # the tables themselves: every streamed expert byte-equals its resident dequant
    ex = m.experts
    assert ex._slots, "the routing answered no experts"
    for e, slot in ex._slots.items():
        for proj in stream.PROJS:
            got = box["streamer"].pool[proj][slot].cpu()
            want = getattr(resident, f"{proj}_t")[e].cpu()
            assert torch.equal(got, want), f"{proj} expert {e}"


def test_slot_bookkeeping_is_exact(box):
    s = box["streamer"]
    s.lru.clear()
    s.free = list(range(SLOTS - 1, FIXTURE["experts"] - 1, -1))
    s.window = [-1] * FIXTURE["experts"]
    e = Engine(box["sx"], capacity=128, max_rows=8, prefill_rows=64)
    first = dec.prefill(e, list(PROMPT), None)
    for attr in ("hits", "misses", "bytes_read", "window_fills"):
        setattr(s, attr, 0)
    s.calls.clear()
    dec.serial_decode(e, first, DECODED + 1, None)
    rows = [(layer, ids) for layer, kind, ids in s.calls if kind == "rows"]
    assert rows, "the decode steps never routed"
    assert not [1 for _layer, kind, _ids in s.calls if kind == "window"], "decode used the window"
    # replay the documented LRU on the same call log: every resolution is a hit or a miss, no more, no less
    free = list(range(SLOTS - 1, FIXTURE["experts"] - 1, -1))
    lru: OrderedDict = OrderedDict()
    hits = misses = 0
    for layer, ids in rows:
        keep = {(layer, e_) for e_ in ids}
        for e_ in ids:
            key = (layer, e_)
            if key in lru:
                lru.move_to_end(key)
                hits += 1
                continue
            misses += 1
            if free:
                slot = free.pop()
            else:
                slot = next(v for k, v in lru.items() if k not in keep)
                lru.pop(next(k for k, v in lru.items() if v == slot))
            lru[key] = slot
    assert (s.hits, s.misses) == (hits, misses), f"{s.hits}/{s.misses} vs replay {hits}/{misses}"
    each = stream.expert_nbytes(s.found, 0)
    assert each > 0
    assert s.bytes_read == s.misses * each > 0
    for (layer, e_), slot in lru.items():
        assert s.slot_of(layer, e_) == slot
    # the window's own fills: after a fresh prefill every first-touch (layer, expert) of the chunk filled once
    s2 = box["streamer"]
    e2 = Engine(box["sx"], capacity=128, max_rows=8, prefill_rows=64)
    for attr in ("hits", "misses", "bytes_read", "window_fills"):
        setattr(s2, attr, 0)
    s2.lru.clear()                                          # slot_of prefers the LRU: leave only the window's
    s2.free = list(range(SLOTS - 1, FIXTURE["experts"] - 1, -1))
    dec.prefill(e2, list(PROMPT), None)
    touched = {(layer, e_) for layer, kind, ids in s2.calls if kind == "window" for e_ in ids}
    assert s2.window_fills > 0
    assert s2.bytes_read == s2.window_fills * each
    assert all(s2.slot_of(layer, e_) == e_ for (layer, e_) in touched if s2.window[e_] == layer)


def test_pool_refuses_a_window_short_of_the_experts(box):
    with pytest.raises(ValueError, match="window slots"):
        stream.attach(box["xpu"], box["root"], slots=FIXTURE["experts"] * 2 - 1)


# -- gate 4: the real checkpoint on the card -------------------------------------------------------------------------


def test_real_checkpoint_streamed_smoke(dev, monkeypatch):
    import resource

    free_vram, total = torch.xpu.mem_get_info(dev)
    cfg = xpu_weights.Config.read(CKPT)
    each = stream.slot_nbytes(cfg.hidden, cfg.moe_width)
    need = (2 * cfg.experts + 128) * each + 9 * 2**30       # the whole pool plus resident weights, head and KV room
    prompt = [(13 * i + 7) % 100000 for i in range(128)]
    loaded = None
    if free_vram > need + 10 * 2**30:
        try:
            loaded = stream.load(CKPT, f"xpu:{dev_index()}", mtp=False, slots=2 * cfg.experts + 128, workers=8)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            print(f"[gate 4] full-model streamed load declined: {type(exc).__name__}: {str(exc)[:160]}")
            loaded = None
            gc.collect()
            torch.xpu.empty_cache()
    if loaded is not None:                                    # the whole 48-layer model on the card
        w, s = loaded
        assert len(w.layers) == cfg.layers
        assert all(isinstance(l.moe.experts, stream.Pooled4) for l in w.layers)
        torch.xpu.reset_peak_memory_stats(dev)
        e = Engine(w, capacity=256, max_rows=8, prefill_rows=128)
        out, _lgs = _card_tokens(e, prompt, DECODED)
        peak = torch.xpu.max_memory_allocated(dev) / 2**20
        used = (total - torch.xpu.mem_get_info(dev)[0]) / 2**30
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0
        print(f"[gate 4] FULL model (48 layers) streamed: 128-token prefill + {DECODED} decode steps, "
              f"first tokens {out[:4]}... peak VRAM {peak:.0f} MiB, {used:.1f} GiB in use, RSS {rss:.1f} GiB, "
              f"hits {s.hits} misses {s.misses}, {s.bytes_read / 2**30:.1f} GiB read from the checkpoint")
        assert len(out) == DECODED + 1 and all(isinstance(t, int) and t >= 0 for t in out)
        del e, w, s
        gc.collect()
        torch.xpu.empty_cache()
        return
    # the card (or the RAM guard) cannot host the full model: gate at a reduced layer count with the CPU walk
    layers = 2
    with _layers(layers):
        wc = xpu_weights.load(CKPT, "cpu")
        wx, s = stream.load(CKPT, f"xpu:{dev_index()}", mtp=False, slots=2 * cfg.experts + 128, workers=8)
    assert [l.index for l in wx.layers] == list(range(layers))
    torch.xpu.reset_peak_memory_stats(dev)
    e = Engine(wx, capacity=256, max_rows=8, prefill_rows=128)
    out, lgs = _card_tokens(e, prompt, DECODED)                      # the shipped fused lane
    e2 = Engine(wx, capacity=256, max_rows=8, prefill_rows=128)
    out2, _lgs2 = _card_tokens(e2, prompt, DECODED)
    assert out == out2, "the fused lane's greedy walk is not reproducible on the card"
    monkeypatch.setenv("TENSORFOLD_XPU_MOE_ORDERED", "1")            # the exactness lane replays the walk bits
    want, clgs = _reference_walk(wc, prompt, DECODED)
    e3 = Engine(wx, capacity=256, max_rows=8, prefill_rows=128)
    o3, lgs3 = _card_tokens(e3, prompt, DECODED)
    monkeypatch.delenv("TENSORFOLD_XPU_MOE_ORDERED")
    peak = torch.xpu.max_memory_allocated(dev) / 2**20
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0
    agree = sum(1 for a, b in zip(out, want) if a == b)
    print(f"[gate 4] reduced to {layers} of {cfg.layers} layers (streamed experts): rank-one argmax sequence "
          f"equal: {o3 == want}, fused lane agrees on {agree}/{DECODED + 1} (a greedy near-tie forks the walk; "
          f"serving text parity is the serve gate's), per-step rel "
          f"{' '.join(f'{rel(a, b):.1e}' for a, b in zip(lgs3, clgs))}, "
          f"peak VRAM {peak:.0f} MiB, RSS {rss:.1f} GiB, hits {s.hits} misses {s.misses}")
    assert o3 == want
    for i, (a, b) in enumerate(zip(lgs3, clgs)):
        assert rel(a, b) < REL, f"rank-one decode step {i}"
    first = min((i for i, (a, b) in enumerate(zip(out, want)) if a != b), default=DECODED + 1)
    for i in range(min(first + 1, len(lgs), len(clgs))):             # the shared prefix: fused logits track it
        assert rel(lgs[i], clgs[i]) < REL, f"fused decode step {i}"
    del e, e2, e3, wx, wc
    gc.collect()
    torch.xpu.empty_cache()


# -- gate 5: graphs ---------------------------------------------------------------------------------------------------


def test_graphs_decline_and_gdn_chunks_stay_exact(box, capsys):
    want = _reference_tokens(box["cpu"], PROMPT, DECODED)
    e = Engine(box["xpu"], capacity=128, max_rows=8, prefill_rows=64)
    assert e.graphs is None and e._graphs_declined                 # graphs off by default
    assert dec.serial_decode(e, dec.prefill(e, list(PROMPT), None), DECODED + 1, None).tokens == want
    eg = Engine(box["xpu"], capacity=128, max_rows=8, prefill_rows=64, graphs=True)
    assert eg.graphs is None
    assert "graphs off" in capsys.readouterr().out                 # the refusal is said out loud, once
    # the 64-row prompt chunk still ran the captured GDN path: the exact same walk
    assert dec.serial_decode(eg, dec.prefill(eg, list(PROMPT), None), DECODED + 1, None).tokens == want
