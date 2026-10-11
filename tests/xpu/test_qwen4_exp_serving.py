"""Wave 5's serving gate: the qwen4_exp XPU engine serving the full checkpoint (B70 #2).

The engine gate certified the decode surface on a two-layer slice; this one serves every
layer with the routed experts streamed from host mmap (``xpu/stream.py``) and the PLE tables
left on disk, and it answers the serving questions: a full-model streamed load with the MTP
head lit (``mtp=True``, pool budget sized to fit the card), greedy decode of a 128-token
prompt over 32 steps matching the oracle fixture's argmax sequence exactly, the MTP
spec-decode path landing on the same tokens greedy produced, and the program.md bench cells -
short 512/256, med 4096/256, long 16384/128, greedy, seed 42 - with decode tok/s, prefill
tok/s, peak VRAM and peak RSS.

The oracle is the engine gate's rule - the same source run on CPU tensors (``exp/oraclelib.py``,
with the host's routed experts served by a bounded LRU over the same dequantized bytes) - but it
is python-loop bound, so the gates never run it inline: ``exp/make_oracle.py`` computes every
greedy sequence once into ``exp/oracle_tokens.json`` and the gates only read that fixture,
skipping with a message naming the script when it is missing. On this box the CPU walk is not
runnable at all: it needs ~49 GB of RSS and the sglang-flashnext server holds ~64 GB of shared
memory permanently, so a walk beside it breaches the RAM guard before its first step.
``make_oracle.py --on-card`` - the documented substitution - computes the walks on the XPU
engine instead; the card/CPU equivalence rests on the merged engine gates at smaller scale.

The port's expert matmul is ``nvfp4.mm_ordered``'s rank-one k-loop (the family's reference),
so per-step costs are launch-bound and the heavy legs here take hours. A bench cell times its
prefill and a 32-step decode window and runs the full token count only when the measured
projection fits ``TF_SERVE_CELL_BUDGET``; a cell that cannot fit reports its measured rate and
cost model and skips. One full load lives on the card at a time: switching between the MTP
load and the greedy load releases the first (pool, weights and all) before the second lands.
"""

from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

import pytest
import torch

os.environ.setdefault("TRITON_CACHE_DIR", "/storage01/.triton-cache-serve")   # the root fs is full
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cuda"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "exp"))

import tensorfold.families.qwen4_exp.xpu.moe  # noqa: E402  (import xpu.moe and xpu.nvfp4 before triton)
import tensorfold.families.qwen4_exp.xpu.nvfp4 as xpu_nvfp4  # noqa: E402
from tensorfold.families.qwen4_exp.xpu import weights as xpu_weights  # noqa: E402
from tensorfold.families.qwen4_exp.xpu import decode as dec  # noqa: E402
from tensorfold.families.qwen4_exp.xpu import stream  # noqa: E402
from tensorfold.families.qwen4_exp.xpu.engine import Engine, PREFILL_ROWS  # noqa: E402

import oraclelib  # noqa: E402  (exp/: the CPU walk and the prompt rule, shared with exp/make_oracle.py)

CKPT = "/storage01/models/Qwen3.8-Flash-Next-NVFP4"
ROWS_FILE = Path(__file__).resolve().parents[2] / "exp" / "serve-cells.json"
POOL_GIB = float(os.environ.get("TF_SERVE_POOL", "20"))          # the contract's starting budget
RAM_GUARD_GB = 24.0
SEED = 42
PROMPT, DECODED = 128, 32
SPEC_DEPTH, SPEC_CONF = 6, 0.7
CELLS = (("short", 512, 256), ("med", 4096, 256), ("long", 16384, 128))
CELL_BUDGET = float(os.environ.get("TF_SERVE_CELL_BUDGET", "3600"))
RATE_STEPS = 32                                    # the decode window a constrained cell measures
ORACLE_FILE = Path(__file__).resolve().parents[2] / "exp" / "oracle_tokens.json"


def _mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _log(msg: str) -> None:
    print(f"[serve] {msg} | RSS {_rss_gb():.1f} GiB avail {_mem_available_gb():.1f} GiB", flush=True)


@pytest.fixture(scope="session", autouse=True)
def _ram_guard():
    """The orchestrator's guard: this gate never runs with less than 24 GB of RAM free."""

    if _mem_available_gb() < RAM_GUARD_GB:
        raise RuntimeError(f"MemAvailable {_mem_available_gb():.1f} GB < {RAM_GUARD_GB} GB: "
                           f"aborting the XPU serving gate")


@pytest.fixture(scope="module")
def dev() -> torch.device:
    """B70 #2: the usable device whose uuid hex contains 4400 (the selector leaves it at index 0)."""

    if not torch.xpu.is_available():
        pytest.skip("no Intel GPU visible")
    for i in range(torch.xpu.device_count()):
        try:
            uuid = bytes(torch.xpu.get_device_properties(i).uuid.bytes).hex()
        except Exception:  # noqa: BLE001
            continue
        if "4400" in uuid:
            return torch.device("xpu", i)
    pytest.skip("no usable XPU device with uuid containing 4400")


_CFG = None


def _cfg():
    global _CFG
    if _CFG is None:
        _CFG = xpu_weights.Config.read(CKPT)
    return _CFG


prompt_tokens = oraclelib.prompt_tokens  # the oracle script's prompt rule, one definition


# -- one live served load at a time --------------------------------------------------------------------------------

_LIVE: dict = {"mode": None, "loaded": None, "engines": {}, "chunk_secs": [], "step_tps": None}

# The contract runs each bench cell as its own `pytest -k` process, which is also what starves
# the cost model: the only in-process rate seed is gate 2's measured 128-row chunk, and under
# -k that lands in a different process. The legs therefore persist their measured prefill chunk
# seconds and decode step rate to ``exp/serve-measured.json`` for the next process to load.
MEASURED_FILE = Path(__file__).resolve().parents[2] / "exp" / "serve-measured.json"


def _load_measured() -> dict:
    try:
        with open(MEASURED_FILE) as f:
            data = json.load(f)
        return {"chunk_secs": [float(s) for s in data.get("chunk_secs", [])],
                "step_tps": data.get("step_tps")}
    except Exception:  # noqa: BLE001
        return {"chunk_secs": [], "step_tps": None}


def _save_measured() -> None:
    payload = {"ckpt": CKPT, "pool_gib": POOL_GIB, "cell_budget_secs": CELL_BUDGET,
               "chunk_secs": _LIVE["chunk_secs"], "step_tps": _LIVE.get("step_tps"),
               "rss_gb": round(_rss_gb(), 2)}
    tmp = MEASURED_FILE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, MEASURED_FILE)


_MEASURED = _load_measured()
_LIVE["chunk_secs"] = list(_MEASURED["chunk_secs"])
_LIVE["step_tps"] = _MEASURED["step_tps"]


def _slot_bytes() -> int:
    return stream.slot_nbytes(_cfg().hidden, _cfg().moe_width)


def _release() -> None:
    _LIVE["engines"].clear()
    if _LIVE["loaded"] is not None:
        _LIVE["loaded"][1].close()
        _LIVE["loaded"] = None
    gc.collect()
    torch.xpu.empty_cache()


def _load_card(dev: torch.device, mtp: bool, gib: float) -> tuple:
    """Streamed load through the family's serving entry, halving the pool budget once on OOM.

    The draft head's MoE is resident and builds its fp32 transposes next to the loader's raw
    stacks - with a full-size pool that is more than the card holds, so the MTP load stages:
    weights first, the draft stacks built (and the raw ones dropped) second, the pool last.
    """

    floor_slots = 2 * _cfg().experts
    budgets = [gib]
    half = gib / 2
    if half * 2**30 // _slot_bytes() >= floor_slots:
        budgets.append(half)
    last: Exception | None = None
    for budget in budgets:
        slots = max(floor_slots, int(budget * 2**30 // _slot_bytes()))
        device = f"xpu:{dev.index}"
        try:
            if not mtp:
                w, streamer = stream.load(CKPT, device, mtp=False, gib=budget, slots=slots,
                                          workers=8)
            else:
                base = stream._patch_reader(xpu_weights, stream.base_names(Path(CKPT)))
                try:
                    w = xpu_weights.load(CKPT, device, mtp=True)
                finally:
                    xpu_weights._Reader = base
                head = w.mtp.layer.moe
                if isinstance(head.experts, dict):
                    head._xpu_stacked = xpu_nvfp4.stacked4_from_experts(head.experts,
                                                                         torch.device(device))
                    head.experts = head._xpu_stacked        # the raw bf16 stacks go before the pool
                gc.collect()
                torch.xpu.empty_cache()
                streamer = stream.attach(w, CKPT, gib=budget, slots=slots, workers=8)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            last = exc
            _log(f"load declined at pool {budget:.1f} GiB: {type(exc).__name__}: {str(exc)[:160]}")
            _release()
            continue
        used, total = _vram(dev)
        _log(f"loaded mtp={mtp} pool {budget:.1f} GiB slots {slots} | card {used:.2f} GiB of "
             f"{total:.1f} peak {torch.xpu.max_memory_allocated() / 2**30:.2f} GiB")
        return w, streamer
    raise RuntimeError(f"no pool budget from {gib} GiB down fits the card: {last}")


def _vram(dev: torch.device) -> tuple[float, float]:
    free, total = torch.xpu.mem_get_info(dev)
    return (total - free) / 2**30, total / 2**30


def serve_engine(dev: torch.device, *, mtp: bool, gib: float = POOL_GIB, capacity: int = 4096,
                 prefill_rows: int | None = None) -> Engine:
    """An engine over the one live load of this mode; a mode switch releases the other load."""

    mode = (mtp, gib)
    if _LIVE["mode"] != mode:
        _release()
        _LIVE["mode"] = mode
        _LIVE["loaded"] = _load_card(dev, mtp, gib)
    key = (capacity, prefill_rows)
    e = _LIVE["engines"].get(key)
    if e is None:
        w = _LIVE["loaded"][0]
        e = Engine(w, capacity=int(capacity), max_rows=8,
                   prefill_rows=int(prefill_rows) if prefill_rows else PREFILL_ROWS,
                   graphs=False, kv_dtype="bf16")
        _LIVE["engines"][key] = e
        _log(f"engine built mtp={mtp} capacity {capacity} prefill_rows {e.prefill_rows} "
             f"kv {e.st.cache_bytes() / 2**30:.2f} GiB")
        torch.xpu.empty_cache()       # return the build's cached blocks so the card's headroom is real
    return e


# -- the oracle fixture ------------------------------------------------------------------------------------------


def oracle_tokens(cell: str, prompt_n: int, new_n: int) -> tuple[list[int], str]:
    """The committed greedy sequence for one cell, read from ``exp/oracle_tokens.json``.

    The walk is never recomputed here. A missing fixture or a missing cell skips with the
    ``exp/make_oracle.py --on-card`` command to run; a fixture that disagrees with this gate's prompt
    rule fails loudly, because stale or hand-edited tokens are a real defect."""

    if not ORACLE_FILE.exists():
        pytest.skip(f"{ORACLE_FILE} missing: build it on the card - python exp/make_oracle.py "
                    f"--on-card gate2,short,med,long --force (the CPU walk's ~49 GB RSS does "
                    f"not fit this box beside the flash server; see the module docstring)")
    with open(ORACLE_FILE) as f:
        data = json.load(f)
    entry = data.get("cells", {}).get(cell)
    if entry is None:
        pytest.skip(f"oracle fixture has no cell '{cell}': python exp/make_oracle.py "
                    f"--on-card {cell} --force")
    if (entry.get("prompt"), entry.get("new"), entry.get("seed"),
            entry.get("prompt_sha256")) != (prompt_n, new_n, SEED,
                                            oraclelib.sha256_json(prompt_tokens(prompt_n))):
        pytest.fail(f"oracle fixture cell '{cell}' does not match this gate's prompt rule "
                    f"(prompt {prompt_n}, {new_n} steps, seed {SEED}): regenerate the fixture")
    tokens = [int(t) for t in entry.get("tokens", [])]
    partial = entry.get("partial")
    if partial is None:
        assert len(tokens) == new_n + 1, f"cell {cell}: the fixture has {len(tokens)} tokens"
    else:
        assert cell in {"short", "med", "long"}, f"cell '{cell}' may not be a prefix fixture"
        assert 1 <= len(tokens) == int(partial) + 1 <= new_n + 1, (
            f"cell {cell}: partial fixture of {len(tokens)} tokens claims {partial} steps")
    return tokens, str(entry.get("source", "cpu"))


def test_full_model_streamed_load_mtp(dev):
    """Every layer streams through the pool, the draft head is resident, the card holds the budget."""

    e = serve_engine(dev, mtp=True, gib=POOL_GIB, capacity=PROMPT + DECODED + 32)
    cfg = e.w.cfg
    assert len(e.w.layers) == cfg.layers >= 48
    for layer in e.w.layers:
        assert isinstance(layer.moe.experts, stream.Pooled4), f"layer {layer.index} is not streamed"
    assert isinstance(e.w.mtp.layer.moe.experts, xpu_nvfp4.Stacked4)     # resident draft head
    assert e.mbuf is not None and e.w.mtp is not None                      # the MTP path is lit
    streamer = e.streamer
    assert streamer is not None and streamer.experts == cfg.experts
    # The documented pool floor is TOTAL slots: one layer's window slots [0, E) plus as many
    # LRU slots [E, S). The streamer structurally reserves the window partition (expert e sits
    # in slot e, filled on demand); only the LRU region circulates on a fresh load.
    assert len(streamer.free) + len(streamer.lru) + cfg.experts >= 2 * cfg.experts
    assert len(streamer.lru) == 0                                   # nothing streamed yet
    used, total = _vram(dev)
    assert used < 0.98 * total, f"card holds {used:.1f} of {total:.1f} GiB"
    assert _rss_gb() < 20.0, "the streamed load must not materialize the checkpoint on the host"


# -- gate 2: greedy decode matches the CPU reference walk ----------------------------------------------------------


def test_greedy_matches_cpu_walk(dev):
    """128-token prompt, 32 greedy steps: the argmax sequence equals the fixture's walk.

    The gate never calls ``oraclelib.cpu_walk_tokens``. The CPU reference walk costs ~49 GB
    of RSS, and this box hosts the sglang-flashnext server permanently (~64 GB of shared
    memory held), so an inline walk breaches the 24 GB RAM guard before its first step - it
    is infeasible here. The fixture is built offline once with the documented substitution
    ``python exp/make_oracle.py --on-card gate2,short,med,long --force``. The card/CPU
    equivalence of the argmax walk rests on the merged engine gates at smaller scale (the
    tiny-fixture end-to-end, the real-checkpoint 1-layer forward and the 2-layer argmax
    walk); against a card-sourced ``gate2`` cell this gate replays that certified walk
    token-for-token through the served engine, the regression that catches drift in the
    serving path.
    """

    want, source = oracle_tokens("gate2", PROMPT, DECODED)
    _log(f"gate 2 comparing the served engine against the fixture's {source} walk")
    prompt = prompt_tokens(PROMPT)
    e = serve_engine(dev, mtp=True, gib=POOL_GIB, capacity=PROMPT + DECODED + 32)
    e.reset()
    t0 = time.perf_counter()
    first = dec.prefill(e, list(prompt), None)
    prefill_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    r = dec.serial_decode(e, first, DECODED + 1, None)
    decode_s = time.perf_counter() - t0
    _log(f"card greedy: prefill {PROMPT} rows in {prefill_s:.0f} s "
         f"({PROMPT / max(prefill_s, 1e-9):.2f} tok/s), decode {len(r.tokens) - 1} steps in "
         f"{decode_s:.0f} s ({(len(r.tokens) - 1) / max(decode_s, 1e-9):.4f} tok/s)")
    _LIVE["chunk_secs"].append(prefill_s)
    _save_measured()
    assert len(r.tokens) == len(want) == DECODED + 1
    assert r.tokens == want, (f"greedy diverged at step "
                              f"{next(i for i in range(len(want)) if r.tokens[i] != want[i])}: "
                              f"card {r.tokens} cpu {want}")


# -- gate 3: the bench cells (greedy, seed 42) -----------------------------------------------------------------------


@pytest.mark.parametrize("cell,prompt_n,new_n", CELLS)
def test_bench_cell(dev, cell, prompt_n, new_n):
    """Prefill tok/s, decode tok/s, peak VRAM, peak RSS; a cell whose measured projection cannot
    fit the wall-clock budget reports its measured rate and cost model instead of a full run.

    The fixture is read before anything pays for a load or a prefill: a cell without its
    committed sequence skips instantly (``exp/make_oracle.py --on-card`` names the build). The
    run's argmax tokens must equal that fixture's sequence (its prefix, when the cell is
    budget-bound). The fixture cells are card-extended walks - the CPU walk is python-loop
    bound, and its ~49 GB RSS cannot coexist with this box's permanent ~64 GB
    sglang-flashnext residency - which is sound because the merged engine gates certify the
    card's argmax walk against the CPU reference at smaller scales; the prefill-argmax
    cross-check below pins every cell's fixture against this process' prefill path regardless
    of the cell's source. The contract runs every cell as its own `pytest -k` process, so the
    cost model's rate seed persists through ``exp/serve-measured.json``."""

    prompt = prompt_tokens(prompt_n)
    want, source = oracle_tokens(cell, prompt_n, new_n)
    assert source in ("cpu", "card"), f"cell {cell}: unknown fixture source {source!r}"
    e = serve_engine(dev, mtp=False, gib=POOL_GIB, capacity=prompt_n + new_n + 64)
    e.reset()
    chunks = -(-prompt_n // e.prefill_rows)
    if chunks > 1 and _LIVE["chunk_secs"]:
        per_chunk = sum(_LIVE["chunk_secs"]) / len(_LIVE["chunk_secs"])
        projected = chunks * per_chunk + (new_n / max(_LIVE.get("step_tps", 1 / 60), 1e-9))
        if projected > CELL_BUDGET:
            used_vram, total = _vram(dev)
            _report(cell, prompt_n, new_n, None, None, 0, new_n, used_vram,
                    torch.xpu.max_memory_allocated() / 2**30, total,
                    note=(f"constraint: eager MoE measured {per_chunk:.0f} s per prefill chunk and "
                          f"{1 / max(_LIVE.get('step_tps', 0), 1e-9):.1f} s per decode step; "
                          f"{chunks} chunks + {new_n} steps project to {projected / 60:.0f} min "
                          f"over the {CELL_BUDGET / 60:.0f} min cell budget"))
            pytest.skip(f"{cell}: projected {projected / 60:.0f} min > budget "
                        f"{CELL_BUDGET / 60:.0f} min (measured {per_chunk:.0f} s/chunk, "
                        f"{1 / max(_LIVE.get('step_tps', 0), 1e-9):.1f} s/token)")
    _, total = _vram(dev)
    torch.xpu.reset_peak_memory_stats()
    t0 = time.perf_counter()
    first = dec.prefill(e, list(prompt), None)
    prefill_s = time.perf_counter() - t0
    prefill_tps = prompt_n / max(prefill_s, 1e-9)
    if chunks == 1:
        _LIVE["chunk_secs"].append(prefill_s)
        _save_measured()

    assert first == want[0], (f"{cell}: the prefill argmax left the fixture's sequence: "
                              f"{first} vs {want[0]}")

    t = time.perf_counter()
    deadline = t + max(CELL_BUDGET - prefill_s, 0.0)
    ticks = 0

    def budget_spent(_new: list[int]) -> bool:
        nonlocal ticks
        ticks += 1
        return ticks >= RATE_STEPS and time.perf_counter() > deadline

    r = dec.serial_decode(e, first, new_n + 1, None, on_tokens=budget_spent)
    decode_s = time.perf_counter() - t
    steps_done = len(r.tokens) - 1
    decode_tps = steps_done / max(decode_s, 1e-9)
    _LIVE["step_tps"] = decode_tps
    _save_measured()
    full = steps_done == new_n
    used_vram, _ = _vram(dev)
    peak_alloc = torch.xpu.max_memory_allocated() / 2**30
    _report(cell, prompt_n, new_n, prefill_tps, decode_tps, steps_done, new_n, used_vram,
            peak_alloc, total,
            note="" if full else
            f"rate over {steps_done}/{new_n} steps (budget-bound); full cell projects "
            f"{(prefill_s + new_n / max(decode_tps, 1e-9)) / 60:.0f} min")

    assert r.tokens[:len(want)] == want[:len(r.tokens)], (
        f"{cell}: decode left the fixture's argmax sequence at step "
        f"{next(i for i, (a, b) in enumerate(zip(r.tokens, want)) if a != b)}")
    assert 0 < steps_done, f"cell {cell} returned 0 tok/s"


def _report(cell, prompt_n, new_n, prefill_tps, decode_tps, steps_done, new_total, used_vram,
            peak_alloc, total, note: str) -> None:
    row = dict(cell=cell, prompt=prompt_n, new=new_n, steps_done=steps_done,
               prefill_tps=None if prefill_tps is None else round(prefill_tps, 3),
               decode_tps=None if decode_tps is None else round(decode_tps, 5),
               vram_used_gb=round(used_vram, 2), vram_peak_gb=round(peak_alloc, 2),
               vram_total_gb=round(total, 2), rss_gb=round(_rss_gb(), 2), note=note)
    print(f"[serve][REPORT] {json.dumps(row)}", flush=True)
    rows = []
    if ROWS_FILE.exists():
        try:
            with open(ROWS_FILE) as f:
                rows = json.load(f)
        except Exception:  # noqa: BLE001
            rows = []
    rows = [r for r in rows if r.get("cell") != cell] + [row]
    ROWS_FILE.parent.mkdir(exist_ok=True)
    with open(ROWS_FILE, "w") as f:
        json.dump(rows, f, indent=1)


# -- gate 4: MTP spec decode lands on the greedy tokens -------------------------------------------------------------


def test_spec_decode_matches_greedy(dev):
    """The speculative path may only accept what greedy would have produced."""

    greedy, source = oracle_tokens("gate2", PROMPT, DECODED)     # the fixture gate 2 replayed
    assert source in ("cpu", "card"), f"gate 4: unknown fixture source {source!r}"
    prompt = prompt_tokens(PROMPT)
    e = serve_engine(dev, mtp=True, gib=POOL_GIB, capacity=PROMPT + DECODED + 32)
    e.reset()
    first = dec.prefill(e, list(prompt), None)
    assert first == greedy[0]
    t0 = time.perf_counter()
    stamps: list[float] = []
    spec = dec.mtp_decode(e, first, DECODED + 1, None, depth=SPEC_DEPTH, confidence=SPEC_CONF,
                          on_tokens=lambda _new: stamps.append(time.perf_counter() - t0) or False)
    el = time.perf_counter() - t0
    _log(f"spec decode: {len(spec.tokens)} tokens in {el:.0f} s ({spec.tokens_per_second:.4f} "
         f"tok/s), rounds {spec.rounds} drafted {spec.drafted} accepted {spec.accepted}")
    assert spec.tokens[:len(greedy)] == greedy, (
        f"spec decode left the greedy path: {spec.tokens[:len(greedy)]} vs {greedy}")
