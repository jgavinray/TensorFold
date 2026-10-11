"""Write ``exp/oracle_tokens.json`` once: the greedy sequences every serving gate reads.

The suite never recomputes the oracle inline. This script computes the reference walks and
commits the fixture:

  gate2            - the 128-token prompt (seed 42) plus 32 greedy steps: the CPU reference
                     walk, the oracle gate 2 and gate 4 compare against.
  short/med/long   - the program.md bench cells (512/256, 4096/256, 16384/128), greedy, seed 42.

The CPU walk streams the routed experts from host mmap through a bounded host LRU and is
python-loop bound - the med/long cells cost days on it, and its ~49 GB peak RSS does not fit
the serving box at all: the sglang-flashnext server holds ~64 GB of shared memory there
permanently, so the walk breaches the RAM guard before its first step. ``--on-card`` computes
a cell's sequence with the XPU engine instead - the sanctioned path on that box, ``gate2``
included: the card's argmax walk is the same function the serving gates replay token-for-token
against the fixture, and its equivalence to the CPU reference is established at smaller scale
by the merged engine gates (tiny-fixture end-to-end, real-checkpoint 1-layer forward,
2-layer argmax walk). The fixture records the source per cell, and the suite's gate docstrings
document the substitution.

Cells already in the JSON are kept unless ``--force``. Every prompt is
``oraclelib.prompt_tokens(n)`` and the fixture stores its sha256, so a prompt-rule change fails
the gates loudly instead of comparing against stale tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "exp"))

import oraclelib  # noqa: E402

OUT_DEFAULT = REPO / "exp" / "oracle_tokens.json"
# cell -> (prompt tokens, greedy steps), the suite's gate 2 and the program.md bench cells
CELLS = {"gate2": (128, 32), "short": (512, 256), "med": (4096, 256), "long": (16384, 128)}


sha256_json = oraclelib.sha256_json


def write_fixture(path: Path, cells: dict) -> None:
    payload = {"ckpt": oraclelib.CKPT, "vocab": int(oraclelib.cfg().vocab), "seed": oraclelib.SEED,
               "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "cells": cells, "sha256": sha256_json({k: v["tokens"] for k, v in cells.items()})}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)
    oraclelib.log(f"fixture {path.name}: {sorted(cells)} sha256 {payload['sha256'][:16]}")


def card_tokens(cell: str, prompt: list[int], new_n: int, gib: float,
                steps: int | None = None) -> tuple[list[int], float]:
    """The cell's greedy sequence from the card engine, called exactly as the serving gates call it.

    ``steps`` may stop the walk early (a prefix fixture); the gates then compare the prefix."""

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "serve_gate", REPO / "tests" / "xpu" / "test_qwen4_exp_serving.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    import torch
    if not torch.xpu.is_available():
        raise RuntimeError("no Intel GPU visible for --on-card")
    dev = None
    for i in range(torch.xpu.device_count()):
        try:
            uuid = bytes(torch.xpu.get_device_properties(i).uuid.bytes).hex()
        except Exception:  # noqa: BLE001
            continue
        if "4400" in uuid:
            dev = torch.device("xpu", i)
            break
    if dev is None:
        raise RuntimeError("no usable XPU device with uuid containing 4400")
    e = suite.serve_engine(dev, mtp=False, gib=gib, capacity=len(prompt) + new_n + 64)
    e.reset()
    t0 = time.perf_counter()
    first = suite.dec.prefill(e, list(prompt), None)
    oraclelib.log(f"{cell}: card prefill {len(prompt)} rows in "
                  f"{time.perf_counter() - t0:.0f} s ({len(prompt) / max(time.perf_counter() - t0, 1e-9):.2f} tok/s)")
    t0 = time.perf_counter()
    ticks = [0]

    def hear(_new: list[int]) -> bool:
        ticks[0] += 1
        if ticks[0] % 8 == 0:
            oraclelib.log(f"{cell}: card decode {ticks[0]}/{(steps or new_n) + 1} "
                          f"+{time.perf_counter() - t0:.0f} s")
        return False

    r = suite.dec.serial_decode(e, first, (steps or new_n) + 1, None, on_tokens=hear)
    secs = time.perf_counter() - t0
    oraclelib.log(f"{cell}: card decode {len(r.tokens) - 1} steps in {secs:.0f} s "
                  f"({(len(r.tokens) - 1) / max(secs, 1e-9):.4f} tok/s)")
    assert len(r.tokens) == (steps or new_n) + 1, f"{cell}: card walk returned {len(r.tokens)} tokens"
    assert r.tokens[0] == first
    return list(r.tokens), secs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", default="gate2,short,med,long",
                    help="comma list from gate2,short,med,long (default all)")
    ap.add_argument("--on-card", default="",
                    help="comma list of cells computed on the XPU engine instead of the CPU walk")
    ap.add_argument("--force", action="store_true", help="recompute cells already in the fixture")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--threads", type=int, default=16, help="torch CPU threads for the walk")
    ap.add_argument("--pool", type=float, default=float(os.environ.get("TF_SERVE_POOL", "20")),
                    help="GiB pool budget for --on-card loads")
    ap.add_argument("--steps", type=int, default=0,
                    help="card walk prefix length (< the cell's new tokens; partial fixture)")
    args = ap.parse_args()
    wanted = [c.strip() for c in args.cells.split(",") if c.strip()]
    card = {c.strip() for c in args.on_card.split(",") if c.strip()}
    unknown = (set(wanted) | card) - set(CELLS)
    if unknown:
        ap.error(f"unknown cells: {sorted(unknown)} (known: {sorted(CELLS)})")

    import torch
    torch.set_num_threads(max(1, args.threads))
    if oraclelib.mem_available_gb() < 24.0:
        raise RuntimeError(f"MemAvailable {oraclelib.mem_available_gb():.1f} GB < 24 GB guard")

    cells: dict = {}
    if args.out.exists():
        with open(args.out) as f:
            cells = json.load(f).get("cells", {})

    if not args.force and all(c in cells for c in wanted):
        oraclelib.log(f"{args.out} already holds every requested cell (use --force to recompute)")
        return 0

    for cell in wanted:
        if cell in cells and not args.force:
            oraclelib.log(f"{cell}: already in the fixture "
                          f"({len(cells[cell]['tokens'])} tokens, source {cells[cell]['source']})")
            continue
        prompt_n, new_n = CELLS[cell]
        prompt = oraclelib.prompt_tokens(prompt_n)
        t0 = time.perf_counter()
        partial: int | None = None
        if cell in card:
            steps = min(args.steps, new_n) if args.steps else new_n
            tokens, _ = card_tokens(cell, prompt, new_n, args.pool, steps)
            source = "card"
            partial = steps if steps < new_n else None
        else:
            if args.steps:
                ap.error("--steps is a --on-card option; the CPU walk computes the full cell")
            seq = oraclelib.cpu_walk_tokens(prompt, new_n)
            assert len(seq) == new_n + 1, f"{cell}: walk returned {len(seq)} tokens"
            tokens, source = seq, "cpu"
        cells[cell] = {"prompt": prompt_n, "new": new_n, "seed": oraclelib.SEED, "source": source,
                       "prompt_sha256": sha256_json(prompt), "tokens": [int(t) for t in tokens],
                       "secs": round(time.perf_counter() - t0, 1),
                       **({"partial": partial} if partial is not None else {})}
        write_fixture(args.out, cells)
        oraclelib.log(f"{cell} ({source}): {len(tokens)} tokens in "
                      f"{cells[cell]['secs']:.0f} s; head {tokens[:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
