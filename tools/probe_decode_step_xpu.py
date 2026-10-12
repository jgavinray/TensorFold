#!/usr/bin/env python
"""Price one full-model decode step layer by layer on B70 #2 (slice 10 diagnostics).

Loads the real checkpoint through the serving loader (streamed routed experts, fp32 slot
pool), walks a short prompt, then times ``nvfp4.moe`` per layer with the card synchronized
around each call, alongside the streamer's hit/miss bookkeeping - so a walk that "hangs" is
attributed to routing sync, slot fills, or the matmuls themselves instead of guessed at.
Prints ``STEP_PROBE`` lines live; one run at a time, idle card, RAM guard on.

    PYTHONPATH=src ONEAPI_DEVICE_SELECTOR=level_zero:1 \
      .venv-xpu/bin/python tools/probe_decode_step_xpu.py --steps 3 > exp/probe-decode.log 2>&1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "exp"))

import oraclelib  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--pool", type=float, default=float(os.environ.get("TF_SERVE_POOL", "20")))
    ap.add_argument("--mtp", action="store_true", help="load the MTP engine (gate 2's config)")
    a = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "serve_gate", REPO / "tests" / "xpu" / "test_qwen4_exp_serving.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    import torch
    from tensorfold.families.qwen4_exp.xpu import nvfp4 as xpu_nvfp4

    if oraclelib.mem_available_gb() < 24.0:
        print(f"abort: MemAvailable {oraclelib.mem_available_gb():.1f} GB < 24 GB", file=sys.stderr)
        return 2
    dev = None
    for i in range(torch.xpu.device_count()):
        try:
            uuid = bytes(torch.xpu.get_device_properties(i).uuid.bytes).hex()
        except Exception:  # noqa: BLE001
            continue
        if "4400" in uuid:
            dev = torch.device("xpu", i)
    if dev is None:
        print("abort: no XPU device with uuid containing 4400 (B70 #2)", file=sys.stderr)
        return 2

    e = suite.serve_engine(dev, mtp=a.mtp, gib=a.pool, capacity=a.prompt + a.steps + 64)
    streamer = e.streamer

    orig = xpu_nvfp4.moe
    calls: list[float] = []

    def timed_moe(x, xs, router_rows, ex, buf, cfg) -> None:
        torch.xpu.synchronize(dev)
        t0 = time.perf_counter()
        orig(x, xs, router_rows, ex, buf, cfg)
        torch.xpu.synchronize(dev)
        dt = time.perf_counter() - t0
        calls.append(dt)
        print(f"STEP_PROBE call={len(calls)} moe_s={dt:.3f} hits={streamer.hits} "
              f"misses={streamer.misses} gib={streamer.bytes_read / 2**30:.2f}", flush=True)
        return buf

    xpu_nvfp4.moe = timed_moe
    e.reset()
    t0 = time.perf_counter()
    first = suite.dec.prefill(e, list(oraclelib.prompt_tokens(a.prompt)), None)
    prefill_s = time.perf_counter() - t0
    print(f"STEP_PROBE prefill rows={a.prompt} s={prefill_s:.1f} "
          f"streamer hits={streamer.hits} misses={streamer.misses} "
          f"bytes_read={streamer.bytes_read / 2**30:.1f} GiB", flush=True)
    t0 = time.perf_counter()
    r = suite.dec.serial_decode(e, first, a.steps + 1, None)
    decode_s = time.perf_counter() - t0
    print(f"STEP_PROBE decode steps={len(r.tokens) - 1} s={decode_s:.1f} moe_calls={len(calls)} "
          f"hits={streamer.hits} misses={streamer.misses} "
          f"bytes_read={streamer.bytes_read / 2**30:.1f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
