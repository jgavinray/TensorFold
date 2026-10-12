#!/usr/bin/env python
"""Drive the streamer's slot-pool fill alone on B70 #2: routing answers, worker fills, no engine.

Builds a ``Streamer`` over the real checkpoint's per-expert NVFP4 tensors (``stream.sources``)
without loading the model, then calls ``begin``/``route`` for every layer with random picks,
the way ``grouped_moe`` does through ``Pooled4``. Times each call and reports hits/misses/bytes,
so a walk that stalls inside ``grouped_moe`` is attributed to the worker-pool fill path or
cleared of it. One run at a time, idle card, RAM guard on.

    PYTHONPATH=src ONEAPI_DEVICE_SELECTOR=level_zero:1 \
      .venv-xpu/bin/python tools/probe_stream_fill_xpu.py --steps 3 > exp/probe-fill.log 2>&1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "exp"))
sys.path.insert(0, str(REPO / "src"))

import oraclelib  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=1092)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    if oraclelib.mem_available_gb() < 24.0:
        print(f"abort: MemAvailable {oraclelib.mem_available_gb():.1f} GB < 24 GB", file=sys.stderr)
        return 2
    import torch
    from tensorfold.families.qwen4_exp.xpu import stream as xstream

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

    cfg = oraclelib.cfg()
    layers, experts = int(cfg.num_hidden_layers), int(cfg.num_experts)
    dims, width = int(cfg.hidden_size), int(cfg.moe_intermediate_size)
    model_dir = Path(oraclelib.CKPT)
    base = xpu = xstream.base_names(model_dir)
    names = xstream.expert_names(layers, experts, f"{base}language_model.layers")
    found, shapes = xstream.sources(model_dir, names)
    print(f"FILL_PROBE layers={layers} experts={experts} dims={dims} width={width} "
          f"slots={a.slots} workers={a.workers} device={dev}", flush=True)

    st = xstream.Streamer(found, shapes, device=dev, layers=layers, experts=experts,
                          top_k=a.topk, slots=a.slots, dims=dims, width=width, workers=a.workers)
    rng = np.random.default_rng(a.seed)
    rows = 1
    pick = torch.zeros((rows, a.topk + 1), dtype=torch.int32, device=dev)
    scratch = types.SimpleNamespace(pick=pick)
    t_all = time.perf_counter()
    for step in range(a.steps):
        for layer in range(layers):
            ids = rng.integers(0, experts, size=rows * a.topk)
            pick[:rows, :a.topk] = torch.from_numpy(ids.astype(np.int32)).to(dev)
            pick[rows - 1, a.topk] = experts                      # the shared slot stays resident
            st.begin("rows", rows, scratch)
            t0 = time.perf_counter()
            table = st.route(layer, "rows", rows)
            torch.xpu.synchronize(dev)
            dt = time.perf_counter() - t0
            if dt > 1.0 or (layer in (0, 1) and step < 2):
                print(f"FILL_PROBE step={step} layer={layer} experts={sorted(table)} "
                      f"route_s={dt:.3f} hits={st.hits} misses={st.misses} "
                      f"gib={st.bytes_read / 2**30:.2f}", flush=True)
    print(f"FILL_PROBE done steps={a.steps} wall={time.perf_counter() - t_all:.1f}s "
          f"hits={st.hits} misses={st.misses} gib={st.bytes_read / 2**30:.2f} "
          f"error={st.error}", flush=True)
    st.close()
    return 1 if st.error is not None else 0


if __name__ == "__main__":
    sys.exit(main())
