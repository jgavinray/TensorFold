#!/usr/bin/env python
"""Where the serve time goes per forward: router vs select vs the fused grouped MoE (slice 10).

The bench cells measure whole prefills and whole decode steps; this probe prices the three
``nvfp4.moe``-contract components separately on B70 #2 at the checkpoint's widths (E=512,
top-k 10, D=2560, NI=640), for a decode row and for prefill chunk shapes, so the report can
say which component a next optimization must attack. The tables are synthetic (the launch
storm and the fp32 matmul do not care where the weights came from - the same substitution
``bench_fused_moe_xpu.py`` and the serve cost model use); the expert picks are uniform-random,
which matches the real router's spread well enough for launch-count and bandwidth pricing.

    PYTHONPATH=src ONEAPI_DEVICE_SELECTOR=level_zero:1 \
      .venv-xpu/bin/python tools/probe_prefill_xpu.py > exp/serve-fused-probe.log 2>&1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch


def _mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--moe-width", type=int, default=640)
    ap.add_argument("--shared-width", type=int, default=640)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--layers", type=int, default=48, help="full-model depth for the projections")
    ap.add_argument("--rows", default="1,512,2048", help="comma list of chunk shapes to price")
    ap.add_argument("--oracle-rows", type=int, default=0,
                    help="also price the rank-one oracle MoE at this row count (slow)")
    a = ap.parse_args()

    free = _mem_available_gb()
    if free < 24.0:
        print(f"abort: MemAvailable {free:.1f} GB < 24 GB", file=sys.stderr)
        return 2

    os.environ.setdefault("TRITON_DEFAULT_BACKEND", "intel")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tensorfold.families.qwen4_exp.xpu import moe as xpu_moe
    from tensorfold.families.qwen4_exp.xpu import nvfp4 as xpu

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

    g = torch.Generator().manual_seed(31)
    slots = a.topk + 1

    def t(*shape):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.02).to(torch.bfloat16)

    ex = xpu.Stacked4(*[t(*shape).to(dev).float() for shape in
                       ((a.experts, a.hidden, a.moe_width), (a.experts, a.hidden, a.moe_width),
                        (a.experts, a.moe_width, a.hidden), (a.hidden, a.shared_width),
                        (a.hidden, a.shared_width), (a.shared_width, a.hidden))])
    router_rows = (torch.randn(a.experts + 1, a.hidden, generator=g, dtype=torch.float32)
                   * 0.02).to(torch.bfloat16).to(dev)
    cfg = type("C", (), {"num_experts": a.experts, "num_experts_per_tok": a.topk,
                         "hidden_size": a.hidden, "moe_intermediate_size": a.moe_width,
                         "shared_expert_intermediate_size": a.shared_width})
    print(f"probe: experts={a.experts} hidden={a.hidden} moe={a.moe_width} "
          f"shared={a.shared_width} topk={a.topk} layers={a.layers} device={dev}")

    def timed(fn, reps: int = 1) -> float:
        fn()  # warm: JIT, allocator, first table touch
        torch.xpu.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.xpu.synchronize(dev)
        return (time.perf_counter() - t0) / reps

    for rows in [int(r) for r in a.rows.split(",") if r.strip()]:
        x = torch.randn(rows, a.hidden, generator=g, dtype=torch.float32).to(torch.bfloat16).to(dev)
        logits = torch.empty(rows, a.experts + 1, device=dev, dtype=torch.float32)
        buf = xpu_moe.MoEBuffers(rows, cfg, dev)
        t_router = timed(lambda: xpu_moe.router(x, router_rows, logits))
        xpu_moe.select_rows(logits, buf, a.topk, ex.routed)
        t_select = timed(lambda: xpu_moe.select_rows(logits, buf, a.topk, ex.routed))
        pre = rows > 1
        t_moe = timed(lambda: xpu.grouped_moe(x, ex, buf.pick[:rows], prefill=pre))
        line = (f"PREFILL_PROBE rows={rows} router_s={t_router:.4f} select_s={t_select:.4f} "
                f"moe_fused_s={t_moe:.4f}")
        if a.oracle_rows and rows == a.oracle_rows:
            t_or = timed(lambda: xpu.grouped_moe(x, ex, buf.pick[:rows], prefill=pre, fused=False))
            line += f" moe_oracle_s={t_or:.2f}"
        total = t_router + t_select + t_moe
        print(line + f" total_s={total:.4f} x{a.layers}_layers={total * a.layers:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
