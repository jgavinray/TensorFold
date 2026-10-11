#!/usr/bin/env python
"""Bench the fused grouped NVFP4 GEMM against the rank-one oracle on B70 #2 (slice 10, phase 2).

A 2-layer decode step at the checkpoint's widths (E=512, top-k 10, D=2560, NI=640), the ``nvfp4.moe``
contract per layer: router, select, grouped experts, both lanes, timed. Prints the metric lines the
gate greps: ``FUSED_MOE seconds_per_step ...`` and ``FUSED_MOE quality ...``. Tables are synthetic
(the launch storm and the fp32 matmul do not care where the weights came from); one run at a time,
idle card, RAM guard on — the orchestrator's rules.

    PYTHONPATH=src ONEAPI_DEVICE_SELECTOR=level_zero:1 \
      .venv-xpu/bin/python tools/bench_fused_moe_xpu.py > exp/fused-gemm-bench.log 2>&1
"""

from __future__ import annotations

import argparse
import gc
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
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--moe-width", type=int, default=640)
    ap.add_argument("--shared-width", type=int, default=640)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--reps", type=int, default=3, help="timed decode steps for the fused lane")
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
    buf = xpu_moe.MoEBuffers(1, cfg, dev)
    x = torch.randn(1, a.hidden, generator=g, dtype=torch.float32).to(torch.bfloat16).to(dev)
    print(f"bench: layers={a.layers} experts={a.experts} hidden={a.hidden} moe={a.moe_width} "
          f"shared={a.shared_width} topk={a.topk} rows=1 device={dev}")

    def lane(fused: bool, reps: int) -> float:
        gc.collect()
        torch.xpu.reset_peak_memory_stats(dev)

        def fn():
            for _ in range(a.layers):
                xpu.moe(x, None, router_rows, ex, buf, cfg, fused=fused)
        fn()                                                      # warm: JIT, allocator, first table touch
        torch.xpu.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.xpu.synchronize(dev)
        return (time.perf_counter() - t0) / reps

    fast = lane(True, a.reps)
    slow = lane(False, max(1, (a.reps + 1) // 2))                 # the oracle costs seconds per step
    out = xpu.grouped_moe(x, ex, buf.pick[:1]).cpu()
    oracle = xpu.grouped_moe(x, ex, buf.pick[:1], fused=False).cpu()
    rel = max(float((out[r, s] - oracle[r, s]).abs().max() / oracle[r, s].abs().max().clamp(min=1e-12))
              for r in range(out.shape[0]) for s in range(out.shape[1]))
    am = bool(torch.equal(out.argmax(-1), oracle.argmax(-1)))
    peak = torch.xpu.max_memory_allocated(dev) / 2**30
    print(f"FUSED_MOE config layers={a.layers} experts={a.experts} hidden={a.hidden} "
          f"moe_width={a.moe_width} topk={a.topk}")
    print(f"FUSED_MOE seconds_per_step oracle={slow / a.layers:.3f} fused={fast / a.layers:.4f} "
          f"speedup={slow / fast:.1f}")
    print(f"FUSED_MOE two_layer_decode oracle={slow:.3f}s fused={fast:.4f}s")
    print(f"FUSED_MOE quality max_rel={rel:.2e} argmax_match={am} peak_vram_gb={peak:.2f}")
    return 0 if (rel < 2e-2 and am and slow / fast >= 10.0) else 1


if __name__ == "__main__":
    sys.exit(main())
