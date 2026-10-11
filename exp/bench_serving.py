"""Serve the qwen4_exp XPU gate: tests/xpu/test_qwen4_exp_serving.py, logged to exp/serve-xpu.log.

    /storage01/qwen-for-intel/.venv-xpu/bin/python exp/bench_serving.py [pytest args...]

Runs the serving gate in the foreground with the orchestrator's environment (B70 #2 only, the
Triton cache off the full root fs) and tees everything to the log. Heavy legs take hours.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "exp" / "serve-xpu.log"
GATE = "tests/xpu/test_qwen4_exp_serving.py"


def main() -> int:
    env = dict(os.environ)
    env.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:1")   # the 4400 card, and only it
    env.setdefault("PYTHONPATH", str(ROOT / "src"))
    env.setdefault("TRITON_CACHE_DIR", "/storage01/.triton-cache-serve")
    env["PYTHONUNBUFFERED"] = "1"                     # stream the gate's [serve] lines into the tee
    cmd = [sys.executable, "-m", "pytest", "-q", "-s", "--tb=short", "-x", GATE, *sys.argv[1:]]
    print(" ".join(cmd), flush=True)
    with open(LOG, "a", buffering=1) as log:
        log.write(f"\n===== bench_serving {' '.join(sys.argv[1:]) or 'gate'} "
                  f"({__import__('datetime').datetime.now().isoformat(timespec='seconds')}) =====\n")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return int(proc.wait())


if __name__ == "__main__":
    raise SystemExit(main())
