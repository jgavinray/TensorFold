#!/bin/bash
# Slice 10's fused serving legs, sequential (never two heavy legs at once), B70 #2 only.
# Contract: each gate/cell is its own pytest -k process; the fixture builder prices the
# cell's walk once (prefix), the cell process then replays and measures it.
# Logs: exp/serve-fused-*.log; every leg prints "MARK <leg> rc=<n> <utc>" on completion.
set -u
V=/storage01/qwen-for-intel/.venv-xpu/bin/python
cd /storage01/tensorfold-s10
export PYTHONUNBUFFERED=1 PYTHONPATH=src
export ONEAPI_DEVICE_SELECTOR=level_zero:1
export TRITON_CACHE_DIR=/storage01/.triton-cache-serve-s10

mem_gb=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
echo "MARK preflight MemAvailable=${mem_gb}G $(date -u +%H:%M:%S)"
[ "$mem_gb" -ge 24 ] || { echo "MARK abort-low-ram"; exit 2; }

export TF_SERVE_REQUIRE_FIXTURE=1
exec 3>&1
leg() { local name=$1 rc; shift; "$@"; rc=$?; echo "MARK $name rc=$rc $(date -u +%H:%M:%S)" >&3; return $rc; }

leg probe $V tools/probe_prefill_xpu.py > exp/serve-fused-probe.log 2>&1
leg build-gate2 $V exp/make_oracle.py --cells gate2 --on-card gate2 --force \
    > exp/serve-fused-oracle-gate2.log 2>&1 || { echo "MARK abort-build-gate2"; exit 3; }
leg gate2 $V -m pytest tests/xpu/test_qwen4_exp_serving.py -q -s \
    -k test_greedy_matches_cpu_walk > exp/serve-fused-gate2.log 2>&1 || { echo "MARK abort-gate2"; exit 3; }

for cell in short med long; do
    mem_gb=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    [ "$mem_gb" -ge 24 ] || { echo "MARK abort-low-ram-before-$cell"; exit 2; }
    leg build-$cell $V exp/make_oracle.py --cells "$cell" --on-card "$cell" --steps 65 --force \
        > "exp/serve-fused-oracle-$cell.log" 2>&1 || { echo "MARK abort-build-$cell"; exit 3; }
    TF_SERVE_CELL_BUDGET=21600 leg cell-$cell $V -m pytest tests/xpu/test_qwen4_exp_serving.py -q -s \
        -k "test_bench_cell and $cell" > "exp/serve-fused-cell-$cell.log" 2>&1
done
echo "MARK ALLDONE $(date -u +%H:%M:%S)"
