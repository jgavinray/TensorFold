#!/bin/bash
# The XPU gate runner: one heavy file at a time, hard per-file timeout, RAM guard between files,
# and a machine-readable verdict line per file. Exists because the naive `pytest tests/xpu -q` sweep
# hung for 2h17m on a single test with no timeout, and a 13-way fan-out drove host RAM to 3 GB.
#
# Usage:  exp/run_gate.sh [file ...]        (default: the heavy files, in dependency order)
# Env:    GATE_TIMEOUT (default 1800)  per-file seconds
#         GATE_RAM_MIN (default 24)    abort if MemAvailable drops below this many GiB
# Output: exp/gate-<file>.log per file; a VERDICT line on stdout per file.
set -u

REPO=${REPO:-/storage01/tensorfold}
V=${V:-/storage01/qwen-for-intel/.venv-xpu/bin/python}
TIMEOUT=${GATE_TIMEOUT:-1800}
RAM_MIN=${GATE_RAM_MIN:-24}

cd "$REPO" || exit 1
export PYTHONUNBUFFERED=1 PYTHONPATH=src
export ONEAPI_DEVICE_SELECTOR=level_zero:1          # B70 #2 only; never combine with ZE_AFFINITY_MASK
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/storage01/.triton-cache-gate}

avail_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }

# default set: heavy files only — the light ones (attention_qsa, gdn, matmul, fused_gemm*) finish in
# seconds and are already covered; re-running them buys nothing.
FILES=("$@")
if [ ${#FILES[@]} -eq 0 ]; then
    FILES=(tests/xpu/test_qwen4_exp_weights.py tests/xpu/test_qwen4_exp_ple.py
           tests/xpu/test_qwen4_exp_hc.py tests/xpu/test_qwen4_exp_mtp.py
           tests/xpu/test_qwen4_exp_engine.py tests/xpu/test_qwen4_exp_serving.py)
fi

rc_all=0
for f in "${FILES[@]}"; do
    b=$(basename "$f" .py)
    log="exp/gate-$b.log"
    a=$(avail_gb)
    if [ "$a" -lt "$RAM_MIN" ]; then
        echo "VERDICT $b SKIP low-ram ${a}G"
        rc_all=1
        continue
    fi
    t0=$(date +%s)
    timeout "$TIMEOUT" "$V" -m pytest "$f" -q -p no:cacheprovider >"$log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    last=$(grep -oE '[0-9]+ (passed|failed|error)[^,]*' "$log" | tail -1)
    case $rc in
        0)   echo "VERDICT $b PASS ${dt}s ${last}" ;;
        124) echo "VERDICT $b TIMEOUT ${dt}s (limit ${TIMEOUT}s)" ;;
        *)   echo "VERDICT $b FAIL rc=$rc ${dt}s ${last}" ;;
    esac
    [ $rc -ne 0 ] && rc_all=1
done
exit $rc_all
