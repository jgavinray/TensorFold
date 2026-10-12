#!/bin/bash
# Slice 10's rank-one greedy replay under fixed attention: rebuild the gate-2 fixture on the
# rank-one lane (post rope-fix), then replay the served engine against it. Re-certifies
# b4e66c0's parity claim with the corrected attention math. Logs: exp/serve-rankone-*.log
set -u
V=/storage01/qwen-for-intel/.venv-xpu/bin/python
cd /storage01/tensorfold-s10
export PYTHONUNBUFFERED=1 PYTHONPATH=src
export ONEAPI_DEVICE_SELECTOR=level_zero:1
export TRITON_CACHE_DIR=/storage01/.triton-cache-serve-s10
export TF_SERVE_REQUIRE_FIXTURE=1
exec 3>&1
leg() { local name=$1 rc; shift; "$@"; rc=$?; echo "MARK $name rc=$rc $(date -u +%H:%M:%S)" >&3; return $rc; }

mem_gb=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
echo "MARK preflight MemAvailable=${mem_gb}G $(date -u +%H:%M:%S)"
[ "$mem_gb" -ge 24 ] || { echo "MARK abort-low-ram"; exit 2; }

# The rank-one fixture: ~9 min load + 408 s prefill + 33 steps x ~241 s ~= 2.4 h.
leg build-rankone env TENSORFOLD_XPU_ORDERED=1 $V exp/make_oracle.py --cells gate2 --on-card gate2 --force \
    > exp/serve-rankone-oracle-gate2.log 2>&1 || { echo "MARK abort-build-rankone"; exit 3; }

# The served engine is the fused lane: it replays the rank-one fixture's tokens and must
# reproduce them under the tolerance rule (fork caveats allowed); the fixture head printed by
# the builder is this replay's new sha input.
leg replay env TENSORFOLD_XPU_ORDERED=1 $V -m pytest tests/xpu/test_qwen4_exp_serving.py -q -s \
    -k test_greedy_matches_cpu_walk > exp/serve-rankone-gate2.log 2>&1 || { echo "MARK abort-replay"; exit 3; }
grep -qE '[0-9]+ passed' exp/serve-rankone-gate2.log || { echo "MARK abort-replay-empty-selection"; exit 4; }
echo "MARK ALLDONE $(date -u +%H:%M:%S)"
