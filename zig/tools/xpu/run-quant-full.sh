#!/usr/bin/env bash
# run-quant-full.sh — run the full fresh-EAQ quant of Qwen3.8-27B on the B70 to completion.
#
# Usage (on hyper03):
#   ~/xpu-eaq/run-quant-full.sh                    # the full run (all 64 layers + head + publish)
#   ~/xpu-eaq/run-quant-full.sh --dry-run          # the gates only: --dry-run --layers 0..0
#   ~/xpu-eaq/run-quant-full.sh --plan PLAN.json   # allocator-plan run: the driver gets
#                                                  # --plan PLAN.json on every invocation
#
# What it does:
#   1. Asserts the tf-main-8081 user service is stopped (stops it if running) so the
#      card is dedicated to the run; restarts it on EVERY exit path via an EXIT trap.
#   2. Runs xpu-eaq-quant in a loop, up to $ATTEMPTS attempts, every line timestamped
#      into $LOG. When a work dir already carries progress.json each attempt passes
#      --resume (skipping layers the manifest records); a fresh work dir runs without
#      --resume, which the driver refuses (Usage), with --seed providing the key.
#   3. Abort-vs-crash policy on a nonzero exit:
#        - ABORT (bug signal, never retried): the attempt's last log line matches the
#          driver's failed-tile or Cholesky abort pattern — the CLI's terminal
#          `xpu-eaq-quant: TileFailed` / `xpu-eaq-quant: CholeskyFailed`, or their
#          signatures (`N failed tiles of M`, `the damped rotated H failed at pivot`).
#          These are deterministic quantizer bugs; resuming cannot fix them, so the
#          script stops, writes the abort line to $FAIL, and exits 1. Inspect the
#          dumped failed_H sidecar / tile stats before touching the pipeline.
#        - CRASH (retried with --resume): any other nonzero exit (device hiccup,
#          power cap, OOM of a transient kind). The next attempt resumes from the last
#          layer the progress manifest recorded.
#      $ATTEMPTS consecutive crashes without progress => $FAIL "attempts exhausted",
#      exit 2.
#   4. On success (exit 0; full mode only) writes $DONE with the pack path and a
#      summary of $WORK/progress.json (layers complete, head recorded, run parameters).
#      --dry-run publishes nothing and records no progress, so it writes no $DONE.
#   5. Prints free memory before the run and after the loop (in the exit trap).
#   6. Pre-flight, all host-side, before the first attempt:
#        (1) the calibration ids file parses: rows x cols decimal tokens
#            (204800 for rows=100 cols=2048), the first and last ids sane, every id
#            below the model vocab (.text_config.vocab_size in the source config.json);
#        (2) the eval prompts file exists and its sha256 matches the
#            content_sha256 recorded in eval.manifest.json;
#        (3) at least MIN_FREE_GB (default 120; override via the environment for
#            validation) is free on the work filesystem — the pack, the fragments
#            and the streams (~10-15 GB plus checkpoints) all live under
#            /storage01/xpu-eaq, and the parents are created as needed;
#        (4) the driver binary exists and answers --help with its usage text;
#        (5) with --plan FILE: the file parses as JSON and carries
#            target_bpw/method/parts/plan_sha. Until the driver's --plan parser
#            lands (its usage must list --plan; QuantFix owns the build), a --plan
#            run is REFUSED at preflight rather than burning attempts on unknown
#            options. Once the parser is in, full mode runs a one-shot
#            `--dry-run --layers 0..0 --plan FILE` acceptance pass right after
#            dedicating the card: the driver's own refusal exits 4, an abort inside
#            that pass follows the same abort policy ($FAIL, exit 1), and only then
#            does the publish loop start. --plan passes through verbatim to every
#            driver invocation, dry mode included (the dry-run loop itself carries
#            the plan to the gate pass).
#
# Exit codes: 0 success | 1 abort (bug signal) | 2 attempts exhausted |
#             3 service would not stop | 4 precondition/preflight/plan refused.

set -u -o pipefail

ROOT="$HOME/xpu-eaq"
BIN="$ROOT/tensorfold/zig-out/bin/xpu-eaq-quant"
SRC="/storage01/models/Qwen3.8-27B"
OUT="/storage01/xpu-eaq/Qwen3.8-27B-EAQ-3.0bpw"
CALIB="$ROOT/calib-200k.ids.txt"
WORK="/storage01/xpu-eaq/quantwork"
DRY_WORK="/storage01/xpu-eaq/quantwork-dry"
LOG="$ROOT/quant-full.log"
DONE="$ROOT/quant-full.DONE"
FAIL="$ROOT/quant-full.FAILED"
EVAL_DIR="$ROOT/eval"
EVAL_PROMPTS="$EVAL_DIR/eval.i32"
EVAL_MANIFEST="$EVAL_DIR/eval.manifest.json"

ROWS=100
COLS=2048
BITS=3
HEAD_BITS=4
SEED=0xE0
ATTEMPTS=5
MIN_FREE_GB="${MIN_FREE_GB:-120}"
UNIT=tf-main-8081

# The driver's failed-tile / Cholesky abort signatures (quant.zig; the CLI prints the
# error name as its terminal line, quant_main.zig). A last line matching this is a bug.
ABORT_SIG_RE='[0-9]+ failed tiles of [0-9]+ |the damped rotated H failed at pivot '
ABORT_RE="xpu-eaq-quant: (TileFailed|CholeskyFailed)\$|$ABORT_SIG_RE"

# systemctl --user needs the session bus even under a plain ssh login.
: "${XDG_RUNTIME_DIR:=/run/user/$(id -u)}"
export XDG_RUNTIME_DIR
: "${DBUS_SESSION_BUS_ADDRESS:=unix:path=$XDG_RUNTIME_DIR/bus}"
export DBUS_SESSION_BUS_ADDRESS

now() { date -u +%FT%TZ; }
ts() { gawk '{ print strftime("[%Y-%m-%dT%H:%M:%SZ] "), $0; fflush() }'; }
log() { printf '[%s] %s\n' "$(now)" "$*" | tee -a "$LOG"; }

cleanup() {
  local rc=$?
  log "restart $UNIT (exit path rc=$rc)"
  systemctl --user start "$UNIT" 2>&1 | ts | tee -a "$LOG" \
    || log "WARNING: systemctl --user start $UNIT failed — start it manually"
  log "free memory after run"
  free -h 2>&1 | ts | tee -a "$LOG"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

die() { log "ERROR: $*"; exit 4; }

write_abort_failed() { # $1 = context, $2 = exit code, $3 = terminal abort line
  log "ABORT detected (failed-tile / Cholesky) — deterministic bug signal, NOT retrying"
  {
    printf '%s\n' "$3"
    local sig
    sig=$(grep -E "$ABORT_SIG_RE" "$LOG" | tail -n 1 || true)
    if [ -n "$sig" ] && [ "$sig" != "$3" ]; then
      printf '%s\n' "$sig"
    fi
    printf '# run-quant-full.sh: %s hit an abort (exit %s), %s\n' "$1" "$2" "$(now)"
    printf '# failed-tile / Cholesky aborts are bug signals: resuming cannot fix them.\n'
    printf '# Inspect the failed_H dump named in the line and the tile stats in %s.\n' "$LOG"
  } >"$FAIL"
  log "wrote $FAIL"
  exit 1
}

MODE=full
PLAN=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      MODE=dry ;;
    --plan)
      if [ "$#" -lt 2 ] || [ -z "${2:-}" ] || [ "${2#--}" != "$2" ]; then
        printf 'usage: %s [--dry-run] [--plan FILE]\n' "$0" >&2
        exit 4
      fi
      PLAN=$2
      shift ;;
    *)
      printf 'usage: %s [--dry-run] [--plan FILE]\n' "$0" >&2
      exit 4 ;;
  esac
  shift
done

# --------------------------------------------------------------- preconditions
for f in "$BIN" "$CALIB"; do
  [ -e "$f" ] || die "missing $f"
done
[ -d "$SRC" ] || die "missing source model dir $SRC"
for t in gawk jq free systemctl sha256sum; do
  command -v "$t" >/dev/null || die "missing tool $t"
done
systemctl --user cat "$UNIT" >/dev/null 2>&1 || die "user unit $UNIT not found"
mkdir -p "$WORK" "$DRY_WORK" "$OUT" || die "cannot create work dirs"

log "==== run-quant-full start mode=$MODE plan=${PLAN:-none} pid=$$ attempts=$ATTEMPTS"
log "driver: $BIN"
log "src=$SRC out=$OUT calib=$CALIB work=$WORK"

# ------------------------------------------------------- pre-flight host gates
# (1) the calibration ids the whole run keys on: rows x cols decimal tokens, the
#     first and last sane, none outside the model's vocab range.
[ -s "$CALIB" ] || die "calib file missing or empty: $CALIB"
vocab=$(jq -r '(.text_config.vocab_size // .vocab_size) // "none"' "$SRC/config.json" 2>/dev/null)
: "${vocab:=none}"
calib_stats=$(awk -v rows="$ROWS" -v cols="$COLS" -v vocab="$vocab" '
  {
    for (i = 1; i <= NF; i++) {
      if ($i !~ /^[0-9]+$/) {
        err = sprintf("field %d on line %d is not a decimal id: %s", i, NR, $i)
        exit
      }
      if (vocab != "none" && $i + 0 >= vocab + 0) {
        err = sprintf("id %s on line %d is not below the vocab size %s", $i, NR, vocab)
        exit
      }
      n++
      if (n == 1) { first = $i }
      last = $i
    }
  }
  END {
    if (err) { print err; exit 1 }
    if (n + 0 != rows * cols) { printf "%d ids, expected %d (rows x cols)\n", n + 0, rows * cols; exit 1 }
    printf "%d ids, first=%s last=%s vocab=%s\n", n, first, last, vocab
  }' "$CALIB") || die "calib $CALIB does not parse: $calib_stats"
log "calib ok: $calib_stats"

# (2) the eval prompts corpus against its recorded sha256.
[ -f "$EVAL_PROMPTS" ] || die "eval prompts file missing: $EVAL_PROMPTS"
[ -f "$EVAL_MANIFEST" ] || die "eval prompts manifest missing: $EVAL_MANIFEST"
recorded=$(jq -r '.content_sha256 // empty' "$EVAL_MANIFEST" 2>/dev/null)
[ -n "$recorded" ] || die "no content_sha256 recorded in $EVAL_MANIFEST"
actual=$(sha256sum "$EVAL_PROMPTS" | awk '{ print $1 }')
[ "$actual" = "$recorded" ] \
  || die "eval prompts sha256 mismatch: $actual != recorded $recorded ($EVAL_MANIFEST)"
log "eval prompts ok: $EVAL_PROMPTS sha256 $actual"

# (3) space for the pack, the fragments and the streams — all of them under
#     /storage01/xpu-eaq, so the work filesystem alone carries the gate.
need_kb=$((MIN_FREE_GB * 1024 * 1024))
kb=$(df -Pk /storage01 | awk 'NR == 2 { print $4 }')
{ [ -n "$kb" ] && [ "$kb" -ge "$need_kb" ]; } \
  || die "only ${kb:-unknown} KB free on /storage01 — the run needs ${MIN_FREE_GB} GB free; clear space or move the work tree"
log "disk ok: /storage01 has $(df -hP /storage01 | awk 'NR == 2 { print $4 }') free (>= ${MIN_FREE_GB} GB)"

# (4) the driver is there and prints its usage.
usage_txt=$("$BIN" plan-src-probe plan-out-probe --help 2>&1) \
  || die "$BIN --help exited $? (expected 0 with a usage text)"
grep -q 'usage: xpu-eaq-quant' <<<"$usage_txt" \
  || die "$BIN --help did not print its usage text"
log "driver ok: $BIN answers --help with usage"

# (5) the allocator plan: JSON shape first; the driver gate needs the --plan parser.
if [ -n "$PLAN" ]; then
  [ -f "$PLAN" ] || die "plan file missing: $PLAN"
  jq -e . "$PLAN" >/dev/null 2>&1 || die "plan $PLAN is not valid JSON"
  jq -e 'has("target_bpw") and has("method") and has("parts") and has("plan_sha")' \
    "$PLAN" >/dev/null 2>&1 \
    || die "plan $PLAN must carry target_bpw, method, parts and plan_sha"
  log "plan ok: $(jq -r '"target_bpw=\(.target_bpw // "?") method=\(.method // "?") parts=\(.parts | length) plan_sha=\((.plan_sha // "?") | tostring | .[0:12])"' "$PLAN") in $PLAN"
  grep -q -- '--plan' <<<"$usage_txt" \
    || die "the driver build has no --plan parser yet (its usage does not list --plan) — plan $PLAN validated, but the run is refused; get the --plan build from QuantFix"
fi

log "free memory before run"
free -h 2>&1 | ts | tee -a "$LOG"

# ------------------------------------------------------- dedicate the card (1)
state=$(systemctl --user is-active "$UNIT" 2>/dev/null || true)
log "$UNIT is $state"
if [ "$state" != inactive ] && [ "$state" != failed ]; then
  log "stopping $UNIT"
  if ! systemctl --user stop "$UNIT" 2>&1 | ts | tee -a "$LOG"; then
    log "ERROR: could not stop $UNIT"
    exit 3
  fi
fi
for _ in $(seq 30); do
  state=$(systemctl --user is-active "$UNIT" 2>/dev/null || true)
  { [ "$state" = inactive ] || [ "$state" = failed ]; } && break
  sleep 1
done
if [ "$state" != inactive ] && [ "$state" != failed ]; then
  log "ERROR: $UNIT never reached inactive (last state $state)"
  exit 3
fi
log "$UNIT is stopped — card dedicated to the run"

# ------------------------------------------------------------------- (5 cont.)
# The driver-side plan gate: one gate pass on layer 0 with the plan, before any
# publish attempt. A driver refusal is final (exit 4); an abort follows the abort
# policy; dry mode needs no separate pass — its loop below carries --plan itself.
if [ -n "$PLAN" ] && [ "$MODE" = full ]; then
  log "plan acceptance: one-shot --dry-run --layers 0..0 with $PLAN"
  AARGS=("$SRC" "$OUT" --calib "$CALIB" --rows "$ROWS" --cols "$COLS"
        --bits "$BITS" --head-bits "$HEAD_BITS" --seed "$SEED"
        --layers 0..0 --dry-run --work "$DRY_WORK" --plan "$PLAN")
  log "driver: $BIN ${AARGS[*]}"
  "$BIN" "${AARGS[@]}" 2>&1 | ts >>"$LOG"
  arc=${PIPESTATUS[0]}
  last_line=$(tail -n 1 "$LOG")
  log "plan acceptance exited rc=$arc"
  if [ "$arc" -ne 0 ]; then
    if printf '%s\n' "$last_line" | grep -Eq "$ABORT_RE"; then
      write_abort_failed "plan acceptance" "$arc" "$last_line"
    fi
    die "the driver refused the plan (exit $arc): $last_line"
  fi
  log "plan accepted by the driver's layer-0 gate pass"
fi

# ----------------------------------------------------- the attempt loop (2),(3)
attempt=1
while [ "$attempt" -le "$ATTEMPTS" ]; do
  ARGS=("$SRC" "$OUT" --calib "$CALIB" --rows "$ROWS" --cols "$COLS"
        --bits "$BITS" --head-bits "$HEAD_BITS" --seed "$SEED")
  if [ "$MODE" = dry ]; then
    # The layer-0 gate pass: run the gates, write no pack, record no progress, in an
    # isolated work dir so the real run's checkpoint is untouched.
    ARGS+=(--layers 0..0 --dry-run --work "$DRY_WORK")
  else
    ARGS+=(--work "$WORK")
    if [ -f "$WORK/progress.json" ]; then
      ARGS+=(--resume)
    fi
  fi
  if [ -n "$PLAN" ]; then
    ARGS+=(--plan "$PLAN")
  fi
  log "attempt $attempt of $ATTEMPTS ($MODE): $BIN ${ARGS[*]}"
  "$BIN" "${ARGS[@]}" 2>&1 | ts >>"$LOG"
  rc=${PIPESTATUS[0]}
  last_line=$(tail -n 1 "$LOG")
  log "attempt $attempt exited rc=$rc"

  if [ "$rc" -eq 0 ]; then
    if [ "$MODE" = dry ]; then
      log "dry run: gates passed; nothing published, no $DONE written"
      exit 0
    fi
    # ----------------------------------------------------- success (4)
    {
      printf '[%s] SUCCESS after attempt %s\n' "$(now)" "$attempt"
      printf 'pack: %s\n' "$OUT"
      printf 'progress: %s\n' "$WORK/progress.json"
      if [ -f "$WORK/progress.json" ]; then
        jq -r '
          "layers complete : \([.layers[].i] | length) of 64",
          "layers recorded : \([.layers[].i]
             | reduce .[] as $x ([];
                 if (length > 0 and ($x == (.[-1][-1] + 1)))
                 then (.[-1] += [$x]) else . += [[$x]] end)
             | map(if length == 1 then "\(.[0])" else "\(.[0])..\(.[-1])" end)
             | join(", "))",
          "head            : \(if .head == null then "absent" else "recorded (\(.head[0:12])...)" end)",
          "params          : bits=\(.bits) head-bits=\(.head_bits) rows=\(.rows) cols=\(.cols) seed=\(.seed) lambda=\(.lambda) scale=\(.scale)",
          "calib sha256    : \(.calib_sha)"
        ' "$WORK/progress.json" 2>&1 || printf 'progress summary unavailable (jq failed)\n'
      else
        printf 'no progress.json in %s\n' "$WORK"
      fi
      printf 'log: %s\n' "$LOG"
    } >"$DONE"
    log "success: $DONE written"
    exit 0
  fi

  # Nonzero: is this the driver's own abort (bug) or a crash (resume-worthy)?
  if printf '%s\n' "$last_line" | grep -Eq "$ABORT_RE"; then
    write_abort_failed "attempt $attempt" "$rc" "$last_line"
  fi

  if [ "$attempt" -eq "$ATTEMPTS" ]; then
    log "attempts exhausted after $ATTEMPTS crashes without the gates passing"
    {
      printf 'attempts exhausted: last exit %s at %s\n' "$rc" "$(now)"
      printf '# last log line:\n%s\n' "$last_line"
      printf '# full log: %s\n' "$LOG"
    } >"$FAIL"
    log "wrote $FAIL"
    exit 2
  fi
  log "crash (not an abort signature) — resuming with attempt $((attempt + 1))"
  attempt=$((attempt + 1))
done
