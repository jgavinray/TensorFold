#!/bin/bash
# KlFixRef card chain: holds until the CARD-FREE marker lands (QuantFix owns the
# card until then), then runs, in order:
#   1) xpu-eaq-kl --probe            (A/B dump + fp64 oracle, m=1024 and m=128)
#   2) xpu-eaq-kl --self             (stock 2.0bpw pack self-check: KL 0 / top-1 100%)
#   3) xpu-eaq-ref canary            (2 rows, bisect + canary lines)
#   4) xpu-eaq-ref full reference    (20 rows -> /storage01/ref_logits.f32 + sidecar)
# Every stage's exit code is recorded; the chain stops at the first failure except
# that the canary gates the full dump (a red canary must not burn 40 GB of card time).
cd /home/jgavinray/xpu-eaq || exit 9
# the root fs is full (192K free): binaries, logs, spot dumps and the canary all
# live on tmpfs; only the 40 GB reference lands on its own volume.
BIN=/tmp/klfixbin/bin
W=/tmp/klfixwork
PACK=$PWD/Qwen3.8-27B-EAQ-2.0bpw
SRC=/storage01/models/Qwen3.8-27B
# the ruling's one-run diagnostic: stream and slab hashes ride the ref runs
export XPU_REF_SPOT=1
# the spot prefixes land as bytes too: a hash names the stage, the bytes name the bug
export XPU_REF_SPOT_FILES=$W/spot
mkdir -p $W/spot $W/resid
[ -f klfix/CHAIN-STARTED ] && exit 1   # one shot
touch klfix/CHAIN-STARTED

echo "chain: waiting for CARD-FREE $(date +%H:%M:%S)" >> "$W/klfix-chain.log"
until [ -f klfix/CARD-FREE ]; do sleep 15; done
echo "chain: CARD-FREE seen $(date +%H:%M:%S)" >> "$W/klfix-chain.log"

# 1) A/B probe (its own runner init; nothing else may hold the card)
XPU_KL_SPOT_DUMP=$W/ab.bin "$BIN/xpu-eaq-kl" "$PACK" --prompts eval/eval.i32 --probe \
  > "$W/klfix-probe.log" 2>&1
echo "PROBE-EXIT=$? $(date +%H:%M:%S)" >> "$W/klfix-chain.log"
tail -3 "$W/klfix-probe.log" >> "$W/klfix-chain.log"

# 2) self-check on the stock pack
"$BIN/xpu-eaq-kl" "$PACK" --prompts eval/eval.i32 --self --report "$W/klfix-self.json" \
  > "$W/klfix-self.log" 2>&1
echo "SELF-EXIT=$? $(date +%H:%M:%S)" >> "$W/klfix-chain.log"
tail -6 "$W/klfix-self.log" >> "$W/klfix-chain.log"

# 3) ref canary: 2 rows, residual bisect on
"$BIN/xpu-eaq-ref" "$SRC" --prompts eval/refcanary2.i32 --out "$W/canary2.f32" \
  --residual-dump "$W/resid" > "$W/klfix-canary.log" 2>&1
CE=$?
echo "CANARY-EXIT=$CE $(date +%H:%M:%S)" >> "$W/klfix-chain.log"
tail -14 "$W/klfix-canary.log" >> "$W/klfix-chain.log"
if [ $CE -ne 0 ]; then
  echo "chain: canary RED, full dump NOT attempted $(date +%H:%M:%S)" >> "$W/klfix-chain.log"
  touch klfix/CHAIN-DONE
  exit $CE
fi

# 4) the full reference dump
"$BIN/xpu-eaq-ref" "$SRC" --prompts eval/eval.i32 --out /storage01/ref_logits.f32 \
  > "$W/klfix-refdump.log" 2>&1
RE=$?
echo "REFDUMP-EXIT=$RE $(date +%H:%M:%S)" >> "$W/klfix-chain.log"
tail -16 "$W/klfix-refdump.log" >> "$W/klfix-chain.log"
sha256sum /storage01/ref_logits.f32 >> "$W/klfix-chain.log" 2>&1
touch klfix/CHAIN-DONE
echo "chain: done $(date +%H:%M:%S)" >> "$W/klfix-chain.log"
