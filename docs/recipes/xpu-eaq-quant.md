# Fresh EAQ quantization of Qwen3.8-27B on the B70

`xpu-eaq-quant` builds an EAQ pack from a bf16 source directory on the Intel B70 alone: calibration
capture, per-layer Hessians, block LDL, block LDLQ, the trellis encoder, fragments, and the pack
writer. `xpu-eaq-kl` scores the result against reference logits from `xpu-eaq-ref`. This recipe is the
repeatable process, end to end. The container is read by `zig/src/xpu/exl3_eaq.zig`; the writer's
acceptance rules are in `zig/src/core/eaq_write.zig`; the KL statistic is in `zig/src/xpu/kl.zig`.

## What a run produces

- The EAQ pack directory: `config.json` with the merged `eaq` block, sharded safetensors with `eaq.*`
  entries, `eaq_manifest.json`. Publication is last-to-first — shards, then the manifest, then
  `EAQ_COMPLETE.json`; a pack without the marker is incomplete.
- `progress.json` in the work directory: one entry per completed layer plus the head. A rerun with
  `--resume` skips what it records, so a crashed run continues from its last layer.
- The quant report (§Q3): per-part halves, achieved bpw, counters, the failed-tile and Cholesky
  evidence. Its sha256 is recorded in both published shards' `__metadata__` and in the manifest, so a
  hand-edited pack refuses at open.

## Prerequisites

- Build on the card host:

  ```sh
  zig build xpu-eaq-quant xpu-eaq-kl xpu-eaq-ref \
      -Doptimize=ReleaseFast -Dxpu-device=bmg -Dngen-include=$HOME/xpu-eaq/ngen
  ```

- A bf16 source directory (`/storage01/models/Qwen3.8-27B`), a calibration ids file, an eval prompts
  file with its manifest, at least 120 GB free on the work filesystem, and the card dedicated to the
  run (the run script stops `tf-main-8081` and restarts it on every exit path).
- Host gates for the machinery: `zig build test-xpu-eaq test-exl3-encode test-exl3-ldlq
  test-exl3-halfstep xpu-eaq-quant-tests`.

## 1. Corpus

- `zig/tools/xpu/calib_tokenize.py` tokenizes the local prose corpus into `calib-200k.ids.txt`
  (204,800 ids: 100 rows × 2,048) and writes its manifest. Run it with the `mlx-ref` venv, the only
  environment with the `tokenizers` library.
- `zig/tools/xpu/calib_evalprompts.py` carves the held-out KL rows (`eval/eval.i32`: 20 rows × 2,048,
  16 disjoint from every calibration window, 4 uniform-random reported separately) and writes
  `eval.manifest.json` with the `content_sha256` the run script checks.

## 2. Gates only (dry run)

```sh
zig/tools/xpu/run-quant-full.sh --dry-run
```

The pre-flight (calibration parse, eval sha256, disk, driver `--help`) and the driver's own
`--dry-run --layers 0..0`. Publishes nothing and records no progress.

## 3. The run

```sh
zig/tools/xpu/run-quant-full.sh
```

Up to five attempts, every line timestamped. A work directory that carries `progress.json` resumes;
a fresh one runs without `--resume` (the driver refuses that combination as a usage error, `--seed`
providing the key). Exit codes: 0 success, 1 abort (failed tiles or Cholesky — a deterministic
quantizer bug, never retried; the failed `H` sidecar is dumped for inspection), 2 attempts exhausted,
3 the serving unit would not stop, 4 pre-flight or plan refusal, 69 Level Zero cannot open,
75 the device is short or the lease is busy.

The device gate runs before the first attempt: `--device-check` prints the Level Zero free bytes
against this run's estimate; when short, a root `fdinfo` scan names every DRM client on the card
(`drm-client-id`, `drm-total-vram0`, `drm-resident-vram0` per pid). A live holder is a refusal
(exit 75 — nothing here kills a live process); no accountable client means kernel reclaim is pending,
so it polls in 30 s steps up to `DEVICE_POLL_SECS` (600 by default) and refuses with the numbers and
a dmesg hint. A lease (`/tmp/xpu-eaq-quant.lease`, `O_CREAT` + `flock`, pid/start/work/need) keeps two
runs from colliding. `InsufficientDeviceMemory` is treated as transient: 60 s backoff, up to
`DEVICE_TRIES` (8) retries, without consuming the five crash attempts.

Geometry: the encoder's strip loop runs at count=32 (measured 604.7 tiles/s on a real part, ~51 % of
the free-running batched kernel); the LDLQ update kernel's cost is under the noise floor (≤1.3 %) and
needs no tuning.

The probe's G2 block band is width-dependent by design: the o_proj/out_proj head-strip family is a
16-row-per-column strip whose error runs systematically higher than the wide parts (family mean
0.151, sigma 0.017 across the measured series), so it is graded on [0.10, 0.20] with a hard red
above 0.20 and a joint amber at corr < 0.99 with block RMS > 0.18, while wide parts keep
[0.10, 0.16] and their per-column band. A joint amber is not a stop: it flags the family for the
pack's self-check (ruling 2026-10-27).

Plan mode:

```sh
zig/tools/xpu/run-quant-full.sh --plan plan-2.9.json
```

`zig run zig/src/core/exl3_alloc_cli.zig -- 2.9 plan-2.9.json` emits the plan (method
`eaq-alloc/v1`, target bpw, head halves, 401 parts, embedded `plan_sha`); the driver refuses a
hand-edited plan. A `--plan` run first takes a one-shot
`--dry-run --layers 0..0 --plan` acceptance pass, then publishes. Plans with odd halves (2.9 / 3.9)
need the half-step serving path to run the result; a uniform even-width pack (3.0) does not.

## 4. Validate

- Open and serve: `tf-xpu-qwen27 PACK ids N`. The load table must show every group's halves, then the
  KL row.
- KL gate:

  ```sh
  xpu-eaq-kl PACK --prompts eval/eval.i32 --ref ref_logits.f32 --report kl.json
  ```

  The reference comes from the dense bf16 forward:

  ```sh
  xpu-eaq-ref SRC --prompts eval/eval.i32 --out ref_logits.f32
  ```

  layer-major `[positions][VOCAB]` fp32. `xpu-eaq-kl --self` scores the pack against itself, which
  checks the instrument before the pack.

## 5. The cheaper route: repack an existing EXL3 pack

`tf-eaq-repack SRC DST` (build step `eaq-repack`) wraps a stock EXL3 pack's trellis tensors in the
EAQ container: payloads verbatim, `model.visual.*` dropped, the `eaq` block merged into `config.json`,
manifest then marker. No new quantization math. The 2.0bpw pack repacked this way opens with all 496
projections mapped and serves through the container path.
