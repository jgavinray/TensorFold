#!/usr/bin/env python3
"""Build the held-out KL-gate evaluation prompts (eaq-ldlq-spec-fable.md §5.2).

Sixteen 2048-token rows from the calibration corpus, disjoint from every
calibration window, plus four uniform-random rows reported separately. The
calibration stream (calib-200k.manifest.json) consumed whole files, so any
file NOT in that manifest is disjoint by construction; we replay the
calibration tool's enumeration, tokenizer, truncation and per-file cap over
the remaining files and carve rows from the resulting stream the same way the
driver framed the calibration rows.

The corpus enumeration, model/tokenizer path and caps below are inlined from
calib_tokenize.py, the calibration tool that produced calib-200k.manifest.json.
That tool is not shipped in this repository, so the shared logic is reproduced
here verbatim (same roots, filters, truncation, caps and manifest join) and the
ids are unchanged; this script now runs from the repository alone, given the
corpus and the mlx-ref venv.

Run on hyper03 with the mlx-ref venv (the only environment with `tokenizers`):

    /storage01/venvs/mlx-ref/bin/python zig/tools/xpu/calib_evalprompts.py

Writes ~/xpu-eaq/eval/eval.i32 (little-endian i32 ids, 20 rows x 2048) and
eval.manifest.json (content_sha256 + per-row kind/domain), prints both.
"""

import hashlib
import json
import pathlib
from array import array

# Inlined verbatim from calib_tokenize.py (the calibration tool, not shipped here).
MODEL = "/storage01/models/Qwen3.8-27B"
ROOTS = [
    "/storage01/qwen-flash-on-intel",
    "/storage01/qwen-for-intel",
    "/storage01/tensorfold",
    "/storage01/models",
]
CALIB = pathlib.Path.home() / "xpu-eaq"
VOCAB = 248_320  # the runner's VOCAB (qwen27.zig); ids must stay below it
SKIP_PARTS = {"node_modules", ".git", ".cache", "venv", "venvs", "__pycache__"}
SKIP_NAMES = {"merges.txt", "vocab.json", "tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json", "crc32.txt"}


def corpus_files():
    fs = []
    for root in ROOTS:
        p = pathlib.Path(root)
        if not p.is_dir():
            continue
        for f in p.rglob("*"):
            if f.suffix not in (".md", ".txt", ".rst") or not f.is_file():
                continue
            name = f.name.lower()
            if f.name in SKIP_NAMES or any(bad in name for bad in ("merges", "vocab", "license", "notice", "checksum", "changes", "contributing", "security")):
                continue
            if any(part.startswith(".") for part in f.relative_to(p).parts):
                continue
            if any(part in SKIP_PARTS for part in f.parts):
                continue
            if f.stat().st_size > 4_000_000:  # no monster dumps; many modest docs read better
                continue
            fs.append(f)
    return sorted(fs)


OUT = CALIB / "eval"
ROWS = 16          # held-out rows (§5.2)
RANDOM_ROWS = 4    # random rows, reported separately
COLS = 2048        # tokens per row (the serving context)
RAND_SEED = 0x5EAF00D


def classify(path: str) -> str:
    p = path.lower()
    if p.startswith(MODEL) or "/models/" in p:
        return "modelcard"
    if "/exp/workers/" in p:
        return "logs"
    if "/docs/" in p or p.endswith(".rst") or "/spec" in p:
        return "technical"
    if "/readme" in p:
        return "prose"
    return "other"


def main() -> None:
    from tokenizers import Tokenizer  # mlx-ref venv only (calib's precondition)

    tok = Tokenizer.from_file(str(MODEL + "/tokenizer.json"))
    m = json.loads((CALIB / "calib-200k.manifest.json").read_text())
    used = {f["path"] for f in m["files"]}

    stream: list[int] = []
    row_sources: dict[int, list[str]] = {}
    for f in corpus_files():
        if str(f) in used:
            continue  # disjoint from every calibration window
        if len(stream) >= ROWS * COLS:
            break
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        enc = tok.encode(text[:24_000], add_special_tokens=False)
        if len(enc.ids) < 16:
            continue
        take = enc.ids[:4096]  # calib's per-file cap, same rule
        start = len(stream)
        stream.extend(take)
        for r in range(start // COLS, (len(stream) + COLS - 1) // COLS):
            row_sources.setdefault(r, []).append(str(f))
    need = ROWS * COLS
    if len(stream) < need:
        raise SystemExit(f"only {len(stream)} held-out tokens available, need {need}")

    rng = 0x5EAF00D0BBAADF00  # splitmix64, deterministic like the rest of the harness

    def rand_id() -> int:
        nonlocal rng
        rng = (rng + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = rng
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return (z ^ (z >> 31)) % VOCAB

    ids = stream[:need] + [rand_id() for _ in range(RANDOM_ROWS * COLS)]
    assert all(0 <= t < VOCAB for t in ids)

    OUT.mkdir(parents=True, exist_ok=True)
    a = array("i", ids)
    if a.itemsize != 4:
        raise SystemExit("array('i') is not 4 bytes here")
    blob = a.tobytes()  # little-endian on every host we run this on (x86/ARM)
    (OUT / "eval.i32").write_bytes(blob)
    sha = hashlib.sha256(blob).hexdigest()

    rows = []
    for r in range(ROWS):
        srcs = row_sources.get(r, [])
        doms = [classify(s) for s in srcs]
        domain = max(set(doms), key=doms.count) if doms else "other"
        rows.append({"row": r, "kind": "heldout", "domain": domain, "sources": srcs})
    for r in range(ROWS, ROWS + RANDOM_ROWS):
        rows.append({"row": r, "kind": "random", "domain": "random"})

    manifest = {
        "created_by": "calib_evalprompts.py",
        "model_dir": MODEL,
        "tokenizer": m["tokenizer"],
        "cols": COLS,
        "rows_total": ROWS + RANDOM_ROWS,
        "heldout_rows": ROWS,
        "random_rows": RANDOM_ROWS,
        "rand_seed": RAND_SEED,
        "vocab_ref": VOCAB,
        "content_sha256": sha,
        "disjoint_note": "held-out files are the corpus_files() enumeration minus calib-200k.manifest.json files; the calibration stream consumed whole files, so no token overlaps",
        "rows": rows,
    }
    (OUT / "eval.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"eval.i32: {(ROWS + RANDOM_ROWS) * COLS} ids, sha256 {sha}")
    print(f"wrote {OUT / 'eval.i32'} ({len(blob)} B), {OUT / 'eval.manifest.json'}")
    for r in rows:
        if r["kind"] == "heldout":
            print(f"  row {r['row']:2d} {r['domain']:10s} {', '.join(s.split('/')[-2] + '/' + s.split('/')[-1] for s in r['sources'])}")


if __name__ == "__main__":
    main()
