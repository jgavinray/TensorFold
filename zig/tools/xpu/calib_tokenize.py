#!/usr/bin/env python3
"""Build the EAQ calibration corpus for the Qwen3.8-27B runner on hyper03 (no torch).

Tokenizer: the model's own HF fast tokenizer, /storage01/models/Qwen3.8-27B/tokenizer.json,
loaded with the `tokenizers` library of /storage01/venvs/mlx-ref (0.23.3). No torch, no
transformers, no network.

Corpus: the sorted markdown/txt/rst prose of the serving repos and model cards on
/storage01 (local text; deterministic order). Raw text until TARGET ids.

Outputs (identical content, two containers):
  ~/xpu-eaq/calib-200k.ids.txt      space-separated decimal ids — exactly what the runner's
                                    prompt-file parser accepts (qwen27_main.zig tokenizeAny " \\n,[]")
  ~/xpu-eaq/calib-200k.ids.i32.bin  the same ids as raw little-endian i32 (a cast/mmap away for the quantizer)
  ~/xpu-eaq/calib-200k.manifest.json provenance: file list, sizes, token counts, tokenizer hash

Usage: /storage01/venvs/mlx-ref/bin/python calib_tokenize.py
"""
import hashlib
import json
import pathlib
from array import array

import tokenizers  # noqa: F401  (importability is the precondition; checked below)
from tokenizers import Tokenizer

MODEL = "/storage01/models/Qwen3.8-27B"
ROOTS = [
    "/storage01/qwen-flash-on-intel",
    "/storage01/qwen-for-intel",
    "/storage01/tensorfold",
    "/storage01/models",
]
OUT = pathlib.Path.home() / "xpu-eaq"
TARGET = 204_800  # ids: >= 200k (exllamav3's order of magnitude), a multiple of the GDN's 64-row chunks
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


def main():
    tok_path = pathlib.Path(MODEL) / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    tok_sha = hashlib.sha256(tok_path.read_bytes()).hexdigest()
    ids, used = [], []
    for f in corpus_files():
        if len(ids) >= TARGET:
            break
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        enc = tok.encode(text[:24_000], add_special_tokens=False)  # the runner frames segments itself
        if len(enc.ids) < 16:
            continue
        take = enc.ids[:4096]  # per-file cap: the corpus spans many documents, not one monster
        ids.extend(take)
        used.append({"path": str(f), "bytes": min(len(text.encode()), 24_000), "tokens": len(take)})
    if len(ids) < TARGET:
        raise SystemExit(f"corpus ran out: {len(ids)} ids < {TARGET}")
    ids = ids[:TARGET]
    vmax, vmin = max(ids), min(ids)
    if vmax >= VOCAB or vmin < 0:
        raise SystemExit(f"id out of range: {vmin}..{vmax} vs vocab {VOCAB}")
    out_txt = OUT / "calib-200k.ids.txt"
    out_bin = OUT / "calib-200k.ids.i32.bin"
    out_txt.write_text(" ".join(str(t) for t in ids))
    a = array("i", ids)
    if a.itemsize != 4:
        raise SystemExit("unexpected int size")
    import sys
    if sys.byteorder == "big":
        a.byteswap()
    out_bin.write_bytes(a.tobytes())
    manifest = {
        "created_by": "calib_tokenize.py",
        "model_dir": MODEL,
        "tokenizer": {"path": str(tok_path), "sha256": tok_sha, "tokenizers_lib": tokenizers.__version__},
        "target_ids": TARGET,
        "ids": len(ids),
        "id_min": vmin,
        "id_max": vmax,
        "vocab_ref": VOCAB,
        "files_used": len(used),
        "raw_bytes": sum(u["bytes"] for u in used),
        "files": used,
        "note": "ids[i] in [0, VOCAB); txt and bin carry the same sequence; segments are framed by the driver (add_special_tokens was False)",
    }
    (OUT / "calib-200k.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"ids={len(ids)} range={vmin}..{vmax} files={len(used)} raw_bytes={manifest['raw_bytes']}")
    print(f"wrote {out_txt} ({out_txt.stat().st_size} B), {out_bin} ({out_bin.stat().st_size} B)")


if __name__ == "__main__":
    main()
