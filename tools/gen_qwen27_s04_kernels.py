#!/usr/bin/env python3
"""Generate the S04 (27B decode) kernel sources that are textual variants of an existing one.

Why a generator: the two S04 kernels differ from their proven parents by one scalar (a row stride
or an output width) and by nothing else - same tiling, same fp32 accumulation, same single bf16
rounding store, same reduction tree. Hand-copying a kernel to change one `#define` invites a silent
drift (the 27B `attention_gate` bug came from exactly that). Here the variant is defined by the
replacement list, so `diff parent variant` shows only the intended constants.

Usage: python3 tools/gen_qwen27_s04_kernels.py [--write]   (default: print what would change)
"""
import difflib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHARED = ROOT / "zig" / "kernels" / "xpu" / "shared"

# (parent key, variant key, [(old, new), ...])
VARIANTS = [
    (
        "layout",
        "layout27",
        [
            (
                "#define QW_QGATE_STRIDE (2 * QW_HEADS * QW_HEAD_DIM) // q_proj row stride: query + gate per head",
                "#define QW_QGATE_STRIDE (2 * QW_HEADS * QW_HEAD_DIM) // q_proj row stride: query + gate per head\n"
                "#ifndef QW_QROWS\n"
                "// Head stride of the `queries` buffer: 0 means \"read dims[1]\", the round's row count. A nonzero\n"
                "// value overrides that stride at compile time; it must be the round's row count, never a model\n"
                "// width, or the kernel writes past the buffer.\n"
                "#define QW_QROWS 0\n"
                "#endif",
            ),
            # `queries` writes at out[(h*dims[1] + m)*D + d]. The 2B geometry passed dims[1] = T, which is the
            # 27B's own bug: the buffer is strided by the whole round's rows (QW_QROWS), and the segment's base
            # sits in dims[2]. QW_QROWS = 0 keeps the parent's behaviour bit for bit (the proven 2B gate).
            (
                "out[((size_t)h * dims[1] + (size_t)m) * QW_HEAD_DIM + (size_t)d] = rope_rotate(src, d, dims[0] + r);",
                "out[((size_t)h * (QW_QROWS ? QW_QROWS : dims[1]) + (size_t)m) * QW_HEAD_DIM + (size_t)d] = rope_rotate(src, d, dims[0] + r);",
            ),
        ],
    ),
    (
        "attn_merge",
        "attn_merge27",
        [
            (
                "#define D    QW_HEAD_DIM",
                "#ifndef QW_OUT_WIDTH\n"
                "// OUT's per-row stride in heads, not in elements: the index below is (w*QW_OUT_WIDTH + qh) * D,\n"
                "// so the leading factor is the head count. The parent hard-codes QW_Q_HEADS there; a variant that\n"
                "// set this to heads*head_dim would multiply by D twice and run 6144x past the buffer.\n"
                "#define QW_OUT_WIDTH QW_Q_HEADS\n"
                "#endif\n\n"
                "#define D    QW_HEAD_DIM",
            ),
            # attn_merge.cl writes at ((w * QW_Q_HEADS) + qh) * D; the 27B's mix buffer is strided by hidden
            # (5120), not by heads*head_dim (6144). One scalar, no arithmetic change.
            (
                "const size_t o = ((size_t)w * QW_Q_HEADS + (size_t)qh) * D + (size_t)(d0 * DPL);",
                "const size_t o = ((size_t)w * QW_OUT_WIDTH + (size_t)qh) * D + (size_t)(d0 * DPL);",
            ),
        ],
    ),
]


def variant_source(parent: str, repls) -> str:
    text = (SHARED / parent / f"{parent}.cl").read_text()
    for old, new in repls:
        if text.count(old) != 1:
            raise SystemExit(f"{parent}: pattern appears {text.count(old)} times, expected once:\n{old}")
        text = text.replace(old, new)
    return text


def main() -> int:
    write = "--write" in sys.argv
    for parent, key, repls in VARIANTS:
        src = variant_source(parent, repls)
        out = SHARED / key / f"{key}.cl"
        if write:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(src)
            print(f"wrote {out.relative_to(ROOT)}")
        else:
            cur = out.read_text() if out.exists() else ""
            diff = list(
                difflib.unified_diff(
                    cur.splitlines(), src.splitlines(), fromfile=f"{key}.cl (on disk)", tofile=f"{key}.cl (generated)", lineterm=""
                )
            )
            print("\n".join(diff) if diff else f"{out.relative_to(ROOT)} is up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
