"""Flash Next on Intel GPUs: the NVFP4 checkpoint, eager decode windows, and streamed routed experts.

``xpu_engine()`` is the family's serving entry: resident dense, attention, GDN and PLE staging with
the routed experts streaming through ``xpu/stream.py``'s slot pool when ``ssd_experts`` gives them a
GiB budget (a 126 GB checkpoint's experts do not fit a 32 GB card). MTP chains light up when the
head's kernels join the build; until then the engine serves the serial reference exactly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .weights import load    # noqa: F401 (re-exported: the slice-1 entry point)

__all__ = ["load", "xpu_engine"]


def xpu_engine(model_dir: str | Path, *, tp: int = 1, no_drafts: bool = False,
               mtp_drafts: int | None = None, context: int | None = None,
               ple_on_ssd: bool = False, kv_dtype: str = "bf16", ssd_experts: float | None = None,
               draft_vocab: int | str | None = None, capacity: int = 4096, max_rows: int = 8,
               prefill_rows: int | None = None, graphs: bool = False, device: str = "xpu",
               **options: Any):
    """Serve ``model_dir`` on one Intel GPU: weights resident, routed experts streamed at ``ssd_experts`` GiB."""

    from torch import xpu as _xpu

    if int(tp) != 1:
        raise ValueError("the XPU engine serves one GPU; tensor parallel ranks are a later slice")
    if not _xpu.is_available():
        raise RuntimeError("no Intel GPU visible: serve with the CUDA engine, or install the oneAPI stack")
    from tensorfold.families.qwen4_exp import CUDA_KV_DTYPES, has_mtp

    if kv_dtype not in CUDA_KV_DTYPES:
        raise ValueError(f"kv-dtype {kv_dtype!r}: this engine serves {' or '.join(CUDA_KV_DTYPES)}")
    want_mtp = has_mtp(Path(model_dir)) and not no_drafts and (mtp_drafts is None or int(mtp_drafts) > 0)
    from .engine import Engine
    from .engine import PREFILL_ROWS

    if ssd_experts is not None:
        from .stream import load as load_streamed

        w, _streamer = load_streamed(model_dir, device, mtp=want_mtp, gib=float(ssd_experts),
                                     draft_vocab=draft_vocab, ple_on_ssd=ple_on_ssd)
    else:
        w = load(model_dir, device, mtp=want_mtp, draft_vocab=draft_vocab, ple_on_ssd=ple_on_ssd)
    return Engine(w, capacity=int(capacity if context is None else min(int(context), capacity)),
                  max_rows=int(max_rows),
                  prefill_rows=int(prefill_rows) if prefill_rows else PREFILL_ROWS,
                  graphs=bool(graphs), kv_dtype=kv_dtype)
