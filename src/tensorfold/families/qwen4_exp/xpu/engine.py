"""The Flash Next XPU engine: weights, one sequence's committed state, and the buffers around them.

The serving surface of ``qwen4_exp`` on Intel GPUs: the decode-window, MTP-head and prompt-chunk
buffers and the serial state from ``xpu/state.py``, driven by ``xpu/forward.py``'s eager path. It
mirrors ``cuda/decode.py``'s ``Engine``; MTP lights up when the head's weights arrive in the build
(``w.mtp`` non-None), and ``w.streamer`` (``xpu/stream.py``) routes the routed experts through the
host-mapped checkpoint's slot pool."""

from __future__ import annotations

from typing import Sequence

import torch

from .forward import forward
from .state import Buffers, State

DEPTH = 6            # most MTP drafts a round
CONFIDENCE = 0.7     # a chain ends before a draft the MTP head gives less than this
PREFILL_ROWS = 2048  # rows of a prompt chunk
GDN_CHUNK = 64        # the DeltaNet chain's captured width: the widest prompt chunk this engine issues


def entry_end(prompt: Sequence[int]) -> int:
    """Where a prompt's kept state ends: one token early, since a next turn sent back without its reasoning renders ``<|im_start|>`` and two newlines there."""

    return max(1, len(prompt) - 1)


def _sync(x: torch.Tensor | None = None) -> None:
    """Block until the device has drained (an engine call returns once every kernel is enqueued)."""

    torch.xpu.synchronize()


class Engine:
    """Weights, one sequence's state, buffers for decode windows (main model and MTP head) and for prompt chunks.

    ``w.streamer`` (``xpu/stream.py``'s ``attach`` or ``load``) routes the routed experts through the
    slot pool; the engine alone registers the ``pick`` scratch that routing reads. Graphs: the NVFP4
    grouped path reads its item list on the host, which a capture rejects, so decode runs eagerly —
    the GDN chunks keep their own captures inside ``xpu/gdn.py`` either way."""

    def __init__(self, w, *, capacity: int = 4096, max_rows: int = 8, prefill_rows: int = PREFILL_ROWS,
                 graphs: bool = False, kv_dtype: str = "bf16") -> None:
        if getattr(w, "comm", None) is not None:
            raise ValueError("the XPU engine serves tensor parallelism of 1; multi-rank decode is a later slice")
        self.w = w
        self.capacity = capacity
        self.rows, self.prefill_rows = max_rows, prefill_rows
        self.kv_dtype = kv_dtype
        self.buf = Buffers(w, max_rows, capacity, moe_prefill=True)       # the experts' prefill arithmetic, shared
        self.mbuf = Buffers(w, max_rows, capacity) if w.mtp is not None else None
        self.pbuf = Buffers(w, prefill_rows, capacity, prefill=True)
        self.st = State(w, capacity, max(max_rows, min(prefill_rows, GDN_CHUNK)), kv_dtype)
        self.streamer = getattr(w, "streamer", None)
        if self.streamer is not None:
            self.streamer.attach_buffers(self.buf.moe, self.pbuf.moe)
        self.graphs = None
        self._graphs_declined = not graphs
        if graphs:
            self._decline_graphs()
        self.kept: dict | None = None
        self.first: int | None = None
        self.last_streams: torch.Tensor | None = None

    def _decline_graphs(self) -> None:
        if not self._graphs_declined:
            print("[tensorfold] XPU graphs off: the NVFP4 experts read their plan's item list on the host, "
                  "which a capture rejects; decode runs eagerly (the GDN chunks keep their own captures)")
            self._graphs_declined = True

    def reset(self) -> None:
        self.st.reset(self.w)

    def twin(self) -> "Engine":
        """Share weights and scratch with an independent serial state, without MTP or graphs; requests run sequentially so scratch reuse leaves this engine's state and prefix cache intact."""

        other = object.__new__(Engine)
        other.w, other.capacity, other.rows, other.prefill_rows = self.w, self.capacity, self.rows, self.prefill_rows
        other.buf, other.mbuf, other.pbuf, other.graphs = self.buf, None, self.pbuf, None
        other.kv_dtype, other.streamer = self.kv_dtype, self.streamer
        other._graphs_declined = True
        other.st = State(self.w, self.capacity, max(self.rows, min(self.prefill_rows, GDN_CHUNK)),
                         self.st.kv_dtype)
        other.kept, other.first, other.last_streams = None, None, None
        return other

    def forward(self, tokens: Sequence[int]) -> torch.Tensor:
        """A decode step's forward: logits [R, V]. Streamed experts answer this window's routing."""

        if self.streamer is not None:
            self.streamer.begin("rows", len(tokens), self.buf.moe)
        return forward(self.w, self.st, self.buf, tokens)

    def sample(self, logits: torch.Tensor, positions: Sequence[int], sampling, *,
               draft: bool = False, gathered: bool = True) -> list[int]:
        """Rows of logits at their positions -> tokens (``draft``: the MTP head's over its mapped vocabulary)."""

        if getattr(self.w, "comm", None) is not None:
            raise NotImplementedError("gathered candidate sampling needs a comm; the XPU engine serves TP=1")
        if draft and self.w.draft_ids is not None:
            from .decode import sample_mapped

            return sample_mapped(logits, positions, sampling, self.w.draft_ids)
        from tensorfold.cuda.sampling import sample_rows

        return sample_rows(logits, positions, sampling)

    def sample_draft(self, logits: torch.Tensor, position: int, sampling) -> tuple[int, float]:
        """The position's keyed draft and its temperature-1 probability; this confidence ends chains early without changing output."""

        import numpy as np

        from tensorfold.engine.exact_sampling import MARGIN, choose_rows

        w = self.w
        if getattr(w, "comm", None) is not None:
            raise NotImplementedError("gathered candidate sampling needs a comm; the XPU engine serves TP=1")
        mapped = w.draft_ids is not None
        if mapped and getattr(self, "_draft_host", None) is None:
            self._draft_host = w.draft_ids.cpu().numpy()
        row = logits[:1].float()
        lse = torch.logsumexp(row, dim=-1, keepdim=True)
        if sampling is None or sampling.temperature <= 0:
            top, col = row.max(dim=-1, keepdim=True)            # the first maximum: argmax's (and serial's) choice
            got = torch.cat([top, lse, col.float()], dim=1).cpu().numpy()[0]       # one sync
            c = int(got[2])
            tok = int(self._draft_host[c]) if mapped else c
            return tok, float(np.exp(float(got[0]) - float(got[1])))
        k = min(row.shape[1], int(sampling.top_k) + MARGIN) if sampling.top_k else row.shape[1]
        vals, idx = torch.topk(row, k, dim=-1, sorted=False)
        got = torch.cat([vals, lse, idx.float()], dim=1).cpu().numpy()[0]          # one sync
        cols = got[k + 1:].astype(np.int64)
        ids = self._draft_host[cols] if mapped else cols
        tok = choose_rows(got[None, :k].astype(np.float32), ids[None, :], [position], sampling)[0]
        hit = np.nonzero(ids == tok)[0]
        return int(tok), float(np.exp(float(got[hit[0]]) - float(got[k]))) if len(hit) else 0.0

    def mtp_forward(self, next_tokens: Sequence[int], streams: torch.Tensor) -> torch.Tensor:
        """The MTP head's window over ``streams``, once the head's kernels join the build."""

        if self.w.mtp is None or self.mbuf is None:
            raise RuntimeError("this engine has no MTP head")
        from .mtp import mtp_forward

        return mtp_forward(self.w, self.st, self.mbuf, next_tokens, streams)


__all__ = ["CONFIDENCE", "DEPTH", "Engine", "PREFILL_ROWS", "entry_end"]
