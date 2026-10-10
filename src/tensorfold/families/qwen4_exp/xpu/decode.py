"""Decode one sequence on Intel GPUs: prompt chunks, serial steps, and the MTP verify window once the head lands.

The torch port of ``cuda/decode.py``'s serving loop without the multi-rank paths (the XPU engine is
TP=1): the same keyed sampling as serial decoding, the same verify rule that commits only rows before
the first mismatched draft, and the streamed experts' per-pass ``begin`` on the window. Draft and
absorb call sites stay guarded on ``w.mtp``: they raise a clear error until the head's kernels merge."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows

from .engine import CONFIDENCE, DEPTH, GDN_CHUNK, PREFILL_ROWS, Engine, _sync, entry_end
from .forward import Cut, commit, cut_snapshot, forward

__all__ = ["CONFIDENCE", "DEPTH", "PREFILL_ROWS", "DecodeResult", "Engine", "absorb", "draft",
           "entry_end", "mtp_decode", "prefill", "prefill_begin", "prefill_chunk", "sample_mapped",
           "serial_decode", "warm"]


def sample_mapped(logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None,
                  id_map: torch.Tensor) -> list[int]:
    """Rows of logits over a token subset (column j is token id_map[j]) -> tokens, with the keyed rule on the real ids (the draft head over the draft vocabulary)."""

    if sampling is None or sampling.temperature <= 0:
        return [int(t) for t in id_map[logits.argmax(dim=-1)].cpu().tolist()]
    k = min(logits.shape[1], int(sampling.top_k) + MARGIN) if sampling.top_k else logits.shape[1]
    vals, idx = torch.topk(logits.float(), k, dim=-1, sorted=False)
    return choose_rows(vals.cpu().numpy(), id_map[idx].cpu().numpy().astype(np.int64), positions, sampling)


def _mtp_forward(w, st, buf, next_tokens: Sequence[int], streams: torch.Tensor) -> torch.Tensor:
    """The MTP head's pass, resolved at call time: the head's kernels merge separately from the engine."""

    try:
        from .mtp import mtp_forward
    except ImportError as exc:                                    # the sibling slice has not landed
        raise RuntimeError("the XPU MTP head has not landed in this build") from exc
    return mtp_forward(w, st, buf, next_tokens, streams)


def _absorbs(e: Engine, mtp: bool) -> bool:
    return mtp and e.w.mtp is not None and e.mbuf is not None


@torch.no_grad()
def prefill_begin(e: Engine, prompt: Sequence[int], *, mtp: bool = True, resume: dict | None = None) -> int:
    """Empty the state, or restore a kept prompt end and absorb its tail; returns the first prompt row to commit."""

    if not prompt:
        raise ValueError("prefill requires at least one token")
    if resume is None:
        e.reset()
        return 0
    st = e.st
    st.restore(resume["state"])
    if not 0 < st.pos < len(prompt):
        raise ValueError("a resumed prompt must extend the cached tokens")
    if _absorbs(e, mtp) and resume.get("tail") is not None:
        _mtp_forward(e.w, st, e.pbuf, [prompt[st.pos]], resume["tail"])
        st.set_mtp_len(st.mtp_len + 1)
    return st.pos


@torch.no_grad()
def prefill_chunk(e: Engine, prompt: Sequence[int], start: int, *, mtp: bool = True,
                  keep_at: int | None = None, end: int | None = None) -> torch.Tensor | None:
    """Commit up to ``e.prefill_rows`` rows from ``start`` (the last chunk returns its logits); a chunk holding ``keep_at`` sets ``e.kept``."""

    w, st, pb = e.w, e.st, e.pbuf
    end = min(start + e.prefill_rows, len(prompt) if end is None else end)
    chunk = list(prompt[start:end])
    R = len(chunk)
    final = end == len(prompt)
    point = keep_at - start if keep_at is not None and start < keep_at <= end else 0     # the kept point's row
    cut = Cut(point) if 0 < point < R else None           # inside the chunk, not at its end
    if e.streamer is not None:                            # the streamed window slots hold this chunk's experts
        e.streamer.begin("window", R, pb.moe)
    # only the prompt's last row is sampled: the head runs on the final chunk alone
    logits = forward(w, st, pb, chunk, logits=final, cut=cut)
    last = logits.clone() if final else None
    e.last_streams = pb.streams[R - 1:R].clone()
    use_mtp = _absorbs(e, mtp)
    if point:                    # before the MTP head writes the streams: the point's tail, its state inside the chunk
        mtp_len = st.mtp_len + point - 1 if use_mtp else st.mtp_len       # every row but the point's last
        tail = pb.streams[point - 1:point].clone() if use_mtp else None
        snap = cut_snapshot(w, st, pb, cut, mtp_len) if cut is not None else None
    nxt = list(prompt[start + 1:end + 1])
    if use_mtp and nxt:
        _mtp_forward(w, st, pb, nxt, pb.streams[:len(nxt)])
        st.set_mtp_len(st.mtp_len + len(nxt))
    commit(w, st, pb, R, R)
    if point:                                            # as a fresh prefill of prompt[:keep_at] leaves it
        e.kept = {"state": snap if snap is not None else {**st.snapshot(), "mtp_len": mtp_len}, "tail": tail}
    return last


@torch.no_grad()
def prefill(e: Engine, prompt: Sequence[int], sampling: Sampling | None, *, mtp: bool = True,
           resume: dict | None = None, constraint=None, probabilities=None, keep_at: int | None = None,
           stops: Sequence[int] = (), keep=None) -> int:
    """Commit the prompt in chunks and sample the first token (``resume`` equals a fresh run); ``e.kept`` resumes prompt[:keep_at]."""

    start, last = prefill_begin(e, prompt, mtp=mtp, resume=resume), None
    if keep_at is not None and not start <= keep_at <= len(prompt):
        raise ValueError(f"keep_at {keep_at} is outside the prefilled range [{start}, {len(prompt)}]")
    saved = e.kept = resume if keep_at == start else None
    stops = sorted({p for p in stops if start < p < len(prompt)})
    while start < len(prompt):
        # the DeltaNet chain runs a chunk whole, so chunks stay inside the captured GDN width
        end = min(start + min(e.prefill_rows, GDN_CHUNK), next((p for p in stops if p > start), len(prompt)))
        if keep_at is not None and start < keep_at < end and end in stops:
            end = keep_at
        point = keep_at if keep_at is not None and start < keep_at <= end else end if end in stops else None
        last = prefill_chunk(e, prompt, start, mtp=mtp, keep_at=point, end=end)
        if point is not None:
            if point == keep_at:
                saved = e.kept
            if keep is not None:
                keep(point, e.kept["state"], e.kept["tail"])
        start = end
    if keep_at is not None:
        e.kept = saved
    if constraint is not None:                           # a reply's grammar: this rank's vocabulary columns
        last = constraint.mask(last, None, e.w.meta.get("vocab_offset", 0))
    first = e.sample(last, [len(prompt)], sampling)[0]
    if probabilities is not None:
        from tensorfold.cuda.logprobs import capture

        capture(last, [first], [len(prompt)], probabilities)
    if constraint is not None:
        constraint.advance([first])
    e.first = first
    return first


WARM_TAIL = 18      # a partial chunk after a full one: neither its rows nor the MTP head's 17 divide by 16


@torch.no_grad()
def warm(e: Engine) -> None:
    """Prefill a synthetic prompt (a full chunk, then a partial one cut at the kept point a row before its end) and empty the state, so no request compiles or loads a prompt kernel."""

    prompt = [0] * min(e.prefill_rows + WARM_TAIL + 1, e.capacity)
    prefill(e, prompt, None, keep_at=entry_end(prompt))
    e.kept = None
    e.reset()


@dataclass
class DecodeResult:
    tokens: list[int]
    seconds: float
    rounds: int
    drafted: int = 0
    accepted: int = 0
    keeps: list[int] = field(default_factory=list)      # tokens each round kept
    committed: list[int] = field(default_factory=list)  # the tokens now in the caches (all but the pending one)
    widths: list[int] = field(default_factory=list)     # rows each round verified

    @property
    def tokens_per_second(self) -> float:
        return (len(self.tokens) - 1) / self.seconds if self.seconds else 0.0


@torch.no_grad()
def serial_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *, stop_eos: bool = False,
                  on_tokens=None, constraint=None, probabilities=None) -> DecodeResult:
    """One token a step through the same kernels and sampler; ``pending`` is the first sampled token. ``on_tokens(new)`` hears each step's token; it returns True to stop early."""

    w, st, b = e.w, e.st, e.buf
    out = [pending]
    _sync(b.logits)
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        logits = e.forward([out[-1]])
        if constraint is not None:
            constraint.mask(logits[:1], None, w.meta.get("vocab_offset", 0))
        tok = e.sample(logits[:1], [st.pos + 1], sampling, gathered=constraint is None)[0]
        if probabilities is not None:
            from tensorfold.cuda.logprobs import capture

            capture(logits[:1], [tok], [st.pos + 1], probabilities)
        commit(w, st, b, 1, 1)
        out.append(tok)
        if constraint is not None:
            constraint.advance([tok])
        if on_tokens is not None and on_tokens([tok]):
            break
    _sync(b.logits)
    return DecodeResult(out, time.perf_counter() - start, len(out) - 1, committed=out[:-1], widths=[1] * (len(out) - 1))


def absorb(e: Engine, streams: torch.Tensor, next_tokens: Sequence[int]) -> torch.Tensor:
    """The MTP cache takes positions with main-model streams [n, S*D] and next tokens; logits of the last."""

    if not _absorbs(e, True):
        raise RuntimeError("the XPU MTP head has not landed in this build")
    st = e.st
    if st.mtp_drafted:
        st.set_mtp_len(st.mtp_len - st.mtp_drafted)
        st.mtp_drafted = 0
    logits = e.mtp_forward(next_tokens, streams)
    st.set_mtp_len(st.mtp_len + len(next_tokens))
    return logits


def draft(e: Engine, streams: torch.Tensor, next_tokens: Sequence[int], position: int, count: int,
          sampling: Sampling | None, confidence: float = 0.0) -> list[int]:
    """Absorb kept rows and chain drafts, always retaining the first even below ``confidence``, then stopping before later drafts below it or after a low-confidence first draft."""

    if not _absorbs(e, True):
        raise RuntimeError("the XPU MTP head has not landed in this build")
    st = e.st
    logits = absorb(e, streams, next_tokens)
    drafts: list[int] = []
    for j in range(count):
        low = False
        if confidence > 0:
            d, p = e.sample_draft(logits, position + j, sampling)
            low = p < confidence
            if low and j > 0:
                break
        else:
            d = e.sample(logits[:1], [position + j], sampling, draft=True)[0]
        drafts.append(d)
        if low:
            break
        if j + 1 < count:
            prev = e.mbuf.streams[len(next_tokens) - 1:len(next_tokens)] if j == 0 else e.mbuf.streams[:1]
            logits = e.mtp_forward([d], prev)
            st.set_mtp_len(st.mtp_len + 1)
            st.mtp_drafted += 1
            next_tokens = [d]
    return drafts


@torch.no_grad()
def mtp_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *, depth: int = DEPTH,
               confidence: float = CONFIDENCE, stop_eos: bool = False, on_tokens=None, constraint=None,
               probabilities=None) -> DecodeResult:
    """Verify pending and drafted tokens from the prefill state, commit rows before the first mismatched draft, and call ``on_tokens(new)`` with kept tokens after pending, stopping on True."""

    if not _absorbs(e, True):
        raise RuntimeError("the XPU MTP head has not landed in this build")
    w, st, b = e.w, e.st, e.buf
    out = [pending]
    rounds = drafted = accepted = 0
    keeps: list[int] = []
    widths: list[int] = []
    pos0 = st.pos
    unabsorbed = None                                  # the last round's kept rows, not yet in the MTP cache
    _sync(b.logits)
    start = time.perf_counter()
    drafts = draft(e, e.last_streams, [pending], st.pos + 1, min(depth, count - len(out)), sampling, confidence)
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        tokens = [out[-1]] + drafts
        window = constraint.window(tokens, list(range(-1, len(tokens) - 1))) if constraint is not None else None
        if window is not None:                           # the drafts no accepted path can hold are cut first
            tokens, drafts = window.tokens, window.tokens[1:]
        R = len(tokens)
        logits = e.forward(tokens)
        if window is not None:
            constraint.mask(logits[:R], window, w.meta.get("vocab_offset", 0))
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling, gathered=window is None)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        if probabilities is not None:
            from tensorfold.cuda.logprobs import capture

            n = min(keep, count - len(out))
            capture(logits[:n], sampled[:n], list(range(st.pos + 1, st.pos + 1 + n)), probabilities)
        commit(w, st, b, R, keep)
        unabsorbed = (keep, sampled[:keep])
        rounds += 1
        drafted += len(drafts)
        accepted += keep - 1
        keeps.append(keep)
        widths.append(R)
        new = sampled[:keep][:max(0, count - len(out))]
        if constraint is not None:
            constraint.advance(sampled[:keep])
        out.extend(sampled[:keep])
        if on_tokens is not None and new and on_tokens(new):
            break
        if len(out) >= count or (stop_eos and out[-1] in w.cfg.eos):
            break
        n = min(depth, count - len(out))
        drafts = []
        if n > 0:
            drafts = draft(e, b.streams[:keep], sampled[:keep], st.pos + 1, n, sampling, confidence)
        unabsorbed = None
    _sync(b.logits)
    seconds = time.perf_counter() - start
    if unabsorbed is not None: # the MTP cache takes the last kept rows: it then covers the sequence
        absorb(e, b.streams[:unabsorbed[0]], unabsorbed[1])
    committed = out[:st.pos - pos0]
    return DecodeResult(out[:count], seconds, rounds, drafted, accepted, keeps, committed, widths)
