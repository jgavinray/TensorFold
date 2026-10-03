"""The RadixArk DSpark drafter: a five-layer DFlash backbone over the target's layer taps plus a
rank-256 Markov head and an acceptance-rate head, drafting a seven-token block per round.

The checkpoint (/storage01/models/dspark-qwen38-27b, SpecForge ``DSparkDraftModel``) is a plain
Qwen3 stack of dual-source blocks: each layer projects keys and values from both the projected
target taps (written once, on arrival, like DFlash2's context ring) and the block's own rows
(``[anchor, mask...]``); every block row attends every context row and every block row. The final
normed rows go through the *target's* full-vocabulary ``lm_head`` (the draft owns no head), the
Markov head adds a learned bigram bias conditioned on the token before each row, and the chain is
drawn row by row with the verifier's keyed Gumbel noise, so a proposal the target agrees with is
accepted exactly. The confidence head predicts each row's acceptance probability for adaptive
truncation; it is loaded and exposed but only applied when a threshold is passed, since its
calibration is an A/B decision (the checkpoint's own serving draws the full block).

Interface mirrors the engine's DFlash2 drafter (``dflash2.DFlash2``) so ``Qwen27Engine``'s draft
hooks accept it unchanged: ``add_taps``/``snapshot``/``restore``/``skip``/``propose_tree`` (plus
``launch_blocks``/``finish_tree`` for the multi-stream path), ``layers``/``window``/``block``/
``trained``/``in_vocab``/``last_candidates``. The engine's ``tree_forward``/``prefill`` already
collect the target taps this drafter consumes (``TAP_LAYERS`` == this checkpoint's
``target_layer_ids``), so no forward-path change is needed. Pure torch throughout: the only
device dispatch is the head matmul, which uses the engine's qmm kernel on XPU and a plain
dequantized matmul elsewhere; no CUDA-only calls.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tensorfold.cuda.direct_read import SafeTensors
from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows, uniform

from .weights import Exl3, Plain, QLinear

# the engine's tap layers (forward.py and prefill.py): the checkpoint must agree, or its fc would
# silently mix the wrong layers' hidden states
ENGINE_TAPS = (5, 19, 33, 47, 61)

NIBBLE_SHIFTS = (0, 4, 8, 12, 16, 20, 24, 28)


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Qwen rotate-half RoPE; x is (rows, heads, head_dim), cos/sin (rows, head_dim // 2)."""
    c, s = cos[:, None, :].float(), sin[:, None, :].float()
    a, b = x[..., : c.shape[-1]].float(), x[..., c.shape[-1]:].float()
    return torch.cat((a * c - b * s, b * c + a * s), dim=-1).to(x.dtype)


def _correction_dim(rotations: float, dim: int, base: float, original_max: int) -> float:
    return dim * math.log(original_max / (rotations * 2 * math.pi)) / (2 * math.log(base))


def _inv_freq(head_dim: int, theta: float, scaling: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, float]:
    """The drafter's own rotary (the checkpoint's rope_scaling, yarn per the RadixArk config)."""
    half = head_dim // 2
    inv = theta ** (-torch.arange(0, half, dtype=torch.float64) * 2 / head_dim)
    factor = 1.0
    if str(scaling.get("rope_type", "")) == "yarn":
        original = int(scaling["original_max_position_embeddings"])
        low = max(math.floor(_correction_dim(float(scaling["beta_fast"]), head_dim, theta, original)), 0)
        high = min(math.ceil(_correction_dim(float(scaling["beta_slow"]), head_dim, theta, original)), half - 1)
        if low == high:
            high += 0.001
        ramp = torch.clamp((torch.arange(half, dtype=torch.float64) - low) / (high - low), 0.0, 1.0)
        inv = inv * (1 - ramp) + inv / float(scaling["factor"]) * ramp
        factor = 0.1 * math.log(original) + 1.0
    return inv.float().to(device), factor


def _dequant_rows(weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, rows: torch.Tensor,
                  k: int, gs: int) -> torch.Tensor:
    """Selected rows of an MLX affine 4-bit table (little-endian nibbles, low nibble first per
    32-bit word, groups of ``gs`` inputs per scale/bias) as bf16."""
    if weight.shape[1] * 8 != k:
        raise ValueError(f"packed words {tuple(weight.shape)} do not hold {k} 4-bit values per row")
    words = weight.index_select(0, rows).to(torch.int64)                            # (R, K // 8)
    shifts = torch.tensor(NIBBLE_SHIFTS, device=weight.device, dtype=torch.int64)
    nibbles = ((words.unsqueeze(-1) >> shifts) & 0xF).reshape(rows.shape[0], k)     # (R, K)
    group = scales.index_select(0, rows).repeat_interleave(gs, dim=-1)
    bias = biases.index_select(0, rows).repeat_interleave(gs, dim=-1)
    return (nibbles.float() * group.float() + bias.float()).to(torch.bfloat16)


class DSpark:
    """The loaded RadixArk drafter: target taps in, a chain of at most ``block`` draft tokens out."""

    def __init__(self, draft_dir: str | Path, target: Any, *, rank: int = 0, world: int = 1,
                 keep: int | None = None, conf_threshold: float | None = None):
        if world != 1:
            raise ValueError("the DSpark drafter runs on one rank (tp_draft is a DFlash2 feature)")
        self.rank, self.world = rank, world
        path = Path(draft_dir)
        cfg = json.loads((path / "config.json").read_text())
        self.hidden = int(cfg["hidden_size"])
        self.head_dim = int(cfg["head_dim"])
        self.heads = int(cfg["num_attention_heads"])
        self.kv_heads = int(cfg["num_key_value_heads"])
        self.eps = float(cfg["rms_norm_eps"])
        self.block = int(cfg["block_size"])                              # block rows per round, anchor included
        self.trained = int(cfg.get("training_block_size", self.block))   # the planner's floor
        self.layers = int(cfg["num_hidden_layers"])
        self.mask_id = int(cfg["mask_token_id"])
        raw_taps = cfg.get("target_layer_ids") or (cfg.get("dflash_config") or {}).get("target_layer_ids")
        if raw_taps is None:
            raise ValueError("the checkpoint declares no target_layer_ids")
        self.taps = tuple(int(t) for t in raw_taps)
        if self.taps != ENGINE_TAPS:
            raise ValueError(f"checkpoint taps {self.taps} do not match the engine's TAP_LAYERS {ENGINE_TAPS}")
        if str(cfg.get("markov_head_type", "vanilla")) != "vanilla":
            raise ValueError(f"markov_head_type {cfg.get('markov_head_type')!r} is not supported")
        self.with_markov_conf = bool(cfg.get("confidence_head_with_markov", True))
        self.conf_threshold = conf_threshold
        self.embed, self.head = target.embed, target.head
        self.device = target.norm.device
        self.vocab = int(target.config.vocab)
        self.window = keep if keep is not None else 1 << 40              # the engine taps the last `window` rows
        self.keep = keep                                                  # None: the full context stays in the ring
        self.last_candidates: np.ndarray | None = None

        self.weights: dict[str, torch.Tensor] = {}
        f = SafeTensors([path / "model.safetensors"])
        for name in f.keys():
            self.weights[name] = f.get(name)                              # on the host until the one device move
        del f
        if self.weights["markov_head.markov_w1.weight"].shape[0] != self.vocab:
            raise ValueError("the checkpoint's Markov table does not cover the target's vocabulary")
        if self.weights["fc.weight"].shape[1] != len(self.taps) * self.hidden:
            raise ValueError("fc does not consume exactly the checkpoint's taps")
        self.inv_freq, self.attn_factor = _inv_freq(self.head_dim, float(cfg["rope_theta"]),
                                                    cfg.get("rope_scaling") or cfg.get("rope_parameters") or {},
                                                    self.device)
        self.weights = {name: t.to(self.device) for name, t in self.weights.items()}
        self.kc: list[torch.Tensor | None] = [None] * self.layers         # per-layer context keys (kv_heads, rows, dim)
        self.vc: list[torch.Tensor | None] = [None] * self.layers
        self.context_len = 0
        self.context_end = 0
        self._dense_head: torch.Tensor | None = None

    # -- target pieces ----------------------------------------------------

    def _token_rows(self, ids: torch.Tensor) -> torch.Tensor:
        """The target's embedding rows for the block's ids, at the checkpoint's own bits."""
        q = self.embed
        if isinstance(q, Plain) or getattr(q, "layout", "") in ("dense", "b16"):
            return q.weight[ids].to(torch.bfloat16).contiguous()
        if q.bits != 4 or q.scales is None:
            raise NotImplementedError(f"embedding format {q.bits}-bit is not supported")
        flat, inverse = torch.unique(ids, return_inverse=True)            # the mask rows dequant once
        return _dequant_rows(q.weight, q.scales, q.biases, flat, q.k, q.gs)[inverse].contiguous()

    def _head_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary logits through the target's own head (the draft owns no head)."""
        head = self.head
        if isinstance(head, Exl3):
            return head(h)                                                # the engine's row-batched EXL3 linear
        if isinstance(head, Plain) or head.layout == "dense":
            weight = head.weight if head.weight.dtype == h.dtype else head.weight.to(h.dtype)
            return F.linear(h, weight)
        if head.bits != 4 or head.gs != 64 or head.scales is None:
            raise NotImplementedError(f"head format {head.bits}-bit g{getattr(head, 'gs', 0)} is not supported")
        if h.device.type == "xpu":                                        # the engine's own 4-bit matmul, bit for bit
            from .qmm_fast import matmul

            return matmul(h.contiguous(), head)
        if self._dense_head is None:                                      # off-XPU: dequant once, reuse
            rows = torch.arange(head.n, device=head.weight.device)
            self._dense_head = _dequant_rows(head.weight, head.scales, head.biases, rows, head.k, head.gs)
        return F.linear(h, self._dense_head)

    def _rotary(self, start: int, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        phase = (torch.arange(start, start + rows, device=self.device, dtype=torch.float32)[:, None]
                 * self.inv_freq[None, :])
        return (phase.cos() * self.attn_factor).contiguous(), (phase.sin() * self.attn_factor).contiguous()

    def _draw(self, values: torch.Tensor, position: int, sampling: Sampling | None) -> tuple[int, np.ndarray]:
        """One row's token by the verifier's own rule (keyed Gumbel over the kept candidates)."""
        if sampling is None or sampling.temperature <= 0:
            return int(values.argmax()), np.empty(0, dtype=np.int64)
        if sampling.top_k:
            count = min(values.shape[0], int(sampling.top_k) + MARGIN)
            top, ids = torch.topk(values, count)
            picked = choose_rows(top[None].cpu().numpy(), ids[None].cpu().numpy().astype(np.int64),
                                 [position], sampling)[0]
            return picked, ids.cpu().numpy().astype(np.int64)
        scaled = (values / max(float(sampling.temperature), 1e-6)).double()   # the exact nucleus: the whole row
        order = torch.argsort(scaled, descending=True)
        row, ids = scaled.cpu().numpy(), order.cpu().numpy().astype(np.int64)
        keep = len(ids)
        if 0.0 < sampling.top_p < 1.0:
            norm = float(torch.logsumexp(scaled, dim=-1))
            keep = int((np.cumsum(np.exp(row[ids] - norm)) < sampling.top_p).sum()) + 1
        if sampling.min_p > 0.0:
            keep = min(keep, int((row[ids] >= row[ids[0]] + sampling.min_log).sum()))
        gumbel = -np.log(-np.log(uniform(sampling.seed, position, ids[:keep])))
        return int(ids[:keep][int(np.argmax(row[ids][:keep] + gumbel))]), ids[:keep]

    # -- engine interface -------------------------------------------------

    def snapshot(self) -> tuple:
        return (list(self.kc), list(self.vc), self.context_len, self.context_end)

    def restore(self, snap) -> None:
        kc, vc, self.context_len, self.context_end = snap
        self.kc, self.vc = list(kc), list(vc)

    def skip(self, n: int) -> None:
        """Skip ``n`` rows untapped (their positions still count); the caller adds fresh rows after."""
        self.kc, self.vc = [None] * self.layers, [None] * self.layers
        self.context_len, self.context_end = 0, self.context_end + n

    @torch.no_grad()
    def add_taps(self, taps: torch.Tensor) -> None:
        """The target's committed rows (their taps [n, taps * hidden]) as context at the next positions."""
        if taps.ndim != 2 or taps.shape[1] != len(self.taps) * self.hidden:
            raise ValueError(f"DSpark expects {len(self.taps)} target layer taps per committed row")
        taps = taps.to(self.device)
        w = self.weights
        projected = _rmsnorm(F.linear(taps, w["fc.weight"]), w["hidden_norm.weight"], self.eps)
        n = projected.shape[0]
        cos, sin = self._rotary(self.context_end, n)
        for layer in range(self.layers):
            prefix = f"layers.{layer}.self_attn."
            k = F.linear(projected, w[prefix + "k_proj.weight"]).view(n, self.kv_heads, self.head_dim)
            v = F.linear(projected, w[prefix + "v_proj.weight"]).view(n, self.kv_heads, self.head_dim)
            k = _rope(_rmsnorm(k, w[prefix + "k_norm.weight"], self.eps), cos, sin).transpose(0, 1)
            v = v.transpose(0, 1)
            self.kc[layer] = k if self.kc[layer] is None else torch.cat((self.kc[layer], k), dim=1)
            self.vc[layer] = v if self.vc[layer] is None else torch.cat((self.vc[layer], v), dim=1)
            if self.keep is not None:
                self.kc[layer] = self.kc[layer][:, -self.keep:].contiguous()
                self.vc[layer] = self.vc[layer][:, -self.keep:].contiguous()
        self.context_len = self.context_len + n if self.keep is None else min(self.keep, self.context_len + n)
        self.context_end += n

    @torch.no_grad()
    def add_taps_streams(self, snaps: list, taps: list[torch.Tensor]) -> list:
        """``add_taps`` for several streams; returns each stream's new context (the engine's snapshots)."""
        out = []
        for snap, t in zip(snaps, taps):
            self.restore(snap)
            self.add_taps(t)
            out.append(self.snapshot())
        return out

    def _block(self, snap, pending: int, rows: int) -> torch.Tensor:
        """The block's rows through the dual-source backbone: their final normed hidden states."""
        w = self.weights
        ids = torch.tensor([pending] + [self.mask_id] * (rows - 1), dtype=torch.int64, device=self.device)
        x = self._token_rows(ids)
        cos, sin = self._rotary(snap[3], rows)
        for layer in range(self.layers):
            base = f"layers.{layer}."
            prefix = base + "self_attn."
            normed = _rmsnorm(x, w[base + "input_layernorm.weight"], self.eps)
            q = F.linear(normed, w[prefix + "q_proj.weight"]).view(rows, self.heads, self.head_dim)
            kn = F.linear(normed, w[prefix + "k_proj.weight"]).view(rows, self.kv_heads, self.head_dim)
            vn = F.linear(normed, w[prefix + "v_proj.weight"]).view(rows, self.kv_heads, self.head_dim)
            q = _rope(_rmsnorm(q, w[prefix + "q_norm.weight"], self.eps), cos, sin).transpose(0, 1)
            kn = _rope(_rmsnorm(kn, w[prefix + "k_norm.weight"], self.eps), cos, sin)
            keys = torch.cat((snap[0][layer], kn.transpose(0, 1)), dim=1)
            values = torch.cat((snap[1][layer], vn.transpose(0, 1)), dim=1)
            groups = self.heads // self.kv_heads
            out = F.scaled_dot_product_attention(q, keys.repeat_interleave(groups, dim=0),
                                                 values.repeat_interleave(groups, dim=0),
                                                 scale=self.head_dim ** -0.5)
            x = (x.float() + F.linear(out.transpose(0, 1).reshape(rows, self.heads * self.head_dim),
                                      w[prefix + "o_proj.weight"]).float()).to(torch.bfloat16)
            normed = _rmsnorm(x, w[base + "post_attention_layernorm.weight"], self.eps)
            mlp = F.silu(F.linear(normed, w[base + "mlp.gate_proj.weight"])) \
                * F.linear(normed, w[base + "mlp.up_proj.weight"])
            x = (x.float() + F.linear(mlp, w[base + "mlp.down_proj.weight"]).float()).to(torch.bfloat16)
        return _rmsnorm(x, w["norm.weight"], self.eps)

    def _confidences(self, h: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Block rows' predicted acceptance probabilities (sigmoid of the head's logit)."""
        if "confidence_head.proj.weight" not in self.weights:
            return torch.ones(h.shape[0], device=h.device)
        features = torch.cat((h, latents), dim=-1) if self.with_markov_conf else h
        return torch.sigmoid(F.linear(features.float(), self.weights["confidence_head.proj.weight"].float(),
                                      self.weights["confidence_head.proj.bias"].float()).squeeze(-1))

    @torch.no_grad()
    def launch_blocks(self, snaps: list, pendings: list[int], max_nodes: int, block: int | None = None) -> list:
        """The drafter half of ``propose_tree`` for several streams; None where a context is empty."""
        out: list = [None] * len(snaps)
        if max_nodes < 1:
            return out
        rows = min(block or self.block, self.block, max_nodes)
        for i, (snap, pending) in enumerate(zip(snaps, pendings)):
            if snap[2] <= 0:
                continue
            h = self._block(snap, int(pending), rows)
            out[i] = ([self._head_logits(h), h, None], 0, rows, int(pending))
        return out

    def launch_block(self, pending: int, max_nodes: int, block: int | None = None):
        """``launch_blocks`` for the current context alone."""
        return self.launch_blocks([self.snapshot()], [pending], max_nodes, block)[0]

    @torch.no_grad()
    def finish_tree(self, launched, context_length: int, max_nodes: int,
                    sampling: Sampling | None = None) -> tuple[list[int], list[int], list[float]]:
        """The host half: the Markov chain over the launched block's rows, each row drawn with the
        verifier's keyed noise at its own position, the confidence head cutting the chain short when
        a threshold is set. Nodes, parents and cumulative path scores in chain order."""
        if launched is None:
            return [], [], []
        shared, _, rows, pending = launched
        logits, h = shared[0].float(), shared[1]
        w1, w2 = self.weights["markov_head.markov_w1.weight"], self.weights["markov_head.markov_w2.weight"]
        tokens: list[int] = []
        parents: list[int] = []
        scores: list[float] = []
        cands: list[np.ndarray] = []
        temp = max(float(sampling.temperature), 1e-6) if sampling is not None else 1.0
        prev, score = int(pending), 0.0
        for j in range(rows):
            latent = w1[prev]
            if self.conf_threshold is not None:
                if float(self._confidences(h[j:j + 1], latent[None])[0]) < self.conf_threshold:
                    break
            values = logits[j] + (w2 @ latent).float()
            token, top = self._draw(values, context_length + j, sampling)
            tokens.append(token)
            parents.append(-1 if j == 0 else j - 1)
            score += -float(values[token]) / temp
            scores.append(score)
            cands.append(top)
            prev = token
        self.last_candidates = np.stack(cands) if cands else None
        return tokens, parents, scores

    @torch.no_grad()
    def propose_tree(self, pending: int, context_length: int, max_nodes: int,
                     sampling: Sampling | None = None, block: int | None = None) -> tuple[list[int], list[int]]:
        return self.finish_tree(self.launch_block(pending, max_nodes, block), context_length, max_nodes, sampling)[:2]

    def in_vocab(self, token: int) -> bool:
        """Whether the drafter's head can propose ``token`` at all: the target's full vocabulary."""
        return 0 <= int(token) < self.vocab


def load(draft_dir: str | Path, target: Any, **options: Any) -> DSpark:
    """The drafter from a RadixArk directory (``config.json`` + ``model.safetensors``), sharing the
    target's embedding, head and device."""
    return DSpark(draft_dir, target, **options)
