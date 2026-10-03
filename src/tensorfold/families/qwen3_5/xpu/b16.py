"""A plain fp16/bf16 linear for what an EXL3 pack leaves unquantized, in torch ops (row-invariant)."""

from __future__ import annotations

import torch

# b16 CUDA ext unavailable on XPU; torch fallback (PLAN.md Phase 2 option B)


def matmul(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """x [M, K] @ w [N, K]^T (+ bias [N]), x cast to the weight's dtype when they differ."""

    if x.dtype != w.dtype:
        x = x.to(w.dtype)
    b = bias if bias is not None and bias.numel() else None
    return torch.nn.functional.linear(x.contiguous(), w.contiguous(), b)


def prompt(x: torch.Tensor, w: torch.Tensor, bm: int = 0) -> torch.Tensor:
    """Prompt rows x [M, K] @ w [N, K]^T, chunk-invariant (``bm`` shaped the CUDA ext's launch; unused here)."""

    return matmul(x, w)


def matmul_pair(x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor) -> list[torch.Tensor]:
    """x @ w0^T and x @ w1^T (the GDN gates b and a), each ``matmul``'s semantics."""

    return [matmul(x, w0), matmul(x, w1)]


def prompt_pair(x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor) -> list[torch.Tensor]:
    """Prompt rows times two weights, each ``prompt``'s semantics."""

    return [prompt(x, w0), prompt(x, w1)]
