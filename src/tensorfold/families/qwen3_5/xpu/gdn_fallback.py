"""Serial torch GDN for XPU: ``reference._gdn``'s recurrence as the tree, chain and replay path.

The CUDA extension (``kernels/gdn``) runs the same serial step per row, so a row's bits never depend on
its window; this module reproduces that step in plain torch fp32 on the device's tensors, for engines on
accelerators without the CUDA ext (PLAN.md Phase 2).  Planning is shared with the CUDA module unchanged:
``schedule``, ``plan_host``, ``plan`` and ``Plan`` are pure python and device-free.  Where the CUDA ext
takes int64 device-pointer tables (a kernel-launch artifact), this module takes the state tensors
themselves and steps them in place.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Sequence

import torch

from tensorfold.cuda.kernels.gdn import Plan, plan_host, schedule

__all__ = ["Plan", "plan", "plan_host", "schedule", "to_device", "tree", "chain", "replay", "replay_table"]


def to_device(values: Sequence[int], dtype: torch.dtype, device) -> torch.Tensor:
    """One device copy (the CUDA path's pinned staging is a CUDA-allocator feature; the fallback syncs)."""

    return torch.tensor(values, dtype=dtype, device=device)


def plan(streams: Sequence[Sequence[int]], device) -> Plan:
    """The shared planning with this module's copy (no pinned staging); kernels/gdn.plan otherwise."""

    entries, starts, slots, rows = plan_host(streams)
    dev = to_device(entries + starts, torch.int32, device)
    w = starts[-1]
    return Plan(dev[:3 * w].view(w, 3), dev[3 * w:], slots, rows)


def _step(S: torch.Tensor, qt: torch.Tensor, kt: torch.Tensor, vt: torch.Tensor,
          gt: torch.Tensor, betat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One row's step, ``reference._gdn``'s inner loop in fp32: S <- S*g + k ((v - k.S) beta), y <- q.S."""

    S = S * gt[:, None, None]
    kv_mem = (S * kt[:, None, :]).sum(-1)
    delta = (vt - kv_mem) * betat[:, None]
    S = S + kt[:, None, :] * delta[:, :, None]
    return S, (S * qt[:, None, :]).sum(-1)


def _expanded(rows: torch.Tensor, rep: int) -> torch.Tensor:
    """(W, Hk, Dk) rows (or one (Hk, Dk) row) to fp32 with each key head shared by its ``rep`` value heads."""

    return rows.float().repeat_interleave(rep, -2)


def tree(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, p: Plan,
         state: torch.Tensor | None = None, table: Sequence[torch.Tensor] | None = None,
         pending: Sequence[torch.Tensor] | None = None, final: torch.Tensor | Sequence[torch.Tensor] | None = None
         ) -> torch.Tensor:
    """Outputs (W, Hv, Dv) bf16; each entry applies its parent's resulting state (a slot, the previous row
    or the committed state), so a row's bits equal its serial decode.

    ``table`` is the fallback's list of every stream's fp32 state tensors (the int64 pointer table the CUDA
    ext takes is a launch artifact).  ``pending`` — k, v, g, beta, rows (streams, P) int32 indexing those
    tensors, counts (streams,) int32 — steps committed states in place first (gdn.cu: its pending rows only
    advance the state, no output).  ``final`` (chain windows only, ``p.slots == 0``) receives the last row's
    resulting state: one state-shaped tensor, or one per stream.
    """

    if (state is None) == (table is None):
        raise ValueError("pass one stream's state or a table of every stream's")
    W, Hk, dk = q.shape
    Hv, dv = v.shape[1], v.shape[2]
    rep = Hv // Hk
    qf, kf = _expanded(q, rep), _expanded(k, rep)
    vf = v.float()
    entries = p.entries.reshape(-1, 3).tolist()
    if state is not None:
        states, starts = [state], [0, W]
    else:
        states = list(table)
        starts = p.starts.tolist()
        if len(states) + 1 != len(starts):
            raise ValueError("a table of every stream's state, with starts offsets")
    if pending is not None:
        pk, pv, pg, pb, prows, pcounts = pending
        pkf, pvf, pgf, pbf = _expanded(pk, rep), pv.float(), pg.float(), pb.float()
        held, counts = prows.tolist(), pcounts.tolist()
        for s, state_t in enumerate(states):
            for row in held[s][:counts[s]]:
                state_t.copy_(_step(state_t, pkf[row], pkf[row], pvf[row], pgf[row], pbf[row])[0])
    slots: list[dict[int, torch.Tensor]] = [{} for _ in states]
    last: list[torch.Tensor | None] = [None] * len(states)
    y = torch.empty((W, Hv, dv), dtype=v.dtype, device=q.device)
    for node, source, dest in entries:
        s = bisect_right(starts, node) - 1
        r = node - starts[s]
        if source == -1:
            S = states[s]
        elif source == -2:
            S = last[s]
        else:
            S = slots[s][source]
        S, yr = _step(S, qf[r], kf[r], vf[r], g[r].float(), beta[r].float())
        y[r] = yr.to(v.dtype)
        last[s] = S
        if dest >= 0:
            slots[s][dest] = S
    if final is not None:
        if p.slots:
            raise ValueError("a final state is only defined for chain windows")
        if state is not None:
            final.copy_(last[0])
        else:
            for state_t, last_t in zip(final, last):
                state_t.copy_(last_t)
    return y


def chain(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
          state: torch.Tensor, final: torch.Tensor) -> torch.Tensor:
    """Prefill chain from ``state`` (read only) to ``final``: the serial step a row at a time, chunk-invariant."""

    rep = v.shape[1] // q.shape[1]
    qf, kf = _expanded(q, rep), _expanded(k, rep)
    vf = v.float()
    S = state
    ys = []
    for t in range(q.shape[0]):
        S, yr = _step(S, qf[t], kf[t], vf[t], g[t].float(), beta[t].float())
        ys.append(yr)
    final.copy_(S)
    return torch.stack(ys).to(v.dtype)


def replay_table(k: Sequence[torch.Tensor], v: Sequence[torch.Tensor], g: Sequence[torch.Tensor],
                 beta: Sequence[torch.Tensor], states: Sequence[Sequence[torch.Tensor]]) -> list[torch.Tensor]:
    """The fallback's table for ``replay``: k, v, g, beta of each layer, then states[stream][layer] — the
    tensors themselves, where the CUDA path packs device addresses."""

    layers = len(k)
    if not (len(v) == len(g) == len(beta) == layers) or any(len(s) != layers for s in states):
        raise ValueError("one k, v, g, beta per layer and one state per stream and layer")
    shape = states[0][0].shape
    if any(t.shape != shape or t.dtype != torch.float32 for s in states for t in s):
        raise ValueError("every state is one fp32 (Hv, Dv, 128) tensor")
    table: list[torch.Tensor] = []
    for layer in range(layers):
        table += [k[layer], v[layer], g[layer], beta[layer]]
    for s in states:
        table += list(s)
    return table


def replay(table: Sequence[torch.Tensor], layers: int, streams: int, rows: torch.Tensor, counts: torch.Tensor,
           k0: torch.Tensor, v0: torch.Tensor, *, in_place: bool = False) -> torch.Tensor | None:
    """States after the accepted rows, (streams, layers, Hv, Dv, Dk) fp32; ``in_place`` overwrites the
    table's state tensors, which nothing else may hold.  ``k0``/``v0`` give the heads' shapes, as for the ext."""

    Hk, dk = k0.shape[1], k0.shape[2]
    Hv, dv = v0.shape[1], v0.shape[2]
    rep = Hv // Hk
    held = table[4 * layers:]
    if len(held) != streams * layers:
        raise ValueError("one state per stream and layer")
    out = None if in_place else torch.empty((streams, layers, Hv, dv, dk), dtype=torch.float32, device=v0.device)
    take, counts_l = rows.tolist(), counts.tolist()
    for s in range(streams):
        path = take[s][:counts_l[s]]
        for l in range(layers):
            k_l, v_l = table[4 * l], table[4 * l + 1]
            g_l, beta_l = table[4 * l + 2], table[4 * l + 3]
            stepped = held[s * layers + l]
            for r in path:
                kr = k_l[r].float().repeat_interleave(rep, -2)
                stepped, _ = _step(stepped, kr, kr, v_l[r].float(), g_l[r].float(), beta_l[r].float())
            if in_place:
                held[s * layers + l].copy_(stepped)
            else:
                out[s, l] = stepped
    return out
