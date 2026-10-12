"""Routed experts streamed from the checkpoint into a card-resident slot pool; the rest stays resident.

The torch port of ``tensorfold/streaming/pool.py``: that module answers the routing with MLX arrays and
hostsync signals (``import mlx.core`` at its top), so an XPU box without the MLX wheel cannot import it
at all. The slot-pool semantics are ported here on plain torch staging instead: the same ``Source``
discovery, the same LRU behind ``expert`` ids, the same window slots for prompt chunks, and the same
``hits / misses / bytes_read`` bookkeeping, but the pool stacks hold the *dequantized fp32* tables the
XPU grouped matmul reads (``nvfp4.Stacked4``'s layout, one expert per slot), filled by host ``pread``
of the checkpoint's packed NVFP4 bytes, uploaded, and dequantized on card with the resident
loader's own ``nvfp4.dequantize`` call - only the ~2.8 MB of packed bytes cross the PCIe bus per
expert (not the 59 MB of fp32), the decode bits are the loader's by construction, and a streamed
expert is ``torch.equal`` to the resident one.

The CUDA/MLX engine hands the routing to a host thread through graph marks because its expert launch
reads its plan on the host mid-graph; the XPU engine decodes eagerly, so ``route`` answers on the
calling thread: the card wrote ``pick`` in this very MoE call, the host reads it back, loads every
picked expert into the pool, and only then does ``grouped_moe`` slice a table (``Pooled4`` defers
those slices to this step's slot map)."""

from __future__ import annotations

import json
import os
import struct
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


from . import nvfp4

PARTS = ("weight", "scales", "scales_2")          # one projection's checkpoint tensors
_SUFFIX = {"weight": "weight", "scales": "weight_scale", "scales_2": "weight_scale_2"}
PROJS = ("gate", "up", "down")
PROJECT = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
_DTYPES = {"U8": torch.uint8, "I8": torch.int8, "U32": torch.int32, "I32": torch.int32,
           "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
           "F8_E4M3": torch.float8_e4m3fn}


@dataclass
class Source:
    """Where one expert's tensor bytes start in a checkpoint file, and how many there are."""

    path: Path
    offset: int
    nbytes: int


def expert_names(layers: int, experts: int, base: str) -> dict[tuple, str]:
    """Pool sources keyed (layer, proj, part, expert): the checkpoint's per-expert NVFP4 tensors."""

    return {(i, proj, part, e): f"{base}layers.{i}.mlp.experts.{e}.{PROJECT[proj]}.{_SUFFIX[part]}"
            for i in range(layers) for proj in PROJS for part in PARTS for e in range(experts)}


def base_names(model_dir: Path) -> str:
    """The decoder layers' name base: NVFP4 groups the language model under ``model.language_model.``."""

    index = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    if "language_model.model.embed_tokens.weight" in index:
        return "language_model.model."
    if "model.language_model.embed_tokens.weight" in index:
        return "model.language_model."
    return "model."


def sources(model_dir: Path, names: dict[tuple, str]) -> tuple[dict, dict]:
    """Sources and each ``(proj, part)``'s (shape, dtype-string) for the keys of ``expert_names``."""

    model_dir = Path(model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    heads: dict[str, tuple[Path, int, dict]] = {}
    for shard in {index[name] for name in names.values() if name in index}:
        path = model_dir / shard
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            heads[shard] = (path, 8 + n, json.loads(f.read(n)))
    found, shapes = {}, {}
    for key, name in names.items():
        shard = index.get(name)
        if shard is None:
            raise ValueError(f"expert tensor {name}: not in the checkpoint index")
        path, base, head = heads[shard]
        entry = head[name]
        begin, end = entry["data_offsets"]
        found[key] = Source(path, base + begin, end - begin)
        spec = (tuple(entry["shape"]), entry["dtype"])
        if shapes.setdefault((key[1], key[2]), spec) != spec:
            raise ValueError(f"expert tensor {name}: shape or dtype differs between layers or experts")
    return found, shapes


def expert_nbytes(found: dict, layer: int) -> int:
    """Checkpoint bytes of one expert of ``layer``: every projection's weight, scales and scale_2."""

    return sum(src.nbytes for key, src in found.items() if key[0] == layer and key[3] == 0)


def slot_nbytes(dims: int, width: int) -> int:
    """Bytes of one expert in the pool: its dequantized fp32 gate, up and down tables."""

    return 3 * dims * width * 4


def _uncached(path: Path) -> int:
    """A read-only fd that skips readahead: the pool caches, the page cache should not."""

    fd = os.open(path, os.O_RDONLY)
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
    return fd


def _is_routed(name: str) -> bool:
    """A decoder layer's per-expert tensor (the MTP head's stacked experts are resident, not streamed)."""

    tail = name.split(".mlp.")[-1] if ".mlp.experts." in name else ""
    return ".mlp.experts." in name and tail.startswith("experts.") \
        and not name.startswith("mtp.") and not name.startswith("model.mtp.") \
        and not name.startswith("language_model.mtp.")


class Streamer:
    """Slots ``[0, E)``: one layer's window (expert e in slot e) for prompt chunks; ``[E, S)``: an LRU
    for decode rows. ``route`` answers one MoE call from the ``pick`` rows the card just wrote."""

    def __init__(self, found: dict, shapes: dict, *, device, layers: int, experts: int, top_k: int,
                 slots: int, dims: int, width: int, workers: int = 8) -> None:
        if slots < 2 * experts:
            raise ValueError(f"the expert pool needs a layer's {experts} window slots plus as many for decode, "
                             f"got {slots}")
        self.found, self.shapes = found, shapes
        self.layers, self.experts, self.top_k = layers, experts, top_k
        self.dims, self.width, self.device = dims, width, torch.device(device)
        self.pool = {
            "gate": torch.zeros((slots, dims, width), dtype=torch.float32, device=self.device),
            "up": torch.zeros((slots, dims, width), dtype=torch.float32, device=self.device),
            "down": torch.zeros((slots, width, dims), dtype=torch.float32, device=self.device),
        }
        self.fds = {path: _uncached(path) for path in {s.path for s in found.values()}}
        self.window: list[int] = [-1] * experts                     # the layer whose expert sits in slot e
        self.lru: OrderedDict[tuple[int, int], int] = OrderedDict()  # (layer, expert) -> slot, oldest first
        self.free = list(range(slots - 1, experts - 1, -1))
        self.hits = self.misses = self.bytes_read = 0
        self.window_fills = 0
        self.calls: list[tuple[int, str, tuple[int, ...]]] = []     # (layer, kind, routed ids) per MoE call
        self.error: BaseException | None = None
        self._readers = ThreadPoolExecutor(workers, thread_name_prefix="expert-read")
        self._buffers: dict[str, Any] = {"rows": None, "window": None}
        self._gen = -1
        self._kind, self._rows = "rows", 0

    # -- per-step routing ---------------------------------------------------------------------------

    def attach_buffers(self, rows_buf: Any, window_buf: Any) -> None:
        """The engine's MoE scratch per pass kind: ``pick`` rows read here answer each call."""

        self._buffers["rows"], self._buffers["window"] = rows_buf, window_buf

    def begin(self, kind: str, rows: int, buf: Any = None) -> None:
        """Start one forward pass: every layer's first table slice answers its routing afresh.

        ``buf`` is that pass' MoE scratch (its ``pick`` rows are what routing reads); without it the
        buffers from ``attach_buffers`` answer, which serves direct tests outside any engine."""

        if kind not in ("rows", "window"):
            raise ValueError(f"unknown expert pass {kind!r}, expected 'rows' or 'window'")
        self._kind, self._rows = kind, int(rows)
        if buf is not None:
            self._buffers[kind] = buf
        self._gen += 1

    def route(self, layer: int, kind: str, rows: int) -> dict[int, int]:
        """The slots this layer's ``rows`` picks need, loaded: expert -> slot (one host sync on ``pick``)."""

        if self.error is not None:
            raise RuntimeError("expert streaming stopped") from self.error
        buf = self._buffers[kind]
        if buf is None:
            raise RuntimeError("the engine never registered its MoE scratch (attach_buffers)")
        got = buf.pick[:rows, :self.top_k].to("cpu").numpy().reshape(-1).astype(np.int64)
        ids = sorted({int(e) for e in got if int(e) != self.experts})   # the shared expert stays resident
        for e in ids:
            if not 0 <= e < self.experts:
                raise ValueError(f"routed expert {e} is outside the model")
        self.calls.append((layer, kind, tuple(ids)))
        if kind == "window":
            self._fill_window(layer, ids)
            return {e: e for e in ids}
        return self._load(layer, ids)

    def _load(self, layer: int, ids: list[int]) -> dict[int, int]:
        """Every id resident for ``layer`` hits its LRU slot; the rest take free slots or evict, oldest first."""

        need, table = [], {}
        for e in ids:
            slot = self.lru.pop((layer, e), None)
            if slot is None:
                slot = self.free.pop() if self.free else self._evict({(layer, i) for i in ids})
                need.append((e, slot))
                self.misses += 1
            else:
                self.hits += 1
            self.lru[(layer, e)] = slot
            table[e] = slot
        self._read_all(layer, need)
        return table

    def _fill_window(self, layer: int, ids: list[int]) -> None:
        need = [(e, e) for e in ids if self.window[e] != layer]
        self._read_all(layer, need)
        self.window_fills += len(need)
        for e, _ in need:
            self.window[e] = layer

    def _evict(self, keep: set) -> int:
        for key in self.lru:
            if key not in keep:
                return self.lru.pop(key)
        raise MemoryError("the expert pool is smaller than one call's experts")

    def _read_all(self, layer: int, need: list[tuple[int, int]]) -> None:
        if not need:
            return
        # The worker threads only ``pread`` the packed bytes (pure host); every card launch - the
        # upload, the dequantize, the slot copy - is issued right here, on the engine's own thread.
        # Launching from pool workers races the main thread's event syncs (a walk can wedge with
        # the sync spinning while a worker waits for the GIL to enqueue what it waits for).
        items = [(proj, e, slot) for e, slot in need for proj in PROJS]
        for (proj, _e, slot), (words, scales, scale2) in zip(
                items, self._readers.map(lambda item: self._read_project(layer, *item), items)):
            self._fill_slot(proj, words, scales, scale2, slot)

    def _fill_slot(self, proj: str, words: torch.Tensor, scales: torch.Tensor,
                   scale2: float, slot: int) -> None:
        rows = nvfp4.dequantize(words.to(self.device), scales.to(self.device), scale2)
        self.pool[proj][slot].copy_(rows.t())

    def _read_project(self, layer: int, proj: str, expert: int, slot: int) -> tuple:
        """One expert's packed checkpoint bytes, read on the host: the card's dequantize of them
        (the resident loader's own ``nvfp4.dequantize``, so the loader's bits) is launched by
        ``_read_all`` on the engine thread."""

        def part(name_part: str) -> torch.Tensor:
            src = self.found[(layer, proj, name_part, expert)]
            raw = os.pread(self.fds[src.path], src.nbytes, src.offset)
            self.bytes_read += src.nbytes
            dtype = self.shapes[(proj, name_part)][1]
            t = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            if dtype == "F32":
                return t.view(torch.float32)
            if dtype == "F8_E4M3":
                return t.view(torch.float8_e4m3fn)
            if dtype == "BF16":
                return t.view(torch.bfloat16)
            return t                                                    # U8 words

        words = part("weight").view(self.shapes[(proj, "weight")][0])
        scales = part("scales").view(self.shapes[(proj, "scales")][0])
        scale2 = float(part("scales_2").reshape(-1)[0])
        return words, scales, scale2

    # -- inspection and teardown ----------------------------------------------------------------------

    def slot_of(self, layer: int, expert: int) -> int | None:
        """The slot holding one expert now (the LRU's, or the window's when its layer sits there)."""

        slot = self.lru.get((layer, expert))
        return slot if slot is not None else (expert if self.window[expert] == layer else None)

    def close(self) -> None:
        self.error = self.error or RuntimeError("streamer closed")
        self._readers.shutdown(wait=True)
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


class Pooled4:
    """A ``nvfp4.Stacked4`` read through the pool: expert ``e``'s tables are the slots the routing
    answered for the current pass, and ``grouped_moe`` slices a table only after it wrote its picks."""

    def __init__(self, streamer: Streamer, layer: int, shared: tuple[torch.Tensor, torch.Tensor,
                                                                     torch.Tensor]) -> None:
        self.streamer, self.layer = streamer, layer
        self.shared_gate_t, self.shared_up_t, self.shared_down_t = shared
        self.gate_t = _Face(self, "gate")
        self.up_t = _Face(self, "up")
        self.down_t = _Face(self, "down")
        self._gen = -1
        self._slots: dict[int, int] = {}

    @property
    def routed(self) -> int:
        return self.streamer.experts

    @property
    def dims(self) -> int:
        return self.streamer.dims

    @property
    def width(self) -> int:
        return self.streamer.width

    @property
    def shared_width(self) -> int:
        return int(self.shared_gate_t.shape[1])

    @property
    def device(self):
        return self.streamer.device

    def ensure(self) -> None:
        """One routing answer per pass: read the card's picks, load what they need, map experts to slots."""

        if self._gen != self.streamer._gen:
            self._slots = self.streamer.route(self.layer, self.streamer._kind, self.streamer._rows)
            self._gen = self.streamer._gen

    def table(self, proj: str, expert: int) -> torch.Tensor:
        self.ensure()
        return self.streamer.pool[proj][self._slots[expert]]


class _Face:
    """``Pooled4``'s ``gate_t``/``up_t``/``down_t``: indexing by expert resolves this pass's slot."""

    def __init__(self, owner: Pooled4, proj: str) -> None:
        self.owner, self.proj = owner, proj

    def __getitem__(self, expert: int) -> torch.Tensor:
        return self.owner.table(self.proj, int(expert))

    @property
    def shape(self):
        return self.owner.streamer.pool[self.proj].shape[1:]


def _patch_reader(module: Any, marker: str) -> Any:
    """Swap ``module._Reader`` for one that leaves the routed experts on disk: their tensors come back as
    meta-device placeholders (shapes and dtypes only, no bytes read), so a streamed load never uploads
    what the pool will stream. Returns the original class to restore."""

    base = module._Reader

    class _Reader(base):
        def queue(self, names):
            super().queue([n for n in names if not _is_routed(n)])

        def get(self, name):
            if _is_routed(name):
                shard = self.where[name]
                _, head = self._header(shard)
                entry = head[name]
                return torch.empty(tuple(entry["shape"]), dtype=_DTYPES[entry["dtype"]], device="meta")
            return super().get(name)

    module._Reader = _Reader
    return base


def attach(w: Any, model_dir: str | Path, *, gib: float | None = None, slots: int | None = None,
           workers: int = 8) -> Streamer:
    """Build the pool for ``gib`` GiB of slots (or exactly ``slots`` slots) and route every decoder MoE through it."""

    model_dir = Path(model_dir)
    c = w.cfg
    found, shapes = sources(model_dir, expert_names(c.layers, c.experts, base_names(model_dir)))
    dims, width = c.hidden, c.moe_width
    if slots is None:
        if gib is None:
            raise ValueError("streaming needs a slot count or a GiB budget")
        slots = int(gib * 2**30) // slot_nbytes(dims, width)
    streamer = Streamer(found, shapes, device=w.device, layers=c.layers, experts=c.experts,
                        top_k=c.top_k, slots=slots, dims=dims, width=width, workers=workers)
    for index, layer in enumerate(w.layers):
        moe = layer.moe
        if not isinstance(moe.experts, dict):
            continue                                      # an already-packed table
        shared = moe.experts["shared"]
        if any(getattr(t, "dtype", None) not in (torch.bfloat16, torch.float32) for t in shared):
            raise NotImplementedError("streamed shared experts read bf16 or fp32 faces only "
                                      "(an MXFP8 shared expert has no lane matmul here)")
        pooled = Pooled4(
            streamer, index,
            (shared[0].to(torch.float32).t().contiguous(),      # the shared expert stays resident
             shared[1].to(torch.float32).t().contiguous(),
             shared[2].to(torch.float32).t().contiguous()))
        moe.experts = pooled
        moe._xpu_pooled = pooled
    w.streamer = streamer
    return streamer


def load(model_dir: str | Path, device: str = "xpu", *, mtp: bool = False, gib: float = 16.0,
         slots: int | None = None, draft_vocab: int | str | None = None, ple_on_ssd: bool = False,
         workers: int = 8):
    """Load with the routed experts left on disk (never uploaded), then attach their pool: the weights
    a card cannot hold stream through slots at each MoE call's routing."""

    from . import weights as xpu_weights

    base = _patch_reader(xpu_weights, base_names(Path(model_dir)))
    try:
        w = xpu_weights.load(model_dir, device, mtp=mtp, draft_vocab=draft_vocab, ple_on_ssd=ple_on_ssd)
    finally:
        xpu_weights._Reader = base
    streamer = attach(w, model_dir, gib=gib, slots=slots, workers=workers)
    return w, streamer


__all__ = ["PARTS", "Pooled4", "Source", "Streamer", "attach", "expert_nbytes", "expert_names",
           "load", "slot_nbytes", "sources"]
