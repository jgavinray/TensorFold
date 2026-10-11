"""The CPU reference walk behind the serving gate's oracle, shared by ``exp/make_oracle.py``.

The oracle is the engine gate's rule - the same source run on CPU tensors - with the host's
routed experts served by a bounded LRU over the same dequantized bytes: the shipped pool's
floor (``2 * experts`` fp32 slots) would breach the box's memory guard twice over on the host.

The gates never run this walk inline; ``exp/make_oracle.py`` computes every greedy sequence
once into ``exp/oracle_tokens.json`` and the suite only reads that fixture.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import resource
import sys
import time
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("TRITON_CACHE_DIR", "/storage01/.triton-cache-serve")   # the root fs is full
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import tensorfold.families.qwen4_exp.xpu.moe  # noqa: E402,F401  (import xpu.moe and xpu.nvfp4 before triton)
import tensorfold.families.qwen4_exp.xpu.nvfp4 as xpu_nvfp4  # noqa: E402
from tensorfold.families.qwen4_exp.xpu import stream  # noqa: E402
from tensorfold.families.qwen4_exp.xpu import weights as xpu_weights  # noqa: E402
from tensorfold.families.qwen4_exp.xpu import forward as fw  # noqa: E402
from tensorfold.families.qwen4_exp.xpu.state import Buffers, State  # noqa: E402
from tensorfold.families.qwen4_exp.xpu.engine import GDN_CHUNK  # noqa: E402

CKPT = "/storage01/models/Qwen3.8-Flash-Next-NVFP4"
SEED = 42
HOST_LRU = int(os.environ.get("TF_SERVE_HOST_LRU", "48"))   # oracle tables resident at once
ORACLE_CHUNK = int(os.environ.get("TF_ORACLE_CHUNK", "16"))  # oracle prompt rows per pass (bounds its peak)
RAM_FLOOR_GB = float(os.environ.get("TF_ORACLE_FLOOR", "1.5"))  # the walk aborts below this much free RAM


def mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def sha256_json(obj) -> str:
    """The fixture's identity hash: sha256 over the canonical JSON of the object."""

    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def log(msg: str) -> None:
    print(f"[oracle] {msg} | RSS {rss_gb():.1f} GiB avail {mem_available_gb():.1f} GiB", flush=True)


_CFG = None


def cfg():
    global _CFG
    if _CFG is None:
        _CFG = xpu_weights.Config.read(CKPT)
    return _CFG


def prompt_tokens(n: int, seed: int = SEED) -> list[int]:
    """The fixed prompt: ``numpy.random.default_rng(seed)`` integers over the checkpoint's vocab."""

    return [int(t) for t in np.random.default_rng(seed).integers(0, int(cfg().vocab), n,
                                                                  dtype=np.int64)]


class HostWalk:
    """A ``Streamer`` stand-in for the CPU walk: its pool is a bounded host LRU of dequantized tables.

    The bytes and the dequantization are the shipped loader's (``stream.sources``,
    ``nvfp4.dequantize``), so every table equals a card slot's; only the residency differs."""

    def __init__(self, found: dict, shapes: dict, experts: int, top_k: int, dims: int, width: int,
                 cap: int) -> None:
        self.found, self.shapes = found, shapes
        self.experts, self.top_k = experts, top_k
        self.dims, self.width = dims, width
        self.device = torch.device("cpu")
        self.cap = int(cap)
        self.lru: OrderedDict[tuple[int, int], dict] = OrderedDict()
        self.hits = self.misses = 0
        self.bytes_read = 0
        self.fds = {p: stream._uncached(p) for p in {s.path for s in found.values()}}
        self._kind, self._rows, self._gen = "rows", 0, -1
        self._buffers: dict[str, object] = {"rows": None, "window": None}

    def begin(self, kind: str, rows: int, buf=None) -> None:
        self._kind, self._rows = kind, int(rows)
        if buf is not None:
            self._buffers[kind] = buf
        self._gen += 1

    def route(self, layer: int, kind: str, rows: int) -> dict[int, dict]:
        buf = self._buffers[kind]
        got = buf.pick[:rows, :self.top_k].to("cpu").numpy().reshape(-1).astype(np.int64)
        ids = sorted({int(e) for e in got if int(e) != self.experts})
        table: dict[int, dict] = {}
        for e in ids:
            key = (layer, e)
            t = self.lru.pop(key, None)
            if t is None:
                t = self._read(layer, e)
                self.misses += 1
                while len(self.lru) >= self.cap:
                    self.lru.popitem(last=False)
            else:
                self.hits += 1
            self.lru[key] = t
            table[e] = t
        return table

    def _read(self, layer: int, expert: int) -> dict:
        out = {}
        for proj in stream.PROJS:
            def part(name_part: str) -> torch.Tensor:
                src = self.found[(layer, proj, name_part, expert)]
                raw = os.pread(self.fds[src.path], src.nbytes, src.offset)
                self.bytes_read += src.nbytes
                if hasattr(os, "posix_fadvise"):   # our read-through pages must not eat the box's headroom
                    try:
                        os.posix_fadvise(self.fds[src.path], src.offset, src.nbytes,
                                         os.POSIX_FADV_DONTNEED)
                    except OSError:
                        pass
                dtype = self.shapes[(proj, name_part)][1]
                t = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                return (t.view(torch.float32) if dtype == "F32"
                        else t.view(torch.float8_e4m3fn) if dtype == "F8_E4M3"
                        else t.view(torch.bfloat16) if dtype == "BF16" else t)

            words = part("weight").view(self.shapes[(proj, "weight")][0])
            scales = part("scales").view(self.shapes[(proj, "scales")][0])
            scale2 = float(part("scales_2").reshape(-1)[0])
            out[proj] = xpu_nvfp4.dequantize(words, scales, scale2).t().contiguous()
        return out

    def close(self) -> None:
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


class Host4:
    """The ``Pooled4`` the CPU walk reads: this pass' routed tables plus the resident shared expert."""

    def __init__(self, walk: HostWalk, layer: int, shared: tuple) -> None:
        self.walk, self.layer = walk, layer
        self.shared_gate_t, self.shared_up_t, self.shared_down_t = shared
        self.gate_t, self.up_t, self.down_t = (_HostFace(self, "gate"), _HostFace(self, "up"),
                                               _HostFace(self, "down"))
        self._gen, self._slots = -1, {}

    @property
    def routed(self) -> int:
        return self.walk.experts

    @property
    def dims(self) -> int:
        return self.walk.dims

    @property
    def width(self) -> int:
        return self.walk.width

    @property
    def shared_width(self) -> int:
        return int(self.shared_gate_t.shape[1])

    @property
    def device(self):
        return self.walk.device

    def ensure(self) -> None:
        if self._gen != self.walk._gen:
            self._slots = self.walk.route(self.layer, self.walk._kind, self.walk._rows)
            self._gen = self.walk._gen

    def table(self, proj: str, expert: int) -> torch.Tensor:
        self.ensure()
        return self._slots[expert][proj]


class _HostFace:
    def __init__(self, owner: Host4, proj: str) -> None:
        self.owner, self.proj = owner, proj

    def __getitem__(self, expert: int) -> torch.Tensor:
        return self.owner.table(self.proj, int(expert))


_WALK: dict = {}


def cpu_walk_tokens(prompt: list[int], count: int) -> list[int]:
    """The oracle: the same forward on host tensors, chunked prompt and single-row decode, greedy."""

    key = ("walk", tuple(prompt), count)
    if key in _WALK:
        return _WALK[key]
    if mem_available_gb() < RAM_FLOOR_GB:
        raise RuntimeError(f"MemAvailable {mem_available_gb():.1f} GB < {RAM_FLOOR_GB} GB floor: "
                           f"the CPU walk would risk the box")
    base = stream._patch_reader(xpu_weights, stream.base_names(Path(CKPT)))
    try:
        w = xpu_weights.load(CKPT, "cpu", mtp=False)
    finally:
        xpu_weights._Reader = base
    cfg_ = w.cfg
    walk = HostWalk(*stream.sources(CKPT, stream.expert_names(cfg_.layers, cfg_.experts,
                                                              stream.base_names(Path(CKPT)))),
                    cfg_.experts, cfg_.top_k, cfg_.hidden, cfg_.moe_width, HOST_LRU)
    for layer in w.layers:
        shared = layer.moe.experts["shared"]
        layer.moe.experts = Host4(
            walk, layer.index,
            (shared[0].to(torch.float32).t().contiguous(),
             shared[1].to(torch.float32).t().contiguous(),
             shared[2].to(torch.float32).t().contiguous()))
    capacity = len(prompt) + count + 16
    st = State(w, capacity, max(8, min(ORACLE_CHUNK, GDN_CHUNK)), "bf16")
    bp = Buffers(w, max(16, ORACLE_CHUNK), capacity, prefill=True)
    bd = Buffers(w, 8, capacity)
    t0 = time.perf_counter()
    seq: list[int] = []
    for start in range(0, len(prompt), ORACLE_CHUNK):
        rows = list(prompt[start:start + ORACLE_CHUNK])
        walk.begin("rows", len(rows), bp.moe)
        last = start + ORACLE_CHUNK >= len(prompt)
        lg = fw.forward(w, st, bp, rows, logits=last)
        if last:
            seq = [int(lg[-1:].argmax(dim=-1)[0])]
        fw.commit(w, st, bp, len(rows), len(rows))
        log(f"prefill {start + len(rows)}/{len(prompt)} +{time.perf_counter() - t0:.0f} s "
            f"hits {walk.hits} misses {walk.misses}")
        t0 = time.perf_counter()
    bp = None                       # the decode leg must not carry the prompt's buffers
    walk.lru.clear()                # or the prompt's route tables - the steps re-route anyway
    gc.collect()
    for i in range(count):
        walk.begin("rows", 1, bd.moe)
        lg = fw.forward(w, st, bd, [seq[-1]])[:1].clone()
        seq.append(int(lg.argmax(dim=-1)[0]))
        fw.commit(w, st, bd, 1, 1)
        if (i + 1) % 8 == 0 or i == count - 1:
            log(f"decode {i + 1}/{count} +{time.perf_counter() - t0:.0f} s "
                f"hits {walk.hits} misses {walk.misses}")
            if mem_available_gb() < RAM_FLOOR_GB:
                raise RuntimeError(f"MemAvailable fell under {RAM_FLOOR_GB} GB mid-walk; aborting")
            t0 = time.perf_counter()
    log(f"walk done ({len(prompt)} + {count}): read {walk.bytes_read / 2**30:.1f} GiB, "
        f"{walk.hits} hits {walk.misses} misses")
    walk.close()
    _WALK[key] = seq
    return seq
