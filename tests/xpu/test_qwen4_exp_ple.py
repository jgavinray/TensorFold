"""Slice 2's gate: the PLE n-gram rows on the B70 - row ids against a naive oracle, gathers against the host tables.

The ids come from ``cuda/ngram.py``'s NGram (pure numpy, device-free); the gathers stage the tables' raw rows
over pinned memory and dequantize on the card (``xpu/ple.py``). The real checkpoint's tables are fp8
(F8_E4M3 rows, one bf16 table scale); its PLE layer sits at index 1 (``ple_layer_ids`` [2], one-indexed).
"""

import hashlib
import json
import resource
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda.ngram import NGram  # noqa: E402
from tensorfold.families.qwen4_exp.host_table import FP8Table, open_table  # noqa: E402
from tensorfold.families.qwen4_exp.xpu import load, ple  # noqa: E402
from tensorfold.families.qwen4_exp.xpu.weight_types import PLEW  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[1] / "cuda"))
from nvfp4_tiny import write  # noqa: E402

REAL = Path("/storage01/models/Qwen3.8-Flash-Next-NVFP4")
ROWS, DIMS = 1000, 160                    # the real checkpoint's row width: 160 values (dims) a head's row
PROMPT = np.random.default_rng(0).integers(0, 248320, 64).astype(np.int64)
PROMPT[::11] = 248044                     # the checkpoint's eos: a reset inside the fixed prompt


def _device() -> torch.device:
    """B70 #2: the XPU whose uuid hex carries 4400 (the selector pins the runtime; never a fixed index)."""

    for i in range(torch.xpu.device_count()):
        if "4400" in str(torch.xpu.get_device_properties(i).uuid):
            return torch.device("xpu", i)
    return torch.device("xpu")


def _free_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1 << 20)
    return 1e9


@pytest.fixture(scope="module", autouse=True)
def _ram_guard():
    """The slice's runs abort under 24 GB MemAvailable."""

    free = _free_gb()
    if free < 24:
        pytest.skip(f"MemAvailable {free:.1f} GB is under the 24 GB guard")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    return write(tmp_path_factory.mktemp("ple-tiny"))


@pytest.fixture(scope="module")
def real():
    if not (REAL / "model.safetensors.index.json").is_file():
        pytest.skip("the real checkpoint is not mounted")
    return load(REAL, device="cpu", mtp=False)


def _oracle(ng: NGram, history: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """Row ids straight from the spec: one position at a time, the segment start after the last eos, then the
    n-gram hash (the int64 wrapping written out) and the per-head modulus - no vectorization, no numpy semantics."""

    mask, half = (1 << 64) - 1, 1 << 63
    seq = [int(t) for t in history] + [int(t) for t in tokens]
    sizes, offs, mult = ng.head_sizes, ng.head_offsets, ng.multipliers
    out = []
    for i in range(len(history), len(seq)):
        seg = 0
        for j in range(i):
            if seq[j] == ng.eos:
                seg = j + 1
        rows = []
        for ngram in range(2, ng.n + 1):
            mixed = 0
            for p in range(ngram):
                src = i - p
                tok = seq[src] if src >= seg and src >= 0 else ng.eos
                mixed ^= (tok * int(mult[p])) & mask
            signed = mixed - (1 << 64) if mixed >= half else mixed
            first = (ngram - 2) * ng.per_ngram
            rows += [signed % int(sizes[first + h]) + int(offs[first + h]) for h in range(ng.per_ngram)]
        out.append(rows)
    return np.array(out, dtype=np.int64)


def _to_bf16(v: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 bits, round to nearest even (host_table's own formula)."""

    f = np.ascontiguousarray(v, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((f + 0x7FFF + ((f >> 16) & 1)) >> 16).astype(np.uint16)


def _bf16_value(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def _dequant4(words: np.ndarray, scales: np.ndarray, biases: np.ndarray, scale: float) -> np.ndarray:
    """The MLX 4-bit rows' values in bf16 bits: nibble x scale + bias (fp32, one rounding), then the table scale."""

    n = words.shape[0]
    q = ((words.reshape(n, -1, 1) >> np.arange(0, 32, 4, dtype=np.uint32)) & np.uint32(0xF)).astype(np.float32)
    v = _to_bf16(q.reshape(n, -1, 32) * _bf16_value(scales)[..., None] + _bf16_value(biases)[..., None])
    if scale != 1.0:
        v = _to_bf16(_bf16_value(v) * np.float32(scale))
    return v.reshape(n, -1)


def _finite_codes(rng: np.random.Generator, shape: tuple) -> np.ndarray:
    """e4m3 codes without the two NaN patterns (torch.equal is element-wise; a NaN would read as a mismatch)."""

    return rng.integers(0, 0x7F, shape).astype(np.uint8) | (rng.integers(0, 2, shape).astype(np.uint8) << 7)


def _shard(dir_: Path, entries: list[tuple[str, torch.Tensor]]) -> Path:
    """One safetensors file of ``entries`` (nvfp4_tiny's layout: a json header, then the bytes in order)."""

    names = {torch.uint8: "U8", torch.int32: "I32", torch.bfloat16: "BF16", torch.float32: "F32",
             torch.float8_e4m3fn: "F8_E4M3"}
    header, blobs, offset = {}, [], 0
    for name, t in entries:
        t = t.reshape(1).contiguous() if t.dim() == 0 else t.contiguous()
        header[name] = {"dtype": names[t.dtype], "shape": list(t.shape),
                        "data_offsets": [offset, offset + t.numel() * t.element_size()]}
        offset += t.numel() * t.element_size()
        blobs.append(t)
    blob = json.dumps(header).encode()
    path = dir_ / "tables.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for t in blobs:
            f.write(t.view(torch.uint8).numpy().tobytes())
    return path


def _ple(table, heads: int, dims: int) -> PLEW:
    """A minimal PLEW stand-in: the gather reads only the table and the n-gram constants."""

    ng = NGram(vocab=1024, ngram_size=3, heads_per_ngram=heads // 2, vocab_base=512, divisor=8, shards=4,
               seed=3, eos=0, embed_dim=heads * dims, ple_index=0)
    return PLEW(table, None, None, None, None, None, None, ng)


@pytest.mark.parametrize("kind", ["fp8", "fp8_lut", "bf16", "nvfp4", "mlx4"])
def test_gather_matches_the_host_reference(tmp_path, kind):
    """A table slice gathered on the card, bit-equal against the CUDA-family host tables after both sides go fp32."""

    heads, dims = (2, 160) if kind == "mlx4" else (16, 160)  # a row holds ``dims`` values; heads from the ids
    rng = np.random.default_rng(11 + ["fp8", "fp8_lut", "bf16", "nvfp4", "mlx4"].index(kind))
    ids = rng.integers(0, ROWS, (7, heads)).astype(np.int64)  # duplicates and disorder included
    dev = _device()
    key = "ngram_embedding.shard_0"
    if kind.startswith("fp8"):
        # a bf16-exact scale: the card's arithmetic dequant is provable against the LUT; the inexact one takes the LUT
        scale = float(torch.tensor(0.1, dtype=torch.bfloat16).float()) if kind == "fp8" \
            else float(np.float32(0.1234567))
        _shard(tmp_path, [(key + ".weight", torch.from_numpy(_finite_codes(rng, (ROWS, dims)))
                           .view(torch.float8_e4m3fn))])
        table = open_table(tmp_path, [("tables.safetensors", key)], lambda field: scale)
        ref = torch.from_numpy(table.gather(ids).view(np.int16)).view(torch.bfloat16)
    elif kind == "bf16":
        _shard(tmp_path, [(key + ".weight", torch.randn(ROWS, dims).to(torch.bfloat16))])
        table = open_table(tmp_path, [("tables.safetensors", key)], lambda field: 1.5)
        ref = (torch.from_numpy(table.gather(ids).view(np.int16)).view(torch.bfloat16).to(torch.float32)
               * 1.5).to(torch.bfloat16)
    elif kind == "nvfp4":
        _shard(tmp_path, [(key + ".weight", torch.from_numpy(rng.integers(0, 256, (ROWS, dims // 2),
                                                                           dtype=np.uint8))),
                          (key + ".weight_scale", torch.from_numpy(_finite_codes(rng, (ROWS, dims // 16)))
                           .view(torch.float8_e4m3fn))])
        table = open_table(tmp_path, [("tables.safetensors", key)], lambda field: 2.0 ** -9)
        ref = torch.from_numpy(table.gather(ids).view(np.int16)).view(torch.bfloat16)
    else:
        _shard(tmp_path, [(key + ".weight", torch.from_numpy(
                              rng.integers(-(2 ** 31), 2 ** 31, (ROWS, dims // 8), dtype=np.int32))),
                          (key + ".scales", torch.randn(ROWS, dims // 32).to(torch.bfloat16)),
                          (key + ".biases", torch.randn(ROWS, dims // 32).to(torch.bfloat16))])
        table = open_table(tmp_path, [("tables.safetensors", key)], lambda field: 0.5)
        ref = torch.from_numpy(_dequant4(*table.gather(ids), 0.5).view(np.int16)).view(torch.bfloat16)

    got = ple.gather(_ple(table, heads, dims), ids, dev)
    torch.xpu.synchronize(dev)
    assert got.shape == (7, heads, dims) and got.dtype is torch.bfloat16 and got.device == dev
    if kind == "fp8":
        assert ple._FP8[table][dev][0] == "scale"     # the proved arithmetic dequant, not the LUT
    assert torch.equal(got.reshape(7, -1).cpu().to(torch.float32), ref.reshape(7, -1).to(torch.float32))


def test_row_ids_match_the_oracle():
    """NGram.ids against the naive per-position oracle: random histories, EOS resets, several derived tables."""

    for seed in range(4):
        rng = np.random.default_rng(seed)
        ng = NGram(vocab=1000, ngram_size=3, heads_per_ngram=3, vocab_base=500, divisor=8, shards=4,
                   seed=seed, eos=0, embed_dim=96, ple_index=seed)
        hist = rng.integers(0, 1000, 2).astype(np.int64)
        hist[1] = 0                                   # a reset carried in from before the prompt
        toks = rng.integers(0, 1000, 200).astype(np.int64)
        toks[rng.random(200) < 0.15] = 0
        assert torch.equal(torch.from_numpy(ng.ids(hist, toks)), torch.from_numpy(_oracle(ng, hist, toks)))


def test_tiny_plew_gather(tiny):
    """The loader's own PLEW (the tiny fixture's MLX 4-bit table), gathered on the card against the host rows."""

    w = load(tiny, device=_device())
    p = w.layers[1].ple
    assert isinstance(p, PLEW) and p.ngram.heads == 2 and p.ngram.dims == 32
    toks = np.arange(1, 65) % 256
    toks[::7] = 0                                     # the fixture's eos: resets inside the prompt
    hist = p.ngram.initial_history()
    ids = p.ngram.ids(hist, toks)
    assert torch.equal(ple.ids(p, hist, toks), torch.from_numpy(ids))
    dev = _device()
    got = ple.gather(p, ids, dev)
    torch.xpu.synchronize(dev)
    assert got.shape == (64, 2, 32)
    assert torch.equal(ple.embed(p, ids, dev), got.reshape(64, 64))
    words, scales, biases = p.table.gather(ids)
    ref = torch.from_numpy(_dequant4(words, scales, biases, getattr(p.table, "weight_scale", 1.0)).view(np.int16))
    assert torch.equal(got.reshape(-1, 32).cpu().view(torch.int16), ref)


def test_real_ple_gather(real):
    """The real checkpoint's fp8 table: the first 64 tokens of a fixed prompt, ids and rows against the host
    reference - bit-equal, hashes and peak RSS reported."""

    p = real.layers[real.cfg.ple_layers[0]].ple
    assert isinstance(p, PLEW) and isinstance(p.table, FP8Table)
    ng = p.ngram
    hist = ng.initial_history()
    ids = ng.ids(hist, PROMPT)
    assert torch.equal(torch.from_numpy(ids), torch.from_numpy(_oracle(ng, hist, PROMPT)))
    dev = _device()
    rows = ple.gather(p, ids, dev)
    torch.xpu.synchronize(dev)
    ref = torch.from_numpy(p.table.gather(ids).view(np.int16)).view(torch.bfloat16)
    assert torch.equal(rows.reshape(-1, ng.dims).cpu().to(torch.float32), ref.to(torch.float32))
    assert torch.equal(ple.embed(p, ids, dev), rows.reshape(len(PROMPT), -1))
    flat = rows.reshape(len(PROMPT), -1).cpu().contiguous().view(torch.int16).numpy().tobytes()
    print(f"\nrow ids sha256 {hashlib.sha256(ids.tobytes()).hexdigest()}")
    print(f"gathered rows sha256 {hashlib.sha256(flat).hexdigest()}")
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20):.2f} GiB")
