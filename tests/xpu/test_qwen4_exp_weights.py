"""Slice 1's gate: the XPU modelopt loader against the tiny fixture and the real checkpoint's index ledger.

The tiny NVFP4 fixture loads on the Intel GPU and is checked for structure, shapes, dtypes and devices; the
real checkpoint loads on the host (the CPU-fallback path) and is verified against its own safetensors index —
names, shapes, dtypes and raw bytes — plus a device-invariance spot-check of the reference dequantize.
No forward pass anywhere.
"""

import json
import struct
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda import qmm  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.reader import _Reader  # noqa: E402
from tensorfold.families.qwen4_exp.host_table import FP8Table, HostTable  # noqa: E402
from tensorfold.families.qwen4_exp.xpu import load  # noqa: E402
from tensorfold.families.qwen4_exp.xpu.weight_types import AttnW, GDNW, MoEW, PLEW  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[1] / "cuda"))
from nvfp4_tiny import write  # noqa: E402

REAL = Path("/storage01/models/Qwen3.8-Flash-Next-NVFP4")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    return write(tmp_path_factory.mktemp("xpu-tiny"))


def _tensors(x):
    """Every torch tensor in a loaded structure: dataclass fields, tuples, lists and dicts walked."""

    if isinstance(x, torch.Tensor):
        yield x
    elif hasattr(x, "__dataclass_fields__"):
        for f in x.__dataclass_fields__:
            yield from _tensors(getattr(x, f))
    elif isinstance(x, dict):
        for v in x.values():
            yield from _tensors(v)
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _tensors(y)


def _all_on(x, kind: str) -> bool:
    return all(t.device.type == kind for t in _tensors(x))


def test_tiny_structure_on_xpu(tiny: Path) -> None:
    w = load(tiny, device="xpu", mtp=True)
    assert w.cfg.quant == "modelopt" and len(w.layers) == 2 and _all_on(w, "xpu")
    assert w.embed[0].shape == (256, 256) and w.embed[0].dtype is torch.bfloat16
    assert w.head.b.weight.shape == (256, 256) and w.head.b.weight.device.type == "xpu"
    assert w.inv_freq.shape == (8,) and w.inv_freq.dtype is torch.float32
    assert w.mixer.down.b.weight.shape == (64, 1024) and w.mixer.up.b.weight.shape == (1024, 64)
    assert w.mixer.scale.dtype is torch.float32 and w.mixer.inject is False
    g = w.layers[0].gdn
    assert w.layers[0].linear and w.layers[0].attn is None and isinstance(g, GDNW)
    assert g.proj.b.weight.shape == (8240, 256) and g.out.b.weight.shape == (256, 3072)
    assert g.conv.shape == (5120, 4) and g.conv.dtype is torch.bfloat16
    assert g.a_log.shape == (24,) and g.a_log.dtype is torch.float32
    assert g.dt_bias.shape == (24,) and g.norm.shape == (128,) and g.norm.dtype is torch.bfloat16
    a = w.layers[1].attn
    assert w.layers[1].gdn is None and isinstance(a, AttnW)
    assert a.proj.b.weight.shape == (1152, 256) and a.o.b.weight.shape == (256, 128)
    assert a.q_scale.shape == (64,) and a.ik_scale.shape == (128,) and a.q_scale.dtype is torch.float32
    p = w.layers[1].ple
    assert isinstance(p, PLEW) and isinstance(p.table, HostTable)
    assert p.key.b.weight.shape == (1024, 64) and p.value.b.weight.shape == (256, 64)
    assert p.conv.shape == (1024, 4) and p.table.rows > 0
    for layer in w.layers:
        m = layer.moe
        assert isinstance(m, MoEW) and m.router.shape == (3, 256) and m.router.dtype is torch.bfloat16
        ex = m.experts
        assert set(ex) == {"gate", "up", "down", "shared", "router"}
        for key in ("gate", "up"):
            words, scales, s2 = ex[key]
            assert words.shape == (2, 128, 128) and words.dtype is torch.uint8
            assert scales.shape == (2, 128, 16) and scales.dtype is torch.float8_e4m3fn
            assert s2.shape == (2,) and s2.dtype is torch.float32
        words, scales, _ = ex["down"]
        assert words.shape == (2, 256, 64) and scales.shape == (2, 256, 8)
        assert all(t.dtype is torch.bfloat16 and t.device.type == "xpu" for t in ex["shared"])
        assert ex["shared"][2].shape == (256, 128)


def test_tiny_mtp_bf16_experts_stack(tiny: Path) -> None:
    """The MTP layer's BF16 stacked experts land as gate/up/down rows split from the fused gate_up."""
    w = load(tiny, device="xpu", mtp=True)
    assert w.mtp is not None and _all_on(w.mtp, "xpu")
    ex = w.mtp.layer.moe.experts
    assert ex["gate"].shape == (2, 128, 256) and ex["up"].shape == (2, 128, 256)
    assert ex["down"].shape == (2, 256, 128) and ex["gate"].dtype is torch.bfloat16
    raw = _Reader(tiny, "cpu").get("mtp.layers.0.mlp.experts.gate_up_proj")
    assert torch.equal(ex["gate"][0].cpu(), raw[0, :128]) and torch.equal(ex["up"][1].cpu(), raw[1, 128:])


def test_tiny_fp8_mtp_experts_requantize(tmp_path: Path) -> None:
    """The MTP drafter's FP8 experts dequantize at load: codes x scale in fp32, rounded once to bf16."""
    dir_ = write(tmp_path / "fp8-mtp", mtp_experts="fp8")
    w = load(dir_, device="xpu", mtp=True)
    ex = w.mtp.layer.moe.experts
    assert ex["gate"].shape == (2, 128, 256) and ex["gate"].dtype is torch.bfloat16
    rd = _Reader(dir_, "cpu")
    for proj, got in (("gate_proj", ex["gate"]), ("down_proj", ex["down"])):
        codes = rd.get(f"mtp.layers.0.mlp.experts.0.{proj}.weight")
        scale = rd.get(f"mtp.layers.0.mlp.experts.0.{proj}.weight_scale")
        want = (codes.float() * scale.float().reshape(-1, 1)).to(torch.bfloat16)
        assert torch.equal(got[0].cpu(), want)


def _add_fp8_expert(dir_: Path) -> None:
    """Rewrite the single shard with an FP8 routed expert added to a main layer (the refusal's trigger)."""

    shard = dir_ / "model-00001-of-00001.safetensors"
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        blob = f.read()
    runs = [(name, e["dtype"], e["shape"], blob[e["data_offsets"][0]:e["data_offsets"][1]])
            for name, e in header.items() if name != "__metadata__"]
    payload = torch.zeros(128, 256, dtype=torch.float8_e4m3fn).view(torch.uint8).numpy().tobytes()
    runs.append(("model.layers.0.mlp.experts.0.gate_proj.weight", "F8_E4M3", [128, 256], payload))
    entries, at = {}, 0
    for name, dtype, shape, raw in runs:
        entries[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + len(raw)]}
        at += len(raw)
    header = json.dumps({"__metadata__": {"format": "pt"}, **entries}, separators=(",", ":")).encode()
    with open(shard, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        for *_, raw in runs:
            f.write(raw)


def test_fp8_routed_experts_outside_mtp_refuse(tmp_path: Path) -> None:
    dir_ = write(tmp_path / "fp8-main")
    _add_fp8_expert(dir_)
    with pytest.raises(ValueError, match="FP8 routed experts"):
        load(dir_, device="cpu", mtp=False)


def test_ple_on_ssd_refuses_on_modelopt(tiny: Path) -> None:
    with pytest.raises(ValueError, match="--ple-on-ssd"):
        load(tiny, device="cpu", mtp=False, ple_on_ssd=True)


def test_dequantize_is_device_invariant(tiny: Path) -> None:
    """qmm.dequantize on the host vs the same words on the card: exact integer math, the same bits."""

    w = load(tiny, device="cpu", draft_vocab=64)      # the draft head is the loader's MLX-format 4-bit copy
    words, scales, biases = qmm.to_mlx(w.draft_head)
    host = qmm.dequantize(words, scales, biases).float()
    card = qmm.dequantize(words.to("xpu"), scales.to("xpu"), biases.to("xpu")).float()
    assert torch.equal(host, card.cpu())


def _mbase(rd: _Reader) -> str:
    return "model.language_model." if rd.has("model.language_model.embed_tokens.weight") else "model."


@pytest.fixture(scope="module")
def real():
    if not (REAL / "model.safetensors.index.json").is_file():
        pytest.skip("the real checkpoint is not mounted")
    return load(REAL, device="cpu", mtp=False)


def test_real_loads_cpu_fallback(real) -> None:
    assert real.cfg.quant == "modelopt" and len(real.layers) == real.cfg.layers == 48
    assert real.mtp is None and _all_on(real, "cpu")


def test_real_embed_lm_head_mixer(real) -> None:
    rd = _Reader(REAL, "cpu")
    base = _mbase(rd)
    assert torch.equal(real.embed[0], rd.get(base + "embed_tokens.weight").to(torch.bfloat16))
    assert torch.equal(real.head.b.weight, rd.get("lm_head.weight").to(torch.bfloat16))
    for field, name in (("down", "input_mix_weight_down"), ("up", "input_mix_weight_up")):
        raw = rd.get(base + f"hyper_connection_mixer.{name}.weight").to(torch.bfloat16)
        assert torch.equal(getattr(real.mixer, field).b.weight, raw)
    norm = rd.get(base + "hyper_connection_mixer.hc_norm.weight").float()
    assert torch.equal(real.mixer.scale, norm if real.around_one else 1.0 + norm)
    assert real.mixer.inject is False


def test_real_gdn_layer_ledger(real) -> None:
    """The GDN layer's fields are the index's bytes: rows stacked in order, the conv channels permuted."""
    rd = _Reader(REAL, "cpu")
    base = _mbase(rd)
    full = real.meta["full"]
    i = real.cfg.layer_types.index("linear")
    g = real.layers[i].gdn
    assert isinstance(g, GDNW) and real.layers[i].linear
    b = f"{base}layers.{i}.linear_attn."
    want = torch.cat([rd.get(b + n).to(torch.bfloat16) for n in
                      ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight")])
    assert torch.equal(g.proj.b.weight, want)
    channels = torch.cat([torch.arange(full.nk * full.dk), full.nk * full.dk + torch.arange(full.nk * full.dk),
                          2 * full.nk * full.dk + torch.arange(full.nv * full.dv)])
    conv = rd.get(b + "conv1d.weight").reshape(full.conv_dim, full.conv_kernel).to(torch.bfloat16)
    assert torch.equal(g.conv, conv.index_select(0, channels).contiguous())
    assert torch.equal(g.a_log, rd.get(b + "A_log").float())
    assert torch.equal(g.dt_bias, rd.get(b + "dt_bias").float())
    assert torch.equal(g.norm, rd.get(b + "norm.weight").to(torch.bfloat16))
    assert torch.equal(g.out.b.weight, rd.get(b + "out_proj.weight").to(torch.bfloat16))


def test_real_attention_layer_ledger(real) -> None:
    rd = _Reader(REAL, "cpu")
    base = _mbase(rd)
    i = real.cfg.layer_types.index("attention")
    a = real.layers[i].attn
    assert isinstance(a, AttnW) and real.layers[i].gdn is None
    b = f"{base}layers.{i}.self_attn."
    want = torch.cat([rd.get(b + n).to(torch.bfloat16) for n in
                      ("q_proj.weight", "k_proj.weight", "v_proj.weight", "indexer.index_qk_proj.weight")])
    assert torch.equal(a.proj.b.weight, want)
    assert torch.equal(a.o.b.weight, rd.get(b + "o_proj.weight").to(torch.bfloat16))
    for field, name in (("q_scale", "q_norm.weight"), ("k_scale", "k_norm.weight"),
                        ("iq_scale", "indexer.q_layernorm.weight"), ("ik_scale", "indexer.k_layernorm.weight")):
        raw = rd.get(b + name).float()
        assert torch.equal(getattr(a, field), raw if real.around_one else 1.0 + raw)


def test_real_moe_layer_ledger(real) -> None:
    """The MoE's raw stacks: the router rows, the shared expert's bf16 and the first/last expert's FP4 bytes."""
    rd = _Reader(REAL, "cpu")
    base = _mbase(rd)
    full = real.meta["full"]
    e = full.experts
    moe = real.layers[0].moe
    assert isinstance(moe, MoEW)
    b = f"{base}layers.0.mlp."
    router = torch.cat([rd.get(b + "gate.weight").to(torch.bfloat16),
                        rd.get(b + "shared_expert_gate.weight").to(torch.bfloat16)])
    assert torch.equal(moe.router, router) and torch.equal(moe.experts["router"], router)
    shared = moe.experts["shared"]
    assert torch.equal(shared[0], rd.get(b + "shared_expert.gate_proj.weight").to(torch.bfloat16))
    assert torch.equal(shared[1], rd.get(b + "shared_expert.up_proj.weight").to(torch.bfloat16))
    assert torch.equal(shared[2], rd.get(b + "shared_expert.down_proj.weight").to(torch.bfloat16))
    assert shared[0].shape == (full.shared_width, full.hidden)
    assert shared[2].shape == (full.hidden, full.shared_width)
    for proj, key in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
        w_, s, s2 = moe.experts[key]
        n, k = (full.moe_width, full.hidden) if key != "down" else (full.hidden, full.moe_width)
        assert w_.dtype is torch.uint8 and s.dtype is torch.float8_e4m3fn and s2.dtype is torch.float32
        assert tuple(w_.shape) == (e, n, k // 2) and tuple(s.shape) == (e, n, k // 16) and tuple(s2.shape) == (e,)
        for j in (0, e - 1):
            assert torch.equal(w_[j], rd.get(b + f"experts.{j}.{proj}.weight"))
            assert torch.equal(s[j].view(torch.uint8), rd.get(b + f"experts.{j}.{proj}.weight_scale").view(torch.uint8))
            assert torch.equal(s2[j], rd.get(b + f"experts.{j}.{proj}.weight_scale_2"))


def test_real_ple_layer_ledger(real) -> None:
    rd = _Reader(REAL, "cpu")
    base = _mbase(rd)
    i = real.cfg.ple_layers[0]
    ple = real.layers[i].ple
    assert isinstance(ple, PLEW)
    b = f"{base}layers.{i}.ple."
    conv = rd.get(b + "conv1d.weight").reshape(real.cfg.streams * real.cfg.hidden, real.cfg.ple_kernel)
    assert torch.equal(ple.conv, conv.to(torch.bfloat16).contiguous())
    assert torch.equal(ple.key.b.weight, rd.get(b + "key_proj.weight").to(torch.bfloat16))
    assert torch.equal(ple.value.b.weight, rd.get(b + "value_proj.weight").to(torch.bfloat16))
    for field, name in (("norm_key", "norm_key.weight"), ("norm_query", "norm_query.weight"),
                        ("norm_conv", "norm_conv.weight")):
        raw = rd.get(b + name).float()
        assert torch.equal(getattr(ple, field), raw if real.around_one else 1.0 + raw)
    ng = real.cfg.ngram(0)
    assert isinstance(ple.table, FP8Table) and ple.table.rows == ng.rows
    assert getattr(ple.table, "width", ng.dims) == ng.dims
