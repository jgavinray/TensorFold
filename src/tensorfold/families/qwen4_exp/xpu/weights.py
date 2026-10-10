"""Load the ModelOpt (NVFP4) weights on XPU: the CUDA loader's checkpoint path with torch.xpu where it queued."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tensorfold.cuda.exl3.format import is_exl3

from ..cuda.bf16 import b16_from_rows, quantize4, stack_b16
from ..cuda.reader import _Reader, norms_around_one
from ..host_table import open_table, shard_keys
from .weight_types import (
    AttnW, Config, GDNW, HC, LayerW, MoEW, MTPW, PLEW, Weights, draft_token_ids)  # noqa: F401 (re-exported)


_PLAIN = (torch.bfloat16, torch.float16, torch.float32)


def _plain(name: str, w: torch.Tensor) -> torch.Tensor:
    """Refuse quantized bytes where a bf16 linear needs real values."""

    if w.dtype not in _PLAIN:
        raise ValueError(f"{name}: {str(w.dtype).removeprefix('torch.')} weights without a scale this loader reads; "
                         "Flash Next reads its non-expert linears as bf16, MXFP8 or 128x128-block FP8")
    return w


def _api(device: str | torch.device):
    """torch.xpu or torch.cuda when the weights land on a GPU, None on the host: the load's accelerator calls."""
    return {"xpu": torch.xpu, "cuda": torch.cuda}.get(torch.device(device).type)


def load(model_dir: str | Path, device: str = "xpu", *, mtp: bool = False, tp: tuple[int, int] | None = None,
         draft_vocab: int | str | None = None, ple_on_ssd: bool = False) -> Weights:
    """Load rank ``tp``'s shares; ``draft_vocab`` selects default/file ids or ids below N, None scores all ids."""

    import time
    from dataclasses import replace

    model_dir = Path(model_dir)
    if is_exl3(model_dir):
        raise NotImplementedError("slice 4")          # the EXL3 packs' own loader
    full = Config.read(model_dir)
    if full.quant == "mlx":
        raise NotImplementedError("slice 4")          # the MLX 4-bit checkpoint's affine loader
    rank, world = tp if tp is not None else (0, 1)
    cfg = full if world == 1 else replace(full, heads=full.heads // world, kv_heads=full.kv_heads // world,
                                          nk=full.nk // world, nv=full.nv // world,
                                          moe_width=full.moe_width // world, shared_width=full.shared_width // world)
    rd = _Reader(model_dir, device)
    api = _api(device)
    prefix = "language_model." if rd.has("language_model.model.embed_tokens.weight") else ""
    # NVFP4 names the language model ``model.language_model.*``; its lm_head and mtp sit at the top level
    mbase = "model.language_model." if rd.has("model.language_model.embed_tokens.weight") else "model."
    chosen = list(range(cfg.layers))
    around_one = norms_around_one(rd, prefix + mbase, chosen)

    def raw(name: str) -> torch.Tensor:
        return rd.get(prefix + name)

    def table_scale(base: str, field: str) -> float:
        name = base + "ngram_embedding." + field
        if not rd.has(prefix + name):
            return 1.0
        value = raw(name)
        if value.numel() != 1:
            raise ValueError(f"{name}: expected one n-gram table scale")
        return float(value.float().reshape(-1)[0])

    def cscale(name: str) -> torch.Tensor:
        w = raw(name).float()
        return (w if around_one else 1.0 + w).contiguous()

    def b16(name: str):
        """One linear as the NVFP4 checkpoint stores it (BF16, torch layout [out, in]), on the qmm matmul face."""
        return b16_from_rows(raw(name + ".weight"))

    def dense(name: str, rows=None, cols: slice | None = None):
        """A linear's weight and its scales: e8m0 (MXFP8), ("block", fp32 per row and 64 inputs) or None (bf16)."""

        w = raw(name + ".weight")
        if w.dtype == torch.float8_e4m3fn and rd.has(prefix + name + ".weight_scale_inv"):   # FP8_PB_WO blocks
            if rows is not None or cols is not None:
                raise ValueError(f"{name}: block-scaled FP8 is read on one GPU only (--tp 1)")
            from tensorfold.cuda.nvfp4.linear import Fp8BlockLinear

            return w, ("block", Fp8BlockLinear.column_scales(raw(name + ".weight_scale_inv"), *w.shape))
        s = raw(name + ".weight_scale") if w.dtype == torch.float8_e4m3fn else None
        if s is not None and s.dtype != torch.uint8:
            raise ValueError(f"{name}: FP8 with a per-tensor scale; Flash Next reads MXFP8 (a scale every 32 inputs)")
        if s is None:
            _plain(name, w)
        if rows is not None:
            w, s = w[rows], None if s is None else s[rows]
        if cols is not None:
            w, s = w[:, cols], None if s is None else s[:, cols.start // 32:cols.stop // 32]
        return w, s

    def face(*parts):
        """Linears of one input as one face by their storage: bf16 rows on ``bf16.matmul``, MXFP8 on the lane matmul."""

        got = [dense(*p) for p in parts]
        if any(isinstance(s, tuple) for _, s in got):     # block FP8: its own lane-matmul face; bf16 parts beside it
            from tensorfold.cuda.nvfp4.linear import Concat, Fp8BlockLinear

            runs: list[list] = []
            for w, s in got:
                kind = "block" if isinstance(s, tuple) else "bf16" if s is None else "mx"
                if kind == "mx":
                    raise ValueError(f"{parts[0][0]}: a projection stack mixes MXFP8 and block FP8 weights")
                if runs and runs[-1][0] == kind:
                    runs[-1][1].append((w, s))
                else:
                    runs.append([kind, [(w, s)]])
            faces = [Fp8BlockLinear.from_rows(torch.cat([w for w, _ in ws]), torch.cat([s[1] for _, s in ws]))
                     if kind == "block" else b16_rows(torch.cat([w for w, _ in ws]).to(torch.bfloat16))
                     for kind, ws in runs]
            return faces[0] if len(faces) == 1 else Concat(faces)
        if all(s is None for _, s in got):
            faces = [b16_rows(w.to(torch.bfloat16)) for w, _ in got]
            return faces[0] if len(faces) == 1 else stack_b16(faces)
        if all(s is not None for _, s in got):
            from tensorfold.cuda.nvfp4.linear import Mx8Linear

            return Mx8Linear.from_checkpoint(torch.cat([w for w, _ in got]), torch.cat([s for _, s in got]))
        raise ValueError(f"{parts[0][0]}: a projection stack mixes MXFP8 and bf16 weights")

    def hc_nvfp4(name: str, inject: bool) -> HC:
        parts = [b16(name + ".input_mix_weight_down")]
        if inject:
            parts.append(b16(name + ".block_inject_weight"))
        return HC(stack_b16(parts), b16(name + ".input_mix_weight_up"), cscale(name + ".hc_norm.weight"), inject)

    def gdn_nvfp4(name: str) -> GDNW:
        """A rank's DeltaNet rows, conv and head vectors from bf16 or MXFP8 checkpoint linears."""

        kl, vl = full.nk // world, full.nv // world
        dk, dv = full.dk, full.dv
        q_rows = torch.arange(rank * kl * dk, (rank + 1) * kl * dk, device=device)
        v_rows = 2 * full.nk * dk + torch.arange(rank * vl * dv, (rank + 1) * vl * dv, device=device)
        channels = torch.cat([q_rows, full.nk * dk + q_rows, v_rows])
        one = world == 1
        proj = face((name + ".in_proj_qkv", None if one else channels),
                    (name + ".in_proj_z", None if one else slice(rank * vl * dv, (rank + 1) * vl * dv)),
                    (name + ".in_proj_b", None if one else slice(rank * vl, (rank + 1) * vl)),
                    (name + ".in_proj_a", None if one else slice(rank * vl, (rank + 1) * vl)))
        conv = raw(name + ".conv1d.weight").reshape(full.conv_dim, full.conv_kernel).to(torch.bfloat16)
        conv = conv.index_select(0, channels).contiguous()
        return GDNW(proj, conv, raw(name + ".A_log").float()[rank * vl:(rank + 1) * vl].contiguous(),
                    raw(name + ".dt_bias").float()[rank * vl:(rank + 1) * vl].contiguous(),
                    raw(name + ".norm.weight").to(torch.bfloat16).contiguous(),
                    face((name + ".out_proj", None, None if one else slice(rank * vl * dv, (rank + 1) * vl * dv))))

    def attention_nvfp4(name: str) -> AttnW:
        """An attention block from the NVFP4 checkpoint (bf16 or MXFP8 linears)."""

        hd = full.head_dim
        hl, kl = full.heads // world, full.kv_heads // world
        one = world == 1
        proj = face((name + ".q_proj", None if one else slice(rank * hl * 2 * hd, (rank + 1) * hl * 2 * hd)),
                    (name + ".k_proj", None if one else slice(rank * kl * hd, (rank + 1) * kl * hd)),
                    (name + ".v_proj", None if one else slice(rank * kl * hd, (rank + 1) * kl * hd)),
                    (name + ".indexer.index_qk_proj", None))
        o = face((name + ".o_proj", None, None if one else slice(rank * hl * hd, (rank + 1) * hl * hd)))
        return AttnW(proj, cscale(name + ".q_norm.weight"), cscale(name + ".k_norm.weight"),
                     cscale(name + ".indexer.q_layernorm.weight"), cscale(name + ".indexer.k_layernorm.weight"),
                     o)

    def ple_nvfp4(name: str, ple_index: int) -> PLEW:
        """A PLE layer with bf16, FP8, NVFP4 or MLX 4-bit n-gram shards and bf16 projections."""

        if ple_on_ssd:
            raise ValueError("--ple-on-ssd reads the MLX checkpoint's n-gram shards from disk; an NVFP4 checkpoint's "
                             "tables stay memory-mapped, so drop --ple-on-ssd")
        ngram = cfg.ngram(ple_index)
        base = name + ".ple_embedding."
        ngram.check(raw(base + "layer_multipliers").cpu().numpy(), raw(base + "ngram_heads_offsets").cpu().numpy(),
                    raw(base + "ngram_heads_vocab_sizes").cpu().numpy())
        keys = shard_keys(prefix + base + "ngram_embedding", cfg.ngram_shards, rd.where)
        table = open_table(model_dir, [(rd.where[k + ".weight"], k) for k in keys],
                           lambda n: table_scale(base, n))
        if getattr(table, "width", ngram.dims) != ngram.dims:
            raise ValueError(f"the n-gram rows hold {table.width} values, expected {ngram.dims}")
        if table.rows != ngram.rows:
            raise ValueError(f"n-gram tables hold {table.rows} rows, expected {ngram.rows}")
        conv = raw(name + ".conv1d.weight").reshape(cfg.streams * cfg.hidden, cfg.ple_kernel).to(torch.bfloat16)
        return PLEW(table, b16(name + ".key_proj"), b16(name + ".value_proj"),
                    cscale(name + ".norm_key.weight"), cscale(name + ".norm_query.weight"),
                    cscale(name + ".norm_conv.weight"), conv.contiguous(), ngram)

    def weight_bf16(name: str, index: torch.Tensor | None = None) -> torch.Tensor:
        """A linear's weight as bf16 rows: block FP8 (``weight_scale_inv``) dequantized in row chunks, else cast."""

        full = raw(name + ".weight")
        w = full if index is None else full.index_select(0, index)
        if w.dtype != torch.float8_e4m3fn or not rd.has(prefix + name + ".weight_scale_inv"):
            return _plain(name, w).to(torch.bfloat16)
        from tensorfold.cuda.nvfp4.linear import Fp8BlockLinear

        cols = Fp8BlockLinear.column_scales(raw(name + ".weight_scale_inv"), *full.shape)
        if index is not None:
            cols = cols.index_select(0, index)
        out = torch.empty(w.shape, dtype=torch.bfloat16, device=w.device)
        for r in range(0, w.shape[0], 16384):
            blk = w[r:r + 16384].float().view(-1, w.shape[1] // 64, 64) * cols[r:r + 16384, :, None]
            out[r:r + 16384] = blk.view(-1, w.shape[1]).to(torch.bfloat16)
        return out

    def b16_rows(t: torch.Tensor):
        return b16_from_rows(t.to(torch.bfloat16).contiguous())

    def moe_nvfp4(name: str) -> MoEW:
        """The NVFP4 checkpoint's MoE: the routed experts' raw stacks (FP4 words, or bf16 rows in the MTP layer)
        and the shared expert's, with the gate rows; slice 4 packs the grids from these."""

        router = raw(name + ".gate.weight").to(torch.bfloat16)
        sgate = raw(name + ".shared_expert_gate.weight").to(torch.bfloat16).reshape(full.hidden).contiguous()
        router = torch.cat([router, sgate[None]]).contiguous()
        e = full.experts
        w_ = full.moe_width
        gs = full.nvfp4_group
        lo, hi = rank * w_ // world, (rank + 1) * w_ // world
        dlo, dhi = rank * w_ // world // gs, (rank + 1) * w_ // world // gs
        se = f"{name}.shared_expert."
        if raw(se + "gate_proj.weight").dtype == torch.float8_e4m3fn:     # MXFP8: its own lane-matmul faces
            shared = (face((se + "gate_proj", slice(lo, hi)), (se + "up_proj", slice(lo, hi))),
                      face((se + "down_proj", None, slice(dlo * gs, dhi * gs))))
        else:
            shared = (raw(se + "gate_proj.weight").to(torch.bfloat16)[lo:hi].contiguous(),
                      raw(se + "up_proj.weight").to(torch.bfloat16)[lo:hi].contiguous(),
                      raw(se + "down_proj.weight").to(torch.bfloat16)[:, dlo * gs:dhi * gs].contiguous())
        fp8_drafter = (rd.has(prefix + f"{name}.experts.0.gate_proj.weight")
                       and raw(f"{name}.experts.0.gate_proj.weight").dtype == torch.float8_e4m3fn)
        if fp8_drafter and not name.startswith("mtp."):
            raise ValueError(f"{name}: FP8 routed experts; Flash Next reads the routed experts as NVFP4 (FP8 only in "
                             "the MTP drafter, which is re-quantized at load)")
        if fp8_drafter:                                  # MTP experts dequantize before draft-only NVFP4 packing
            def expert_bf16(base: str) -> torch.Tensor:
                codes = raw(base + ".weight")
                if codes.dtype != torch.float8_e4m3fn:
                    raise ValueError(f"{base}: expected FP8 e4m3 MTP expert weights")
                if rd.has(prefix + base + ".weight_scale_inv"):          # 128x128 blocks
                    return weight_bf16(base)
                if not rd.has(prefix + base + ".weight_scale"):
                    raise ValueError(f"{base}: FP8 MTP experts need a tensor, row or 128x128-block scale")
                scale = raw(base + ".weight_scale")
                if scale.dtype not in _PLAIN or scale.numel() not in (1, codes.shape[0]):
                    raise ValueError(f"{base}: FP8 MTP weight_scale must hold one float per tensor or output row")
                s = scale.float().reshape(-1, 1)
                return (codes.float() * s).to(torch.bfloat16)

            def stacked(proj: str) -> torch.Tensor:
                return torch.stack([expert_bf16(f"{name}.experts.{i}.{proj}") for i in range(e)])

            gate, up, dn = stacked("gate_proj"), stacked("up_proj"), stacked("down_proj")
            if world > 1:
                gate, up, dn = gate[:, lo:hi], up[:, lo:hi], dn[:, :, dlo * gs:dhi * gs]
        elif rd.has(prefix + f"{name}.experts.0.gate_proj.weight"):    # the main layers: per-expert FP4
            def stack(proj: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                weights = [raw(f"{name}.experts.{i}.{proj}.weight") for i in range(e)]
                if any(w.dtype != torch.uint8 for w in weights):
                    raise ValueError(f"{name}.{proj}: expected packed NVFP4 experts; FP8 is read only in MTP")
                w = torch.stack(weights)
                s = torch.stack([raw(f"{name}.experts.{i}.{proj}.weight_scale") for i in range(e)])
                s2 = torch.stack([raw(f"{name}.experts.{i}.{proj}.weight_scale_2") for i in range(e)])
                return w, s, s2

            gate = stack("gate_proj")
            up = stack("up_proj")
            dn = stack("down_proj")
            if world > 1:
                gate = (gate[0][:, lo:hi], gate[1][:, lo:hi], gate[2])
                up = (up[0][:, lo:hi], up[1][:, lo:hi], up[2])
                dn = (dn[0][:, :, dlo * gs // 2:dhi * gs // 2], dn[1][:, :, dlo:dhi], dn[2])   # 8 bytes a block
        else:                                            # the MTP layer: BF16 stacked experts (excluded)
            gu = raw(name + ".experts.gate_up_proj").to(torch.bfloat16)          # [E, 2*NI, D]
            dn = raw(name + ".experts.down_proj").to(torch.bfloat16)             # [E, D, NI]
            if world > 1:
                gu, dn = gu[:, lo:hi], dn[:, :, dlo * gs:dhi * gs]
            gate, up = gu[:, :w_], gu[:, w_:]            # the fused rows split back apart
        return MoEW(router, {"gate": gate, "up": up, "down": dn, "shared": shared, "router": router})

    def layer(i: int, base: str, kind: str, with_ple: bool) -> LayerW:
        linear = kind == "linear"
        entry = LayerW(i, linear,
                       hc_nvfp4(base + ".attn_hyper_connection", True),
                       hc_nvfp4(base + ".mlp_hyper_connection", True),
                       gdn_nvfp4(base + ".linear_attn") if linear else None,
                       None if linear else attention_nvfp4(base + ".self_attn"),
                       moe_nvfp4(base + ".mlp"))
        if with_ple and i in cfg.ple_layers:
            entry.ple = ple_nvfp4(base + ".ple", cfg.ple_layers.index(i))
        return entry

    t0 = time.time()
    if cfg.quant not in ("mlx", "modelopt"):
        raise ValueError(f"Flash Next's XPU engine reads NVFP4 (experts-only) checkpoints, not {cfg.quant}")
    try:                                              # a failed load cancels the reads queued ahead
        embed = (raw(mbase + "embed_tokens.weight").to(torch.bfloat16).contiguous(),)
        loaded = []
        ahead = rd.layer_names(prefix, mbase, chosen, mtp)   # read ahead of the layer that takes them
        layer_events: list = []                           # each layer's event, recorded once its work is queued
        for k, i in enumerate(chosen):
            if len(layer_events) >= 2:                    # at most two layers queued ahead of the GPU
                layer_events.pop(0).synchronize()
            for names in ahead[k:k + 2]:                  # two layers in flight: reads overlap this one's packing
                rd.queue(names)
            loaded.append(layer(i, f"{mbase}layers.{i}", cfg.layer_types[i], True))
            if api is not None:
                layer_events.append(api.current_stream().record_event())
            rd.drop(ahead[k])                             # what the layer never took
            rd.release()
            if i % 8 == 7 and api is not None:    # each release waits for the device; a layer leaves few temporaries
                api.empty_cache()
        mixer = hc_nvfp4(mbase + "hyper_connection_mixer", False)
        vl = full.vocab // world
        if rd.has(prefix + "lm_head.weight_scale_inv"):     # block FP8: its stored bytes
            from tensorfold.cuda.nvfp4.linear import Fp8BlockLinear

            head = replace(Fp8BlockLinear.from_checkpoint(raw("lm_head.weight"), raw("lm_head.weight_scale_inv")),
                           lane=True)
        else:
            head = b16_rows(weight_bf16("lm_head")[rank * vl:(rank + 1) * vl])
        # NVFP4: the draft head is the bf16 lm_head's draft rows requantized 4-bit at load (drafts only)
        draft_head, draft_ids = None, None
        ids = draft_token_ids(draft_vocab)
        if ids is not None:
            ids = np.array_split(ids[ids < full.vocab], world)[rank]
            ids = torch.from_numpy(ids).to(device)
            draft_ids = ids
            draft_head = quantize4(weight_bf16("lm_head", ids))
        inv = torch.tensor(cfg.rope_theta, dtype=torch.float64) ** (
            -torch.arange(0, cfg.rotary_dim // 2, dtype=torch.float64) / (cfg.rotary_dim // 2))
        w = Weights(cfg, embed, loaded, mixer, head, inv.to(torch.float32).to(device), around_one=around_one)
        w.meta.update(rank=rank, world=world, vocab_offset=rank * vl, full=full)
        w.draft_head, w.draft_ids = draft_head, draft_ids
        if mtp and rd.has(prefix + "mtp.fc_embedding.weight"):
            w.mtp = MTPW(cscale("mtp.pre_fc_norm_embedding.weight"), cscale("mtp.pre_fc_norm_hidden.weight"),
                         b16("mtp.fc_embedding"), b16("mtp.fc_hidden"),
                         layer(-1, "mtp.layers.0", "attention", False),
                         hc_nvfp4("mtp.hyper_connection_mixer", False))
    except BaseException:
        rd.close()
        raise
    rd.close()
    rd.release()
    if api is not None:
        api.empty_cache()
    w.meta["load_seconds"] = time.time() - t0
    return w
