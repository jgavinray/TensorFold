"""Slice 5's gate: the hyper-connections' branch/merge on the B70.

Branch (write-back into the streams) and merge (slot/rank/slice sums, the mix) run on the card
through ``xpu/hc.py`` and must be ``torch.equal`` with the host reference: the ``cuda/glue.py``
kernels' arithmetic transcribed line-by-line (the fp32/bf16 rounding the CUDA kernels define), run
on CPU over the same input bits. The four checks the slice owes: the branch/merge ops against that
host reference (modes 0-4, the act/reduce and the upmix mix), batch-shape invariance (rows
one-at-a-time vs batched on the card), one real-checkpoint HCW end-to-end card-vs-CPU, and the
``hc_check.py`` gate replicated (its ``_check``/``_check_upmix`` fixtures, shapes and seeds, byte
comparison) - ``hc_check.fuser`` itself is CUDA-ext bound (torch.cuda + the glue/qmm kernels), so
its assertions are replicated here rather than run.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.xpu import hc  # noqa: E402
from tensorfold.families.qwen4_exp.xpu.weights import load  # noqa: E402

REAL = Path("/storage01/models/Qwen3.8-Flash-Next-NVFP4")
ROWS, DIMS, STREAMS, SLOTS, LOW = 7, 2560, 4, 9, 320      # hc_check._check's fixture shapes
EPS = 1e-6


def _device() -> torch.device:
    """B70 #2: the usable device whose uuid hex contains 4400 (the selector leaves it at index 0)."""

    for i in range(torch.xpu.device_count()):
        try:
            uuid = bytes(torch.xpu.get_device_properties(i).uuid.bytes).hex()
        except Exception:
            continue
        if "4400" in uuid:
            return torch.device("xpu", i)
    pytest.skip("no usable XPU device with uuid containing 4400")


@pytest.fixture(scope="module")
def dev() -> torch.device:
    return _device()


def _free_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1 << 20)
    return 1e9


@pytest.fixture(scope="module", autouse=True)
def _ram_guard():
    """The slice's runs abort under 24 GB MemAvailable."""

    free = _free_gb()
    if free < 24.0:
        pytest.skip(f"MemAvailable {free:.1f} GB is under the 24 GB guard")


@pytest.fixture(scope="module")
def real():
    if not (REAL / "model.safetensors.index.json").is_file():
        pytest.skip("the real checkpoint is not mounted")
    return load(REAL, device="cpu", mtp=False)


def _gen(seed: int):
    """One CPU generator, so the card and the host see the same input bits."""

    return torch.Generator().manual_seed(seed)


def _rnd(g, *shape, dt=torch.bfloat16):
    return torch.randn(*shape, generator=g).to(dt)


# ---------------------------------------------------------------- the host reference
# cuda/glue.py's _hc_writeback/_hc_normed/_hc_act/_hc_reduce_act/_hc_mix and cuda/hc_upmix.py's
# mix, transcribed with the module's portable primitives (the shared fp32/bf16 arithmetic contract).


def ref_branch(g, mode, rows, d, br=None, y=None, wts=None):
    if mode == 1:
        return br.float()
    if mode in (3, 4):
        acc = br[0].clone()
        for k in range(1, br.shape[0]):
            acc = acc + br[k]
        return hc._bf(acc)
    if mode == 2:
        acc = torch.zeros(rows, d)
        for k in range(SLOTS):
            acc = acc + y[:, k].float() * wts[:, k][:, None]
        return hc._bf(acc)
    return None


def ref_write_norm(h, pss, scale, xs, normed, mode, branch, inject, y, wts, streams, eps, d):
    """The two released kernels in sequence: write-back then norm, rows x (streams * d)."""

    rows = h.shape[0]
    br = ref_branch(None, mode, rows, d, branch, y, wts)
    for s in range(streams):
        hv = h.view(rows, streams, d)[:, s].float().clone()
        if mode:
            hv = hc._bf(hv + br * inject.float()[:, s:s + 1])
            h.view(rows, streams, d)[:, s].copy_(hv.to(torch.bfloat16))
        pss[:, :, s].copy_(hc.group_sums(hv * hv, d // 256, 256))
    total = torch.zeros(rows, streams)
    for c in range(pss.shape[1]):
        total = total + pss[:, c, :]
    rinv = hc.div(torch.ones_like(total), hc.sqrt(hc.div(total, d) + eps))
    for s in range(streams):
        v = ((h.view(rows, streams, d)[:, s].float() * rinv[:, s][:, None])
             * scale.float().view(streams, d)[s]).to(torch.bfloat16)
        normed.view(rows, streams, d)[:, s].copy_(v)
        xs.view(rows, streams, d // 32)[:, s].copy_(hc.group_sums(v.float(), d // 32, 32))


def ref_act(dn, act, xs, inj, streams, low):
    v = hc._bf(hc.div(dn[:, :low], streams))
    a = hc.bsilu(v)
    act.copy_(a.to(torch.bfloat16))
    xs.copy_(hc.group_sums(a, low // 32, 32))
    if inj is not None:
        iv = hc._bf(hc.div(dn[:, low:low + streams], streams))
        inj.copy_((2.0 * hc.bsig(iv)).to(torch.bfloat16))


def ref_reduce_act(part, act, xs, inj, sk, streams, low):
    v = part[0][:, :low].clone()
    for k in range(1, sk):
        v = v + part[k][:, :low]
    v = hc._bf(v)
    v = hc._bf(hc.div(v, streams))
    a = hc.bsilu(v)
    act.copy_(a.to(torch.bfloat16))
    xs.copy_(hc.group_sums(a, low // 32, 32))
    if inj is not None:
        iv = part[0][:, low:low + streams].clone()
        for k in range(1, sk):
            iv = iv + part[k][:, low:low + streams]
        iv = hc._bf(iv)
        iv = hc._bf(hc.div(iv, streams))
        inj.copy_((2.0 * hc.bsig(iv)).to(torch.bfloat16))


def ref_mix(up, normed, mixed, xs, streams):
    """glue._hc_mix and hc_upmix's mix epilogue: ordered bf16-rounded sigmoid*normed, /S, 32-sums."""

    rows, wide = up.shape
    d = wide // streams
    total = torch.zeros(rows, d)
    for s in range(streams):
        total = total + hc._bf(hc.bsig(up.view(rows, streams, d)[:, s].float())
                               * normed.view(rows, streams, d)[:, s].float())
    m = hc.div(total, streams).to(torch.bfloat16)
    mixed.copy_(m)
    xs.copy_(hc.group_sums(m.float(), d // 32, 32))


def _write_norm_case(dev, mode, seed=1927):
    """The hc_check._check fixture for one mode: host reference vs the card, byte-for-byte."""

    g = _gen(seed)
    h0 = _rnd(g, ROWS, STREAMS * DIMS)
    scale = _rnd(g, STREAMS * DIMS, dt=torch.float32)
    inject = _rnd(g, ROWS, STREAMS)
    branch = y = wts = None
    if mode == 1:
        branch = _rnd(g, ROWS, DIMS)
    elif mode == 2:
        y, wts = _rnd(g, ROWS, SLOTS, DIMS), _rnd(g, ROWS, SLOTS, dt=torch.float32)
    elif mode in (3, 4):
        branch = _rnd(g, 3, ROWS, DIMS, dt=torch.float32)
    pss0 = torch.zeros(ROWS, DIMS // 256, STREAMS)
    n0 = torch.zeros_like(h0)
    x0 = torch.zeros(ROWS, STREAMS * DIMS // 32)
    h1 = h0.clone().to(dev)
    pss1 = torch.zeros(ROWS, DIMS // 256, STREAMS, device=dev)
    n1 = torch.zeros_like(h1)
    x1 = torch.zeros(ROWS, STREAMS * DIMS // 32, device=dev)
    ref_write_norm(h0, pss0, scale, x0, n0, mode, branch, inject, y, wts, STREAMS, EPS, DIMS)
    hc.write_norm(h1, pss1, scale.to(dev), n1, x1, STREAMS, EPS, mode,
                  branch=branch.to(dev) if branch is not None else None,
                  inject=inject.to(dev), y=y.to(dev) if y is not None else None,
                  wts=wts.to(dev) if wts is not None else None)
    for a, b in ((h0, h1), (pss0, pss1), (n0, n1), (x0, x1)):
        assert torch.equal(a.view(torch.uint8), b.cpu().view(torch.uint8))


# ---------------------------------------------------------------- gate 1: branch/merge vs the host reference


@pytest.mark.parametrize("mode", [0, 1, 2, 3, 4])
def test_writeback_normed_match_the_host_reference(mode, dev):
    """All five write-back modes and the normed read-out, card vs the host transcription: torch.equal."""

    _write_norm_case(dev, mode)


@pytest.mark.parametrize("with_inject", [False, True])
def test_act_matches_the_host_reference(with_inject, dev):
    """The down-projection epilogue (SiLU and the inject gates), card vs host reference: torch.equal."""

    g = _gen(7717)
    dn = _rnd(g, ROWS, LOW + (STREAMS if with_inject else 0), dt=torch.float32)
    a0, x0 = torch.zeros(ROWS, LOW, dtype=torch.bfloat16), torch.zeros(ROWS, LOW // 32)
    i0 = torch.zeros(ROWS, STREAMS, dtype=torch.bfloat16) if with_inject else None
    a1 = torch.zeros_like(a0).to(dev)
    x1 = torch.zeros_like(x0).to(dev)
    i1 = torch.zeros_like(i0).to(dev) if with_inject else None
    ref_act(dn.clone(), a0, x0, i0, STREAMS, LOW)
    hc.hc_act(dn.to(dev), a1, x1, i1, STREAMS, LOW)
    assert torch.equal(a0.cpu().view(torch.uint8), a1.cpu().view(torch.uint8))
    assert torch.equal(x0.cpu().view(torch.uint8), x1.cpu().view(torch.uint8))
    if with_inject:
        assert torch.equal(i0.cpu().view(torch.uint8), i1.cpu().view(torch.uint8))


def test_reduce_act_matches_the_host_reference(dev):
    """The split-K reduce fused with the act, card vs host reference (slices summed in order)."""

    g = _gen(313)
    sk = 3
    part = _rnd(g, sk, ROWS, LOW + STREAMS, dt=torch.float32)
    a0, x0 = torch.zeros(ROWS, LOW, dtype=torch.bfloat16), torch.zeros(ROWS, LOW // 32)
    i0 = torch.zeros(ROWS, STREAMS, dtype=torch.bfloat16)
    a1 = torch.zeros_like(a0).to(dev)
    x1 = torch.zeros_like(x0).to(dev)
    i1 = torch.zeros_like(i0).to(dev)
    ref_reduce_act(part.clone(), a0, x0, i0, sk, STREAMS, LOW)
    hc.hc_reduce_act(part.to(dev), a1, x1, i1, STREAMS, LOW)
    assert torch.equal(a0.cpu().view(torch.uint8), a1.cpu().view(torch.uint8))
    assert torch.equal(x0.cpu().view(torch.uint8), x1.cpu().view(torch.uint8))
    assert torch.equal(i0.cpu().view(torch.uint8), i1.cpu().view(torch.uint8))


@pytest.mark.parametrize("rows", [17, 32])
def test_mix_matches_the_upmix_reference(rows, dev):
    """The mix half of cuda/hc_upmix.py (gate*S * normed, ordered, /S, the 32-group sums): the
    _check_upmix fixture's rows and seed, card vs host reference, torch.equal."""

    g = _gen(2639)
    up = _rnd(g, rows, STREAMS * DIMS)
    normed = _rnd(g, rows, STREAMS * DIMS)
    m0, x0 = torch.zeros(rows, DIMS, dtype=torch.bfloat16), torch.zeros(rows, DIMS // 32)
    m1 = torch.zeros_like(m0).to(dev)
    x1 = torch.zeros_like(x0).to(dev)
    ref_mix(up.clone(), normed.clone(), m0, x0, STREAMS)
    hc.hc_mix(up.to(dev), normed.to(dev), m1, x1, STREAMS)
    assert torch.equal(m0.cpu().view(torch.uint8), m1.cpu().view(torch.uint8))
    assert torch.equal(x0.cpu().view(torch.uint8), x1.cpu().view(torch.uint8))
    hc.hc_mix(up.to(dev), normed.to(dev), m1, x1, STREAMS)                # the candidate's second call
    assert torch.equal(m0.cpu().view(torch.uint8), m1.cpu().view(torch.uint8))
    assert torch.equal(x0.cpu().view(torch.uint8), x1.cpu().view(torch.uint8))


# ---------------------------------------------------------------- gate 2: batch-shape invariance


def test_rows_one_at_a_time_match_the_batch(dev):
    """Every op on the 7-row batch equals its rows run one at a time, torch.equal: the reductions
    are elementwise fixed-order folds, so no launch or batch shape can rechoose a tree."""

    g = _gen(9091)
    h0 = _rnd(g, ROWS, STREAMS * DIMS).clone()
    scale = _rnd(g, STREAMS * DIMS, dt=torch.float32)
    inject = _rnd(g, ROWS, STREAMS)
    branch = _rnd(g, ROWS, DIMS)
    pss_b = torch.zeros(ROWS, DIMS // 256, STREAMS, device=dev)
    n_b = torch.zeros(ROWS, STREAMS * DIMS, dtype=torch.bfloat16, device=dev)
    x_b = torch.zeros(ROWS, STREAMS * DIMS // 32, device=dev)
    h_b = h0.clone().to(dev)
    hc.write_norm(h_b, pss_b, scale.to(dev), n_b, x_b, STREAMS, EPS, 1, branch=branch.to(dev),
                  inject=inject.to(dev))
    h_1 = h0.clone().to(dev)
    pss_1 = torch.zeros_like(pss_b)
    n_1 = torch.zeros_like(n_b)
    x_1 = torch.zeros_like(x_b)
    for i in range(ROWS):
        hc.write_norm(h_1[i:i + 1], pss_1[i:i + 1], scale.to(dev), n_1[i:i + 1], x_1[i:i + 1],
                      STREAMS, EPS, 1, branch=branch[i:i + 1].to(dev), inject=inject[i:i + 1].to(dev))
    assert torch.equal(h_b, h_1) and torch.equal(pss_b, pss_1)
    assert torch.equal(n_b, n_1) and torch.equal(x_b, x_1)

    dn = _rnd(g, ROWS, LOW + STREAMS, dt=torch.float32)
    a_b = torch.zeros(ROWS, LOW, dtype=torch.bfloat16, device=dev)
    xs_b = torch.zeros(ROWS, LOW // 32, device=dev)
    i_b = torch.zeros(ROWS, STREAMS, dtype=torch.bfloat16, device=dev)
    hc.hc_act(dn.to(dev), a_b, xs_b, i_b, STREAMS, LOW)
    a_1 = torch.zeros_like(a_b)
    xs_1 = torch.zeros_like(xs_b)
    i_1 = torch.zeros_like(i_b)
    for k in range(ROWS):
        hc.hc_act(dn[k:k + 1].to(dev), a_1[k:k + 1], xs_1[k:k + 1], i_1[k:k + 1], STREAMS, LOW)
    assert torch.equal(a_b, a_1) and torch.equal(xs_b, xs_1) and torch.equal(i_b, i_1)

    part = _rnd(g, 3, ROWS, LOW + STREAMS, dt=torch.float32)
    a_b.zero_()
    xs_b.zero_()
    i_b.zero_()
    hc.hc_reduce_act(part.to(dev), a_b, xs_b, i_b, STREAMS, LOW)
    a_1.zero_()
    xs_1.zero_()
    i_1.zero_()
    for k in range(ROWS):
        hc.hc_reduce_act(part[:, k:k + 1].to(dev), a_1[k:k + 1], xs_1[k:k + 1], i_1[k:k + 1],
                         STREAMS, LOW)
    assert torch.equal(a_b, a_1) and torch.equal(xs_b, xs_1) and torch.equal(i_b, i_1)

    up = _rnd(g, ROWS, STREAMS * DIMS)
    m_b = torch.zeros(ROWS, DIMS, dtype=torch.bfloat16, device=dev)
    xm_b = torch.zeros(ROWS, DIMS // 32, device=dev)
    hc.hc_mix(up.to(dev), n_b, m_b, xm_b, STREAMS)
    m_1 = torch.zeros_like(m_b)
    xm_1 = torch.zeros_like(xm_b)
    for k in range(ROWS):
        hc.hc_mix(up[k:k + 1].to(dev), n_b[k:k + 1], m_1[k:k + 1], xm_1[k:k + 1], STREAMS)
    assert torch.equal(m_b, m_1) and torch.equal(xm_b, xm_1)


# ---------------------------------------------------------------- gate 3: one real HCW


def test_real_checkpoint_hcw_card_vs_cpu(real, dev):
    """One HCW from the real checkpoint (slice-1's loader, device='cpu'): the write_norm with the
    real hc_norm gamma, then act on the real down projection's output and mix on the real up's,
    each card-vs-CPU torch.equal."""

    hcw = real.layers[0].attn_hc
    streams, low = real.cfg.streams, real.cfg.low
    d = hcw.scale.numel() // streams                                    # the per-stream width
    assert hcw.inject and low == LOW and d == real.cfg.hidden
    scale = hcw.scale.contiguous()                                   # [S*D] fp32
    down = hcw.down.b.weight.to(torch.float32)                      # [low + S, S*D] bf16 -> fp32
    up_w = hcw.up.b.weight.to(torch.float32)                        # [S*D, low]
    rows = 7
    g = _gen(5150)
    h0 = _rnd(g, rows, streams * d).clone()
    inject = _rnd(g, rows, streams)
    branch = _rnd(g, rows, d)
    pss0 = torch.zeros(rows, d // 256, streams)
    n0 = torch.zeros_like(h0)
    x0 = torch.zeros(rows, streams * d // 32)
    h1 = h0.clone().to(dev)
    pss1 = torch.zeros(rows, d // 256, streams, device=dev)
    n1 = torch.zeros_like(h1)
    x1 = torch.zeros(rows, streams * d // 32, device=dev)
    ref_write_norm(h0, pss0, scale, x0, n0, 1, branch, inject, None, None, streams, real.cfg.eps, d)
    hc.write_norm(h1, pss1, scale.to(dev), n1, x1, streams, real.cfg.eps, 1,
                  branch=branch.to(dev), inject=inject.to(dev))
    assert torch.equal(h0.view(torch.uint8), h1.cpu().view(torch.uint8))
    assert torch.equal(pss0.view(torch.uint8), pss1.cpu().view(torch.uint8))
    assert torch.equal(n0.view(torch.uint8), n1.cpu().view(torch.uint8))
    assert torch.equal(x0.view(torch.uint8), x1.cpu().view(torch.uint8))

    dn = (n0.view(rows, streams * d).float() @ down.T).contiguous()   # the down projection, host bits
    a0, xa0 = torch.zeros(rows, low, dtype=torch.bfloat16), torch.zeros(rows, low // 32)
    i0 = torch.zeros(rows, streams, dtype=torch.bfloat16)
    a1 = torch.zeros_like(a0).to(dev)
    xa1 = torch.zeros_like(xa0).to(dev)
    i1 = torch.zeros_like(i0).to(dev)
    ref_act(dn.clone(), a0, xa0, i0, streams, low)
    hc.hc_act(dn.to(dev), a1, xa1, i1, streams, low)
    assert torch.equal(a0.view(torch.uint8), a1.cpu().view(torch.uint8))
    assert torch.equal(xa0.view(torch.uint8), xa1.cpu().view(torch.uint8))
    assert torch.equal(i0.view(torch.uint8), i1.cpu().view(torch.uint8))

    up = (a0.float() @ up_w.T).to(torch.bfloat16).contiguous()        # the up projection, host bits
    m0, xm0 = torch.zeros(rows, d, dtype=torch.bfloat16), torch.zeros(rows, d // 32)
    m1 = torch.zeros_like(m0).to(dev)
    xm1 = torch.zeros_like(xm0).to(dev)
    ref_mix(up.clone(), n0.clone(), m0, xm0, streams)
    hc.hc_mix(up.to(dev), n0.clone().to(dev), m1, xm1, streams)
    assert torch.equal(m0.view(torch.uint8), m1.cpu().view(torch.uint8))
    assert torch.equal(xm0.view(torch.uint8), xm1.cpu().view(torch.uint8))


# ---------------------------------------------------------------- gate 4: hc_check parity


def test_hc_check_gate_is_cuda_bound_and_its_assertions_replicated(dev):
    """hc_check.fuser returns None for an XPU device (it is CUDA-ext bound: torch.cuda.device, the
    glue/qmm triton kernels); _check's and _check_upmix's fixtures - shapes, seeds, modes and the
    uint8 byte comparison - are replicated by the tests above, which therefore carry its gate."""

    from tensorfold.families.qwen4_exp.cuda.hc_check import fuser

    assert fuser(torch.device("xpu")) is None
    assert fuser(torch.device("xpu"), upmix=True) is None
    _write_norm_case(dev, 1)                                   # _check's mode 1, byte-compared
    _write_norm_case(dev, 2)                                   # _check's MoE slots, byte-compared
