"""Slice 3's gate: the XPU GDN chain's per-row bits.

Every check is ``torch.equal`` on the device (the per-tensor bit-for-bit pattern of the CUDA prefill
tests): a row's bits never depend on its window (a chain against its state-composed splits, final state
and bf16 out rows), a replayed prefix has the serial state's bits, the copied qwen3_5 step matches the
fallback primitive at this family's widths and the tiny fixture's rank share, and the XPUGraph capture is
bit-exact with the eager chain at 64 rows.
"""

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen3_5.xpu import gdn_fallback  # noqa: E402
from tensorfold.families.qwen4_exp.xpu import gdn  # noqa: E402

WIDTHS = [(16, 48), (8, 24)]          # the layer and the tiny fixture's per-rank share
EPS = 1e-6


@pytest.fixture(scope="module")
def dev() -> torch.device:
    """B70 #2: the usable device whose uuid hex contains 4400 (the selector leaves it at index 0)."""

    for i in range(torch.xpu.device_count()):
        try:
            uuid = bytes(torch.xpu.get_device_properties(i).uuid.bytes).hex()
        except Exception:
            continue
        if "4400" in uuid:
            return torch.device("xpu", i)
    pytest.skip("no usable XPU device with uuid containing 4400")


def _inputs(nk: int, nv: int, rows: int, dev, seed: int):
    """Synthetic layer inputs on ``dev``: a projection window, the conv state/weight, the state and the
    head vectors, all from one CPU generator (deterministic across runs and devices)."""

    conv, pw = gdn.widths(nk, nv)
    g = torch.Generator().manual_seed(seed)

    def bf(*shape):
        return torch.randn(*shape, generator=g).to(torch.bfloat16).to(dev)

    p = bf(rows, pw)
    cs = bf(gdn.TAPS - 1, conv)
    cw = bf(conv, gdn.TAPS)
    state = (torch.randn(nv, 128, 128, generator=g) * 0.1).to(dev)
    a_log = torch.randn(nv, generator=g).to(dev)
    dt_bias = torch.randn(nv, generator=g).to(dev)
    return p, cs, cw, state, a_log, dt_bias, bf(128)


def _run(nk, nv, p, cs, cw, state, a_log, dt_bias, norm_w, dev):
    """One chain call on row-view slices: (scratch, state, out, xs)."""

    rows = p.shape[0]
    scratch = gdn.GDNScratch(rows, dev, nk, nv)
    st = torch.empty_like(state)
    out = torch.empty((rows, nv * 128), dtype=torch.bfloat16, device=dev)
    xs = torch.empty((rows, nv * 128 // 32), dtype=torch.float32, device=dev)
    gdn.chain(p, cs, cw, state, a_log, dt_bias, norm_w, EPS, rows, scratch, st, out, xs)
    return scratch, st, out, xs


@pytest.mark.parametrize("nk,nv", WIDTHS)
@pytest.mark.parametrize("T", [1, 2, 3, 16, 17, 64, 65])
def test_row_invariance(nk, nv, T, dev):
    """A chain on rows [0..T) equals its state-composed split at every m: the final state and both sides'
    row outputs, torch.equal — across the graph/eager boundary at 64 rows."""

    p, cs, cw, state, a_log, dt_bias, norm_w = _inputs(nk, nv, T, dev, 1000 * T + 17 * nv)
    sc, st, out, xs = _run(nk, nv, p, cs, cw, state, a_log, dt_bias, norm_w, dev)
    cs0 = cs.clone()
    conv = cs.shape[1]
    for m in range(1, T):
        sc1, st1, out1, xs1 = _run(nk, nv, p[:m], cs, cw, state, a_log, dt_bias, norm_w, dev)
        assert torch.equal(out1, out[:m]) and torch.equal(xs1, xs[:m])
        cs1 = torch.cat([cs, p[:m, :conv]])[m:m + gdn.TAPS - 1]      # forward.py's shift_windows
        sc2, st2, out2, xs2 = _run(nk, nv, p[m:], cs1, cw, st1, a_log, dt_bias, norm_w, dev)
        assert torch.equal(st2, st)
        assert torch.equal(out2, out[m:]) and torch.equal(xs2, xs[m:])
    assert torch.equal(cs, cs0)                                      # the conv state is read only


@pytest.mark.parametrize("nk,nv", WIDTHS)
@pytest.mark.parametrize("T", [1, 2, 17, 64, 65])
def test_replay_equals_serial(nk, nv, T, dev):
    """After a chain, replay(state_in, scratch, rows) is the serial state, torch.equal — against the
    chain's committed state and against the step loop re-run from the retained scratch."""

    p, cs, cw, state, a_log, dt_bias, norm_w = _inputs(nk, nv, T, dev, 2000 * T + 17 * nv)
    sc, st, out, xs = _run(nk, nv, p, cs, cw, state, a_log, dt_bias, norm_w, dev)
    st_re = torch.empty_like(state)
    gdn.replay(state, sc, T, st_re)
    assert torch.equal(st_re, st)
    S = state.clone()
    kf = gdn._expanded(sc.k[:T], nv // nk)
    vf = sc.v[:T].float()
    for t in range(T):
        S, _ = gdn._step(S, kf[t], kf[t], vf[t], sc.g[:T][t].float(), sc.b[:T][t].float())
    assert torch.equal(S, st)
    assert torch.equal(st_re, S)


@pytest.mark.parametrize("nk,nv", WIDTHS)
def test_fallback_primitive(nk, nv, dev, monkeypatch):
    """The copied step is the qwen3_5 fallback's: bit-identical y and final state through both modules'
    chain paths.  The fallback's capture rounds q/k to bf16, so the graph case feeds bf16-representable
    q/k; the eager case feeds arbitrary fp32 with the fallback's capture disabled."""

    g = torch.Generator().manual_seed(31 + nv)
    T = 5
    q = torch.randn(T, nk, 128, generator=g).to(torch.bfloat16).float().to(dev)
    k = torch.randn(T, nk, 128, generator=g).to(torch.bfloat16).float().to(dev)
    v = torch.randn(T, nv, 128, generator=g).to(torch.bfloat16).to(dev)
    gt = torch.rand(T, nv, generator=g).to(dev)
    beta = torch.rand(T, nv, generator=g).to(torch.bfloat16).float().to(dev)
    state = (torch.randn(nv, 128, 128, generator=g) * 0.1).to(dev)
    f1, f2 = torch.empty_like(state), torch.empty_like(state)
    y1 = gdn._delta(q, k, v, gt, beta, state, f1)
    y2 = gdn_fallback.chain(q, k, v, gt, beta, state, f2)
    assert torch.equal(y1, y2) and torch.equal(f1, f2)
    monkeypatch.setattr(gdn_fallback, "_GRAPH_CHUNKS", None)      # the fallback's eager path
    q = torch.randn(T, nk, 128, generator=g).to(dev)
    k = torch.randn(T, nk, 128, generator=g).to(dev)
    f1, f2 = torch.empty_like(state), torch.empty_like(state)
    y1 = gdn._delta(q, k, v, gt, beta, state, f1)
    y2 = gdn_fallback.chain(q, k, v, gt, beta, state, f2)
    assert torch.equal(y1, y2) and torch.equal(f1, f2)


def test_widths_and_scratch():
    """widths(16, 48) is the ledger's (CONV, PW) and the scratch holds the retained dtypes exactly."""

    conv, pw = gdn.widths(16, 48)
    assert conv == 2 * 16 * 128 + 48 * 128 == 10240
    assert pw == conv + 48 * 128 + 2 * 48 == 16480
    assert (gdn.CONV, gdn.PW) == (10240, 16480)
    assert gdn.widths(8, 24) == (5120, 8240)                      # the tiny fixture's rank share
    s = gdn.GDNScratch(7, "cpu")
    assert s.k.shape == (7, 16, 128) and s.k.dtype is torch.float32
    assert s.v.shape == (7, 48, 128) and s.v.dtype is torch.bfloat16
    assert s.g.shape == (7, 48) and s.g.dtype is torch.float32
    assert s.b.shape == (7, 48) and s.b.dtype is torch.float32


@pytest.mark.parametrize("nk,nv", WIDTHS)
def test_capture_equals_eager(nk, nv, dev, monkeypatch):
    """The 64-row XPUGraph capture is the eager chain, torch.equal — out, xs, the state and the scratch."""

    p, cs, cw, state, a_log, dt_bias, norm_w = _inputs(nk, nv, 64, dev, 3000 + 17 * nv)
    monkeypatch.setattr(gdn, "_GRAPH_CHUNKS", {})
    sc1, st1, out1, xs1 = _run(nk, nv, p, cs, cw, state, a_log, dt_bias, norm_w, dev)
    assert len(gdn._GRAPH_CHUNKS) == 1                            # the capture happened for this width
    monkeypatch.setattr(gdn, "_GRAPH_CHUNKS", None)               # the eager path
    sc2, st2, out2, xs2 = _run(nk, nv, p, cs, cw, state, a_log, dt_bias, norm_w, dev)
    assert torch.equal(out1, out2) and torch.equal(xs1, xs2) and torch.equal(st1, st2)
    assert torch.equal(sc1.k, sc2.k) and torch.equal(sc1.v, sc2.v)
    assert torch.equal(sc1.g, sc2.g) and torch.equal(sc1.b, sc2.b)
