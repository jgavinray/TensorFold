"""fp64 mirror of the Metal kernels' layer-0 arithmetic (every bf16 rounding point of norm_nores / qmm / gdn_pre / gdn_chain /
gdn_post / mlp_act / norm / embed, as in exp/kernels/<key>/case.py, from the real checkpoint tensors) on the 176-token Gate A
input. Writes mirror_l0_{ln2,h,out}.f32 [176][2048] next to the capture. Pure numpy, no MLX run.
Usage: /storage01/venvs/mlx-ref/bin/python qwen35_layer0_mirror.py <capture_dir>"""
import json, struct, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/storage01/qwen-flash-on-intel/exp/kernels")
import common
r = common.bf16_round64
cap = Path(sys.argv[1])
ids = np.fromfile(cap / "ids.u32", "<u4")
M = len(ids)

path = "/storage01/models/Qwen3.5-2B-MLX-4bit/model.safetensors"
fh = open(path, "rb")
hl = struct.unpack("<Q", fh.read(8))[0]
hdr = json.loads(fh.read(hl))
base = 8 + hl
def T(name, rows=None):
    e = hdr["language_model." + name]
    a, b = e["data_offsets"]
    raw = np.memmap(path, mode="r", offset=base + a, shape=(b - a,), dtype=np.uint8)
    dt = {"U32": "<u4", "BF16": "<u2", "F32": "<f4"}[e["dtype"]]
    arr = np.frombuffer(raw, dt).reshape(e["shape"])
    return arr if rows is None else arr[rows]
bf = lambda w: common.bf16_f32(w).astype(np.float64)
def dequant(prefix, rows=None):
    w, s, b = T(prefix + ".weight", rows), T(prefix + ".scales", rows), T(prefix + ".biases", rows)
    N = w.shape[0]
    q = np.stack([(w >> (4 * e)) & 0xF for e in range(8)], -1).reshape(N, -1).astype(np.float64)   # element 8j+e
    g = q.reshape(N, -1, 64)
    return (g * bf(s)[..., None] + bf(b)[..., None]).reshape(N, -1)
def qmm(x, prefix):                                   # fp64 exact of the dequantised matmul, bf16-rounded
    return r(x @ dequant(prefix).T)
def norm(h, w, eps=1e-6):
    inv = 1.0 / np.sqrt((h * h).sum(-1, keepdims=True) / 2048 + eps)
    return r(bf(w) * (h * inv))
silu = lambda x: x / (1.0 + np.exp(-x))

L = "model.layers.0."
emb = r(dequant("model.embed_tokens", ids))
x = norm(emb, T(L + "input_layernorm.weight"))
P = L + "linear_attn."
qkv, z, a, b = (qmm(x, P + n) for n in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"))
C, NK, DK = 6144, 16, 128
cw = bf(T(P + "conv1d.weight").reshape(C, 4))
pad = np.concatenate([np.zeros((3, C)), qkv])
acc = sum(cw[None, :, t] * pad[t:t + M] for t in range(4))
conv = r(acc)
vals = r(conv * r(1.0 / (1.0 + np.exp(-conv))))
q, k, v = (vals[:, :2048].reshape(M, 16, 128), vals[:, 2048:4096].reshape(M, 16, 128), vals[:, 4096:].reshape(M, 16, 128))
inv = lambda t: 1.0 / np.sqrt((t * t).sum(-1, keepdims=True) / DK + 1e-6 / DK)
Q = r(r(1.0 / DK) * r(q * inv(q)))
K = r(r(DK ** -0.5) * r(k * inv(k)))
s_ = r(a + bf(T(P + "dt_bias")))
sp = r(np.maximum(s_, 0) + np.log(1 + np.exp(-np.abs(s_))))
G = np.exp(-np.exp(T(P + "A_log").astype(np.float64))[None] * sp).astype(np.float32).astype(np.float64)
BETA = r(1.0 / (1.0 + np.exp(-b)))
S = np.zeros((16, 128, 128))                          # [hv][dv][dk]
y = np.zeros((M, 16, 128))
for n in range(M):
    S *= G[n][:, None, None]
    kv = np.einsum("hvk,hk->hv", S, K[n])
    delta = (v[n] - kv) * BETA[n][:, None]
    S += delta[:, :, None] * K[n][:, None, :]
    y[n] = np.einsum("hvk,hk->hv", S, Q[n])
y = r(y)
yn = y / np.sqrt((y * y).sum(-1, keepdims=True) / 128 + 1e-6)
xn = r(bf(T(P + "norm.weight"))[None, None] * yn)
mix = r(silu(z.reshape(M, 16, 128)) * xn).reshape(M, 2048)
rr = qmm(mix, P + "out_proj")
h = r(emb + rr)
ln2 = norm(h, T(L + "post_attention_layernorm.weight"))
gate, up = qmm(ln2, L + "mlp.gate_proj"), qmm(ln2, L + "mlp.up_proj")
act = r(silu(gate) * up)
out = r(h + qmm(act, L + "mlp.down_proj"))
for n, t in (("ln2", ln2), ("h", h), ("out", out)):
    t.astype("<f4").tofile(cap / f"mirror_l0_{n}.f32")
ref = np.fromfile(cap / "l0_out.f32", "<f4").astype(np.float64)
fl = lambda t: np.sqrt(np.mean((r(t) - t) ** 2))
print("mirror vs fp32 capture, layer output: rms %.3e, floor %.3e, ratio %.2fx" % (np.sqrt(np.mean((out.ravel() - ref) ** 2)), fl(ref), np.sqrt(np.mean((out.ravel() - ref) ** 2)) / fl(ref)))
