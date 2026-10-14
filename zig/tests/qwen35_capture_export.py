"""Export the cached MLX-CPU fp32 layer-0 capture (exp/kernels/_cache/capture.npz, via exp/kernels/common.capture())
as raw little-endian files for the Zig Gate A: ids.u32 (176) and, for the 176-token sequence (160 prefix rows then 16
window rows), emb / l0_ln1 / l0_h / l0_ln2 / l0_out as float32 [176][2048]. Never starts an MLX run unless the cache is missing.
Usage: /storage01/venvs/mlx-ref/bin/python qwen35_capture_export.py <out_dir>"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/storage01/qwen-flash-on-intel/exp/kernels")
import common

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
if not common.CACHE.exists():
    sys.exit("capture cache missing; refusing to start a long MLX run")
z = common.capture()
z["ids"].astype("<u4").tofile(out / "ids.u32")
for name, key in (("emb", "emb"), ("l0_ln1", "L0.ln1"), ("l0_h", "L0.h"), ("l0_ln2", "L0.ln2"), ("l0_out", "L0.out")):
    a = np.concatenate([z["pre." + key], z["win." + key]], axis=1)[0].astype("<f4")
    assert a.shape == (176, 2048), a.shape
    a.tofile(out / (name + ".f32"))
print("exported", sorted(p.name for p in out.iterdir()))
