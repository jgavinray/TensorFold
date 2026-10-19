// qwen35_gdn_pre: conv1d(QW_TAPS taps) + SiLU, per-head L2 normalisation of q/k, v copy, gate g, beta,
// and the conv-state hand-off, for the Qwen3.5/Qwen 3.8 GatedDeltaNet layer, Intel Arc Pro B70.
//
// Work split: one work-group of QW_DK work-items per (head, row). Work-item d owns channel c0 + d.
// QW_DK work-items are four Intel sub-groups of 32 (or eight of 16): the per-head sum of squares uses
// work_group_reduce_add, so the kernel is correct for any sub-group width the runtime picks.
//
// Shapes are compile-time defines; defaults are the Qwen3.5-2B geometry, so the proven 2B gate applies
// to this file unchanged. The 27B build overrides QW_NV (48 value heads; 2B has 16), sets QW_ALOG_BF16 (A_log's
// storage, see below) and QW_TAPS stays 4;
// QW_NK (16 key/query heads), QW_DK/QW_DV (128) and the channel layout are shared. The total channel
// count QW_CC = 2*QW_NK*QW_DK + QW_NV*QW_DV is 6144 on 2B and 10240 on 27B.
#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#ifndef QW_NK
#define QW_NK 16
#endif
#ifndef QW_NV
#define QW_NV 16
#endif
#ifndef QW_DK
#define QW_DK 128
#endif
#ifndef QW_DV
#define QW_DV 128
#endif
#ifndef QW_TAPS
#define QW_TAPS 4
#endif
#define QW_CC (2 * QW_NK * QW_DK + QW_NV * QW_DV)   // total channels: q heads, k heads, v heads
#define QW_HEADS3 (2 * QW_NK + QW_NV)               // work-groups along y: q heads + k heads + v heads

#define BF(x) intel_convert_as_bfloat16_float(x)
#define TOB(x) intel_convert_bfloat16_as_ushort(x)
#define RBF(x) BF(TOB(x))            // round to nearest-even bf16 and back

// A_log's storage follows the checkpoint: the Qwen3.5-2B keeps it fp32 (the default here), the Qwen 3.8 27B MLX
// checkpoint stores it bf16 and its build passes -DQW_ALOG_BF16 (xpu_weights.zig loads it as .bf16, raw bytes). Read
// as fp32, a bf16 A_log gave value head hv the bits of heads 2hv+1/2hv (hv < 24) and arena padding (hv >= 24): every
// GDN layer ran with the wrong decay rates (mean |dNLL| 0.29 vs MLX fp32, 2026-10-19; exp/precision).
#ifdef QW_ALOG_BF16
typedef ushort alog_t;
#define ALOG_F(x) BF(x)
#else
typedef float alog_t;
#define ALOG_F(x) (x)
#endif

__kernel void qwen35_gdn_pre(__global const ushort* QKV, __global const ushort* CS, __global const ushort* CW,
                             __global const int* windows, __global const ushort* Ain, __global const ushort* Bin,
                             __global const alog_t* ALOG, __global const ushort* DT,
                             __global ushort* Q, __global ushort* Kout, __global ushort* Vout,
                             __global float* G, __global ushort* BETA, __global ushort* CO) {
  const int d = get_local_id(0);
  const int head = get_group_id(1);
  const int w = get_group_id(2);

  const bool isq = head < QW_NK;
  const bool isk = !isq && head < 2 * QW_NK;
  const int c0 = isq ? head * QW_DK : (isk ? QW_NK * QW_DK + (head - QW_NK) * QW_DK : 2 * QW_NK * QW_DK + (head - 2 * QW_NK) * QW_DV);
  const int c = c0 + d;

  // conv1d over the QW_TAPS window taps, then SiLU, with the reference's bf16 rounding points.
  float acc = 0.0f;
  for (int tap = 0; tap < QW_TAPS; tap++) {
    const int row = windows[w * QW_TAPS + tap];
    const float xv = row < QW_TAPS - 1 ? BF(CS[row * QW_CC + c]) : BF(QKV[(row - (QW_TAPS - 1)) * QW_CC + c]);
    acc += BF(CW[c * QW_TAPS + tap]) * xv;
  }
  const float conv = RBF(acc);
  const float sig = RBF(1.0f / (1.0f + exp(-conv)));
  const float val = RBF(conv * sig);

  if (isq || isk) {
    const float ss = work_group_reduce_add(val * val);
    const float inv = rsqrt(ss / (float)QW_DK + (1e-6f / (float)QW_DK));
    const float scale = isq ? RBF(1.0f / (float)QW_DK) : RBF(rsqrt((float)QW_DK));
    const ushort o = TOB(scale * RBF(val * inv));
    if (isq) {
      Q[(w * QW_NK + head) * QW_DK + d] = o;
    } else {
      Kout[(w * QW_NK + head - QW_NK) * QW_DK + d] = o;
    }
  } else {
    const int hv = head - 2 * QW_NK;
    Vout[(w * QW_NV + hv) * QW_DV + d] = TOB(val);
    if (d == 0) {
      const float s = RBF(BF(Ain[w * QW_NV + hv]) + BF(DT[hv]));
      const float sp = RBF(fmax(s, 0.0f) + log(1.0f + exp(-fabs(s))));
      G[w * QW_NV + hv] = exp(-exp(ALOG_F(ALOG[hv])) * sp);
      BETA[w * QW_NV + hv] = TOB(1.0f / (1.0f + exp(-BF(Bin[w * QW_NV + hv]))));
    }
  }

  // Conv-state hand-off: rows windows[w][1..QW_TAPS-1] copied bit for bit.
  for (int r = 0; r < QW_TAPS - 1; r++) {
    const int row = windows[w * QW_TAPS + 1 + r];
    CO[(w * (QW_TAPS - 1) + r) * QW_CC + c] =
        row < QW_TAPS - 1 ? CS[row * QW_CC + c] : QKV[(row - (QW_TAPS - 1)) * QW_CC + c];
  }
}
