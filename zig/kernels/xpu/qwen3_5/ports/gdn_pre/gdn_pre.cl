// qwen35_gdn_pre: conv1d(4 taps) + SiLU, per-head L2 normalisation of q/k, v copy, gate g, beta,
// and the conv-state hand-off, for the Qwen3.5-2B GatedDeltaNet layer (layer 0), Intel Arc Pro B70.
//
// Work split: one work-group of 128 work-items per (head, row). Work-item d owns channel c0 + d.
// 128 work-items = four Intel sub-groups of 32 (or eight of 16): the per-head sum of squares uses
// work_group_reduce_add, so the kernel is correct for any sub-group width the runtime picks.
#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#define NK 16
#define NV 16
#define DK 128
#define DV 128
#define TAPS 4
#define CC (2 * NK * DK + NV * DV)   // 6144 channels

#define BF(x) intel_convert_as_bfloat16_float(x)
#define TOB(x) intel_convert_bfloat16_as_ushort(x)
#define RBF(x) BF(TOB(x))            // round to nearest-even bf16 and back

__kernel void qwen35_gdn_pre(__global const ushort* QKV, __global const ushort* CS, __global const ushort* CW,
                             __global const int* windows, __global const ushort* Ain, __global const ushort* Bin,
                             __global const float* ALOG, __global const ushort* DT,
                             __global ushort* Q, __global ushort* Kout, __global ushort* Vout,
                             __global float* G, __global ushort* BETA, __global ushort* CO) {
  const int d = get_local_id(0);
  const int head = get_group_id(1);
  const int w = get_group_id(2);

  const bool isq = head < NK;
  const bool isk = !isq && head < 2 * NK;
  const int c0 = isq ? head * DK : (isk ? NK * DK + (head - NK) * DK : 2 * NK * DK + (head - 2 * NK) * DV);
  const int c = c0 + d;

  // conv1d over the four window taps, then SiLU, with the reference's bf16 rounding points.
  float acc = 0.0f;
  for (int tap = 0; tap < TAPS; tap++) {
    const int row = windows[w * TAPS + tap];
    const float xv = row < TAPS - 1 ? BF(CS[row * CC + c]) : BF(QKV[(row - (TAPS - 1)) * CC + c]);
    acc += BF(CW[c * TAPS + tap]) * xv;
  }
  const float conv = RBF(acc);
  const float sig = RBF(1.0f / (1.0f + exp(-conv)));
  const float val = RBF(conv * sig);

  if (isq || isk) {
    const float ss = work_group_reduce_add(val * val);
    const float inv = rsqrt(ss / (float)DK + (1e-6f / (float)DK));
    const float scale = isq ? RBF(1.0f / (float)DK) : RBF(rsqrt((float)DK));
    const ushort o = TOB(scale * RBF(val * inv));
    if (isq) {
      Q[(w * NK + head) * DK + d] = o;
    } else {
      Kout[(w * NK + head - NK) * DK + d] = o;
    }
  } else {
    const int hv = head - 2 * NK;
    Vout[(w * NV + hv) * DV + d] = TOB(val);
    if (d == 0) {
      const float s = RBF(BF(Ain[w * NV + hv]) + BF(DT[hv]));
      const float sp = RBF(fmax(s, 0.0f) + log(1.0f + exp(-fabs(s))));
      G[w * NV + hv] = exp(-exp(ALOG[hv]) * sp);
      BETA[w * NV + hv] = TOB(1.0f / (1.0f + exp(-BF(Bin[w * NV + hv]))));
    }
  }

  // Conv-state hand-off: rows windows[w][1..3] copied bit for bit.
  for (int r = 0; r < TAPS - 1; r++) {
    const int row = windows[w * TAPS + 1 + r];
    CO[(w * (TAPS - 1) + r) * CC + c] =
        row < TAPS - 1 ? CS[row * CC + c] : QKV[(row - (TAPS - 1)) * CC + c];
  }
}
