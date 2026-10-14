#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable
#define NK 16
#define NV 16
#define DK 128
#define DV 128
#define TAPS 4
#define CC (2 * NK * DK + NV * DV)
#define BF(x) intel_convert_as_bfloat16_float(x)
#define TOB(x) intel_convert_bfloat16_as_ushort(x)
#define RBF(x) BF(TOB(x))
__kernel void qwen35_gdn_pre(__global const ushort* QKV, __global const ushort* CS, __global const ushort* CW,
                             __global const int* windows, __global const ushort* Ain, __global const ushort* Bin,
                             __global const float* ALOG, __global const ushort* DT,
                             __global ushort* Q, __global ushort* Kout, __global ushort* Vout,
                             __global float* G, __global ushort* BETA, __global ushort* CO) {
  const int d = get_local_id(0), head = get_group_id(1), w = get_group_id(2);   // 128 work-items per (head, w)
  const bool isq = head < NK, isk = !isq && head < 2 * NK;
  const int c0 = isq ? head * DK : (isk ? NK * DK + (head - NK) * DK : 2 * NK * DK + (head - 2 * NK) * DV);
  const int c = c0 + d;
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
    if (isq) Q[(w * NK + head) * DK + d] = o; else Kout[(w * NK + head - NK) * DK + d] = o;
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
  for (int r = 0; r < TAPS - 1; r++) {
    const int row = windows[w * TAPS + 1 + r];
    CO[(w * (TAPS - 1) + r) * CC + c] = row < TAPS - 1 ? CS[row * CC + c] : QKV[(row - (TAPS - 1)) * CC + c];
  }
}
