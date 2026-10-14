#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable
#define BF(x) intel_convert_as_bfloat16_float(x)
__kernel void qwen35_qmm(__global const ushort* X, const int R, __global const uint* W, __global const ushort* SC,
                         __global const ushort* BI, __global ushort* OUT, const int N, const int K) {
  const int n = get_global_id(0), r = get_global_id(1);
  if (n >= N || r >= R) return;
  float acc = 0.0f;
  for (int g = 0; g < K / 64; g++) {
    const float s = BF(SC[n * (K / 64) + g]), b = BF(BI[n * (K / 64) + g]);
    float qx = 0.0f, xs = 0.0f;
    for (int j = 0; j < 8; j++) {
      const uint w = W[n * (K / 8) + g * 8 + j];
      for (int e = 0; e < 8; e++) {
        const float x = BF(X[r * K + g * 64 + j * 8 + e]);
        qx = fma((float)((w >> (4 * e)) & 0xF), x, qx);
        xs += x;
      }
    }
    acc += s * qx + b * xs;
  }
  OUT[r * N + n] = intel_convert_bfloat16_as_ushort(acc);
}
