#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable
#define K 2048
#define E 16
__kernel void qwen35_norm_nores(__global const ushort* H, __global const ushort* Wt, const float eps, __global ushort* XO) {
  const uint t = get_local_id(0), m = get_group_id(1);
  const uint base = m * K + t * E;
  float hv[E], ss = 0.0f;
  for (int i = 0; i < E; i++) { hv[i] = intel_convert_as_bfloat16_float(H[base + i]); ss = fma(hv[i], hv[i], ss); }
  const float inv = rsqrt(work_group_reduce_add(ss) / (float)K + eps);
  for (int i = 0; i < E; i++)
    XO[base + i] = intel_convert_bfloat16_as_ushort(intel_convert_as_bfloat16_float(Wt[t * E + i]) * (hv[i] * inv));
}
