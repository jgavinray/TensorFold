#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable
#define DV 128
#define BF(x) intel_convert_as_bfloat16_float(x)
#define TOB(x) intel_convert_bfloat16_as_ushort(x)
__kernel void qwen35_gdn_post(__global const ushort* Y, __global const ushort* Z, __global const ushort* NW,
                              const float eps, __global ushort* OUT) {
  const uint d = get_local_id(0), hv = get_group_id(1), m = get_group_id(2);   // 128 work-items per (hv, m)
  const float y = BF(Y[(m * 16 + hv) * DV + d]);
  const float inv = rsqrt(work_group_reduce_add(y * y) / (float)DV + eps);
  const float x = BF(TOB(BF(NW[d]) * (y * inv)));
  const float zf = BF(Z[m * 2048 + hv * DV + d]);
  OUT[(m * 16 + hv) * DV + d] = TOB(zf / (1.0f + exp(-zf)) * x);
}
