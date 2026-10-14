#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable
#define N 6144
__kernel void qwen35_mlp_act(__global const ushort* GATE, __global const ushort* UP, __global ushort* HOUT) {
  const uint i = get_global_id(0), m = get_global_id(1);
  if (i >= N) return;
  const float gf = intel_convert_as_bfloat16_float(GATE[m * N + i]);
  HOUT[m * N + i] = intel_convert_bfloat16_as_ushort(gf / (1.0f + exp(-gf)) * intel_convert_as_bfloat16_float(UP[m * N + i]));
}
