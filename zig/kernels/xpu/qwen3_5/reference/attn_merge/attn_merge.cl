#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable
#define D 256
__kernel void qwen35_attn_merge(__global const float* PM, __global const float* PL, __global const float* PO,
                                __global const int* dims, __global ushort* OUT) {
  const int d = get_local_id(0), qh = get_group_id(1), w = get_group_id(2);   // 256 work-items per (qh, w)
  const int W = dims[1], NCH = dims[3];
  const int base = (qh * W + w) * NCH;
  float mx = -INFINITY;
  for (int c = 0; c < NCH; c++) mx = fmax(mx, PM[base + c]);
  float lsum = 0.0f, acc = 0.0f;
  for (int c = 0; c < NCH; c++) {
    const float e = PL[base + c] > 0.0f ? exp(PM[base + c] - mx) : 0.0f;
    lsum = fma(PL[base + c], e, lsum);
    acc = fma(PO[(size_t)(base + c) * D + d], e, acc);
  }
  OUT[((w * 8) + qh) * D + d] = intel_convert_bfloat16_as_ushort(acc / lsum);
}
