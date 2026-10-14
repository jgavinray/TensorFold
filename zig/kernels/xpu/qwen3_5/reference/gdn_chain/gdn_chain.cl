#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable
#define DK 128
#define DV 128
#define HK 16
#define HV 16
#define BF(x) intel_convert_as_bfloat16_float(x)
__kernel void qwen35_gdn_chain(__global const ushort* q, __global const ushort* k, __global const ushort* v,
                               __global const float* g, __global const ushort* beta, __global const float* state_in,
                               __global const int* dims, __global ushort* y, __global float* state_out) {
  const int dv = get_global_id(0), hv = get_global_id(1);        // one work-item owns one state row (128 dk values)
  const int hk = hv / (HV / HK), W = dims[0];
  float s[DK];
  for (int i = 0; i < DK; i++) s[i] = state_in[(hv * DV + dv) * DK + i];
  for (int node = 0; node < W; node++) {
    const float gc = g[node * HV + hv], bc = BF(beta[node * HV + hv]), vc = BF(v[(node * HV + hv) * DV + dv]);
    float kv = 0.0f;
    for (int i = 0; i < DK; i++) { s[i] *= gc; kv = fma(s[i], BF(k[(node * HK + hk) * DK + i]), kv); }
    const float delta = (vc - kv) * bc;
    float out = 0.0f;
    for (int i = 0; i < DK; i++) { s[i] = fma(BF(k[(node * HK + hk) * DK + i]), delta, s[i]); out = fma(s[i], BF(q[(node * HK + hk) * DK + i]), out); }
    y[(node * HV + hv) * DV + dv] = intel_convert_bfloat16_as_ushort(out);
    if (dims[1] || node + 1 == W) {
      const int saved = dims[1] ? node : 0;
      for (int i = 0; i < DK; i++) state_out[((saved * HV + hv) * DV + dv) * DK + i] = s[i];
    }
  }
}
