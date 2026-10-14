#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define DK 128
#define DV 128
#define HK 16
#define HV 16
#define BF(x) intel_convert_as_bfloat16_float(x)
// One 16-lane sub-group owns one state row (hv,dv); lane l holds dk = 4l..4l+3 and 64+4l..64+4l+3.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qwen35_gdn_chain(__global const ushort* q, __global const ushort* k, __global const ushort* v,
                               __global const float* g, __global const ushort* beta, __global const float* state_in,
                               __global const int* dims, __global ushort* y, __global float* state_out) {
  const int lane = get_sub_group_local_id();
  const int row = get_global_id(0) >> 4;           // 0 .. HV*DV-1
  const int hv = row / DV, dv = row % DV;
  const int hk = hv / (HV / HK), W = dims[0], save_all = dims[1];
  const int c0 = 4 * lane, c1 = 64 + 4 * lane;
  __global const float* sp = state_in + (size_t)row * DK;
  float4 s0 = vload4(0, sp + c0), s1 = vload4(0, sp + c1);
  for (int node = 0; node < W; node++) {
    const float gc = g[node * HV + hv], bc = BF(beta[node * HV + hv]), vc = BF(v[(node * HV + hv) * DV + dv]);
    __global const ushort* kp = k + (node * HK + hk) * DK;
    __global const ushort* qp = q + (node * HK + hk) * DK;
    ushort4 ku0 = vload4(0, kp + c0), ku1 = vload4(0, kp + c1);
    float4 k0 = (float4)(BF(ku0.x), BF(ku0.y), BF(ku0.z), BF(ku0.w));
    float4 k1 = (float4)(BF(ku1.x), BF(ku1.y), BF(ku1.z), BF(ku1.w));
    s0 *= gc; s1 *= gc;
    float4 p = s0 * k0 + s1 * k1;
    float kv = sub_group_reduce_add(p.x + p.y + p.z + p.w);
    const float delta = (vc - kv) * bc;
    s0 = fma(k0, (float4)(delta), s0);
    s1 = fma(k1, (float4)(delta), s1);
    ushort4 qu0 = vload4(0, qp + c0), qu1 = vload4(0, qp + c1);
    float4 q0 = (float4)(BF(qu0.x), BF(qu0.y), BF(qu0.z), BF(qu0.w));
    float4 q1 = (float4)(BF(qu1.x), BF(qu1.y), BF(qu1.z), BF(qu1.w));
    float4 o = s0 * q0 + s1 * q1;
    float out = sub_group_reduce_add(o.x + o.y + o.z + o.w);
    if (lane == 0) y[(node * HV + hv) * DV + dv] = intel_convert_bfloat16_as_ushort(out);
    if (save_all || node + 1 == W) {
      __global float* op = state_out + ((size_t)((save_all ? node : 0) * HV * DV) + row) * DK;
      vstore4(s0, 0, op + c0); vstore4(s1, 0, op + c1);
    }
  }
}
