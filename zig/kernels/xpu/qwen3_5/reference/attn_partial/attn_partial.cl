#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable
#define D 256
#define G 4
#define CK 128
#define BF(x) intel_convert_as_bfloat16_float(x)
__kernel void qwen35_attn_partial(__global const ushort* Q, __global const ushort* K, __global const ushort* V,
                                  const float scale, __global const int* dims,
                                  __global float* PM, __global float* PL, __global float* PO) {
  const int t = get_local_id(0), c = get_group_id(1), hw = get_group_id(2);   // 256 work-items
  const int P = dims[0], W = dims[1], CAP = dims[2], NCH = dims[3];
  const int h = hw / W, w = hw % W;
  __local float sq[G][D]; __local float ss[G][CK]; __local float sm[G], sl[G];
  const int last = P + w, k0 = c * CK, k1 = min(k0 + CK, last + 1);
  for (int g = 0; g < G; g++) sq[g][t] = BF(Q[(((h * G + g) * dims[5]) + dims[6] + w) * D + t]);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t < CK) {
    const int key = k0 + t;
    for (int g = 0; g < G; g++) {
      float d = 0.0f;
      if (key < k1) for (int i = 0; i < D; i++) d = fma(sq[g][i], BF(K[((size_t)h * CAP + key) * D + i]), d);
      ss[g][t] = key < k1 ? d * scale : -INFINITY;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (t < G) {
    float m = -INFINITY, l = 0.0f;
    for (int j = 0; j < CK; j++) m = fmax(m, ss[t][j]);
    if (k1 > k0) for (int j = 0; j < k1 - k0; j++) l += exp(ss[t][j] - m);
    sm[t] = m; sl[t] = l;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int g = 0; g < G; g++) {
    float acc = 0.0f;
    for (int j = 0; j < k1 - k0; j++) acc = fma(exp(ss[g][j] - sm[g]), BF(V[((size_t)h * CAP + k0 + j) * D + t]), acc);
    const int slot = ((h * G + g) * W + w) * NCH + c;
    PO[(size_t)slot * D + t] = acc;
    if (t == 0) { PM[slot] = sm[g]; PL[slot] = sl[g]; }
  }
}
