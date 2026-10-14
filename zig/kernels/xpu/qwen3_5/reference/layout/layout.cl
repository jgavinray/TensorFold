#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable
#define BF(x) intel_convert_as_bfloat16_float(x)
#define TOB(x) intel_convert_bfloat16_as_ushort(x)
#define RBF(x) BF(TOB(x))

__kernel void qwen35_embed(__global const uint* ids, __global const uchar* weight, __global const ushort* scales,
                           __global const ushort* biases, __global ushort* out) {
  const uint b = get_global_id(0), r = get_global_id(1);
  const size_t id = ids[r];
  const uchar v = weight[id * 1024 + b];
  const size_t g = id * 32 + (2 * b) / 64;
  const float s = BF(scales[g]), z = BF(biases[g]);
  out[(size_t)r * 2048 + 2 * b] = TOB(s * (float)(v & 15) + z);
  out[(size_t)r * 2048 + 2 * b + 1] = TOB(s * (float)(v >> 4) + z);
}

__kernel void qwen35_head_norm(__global const ushort* x, __global const ushort* weight, __global ushort* out,
                               __global const uint* dims) {
  const uint row = get_group_id(0), t = get_local_id(0);
  const uint head = row % dims[1], m = row / dims[1];
  const size_t base = (size_t)m * dims[2] + head * dims[3];
  float v[4], acc = 0.0f;
  for (int i = 0; i < 4; i++) { v[i] = BF(x[base + 4 * t + i]); acc += v[i] * v[i]; }
  const float inv = rsqrt(work_group_reduce_add(acc) / 256.0f + 1e-6f);
  for (int i = 0; i < 4; i++) {
    const uint d = 4 * t + i;
    out[((size_t)m * dims[1] + head) * 256 + d] = TOB(BF(weight[d]) * (v[i] * inv));
  }
}

inline ushort rotate(__global const ushort* x, uint d, uint position) {
  if (d >= 64) return x[d];
  const uint i = d % 32;
  const float angle = (float)position * pow(10000000.0f, -(float)i / 32.0f);
  const float cs = cos(angle), sn = sin(angle);
  const float a = BF(x[i]), b = BF(x[i + 32]);
  return TOB(d < 32 ? a * cs - b * sn : b * cs + a * sn);
}

__kernel void qwen35_queries(__global const ushort* x, __global ushort* out, __global const uint* dims) {
  const uint d = get_global_id(0), h = get_global_id(1), r = get_global_id(2);
  const uint m = dims[2] + r;
  out[((size_t)h * dims[1] + m) * 256 + d] = rotate(x + ((size_t)m * 8 + h) * 256, d, dims[0] + r);
}

__kernel void qwen35_keys(__global const ushort* k, __global const ushort* v, __global ushort* keys,
                          __global ushort* values, __global const uint* dims) {
  const uint d = get_global_id(0), h = get_global_id(1), r = get_global_id(2);
  const uint m = dims[2] + r;
  const size_t dst = ((size_t)h * dims[1] + dims[0] + r) * 256 + d;
  keys[dst] = rotate(k + ((size_t)m * 2 + h) * 256, d, dims[0] + r);
  values[dst] = v[((size_t)m * 2 + h) * 256 + d];
}

__kernel void qwen35_attention_gate(__global const ushort* x, __global const ushort* q, __global ushort* out) {
  const uint d = get_global_id(0), r = get_global_id(1), head = d / 256;
  const float gate = BF(q[(size_t)r * 4096 + head * 512 + 256 + d % 256]);
  const float s = RBF(1.0f / (1.0f + exp(-gate)));
  out[(size_t)r * 2048 + d] = TOB(BF(x[(size_t)r * 2048 + d]) * s);
}
