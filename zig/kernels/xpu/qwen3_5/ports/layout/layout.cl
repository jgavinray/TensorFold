// Qwen3.5-2B layout kernels - port of layout.metal to OpenCL C for the Intel Arc Pro B70.
//
// Five kernels: embedding dequant, per-head RMSNorm, partial RoPE (queries and keys), and the
// sigmoid attention gate. Apple-only primitives are replaced by plain grid geometry:
//
//   * head_norm: the Metal 2-SIMD-group reduction is re-expressed with work_group_reduce_add, which is
//     correct for 16- and 32-wide Intel sub-groups. Work-group size stays 64, matching the Metal local
//     size, and every group owns exactly one (token, head) row, so the launch grid is a flat 64*rows*heads.
//   * queries/keys: Metal's thread (d, h, r) grid is kept verbatim; dim0 is 256 wide, so the local size
//     must divide 256 (256 or 32 are both valid on Xe).
//   * embed: Metal's (1024, R) grid with a 64-wide group is kept; group g covers bytes [64g, 64g+64) of a
//     row, which is exactly one 64-element dequant group (32 scale/bias entries), so scale and bias are
//     loaded once per work-item instead of once per byte.
//   * attention_gate: thread (d, r) with d < 2048; d / 256 selects the head, d % 256 the element.
//
// All row offsets are size_t: the real embedding table has 248,320 rows (1024 B each) and the RoPE
// outputs are indexed by absolute cache rows, both of which overflow 32-bit byte arithmetic.
// bf16 is carried as ushort bits; every store converts once through intel_convert_bfloat16_as_ushort.

#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#define HEAD_DIM 256

static inline float bf16_to_f32(ushort u) {
  return intel_convert_as_bfloat16_float(u);
}

static inline ushort f32_to_bf16(float x) {
  return intel_convert_bfloat16_as_ushort(x);
}

// Partial RoPE of one element d of a 256-wide head, position `position`:
// d >= 64 is a bit copy; otherwise the pair (i, i + 32), i = d % 32, rotates by
// angle = position * 1e7^(-i/32), the lower half taking a*cs - b*sn and the upper b*cs + a*sn.
// pow/cos/sin are evaluated in fp32 as in Metal (native_* maps to the hardware functions there).
static inline ushort rope_rotate(const __global ushort* x, uint d, uint position) {
  if (d >= 64u) return x[d];
  const uint i = d % 32u;
  const float angle = (float)position * pow(10000000.0f, -((float)i) / 32.0f);
  const float cs = cos(angle), sn = sin(angle);
  const float a = bf16_to_f32(x[i]);
  const float b = bf16_to_f32(x[i + 32u]);
  return f32_to_bf16(d < 32u ? fma(a, cs, -(b * sn)) : fma(b, cs, a * sn));
}

// out[r][2b]   = bf16(s*(byte&15) + z)
// out[r][2b+1] = bf16(s*(byte>>4) + z)      s, z = scale/bias of group (2b)/64 of row ids[r]
__kernel void qwen35_embed(__global const uint* ids,
                           __global const uchar* weight,
                           __global const ushort* scales,
                           __global const ushort* biases,
                           __global ushort* out) {
  const uint b = get_global_id(0);   // byte index within the row, 0..1023
  const uint r = get_global_id(1);   // token (row) index
  const size_t id = (size_t)ids[r];

  const uchar v = weight[id * 1024u + b];
  const size_t g = id * 32u + (size_t)((2u * b) / 64u);
  const float s = bf16_to_f32(scales[g]);
  const float z = bf16_to_f32(biases[g]);

  const size_t o = (size_t)r * 2048u + (size_t)(2u * b);
  out[o] = f32_to_bf16(fma(s, (float)(v & 15u), z));
  out[o + 1u] = f32_to_bf16(fma(s, (float)(v >> 4), z));
}

// One 64-work-item group per (token m, head): row = group index, head = row % dims[1], m = row / dims[1];
// base = m*dims[2] + head*dims[3]; out[(m*dims[1] + head)*256 + d] = bf16(w[d] * v[d] * rsqrt(mean(v^2) + 1e-6)).
__kernel void qwen35_head_norm(__global const ushort* x,
                               __global const ushort* weight,
                               __global ushort* out,
                               __global const uint* dims) {
  const uint row = get_group_id(0);
  const uint t = get_local_id(0);    // 0..63
  const uint heads = dims[1];
  const uint head = row % heads;
  const uint m = row / heads;

  const size_t base = (size_t)m * dims[2] + (size_t)head * dims[3];

  float v[4];
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    v[i] = bf16_to_f32(x[base + (size_t)(4u * t + (uint)i)]);
    acc = fma(v[i], v[i], acc);
  }

  // 64 items * 4 dims = 256, so the group total is the row sum of squares; work_group_reduce_add is
  // width-agnostic (16- or 32-wide sub-groups), unlike Metal's 2-SIMD simd_sum.
  const float total = work_group_reduce_add(acc);
  const float inv = rsqrt(total / 256.0f + 1e-6f);

  const size_t o = ((size_t)m * heads + (size_t)head) * HEAD_DIM;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const uint d = 4u * t + (uint)i;
    out[o + (size_t)d] = f32_to_bf16(bf16_to_f32(weight[d]) * (v[i] * inv));
  }
}

// thread (d, h, r): m = dims[2] + r; position = dims[0] + r;
// out[(h*dims[1] + m)*256 + d] = rotate(x + (m*8 + h)*256, d, position).
// Rows outside [dims[2], dims[2] + R) are never touched, so a windowed launch leaves the rest of the cache alone.
__kernel void qwen35_queries(__global const ushort* x,
                             __global ushort* out,
                             __global const uint* dims) {
  const uint d = get_global_id(0);
  const uint h = get_global_id(1);
  const uint r = get_global_id(2);

  const uint m = dims[2] + r;
  const __global ushort* src = x + ((size_t)m * 8u + (size_t)h) * HEAD_DIM;
  out[((size_t)h * dims[1] + (size_t)m) * HEAD_DIM + (size_t)d] = rope_rotate(src, d, dims[0] + r);
}

// thread (d, h, r): m = dims[2] + r; dst = (h*dims[1] + dims[0] + r)*256 + d;
// keys[dst] = rotate(k + (m*2 + h)*256, d, dims[0] + r); values[dst] = v[(m*2 + h)*256 + d].
// Cache rows outside [dims[0], dims[0] + R) stay untouched.
__kernel void qwen35_keys(__global const ushort* k,
                          __global const ushort* v,
                          __global ushort* keys,
                          __global ushort* values,
                          __global const uint* dims) {
  const uint d = get_global_id(0);
  const uint h = get_global_id(1);
  const uint r = get_global_id(2);

  const uint m = dims[2] + r;
  const size_t src = ((size_t)m * 2u + (size_t)h) * HEAD_DIM;
  const size_t dst = ((size_t)h * dims[1] + (size_t)dims[0] + (size_t)r) * HEAD_DIM + (size_t)d;

  keys[dst] = rope_rotate(k + src, d, dims[0] + r);
  values[dst] = v[src + (size_t)d];
}

// thread (d, r): head = d/256; gate = q[r*4096 + head*512 + 256 + d%256]; s = bf16(1/(1+exp(-gate)));
// out[r*2048 + d] = bf16(x[r*2048 + d] * s). The sigmoid is rounded to bf16 before the product, as in Metal.
__kernel void qwen35_attention_gate(__global const ushort* x,
                                    __global const ushort* q,
                                    __global ushort* out) {
  const uint d = get_global_id(0);
  const uint r = get_global_id(1);
  const uint head = d / HEAD_DIM;

  const float gate = bf16_to_f32(q[(size_t)r * 4096u + (size_t)head * 512u + 256u + (size_t)(d % HEAD_DIM)]);
  const float s = bf16_to_f32(f32_to_bf16(1.0f / (1.0f + exp(-gate))));

  out[(size_t)r * 2048u + (size_t)d] = f32_to_bf16(bf16_to_f32(x[(size_t)r * 2048u + (size_t)d]) * s);
}
