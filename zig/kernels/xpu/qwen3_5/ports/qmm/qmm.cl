// qwen35_qmm: generic int4-affine matmul (group size 64) for the Intel Arc Pro B70.
//
//   OUT[r][n] = sum over groups g of ( SC[n][g] * sum_{k in g} q[n][k]*x[r][k]
//                                     + BI[n][g] * sum_{k in g} x[r][k] )
//
// Weights are never rounded: q is exact int4 -> float, SC/BI are bf16 -> float.
// Activations bf16 -> float, fp32 accumulation, a single round to bf16 at the end.
//
// Split-K geometry. A work-group covers a TILE_M x TILE_N tile of OUT. The 16 lanes of a
// sub-group share one column strip and split K between them: in step m lane L takes the
// RUN words at L*RUN + m*SPLIT*RUN, so the lanes sweep the row in strides of SPLIT*RUN.
// Because a group is GWORDS words and RUN is a multiple of GWORDS, every group belongs to
// exactly one lane in one step, so SC/BI are applied inside the lane on complete groups.
// The lanes then sum their already-scaled partials, which is exact: each lane contributes
// a whole-group term and float addition of whole-group terms is what the reference does.
//
// The payoff is coalescing: in one step the 16 lanes read a contiguous SPLIT*RUN-word span
// of the row, so the 254 MB weight stream arrives as wide coalesced blocks instead of one
// cache line per lane.
#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#define TILE_M 64
#define TILE_N 32
#define WG_M 8
#define WG_N 2
#define TG_M (TILE_M / WG_M)     // output rows per thread
#define TG_N (TILE_N / WG_N)     // output cols per thread
#define GROUP 64
#define GWORDS (GROUP / 8)       // uint32 words per group (8)
#define RUN 16                   // weight words per lane (2 complete groups)
#define SPLIT 16                 // lanes that share one column

static inline float bf16_to_f32(const ushort h) { return intel_convert_as_bfloat16_float(h); }

static inline ushort f32_to_bf16(const float f) { return intel_convert_bfloat16_as_ushort(f); }

// One nibble of a packed word, as an exact small integer float.
static inline float nib(const uint w, const int e) { return (float)((w >> (4 * e)) & 0xFu); }

__attribute__((intel_reqd_sub_group_size(16))) __kernel void qwen35_qmm(
    __global const ushort *X, const int R, __global const uint *W, __global const ushort *SC,
    __global const ushort *BI, __global ushort *OUT, const int N, const int K) {
  const int n0 = (int)get_group_id(0) * TILE_N;
  const int m0 = (int)get_group_id(1) * TILE_M;
  const int lid = (int)get_local_id(0);
  const int sg = lid / SPLIT;      // sub-group id: picks the row strip and column strip
  const int lane = lid % SPLIT;    // lane within the sub-group: picks the k run

  const int ng = K / GROUP;     // groups per output row
  const int kw = K / 8;         // uint32 words per weight row (8 int4 per word)
  const int xw_stride = K / 2;  // uint32 words per activation row when X is viewed as uint*

  // Each sub-group handles one (column strip, row strip) pair: WG_M strips of TG_M rows
  // and WG_N strips of TG_N columns, mapped from the sub-group id.
  const int tm = sg / WG_N;
  const int tn = sg % WG_N;
  const int col = n0 + tn * TG_N;
  const int row = m0 + tm * TG_M;

  int xr[TG_M];
  for (int i = 0; i < TG_M; i++) {
    const int r = row + i;
    xr[i] = r < R ? r : R - 1;
  }

  float acc[TG_M][TG_N];
  for (int i = 0; i < TG_M; i++)
    for (int j = 0; j < TG_N; j++) acc[i][j] = 0.0f;

  // Walk this lane's words for every one of its columns. A lane owns the words at
  // lane*RUN + m*(SPLIT*RUN) for m = 0, 1, 2, ..., so any K is covered: the SPLIT lanes
  // sweep the row in strides of SPLIT*RUN words. Consecutive lanes read words RUN apart,
  // so the sub-group's reads stay contiguous.
  //
  // Both loops must have a uniform trip count across the sub-group: sub_group_reduce_add
  // later requires every lane to reach it from the same control-flow point. So the outer
  // loop is bounded by a value every lane shares and the inner guard skips, not breaks.
  const int nsteps = (kw + SPLIT * RUN - 1) / (SPLIT * RUN);
  for (int m = 0; m < nsteps; m++) {
    const int kbase = lane * RUN + m * (SPLIT * RUN);
    for (int t = 0; t < RUN; t++) {
      const int kword = kbase + t;
      if (kword >= kw) continue;   // uniform guard: no lane leaves the loop early
      const int g = kword / GWORDS;   // group of this word (entirely inside this lane)
      const int xword = kword * 4;    // 8 bf16 = four uint32 of the activation row

      float xa[TG_M][8];
      float xsum[TG_M];
      for (int i = 0; i < TG_M; i++) {
        const __global uint *xrow = (const __global uint *)X + (size_t)xr[i] * xw_stride + xword;
        const uint4 v = *(__global const uint4 *)xrow;
        xa[i][0] = bf16_to_f32((ushort)(v.x & 0xFFFFu));
        xa[i][1] = bf16_to_f32((ushort)(v.x >> 16));
        xa[i][2] = bf16_to_f32((ushort)(v.y & 0xFFFFu));
        xa[i][3] = bf16_to_f32((ushort)(v.y >> 16));
        xa[i][4] = bf16_to_f32((ushort)(v.z & 0xFFFFu));
        xa[i][5] = bf16_to_f32((ushort)(v.z >> 16));
        xa[i][6] = bf16_to_f32((ushort)(v.w & 0xFFFFu));
        xa[i][7] = bf16_to_f32((ushort)(v.w >> 16));
        xsum[i] =
            xa[i][0] + xa[i][1] + xa[i][2] + xa[i][3] + xa[i][4] + xa[i][5] + xa[i][6] + xa[i][7];
      }

      for (int j = 0; j < TG_N; j++) {
        const uint w = W[(size_t)(col + j) * kw + kword];
        const size_t idx = (size_t)(col + j) * ng + g;
        const float sc = bf16_to_f32(SC[idx]);
        const float bi = bf16_to_f32(BI[idx]);
        float q[8];
        q[0] = nib(w, 0);
        q[1] = nib(w, 1);
        q[2] = nib(w, 2);
        q[3] = nib(w, 3);
        q[4] = nib(w, 4);
        q[5] = nib(w, 5);
        q[6] = nib(w, 6);
        q[7] = nib(w, 7);
        for (int i = 0; i < TG_M; i++) {
          const float *xa_i = xa[i];
          float d = 0.0f;
          d = fma(q[0], xa_i[0], d);
          d = fma(q[1], xa_i[1], d);
          d = fma(q[2], xa_i[2], d);
          d = fma(q[3], xa_i[3], d);
          d = fma(q[4], xa_i[4], d);
          d = fma(q[5], xa_i[5], d);
          d = fma(q[6], xa_i[6], d);
          d = fma(q[7], xa_i[7], d);
          acc[i][j] = fma(sc, d, fma(bi, xsum[i], acc[i][j]));
        }
      }
    }
  }

  // Reduce the SPLIT lanes that share a column: each lane contributed whole-group terms
  // already scaled by SC and BI, so a plain sum is exact. sub_group_reduce_add does this
  // in registers and needs no local memory, so every lane ends up holding the total.
  for (int j = 0; j < TG_N; j++) {
    for (int i = 0; i < TG_M; i++) acc[i][j] = sub_group_reduce_add(acc[i][j]);
  }

  if (lane == 0) {
    for (int j = 0; j < TG_N; j++) {
      const int n = col + j;
      if (n >= N) continue;
      for (int i = 0; i < TG_M; i++) {
        const int r = row + i;
        if (r >= R) continue;
        OUT[(size_t)r * N + n] = f32_to_bf16(acc[i][j]);
      }
    }
  }
}
