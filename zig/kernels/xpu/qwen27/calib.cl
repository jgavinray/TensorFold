// calib.cl: the fresh-quant calibration kernels (eaq-ldlq-spec-fable.md sections 1.4 and 2.1-2.2).
// bf16_to_slab stages a dense bf16 [N, K] tensor (HF out-major) into an fp16 [K, NT] weight slab
// the prompt GEMM reads like the dequant scratch; transpose16 builds X^T for the Hessian kernel;
// exl3_syrk32 accumulates H += X^T X in fp32 over the lower 128x128 tiles. Same launch idiom as
// exl3.cl: sub-group size 16, 256 threads as 4x4 subgroups over a 128x128 tile, DPAS k16 blocks.
#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

// ---- dense slab upload (section 1.4): dst[k * NT + n0 + n] = (half) bf16 src[n * K + k] for
// n < N, k < K; columns [n0 + N, n0 + Npad) zero. bf16 to fp16 is exact for |w| in [6.1e-5,
// 65504) through the fp32 widening (the low 16 bf16 bits land inside fp16's 10 explicit mantissa
// bits after the shift); the host counts the underflowing weights (uploadDense logs them).
// grid: x = K/16, y = ceil(Npad/16); a work group transposes a 16x16 tile through local memory.
KERNEL(256) void bf16_to_slab(global const ushort *SRC, global half *DST, int N, int K, int NT, int n0, int Npad) {
  if (K % 16 != 0 || Npad % 16 != 0) return; // workgroup-uniform trap
  const int k0 = get_group_id(0) * 16;
  const int nb0 = get_group_id(1) * 16; // the tile's first column inside the part's slab range
  const int tx = get_local_id(0) & 15, ty = get_local_id(0) >> 4;
  local half tile[16][16];
  const int n = nb0 + ty; // the part's own column (pad columns read nothing and store zero)
  tile[ty][tx] = (n < N) ? convert_half(as_float(((uint)SRC[(long)n * K + k0 + tx]) << 16)) : (half)(0.0f);
  barrier(CLK_LOCAL_MEM_FENCE);
  DST[(long)(k0 + ty) * NT + n0 + nb0 + tx] = tile[tx][ty]; // (n, k) -> DST[k][n0+n]: tile[tx][ty] holds output row nb0+tx, input column k0+ty, so the store addresses (k0+ty, n0+nb0+tx); coalesced along n
}

// ---- Hessian staging (section 2.2): XT[k][t] = X[t][k] over the padded row count; rows
// T..Tp of X are stale (they can hold Inf or NaN from an earlier batch), so the transpose
// reads only t < T and writes zeros below it: 0 x Inf would poison H (the syrk must never
// see a stale row). X is the capture's compact fp16 [Tp, K] staging (xpad), XT its fp16
// [K, Tp] partner (xt), both pitched as the syrk documents. grid: x = K/16, y = Tp/16.
KERNEL(256) void transpose16(global const half *X, global half *XT, int T, int Tp, int K) {
  if (K % 16 != 0 || Tp % 16 != 0) return;
  const int k0 = get_group_id(0) * 16;
  const int t0 = get_group_id(1) * 16;
  const int tx = get_local_id(0) & 15, ty = get_local_id(0) >> 4;
  local half tile[16][16];
  tile[ty][tx] = (t0 + ty < T) ? X[(long)(t0 + ty) * K + k0 + tx] : (half)(0.0f);
  barrier(CLK_LOCAL_MEM_FENCE);
  XT[(long)(k0 + tx) * Tp + t0 + ty] = tile[ty][tx]; // (t, k) -> XT[k][t]: tile[ty][tx] holds row t0+ty, column k0+tx, so the store addresses (k0+tx, t0+ty)
}

// ---- Hessian accumulation (section 2.1): H[bi, bj tile] += XT[K][Tp] . X[Tp][K], fp32 C,
// read-modify-write; tiles with bi < bj (the upper triangle) return before touching memory,
// so a sentinel planted there survives (T-HESS-3). The body is exl3_prompt16's (exl3.cl:302-338)
// with three deltas: fp32 C instead of fp16 out, the C tile read back and summed instead of a
// bare store, and the triangle skip. A = XT (the [i][t] operand, prompt16's A role with the
// pitch Tp), B = X read through the transform block (prompt16's B role with pitch K): the DPAS
// computes H[i][j] = sum_t XT[i][t] X[t][j]. kk runs ascending over Tp/16 once per element:
// no atomics, one owner per tile per launch, fixed order, so two captures of the same rows
// give bit-identical H (section 2.3, T-HESS-1). fp16 products are exact in fp32.
// grid: x = K/128 (bi), y = K/128 (bj). Uniform trap: K % 128 || Tp % 64 || Tp < 64 -> return.
// accumulate = 0: H tile = acc with no read and no triangle skip (the KL tool's plain fp32
// GEMM uses this form with the triangle skip off).
KERNEL(256) void exl3_syrk32(global const half *XT, global const half *X, global float *H,
                             int K, int Tp, int accumulate) {
  if (K % 128 != 0 || Tp % 64 != 0 || Tp < 64) return; // workgroup-uniform trap
  const int bi = get_group_id(0), bj = get_group_id(1);
  if (accumulate && bi < bj) return; // upper tile: exit before touching memory
  const int sg = get_sub_group_id();
  const int i0 = bi * 128 + (sg % 4) * 32; // the subgroup's H rows (XT rows)
  const int j0 = bj * 128 + (sg / 4) * 32; // and H columns (X columns)
  float8 acc[4][2];
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < 4; i++)
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 2; j++) acc[i][j] = 0.0f;
  for (int kk = 0; kk < Tp / 16; kk++) {
    short8 a[4];
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 4; i++)
      intel_sub_group_2d_block_read_16b_8r16x1c((global void *)XT, Tp * 2, K, Tp * 2,
                                                (int2)(kk * 16, i0 + 8 * i), (private ushort *)&a[i]);
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 2; j++) {
      uint8 bb;
      intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void *)X, K * 2, Tp, K * 2,
                                                           (int2)(j0 + 16 * j, kk * 16), (private uint *)&bb);
      __attribute__((opencl_unroll_hint))
      for (int i = 0; i < 4; i++) acc[i][j] = intel_sub_group_f16_f16_matrix_mad_k16(a[i], as_int8(bb), acc[i][j]);
    }
  }
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < 4; i++)
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 2; j++) {
      float8 out = acc[i][j];
      if (accumulate) {
        uint8 cur;
        intel_sub_group_2d_block_read_32b_8r16x1c((global void *)H, K * 4, K, K * 4,
                                                   (int2)(j0 + 16 * j, i0 + 8 * i), (private uint *)&cur);
        out += as_float8(cur);
      }
      intel_sub_group_2d_block_write_32b_8r16x1c((global void *)H, K * 4, K, K * 4,
                                                 (int2)(j0 + 16 * j, i0 + 8 * i), (private uint *)&out);
    }
}
