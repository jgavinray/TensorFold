// The KL gate's fp32-output head GEMM (eaq-ldlq-spec-fable.md §5.1): logits fp32 [M][NT]
// = hidden fp16 [M][K] · head slab fp16 [K][NT]. exl3_prompt16 (exl3.cl:302) verbatim
// except the C store: the same 32x32-subgroup DPAS tile (f16 math, fp32 accumulator)
// writing float through the 32b block store (the gemm.cl W4S3_SPLIT pattern, gemm.cl:210),
// which is the exl3_syrk32 form of §2.1 at accumulate = 0 and the triangle skip off.
// Own file so the compile skips -cl-intel-256-GRF-per-thread, as the other exl3 images do.

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

// Grid: x = ceil(M/128), y = NT/128. K a multiple of 16, NT a multiple of 128. Out-of-bounds
// A rows read 0 (the 2d block IO contract) and stores past row M are dropped; the caller
// pitches C at NT (floats) exactly like exl3_prompt16 pitches its fp16 C.
KERNEL(16 * 4 * 4) void eaq_gemm32(global const half *A, global const half *B, global float *C,
                                   int M, int NT, int K) {
  if (K % 16 != 0 || NT % 128 != 0) return; // workgroup-uniform trap
  const int sg = get_sub_group_id();
  const int m0 = get_group_id(0) * 128 + (sg % 4) * 32;
  const int n0 = get_group_id(1) * 128 + (sg / 4) * 32;
  float8 acc[4][2];
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < 4; i++)
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 2; j++) acc[i][j] = 0.0f;
  for (int kk = 0; kk < K / 16; kk++) {
    short8 a[4];
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 4; i++)
      intel_sub_group_2d_block_read_16b_8r16x1c((global void *)A, K * 2, M, K * 2, (int2)(kk * 16, m0 + 8 * i),
                                                (private ushort *)&a[i]);
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 2; j++) {
      uint8 bb;
      intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void *)B, NT * 2, K, NT * 2,
                                                           (int2)(n0 + 16 * j, kk * 16), (private uint *)&bb);
      __attribute__((opencl_unroll_hint))
      for (int i = 0; i < 4; i++) acc[i][j] = intel_sub_group_f16_f16_matrix_mad_k16(a[i], as_int8(bb), acc[i][j]);
    }
  }
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < 4; i++) {
    if (m0 + 8 * i >= M) continue; // whole rows past M: the 2d block IO contract drops them
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 2; j++) {
      float8 f = acc[i][j];
      intel_sub_group_2d_block_write_32b_8r16x1c(C, NT * 4, M, NT * 4, (int2)(n0 + 16 * j, m0 + 8 * i),
                                                 (private uint *)&f);
    }
  }
}

// eaq_copy16: deliver A on the GEMM's own compute list. The fable ruling on the
// zeroed-rows defect (1b, 2026-10-24): no y/height contract in the 2d block IO can
// zero rows past the first tile row; the non-8-aligned, host-stack-moving band edges
// put the loss in the byte-granular copy delivery, ranked first at the upload. The
// H2D therefore ends at a device staging buffer on the copy list (the download's
// event-signaled path, the one proven to land bytes), and a same-list kernel copy —
// in-order behind nothing and ahead of the GEMM — makes A resident where DPAS reads
// it, so no cross-engine promotion can reorder the GEMM's read behind the DMA.
KERNEL(256) void eaq_copy16(global ushort *dst, global const ushort *src, unsigned int n) {
  const unsigned int gsize = get_global_size(0);
  for (unsigned int i = get_global_id(0); i < n; i += gsize) dst[i] = src[i];
}
// ---- probe-only A/B instrumentation (kl.zig's probeGemm; no serving path launches
// these). The failure mode to isolate: eaq_gemm32 delivers exact zeros for most rows
// of an m=1024 block. Every kernel below is eaq_gemm32's body verbatim plus the one
// thing the probe needs; same sub-group geometry, same IO calls, same buffers.

// eaq_dump_areg / eaq_dump_breg: the same intel_sub_group_2d_block_read_16b_8r16x1c
// (A) and intel_sub_group_2d_block_read_transform_16b_16r16x1c (B) calls, at the same
// pitches/heights/origins, dumping each register's raw bits through plain per-lane
// stores at the sample k-indices KKA/KKB (nk entries; other kk values run the read but
// are not dumped). A dump slot: ((((tileRow*16+sg)*4+i)*nk + x)*16 + lane)*8 + t ushort.
// B dump slot: ((((tileCol*16+sg)*2+j)*nk + x)*16 + lane)*8 + t uint.
KERNEL(16 * 4 * 4) void eaq_dump_areg(global const half *A, global const half *B, global float *C,
                                       global ushort *DA, global uint *DB,
                                       int M, int NT, int K, int nk,
                                       global const int *KKA, global const int *KKB) {
  if (K % 16 != 0 || NT % 128 != 0) return; // workgroup-uniform trap
  const int sg = get_sub_group_id();
  const int ln = get_sub_group_local_id();
  const int m0 = get_group_id(0) * 128 + (sg % 4) * 32;
  const int n0 = get_group_id(1) * 128 + (sg / 4) * 32;
  float8 acc[4][2];
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < 4; i++)
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 2; j++) acc[i][j] = 0.0f;
  for (int kk = 0; kk < K / 16; kk++) {
    int slot_a = -1;
    for (int x = 0; x < nk; x++)
      if (KKA[x] == kk) slot_a = x;
    short8 a[4];
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 4; i++) {
      intel_sub_group_2d_block_read_16b_8r16x1c((global void *)A, K * 2, M, K * 2, (int2)(kk * 16, m0 + 8 * i),
                                                (private ushort *)&a[i]);
      if (slot_a >= 0) { // dump the register exactly as the mad would see it
        const ushort *p = (const ushort *)&a[i];
        long o = ((((long)get_group_id(0) * 16 + sg) * 4 + i) * nk + slot_a) * 128 + ln * 8;
        __attribute__((opencl_unroll_hint))
        for (int t = 0; t < 8; t++) DA[o + t] = p[t];
      }
    }
    int slot_b = -1;
    for (int x = 0; x < nk; x++)
      if (KKB[x] == kk) slot_b = x;
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 2; j++) {
      uint8 bb;
      intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void *)B, NT * 2, K, NT * 2,
                                                           (int2)(n0 + 16 * j, kk * 16), (private uint *)&bb);
      if (slot_b >= 0) {
        const uint *p = (const uint *)&bb;
        long o = ((((long)get_group_id(1) * 16 + sg) * 2 + j) * nk + slot_b) * 128 + ln * 8;
        __attribute__((opencl_unroll_hint))
        for (int t = 0; t < 8; t++) DB[o + t] = p[t];
      }
      __attribute__((opencl_unroll_hint))
      for (int i = 0; i < 4; i++) acc[i][j] = intel_sub_group_f16_f16_matrix_mad_k16(a[i], as_int8(bb), acc[i][j]);
    }
  }
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < 4; i++) {
    if (m0 + 8 * i >= M) continue; // whole rows past M: the 2d block IO contract drops them
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 2; j++) {
      float8 f = acc[i][j];
      intel_sub_group_2d_block_write_32b_8r16x1c(C, NT * 4, M, NT * 4, (int2)(n0 + 16 * j, m0 + 8 * i),
                                                 (private uint *)&f);
    }
  }
}

// eaq_ab_store16: exl3_prompt16 (exl3.cl:302) verbatim — the 16-bit block store into an
// fp16 C — the A/B pair for eaq_gemm32's 32-bit store at the same buffers: if this one
// lands every row and eaq_gemm32 zeroes them, the y/height contract failure is in the
// 32-bit write path, not the reads.
KERNEL(16 * 4 * 4) void eaq_ab_store16(global const half *A, global const half *B, global half *C,
                                       int M, int NT, int K) {
  if (K % 16 != 0 || NT % 128 != 0) return; // workgroup-uniform trap
  const int sg = get_sub_group_id();
  const int m0 = get_group_id(0) * 128 + (sg % 4) * 32;
  const int n0 = get_group_id(1) * 128 + (sg / 4) * 32;
  float8 acc[4][2];
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < 4; i++)
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 2; j++) acc[i][j] = 0.0f;
  for (int kk = 0; kk < K / 16; kk++) {
    short8 a[4];
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 4; i++)
      intel_sub_group_2d_block_read_16b_8r16x1c((global void *)A, K * 2, M, K * 2, (int2)(kk * 16, m0 + 8 * i),
                                                (private ushort *)&a[i]);
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 2; j++) {
      uint8 bb;
      intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void *)B, NT * 2, K, NT * 2,
                                                           (int2)(n0 + 16 * j, kk * 16), (private uint *)&bb);
      __attribute__((opencl_unroll_hint))
      for (int i = 0; i < 4; i++) acc[i][j] = intel_sub_group_f16_f16_matrix_mad_k16(a[i], as_int8(bb), acc[i][j]);
    }
  }
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < 4; i++) {
    if (m0 + 8 * i >= M) continue;
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 2; j++) {
      half8 h = convert_half8_rte(acc[i][j]);
      intel_sub_group_2d_block_write_16b_8r16x1c(C, NT * 2, M, NT * 2, (int2)(n0 + 16 * j, m0 + 8 * i),
                                                 (private ushort *)&h);
    }
  }
}
