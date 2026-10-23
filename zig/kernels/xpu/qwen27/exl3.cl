// EXL3 trellis-decode GEMVs on Intel Xe2 (mul1 codebook, docs/recipes/exl3.md).
// Own file so the compile skips -cl-intel-256-GRF-per-thread: the decode GEMVs are
// small-register latency kernels; exl3_prompt16's 32x32 DPAS tile fits ~88 registers,
// so it does not want the flag either (ocloc: no spills at 128 GRF).

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

// EXL3 probe (mul1 codebook): X * W_q over K-tile slices, W_q decoded on the fly from TR.
KERNEL(16) void exl3_gemv(global const ushort *TR, global const half *X, global float *Y, int K, int N, int T, int bits,
                          int kslices) {
  // the tile is read once per subgroup and distributed by shuffle, like the activations of its 16 K rows
  const int nt = get_group_id(0), ks = get_group_id(1), lane = get_sub_group_local_id();
  const int per = K / 16 / kslices, kt0 = ks * per, R = 256 * bits, nw = 8 * bits;
  const float c1 = (float)as_half((ushort)0x1EEE), c0 = (float)as_half((ushort)0xC931);
  float acc0[8], acc1[8];
  for (int t = 0; t < 8; t++) acc0[t] = acc1[t] = 0.0f;
  for (int kt = kt0; kt < kt0 + per; kt++) {
    global const uint *tw = (global const uint *)(TR + ((long)kt * (N / 16) + nt) * 16 * bits);
    const uint r0 = tw[lane], r1 = lane + 16 < nw ? tw[lane + 16] : 0u;
    float xr[8];
    for (int t = 0; t < 8; t++) xr[t] = t < T ? (float)X[(long)t * K + kt * 16 + lane] : 0.0f;
    __attribute__((opencl_unroll_hint))
    for (int v = 0; v < 16; v++) {
      const int p = lane * 16 + v;
      int a = (p + 1) * bits - 16;
      a += a < 0 ? R : 0;
      const int wi = a >> 5, wj = wi + 1 == nw ? 0 : wi + 1, sh = a & 31;
      const uint w0a = sub_group_shuffle(r0, wi & 15), w0b = sub_group_shuffle(r1, wi & 15);
      const uint w1a = sub_group_shuffle(r0, wj & 15), w1b = sub_group_shuffle(r1, wj & 15);
      const uint w0 = wi < 16 ? w0a : w0b, w1 = wj < 16 ? w1a : w1b;
      const uint hi = sh ? (w0 << sh) | (w1 >> (32 - sh)) : w0; // the 32 bits starting at bit a
      const uint s = hi >> 16;
      const uint xx = s * 0x83DCD12Du;
      const uint bs = (xx & 0x00FF00FFu) + ((xx >> 8) & 0x00FF00FFu);
      const float w = (float)(1024u + (bs & 0xFFFFu) + (bs >> 16)) * c1 + c0;
      const int l = p >> 3, j = p & 7, row = 2 * (l & 3) + (j & 1) + 8 * ((j >> 1) & 1);
      for (int t = 0; t < 8; t++) {
        if (t >= T) break;
        const float xv = sub_group_shuffle(xr[t], row);
        if (j >> 2) acc1[t] += xv * w; else acc0[t] += xv * w;
      }
    }
  }
  for (int t = 0; t < T && t < 8; t++) {
    const float a0 = acc0[t] + sub_group_shuffle_xor(acc0[t], 1), a1 = acc1[t] + sub_group_shuffle_xor(acc1[t], 1);
    if ((lane & 1) == 0) {
      const long base = ((long)ks * T + t) * N + nt * 16 + lane / 2;
      Y[base] = a0;
      Y[base + 8] = a1;
    }
  }
}

// EXL3 decode v2: lane-owned contiguous bit windows, no shuffles, no dynamic indexing.
// Same signature and output as exl3_gemv; value p's 16-bit state window ends at (p+1)*bits
// in the circular ring of 256*bits bits (core/exl3_format.zig tileStates), MSB-first in
// little-endian uint pairs. Lane L maps positions p = 16*L + v (v < 16); its windows span
// [(16L+1)*bits - 16, (16L+16)*bits) mod R — 15*bits + 16 bits. Each window is two ring
// loads and two shifts; the acc arrays are indexed only by unrolled literals (the only
// dynamic index is the global tw[uu] load address, an ordinary load) — ocloc still
// reports ~1.2 KB of register spills for this kernel, see the build warnings.
KERNEL(16) void exl3_decode2(global const ushort *TR, global const half *X, global float *Y, int K, int N, int T, int bits,
                             int kslices) {
  // bits must be whole (the probe uses 4/3/2; half widths would need a reader-side path)
  // and T <= 8; kslices need not divide K/16 — the last slice absorbs the remainder.
  const int nt = get_group_id(0), ks = get_group_id(1), lane = get_sub_group_local_id();
  const int per = K / 16 / kslices, kt0 = ks * per, R = 256 * bits, nw = 8 * bits;
  const int kt_end = ks == kslices - 1 ? K / 16 : kt0 + per;
  float acc0[8], acc1[8], acc0b[8], acc1b[8];
  __attribute__((opencl_unroll_hint))
  for (int t = 0; t < 8; t++) acc0[t] = acc1[t] = acc0b[t] = acc1b[t] = 0.0f;
  const int lo = (16 * lane + 1) * bits - 16;          // the lane's first needed bit, may be negative
  for (int kt = kt0; kt < kt_end; kt++) {
    global const uint *tw = (global const uint *)(TR + ((long)kt * (N / 16) + nt) * 16 * bits);
    __attribute__((opencl_unroll_hint))
    for (int v = 0; v < 16; v++) {
      const int a_v = lo + v * bits;                   // the window's start bit, unwrapped
      const int u = (a_v >= 0 ? a_v : a_v - 31) / 32;  // floor division; -1 at most, since lo >= -16 + bits
      const int uu = u < 0 ? u + nw : u;               // the ring word holding the window's first bit
      const uint w0 = tw[uu], w1 = tw[uu + 1 == nw ? 0 : uu + 1];
      const int s = a_v - u * 32;
      const uint w32 = s ? (w0 << s | w1 >> (32 - s)) : w0;
      const uint state = w32 >> 16;
      const uint xx = state * 0x83DCD12Du;
      const uint bs = (xx & 0x00FF00FFu) + ((xx >> 8) & 0x00FF00FFu);
      // The fp32 product-add below is exact (h <= 2044 is 11 bits, 0x1EEE has an 11-bit
      // significand, the sum fits 24 bits), but it is NOT the format's weight:
      // codebookValue rounds the exact value to fp16 — up to half an fp16 ulp per weight.
      // Measured on the 27B probe (gate18/19): the deviation reaches 0.17 abs, ~1e-4 of
      // the output scale. The integer half-even form — ((1024+S)*1774 - 2721792)*2^-18
      // rounded to 11 significant bits — reproduces codebookValue exactly on-card
      // (host-reference max 0.0022, reorder-level) at ~2x this kernel's time; it is
      // recorded in artifacts/2026-10-22/pr2-final.diff and exl3_format.zig's
      // integerFp16Weight test. Switch if the fidelity gates demand format weights.
      const float wgt = (float)(1024u + (bs & 0xFFFFu) + (bs >> 16)) * (float)as_half((ushort)0x1EEE) + (float)as_half((ushort)0xC931);
      const int p = 16 * lane + v;
      const int l = p >> 3, j = p & 7, row = 2 * (l & 3) + (j & 1) + 8 * ((j >> 1) & 1);
      __attribute__((opencl_unroll_hint))
      for (int t = 0; t < 8; t++) {
        if (t >= T) continue;
        const float xv = X[(long)t * K + kt * 16 + row];
        if (j >> 2) {
          if (v & 1) acc1b[t] += xv * wgt; else acc1[t] += xv * wgt;
        } else {
          if (v & 1) acc0b[t] += xv * wgt; else acc0[t] += xv * wgt;
        }
      }
    }
  }
  __attribute__((opencl_unroll_hint))
  for (int t = 0; t < 8; t++) {
    if (t >= T) continue;
    const float a0 = acc0[t] + acc0b[t] + sub_group_shuffle_xor(acc0[t] + acc0b[t], 1);
    const float a1 = acc1[t] + acc1b[t] + sub_group_shuffle_xor(acc1[t] + acc1b[t], 1);
    if ((lane & 1) == 0) {
      const long base = ((long)ks * T + t) * N + nt * 16 + lane / 2;
      Y[base] = a0;
      Y[base + 8] = a1;
    }
  }
}

// T == 1 fast path of exl3_decode2: no token loop, so no predicated dead bodies and no
// acc arrays — scalar accumulators and one activation load per weight. Same math and
// output as exl3_decode2 with T = 1 (Y is that one row); T must be 1 and bits whole,
// like the general kernel, and the last K slice absorbs a non-divisor kslices remainder.
KERNEL(16) void exl3_decode2_t1(global const ushort *TR, global const half *X, global float *Y, int K, int N, int T, int bits,
                                int kslices) {
  if (T != 1) return;
  const int nt = get_group_id(0), ks = get_group_id(1), lane = get_sub_group_local_id();
  const int per = K / 16 / kslices, kt0 = ks * per, R = 256 * bits, nw = 8 * bits;
  const int kt_end = ks == kslices - 1 ? K / 16 : kt0 + per;
  float acc0 = 0.0f, acc1 = 0.0f, b0 = 0.0f, b1 = 0.0f;
  const int lo = (16 * lane + 1) * bits - 16;
  for (int kt = kt0; kt < kt_end; kt++) {
    global const uint *tw = (global const uint *)(TR + ((long)kt * (N / 16) + nt) * 16 * bits);
    __attribute__((opencl_unroll_hint))
    for (int v = 0; v < 16; v++) {
      const int a_v = lo + v * bits;                   // the window's start bit, unwrapped
      const int av = a_v + R;                          // shifted into [R - 16, 2R)
      const int w0i = av >> 5;                         // a_v >= 0: the ring word plus nw; a_v < 0: nw - 1
      const int uu = w0i >= nw ? w0i - nw : w0i;       // the ring word holding the window's first bit
      const uint w0 = tw[uu], w1 = tw[uu + 1 == nw ? 0 : uu + 1];
      const int s = av & 31;
      const uint w32 = s ? (w0 << s | w1 >> (32 - s)) : w0;
      const uint state = w32 >> 16;
      const uint xx = state * 0x83DCD12Du;
      const uint bs = (xx & 0x00FF00FFu) + ((xx >> 8) & 0x00FF00FFu);
      // The fp32 product-add below is exact (h <= 2044 is 11 bits, 0x1EEE has an 11-bit
      // significand, the sum fits 24 bits), but it is NOT the format's weight:
      // codebookValue rounds the exact value to fp16 — up to half an fp16 ulp per weight.
      // Measured on the 27B probe (gate18/19): the deviation reaches 0.17 abs, ~1e-4 of
      // the output scale. The integer half-even form — ((1024+S)*1774 - 2721792)*2^-18
      // rounded to 11 significant bits — reproduces codebookValue exactly on-card
      // (host-reference max 0.0022, reorder-level) at ~2x this kernel's time; it is
      // recorded in artifacts/2026-10-22/pr2-final.diff and exl3_format.zig's
      // integerFp16Weight test. Switch if the fidelity gates demand format weights.
      const float wgt = (float)(1024u + (bs & 0xFFFFu) + (bs >> 16)) * (float)as_half((ushort)0x1EEE) + (float)as_half((ushort)0xC931);
      const int p = 16 * lane + v;
      const int l = p >> 3, j = p & 7, row = 2 * (l & 3) + (j & 1) + 8 * ((j >> 1) & 1);
      const float xv = (float)X[kt * 16 + row];
      if (j >> 2) {
        if (v & 1) b1 += xv * wgt; else acc1 += xv * wgt;
      } else {
        if (v & 1) b0 += xv * wgt; else acc0 += xv * wgt;
      }
    }
  }
  const float a0 = acc0 + b0 + sub_group_shuffle_xor(acc0 + b0, 1);
  const float a1 = acc1 + b1 + sub_group_shuffle_xor(acc1 + b1, 1);
  if ((lane & 1) == 0) {
    const long base = (long)ks * N + nt * 16 + lane / 2;
    Y[base] = a0;
    Y[base + 8] = a1;
  }
}

// ---- EXL3 prompt path: exact dequant of one [128,128] weight block to fp16 scratch ----
// One workgroup (256 threads, 16 subgroups) owns k rows [128*gid0, +128) and n columns
// [128*gid1, +128). Phase A decodes the block's 64 trellis tiles with the PR 2 window
// extraction and the EXACT integer half-even codebookValue (the variant C form verified
// on-card in pr2 gate19; affordable here because the dequant runs once per layer+chunk —
// see the decode2 comment). Phase B rotates n and phase C rotates k in fp32 via the
// rot128 butterfly (same convention as glue.cl's rot128_reg: out[i] = sum_j
// (-1)^parity(i&j) x[j] / sqrt(128), lane l holding elements 8l+j); each scale follows
// its axis' Hadamard, matching exl3_format.dequantize. The intermediate after phase B
// rounds to fp16 once (measured: total dequant-vs-fp64-oracle error 4-5e-6 at the
// gate's scale magnitudes; with unit scales it is ~half an fp16 ulp of the phase-B
// value). SLM stride 130 keeps the phase-B row reads conflict-free; phase C's column
// reads are 4-way bank-conflicted and its stores are 2-byte strided — a transpose or
// 2d block write would lift the measured ~90 GB/s effective rate (perf, not
// correctness). Requirements: K a multiple of 128 (grid x = K/128 truncates otherwise),
// N a multiple of 16 (tiles), NT a multiple of 128 and >= N; tiles past N/16 decode as
// zero and SVH reads are guarded for n >= N.
static float rot128_lane(float v[8], int lane) {
  __attribute__((opencl_unroll_hint))
  for (int m = 1; m < 8; m <<= 1)
    for (int j = 0; j < 8; j++)
      if (!(j & m)) {
        const float a = v[j], b = v[j | m];
        v[j] = a + b;
        v[j | m] = a - b;
      }
  __attribute__((opencl_unroll_hint))
  for (int m = 1; m < 16; m <<= 1)
    for (int j = 0; j < 8; j++) {
      const float o = sub_group_shuffle_xor(v[j], m);
      v[j] = (lane & m) ? o - v[j] : v[j] + o;
    }
  return 0.0f; // values transformed in place
}

KERNEL(256) void exl3_dequant128(global const ushort *TR, global const half *SUH, global const half *SVH,
                                 global half *W, int K, int N, int NT, int bits) {
  if (K % 128 != 0 || NT % 128 != 0 || bits < 1 || bits > 8) return; // workgroup-uniform trap
  const int bk = get_group_id(0), bn = get_group_id(1);
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int nw = 8 * bits;
  local half blk[128][130];
  for (int i = get_local_id(0); i < 128 * 130; i += 256) blk[i / 130][i % 130] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  // phase A: decode; subgroup sg takes kt' = sg/2 and nt' = (sg&1)*4 + [0,4)
  const int kt0 = (sg >> 1) * 16, nt0 = (sg & 1) * 64;
  const int lo = (16 * lane + 1) * bits - 16;
  __attribute__((opencl_unroll_hint))
  for (int ti = 0; ti < 4; ti++) {
    const int tn_g = bn * 8 + (sg & 1) * 4 + ti;
    if (tn_g < N / 16) {
      global const uint *tw = (global const uint *)(TR + ((long)(bk * 8 + (sg >> 1)) * (N / 16) + tn_g) * 16 * bits);
      const int nt_col = nt0 + ti * 16;
      __attribute__((opencl_unroll_hint))
      for (int v = 0; v < 16; v++) {
        const int a_v = lo + v * bits;                   // the window's start bit, unwrapped
        const int u = (a_v >= 0 ? a_v : a_v - 31) / 32;  // floor division
        const int uu = u < 0 ? u + nw : u;               // the ring word holding the window's first bit
        const uint w0 = tw[uu], w1 = tw[uu + 1 == nw ? 0 : uu + 1];
        const int s = a_v - u * 32;
        const uint w32 = s ? (w0 << s | w1 >> (32 - s)) : w0;
        const uint state = w32 >> 16;
        const uint xx = state * 0x83DCD12Du;
        const uint bs = (xx & 0x00FF00FFu) + ((xx >> 8) & 0x00FF00FFu);
        // exact integer half-even rounding of ((1024+S)*1774 - 2721792)*2^-18 to 11 bits
        const int n2 = (int)(1024u + (bs & 0xFFFFu) + (bs >> 16)) * 1774 - 2721792;
        const int mag = n2 < 0 ? -n2 : n2;
        const int b = 31 - clz(mag);
        const int q = b > 10 ? 1 << (b - 10) : 1;
        const int rem = mag & (q - 1);
        int r = mag - rem;
        if (rem > (q >> 1) || (rem == (q >> 1) && (r & q) != 0)) r += q;
        const float wgt = (n2 < 0 ? -(float)r : (float)r) * 0x1.0p-18f;
        const int p = 16 * lane + v;
        const int l = p >> 3, j = p & 7;
        const int row = 2 * (l & 3) + (j & 1) + 8 * ((j >> 1) & 1);
        blk[kt0 + row][nt_col + (l >> 2) + 8 * (j >> 2)] = convert_half_rte(wgt);
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // phase B: H_N along n; subgroup sg rotates rows k = sg*8 .. sg*8+8. A block that
  // straddles N is left UNROTATED, matching the host oracle (rotate walks whole 128
  // blocks only, exl3_format.rotate); its svh scales still apply in phase C.
  if ((bn + 1) * 128 <= N) {
    __attribute__((opencl_unroll_hint))
    for (int kr = 0; kr < 8; kr++) {
    const int k = sg * 8 + kr;
    float v[8];
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 8; j++) v[j] = (float)blk[k][lane * 8 + j];
    rot128_lane(v, lane);
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 8; j++) blk[k][lane * 8 + j] = convert_half_rte(v[j] * 0.08838834764831845f);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // phase C: H_K along k on the columns, fused with suh[k]*svh[n] and the store
  __attribute__((opencl_unroll_hint))
  for (int nc = 0; nc < 8; nc++) {
    const int n = sg * 8 + nc;
    const int gn = bn * 128 + n;
    const float sv = gn < N ? (float)SVH[gn] : 0.0f;
    float v[8];
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 8; j++) v[j] = (float)blk[lane * 8 + j][n];
    rot128_lane(v, lane);
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 8; j++)
      W[(long)(bk * 128 + lane * 8 + j) * NT + bn * 128 + n] =
          convert_half_rte(v[j] * 0.08838834764831845f * (float)SUH[bk * 128 + lane * 8 + j] * sv);
  }
}

// ---- EXL3 prompt GEMM: plain fp16 B from the dequant scratch, fp32 acc, fp16 out ----
// Subgroup tile 32x32, workgroup 4x4 subgroups = 256 threads over a 128x128 output tile
// (an earlier 32x64-subgroup shape spilled ~9.3 KB of registers and ran at 1/4 the
// throughput; 32x32 holds acc+a+bb in ~88 registers). The B operand is the [K][NT] fp16
// scratch read with the transform block load (the gemm.cl BIAS_TILE pattern) — per lane,
// its n column's next 16 k values as packed half2. No scales and no bias: both are baked
// into the dequantized weights. A is fp16 row-major [M][K]; out-of-bounds rows read 0 and
// out-of-bounds stores are dropped (the 2d block IO contract the engine already relies
// on). C is written NT-pitched (like B); the caller allocates [M][NT] or handles the
// pitch. Grid: x = ceil(M/128), y = NT/128; K must be a multiple of 16 (whole widths).
KERNEL(16 * 4 * 4) void exl3_prompt16(global const half *A, global const half *B, global half *C,
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
