// EXL3 forward kernels on Intel Xe2 (mul1 codebook, docs/recipes/exl3.md): the fp16-output
// twins of the frozen exl3.cl decode GEMVs, plus the fused-column-base dequant. exl3.cl
// stays the fp32-probe file (gate-verified, do not edit): every GEMM-output consumer in
// the engine is fp16 (a.p, a.r, a.logits — argmax_part reads a.logits as half), so the
// forward launches these instead — same mul1 tables, same window extraction and
// placement as exl3_decode2/_t1/exl3_dequant128, with the named deltas only (PR 4 wiring
// contract section 5e): half Y, an NT row-pitch argument that separates the output stride
// from the part's N (fused projections stack parts on N and the caller shifts Y to the
// part's column base n0), and the dequant's n0 column base into the shared wscr slab.
// Own file so the compile skips -cl-intel-256-GRF-per-thread, like exl3.cl: the decode
// GEMVs are small-register latency kernels and the dequant's staged block fits ~88 registers.

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

// fp16-output exl3_decode2 (exl3.cl:62), verbatim except: half Y, the NT pitch and the arg
// trap. Y is the caller's fp16 [rows][NT] (advanced to the part's column base); the slice
// rows follow the seed kernel's layout ((ks*T + t) rows apart), so the forward launches
// with kslices = 1 — the shipped layout puts slice sums in disjoint rows and only the
// probe reduces them host-side. bits must be whole (the loader admits 2..8; the probe uses
// 4/3/2; half widths would need a reader-side path), T <= 8, and kslices need not divide
// K/16 — the last slice absorbs the remainder.
KERNEL(16) void exl3_decode2_h(global const ushort *TR, global const half *X, global half *Y, int K, int N, int T, int bits,
                               int kslices, int NT) {
  if (bits < 1 || bits > 8 || K % 16 != 0 || kslices < 1) return; // workgroup-uniform arg trap
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
      // The exact integer half-even form lives in exl3_dequant128_c below (and in
      // exl3_format.zig's integerFp16Weight test). The prompt/decode weight-form
      // asymmetry is the open operator decision recorded in the PR 3 design note: the
      // prompt path dequantizes exact, the decode GEMVs stay fast, until PR 5 matches
      // them. Switch if the fidelity gates demand format weights.
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
      const long base = ((long)ks * T + t) * NT + nt * 16 + lane / 2;
      Y[base] = convert_half_rte(a0);
      Y[base + 8] = convert_half_rte(a1);
    }
  }
}

// T == 1 fast path of exl3_decode2_h: no token loop, scalar accumulators and one
// activation load per weight. Same math and output as exl3_decode2_h with T = 1 (Y is that
// one row); T must be 1 and bits whole, like the general kernel, and the last K slice
// absorbs a non-divisor kslices remainder.
KERNEL(16) void exl3_decode2_t1_h(global const ushort *TR, global const half *X, global half *Y, int K, int N, int T, int bits,
                                   int kslices, int NT) {
  if (T != 1 || bits < 1 || bits > 8 || K % 16 != 0 || kslices < 1) return; // workgroup-uniform arg trap
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
      // See exl3_dequant128_c's exact integer half-even form; the prompt/decode weight-
      // form asymmetry is the open PR 5 operator decision. Switch if the fidelity gates
      // demand format weights.
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
    const long base = (long)ks * NT + nt * 16 + lane / 2;
    Y[base] = convert_half_rte(a0);
    Y[base + 8] = convert_half_rte(a1);
  }
}

// ---- EXL3 prompt path, fused variant: exact dequant of one [128,128] weight block into
// the part's column range of the shared fp16 scratch ----
// exl3_dequant128 (exl3.cl:213) verbatim except the trailing n0, the workgroup-uniform
// column base: each part dequantizes its own trellis into columns [n0, n0 + N) of one
// wscr slab pitched at the fused NT (the A1 design — the rotation is baked here, so
// exl3_prompt16 runs ONCE over the whole NT and needs no suh/svh arguments). n0 shifts the
// W stores only: the trellis reads, the SVH reads and phase B's straddle check stay
// part-relative (grid y = ceil(N/128) covers the part's blocks, the phase-C store adds n0
// to the column). Requirements as the frozen kernel: K a multiple of 128, N a multiple of
// 16, NT a multiple of 128 and >= the fused Proj.n, and n0 a multiple of 16; tiles past
// N/16 decode as zero and SVH reads are guarded for n >= N (the runner zero-fills the
// slab's pad columns once at allocation, Acts.init, so the fused tail between the last
// part and NT is deterministic).
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

KERNEL(256) void exl3_dequant128_c(global const ushort *TR, global const half *SUH, global const half *SVH,
                                   global half *W, int K, int N, int NT, int bits, int n0) {
  if (K % 128 != 0 || NT % 128 != 0 || bits < 1 || bits > 8 || n0 % 16 != 0) return; // workgroup-uniform trap
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
  // phase C: H_K along k on the columns, fused with suh[k]*svh[n] and the store into the
  // part's column range (n0 shifts the store column; the SUH/SVH reads stay part-relative)
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
      W[(long)(bk * 128 + lane * 8 + j) * NT + n0 + bn * 128 + n] =
          convert_half_rte(v[j] * 0.08838834764831845f * (float)SUH[bk * 128 + lane * 8 + j] * sv);
  }
}

// ---- EXL3 decode path, fused variant: the two scale/rotation passes the raw trellis
// GEMVs cannot do themselves ----
// The format's weight is W_eff[k][n] = suh[k] * svh[n] * (H_K W_q H_N)[k][n] (host oracle
// core/exl3_format.zig:200-212 dequantize: rotate along k, *suh, rotate along n, *svh; the
// prompt dequant bakes all four into the slab in its phases B and C). exl3_decode2_h and
// exl3_decode2_t1_h read the raw trellis words and raw X, so the decode row has to meet
// them in the middle: with H symmetric and the two Hadamards on independent axes,
//   Y[n] = svh[n] * (H_N Z)[n],  Z[b] = sum_a Xt[a] * W_q[a][b],  Xt = H_K (X .* suh),
// which is exl3_xform_h before the GEMV (per part, since suh is per part) and
// exl3_yscale_h after it (the n-axis butterfly over the part's own columns, then svh).
// Same rot128_lane, same 1/sqrt(128) = 0.08838834764831845f as the dequant's phases, so
// prompt and decode agree on the weight form up to the codebook rounding the GEMVs do in
// fp32 (the PR 5 asymmetry above). Both are one sub-group workgroups: a lane holds the 8
// consecutive values rot128_lane expects (element 8*lane+j of the 128-vector).
// Requirements: X, XT fp16 [M][K] with K a multiple of 128; Y the GEMV's fp16 output,
// already advanced to the part's column base and pitched at NT; SVH the part's [N]; the
// admitted parts have N a multiple of 128, so every n block is whole (no straddle).
KERNEL(16) void exl3_xform_h(global const half *X, global const half *SUH, global half *XT, int K, int M) {
  if (K % 128 != 0 || M < 1) return; // workgroup-uniform arg trap
  const int kb = get_group_id(0), row = get_group_id(1), lane = get_sub_group_local_id();
  const int k0 = kb * 128;
  float v[8];
  __attribute__((opencl_unroll_hint))
  for (int j = 0; j < 8; j++) {
    const int k = k0 + lane * 8 + j;
    v[j] = (float)X[(long)row * K + k] * (float)SUH[k];
  }
  rot128_lane(v, lane);
  __attribute__((opencl_unroll_hint))
  for (int j = 0; j < 8; j++)
    XT[(long)row * K + k0 + lane * 8 + j] = convert_half_rte(v[j] * 0.08838834764831845f);
}

KERNEL(16) void exl3_yscale_h(global half *Y, global const half *SVH, int N, int NT, int M) {
  if (N % 128 != 0 || NT % 128 != 0 || M < 1) return; // workgroup-uniform arg trap
  const int nb = get_group_id(0), row = get_group_id(1), lane = get_sub_group_local_id();
  const int n0 = nb * 128;
  float v[8];
  __attribute__((opencl_unroll_hint))
  for (int j = 0; j < 8; j++) v[j] = (float)Y[(long)row * NT + n0 + lane * 8 + j];
  rot128_lane(v, lane);
  __attribute__((opencl_unroll_hint))
  for (int j = 0; j < 8; j++) {
    const int n = n0 + lane * 8 + j;
    Y[(long)row * NT + n] = convert_half_rte(v[j] * 0.08838834764831845f * (float)SVH[n]);
  }
}
