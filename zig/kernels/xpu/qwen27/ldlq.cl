// Block LDLQ device loop (fable LDLQ spec sections 4.3, 4.6): the three kernels of the
// per-block feedback loop - strip_f16 (fp32 target strip to the encoder's fp16 WQ, counting
// |T| > 4 clamps), strip_diff (E32 = fp32(Q16) - W32 rows of the block), ldlq_update (the
// rank-16 feedback T[0:16j] -= (L[16j:16j+16, 0:16j])^T * E, fp32 FMA, i ascending so the
// sum is deterministic) - plus ldlq_eta (the PartReport eta_sq device reduction and the
// optional eta strips) and strip_decode (the standalone raw strip decoder: the oracle for
// the encoder's WHAT output and the proxy dequant that exl3_dequant128_c cannot be, since
// it rotates and scales). All states/values follow core/exl3_format.zig: streamEnds and
// tileStates windows (MSB-first within the little-endian uint32 view of the int16 ring),
// the tilePlacement closed form, and exl3_enc.cl's qcv12 integer weight (the r before
// >>6, value r * 2^-18, exact in fp16). The encoder launch itself is xpu/exl3_encode.zig's
// exl3_encode kernel; xpu/ldlq.zig sequences these around it. Built like the exl3 family
// (no -cl-intel-256-GRF-per-thread in zig/build/xpu.zig): memory-bound streams.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

// the r of exl3_enc.cl's qcv12, before the Q12 shift: the fp16-exact integer whose value
// is r * 2^-18 (exl3_dequant128_c's form, exl3_h.cl:190-201; the T2 host test ties it to
// codebookValue for all 65536 states).
static int qcvInt(uint t) {
  const uint xx = t * 0x83DCD12Du;
  const uint bs = (xx & 0x00FF00FFu) + ((xx >> 8) & 0x00FF00FFu);
  const int n2 = (int)(1024u + (bs & 0xFFFFu) + (bs >> 16)) * 1774 - 2721792;
  const int mag = n2 < 0 ? -n2 : n2;
  const int b = 31 - clz(mag); // (1024+S)*1774 never equals 2721792: mag > 0 for every state
  const int q = b > 10 ? 1 << (b - 10) : 1;
  const int rem = mag & (q - 1);
  int r = mag - rem;
  if (rem > (q >> 1) || (rem == (q >> 1) && (r & q) != 0)) r += q;
  if (n2 < 0) r = -r;
  return r;
}

// S[16][N] = (half) T[16j .. 16j+16][N]; also counts |T| > 4 into a global counter
// (the clamp monitor of section 4.3: the encoder clamps its targets to ±4, and this
// fraction above 1e-4 says to investigate g before trusting the pack).
// grid: x = ceil(N/256), y = 16 (the strip's rows).
KERNEL(256) void strip_f16(global const float *T, global half *S, int j, int N, global uint *CLAMPED) {
  if (j < 0 || N % 16 != 0) return; // workgroup-uniform arg trap
  const int n = get_global_id(0);
  if (n >= N) return;
  const int row = get_group_id(1);
  const float t = T[(long)(16 * j + row) * N + n];
  S[(long)row * N + n] = (half)t;
  if (fabs(t) > 4.0f) atomic_add(CLAMPED, 1u);
}

// E[16][N] = (float) Q[16][N] - W[16j .. 16j+16][N]: the error against the ORIGINAL
// weight, not against the block's target (section 4.1).
// grid: x = ceil(N/256), y = 16.
KERNEL(256) void strip_diff(global const half *Q, global const float *W, global float *E, int j, int N) {
  if (j < 0 || N % 16 != 0) return; // workgroup-uniform arg trap
  const int n = get_global_id(0);
  if (n >= N) return;
  const int row = get_group_id(1);
  E[(long)row * N + n] = convert_float(Q[(long)row * N + n]) - W[(long)(16 * j + row) * N + n];
}

// T[r][n] -= sum_{i<16} L[(16j + i) * K + r] * E[i][n]  for r < rows (= 16j), n < N.
// fp32 FMA, i ascending: deterministic. Only strictly-lower block columns of L are read.
// grid: x = ceil(rows / 16), y = ceil(N / 256); a work-item owns one n and 16 consecutive
// r; the L strip arrives through local memory (one 16x16 tile per group, 256 loads).
KERNEL(256) void ldlq_update(global float *T, global const float *L, global const float *E, int j, int rows, int N, int K) {
  if (j <= 0 || rows % 16 != 0 || rows > 16 * j || N % 16 != 0 || K % 16 != 0) return; // uniform trap
  const int n = get_group_id(1) * 256 + get_local_id(0);
  const int r0 = get_group_id(0) * 16;
  local float Ls[16][16]; // Ls[t][i] = L[(16j + i)][r0 + t]
  { // cooperative preload: barrier() needs every item, so this runs before the n<N
    // tail return — items beyond N still load their 256th of the tile, then retire
    const int idx = get_local_id(0);
    const int i = idx & 15, t = idx >> 4;
    Ls[t][i] = L[(long)(16 * j + i) * K + r0 + t];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (n >= N) return;
  #pragma unroll
  for (int t = 0; t < 16; t++) {
    const int r = r0 + t;
    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) acc = fma(Ls[t][i], E[(long)i * N + n], acc); // i ascending; OpenCL fma(), no fmaf in CL C
    T[(long)r * N + n] -= acc;
  }
}

// The PartReport eta_sq reduction (section 4.4): per block j, eta[r][n] =
// (float)Q[row][n] - (float)S[row][n] with the block's rows r = 16j + row. When ETA is
// non-null the strips are kept (T-LDLQ-3 compares them against the host reference); the
// group's sum of squares lands at PART[get_group_id(0)] in a fixed local order, so the
// host fp64 total over PART[0 .. kb * ngroups) is deterministic.
// grid: x = ceil(N / 256).
KERNEL(256) void ldlq_eta(global const half *Q, global const half *S, global float *ETA, global float *PART, int j, int N) {
  if (j < 0 || N % 16 != 0) return; // workgroup-uniform arg trap
  const int n = get_global_id(0);
  local float loc[256];
  float acc = 0.0f;
  if (n < N) {
    #pragma unroll
    for (int row = 0; row < 16; row++) {
      const float e = convert_float(Q[(long)row * N + n]) - convert_float(S[(long)row * N + n]);
      if (ETA) ETA[(long)(16 * j + row) * N + n] = e;
      acc += e * e;
    }
  }
  loc[get_local_id(0)] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (get_local_id(0) == 0) {
    float tot = 0.0f;
    for (int t = 0; t < 256; t++) tot += loc[t]; // fixed order: deterministic partial
    PART[get_group_id(0)] = tot;
  }
}

// The raw strip decoder (section 4.3's oracle for WHAT, and the no-rotation dequant of
// section 4.6): tile (tk = get_group_id(1), tn = get_group_id(0)) of the part's trellis
// int16 [tiles_k, tiles_n, 8 * halves] decodes to codebook values OUT[(tk*16+row)*N +
// tn*16+col] = (float)r * 2^-18 in fp16, states read through the tileStates window
// (16 ring bits ending at E(p), MSB first; exl3_format.zig:103-114) and placement
// (exl3_format.zig:119-128). grid: x = tiles_n, y = tiles_k.
KERNEL(256) void strip_decode(global const ushort *TR, int halves, global half *OUT, int N, int tiles_n) {
  if (halves < 2 || halves > 16 || N % 16 != 0 || tiles_n <= 0) return; // uniform trap
  const int p = get_local_id(0);
  const int tn = get_group_id(0), tk = get_group_id(1);
  const int halfbit = halves & 1;
  const int ka = (halves - 1) >> 1;
  const uint ring = 128u * (uint)halves;
  const uint E = (uint)(halfbit ? (((p + 1) * (2 * ka + 1) - ((p + 1) & 1)) >> 1) : ((p + 1) * halves / 2));
  const uint start = (E + ring - 16u) % ring;
  global const uint *ringw = (global const uint *)(TR + ((long)tk * tiles_n + tn) * 8 * halves);
  uint s = 0u;
  for (int b = 0; b < 16; b++) { // the tileStates window, MSB first
    const uint pos = (start + (uint)b) % ring;
    s = (s << 1) | ((ringw[pos >> 5] >> (31 - (pos & 31))) & 1u);
  }
  const int l = p >> 3, jj = p & 7; // the tilePlacement closed form (exl3_enc.cl's gather)
  const int row = 2 * (l & 3) + (jj & 1) + 8 * ((jj >> 1) & 1);
  const int col = (l >> 2) + 8 * (jj >> 2);
  OUT[(long)(tk * 16 + row) * N + (long)tn * 16 + col] = (half)((float)qcvInt(s) * 0x1.0p-18f);
}
