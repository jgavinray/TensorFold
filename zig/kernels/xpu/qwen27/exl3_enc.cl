// EXL3 tail-biting Viterbi encoder for one 16x16 mul1 tile (fable spec sections 1-4,
// Layout G): two pinned Viterbi passes over the 65536-state ring trellis with ping-pong
// cost arrays in global memory, one 256-item work-group per tile, back-pointer bytes in
// pass 2, and the ring packed from the traceback states with atomic_or. The oracle is
// core/exl3_format.zig: E(p) is streamEnds, b(p) = E(p) - E(p-1) are the fresh bits
// (closed forms per spec section 1, pinned equal to the differenced oracle by the host
// tests), and the branch metric is (qcv(t) - qt[p])^2 in Q12 with qcv computed from the
// exact integer weight of exl3_dequant128_c (exl3_h.cl:190-201), so costs are
// bit-identical to the host reference's floor(f32(codebookValue(.mul1, t)) * 4096) under
// the periodic-rebase scheme of the fable rebase ruling (eaq-rebase-ruling-fable.md), the
// arithmetic contract exl3-enc/v1: q12 u32-rebased inf=0xFFFFFFFF clamp=4.0. Entries are
// stored relative to the previous step's minimum: rel = m - B(p-1) (B(-1) = 0, both
// passes), v = add_sat(rel, d), and A (u64) accumulates B(0..254), so the reported costs
// are A + the final raw entry and no legal clamped-outlier tile saturates to failed; the
// pruned flags (STAT flags bit0 = pass 1, bit1 = pass 2) mark the steps whose saturation
// pruned a finite branch, where optimality is certified only up to the flag.
// Work-group-uniform arg trap; built like the exl3 family (no -cl-intel-256-GRF-per-thread
// in zig/build/xpu.zig): this kernel is memory bound and wants occupancy, and the rebase
// adds one sub_group_reduce_min and one 16-wide local reduce per step (budget: under 2%
// of T8).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

// floor(f32(codebookValue(.mul1, t)) * 4096) = qcvr(t) >> 6. qcvr's r is the
// half-even-rounded integer whose value is r * 2^-18 (exl3_dequant128_c's form), so the
// arithmetic shift is the floor into Q12 (-905216 >> 6 = -14144 at S = 0; the T2 host
// test ties it to codebookValue for all 65536 states), and r itself is the exact weight
// the WHAT output writes. The byte sum S of t * 0x83DCD12D is realizable only on
// [0, 1005], so Q12 tops out at 13712: the 14128 at S = 1020 that spec section 1's table
// hand-derives is unreachable (update the spec's table and emax note accordingly).
static int qcvr(uint t) {
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

static int qcv12(uint t) { return qcvr(t) >> 6; }

// tile = tile0 + get_group_id(0), k-major over the part; targets gathered through the
// placement closed form; the ring words go to ((global uint *)(TR + tile * 8 * halves))[w],
// the uint view unpack reads (exl3_format.zig:96-98). COST holds count * 2 * 65536 u32
// (ping-pong per slot), BP count * 256 * GMAX bytes (pass 2 only), STAT tiles_total * 32 B
// { cost u64, open_cost u64, pinned u32, status u32: 0 optimal, 1 closed, 2 failed,
//   flags u32: bit0 pruned in pass 1, bit1 pruned in pass 2, reserved u32 }, written
// through this 8-byte-aligned ulong view; the host pre-fills status 2 before launch so an
// unwritten tile reads back failed.
// WHAT (LDLQ driver, spec 4.3): optional fp16 [K, N]; item p writes the exact
// codebookValue(.mul1, sval[p]) = (float)qcvr(sval[p]) * 2^-18 at its placement slot,
// the same closed form the targets gather through. Null WHAT keeps step 1's behavior
// bit-identical (all step-1 tests pass null).
KERNEL(256) void exl3_encode(global const half *WQ, int K, int N, int halves, int tile0,
                             global ushort *TR, global uint *COST, global uchar *BP,
                             global ulong *STAT, global half *WHAT) {
  if (halves < 2 || halves > 16 || K % 16 != 0 || N % 16 != 0) return; // workgroup-uniform arg trap
  const int tile = tile0 + get_group_id(0);
  const int i = get_local_id(0);
  const int halfbit = halves & 1;
  const int ka = (halves - 1) >> 1;
  const int gmax = 1 << (16 - (halfbit ? ka : halves >> 1));
  const int tn16 = N >> 4;
  const int tk = tile / tn16, tn = tile % tn16;

  local short qt[256];    // Q12 targets: item p gathers placement(p) once
  local ushort sval[256]; // traceback states in stream order
  local uint ring[64];    // ring words: 128 * halves bits, 4 * halves words
  local ulong sgm[16];    // pass-1 argmin: one 64-bit key per sub-group
  local uint sgmin[2][16]; // per-step minimum entries, ping-ponged on p & 1: lane 0 of
                           // each sub-group stores sub_group_reduce_min of its writes,
                           // and every item reduces the 16 at the top of the next step
  local ulong acost[2];   // A(p) per pass: item 0's u64 sum of B(0..254)
  local uint lflags;      // pruned flags: bit0 pass 1, bit1 pass 2
  local uint pinned;
  local ulong open_cost;
  local int failed;

  if (i == 0) { acost[0] = 0ul; acost[1] = 0ul; lflags = 0u; } // ordered by the pass-0 start barrier

  { // qt[p] = floor(clamp(W_q[placement(p)], -4, 4) * 4096); the caller guarantees finite
    const int l = i >> 3, j = i & 7;
    const int row = 2 * (l & 3) + (j & 1) + 8 * ((j >> 1) & 1);
    const int col = (l >> 2) + 8 * (j >> 2);
    float f = (float)WQ[(long)(tk * 16 + row) * N + (long)tn * 16 + col];
    f = fmin(fmax(f, -4.0f), 4.0f);
    qt[i] = (short)convert_int_rtn(f * 4096.0f);
  }

  global uint *ca = COST + (long)get_group_id(0) * 131072; // COST[slot][2][65536], ping-pong
  global uint *cb = ca + 65536;
  global uchar *bpv = BP + (long)get_group_id(0) * (256L * gmax); // BP[slot][256][GMAX]

  for (int pass = 0; pass < 2; pass++) {
    int pruned = 0; // this item's private saturation flag for the pass, OR'd once at p = 255
    if (pass == 0) { // pass 1: free start, R[-1][s] = 0 everywhere
      for (int k = 0; k < 256; k++) ca[i + 256 * k] = 0u;
    } else { // pass 2: pinned at e*, R[-1][s] = INF everywhere else
      for (int k = 0; k < 256; k++) ca[i + 256 * k] = (i + 256 * k == (int)pinned) ? 0u : 0xFFFFFFFFu;
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    for (int p = 0; p < 256; p++) {
      const int b = halfbit ? ka + (p & 1) : halves >> 1; // fresh bits b(p), spec section 1
      const int G = 1 << (16 - b);
      const int fan = 1 << b;
      const int sh = 16 - b;
      const int q = (int)qt[p];
      uint base_prev = 0u; // B(p-1), uniform across the work-group: the 16 sub-group minima
      if (p > 0) {         // of the previous step (the end-of-step barrier ordered them)
        base_prev = sgmin[(p - 1) & 1][0];
        for (int sgi = 1; sgi < 16; sgi++) base_prev = min(base_prev, sgmin[(p - 1) & 1][sgi]);
        if (i == 0) acost[pass] += base_prev; // A = sum of B(0..254): added at the top of
                                              // steps 1..255, so B(255) is never applied
      }
      uint wmin = 0xFFFFFFFFu; // min of what this item writes this step
      for (int g = i; g < G; g += 256) { // groups to items, never states: at fixed h the
        uint m = 0xFFFFFFFFu;            // predecessors g | (h << (16-b)) are consecutive
        int hb = 0;                      // strict less: the lowest h wins ties
        for (int h = 0; h < fan; h++) {
          const uint c = ca[g | (h << sh)];
          if (c < m) { m = c; hb = h; }
        }
        if (pass) bpv[(long)p * gmax + g] = (uchar)hb;
        const uint rel = (m == 0xFFFFFFFFu) ? 0xFFFFFFFFu : m - base_prev; // base_prev <=
                                       // m for every finite m (it is the step minimum),
                                       // so the subtraction cannot underflow
        const uint base = (uint)g << b;
        for (int n = 0; n < fan; n++) { // the item's fan writes are contiguous
          const uint t = base | (uint)n;
          const int e = qcv12(t) - q;
          const uint v = (rel == 0xFFFFFFFFu) ? 0xFFFFFFFFu : add_sat(rel, (uint)(e * e));
          if (rel != 0xFFFFFFFFu && v == 0xFFFFFFFFu) pruned = 1; // a finite branch the
                                       // saturating add pruned: optimality becomes
                                       // flag-qualified for this tile (ruling, Q1)
          if (v < wmin) wmin = v;
          cb[t] = v;
        }
      }
      if (p < 255) { // publish the step minimum (always finite: the leader's rel is 0 and
                     // its successors cost d < 2^30) for the next step's base_prev
        const uint sg = sub_group_reduce_min(wmin);
        if (get_sub_group_local_id() == 0) sgmin[p & 1][get_sub_group_id()] = sg;
      }
      if (p == 255 && pruned) atomic_or(&lflags, 1u << pass);
      barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
      global uint *tmp = ca; ca = cb; cb = tmp;
    }
    if (pass == 0) { // e* = lowest cost then lowest state: the min of (cost << 16) | state.
      // R[255] carries the uniform offset A1, so the argmin is the u64 run's; the cost
      // itself is recovered as A1 + the raw entry.
      ulong best = ~0ul;
      for (int k = 0; k < 256; k++) {
        const int s = i + 256 * k;
        const ulong key = ((ulong)ca[s] << 16) | (uint)s;
        if (key < best) best = key;
      }
      best = sub_group_reduce_min(best);
      if (get_sub_group_local_id() == 0) sgm[get_sub_group_id()] = best;
      barrier(CLK_LOCAL_MEM_FENCE);
      if (i == 0) {
        ulong bk = sgm[0];
        for (int sgi = 1; sgi < 16; sgi++) bk = min(bk, sgm[sgi]);
        pinned = (uint)(bk & 0xFFFFu);
        open_cost = acost[0] + (uint)(bk >> 16); // u64: A1 + R1[255][e*]
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  if (i == 0) { // traceback: 256 dependent back-pointer bytes into the local states
    int e = (int)pinned;
    for (int p = 255; p >= 0; p--) {
      const int b = halfbit ? ka + (p & 1) : halves >> 1;
      sval[p] = (ushort)e;
      const int g = e >> b;
      e = g | ((int)bpv[(long)p * gmax + g] << (16 - b));
    }
    failed = e != (int)pinned; // only e* had a finite start cost, so it must close on e*
  }
  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
  if (WHAT != 0) { // the quantized value this tile's trellis carries, at the matrix
    // position item p owns (LDLQ spec 4.3): convert_half_rte of r * 2^-18 is exact for
    // every state (r's magnitude is at most 11 significant bits), so these halves equal
    // codebookValue(.mul1, sval[p]) bit for bit
    const int l = i >> 3, j = i & 7;
    const int row = 2 * (l & 3) + (j & 1) + 8 * ((j >> 1) & 1);
    const int col = (l >> 2) + 8 * (j >> 2);
    WHAT[(long)(tk * 16 + row) * N + (long)tn * 16 + col] = convert_half_rte((float)qcvr((uint)sval[i]) * 0x1p-18f);
  }

  for (int w = i; w < 4 * halves; w += 256) ring[w] = 0u;
  barrier(CLK_LOCAL_MEM_FENCE);
  { // item p packs its fresh bits into [E(p) - b, E(p)), MSB first; E(p) is the oracle's
    // streamEnds closed form (exl3_format.zig:83,87); atomic_or because a position's bits
    // can straddle two ring words (different positions own disjoint bits, so no contention)
    const int b = halfbit ? ka + (i & 1) : halves >> 1;
    const int E = halfbit ? (((i + 1) * (2 * ka + 1) - ((i + 1) & 1)) >> 1) : ((i + 1) * b);
    const int n = (int)sval[i] & ((1 << b) - 1);
    for (int k = 0; k < b; k++) {
      const int pos = E - b + k;
      if ((n >> (b - 1 - k)) & 1) atomic_or(&ring[pos >> 5], 1u << (31 - (pos & 31)));
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (i < 4 * halves) ((global uint *)(TR + (long)tile * 8 * halves))[i] = ring[i];
  if (i == 0) { // the 32-byte TileStat; failed is R2[255][e*] == INF or a traceback that
                // did not close on e*; optimal additionally needs the pass-1 pruned flag
                // clear (pass-2 saturation cannot prune the path itself, only flag it)
    const uint raw = ca[(int)pinned];
    const int is_failed = (raw == 0xFFFFFFFFu) || failed;
    const ulong cost = is_failed ? 0xFFFFFFFFFFFFFFFFull : acost[1] + (ulong)raw;
    uint status;
    if (is_failed) status = 2u;
    else if (cost == open_cost && (lflags & 1u) == 0u) status = 0u;
    else status = 1u;
    global ulong *st = STAT + (long)tile * 4; // little-endian: [2] = status<<32 | pinned,
    st[0] = cost;                             // [3] = flags (reserved stays 0, as prefilled)
    st[1] = open_cost;
    st[2] = ((ulong)status << 32) | (ulong)pinned;
    st[3] = (ulong)lflags;
  }
}
