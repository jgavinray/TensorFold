// Qwen3.5 int4 matmul (all projection shapes, runtime N x K) for Intel Arc Pro B70.
//
//   OUT[r][n] = sum_g ( SC[n][g] * sum_{k in g} q[n][k]*x[r][k] + BI[n][g] * sum_{k in g} x[r][k] )
//   q[n][k] = (W[n*(K/8) + k/8] >> 4*(k%8)) & 0xF        (unsigned nibble, group size QW_GROUP)
//
// fp32 accumulation throughout; the dequantized weight is never materialized, and the only bf16
// rounding is the final round-to-nearest-even store. int4 values (0..15) and bf16 activations are
// exact in fp32, so a plain fma chain sits at the bf16 floor.
//
// Intel design (no Apple simdgroup features, no DPAS - measured slower for R=1): a work-group of 256
// work-items covers QW_TILE output columns of one activation row. Columns come in QW_SETS sets of 8;
// each work-item owns one column set and walks whole QW_GROUP-element groups (lane l takes groups l,
// l+QW_LANES, ...). Per group the work-item converts its 8 bf16 activations to fp32 ONCE, computes the
// plain x-sum once, and then runs 8 nibble-by-nibble fma chains - one per column - so the 8 X
// loads and 8 conversions are amortized over 8 columns of W traffic. W is streamed as uint4. The
// QW_LANES work-items of a set reduce their per-column partials through a local-memory tree; lane 0
// rounds each of its 8 columns once to bf16. Tail work-groups clamp W/SC/BI row indices to N-1
// (reads stay in bounds, dead columns are discarded) and guard the store; work-groups with
// r >= R exit before the barrier is hit by any live set.
//
// N, K and R are runtime arguments (the 2B and 27B shape sets both go through this one kernel); the
// affine group size is the only compile-time shape constant, defaulting to the 2B value (64). K must
// be a multiple of QW_GROUP, and K/8, K/16 multiples of 8/16, which holds for 2048 and 6144 (and the
// 27B 5120).
//
// R = 1 is the bandwidth-critical case: the grid is (ceil(N/QW_TILE), R) work-groups, i.e. thousands
// of SIMD16 waves for the big shapes, which hides latency with thread parallelism while W streams
// at full bandwidth with uint4 loads.

#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#ifndef QW_GROUP
#define QW_GROUP 64     // affine quantisation group size (config.json `group_size`)
#endif
#ifndef QW_SETS
#define QW_SETS 8       // column sets per work-group, 8 columns each
#endif
#ifndef QW_LANES
#define QW_LANES 32     // work-items per column set
#endif
#define QW_TILE (QW_SETS * 8)          // 64 output columns per work-group
#define QW_WG (QW_SETS * QW_LANES)     // 256 work-items per work-group

inline float bf16_to_f32(ushort b) { return intel_convert_as_bfloat16_float(b); }

__kernel void qwen35_qmm(__global const ushort* X,   // bf16 bits, R x K
                         const int R,
                         __global const uint* W,     // N x (K/8) uint32, 8 int4 per word
                         __global const ushort* SC,  // bf16 bits, N x (K/QW_GROUP)
                         __global const ushort* BI,  // bf16 bits, N x (K/QW_GROUP)
                         __global ushort* OUT,       // bf16 bits, R x N
                         const int N, const int K) {
    const int G = K / QW_GROUP;   // groups per row (K is a multiple of QW_GROUP)

    const int tid = get_local_id(0);
    const int s = tid / QW_LANES;         // column set 0..QW_SETS-1 (8 columns each)
    const int l = tid % QW_LANES;         // lane within the set's group stride
    const int n0 = get_group_id(0) * QW_TILE + s * 8;
    const int r = get_group_id(1);

    if (n0 >= N || r >= R) return;        // whole set dead: exit (its lanes hit no barrier)

    const int nl = min(8, N - n0);        // live columns in this set
    const __global uint4* xv = (const __global uint4*)(X + (size_t)r * K);

    float acc[8];
    acc[0] = acc[1] = acc[2] = acc[3] = acc[4] = acc[5] = acc[6] = acc[7] = 0.0f;

    // Lane l owns whole groups: g = l, l+QW_LANES, l+2*QW_LANES, ...  K/8 and K/16 are multiples of 8/16, so
    // the uint4 indices stay aligned for every K in the shape set.
    for (int g = l; g < G; g += QW_LANES) {
        const int xb = 8 * g;             // uint4 index of the group's first activation
        const int wb = 2 * g;             // uint4 index of the group's first W word
        float sqx[8];
        sqx[0] = sqx[1] = sqx[2] = sqx[3] = sqx[4] = sqx[5] = sqx[6] = sqx[7] = 0.0f;
        float sx = 0.0f;

        // The group's W words for all 8 columns, fetched once (both loops are unrolled, so all
        // indices are compile-time constants and these stay in registers).
        uint4 w0[8], w1[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int nc = min(n0 + i, N - 1); // clamp: dead columns read row N-1, results discarded
            const __global uint4* wv = (const __global uint4*)(W + (size_t)nc * (K >> 3));
            w0[i] = wv[wb];
            w1[i] = wv[wb + 1];
        }

        // Word-outer: convert the word's 8 bf16 activations once, then apply to all 8 columns.
        // Word j of the group covers k = QW_GROUP*g + 8j .. QW_GROUP*g + 8j + 7, i.e. x uint4 xb + j.
        #define QMM_STEP(j, comp)                                                  \
            {                                                                      \
                uint4 xj = xv[xb + (j)];                                           \
                float f0 = bf16_to_f32((ushort)xj.x);                              \
                float f1 = bf16_to_f32((ushort)(xj.x >> 16));                      \
                float f2 = bf16_to_f32((ushort)xj.y);                              \
                float f3 = bf16_to_f32((ushort)(xj.y >> 16));                      \
                float f4 = bf16_to_f32((ushort)xj.z);                              \
                float f5 = bf16_to_f32((ushort)(xj.z >> 16));                      \
                float f6 = bf16_to_f32((ushort)xj.w);                              \
                float f7 = bf16_to_f32((ushort)(xj.w >> 16));                      \
                sx += ((f0 + f1) + (f2 + f3)) + ((f4 + f5) + (f6 + f7));           \
                __attribute__((opencl_unroll_hint))                             \
                for (int i = 0; i < 8; ++i) {                                      \
                    float t = ((float)(((comp) >> 28) & 0xFu)) * f7;           \
                    t = fma((float)(((comp) >> 24) & 0xFu), f6, t);            \
                    t = fma((float)(((comp) >> 20) & 0xFu), f5, t);            \
                    t = fma((float)(((comp) >> 16) & 0xFu), f4, t);            \
                    t = fma((float)(((comp) >> 12) & 0xFu), f3, t);            \
                    t = fma((float)(((comp) >> 8) & 0xFu), f2, t);             \
                    t = fma((float)(((comp) >> 4) & 0xFu), f1, t);             \
                    t = fma((float)((comp) & 0xFu), f0, t);                    \
                    sqx[i] += t;                                                   \
                }                                                                  \
            }
        QMM_STEP(0, w0[i].x)  // k = QW_GROUP*g +  0.. 7
        QMM_STEP(1, w0[i].y)  // k = QW_GROUP*g +  8..15
        QMM_STEP(2, w0[i].z)  // k = QW_GROUP*g + 16..23
        QMM_STEP(3, w0[i].w)  // k = QW_GROUP*g + 24..31
        QMM_STEP(4, w1[i].x)  // k = QW_GROUP*g + 32..39
        QMM_STEP(5, w1[i].y)  // k = QW_GROUP*g + 40..47
        QMM_STEP(6, w1[i].z)  // k = QW_GROUP*g + 48..55
        QMM_STEP(7, w1[i].w)  // k = QW_GROUP*g + 56..63
        #undef QMM_STEP

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int nc = min(n0 + i, N - 1);
            const float sc = bf16_to_f32(SC[(size_t)nc * G + g]);
            const float bi = bf16_to_f32(BI[(size_t)nc * G + g]);
            acc[i] = fma(bi, sx, fma(sc, sqx[i], acc[i]));
        }
    }

    __local float red[8][8][32];
    #pragma unroll
    for (int i = 0; i < 8; ++i) red[s][i][l] = acc[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int st = 16; st > 0; st >>= 1) {
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            if (l < st) red[s][i][l] += red[s][i][l + st];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0) {
        #pragma unroll
        for (int i = 0; i < nl; ++i)
            OUT[(size_t)r * N + n0 + i] = intel_convert_bfloat16_as_ushort(red[s][i][0]);
    }
}


// ---------------------------------------------------------------------------------------------------------------
// qwen35_qmm_q8: the decode matmul (every projection outside XMX prefill), W4A8x2 on dp4a.
//
// At the B70's 100 W cap decode is energy-bound: speed = 100 W / energy per weight byte. qwen35_qmm's per-nibble
// unpack + fp32 convert + FMA costs as much energy as its weight traffic (517 mJ/GB: 193 GB/s, where its own load
// pattern alone sustains 404). Here the multiply-adds are 4-wide int8 dot products (cl_khr_integer_dot_product):
// 3 ALU ops unpack a word's 8 nibbles into two 4 x uint8 vectors, 4 dp4a apply them to the activations - 356 mJ/GB,
// 281 GB/s, rms error 0.50x the bf16 floor (exp/kernels/power, exp/kernels/qmm_dp4a).
//
// qwen35_qmm_split, per (row r, group g of QW_GROUP inputs; one 64-item work-group each): s = max|x| / 127 (1 for an all-zero group),
// x/s = hi + lo/254 with hi, lo int8 - about 2^-16 of the group max, far below the bf16 floor. XQ holds, per (r, g),
// 16 words of hi then 16 of lo, in the order the unpack produces: within each 8-input word x0 x2 x4 x6 | x1 x3 x5 x7
// (w & 0x0F0F0F0F holds nibbles 0,2,4,6; (w >> 4) & 0x0F0F0F0F holds 1,3,5,7). XS holds (s, exact fp32 sum of the
// group's bf16 inputs) per (r, g).
// qwen35_qmm_q8: qwen35_qmm's work decomposition and load pattern; per group and column the integer sums are exact and
//   OUT += SC * (s * (sum q*hi + sum q*lo / 254)) + BI * sum x.
// Up to four projections of the same input in one launch (work-groups numbered segment by segment in argument
// order, so callers list narrow projections first: alone, a 48-output projection is one work-group at latency;
// dispatched first it finishes under its wide siblings' traffic). An absent segment has N = 0. Each row is computed
// alone, so a row's bits do not depend on R (decode and verify windows agree byte for byte).

// One 64-item work-group per (group, row): item k owns input k; max and sum are work-group reductions.
__attribute__((reqd_work_group_size(QW_GROUP, 1, 1)))
__kernel void qwen35_qmm_split(__global const ushort* X, const int R, __global uint* XQ, __global float* XS,
                                const int K) {
    const int g = get_group_id(0), r = get_group_id(1), k = get_local_id(0), G = K / QW_GROUP;
    const float v = bf16_to_f32(X[(size_t)r * K + (size_t)g * QW_GROUP + k]);
    const float mx = work_group_reduce_max(fabs(v));
    const float sum = work_group_reduce_add(v);
    const float s = mx > 0.0f ? mx / 127.0f : 1.0f;
    const float t = v / s;
    const float h = rint(t);
    __local uchar hb[QW_GROUP], lb[QW_GROUP];
    const int i = k & 7, slot = (k & ~7) + ((i & 1) ? 4 + (i >> 1) : (i >> 1));
    hb[slot] = (uchar)((int)h & 0xFF);
    lb[slot] = (uchar)((int)rint((t - h) * 254.0f) & 0xFF);
    barrier(CLK_LOCAL_MEM_FENCE);
    __global uint* xq = XQ + ((size_t)r * G + g) * (QW_GROUP / 2);
    if (k < QW_GROUP / 4) xq[k] = ((__local const uint*)hb)[k];
    else if (k < QW_GROUP / 2) xq[k] = ((__local const uint*)lb)[k - QW_GROUP / 4];
    if (k == 0) {
        XS[((size_t)r * G + g) * 2] = s;
        XS[((size_t)r * G + g) * 2 + 1] = sum;
    }
}

inline void q8_tile(__global const uint* XQ, __global const float* XS, const int R, __global const uint* W,
                    __global const ushort* SC, __global const ushort* BI, __global ushort* OUT, const int N,
                    const int K, const int bx, __local float (*red)[8][32]) {
    const int G = K / QW_GROUP;
    const int tid = get_local_id(0);
    const int s = tid / QW_LANES;         // column set 0..QW_SETS-1 (8 columns each)
    const int l = tid % QW_LANES;         // lane within the set's group stride
    const int n0 = bx * QW_TILE + s * 8;
    const int r = get_group_id(1);
    if (n0 >= N || r >= R) return;        // whole set dead: exit (its lanes hit no barrier)
    const int nl = min(8, N - n0);

    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
    for (int g = l; g < G; g += QW_LANES) {
        const __global uint4* xq = (const __global uint4*)(XQ + ((size_t)r * G + g) * (QW_GROUP / 2));
        uint hv[16], lv[16];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint4 h4 = xq[j], l4 = xq[4 + j];
            hv[4 * j] = h4.x; hv[4 * j + 1] = h4.y; hv[4 * j + 2] = h4.z; hv[4 * j + 3] = h4.w;
            lv[4 * j] = l4.x; lv[4 * j + 1] = l4.y; lv[4 * j + 2] = l4.z; lv[4 * j + 3] = l4.w;
        }
        const float2 ss = vload2((size_t)r * G + g, XS);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int nc = min(n0 + i, N - 1); // clamp: dead columns read row N-1, results discarded
            const __global uint4* wv = (const __global uint4*)(W + (size_t)nc * (K >> 3));
            const uint4 w0 = wv[2 * g], w1 = wv[2 * g + 1];
            const uint wd[8] = { w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w };
            int h = 0, lo = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint e = wd[j] & 0x0F0F0F0Fu, o = (wd[j] >> 4) & 0x0F0F0F0Fu;
                h = dot_acc_sat_4x8packed_su_int(hv[2 * j], e, h);
                h = dot_acc_sat_4x8packed_su_int(hv[2 * j + 1], o, h);
                lo = dot_acc_sat_4x8packed_su_int(lv[2 * j], e, lo);
                lo = dot_acc_sat_4x8packed_su_int(lv[2 * j + 1], o, lo);
            }
            // |h| <= 64 * 15 * 127 < 2^24: exact in fp32
            const float sq = fma((float)lo, 1.0f / 254.0f, (float)h) * ss.x;
            const float sc = bf16_to_f32(SC[(size_t)nc * G + g]);
            const float bi = bf16_to_f32(BI[(size_t)nc * G + g]);
            acc[i] = fma(bi, ss.y, fma(sc, sq, acc[i]));
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) red[s][i][l] = acc[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int st = QW_LANES / 2; st > 0; st >>= 1) {
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            if (l < st) red[s][i][l] += red[s][i][l + st];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0) {
        for (int i = 0; i < nl; ++i)
            OUT[(size_t)r * N + n0 + i] = intel_convert_bfloat16_as_ushort(red[s][i][0]);
    }
}

__kernel void qwen35_qmm_q8(__global const uint* XQ, __global const float* XS, const int R, const int K,
                            __global const uint* W0, __global const ushort* SC0, __global const ushort* BI0, __global ushort* OUT0, const int N0,
                            __global const uint* W1, __global const ushort* SC1, __global const ushort* BI1, __global ushort* OUT1, const int N1,
                            __global const uint* W2, __global const ushort* SC2, __global const ushort* BI2, __global ushort* OUT2, const int N2,
                            __global const uint* W3, __global const ushort* SC3, __global const ushort* BI3, __global ushort* OUT3, const int N3) {
    __local float red[8][8][32];
    const int t0 = (N0 + QW_TILE - 1) / QW_TILE, t1 = (N1 + QW_TILE - 1) / QW_TILE, t2 = (N2 + QW_TILE - 1) / QW_TILE;
    int bx = get_group_id(0);
    // Pick the segment, then run one tile body (four inlined bodies spill registers).
    __global const uint* W = W0;
    __global const ushort* SC = SC0;
    __global const ushort* BI = BI0;
    __global ushort* OUT = OUT0;
    int N = N0;
    if (bx >= t0) {
        bx -= t0; W = W1; SC = SC1; BI = BI1; OUT = OUT1; N = N1;
        if (bx >= t1) {
            bx -= t1; W = W2; SC = SC2; BI = BI2; OUT = OUT2; N = N2;
            if (bx >= t2) { bx -= t2; W = W3; SC = SC3; BI = BI3; OUT = OUT3; N = N3; }
        }
    }
    q8_tile(XQ, XS, R, W, SC, BI, OUT, N, K, bx, red);
}

// ---------------------------------------------------------------------------------------------------------------
// qwen35_qmm_xmx: the same affine int4 matmul on the XMX matrix engines, for prefill chunks (R >> 1).
//
// The scalar kernel is compute-bound at ~1.7 TFLOPS (int4 unpack + fp32 fma); the B70 sustains 121 TFLOPS bf16 on
// `dpas` at the 100 W cap (exp/capability/BUDGET-27B.md). Per quantization group g (QW_GROUP inputs), column n:
//     OUT += SC[n][g] * sum_k q x  +  BI[n][g] * sum_k x
// - the scalar kernel's formula. Weights enter dpas as bf16(128 + q) = 0x4300 | q (exact; no float conversion), so
// dpas yields sum (128+q) x and sum q x = that - 128 * sum x. sum_k x per (row, group) comes from
// qwen35_qmm_xsum (one pass over X). Products are exact in fp32; only summation order differs from the scalar kernel
// (not byte-identical: prefill only - decode and verify windows stay scalar, so the startup qualify is unaffected).
// Tiles: a 16-lane sub-group owns 16 columns (lane = column) x QX_ROWS rows (QX_ROWS/8 dpas row blocks).
// dpas layout (bf16, M8 K16 N16, SIMD16) verified on the card: exp/kernels/qmm_xmx/layout_check.py.
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#ifndef QX_ROWS
#define QX_ROWS 16
#endif
#define QX_RB (QX_ROWS / 8)
inline int qx_pair(uint w, int j) {             // nibbles 2j, 2j+1 of w as two bf16 (128 + q), low half first
    const uint t = (w >> (8 * j)) & 0xFFu;
    return (int)((t & 0xFu) | ((t >> 4) << 16) | 0x43004300u);
}
// XS[r][g] = sum of X[r][k] over group g (fp32), the bias term's input. Grid: (G, R).
__kernel void qwen35_qmm_xsum(__global const ushort* X, const int R, __global float* XS, const int K) {
    const int g = get_global_id(0), r = get_global_id(1), G = K / QW_GROUP;
    if (g >= G || r >= R) return;
    const __global ushort* x = X + (size_t)r * K + (size_t)g * QW_GROUP;
    float s = 0.0f;
    for (int k = 0; k < QW_GROUP; ++k) s += intel_convert_as_bfloat16_float(x[k]);
    XS[(size_t)r * G + g] = s;
}
#ifndef QX_NT
#define QX_NT 2            // 16-column tiles per sub-group: one A load feeds QX_NT dpas
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void qwen35_qmm_xmx(__global const ushort* X,   // bf16 bits, R x K
                             const int R,
                             __global const uint* W,     // N x (K/8) uint32, 8 int4 per word
                             __global const ushort* SC,  // bf16 bits, N x (K/QW_GROUP)
                             __global const ushort* BI,  // bf16 bits, N x (K/QW_GROUP)
                             __global ushort* OUT,       // bf16 bits, R x N
                             const int N, const int K,
                             __global const float* XS) { // R x G, from qwen35_qmm_xsum
    const int G = K / QW_GROUP;
    const int l = get_sub_group_local_id();
    const int nbase = (get_global_id(0) / 16) * (16 * QX_NT);   // first column of this sub-group
    const int r0 = get_global_id(1) * QX_ROWS;
    int nc[QX_NT];
    #pragma unroll
    for (int t = 0; t < QX_NT; ++t) nc[t] = min(nbase + 16 * t + l, N - 1);

    float8 acc[QX_NT][QX_RB];
    #pragma unroll
    for (int t = 0; t < QX_NT; ++t)
        #pragma unroll
        for (int rb = 0; rb < QX_RB; ++rb) acc[t][rb] = (float8)(0.0f);

    for (int g = 0; g < G; ++g) {
        uint8 wg[QX_NT];
        #pragma unroll
        for (int t = 0; t < QX_NT; ++t) wg[t] = vload8(g, W + (size_t)nc[t] * (K >> 3));
        float8 cq[QX_NT][QX_RB];
        #pragma unroll
        for (int t = 0; t < QX_NT; ++t)
            #pragma unroll
            for (int rb = 0; rb < QX_RB; ++rb) cq[t][rb] = (float8)(0.0f);
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            int8 b[QX_NT];
            #pragma unroll
            for (int t = 0; t < QX_NT; ++t) {
                const uint w0 = ((uint*)&wg[t])[2 * kk], w1 = ((uint*)&wg[t])[2 * kk + 1];
                b[t] = (int8)(qx_pair(w0, 0), qx_pair(w0, 1), qx_pair(w0, 2), qx_pair(w0, 3),
                              qx_pair(w1, 0), qx_pair(w1, 1), qx_pair(w1, 2), qx_pair(w1, 3));
            }
            const int k0 = g * QW_GROUP + kk * 16;
            #pragma unroll
            for (int rb = 0; rb < QX_RB; ++rb) {
                short8 a;
                #pragma unroll
                for (int m = 0; m < 8; ++m) {
                    const int r = min(r0 + rb * 8 + m, R - 1);
                    ((ushort*)&a)[m] = intel_sub_group_block_read_us(X + (size_t)r * K + k0);
                }
                #pragma unroll
                for (int t = 0; t < QX_NT; ++t) cq[t][rb] = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b[t], cq[t][rb]);
            }
        }
        #pragma unroll
        for (int rb = 0; rb < QX_RB; ++rb) {
            float8 sx;
            #pragma unroll
            for (int m = 0; m < 8; ++m) ((float*)&sx)[m] = XS[(size_t)min(r0 + rb * 8 + m, R - 1) * G + g];
            #pragma unroll
            for (int t = 0; t < QX_NT; ++t) {
                const float sc = intel_convert_as_bfloat16_float(SC[(size_t)nc[t] * G + g]);
                const float bi = intel_convert_as_bfloat16_float(BI[(size_t)nc[t] * G + g]);
                acc[t][rb] = fma(bi, sx, fma(sc, cq[t][rb] - 128.0f * sx, acc[t][rb]));
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < QX_NT; ++t) {
        const int n = nbase + 16 * t + l;
        if (n < N) {
            #pragma unroll
            for (int rb = 0; rb < QX_RB; ++rb)
                #pragma unroll
                for (int m = 0; m < 8; ++m) {
                    const int r = r0 + rb * 8 + m;
                    if (r < R) OUT[(size_t)r * N + n] = intel_convert_bfloat16_as_ushort(((float*)&acc[t][rb])[m]);
                }
        }
    }
}
