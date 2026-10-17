// qwen35_qmm_pair: two int4 projections sharing one activation row, in one launch - the Qwen3.5 GatedDeltaNet
// layer's `in_proj_a` and `in_proj_b` (both N = QW_PAIR_N = 16, K = hidden). Semantics are exactly two calls to
// qmm.cl (one per projection) on the same X, with the same fp32 accumulation and the same single bf16 RNE store,
// so each output is bit-identical to the unfused qmm. Nothing is recomputed and no activation is materialized;
// the only saving is one kernel launch (the two N=16 projections are launch-bound: ~22 KB of traffic each).
//
// Layout: the work-group covers QW_TILE = QW_SETS*8 = 64 columns as qmm.cl does, but only 2*QW_PAIR_N = 32
// columns are live (QW_PAIR_N = 16 per projection). Column set s covers columns 8s..8s+7 of the *pair*: the first
// QW_PAIR_SETS/2 sets (s < 2) belong to A, the remaining sets (s >= 2) to B. Each set's 8 columns come from the
// source triple of its own projection (W/SC/BI) and are stored to that projection's own output buffer.
// Work-group = QW_PAIR_SETS * QW_LANES = 4 * 32 = 128 work-items.
//
// The grid is (1, R): one work-group covers both projections of one row (the 2B/27B pair is 32 columns total,
// well under the 64-column tile). R is a runtime argument.

#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#ifndef QW_PAIR_N
#define QW_PAIR_N 16     // output columns per projection (linear_heads on 2B and 27B alike)
#endif
#ifndef QW_GROUP
#define QW_GROUP 64      // affine quantisation group size
#endif
#ifndef QW_LANES
#define QW_LANES 32      // work-items per column set
#endif
#define QW_PAIR_SETS ((2 * QW_PAIR_N) / 8)   // column sets covering both projections (4 at the default)
#define QW_PAIR_WG (QW_PAIR_SETS * QW_LANES) // work-group size (128 at the default)

inline float bf16_to_f32(ushort b) { return intel_convert_as_bfloat16_float(b); }

__kernel void qwen35_qmm_pair(__global const ushort* X,   // bf16 bits, R x K (shared input)
                              const int R,
                              __global const uint* WA,    // A: N x (K/8) uint32, 8 int4 per word
                              __global const ushort* SCA, // A: bf16 bits, N x (K/QW_GROUP)
                              __global const ushort* BIA, // A: bf16 bits, N x (K/QW_GROUP)
                              __global const uint* WB,    // B: N x (K/8) uint32
                              __global const ushort* SCB, // B: bf16 bits, N x (K/QW_GROUP)
                              __global const ushort* BIB, // B: bf16 bits, N x (K/QW_GROUP)
                              __global ushort* OUTA,      // bf16 bits, R x N
                              __global ushort* OUTB,      // bf16 bits, R x N
                              const int N, const int K) {
    const int G = K / QW_GROUP;

    const int tid = get_local_id(0);
    const int s = tid / QW_LANES;         // column set 0..QW_PAIR_SETS-1
    const int l = tid % QW_LANES;         // lane within the set
    const int r = get_group_id(1);

    if (r >= R) return;

    // Set s < QW_PAIR_SETS/2 covers projection A's columns; the rest cover B's.
    const bool is_a = s < QW_PAIR_SETS / 2;
    const int n0 = (s % (QW_PAIR_SETS / 2)) * 8;   // column within the projection
    if (n0 >= N) return;

    const __global uint* W = is_a ? WA : WB;
    const __global ushort* SC = is_a ? SCA : SCB;
    const __global ushort* BI = is_a ? BIA : BIB;
    __global ushort* OUT = is_a ? OUTA : OUTB;

    const int nl = min(8, N - n0);
    const __global uint4* xv = (const __global uint4*)(X + (size_t)r * K);

    float acc[8];
    acc[0] = acc[1] = acc[2] = acc[3] = acc[4] = acc[5] = acc[6] = acc[7] = 0.0f;

    for (int g = l; g < G; g += QW_LANES) {
        const int xb = 8 * g;
        const int wb = 2 * g;
        float sqx[8];
        sqx[0] = sqx[1] = sqx[2] = sqx[3] = sqx[4] = sqx[5] = sqx[6] = sqx[7] = 0.0f;
        float sx = 0.0f;

        uint4 w0[8], w1[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int nc = min(n0 + i, N - 1);
            const __global uint4* wv = (const __global uint4*)(W + (size_t)nc * (K >> 3));
            w0[i] = wv[wb];
            w1[i] = wv[wb + 1];
        }

        #define QP_STEP(j, comp)                                                     \
            {                                                                        \
                uint4 xj = xv[xb + (j)];                                             \
                float f0 = bf16_to_f32((ushort)xj.x);                                \
                float f1 = bf16_to_f32((ushort)(xj.x >> 16));                        \
                float f2 = bf16_to_f32((ushort)xj.y);                                \
                float f3 = bf16_to_f32((ushort)(xj.y >> 16));                        \
                float f4 = bf16_to_f32((ushort)xj.z);                                \
                float f5 = bf16_to_f32((ushort)(xj.z >> 16));                        \
                float f6 = bf16_to_f32((ushort)xj.w);                                \
                float f7 = bf16_to_f32((ushort)(xj.w >> 16));                        \
                sx += ((f0 + f1) + (f2 + f3)) + ((f4 + f5) + (f6 + f7));             \
                __attribute__((opencl_unroll_hint))                                  \
                for (int i = 0; i < 8; ++i) {                                        \
                    float t = ((float)(((comp) >> 28) & 0xFu)) * f7;                 \
                    t = fma((float)(((comp) >> 24) & 0xFu), f6, t);                  \
                    t = fma((float)(((comp) >> 20) & 0xFu), f5, t);                  \
                    t = fma((float)(((comp) >> 16) & 0xFu), f4, t);                  \
                    t = fma((float)(((comp) >> 12) & 0xFu), f3, t);                  \
                    t = fma((float)(((comp) >> 8) & 0xFu), f2, t);                   \
                    t = fma((float)(((comp) >> 4) & 0xFu), f1, t);                   \
                    t = fma((float)((comp) & 0xFu), f0, t);                          \
                    sqx[i] += t;                                                     \
                }                                                                    \
            }
        QP_STEP(0, w0[i].x)
        QP_STEP(1, w0[i].y)
        QP_STEP(2, w0[i].z)
        QP_STEP(3, w0[i].w)
        QP_STEP(4, w1[i].x)
        QP_STEP(5, w1[i].y)
        QP_STEP(6, w1[i].z)
        QP_STEP(7, w1[i].w)
        #undef QP_STEP

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int nc = min(n0 + i, N - 1);
            const float sc = bf16_to_f32(SC[(size_t)nc * G + g]);
            const float bi = bf16_to_f32(BI[(size_t)nc * G + g]);
            acc[i] = fma(bi, sx, fma(sc, sqx[i], acc[i]));
        }
    }

    __local float red[QW_PAIR_SETS][8][32];
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
