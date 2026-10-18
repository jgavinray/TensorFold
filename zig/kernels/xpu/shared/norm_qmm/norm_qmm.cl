// qwen35_norm_qmm: the fused (residual + RMSNorm) -> int4 qmm prologue for the Qwen3.5 decode path.
//
// Semantics are exactly norm.cl followed by qmm.cl, with every bf16 rounding point preserved:
//   HO[r][k]  = bf16(H[r][k] + R[r][k])                       (the residual stream, written out)
//   hv[k]     = bf16(HO[r][k])                                (the normed input is bf16-rounded first)
//   inv       = rsqrt(sum_k hv[k]^2 / QW_HIDDEN + eps)
//   XO[r][k]  = bf16(Wt[k] * (hv[k] * inv))                   (single RNE round, as norm.cl)
//   OUT[r][n] = bf16( sum_g SC[n][g]*sum_{k in g} q[n][k]*XO[r][k] + BI[n][g]*sum_{k in g} XO[r][k] )
// The qmm accumulation consumes XO after its bf16 rounding (converted back to fp32), so the fused result is
// bit-identical to running norm then qmm on the rounded XO buffer. XO is never materialized in memory.
//
// Geometry: qmm's grid (ceil(N/QW_TILE) work-groups of QW_WG over R rows). Each work-group computes the whole
// row's RMSNorm redundantly - it needs `inv` before it can form any XO value, and the reduction is cheap next
// to the weight stream. HO is written once per row, by column set 0 (each of its 32 lanes stores QW_ELEMS
// contiguous values, so the group covers the row with QW_TPG stores).
//
// Shapes: QW_HIDDEN (row width), QW_GROUP (affine group size), QW_SETS/QW_LANES (qmm tiling) are compile-time
// defines with the 2B defaults, exactly as norm.cl and qmm.cl. R, N and K stay runtime arguments.

#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#ifndef QW_HIDDEN
#define QW_HIDDEN 2048
#endif
#ifndef QW_GROUP
#define QW_GROUP 64
#endif
#ifndef QW_SETS
#define QW_SETS 8
#endif
#ifndef QW_LANES
#define QW_LANES 32
#endif
#ifndef QW_ELEMS
#define QW_ELEMS 16
#endif
#define QW_TILE (QW_SETS * 8)          // 64 output columns per work-group
#define QW_WG (QW_SETS * QW_LANES)     // 256 work-items per work-group
#define QW_TPG (QW_HIDDEN / QW_ELEMS)  // norm.cl's reduction width (128 at the 2B default)

inline float bf16_to_f32(ushort b) { return intel_convert_as_bfloat16_float(b); }

// HO[r][k] for the k's this work-item owns: the residual sum rounded once to bf16, exactly as norm.cl.
// Every work-item of the group takes part (stride QW_WG), so the whole row is covered once.
inline void store_ho(__global const ushort* H, __global const ushort* R, __global ushort* HO, const int row_base) {
    const int end = row_base + QW_HIDDEN;
    for (int i = row_base + (int)get_local_id(0); i < end; i += QW_WG) {
        const float h = bf16_to_f32(H[i]) + bf16_to_f32(R[i]);
        HO[i] = intel_convert_bfloat16_as_ushort(h);
    }
}

__kernel void qwen35_norm_qmm(__global const ushort* H,   // bf16 bits, R x QW_HIDDEN (residual stream)
                              __global const ushort* R,   // bf16 bits, R x QW_HIDDEN (residual)
                              __global const ushort* Wt,  // bf16 bits, QW_HIDDEN (norm weight)
                              const float eps,            // rms_norm_eps
                              __global ushort* HO,        // bf16 bits, R x QW_HIDDEN (H + R rounded)
                              __global const uint* W,     // N x (K/8) uint32, 8 int4 per word
                              __global const ushort* SC,  // bf16 bits, N x (K/QW_GROUP)
                              __global const ushort* BI,  // bf16 bits, N x (K/QW_GROUP)
                              __global ushort* OUT,       // bf16 bits, R x N
                              const int R_, const int N, const int K) {
    const int G = K / QW_GROUP;

    const int tid = get_local_id(0);
    const int s = tid / QW_LANES;         // column set 0..QW_SETS-1
    const int l = tid % QW_LANES;         // lane within the set
    const int n0 = get_group_id(0) * QW_TILE + s * 8;
    const int r = get_group_id(1);

    if (n0 >= N || r >= R_) return;

    const int row_base = r * QW_HIDDEN;
    // Only the row's first work-group writes HO: every work-group computes the same values, so one writer
    // suffices, and it must be all QW_WG work-items of that group (not just one column set).
    if (get_group_id(0) == 0) store_ho(H, R, HO, row_base);

    // Every work-group needs the row's inv. To keep the bf16 rounding points of norm.cl exactly, the reduction
    // must use norm.cl's partition: QW_TPG = QW_HIDDEN/QW_ELEMS work-items, each summing QW_ELEMS contiguous
    // elements in one fma chain, then one work-group reduction. The remaining work-items contribute 0.
    // (A different partition re-associates the fp32 sum, which can flip a bf16 rounding in XO.)
    float ss = 0.0f;
    if (tid < QW_TPG) {
        const int base = row_base + tid * QW_ELEMS;
        #pragma unroll
        for (int i = 0; i < QW_ELEMS; ++i) {
            const float h = bf16_to_f32(H[base + i]) + bf16_to_f32(R[base + i]);
            const float hv = bf16_to_f32(intel_convert_bfloat16_as_ushort(h));   // the bf16 rounding point
            ss = fma(hv, hv, ss);
        }
    }
    const float inv = rsqrt(work_group_reduce_add(ss) / (float)QW_HIDDEN + eps);

    const int nl = min(8, N - n0);

    float acc[8];
    acc[0] = acc[1] = acc[2] = acc[3] = acc[4] = acc[5] = acc[6] = acc[7] = 0.0f;

    for (int g = l; g < G; g += QW_LANES) {
        const int wb = 2 * g;

        uint4 w0[8], w1[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int nc = min(n0 + i, N - 1);
            const __global uint4* wv = (const __global uint4*)(W + (size_t)nc * (K >> 3));
            w0[i] = wv[wb];
            w1[i] = wv[wb + 1];
        }

        // The group's 64 activations k = QW_GROUP*g .. +63, formed in-register: residual sum, bf16 round,
        // norm, weight, bf16 round - exactly XO[r][k] from norm.cl. f[j] holds k = QW_GROUP*g + 8*j .. +7.
        float f[64];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int k0 = QW_GROUP * g + 8 * j;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int k = k0 + e;
                const float hb = bf16_to_f32(intel_convert_bfloat16_as_ushort(
                    bf16_to_f32(H[row_base + k]) + bf16_to_f32(R[row_base + k])));
                const float wv = bf16_to_f32(Wt[k]);
                f[j * 8 + e] = bf16_to_f32(intel_convert_bfloat16_as_ushort(wv * (hb * inv)));
            }
        }

        #define NQ_STEP(j, comp)                                                     \
            {                                                                        \
                const float f0 = f[(j)*8 + 0], f1 = f[(j)*8 + 1], f2 = f[(j)*8 + 2], f3 = f[(j)*8 + 3]; \
                const float f4 = f[(j)*8 + 4], f5 = f[(j)*8 + 5], f6 = f[(j)*8 + 6], f7 = f[(j)*8 + 7]; \
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
        float sqx[8];
        sqx[0] = sqx[1] = sqx[2] = sqx[3] = sqx[4] = sqx[5] = sqx[6] = sqx[7] = 0.0f;
        float sx = 0.0f;
        // The 8 calls are the whole column loop, exactly as qmm.cl: the macro's own `for (int i...)` walks the
        // 8 columns, and each `comp` argument picks one weight word per column. `sx` accumulates once per call
        // (8 values each, 64 total); `sqx[i]` accumulates the 8 columns' partials.
        NQ_STEP(0, w0[i].x)
        NQ_STEP(1, w0[i].y)
        NQ_STEP(2, w0[i].z)
        NQ_STEP(3, w0[i].w)
        NQ_STEP(4, w1[i].x)
        NQ_STEP(5, w1[i].y)
        NQ_STEP(6, w1[i].z)
        NQ_STEP(7, w1[i].w)
        #undef NQ_STEP

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
