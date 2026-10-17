// qwen35_mlp_act_down: the fused SwiGLU -> int4 down-projection for the Qwen3.5 decode path.
//
// Semantics are exactly mlp_act.cl followed by qmm.cl, with every bf16 rounding point preserved:
//   ACT[r][k]  = bf16( silu(bf16(GATE[r][k])) * bf16(UP[r][k]) ),  silu(g) = g / (1 + exp(-g))
//   OUT[r][n]  = bf16( sum_g SC[n][g]*sum_{k in g} q[n][k]*ACT[r][k] + BI[n][g]*sum_{k in g} ACT[r][k] )
// ACT is consumed after its bf16 rounding (converted back to fp32), so the fused result is bit-identical to
// running mlp_act then qmm on the rounded ACT buffer. ACT is never materialized in memory: the down
// projection reads GATE and UP and forms each activation in-register, which removes one full write+read
// round trip of the intermediate (QW_INTERMEDIATE bf16 per row) and one kernel launch.
//
// Geometry: qmm's grid (ceil(N/QW_TILE) work-groups of QW_WG over R rows), with K = QW_INTERMEDIATE.
// Each work-item walks whole QW_GROUP-element groups; per group it forms the group's QW_GROUP activations
// once (silu of GATE times UP, bf16-rounded) and applies them to its 8 columns, exactly as qmm.cl amortizes
// its X loads over 8 columns.
//
// Shapes: QW_INTERMEDIATE (K, the activation width), QW_GROUP (affine group size), QW_SETS/QW_LANES are
// compile-time defines with the 2B defaults. R, N and K stay runtime arguments (K must equal QW_INTERMEDIATE).

#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#ifndef QW_INTERMEDIATE
#define QW_INTERMEDIATE 6144
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
#define QW_TILE (QW_SETS * 8)          // 64 output columns per work-group
#define QW_WG (QW_SETS * QW_LANES)     // 256 work-items per work-group

inline float bf16_to_f32(ushort b) { return intel_convert_as_bfloat16_float(b); }

// ACT[k] = bf16(silu(bf16(GATE[k])) * bf16(UP[k])), the mlp_act.cl rounding point, as an fp32 value.
inline float act_at(__global const ushort* GATE, __global const ushort* UP, const int k) {
    const float g = bf16_to_f32(GATE[k]);
    const float u = bf16_to_f32(UP[k]);
    const float y = g / (1.0f + exp(-g)) * u;
    return bf16_to_f32(intel_convert_bfloat16_as_ushort(y));
}

__kernel void qwen35_mlp_act_down(__global const ushort* GATE, // bf16 bits, R x QW_INTERMEDIATE (gate_proj output)
                                  __global const ushort* UP,   // bf16 bits, R x QW_INTERMEDIATE (up_proj output)
                                  __global const uint* W,      // N x (K/8) uint32, 8 int4 per word
                                  __global const ushort* SC,   // bf16 bits, N x (K/QW_GROUP)
                                  __global const ushort* BI,   // bf16 bits, N x (K/QW_GROUP)
                                  __global ushort* OUT,        // bf16 bits, R x N
                                  const int R_, const int N, const int K) {
    const int G = K / QW_GROUP;

    const int tid = get_local_id(0);
    const int s = tid / QW_LANES;         // column set 0..QW_SETS-1
    const int l = tid % QW_LANES;         // lane within the set
    const int n0 = get_group_id(0) * QW_TILE + s * 8;
    const int r = get_group_id(1);

    if (n0 >= N || r >= R_) return;

    const int row = r * K;
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

        // Word j of the group covers k = QW_GROUP*g + 8j .. +7. Form those 8 activations in registers per
        // step (no 64-wide local array): silu of GATE times UP, bf16-rounded, exactly mlp_act.cl's point.
        #define AD_STEP(j, comp)                                                     \
            {                                                                        \
                const int k0 = row + QW_GROUP * g + 8 * (j);                         \
                const float f0 = act_at(GATE, UP, k0 + 0), f1 = act_at(GATE, UP, k0 + 1); \
                const float f2 = act_at(GATE, UP, k0 + 2), f3 = act_at(GATE, UP, k0 + 3); \
                const float f4 = act_at(GATE, UP, k0 + 4), f5 = act_at(GATE, UP, k0 + 5); \
                const float f6 = act_at(GATE, UP, k0 + 6), f7 = act_at(GATE, UP, k0 + 7); \
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
        AD_STEP(0, w0[i].x)
        AD_STEP(1, w0[i].y)
        AD_STEP(2, w0[i].z)
        AD_STEP(3, w0[i].w)
        AD_STEP(4, w1[i].x)
        AD_STEP(5, w1[i].y)
        AD_STEP(6, w1[i].z)
        AD_STEP(7, w1[i].w)
        #undef AD_STEP

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
