// qwen35_norm_nores: RMSNorm without residual, bf16 H (M x 2048) * inv -> XO, Intel Arc Pro B70.
// Port of norm_nores.metal: one work-group per row, 128 work-items, 16 contiguous elements each.
// The Metal simd_sum/threadgroup two-level reduction is replaced by work_group_reduce_add (CL 3.0 core;
// Intel sub-groups may be 16 or 32 wide, so no fixed-width two-level scheme).
#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#define K   2048
#define E   16                       // elements per work-item
#define TPG (K / E)                  // 128 work-items per row

__kernel void qwen35_norm_nores(__global const ushort* H,   // bf16 bits, M x K
                                __global const ushort* Wt,  // bf16 bits, K
                                const float eps,            // 1e-6
                                __global ushort* XO)        // bf16 bits, M x K
{
    const uint t    = get_local_id(0);   // 0..127 within the row's work-group
    const uint m    = get_group_id(1);   // row
    const uint base = m * K + t * E;

    float hv[E];
    float ss = 0.0f;
    #pragma unroll
    for (int i = 0; i < E; i++) {
        float h = intel_convert_as_bfloat16_float(H[base + i]);
        hv[i] = h;
        ss = fma(h, h, ss);              // fp32 fma chain, per work-item
    }

    const float total = work_group_reduce_add(ss);
    const float inv   = rsqrt(total / (float)K + eps);

    #pragma unroll
    for (int i = 0; i < E; i++) {
        float w = intel_convert_as_bfloat16_float(Wt[t * E + i]);
        XO[base + i] = intel_convert_bfloat16_as_ushort(w * (hv[i] * inv)); // single RNE round
    }
}
