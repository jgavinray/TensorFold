// qwen35_norm_nores: RMSNorm without residual, bf16 H (M x QW_HIDDEN) * inv -> XO, Intel Arc Pro B70.
// Port of norm_nores.metal: one work-group per row, QW_TPG work-items, QW_ELEMS contiguous elements each.
// The Metal simd_sum/threadgroup two-level reduction is replaced by work_group_reduce_add (CL 3.0 core;
// Intel sub-groups may be 16 or 32 wide, so no fixed-width two-level scheme).
//
// Shapes are compile-time defines; defaults are the Qwen3.5-2B geometry, so the proven 2B gate applies
// to this file unchanged. QW_HIDDEN = row width, QW_ELEMS = per-work-item slice, QW_TPG = work-group size.
#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#ifndef QW_HIDDEN
#define QW_HIDDEN 2048
#endif
#ifndef QW_ELEMS
#define QW_ELEMS 16                  // elements per work-item
#endif
#define QW_TPG (QW_HIDDEN / QW_ELEMS) // 128 work-items per row at the 2B default

__kernel void qwen35_norm_nores(__global const ushort* H,   // bf16 bits, M x QW_HIDDEN
                                __global const ushort* Wt,  // bf16 bits, QW_HIDDEN
                                const float eps,            // 1e-6
                                __global ushort* XO)        // bf16 bits, M x QW_HIDDEN
{
    const uint t    = get_local_id(0);   // 0..QW_TPG-1 within the row's work-group
    const uint m    = get_group_id(1);   // row
    const uint base = m * QW_HIDDEN + t * QW_ELEMS;

    float hv[QW_ELEMS];
    float ss = 0.0f;
    #pragma unroll
    for (int i = 0; i < QW_ELEMS; i++) {
        float h = intel_convert_as_bfloat16_float(H[base + i]);
        hv[i] = h;
        ss = fma(h, h, ss);              // fp32 fma chain, per work-item
    }

    const float total = work_group_reduce_add(ss);
    const float inv   = rsqrt(total / (float)QW_HIDDEN + eps);

    #pragma unroll
    for (int i = 0; i < QW_ELEMS; i++) {
        float w = intel_convert_as_bfloat16_float(Wt[t * QW_ELEMS + i]);
        XO[base + i] = intel_convert_bfloat16_as_ushort(w * (hv[i] * inv)); // single RNE round
    }
}
