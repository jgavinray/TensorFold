// Port of gdn_post.metal (Qwen3.5/Qwen 3.8 GatedDeltaNet gated RMSNorm:
// OUT = silu(z) * bf16(NW * rmsnorm(y))) to OpenCL C for the Intel Arc Pro B70.
// One 32-wide work-group per (head, row) with 4 elements per lane; the cross-lane sum of squares uses
// work_group_reduce_add over the whole 32-wide work-group, so correctness is independent of the
// sub-group size (Metal's fixed 32-lane SIMD + simd_sum are not portable).
// Semantics per gdn_post.metal: the normed value x is rounded to bf16 (round-to-nearest-even) *before* the
// silu(z) product; the product is rounded to bf16 once at the end.
//
// Shapes are compile-time defines; defaults are the Qwen3.5-2B geometry, so the proven 2B gate applies
// to this file unchanged. The 27B build overrides QW_NV (48 value heads) and QW_ZS (in_proj_z row
// stride = QW_HIDDEN = 5120); QW_DV (128) is shared. QW_NV * QW_DV = 32 lanes x 4 elements must hold.
#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#ifndef QW_NV
#define QW_NV 16
#endif
#ifndef QW_DV
#define QW_DV 128
#endif
#ifndef QW_ZS
#define QW_ZS 2048      // in_proj_z row stride (= hidden size); ZO is the head's offset within the row
#endif
#ifndef QW_ZO
#define QW_ZO 0
#endif

__kernel void qwen35_gdn_post(__global const ushort* Y,   // bf16 bits, M x QW_NV x QW_DV (recurrence output y)
                              __global const ushort* Z,   // bf16 bits, M x QW_ZS (in_proj_z; head hv = Z[m][hv*QW_DV .. +QW_DV])
                              __global const ushort* NW,  // bf16 bits, QW_DV (norm weight)
                              const float eps,            // 1e-6
                              __global ushort* OUT)       // bf16 bits, M x QW_NV x QW_DV
{
    const uint lane = (uint)get_local_id(0);
    const uint hv = get_global_id(1);
    const uint m = get_global_id(2);

    float yv[4];
    float ss = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        yv[j] = intel_convert_as_bfloat16_float(Y[(m * QW_NV + hv) * QW_DV + lane * 4 + j]);
        ss += yv[j] * yv[j];
    }
    ss = work_group_reduce_add(ss);
    const float inv = rsqrt(ss / (float)QW_DV + eps);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const uint d = lane * 4 + j;
        const float x = intel_convert_as_bfloat16_float(
            intel_convert_bfloat16_as_ushort(intel_convert_as_bfloat16_float(NW[d]) * (yv[j] * inv)));
        const float zf = intel_convert_as_bfloat16_float(Z[m * QW_ZS + QW_ZO + hv * QW_DV + d]);
        OUT[m * QW_NV * QW_DV + hv * QW_DV + d] = intel_convert_bfloat16_as_ushort(zf / (1.0f + exp(-zf)) * x);
    }
}
