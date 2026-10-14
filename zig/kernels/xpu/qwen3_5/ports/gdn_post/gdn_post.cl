// Port of gdn_post.metal (Qwen3.5-2B GatedDeltaNet gated RMSNorm: OUT = silu(z) * bf16(NW * rmsnorm(y)))
// to OpenCL C for the Intel Arc Pro B70. One 32-wide work-group per (head, row) with 4 elements per lane;
// the cross-lane sum of squares uses work_group_reduce_add over the whole 32-wide work-group, so correctness
// is independent of the sub-group size (Metal's fixed 32-lane SIMD + simd_sum are not portable).
// Semantics per gdn_post.metal: the normed value x is rounded to bf16 (round-to-nearest-even) *before* the
// silu(z) product; the product is rounded to bf16 once at the end.
#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

__kernel void qwen35_gdn_post(__global const ushort* Y,   // bf16 bits, M x 16 x 128 (recurrence output y)
                              __global const ushort* Z,   // bf16 bits, M x 2048 (in_proj_z; head hv = Z[m][hv*128 .. +128])
                              __global const ushort* NW,  // bf16 bits, 128 (norm weight)
                              const float eps,            // 1e-6
                              __global ushort* OUT)       // bf16 bits, M x 16 x 128
{
    const uint NV = 16;
    const uint DV = 128;
    const uint ZS = 2048;
    const uint ZO = 0;

    const uint lane = (uint)get_local_id(0);
    const uint hv = get_global_id(1);
    const uint m = get_global_id(2);

    float yv[4];
    float ss = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        yv[j] = intel_convert_as_bfloat16_float(Y[(m * NV + hv) * DV + lane * 4 + j]);
        ss += yv[j] * yv[j];
    }
    ss = work_group_reduce_add(ss);
    const float inv = rsqrt(ss / (float)DV + eps);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const uint d = lane * 4 + j;
        const float x = intel_convert_as_bfloat16_float(
            intel_convert_bfloat16_as_ushort(intel_convert_as_bfloat16_float(NW[d]) * (yv[j] * inv)));
        const float zf = intel_convert_as_bfloat16_float(Z[m * ZS + ZO + hv * DV + d]);
        OUT[m * NV * DV + hv * DV + d] = intel_convert_bfloat16_as_ushort(zf / (1.0f + exp(-zf)) * x);
    }
}
