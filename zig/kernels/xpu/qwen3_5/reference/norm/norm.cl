// Port of metal/norm.metal (Qwen3.5 fused residual + RMSNorm) to OpenCL C for Intel Arc Pro B70.
// bf16 is carried as ushort bit patterns via cl_intel_bfloat16_conversions
// (round-to-nearest-even, same rounding as the Metal `bfloat()` cast).

#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#define K 2048
#define E 16
#define TPG (K / E) // 128 work-items per row

__kernel void qwen35_norm(__global const ushort* H,  // bf16 bits, M x 2048
                          __global const ushort* R,  // bf16 bits, M x 2048 (residual)
                          __global const ushort* Wt, // bf16 bits, 2048
                          const float eps,
                          __global ushort* HO,       // bf16 bits, M x 2048: H + R rounded to bf16
                          __global ushort* XO)       // bf16 bits, M x 2048: RMSNorm(HO) * Wt
{
    const uint t = (uint)get_local_id(0);  // 0..127, lane within the row's work-group
    const uint m = (uint)get_group_id(1);  // one work-group per row
    const int base = (int)m * K + (int)t * E;

    float hv[E];
    float ss = 0.0f;
    for (int i = 0; i < E; i++) {
        float h = intel_convert_as_bfloat16_float(H[base + i]);
        h += intel_convert_as_bfloat16_float(R[base + i]);
        ushort hb = intel_convert_bfloat16_as_ushort(h);
        HO[base + i] = hb;
        hv[i] = intel_convert_as_bfloat16_float(hb);
        ss = fma(hv[i], hv[i], ss);
    }

    // Correct for any sub-group size the compiler picks (16 or 32 on Intel, not Metal's 32).
    const float total = work_group_reduce_add(ss);
    const float inv = rsqrt(total / (float)K + eps);

    for (int i = 0; i < E; i++) {
        float w = intel_convert_as_bfloat16_float(Wt[(int)t * E + i]);
        XO[base + i] = intel_convert_bfloat16_as_ushort(w * (hv[i] * inv));
    }
}
