// Port of metal/norm.metal (Qwen3.5 fused residual + RMSNorm) to OpenCL C for Intel Arc Pro B70.
// bf16 is carried as ushort bit patterns via cl_intel_bfloat16_conversions
// (round-to-nearest-even, same rounding as the Metal `bfloat()` cast).
//
// Shapes are compile-time defines so one source serves both Qwen3.5-2B (defaults) and Qwen 3.8 27B
// (`ocloc -D QW_HIDDEN=5120`). QW_HIDDEN is the row width; QW_ELEMS is the per-work-item slice, and
// QW_TPG = QW_HIDDEN / QW_ELEMS is the work-group size (one work-group per row). The defaults are the
// 2B geometry, so the proven 2B gate applies to this file unchanged.

#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#ifndef QW_HIDDEN
#define QW_HIDDEN 2048
#endif
#ifndef QW_ELEMS
#define QW_ELEMS 16
#endif
#define QW_TPG (QW_HIDDEN / QW_ELEMS) // 128 work-items per row at the 2B default

__kernel void qwen35_norm(__global const ushort* H,  // bf16 bits, M x QW_HIDDEN
                          __global const ushort* R,  // bf16 bits, M x QW_HIDDEN (residual)
                          __global const ushort* Wt, // bf16 bits, QW_HIDDEN
                          const float eps,
                          __global ushort* HO,       // bf16 bits, M x QW_HIDDEN: H + R rounded to bf16
                          __global ushort* XO)       // bf16 bits, M x QW_HIDDEN: RMSNorm(HO) * Wt
{
    const uint t = (uint)get_local_id(0);  // 0..QW_TPG-1, lane within the row's work-group
    const uint m = (uint)get_group_id(1);  // one work-group per row
    const int base = (int)m * QW_HIDDEN + (int)t * QW_ELEMS;

    float hv[QW_ELEMS];
    float ss = 0.0f;
    for (int i = 0; i < QW_ELEMS; i++) {
        float h = intel_convert_as_bfloat16_float(H[base + i]);
        h += intel_convert_as_bfloat16_float(R[base + i]);
        ushort hb = intel_convert_bfloat16_as_ushort(h);
        HO[base + i] = hb;
        hv[i] = intel_convert_as_bfloat16_float(hb);
        ss = fma(hv[i], hv[i], ss);
    }

    // Correct for any sub-group size the compiler picks (16 or 32 on Intel, not Metal's 32).
    const float total = work_group_reduce_add(ss);
    const float inv = rsqrt(total / (float)QW_HIDDEN + eps);

    for (int i = 0; i < QW_ELEMS; i++) {
        float w = intel_convert_as_bfloat16_float(Wt[(int)t * QW_ELEMS + i]);
        XO[base + i] = intel_convert_bfloat16_as_ushort(w * (hv[i] * inv));
    }
}
