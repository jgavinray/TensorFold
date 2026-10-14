// qwen3_5 attn_merge (flash-decoding chunk merge) - port of attn_merge.metal to OpenCL C
// for the Intel Arc Pro B70.
//
// Merges per-chunk softmax partials into the final attention output. For each (qh, w):
//   mx   = max_c PM[slot]                        (slot = (qh*W + w)*NCH + c)
//   e_c  = PL[slot] > 0 ? exp(PM[slot] - mx) : 0 (empty chunks: PM = -inf, PL = 0, PO = 0)
//   lsum = sum_c PL[slot] * e_c
//   acc  = sum_c PO[slot*256 + d] * e_c
//   OUT[(w*8 + qh)*256 + d] = bf16(acc / lsum)   (fp32 math, one round-to-nearest-even)
//
// Geometry mirrors the Metal grid without sub-group intrinsics: global (32, 8, W),
// local (32, 1, 1). Dimension 0 is an explicit 32-wide slice index, not a lane id, and
// each work-item owns DPL = 256/32 = 8 contiguous dims, so the 32 items of a slice
// together cover all 256 dims with fully contiguous 8-float blocks in PM-free PO reads
// and OUT writes. mx/lsum are recomputed per slice exactly as the Metal lanes did.
#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#define D 256
#define DPL (D / 32)

__kernel void qwen35_attn_merge(__global const float* PM,   // 8 x W x NCH   chunk max of scaled scores (-inf for an empty chunk)
                                __global const float* PL,   // 8 x W x NCH   chunk sum of exp(score - PM)  (0 for an empty chunk)
                                __global const float* PO,   // 8 x W x NCH x 256   chunk sum of exp(score - PM) * V
                                __global const int*   dims, // dims[1] = W (window rows), dims[3] = NCH (chunks); other entries unused here
                                __global ushort* OUT)       // bf16 bits, W x 8 x 256
{
    const int d0   = (int)get_global_id(0);                  // 0..31: this item's 8-dim slice
    const int qh   = (int)get_global_id(1);                  // 0..7: query head
    const int w    = (int)get_global_id(2);                  // 0..W-1: window row
    const int W    = dims[1];
    const int NCH  = dims[3];
    const int base = (qh * W + w) * NCH;

    float mx = -INFINITY;
    for (int c = 0; c < NCH; c++) mx = fmax(mx, PM[base + c]);

    float lsum = 0.0f;
    float acc[DPL];
#pragma unroll
    for (int i = 0; i < DPL; i++) acc[i] = 0.0f;

    for (int c = 0; c < NCH; c++) {
        // PL == 0 marks an empty chunk: contribute nothing, never exp(PM - mx) with PM = -inf.
        const float e = PL[base + c] > 0.0f ? exp(PM[base + c] - mx) : 0.0f;
        lsum = fma(PL[base + c], e, lsum);
        const size_t po = (size_t)(base + c) * D + (size_t)(d0 * DPL);
#pragma unroll
        for (int i = 0; i < DPL; i++) acc[i] = fma(PO[po + (size_t)i], e, acc[i]);
    }

    const size_t o = ((size_t)w * 8 + (size_t)qh) * D + (size_t)(d0 * DPL);
#pragma unroll
    for (int i = 0; i < DPL; i++) OUT[o + (size_t)i] = intel_convert_bfloat16_as_ushort(acc[i] / lsum);
}
