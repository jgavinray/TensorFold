// qwen3_5 attn_partial (flash-decoding chunk partials) - port of attn_partial.metal to
// OpenCL C for the Intel Arc Pro B70.
//
// For every (query head qh, window row w, 128-key chunk c) the kernel writes the
// flash-decoding partials of the causal attention over keys 0..P+w (inclusive), where
// slot = (qh*W + w)*NCH + c:
//   PM[slot]         = max_j scale * <q, K_j>   over the chunk's valid keys (-inf if none)
//   PL[slot]         = sum_j exp(s_j - PM[slot])
//   PO[slot*256 + d] = sum_j exp(s_j - PM[slot]) * V_j[d]
// A chunk with no valid key for a row (k0 > P+w) writes PL = 0 and PO = 0 exactly and
// PM = -inf (attn_merge skips PL == 0 slots). bf16 inputs are converted exactly with the
// Intel bf16 conversion instruction; all arithmetic is fp32. Rows past the valid length
// are NaN and are never read: every key index stays below k1 = min(k0 + 128, P+w+1).
//
// Geometry: one work-group per (qh, w, c); global (S, 8*W, NCH), local (S, 1, 1), S = 16.
// Dimension 0 is an explicit slice index, not a lane id, so no sub-group width is assumed
// (Intel sub-groups are 16 or 32 wide); 16 items map onto exactly one 16-wide sub-group.
// The Metal kernel's "each SIMD lane owns D/32 dims, split keys 4 ways" becomes: each
// work-item owns D/S = 16 dims of the query and of the output accumulator o (private
// registers, no 256-float spills), and ALL keys of the chunk, processed in blocks of
// BK = 16. Per block every item computes its 16-dim partial dot of each key into local
// memory (one barrier), then reduces the full score sc_j itself - identical fp32 order on
// all items, so m/l stay uniform without a second cross-item merge - and rescales its own
// o slice with the running max. K and V rows are therefore read exactly once per slot and
// the only local traffic is BK floats per key; no per-key cross-item reduction, no
// sub-group intrinsics. Item dv owns PO dims [dv*DPT, dv*DPT + DPT); PM/PL by item 0.
#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#define D   256
#define CK  128
#define S   16         // work-items per group; item dv owns dims [dv*DPT, dv*DPT + DPT)
#define DPT (D / S)    // 16 dims per item
#define BK  16         // keys per block between barriers

__kernel void qwen35_attn_partial(__global const ushort* Q,    // bf16 bits, 8 x QROWS x 256 (q_norm + RoPE applied)
                                  __global const ushort* K,    // bf16 bits, 2 x CAP x 256 (K cache, RoPE applied)
                                  __global const ushort* V,    // bf16 bits, 2 x CAP x 256
                                  const float scale,           // 256^-0.5
                                  __global const int*   dims,  // [P, W, CAP, NCH, MAXD(unused), QROWS, QOFF, 0]
                                  __global float* PM,          // 8 x W x NCH
                                  __global float* PL,          // 8 x W x NCH
                                  __global float* PO)          // 8 x W x NCH x 256
{
    const int dv = (int)get_global_id(0);                    // 0..S-1: dim-slice index
    const int y  = (int)get_global_id(1);                    // qh*W + w
    const int c  = (int)get_global_id(2);                    // chunk index
    const int P  = dims[0];
    const int W  = dims[1];
    const int CAP  = dims[2];
    const int NCH  = dims[3];
    const int QROWS = dims[5];
    const int QOFF  = dims[6];
    const int w  = y % W;
    const int qh = y / W;
    const int h  = qh / 4;                                   // 4 query heads share one kv head

    __local float part[BK][S];   // per-key partial dots, one column per item

    const int k0   = c * CK;
    const int last = P + w;
    const int k1   = min(k0 + CK, last + 1);

    // This item's query slice and zeroed output accumulator, fp32 registers.
    float q[DPT], o[DPT];
    const size_t qrow = (size_t)(qh * QROWS + QOFF + w) * D;
#pragma unroll
    for (int i = 0; i < DPT; i++) {
        q[i] = intel_convert_as_bfloat16_float(Q[qrow + (size_t)(dv * DPT + i)]);
        o[i] = 0.0f;
    }

    float m = -INFINITY, l = 0.0f;
    for (int j0 = k0; j0 < k1; j0 += BK) {
        // 1) this slice's partial dot for each key of the block (0 for out-of-range keys)
        for (int jj = 0; jj < BK; jj++) {
            const int j = j0 + jj;
            float p = 0.0f;
            if (j < k1) {
                const __global ushort* kr = K + ((size_t)h * CAP + (size_t)j) * D + (size_t)(dv * DPT);
#pragma unroll
                for (int i = 0; i < DPT; i++)
                    p = fma(q[i], intel_convert_as_bfloat16_float(kr[i]), p);
            }
            part[jj][dv] = p;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // 2) every item reduces the full score of each key - identical fp32 summation order
        //    on all items, so m and l come out uniform without a cross-item merge.
        float sc[BK], eb[BK];
        float mn = -INFINITY;
#pragma unroll
        for (int jj = 0; jj < BK; jj++) {
            const float4 s4 = vload4(0, &part[jj][0]) + vload4(0, &part[jj][4])
                            + vload4(0, &part[jj][8]) + vload4(0, &part[jj][12]);
            sc[jj] = ((s4.x + s4.y) + (s4.z + s4.w)) * scale;
            if (j0 + jj < k1) mn = fmax(mn, sc[jj]);
        }
        const float mnew = fmax(m, mn);          // running max; block always has >= 1 valid key
        const float a = exp(m - mnew);           // m = -inf before the first block: exp(-inf) = 0
#pragma unroll
        for (int jj = 0; jj < BK; jj++) eb[jj] = exp(sc[jj] - mnew);

        float ls = 0.0f;
#pragma unroll
        for (int jj = 0; jj < BK; jj++)
            if (j0 + jj < k1) ls += eb[jj];
        l = l * a + ls;
        m = mnew;

        // 3) rescale this item's o slice once, then fold in the block's keys
#pragma unroll
        for (int i = 0; i < DPT; i++) o[i] *= a;
        for (int jj = 0; jj < BK; jj++) {
            if (j0 + jj < k1) {
                const __global ushort* vr = V + ((size_t)h * CAP + (size_t)(j0 + jj)) * D + (size_t)(dv * DPT);
#pragma unroll
                for (int i = 0; i < DPT; i++)
                    o[i] = fma(eb[jj], intel_convert_as_bfloat16_float(vr[i]), o[i]);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);            // part[][] is reused by the next block
    }

    // Every item holds the same (m, l) and its own 16 dims of the partial output.
    const size_t slot = (size_t)(qh * W + w) * (size_t)NCH + (size_t)c;
    if (dv == 0) {
        PM[slot] = m;
        PL[slot] = l;
    }
    const size_t po = slot * (size_t)D;
#pragma unroll
    for (int i = 0; i < DPT; i++)
        PO[po + (size_t)(dv * DPT + i)] = o[i];
}
