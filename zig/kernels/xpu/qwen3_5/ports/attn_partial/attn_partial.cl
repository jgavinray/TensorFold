// qwen3_5 attn_partial (flash-decoding chunk partials) - port of attn_partial.metal to
// OpenCL C for the Intel Arc Pro B70.
//
// For every (query head qh, window row w, 128-key chunk c) the kernel writes the flash-decoding
// partials of the causal attention over keys 0..P+w (inclusive), where slot = (qh*W + w)*NCH + c:
//   PM[slot]         = max_j scale * <q, K_j>   over the chunk's valid keys (-inf if none)
//   PL[slot]         = sum_j exp(s_j - PM[slot])
//   PO[slot*256 + d] = sum_j exp(s_j - PM[slot]) * V_j[d]
// A chunk with no valid key (k0 > P+w) writes PL = 0 and PO = 0 exactly and PM = -inf (attn_merge
// drops PL == 0 slots). Key rows past the valid length are NaN and are never read: every key index
// stays below k1 = min(k0 + 128, P+w+1).
//
// Work split (redesigned for the B70; the Metal original splits keys 4 ways across 4 SIMD groups
// and needs a cross-simdgroup merge, which has no portable Intel equivalent):
// one work-group per (qh, w, c); global (S, 8*W, NCH), local (S, 1, 1) with S = 16. Dimension 0 is
// an explicit dim-slice index, not a lane id, so no sub-group width is assumed - 16 work-items fill
// exactly one 16-wide Intel sub-group whatever the runtime picks. Each item owns D/S = 16 dims of
// the query and of the output accumulator o (private fp32 registers), and ALL keys of the chunk,
// processed in blocks of BK = 16.
//
// Per block each item computes its 16-dim partial dot of every key of the block, the partials go to
// one __local array indexed [key][item] (BK*16 floats = 1 KiB), one barrier, then every item reduces
// each key's full score itself from that array in an identical fp32 order. Because all items run the
// same fp32 sequence on the same values, m and l come out bit-identical on every item, so no second
// cross-item reduction and no sub-group intrinsic is needed anywhere in the kernel. The only local
// traffic is BK floats per key. K and V rows are read exactly once per (slot, key); the block is
// re-read only in the sense that V is touched after the scores are known, never speculatively.
//
// The block's exponentials are computed once into eb[] and reused for both the PL sum and the V fold,
// and the running output is rescaled by exp(m_old - m_new) exactly once per block (the standard
// flash-decoding rescale), instead of recomputing exp() per key per dimension.
#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#define D   256
#define CK  128
#define S   16         // work-items per group; item dv owns dims [dv*DPT, dv*DPT + DPT)
#define DPT (D / S)    // 16 dims per item
#define BK  16         // keys per block between barriers

#define BF(u) intel_convert_as_bfloat16_float(u)

__kernel void qwen35_attn_partial(__global const ushort* Q,    // bf16 bits, 8 x QROWS x 256 (q_norm + RoPE applied)
                                  __global const ushort* K,    // bf16 bits, 2 x CAP x 256 (K cache, RoPE applied)
                                  __global const ushort* V,    // bf16 bits, 2 x CAP x 256
                                  const float scale,           // 256^-0.5
                                  __global const int*   dims,  // [P, W, CAP, NCH, MAXD(unused), QROWS, QOFF, 0]
                                  __global float* PM,          // 8 x W x NCH
                                  __global float* PL,          // 8 x W x NCH
                                  __global float* PO)          // 8 x W x NCH x 256
{
    const int dv = (int)get_local_id(0);          // 0..S-1: dim-slice index (not a lane id)
    const int y  = (int)get_group_id(1);          // qh*W + w
    const int c  = (int)get_group_id(2);          // chunk index
    const int P    = dims[0];
    const int W    = dims[1];
    const int CAP  = dims[2];
    const int NCH  = dims[3];
    const int QROWS = dims[5];
    const int QOFF  = dims[6];
    const int w  = y % W;
    const int qh = y / W;
    const int h  = qh >> 2;                       // 4 query heads share one kv head

    __local float part[BK][S];                    // per-key partial dots, one column per item

    const int k0   = c * CK;
    const int last = P + w;
    const int k1   = min(k0 + CK, last + 1);

    // This item's query slice and zeroed output accumulator, private fp32 registers.
    float q[DPT], o[DPT];
    const __global ushort* qrow = Q + ((size_t)(qh * QROWS + QOFF + w)) * D + (size_t)(dv * DPT);
#pragma unroll
    for (int i = 0; i < DPT; i++) {
        q[i] = BF(qrow[i]);
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
                    p = fma(q[i], BF(kr[i]), p);
            }
            part[jj][dv] = p;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // 2) every item reduces the full score of each key in the same fp32 order, so m and l stay
        //    identical on all items and need no cross-item merge.
        float sc[BK], eb[BK];
        float mn = -INFINITY;
#pragma unroll
        for (int jj = 0; jj < BK; jj++) {
            const float4 s4 = vload4(0, &part[jj][0]) + vload4(0, &part[jj][4])
                            + vload4(0, &part[jj][8]) + vload4(0, &part[jj][12]);
            sc[jj] = ((s4.x + s4.y) + (s4.z + s4.w)) * scale;
            if (j0 + jj < k1) mn = fmax(mn, sc[jj]);
        }
        const float mnew = fmax(m, mn);           // the block always holds >= 1 valid key
        const float a = exp(m - mnew);            // m = -inf before the first block: exp(-inf) = 0
#pragma unroll
        for (int jj = 0; jj < BK; jj++) eb[jj] = exp(sc[jj] - mnew);

        float ls = 0.0f;
#pragma unroll
        for (int jj = 0; jj < BK; jj++)
            if (j0 + jj < k1) ls += eb[jj];
        l = l * a + ls;
        m = mnew;

        // 3) rescale this item's o slice once, then fold in the block's V rows with the eb[] weights
#pragma unroll
        for (int i = 0; i < DPT; i++) o[i] *= a;
        for (int jj = 0; jj < BK; jj++) {
            if (j0 + jj < k1) {
                const __global ushort* vr = V + ((size_t)h * CAP + (size_t)(j0 + jj)) * D + (size_t)(dv * DPT);
#pragma unroll
                for (int i = 0; i < DPT; i++)
                    o[i] = fma(eb[jj], BF(vr[i]), o[i]);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);             // part[][] is reused by the next block
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
