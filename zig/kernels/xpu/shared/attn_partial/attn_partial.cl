// qwen3_5 attn_partial (flash-decoding chunk partials) - port of attn_partial.metal to
// OpenCL C for the Intel Arc Pro B70.
//
// For every (query head qh, window row w, QW_CHUNK-key chunk c) the kernel writes the
// flash-decoding partials of the causal attention over keys 0..P+w (inclusive), where
// slot = (qh*W + w)*NCH + c:
//   PM[slot]              = max_j scale * <q, K_j>   over the chunk's valid keys (-inf if none)
//   PL[slot]              = sum_j exp(s_j - PM[slot])
//   PO[slot*QW_HEAD_DIM+d] = sum_j exp(s_j - PM[slot]) * V_j[d]
// A chunk with no valid key for a row (k0 > P+w) writes PL = 0 and PO = 0 exactly and
// PM = -inf (attn_merge skips PL == 0 slots). bf16 inputs are converted exactly with the
// Intel bf16 conversion instruction; all arithmetic is fp32. Rows past the valid length
// are NaN and are never read: every key index stays below k1 = min(k0 + QW_CHUNK, P+w+1).
//
// Geometry: one work-group per (qh, w, c); global (S, QW_Q_HEADS*W, NCH), local (S, 1, 1), S = 16.
// Dimension 0 is an explicit slice index, not a lane id, so no sub-group width is assumed
// (Intel sub-groups are 16 or 32 wide); 16 items map onto exactly one 16-wide sub-group.
// Each work-item owns QW_HEAD_DIM/S dims of the query and of the output accumulator o (private
// registers, no QW_HEAD_DIM-float spills), and ALL keys of the chunk, processed in blocks of
// BK = 16. Per block every item computes its DPT-dim partial dot of each key into local
// memory (one barrier), then reduces the full score sc_j itself - identical fp32 order on
// all items, so m/l stay uniform without a second cross-item merge - and rescales its own
// o slice with the running max. K and V rows are therefore read exactly once per slot and
// the only local traffic is BK floats per key; no per-key cross-item reduction, no
// sub-group intrinsics. Item dv owns PO dims [dv*DPT, dv*DPT + DPT); PM/PL by item 0.
//
// Shapes are compile-time defines; defaults are the Qwen3.5-2B geometry, so the proven 2B gate applies
// to this file unchanged. QW_HEAD_DIM (256) is the attention head width (the same on 2B and 27B);
// QW_Q_HEADS (8 vs 24) only enters the launch geometry (attn_partial folds heads into the y axis).
#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#ifndef QW_HEAD_DIM
#define QW_HEAD_DIM 256
#endif
#ifndef QW_CHUNK
#define QW_CHUNK 128      // keys per flash-decoding chunk (fixed by the plan, not by the model)
#endif
#ifndef QW_Q_HEADS
#define QW_Q_HEADS 8      // num_attention_heads
#endif
#ifndef QW_KV_HEADS
#define QW_KV_HEADS 2     // num_key_value_heads; QW_Q_HEADS / QW_KV_HEADS query heads share one kv head
#endif
#define QW_Q_PER_KV (QW_Q_HEADS / QW_KV_HEADS)   // 4 on 2B, 6 on 27B

#define D    QW_HEAD_DIM
#define CK   QW_CHUNK
#define S    16           // work-items per group; item dv owns dims [dv*DPT, dv*DPT + DPT)
#define DPT  (D / S)      // dims per item
#define BK   16           // keys per block between barriers

__kernel void qwen35_attn_partial(__global const ushort* Q,    // bf16 bits, QW_Q_HEADS x QROWS x QW_HEAD_DIM (q_norm + RoPE applied)
                                  __global const ushort* K,    // bf16 bits, 2 x CAP x QW_HEAD_DIM (K cache, RoPE applied)
                                  __global const ushort* V,    // bf16 bits, 2 x CAP x QW_HEAD_DIM
                                  const float scale,           // QW_HEAD_DIM^-0.5
                                  __global const int*   dims,  // [P, W, CAP, NCH, MAXD(unused), QROWS, QOFF, 0]
                                  __global float* PM,          // QW_Q_HEADS x W x NCH
                                  __global float* PL,          // QW_Q_HEADS x W x NCH
                                  __global float* PO)          // QW_Q_HEADS x W x NCH x QW_HEAD_DIM
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
    const int h  = qh / QW_Q_PER_KV;                         // QW_Q_PER_KV query heads share one kv head

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

    // Every item holds the same (m, l) and its own DPT dims of the partial output.
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

// ---------------------------------------------------------------------------------------------------------------
#ifndef QA_SPLIT
#define QA_SPLIT 4                 // key slices per chunk; slice t owns keys [k0 + t*CK/QA_SPLIT, +CK/QA_SPLIT)
#endif
#define SK (CK / QA_SPLIT)         // keys per slice
#define NSG (QA_SPLIT * QW_Q_PER_KV) // 16-item sub-groups per work-group

// qwen35_attn_partial_gqa: the same per-(qh, w, chunk) partials as qwen35_attn_partial. One work-group covers the
// QW_Q_PER_KV query heads that share a kv head (so the chunk's K/V rows come from DRAM once and from L1 for the other
// heads - the shipped kernel reads them once per query head) and splits the chunk's keys across QA_SPLIT slices that
// run concurrently, merged in local memory (max-rescaled, fixed order).
// 27B decode at 474 context: 211 -> 31 us per layer; equal at 4K, +10% at 16K (exp/kernels/attn/partial_check.py).
// Geometry: global (S*NSG, QW_KV_HEADS*W, NCH), local (S*NSG, 1, 1). Sub-group g: head j = g % QW_Q_PER_KV,
// slice t = g / QW_Q_PER_KV.
__kernel void qwen35_attn_partial_gqa(__global const ushort* Q, __global const ushort* K, __global const ushort* V,
                                      const float scale, __global const int* dims, __global float* PM,
                                      __global float* PL, __global float* PO)
{
    const int lid = (int)get_local_id(0);
    const int sg = lid / S, dv = lid % S;
    const int j = sg % QW_Q_PER_KV, t = sg / QW_Q_PER_KV;
    const int y  = (int)get_global_id(1);
    const int c  = (int)get_global_id(2);
    const int P  = dims[0], W = dims[1], CAP = dims[2], NCH = dims[3], QROWS = dims[5], QOFF = dims[6];
    const int w  = y % W, h = y / W, qh = h * QW_Q_PER_KV + j;

    // One local buffer for both phases (BK*S == D == 256 floats per sub-group): the per-key partial dots during the
    // key loop, then each sub-group's output slice for the merge. Halves the local memory, which bounds occupancy.
    __local float buf[NSG][D];
    __local float sm[NSG], sl[NSG];
    __local float (*part)[BK][S] = (__local float (*)[BK][S])buf;

    const int last = P + w;
    const int ks = c * CK + t * SK;
    const int ke = min(ks + SK, last + 1);

    float q[DPT], o[DPT];
    const size_t qrow = (size_t)(qh * QROWS + QOFF + w) * D;
#pragma unroll
    for (int i = 0; i < DPT; i++) {
        q[i] = intel_convert_as_bfloat16_float(Q[qrow + (size_t)(dv * DPT + i)]);
        o[i] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    for (int j0 = ks; j0 < ks + SK; j0 += BK) {
        for (int jj = 0; jj < BK; jj++) {
            const int k = j0 + jj;
            float p = 0.0f;
            if (k < ke) {
                const __global ushort* kr = K + ((size_t)h * CAP + (size_t)k) * D + (size_t)(dv * DPT);
#pragma unroll
                for (int i = 0; i < DPT; i++) p = fma(q[i], intel_convert_as_bfloat16_float(kr[i]), p);
            }
            part[sg][jj][dv] = p;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (j0 < ke) {
            float sc[BK], eb[BK];
            float mn = -INFINITY;
#pragma unroll
            for (int jj = 0; jj < BK; jj++) {
                const float4 s4 = vload4(0, &part[sg][jj][0]) + vload4(0, &part[sg][jj][4])
                                + vload4(0, &part[sg][jj][8]) + vload4(0, &part[sg][jj][12]);
                sc[jj] = ((s4.x + s4.y) + (s4.z + s4.w)) * scale;
                if (j0 + jj < ke) mn = fmax(mn, sc[jj]);
            }
            const float mnew = fmax(m, mn);
            const float a = exp(m - mnew);
#pragma unroll
            for (int jj = 0; jj < BK; jj++) eb[jj] = exp(sc[jj] - mnew);
            float ls = 0.0f;
#pragma unroll
            for (int jj = 0; jj < BK; jj++)
                if (j0 + jj < ke) ls += eb[jj];
            l = l * a + ls;
            m = mnew;
#pragma unroll
            for (int i = 0; i < DPT; i++) o[i] *= a;
            for (int jj = 0; jj < BK; jj++) {
                if (j0 + jj < ke) {
                    const __global ushort* vr = V + ((size_t)h * CAP + (size_t)(j0 + jj)) * D + (size_t)(dv * DPT);
#pragma unroll
                    for (int i = 0; i < DPT; i++) o[i] = fma(eb[jj], intel_convert_as_bfloat16_float(vr[i]), o[i]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (dv == 0) { sm[sg] = m; sl[sg] = l; }
#pragma unroll
    for (int i = 0; i < DPT; i++) buf[sg][dv * DPT + i] = o[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t != 0) return;
    float M = -INFINITY;
    for (int u = 0; u < QA_SPLIT; u++) { const int g = u * QW_Q_PER_KV + j; if (sl[g] > 0.0f) M = fmax(M, sm[g]); }
    float wt[QA_SPLIT], L = 0.0f;
    for (int u = 0; u < QA_SPLIT; u++) {
        const int g = u * QW_Q_PER_KV + j;
        wt[u] = sl[g] > 0.0f ? exp(sm[g] - M) : 0.0f;
        L += sl[g] * wt[u];
    }
    const size_t slot = (size_t)(qh * W + w) * (size_t)NCH + (size_t)c;
    if (dv == 0) { PM[slot] = M; PL[slot] = L; }
    const size_t po = slot * (size_t)D;
#pragma unroll
    for (int i = 0; i < DPT; i++) {
        float acc = 0.0f;
        for (int u = 0; u < QA_SPLIT; u++) acc += buf[u * QW_Q_PER_KV + j][dv * DPT + i] * wt[u];
        PO[po + (size_t)(dv * DPT + i)] = acc;
    }
}
