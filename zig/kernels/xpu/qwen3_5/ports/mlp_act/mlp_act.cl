// qwen3_5 mlp_act (SwiGLU) - port of mlp_act.metal to OpenCL C for the Intel Arc Pro B70.
//
// HOUT[m][i] = silu(GATE[m][i]) * UP[m][i] over M x 6144, bf16 carried as ushort bits.
// Math is fp32: g converted from bf16 (lossless), silu computed as g / (1 + exp(-g)),
// multiplied by u, and rounded once (round-to-nearest-even) to bf16 on the store.
//
// Each work-item handles 8 contiguous columns of one row (ushort8 I/O via vload8/vstore8,
// which need no alignment beyond the element type). 6144 % 8 == 0 leaves no tail and the
// 12288-byte row stride keeps every vector access inside one row. The guard keeps a
// mislaunched grid from writing out of bounds, mirroring the Metal original.
#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#define N 6144
#define VEC 8
#define COLS_PER_ROW (N / VEC)

__kernel void qwen35_mlp_act(__global const ushort* GATE,   // bf16 bits, M x 6144 (gate_proj output)
                             __global const ushort* UP,     // bf16 bits, M x 6144 (up_proj output)
                             __global ushort* HOUT)         // bf16 bits, M x 6144
{
    const size_t c = get_global_id(0);                      // column group index, 0..COLS_PER_ROW-1
    const size_t m = get_global_id(1);                      // row
    if (c >= (size_t)COLS_PER_ROW) return;

    const size_t base = m * (size_t)N + c * (size_t)VEC;
    const ushort8 gbits = vload8(0, GATE + base);
    const ushort8 ubits = vload8(0, UP + base);

    const float8 g = (float8)(intel_convert_as_bfloat16_float(gbits.s0),
                              intel_convert_as_bfloat16_float(gbits.s1),
                              intel_convert_as_bfloat16_float(gbits.s2),
                              intel_convert_as_bfloat16_float(gbits.s3),
                              intel_convert_as_bfloat16_float(gbits.s4),
                              intel_convert_as_bfloat16_float(gbits.s5),
                              intel_convert_as_bfloat16_float(gbits.s6),
                              intel_convert_as_bfloat16_float(gbits.s7));
    const float8 u = (float8)(intel_convert_as_bfloat16_float(ubits.s0),
                              intel_convert_as_bfloat16_float(ubits.s1),
                              intel_convert_as_bfloat16_float(ubits.s2),
                              intel_convert_as_bfloat16_float(ubits.s3),
                              intel_convert_as_bfloat16_float(ubits.s4),
                              intel_convert_as_bfloat16_float(ubits.s5),
                              intel_convert_as_bfloat16_float(ubits.s6),
                              intel_convert_as_bfloat16_float(ubits.s7));

    const float8 y = g / (1.0f + exp(-g)) * u;

    vstore8((ushort8)(intel_convert_bfloat16_as_ushort(y.s0),
                      intel_convert_bfloat16_as_ushort(y.s1),
                      intel_convert_bfloat16_as_ushort(y.s2),
                      intel_convert_bfloat16_as_ushort(y.s3),
                      intel_convert_bfloat16_as_ushort(y.s4),
                      intel_convert_bfloat16_as_ushort(y.s5),
                      intel_convert_bfloat16_as_ushort(y.s6),
                      intel_convert_bfloat16_as_ushort(y.s7)), 0, HOUT + base);
}
