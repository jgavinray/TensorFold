// DPAS probes: one dpas.8x1 on a 16-wide sub-group (port of exp/xmx-probe/dpas_onCard.cpp). A is a 1xK row spread over the
// 16 work-items (one short each); B is Kx16 with column n held by work-item n as 8 packed ints; C[0][n] lands in
// work-item n. Operand flags: 0x1/0x2 A/B signed, 0x10/0x20 packed int8, 0x40/0x80 packed int4.
typedef int int8v __attribute__((ext_vector_type(8)));
int __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int, short, int8v, int, int);

__attribute__((intel_reqd_sub_group_size(16)))
kernel void tf_dpas_int8(global const short* a, global const int* b, global int* c) {
    int w = get_local_id(0);
    int8v bv;
    for (int j = 0; j < 8; j++) bv[j] = b[w * 8 + j];
    c[w] = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(32, a[w], bv, 0, 0x33);
}

__attribute__((intel_reqd_sub_group_size(16)))
kernel void tf_dpas_int4(global const short* a, global const int* b, global int* c) {
    int w = get_local_id(0);
    int8v bv;
    for (int j = 0; j < 8; j++) bv[j] = b[w * 8 + j];
    c[w] = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(64, a[w], bv, 0, 0xC3);
}
