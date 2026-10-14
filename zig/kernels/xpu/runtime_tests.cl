// Level Zero runtime qualification kernels (compiled by ocloc for bmg; see zig/build/xpu.zig).

// Writes a value that cannot equal the sentinel the host filled before launch: any stale readback shows.
kernel void tf_xpu_write(global uint* out, uint n) {
    uint i = get_global_id(0);
    if (i < n) out[i] = i * 3u + 7u;
}

// Mixed-width arguments: u32 pointer target, u32, u8, u64.
kernel void tf_xpu_probe(global uint* out, uint n, uchar a, ulong big) {
    uint i = get_global_id(0);
    if (i < n) out[i] = i + (uint)a + (uint)(big >> 32);
}
