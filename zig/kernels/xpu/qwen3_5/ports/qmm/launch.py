"""Launch geometry for qmm.cl (qwen35_qmm).

One work-group is 256 threads covering a 64x32 tile of OUT: 16 sub-groups of 16 lanes,
laid out as 8 row strips x 2 column strips. Grid = (ceil(N/32), ceil(R/64)).
The kernel clamps tail rows and guards tail columns, so the grid may overhang N and R.
"""

TILE_N = 32
TILE_M = 64


def launch(R, N, K):
    gx = max(1, (N + TILE_N - 1) // TILE_N)
    gy = max(1, (R + TILE_M - 1) // TILE_M)
    return (gx * 256, gy, 1), (256, 1, 1)
