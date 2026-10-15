"""Launch geometry for qwen35_gdn_post on the Intel Arc Pro B70.

Same decomposition as the Metal kernel: one 32-lane work-group per (head, row), 4 elements per lane.
x = 32 lanes, y = QW_NV heads, z = M rows -> global (32, QW_NV, M), local (32, 1, 1). The cross-lane
reduction is work_group_reduce_add over the full work-group, so no sub-group-size assumption is made.
Defaults are the Qwen3.5-2B geometry (NV = 16); the 27B build passes NV = 48.
"""

NV = 16


def launch(M):
    M = int(M)
    return ((32, NV, M), (32, 1, 1))
