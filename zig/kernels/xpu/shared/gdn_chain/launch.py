"""Launch geometry for qwen35_gdn_chain.

One 16-lane sub-group per state row; the state has QW_HV * QW_DV rows and QW_ROW_LANES lanes each, so
the grid is QW_ROW_LANES * QW_HV * QW_DV work-items with a local size of 128. Defaults are the
Qwen3.5-2B geometry: HV = 16, DV = 128, ROW_LANES = 16 -> global (16*16*128), local (128,).
The 27B build passes HV = 48 -> global (16*48*128).
"""

DV = 128
HV = 16
ROW_LANES = 16  # one 16-lane sub-group per state row (coupled to QW_DK = 128)


def launch(W):
    return (ROW_LANES * HV * DV,), (128,)
