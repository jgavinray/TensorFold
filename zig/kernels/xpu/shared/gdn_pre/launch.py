"""Launch geometry for qwen35_gdn_pre.

QW_DK work-items per (head, row): one work-item per channel of the head, QW_HEADS3 = 2*QW_NK + QW_NV
heads. Defaults are the Qwen3.5-2B geometry: DK = 128, heads3 = 2*16 + 16 = 48, so global
(128, 48, W), local (128, 1, 1). The 27B build passes heads3 = 2*16 + 48 = 80.
"""

DK = 128
NK = 16
NV = 16
HEADS3 = 2 * NK + NV  # 48 at the 2B default


def launch(W):
    return (DK, HEADS3, W), (DK, 1, 1)
