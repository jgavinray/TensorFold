"""Launch geometry for qwen35_norm: one work-group of QW_TPG work-items per row (QW_ELEMS each).

Defaults are the Qwen3.5-2B geometry: K = 2048, E = 16, TPG = 128. A 27B build that overrides
QW_HIDDEN must launch with the matching QW_TPG (see the defines at the top of norm.cl).
"""

HIDDEN = 2048
ELEMS = 16
TPG = HIDDEN // ELEMS  # 128


def launch(M):
    return ((TPG, M), (TPG, 1))
