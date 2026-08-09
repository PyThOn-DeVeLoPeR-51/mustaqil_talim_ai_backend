"""Frame and title-block score (3 points)."""

from __future__ import annotations

import numpy as np

def score_frame_titleblock_3(metrics: dict):
    sim = float(metrics.get("similarity", 0.0))
    cov = float(metrics.get("coverage", 0.0))
    miss = float(metrics.get("missing_ratio", 1.0))
    extra = float(metrics.get("extra_ratio", 1.0))

    if sim < 0.12 or cov < 0.12:
        pts = 0
    else:
        q = (
            0.45 * sim +
            0.25 * cov +
            0.20 * (1.0 - miss) +
            0.10 * (1.0 - min(extra, 1.0))
        )
        pts = int(np.clip(round(q * 3), 0, 3))

    dbg = {
        "similarity": round(sim, 4),
        "coverage": round(cov, 4),
        "missing_ratio": round(miss, 4),
        "extra_ratio": round(extra, 4),
        "note": "placeholder_logic_until_real_step1_detector_is_inserted",
    }
    return pts, dbg


__all__ = [
    'score_frame_titleblock_3',
]
