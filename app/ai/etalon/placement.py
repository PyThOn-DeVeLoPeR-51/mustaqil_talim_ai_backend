"""Drawing placement score (6 points)."""

from __future__ import annotations

import cv2
import numpy as np

from app.ai.common.hough import normalize_hough_lines

from app.ai.etalon.io import bin_to_lines

def content_bbox(lines_255: np.ndarray, pad=20):
    ys, xs = np.where(lines_255 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    H, W = lines_255.shape[:2]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W - 1, x2 + pad); y2 = min(H - 1, y2 + pad)
    return (x1, y1, x2, y2)


def bbox_center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def angle_hist(lines_255, bins=18):
    edges = cv2.Canny(lines_255, 50, 150, apertureSize=3)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=10)
    if segs is None:
        return np.ones(bins) / bins
    angs = []
    for x1, y1, x2, y2 in normalize_hough_lines(segs):
        dx = x2 - x1; dy = y2 - y1
        L = np.hypot(dx, dy)
        if L < 40:
            continue
        a = (np.degrees(np.arctan2(dy, dx)) % 180.0)
        angs.append(a)
    if not angs:
        return np.ones(bins) / bins
    h, _ = np.histogram(angs, bins=bins, range=(0, 180))
    h = h.astype(np.float32)
    h = h / max(h.sum(), 1.0)
    return h


def cos_sim(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def score_placement_6(et_r: np.ndarray, st_aligned: np.ndarray):
    et_lines = bin_to_lines(et_r)
    st_lines = bin_to_lines(st_aligned)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    et_band = cv2.dilate(et_lines, k, iterations=1)
    st_inside = cv2.bitwise_and(st_lines, et_band)
    coverage = float((st_inside > 0).sum() / max((st_lines > 0).sum(), 1))

    eb = content_bbox(et_lines)
    sb = content_bbox(st_lines)
    H, W = et_lines.shape[:2]

    if eb is None or sb is None:
        return 0, {
            "coverage": round(coverage, 4),
            "shift_norm": None,
            "angle_sim": None,
            "reason": "content bbox not found",
        }

    ecx, ecy = bbox_center(eb)
    scx, scy = bbox_center(sb)
    shift = np.hypot(scx - ecx, scy - ecy)
    shift_norm = float(shift / max(W, H))
    shift_q = float(np.clip(1.0 - (shift_norm / 0.12), 0.0, 1.0))

    et_h = angle_hist(et_lines, bins=18)
    st_h = angle_hist(st_lines, bins=18)
    angle_sim = cos_sim(et_h, st_h)

    placement_score01 = (0.50 * coverage + 0.35 * shift_q + 0.15 * angle_sim)
    placement_pts = int(np.clip(round(placement_score01 * 6), 0, 6))

    dbg = {
        "coverage": round(coverage, 4),
        "shift_px": round(float(shift), 4),
        "shift_norm": round(shift_norm, 4),
        "shift_q": round(shift_q, 4),
        "angle_sim": round(angle_sim, 4),
        "placement_score01": round(float(placement_score01), 4),
    }
    return placement_pts, dbg


__all__ = [
    'content_bbox',
    'bbox_center',
    'angle_hist',
    'cos_sim',
    'score_placement_6',
]
