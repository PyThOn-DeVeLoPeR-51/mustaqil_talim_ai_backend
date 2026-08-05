"""Line-type score (8 points)."""

from __future__ import annotations

import cv2
import numpy as np

from app.ai.etalon.io import bin_to_lines

def dashed_like_count(lines_255: np.ndarray):
    m = (lines_255 > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m = cv2.erode(m, k, iterations=1)

    h_long = cv2.getStructuringElement(cv2.MORPH_RECT, (61, 1))
    v_long = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 61))
    long_h = cv2.morphologyEx(m, cv2.MORPH_OPEN, h_long, iterations=1)
    long_v = cv2.morphologyEx(m, cv2.MORPH_OPEN, v_long, iterations=1)
    m[cv2.bitwise_or(long_h, long_v) > 0] = 0

    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    cnt = 0
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if 10 <= area <= 220 and 2 <= w <= 45 and 2 <= h <= 45:
            cnt += 1
    return int(cnt), m


def stroke_width_stats(binary_thr: np.ndarray):
    inv = (255 - binary_thr)
    inv = (inv > 0).astype(np.uint8)
    if inv.sum() < 800:
        return None
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    vals = dist[inv > 0]
    p25 = float(np.percentile(vals, 25))
    p50 = float(np.percentile(vals, 50))
    p75 = float(np.percentile(vals, 75))
    p90 = float(np.percentile(vals, 90))
    r9050 = float(p90 / max(p50, 1e-6))
    r7525 = float(p75 / max(p25, 1e-6))
    return {"p50": p50, "r9050": r9050, "r7525": r7525}


def score_line_types_8(et_r: np.ndarray, st_aligned: np.ndarray):
    et_lines = bin_to_lines(et_r)
    st_lines = bin_to_lines(st_aligned)

    et_w = stroke_width_stats(et_r)
    st_w = stroke_width_stats(st_aligned)

    et_dash, et_dash_mask = dashed_like_count(et_lines)
    st_dash, st_dash_mask = dashed_like_count(st_lines)

    if et_w is None or st_w is None:
        return 0, {
            "reason": "too few strokes for width analysis",
            "et_dash": et_dash,
            "st_dash": st_dash,
        }, et_dash_mask, st_dash_mask

    div_et = 0.6 * et_w["r9050"] + 0.4 * et_w["r7525"]
    div_st = 0.6 * st_w["r9050"] + 0.4 * st_w["r7525"]

    dash_ratio = float((st_dash + 1) / (et_dash + 1))
    div_ratio = float(div_st / max(div_et, 1e-6))

    ABS_DASH_MIN = 70
    ABS_DASH_RATIO = 2.2

    gated = (st_dash >= ABS_DASH_MIN) and (dash_ratio >= ABS_DASH_RATIO)

    if gated:
        dbg = {
            "gated": True,
            "reason": "Absolute noisy dashed signature => 0",
            "et_dash": et_dash,
            "st_dash": st_dash,
            "dash_ratio": round(dash_ratio, 4),
            "div_et": round(div_et, 4),
            "div_st": round(div_st, 4),
            "div_ratio": round(div_ratio, 4),
        }
        return 0, dbg, et_dash_mask, st_dash_mask

    t_ratio = float(st_w["p50"] / max(et_w["p50"], 1e-6))
    t_ratio = float(max(t_ratio, 1.0 / t_ratio))
    thick_match_q = float(np.clip(1.0 - (t_ratio - 1.0) / 0.80, 0.0, 1.0))

    div_st_q = float(np.clip((div_st - 1.05) / (1.35 - 1.05), 0.0, 1.0))
    div_et_q = float(np.clip((div_et - 1.05) / (1.35 - 1.05), 0.0, 1.0))
    diversity_q = float(np.clip((1.0 - div_et_q) + div_et_q * div_st_q, 0.0, 1.0))

    if et_dash < 20:
        dash_q = 1.0
    else:
        dash_q = float(np.clip(st_dash / max(et_dash, 1), 0.0, 1.0))
        if st_dash < 8:
            dash_q = 0.0

    score01 = 0.45 * thick_match_q + 0.35 * diversity_q + 0.20 * dash_q
    line_pts = int(np.clip(round(score01 * 8), 0, 8))

    dbg = {
        "gated": False,
        "t_ratio_sym": round(t_ratio, 4),
        "thick_match_q": round(thick_match_q, 4),
        "div_et": round(div_et, 4),
        "div_st": round(div_st, 4),
        "div_ratio": round(div_ratio, 4),
        "et_dash": et_dash,
        "st_dash": st_dash,
        "dash_ratio": round(dash_ratio, 4),
        "dash_q": round(dash_q, 4),
        "diversity_q": round(diversity_q, 4),
        "score01": round(score01, 4),
    }
    return line_pts, dbg, et_dash_mask, st_dash_mask


__all__ = [
    'dashed_like_count',
    'stroke_width_stats',
    'score_line_types_8',
]
