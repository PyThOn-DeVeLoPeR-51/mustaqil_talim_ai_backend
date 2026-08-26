"""Dimensioning score (12 points)."""

from __future__ import annotations

import cv2
import numpy as np

from app.ai.common.hough import normalize_hough_lines

from app.ai.etalon.io import bin_to_lines
from app.ai.etalon.line_types import dashed_like_count

def remove_long_axes(lines_255: np.ndarray):
    m = lines_255.copy()
    h_long = cv2.getStructuringElement(cv2.MORPH_RECT, (81, 1))
    v_long = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 81))
    long_h = cv2.morphologyEx(m, cv2.MORPH_OPEN, h_long, iterations=1)
    long_v = cv2.morphologyEx(m, cv2.MORPH_OPEN, v_long, iterations=1)
    long_mask = cv2.bitwise_or(long_h, long_v)
    out = m.copy()
    out[long_mask > 0] = 0
    return out, long_mask


def small_component_stats(mask_255: np.ndarray):
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_255 > 0).astype(np.uint8), connectivity=8)
    small_cnt = 0
    small_area_sum = 0
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if 12 <= area <= 450 and 3 <= w <= 60 and 3 <= h <= 60:
            small_cnt += 1
            small_area_sum += area
    return small_cnt, small_area_sum


def dimension_like_mask(lines_255: np.ndarray):
    short, long_mask = remove_long_axes(lines_255)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    short = cv2.morphologyEx(short, cv2.MORPH_OPEN, k, iterations=1)
    short = cv2.morphologyEx(short, cv2.MORPH_CLOSE, k, iterations=1)
    return short, long_mask


def line_long_ratio(lines_255: np.ndarray):
    edges = cv2.Canny(lines_255, 50, 150, apertureSize=3)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=10)
    if segs is None:
        return 0.0
    lengths = []
    for x1, y1, x2, y2 in normalize_hough_lines(segs):
        L = float(np.hypot(x2 - x1, y2 - y1))
        if L >= 10:
            lengths.append(L)
    if not lengths:
        return 0.0
    lengths = np.array(lengths, dtype=np.float32)
    H, W = lines_255.shape[:2]
    Lthr = max(120.0, 0.10 * max(H, W))
    return float((lengths >= Lthr).mean())


def score_dimension_12(et_r: np.ndarray, st_aligned: np.ndarray, metrics: dict):
    et_lines = bin_to_lines(et_r)
    st_lines = bin_to_lines(st_aligned)

    sim = max(0.0, min(float(metrics.get("similarity", 0.0)), 1.0))
    miss = float(metrics.get("missing_ratio", 1.0))
    cov = float(metrics.get("coverage", 0.0))
    extra = float(metrics.get("extra_ratio", 0.0))

    et_dash, _ = dashed_like_count(et_lines)
    st_dash, _ = dashed_like_count(st_lines)
    dash_ratio = float((st_dash + 1) / (et_dash + 1))
    st_long = line_long_ratio(st_lines)

    hard_bad = (cov < 0.18 and sim < 0.28)
    ABS_DASH_MIN = 90
    ABS_DASH_RATIO = 2.6
    CAD_LONG_OK = 0.12
    noise_bad = (st_dash >= ABS_DASH_MIN) and (dash_ratio >= ABS_DASH_RATIO) and ((st_long < CAD_LONG_OK) or (sim < 0.15))
    extra_bad = (extra > 0.55 and miss > 0.40 and st_long < CAD_LONG_OK)
    noise_override = (sim <= 0.01) and (st_dash >= 70) and (dash_ratio >= 2.2)

    if hard_bad or noise_bad or extra_bad or noise_override:
        why = []
        if hard_bad: why.append("hard_bad")
        if noise_bad: why.append("noise_bad")
        if extra_bad: why.append("extra_bad")
        if noise_override: why.append("noise_override")
        dbg = {
            "mode": "GATED_TO_ZERO",
            "why": " + ".join(why),
            "similarity": round(sim, 4),
            "missing_ratio": round(miss, 4),
            "coverage": round(cov, 4),
            "extra_ratio": round(extra, 4),
            "et_dash": et_dash,
            "st_dash": st_dash,
            "dash_ratio": round(dash_ratio, 4),
            "st_long_ratio": round(st_long, 4),
        }
        return 0, dbg, np.zeros_like(et_lines), np.zeros_like(st_lines)

    et_dim, _ = dimension_like_mask(et_lines)
    st_dim, _ = dimension_like_mask(st_lines)

    et_dim_area = int((et_dim > 0).sum())
    st_dim_area = int((st_dim > 0).sum())
    et_small_cnt, _ = small_component_stats(et_dim)
    st_small_cnt, _ = small_component_stats(st_dim)

    if et_dim_area < 2000 and et_small_cnt < 12:
        pts = 12 if (st_dim_area > 1500 or st_small_cnt >= 10) else 8
        dbg = {
            "mode": "etalonda_dim_kam (soft)",
            "similarity": round(sim, 4),
            "missing_ratio": round(miss, 4),
            "coverage": round(cov, 4),
            "extra_ratio": round(extra, 4),
            "et_dim_area": et_dim_area,
            "st_dim_area": st_dim_area,
            "et_small_cnt": et_small_cnt,
            "st_small_cnt": st_small_cnt,
            "et_dash": et_dash,
            "st_dash": st_dash,
            "dash_ratio": round(dash_ratio, 4),
            "st_long_ratio": round(st_long, 4),
        }
        return pts, dbg, et_dim, st_dim

    area_ratio = float(np.clip(st_dim_area / max(et_dim_area, 1), 0.0, 1.2))
    cnt_ratio = float(np.clip(st_small_cnt / max(et_small_cnt, 1), 0.0, 1.2))
    score01 = 0.40 * min(area_ratio, 1.0) + 0.60 * min(cnt_ratio, 1.0)
    pts = int(np.clip(round(score01 * 12), 0, 12))

    dbg = {
        "mode": "ratio",
        "similarity": round(sim, 4),
        "missing_ratio": round(miss, 4),
        "coverage": round(cov, 4),
        "extra_ratio": round(extra, 4),
        "et_dim_area": et_dim_area,
        "st_dim_area": st_dim_area,
        "et_small_cnt": et_small_cnt,
        "st_small_cnt": st_small_cnt,
        "area_ratio": round(area_ratio, 4),
        "cnt_ratio": round(cnt_ratio, 4),
        "score01": round(score01, 4),
        "et_dash": et_dash,
        "st_dash": st_dash,
        "dash_ratio": round(dash_ratio, 4),
        "st_long_ratio": round(st_long, 4),
    }
    return pts, dbg, et_dim, st_dim


__all__ = [
    'remove_long_axes',
    'small_component_stats',
    'dimension_like_mask',
    'line_long_ratio',
    'score_dimension_12',
]
