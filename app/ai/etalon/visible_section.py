"""Visible-view section and hatching comparison (15 points)."""

from __future__ import annotations

import cv2
import numpy as np

from app.ai.common.hough import normalize_hough_lines

from app.ai.etalon.visible_view import _bin_to_lines, _component_angle_and_aspect, _remove_small_components, crop_box

def _axial_angle_diff(a_deg: float, b_deg: float) -> float:
    d = abs(a_deg - b_deg) % 180.0
    return min(d, 180.0 - d)


def _nearest_allowed_hatch_angle(ang_deg: float):
    allowed = [45.0, 60.0, 120.0, 135.0]
    best = min(allowed, key=lambda t: _axial_angle_diff(ang_deg, t))
    return best, _axial_angle_diff(ang_deg, best)


def is_box_valid_for_current_pair(ref_thr: np.ndarray, box, min_line_px=250):
    if box is None or len(box) != 4:
        return False
    h, w = ref_thr.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x2 <= x1 or y2 <= y1:
        return False
    roi = crop_box(ref_thr, box, pad=8)
    lines = _bin_to_lines(roi)
    line_px = int((lines > 0).sum())
    return line_px >= min_line_px


def detect_short_diag_segments_visible_section(lines_mask: np.ndarray):
    lines = (lines_mask > 0).astype(np.uint8) * 255
    h, w = lines.shape[:2]
    cc_mask = np.zeros_like(lines)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)

    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if not (6 <= area <= 260):
            continue
        comp = np.uint8(labels[y:y + bh, x:x + bw] == i) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        ang, aspect = _component_angle_and_aspect(cnts[0])
        max_dim = max(bw, bh)
        allowed = [30, 45, 60, 120, 135, 150]
        is_diag_family = min(_axial_angle_diff(ang, t) for t in allowed) <= 12
        if is_diag_family and aspect >= 1.5 and 6 <= max_dim <= 90:
            cc_mask[labels == i] = 255

    hough_mask = np.zeros_like(lines)
    segs = cv2.HoughLinesP(lines, rho=1, theta=np.pi / 180, threshold=10,
                           minLineLength=max(8, int(0.035 * min(h, w))), maxLineGap=3)
    if segs is not None:
        allowed = [30, 45, 60, 120, 135, 150]
        for s in normalize_hough_lines(segs):
            x1, y1, x2, y2 = map(int, s)
            dx = x2 - x1; dy = y2 - y1
            L = float(np.hypot(dx, dy))
            if L < 6:
                continue
            ang = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0
            if min(_axial_angle_diff(ang, t) for t in allowed) <= 10:
                if L <= max(22, int(0.42 * max(h, w))):
                    cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 1)

    out = cv2.bitwise_or(cc_mask, hough_mask)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    out = _remove_small_components(out, min_pixels=6)
    return out


def _collect_parallel_hatch_lines(lines_mask: np.ndarray):
    lines = (lines_mask > 0).astype(np.uint8) * 255
    h, w = lines.shape[:2]
    segs = cv2.HoughLinesP(lines, rho=1, theta=np.pi / 180, threshold=10,
                           minLineLength=max(8, int(0.03 * min(h, w))), maxLineGap=3)
    hatch_lines = np.zeros_like(lines)

    if segs is None:
        debug = {"candidate_segments": 0, "accepted_segments": 0, "accepted_families": 0, "offset_bins": 0}
        return hatch_lines, debug

    rows = []
    for s in normalize_hough_lines(segs):
        x1, y1, x2, y2 = map(int, s)
        dx = x2 - x1; dy = y2 - y1
        L = float(np.hypot(dx, dy))
        if L < 6:
            continue
        ang = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0
        target, diff = _nearest_allowed_hatch_angle(ang)
        if diff > 8.5:
            continue
        if L > max(26, int(0.33 * max(h, w))):
            continue

        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)
        nr = np.deg2rad((target + 90.0) % 180.0)
        rho = mx * np.cos(nr) + my * np.sin(nr)

        rows.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "L": L, "target": target, "rho": rho})

    if not rows:
        debug = {"candidate_segments": 0, "accepted_segments": 0, "accepted_families": 0, "offset_bins": 0}
        return hatch_lines, debug

    bin_size = max(6.0, 0.018 * min(h, w))
    accepted_segments = 0
    accepted_families = 0
    total_bins = 0

    for fam in [45.0, 60.0, 120.0, 135.0]:
        fam_rows = [r for r in rows if abs(r["target"] - fam) < 1e-6]
        if len(fam_rows) < 4:
            continue

        bins = {}
        for r in fam_rows:
            k = int(np.round(r["rho"] / bin_size))
            bins.setdefault(k, []).append(r)

        rich_bins = []
        for k, items in bins.items():
            total_len = sum(z["L"] for z in items)
            if len(items) >= 1 and total_len >= 8:
                rich_bins.append((k, items))
        if len(rich_bins) < 4:
            continue

        lengths = [z["L"] for z in fam_rows]
        median_len = float(np.median(lengths))
        p85_len = float(np.percentile(lengths, 85))
        if median_len > 0.16 * max(h, w):
            continue
        if p85_len > 0.28 * max(h, w):
            continue

        accepted_families += 1
        total_bins += len(rich_bins)
        for _, items in rich_bins:
            for r in items:
                cv2.line(hatch_lines, (r["x1"], r["y1"]), (r["x2"], r["y2"]), 255, 1)
                accepted_segments += 1

    hatch_lines = cv2.morphologyEx(hatch_lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    hatch_lines = _remove_small_components(hatch_lines, min_pixels=6)

    debug = {
        "candidate_segments": int(len(rows)),
        "accepted_segments": int(accepted_segments),
        "accepted_families": int(accepted_families),
        "offset_bins": int(total_bins),
    }
    return hatch_lines, debug


def detect_hatch_region(binary_thr: np.ndarray, roi_mask=None):
    lines = _bin_to_lines(binary_thr)
    if roi_mask is not None:
        lines = cv2.bitwise_and(lines, roi_mask)

    hatch_lines, line_debug = _collect_parallel_hatch_lines(lines)
    if int((hatch_lines > 0).sum()) == 0:
        empty = np.zeros_like(lines)
        debug = {"short_diag_px": 0, "hatch_region_px": 0, "hatch_line_px": 0, "cluster_count": 0, **line_debug}
        return empty, empty, debug

    hatch_region = cv2.dilate(hatch_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)
    hatch_region = cv2.morphologyEx(hatch_region, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)), iterations=1)
    hatch_region = cv2.dilate(hatch_region, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((hatch_region > 0).astype(np.uint8), connectivity=8)
    filtered_region = np.zeros_like(hatch_region)
    cluster_count = 0

    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < 120:
            continue
        reg = np.uint8(labels == i) * 255
        inside_lines = cv2.bitwise_and(hatch_lines, reg)
        line_px = int((inside_lines > 0).sum())
        density = line_px / max(area, 1)
        if density >= 0.018:
            filtered_region[reg > 0] = 255
            cluster_count += 1

    filtered_region = cv2.morphologyEx(filtered_region, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)
    filtered_region = _remove_small_components(filtered_region, min_pixels=90)

    debug = {
        "short_diag_px": int((hatch_lines > 0).sum()),
        "hatch_region_px": int((filtered_region > 0).sum()),
        "hatch_line_px": int((hatch_lines > 0).sum()),
        "cluster_count": int(cluster_count),
        **line_debug
    }
    return filtered_region, hatch_lines, debug


def score_visible_section_15(ref_hatch_region: np.ndarray, student_hatch_region: np.ndarray):
    ref_bin = (ref_hatch_region > 0)
    st_bin = (student_hatch_region > 0)
    ref_area = int(ref_bin.sum())
    st_area = int(st_bin.sum())

    if ref_area < 120:
        metrics = {
            "presence_recall": 0.0, "presence_precision": 0.0 if st_area == 0 else float(min(1.0, st_area / 120.0)),
            "iou": 0.0, "area_ratio": float(st_area / max(ref_area, 1)), "area_balance": 0.0,
            "ref_area": int(ref_area), "student_area": int(st_area),
            "zero_reason": "reference_section_region_not_found_or_invalid_roi",
        }
        return 0, metrics

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    ref_band = cv2.dilate(ref_hatch_region, k, iterations=1)
    st_band = cv2.dilate(student_hatch_region, k, iterations=1)

    ref_found = int((ref_bin & (st_band > 0)).sum())
    st_correct = int((st_bin & (ref_band > 0)).sum()) if st_area > 0 else 0

    presence_recall = ref_found / max(ref_area, 1)
    presence_precision = st_correct / max(st_area, 1) if st_area > 0 else 0.0
    inter = int(((ref_band > 0) & (st_band > 0)).sum())
    union = int(((ref_band > 0) | (st_band > 0)).sum())
    iou = inter / max(union, 1)

    area_ratio = st_area / max(ref_area, 1)
    area_balance = min(area_ratio, 1.0 / max(area_ratio, 1e-6))
    area_balance = float(np.clip(area_balance, 0.0, 1.0))

    zero_reason = None
    if st_area < max(100, int(0.12 * ref_area)):
        zero_reason = "student_has_no_section_region"
    elif presence_recall < 0.20:
        zero_reason = "student_section_not_in_correct_place"

    if zero_reason is not None:
        metrics = {
            "presence_recall": float(presence_recall), "presence_precision": float(presence_precision),
            "iou": float(iou), "area_ratio": float(area_ratio), "area_balance": float(area_balance),
            "ref_area": int(ref_area), "student_area": int(st_area), "zero_reason": zero_reason,
        }
        return 0, metrics

    f1 = (2.0 * presence_recall * presence_precision) / max(presence_recall + presence_precision, 1e-9)
    quality = 0.46 * f1 + 0.26 * presence_recall + 0.16 * presence_precision + 0.12 * np.sqrt(max(iou, 0.0))
    quality = float(np.clip(quality, 0.0, 1.0))
    raw = 15.0 * (quality ** 0.82)

    cap = 15
    if presence_recall < 0.40 or presence_precision < 0.35:
        cap = 5
    elif presence_recall < 0.60 or presence_precision < 0.45:
        cap = 9
    elif presence_recall < 0.78 or presence_precision < 0.55:
        cap = 12
    elif presence_recall < 0.88 or presence_precision < 0.65:
        cap = 14

    if presence_recall >= 0.90 and presence_precision >= 0.58 and iou >= 0.48 and area_balance >= 0.45:
        cap = max(cap, 13)
    if presence_recall >= 0.93 and presence_precision >= 0.64 and iou >= 0.54 and area_balance >= 0.50:
        cap = max(cap, 15)

    score = int(np.clip(np.round(min(raw, cap)), 0, 15))
    metrics = {
        "presence_recall": float(presence_recall), "presence_precision": float(presence_precision), "f1": float(f1),
        "iou": float(iou), "quality": float(quality), "area_ratio": float(area_ratio), "area_balance": float(area_balance),
        "ref_area": int(ref_area), "student_area": int(st_area), "zero_reason": None,
    }
    return score, metrics


__all__ = [
    '_axial_angle_diff',
    '_nearest_allowed_hatch_angle',
    'is_box_valid_for_current_pair',
    'detect_short_diag_segments_visible_section',
    '_collect_parallel_hatch_lines',
    'detect_hatch_region',
    'score_visible_section_15',
]
