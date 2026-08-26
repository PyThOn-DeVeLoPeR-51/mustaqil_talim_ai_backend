"""Projection-section analysis (10 points)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.ai.common.hough import normalize_hough_lines

from app.ai.etalon.config import STEP6_HYBRID_THR, STEP6_PER_PROJ

def step6_odd(x: int) -> int:
    return x if x % 2 == 1 else x + 1


def step6_ensure_gray(img):
    if img is None:
        raise ValueError("Input image is None")
    if len(img.shape) == 2:
        gray = img.copy()
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def step6_clamp01(x):
    return max(0.0, min(1.0, float(x)))


def step6_box_area(b):
    x1, y1, x2, y2 = map(int, b)
    return max(0, x2 - x1) * max(0, y2 - y1)


def step6_crop_box(img, box, pad_ratio=0.04):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    px = max(4, int((x2 - x1) * pad_ratio))
    py = max(4, int((y2 - y1) * pad_ratio))
    return img[max(0, y1 - py):min(H, y2 + py), max(0, x1 - px):min(W, x2 + px)].copy()


def step6_auto_canny(img, sigma=0.33):
    v = float(np.median(img))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    if upper <= lower:
        lower, upper = 30, 120
    return cv2.Canny(img, lower, upper, L2gradient=True)


def step6_cluster_positions(vals, min_gap):
    if len(vals) == 0:
        return []
    vals = sorted([float(v) for v in vals])
    clusters = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - np.mean(clusters[-1])) <= min_gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(np.mean(c)) for c in clusters]


def step6_draw_preview(clean_img, boxes, step6_out=None):
    if len(clean_img.shape) == 2:
        vis = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2BGR)
    else:
        vis = clean_img.copy()

    boxes = [tuple(map(int, b)) for b in boxes]
    for i, b in enumerate(boxes, start=1):
        x1, y1, x2, y2 = map(int, b)
        color = (255, 140, 0)
        label = f"P{i}"
        if step6_out is not None and i <= len(step6_out["results"]):
            ok = step6_out["results"][i - 1]["is_section"]
            sc = step6_out["results"][i - 1]["score"]
            color = (0, 180, 0) if ok else (0, 0, 255)
            label = f"P{i} | {'QIRQIM' if ok else 'NO'} | {sc:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return vis


def step6_ink_mask_robust(gray_or_bgr):
    gray = step6_ensure_gray(gray_or_bgr)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    block = step6_odd(max(21, int(min(gray.shape[:2]) * 0.10)))
    bw2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 7)

    def to_ink(bw):
        if np.mean(bw) < 127:
            bw = cv2.bitwise_not(bw)
        ink = cv2.bitwise_not(bw)
        ink = cv2.medianBlur(ink, 3)
        return ink

    ink1 = to_ink(bw1)
    ink2 = to_ink(bw2)

    area = gray.shape[0] * gray.shape[1]
    fill1 = float(np.count_nonzero(ink1)) / max(area, 1.0)
    fill2 = float(np.count_nonzero(ink2)) / max(area, 1.0)
    target = 0.08

    def score_fill(fill):
        if 0.006 <= fill <= 0.30:
            return 1.0 - abs(fill - target)
        return -abs(fill - target)

    ink = ink1 if score_fill(fill1) >= score_fill(fill2) else ink2
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    return ink


def step6_extract_hatch_features(roi_img, debug=False):
    gray = step6_ensure_gray(roi_img)
    h, w = gray.shape[:2]
    area = float(h * w)
    if min(h, w) < 26:
        return {"is_section": False, "diag_coverage": 0.0, "diag_components": 0, "diag_line_count": 0,
                "dom_line_count": 0, "dom_ratio": 0.0, "dom_angle": 0.0, "angle_std": 999.0,
                "fill_ratio": 0.0, "interior_share": 0.0, "stripe_count": 0, "gap_cv": 999.0,
                "median_gap_norm": 0.0, "mean_len_ratio": 0.0, "hybrid_score": -999.0,
                "mode_hint": "none", "target_angle": 0.0, "mask": None}

    inner = np.zeros_like(gray, dtype=np.uint8)
    mx = max(4, int(w * 0.10)); my = max(4, int(h * 0.10))
    inner[my:h - my, mx:w - mx] = 255

    ink = step6_ink_mask_robust(gray)
    ink = cv2.bitwise_and(ink, inner)
    fill_ratio = float(np.count_nonzero(ink)) / max(area, 1.0)
    ink_nonzero = max(np.count_nonzero(ink), 1)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    edges = step6_auto_canny(g)
    edges = cv2.bitwise_and(edges, inner)
    base = min(h, w)

    hk = step6_odd(max(9, int(base * 0.12)))
    hker = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    vker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, hk))
    long_h = cv2.morphologyEx(edges, cv2.MORPH_OPEN, hker)
    long_v = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vker)
    edges_work = cv2.subtract(edges, cv2.bitwise_or(long_h, long_v))
    edges_work = cv2.morphologyEx(edges_work, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    lines = cv2.HoughLinesP(edges_work, rho=1, theta=np.pi / 180,
                            threshold=max(10, int(base * 0.07)),
                            minLineLength=max(10, int(base * 0.08)),
                            maxLineGap=max(2, int(base * 0.02)))

    allowed_targets = [45.0, 60.0]
    angle_tol = 11.0
    candidates = []
    total_candidate_len = {45.0: 0.0, 60.0: 0.0}
    total_candidate_cnt = {45.0: 0, 60.0: 0}

    if lines is not None:
        for l in normalize_hough_lines(lines):
            x1, y1, x2, y2 = map(int, l)
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < max(10, int(base * 0.07)):
                continue
            ang = math.degrees(math.atan2(dy, dx)) % 180.0
            if ang > 90:
                ang = 180.0 - ang
            nearest = min(allowed_targets, key=lambda t: abs(ang - t))
            if abs(ang - nearest) <= angle_tol:
                mxp = 0.5 * (x1 + x2)
                myp = 0.5 * (y1 + y2)
                candidates.append((nearest, ang, length, mxp, myp, x1, y1, x2, y2))
                total_candidate_len[nearest] += length
                total_candidate_cnt[nearest] += 1

    if len(candidates) == 0:
        return {"is_section": False, "diag_coverage": 0.0, "diag_components": 0, "diag_line_count": 0,
                "dom_line_count": 0, "dom_ratio": 0.0, "dom_angle": 0.0, "angle_std": 999.0,
                "fill_ratio": float(fill_ratio), "interior_share": 0.0, "stripe_count": 0, "gap_cv": 999.0,
                "median_gap_norm": 0.0, "mean_len_ratio": 0.0, "hybrid_score": -999.0,
                "mode_hint": "weak", "target_angle": 0.0, "mask": edges_work if debug else None}

    target_angle = max(allowed_targets, key=lambda t: (total_candidate_cnt[t], total_candidate_len[t]))
    selected = [c for c in candidates if c[0] == target_angle]

    line_count = len(candidates)
    dom_line_count = len(selected)
    dom_ratio = float(dom_line_count) / max(line_count, 1)

    selected_angles = [c[1] for c in selected]
    selected_lengths = [c[2] for c in selected]
    selected_mid = [(c[3], c[4]) for c in selected]

    dom_angle = float(np.median(selected_angles)) if len(selected_angles) > 0 else 0.0
    angle_std = float(np.std(selected_angles)) if len(selected_angles) > 1 else 0.0
    mean_len_ratio = float(np.mean(selected_lengths)) / max(math.hypot(w, h), 1.0) if len(selected_lengths) > 0 else 0.0

    rad = math.radians(target_angle)
    nx, ny = -math.sin(rad), math.cos(rad)
    positions = [(mxp * nx + myp * ny) for (mxp, myp) in selected_mid]
    min_gap = max(3.0, base * 0.020)
    stripe_centers = step6_cluster_positions(positions, min_gap=min_gap)
    stripe_count = len(stripe_centers)

    if len(stripe_centers) >= 2:
        gaps = np.diff(sorted(stripe_centers))
        median_gap_norm = float(np.median(gaps)) / max(base, 1.0)
        gap_cv = float(np.std(gaps) / max(np.mean(gaps), 1e-6)) if len(gaps) > 1 else 0.0
    else:
        median_gap_norm = 0.0
        gap_cv = 999.0

    selected_mask = np.zeros_like(edges_work)
    for (_, _, _, _, _, x1, y1, x2, y2) in selected:
        cv2.line(selected_mask, (x1, y1), (x2, y2), 255, 1)

    diag_coverage = float(np.count_nonzero(selected_mask)) / max(area, 1.0)
    interior_share = float(np.count_nonzero(selected_mask)) / max(ink_nonzero, 1)

    ncc, lbl, st, _ = cv2.connectedComponentsWithStats(selected_mask, connectivity=8)
    comp_count = 0
    for i in range(1, ncc):
        x, y, ww, hh, aa = st[i]
        if aa >= max(6, int(area * 0.00010)) and max(ww, hh) >= max(5, int(base * 0.04)):
            comp_count += 1

    hybrid_score = 0.0
    if stripe_count >= 3: hybrid_score += 2.0
    if stripe_count >= 4: hybrid_score += 1.5
    if stripe_count >= 5: hybrid_score += 0.8
    hybrid_score += 2.2 * step6_clamp01((dom_ratio - 0.55) / 0.30)
    hybrid_score += 1.6 * step6_clamp01((12.0 - angle_std) / 12.0)
    hybrid_score += 1.2 * step6_clamp01((dom_line_count - 4) / 5.0)
    hybrid_score += 0.8 * step6_clamp01((interior_share - 0.02) / 0.08)
    if 0.008 <= median_gap_norm <= 0.22: hybrid_score += 0.8
    if gap_cv <= 1.25: hybrid_score += 0.7
    if stripe_count < 3: hybrid_score -= 4.0
    if dom_line_count < 4: hybrid_score -= 2.0
    if dom_ratio < 0.60: hybrid_score -= 1.6
    if mean_len_ratio > 0.80: hybrid_score -= 1.2

    cad_vote = (
        target_angle in (45.0, 60.0) and stripe_count >= 4 and dom_line_count >= 5 and dom_ratio >= 0.72 and
        angle_std <= 8.5 and gap_cv <= 1.25 and mean_len_ratio >= 0.06 and mean_len_ratio <= 0.75 and interior_share >= 0.02
    )
    sketch_vote = (
        target_angle in (45.0, 60.0) and stripe_count >= 3 and dom_line_count >= 4 and dom_ratio >= 0.62 and
        angle_std <= 12.0 and gap_cv <= 1.60 and mean_len_ratio >= 0.05 and mean_len_ratio <= 0.78 and interior_share >= 0.015
    )
    hybrid_vote = target_angle in (45.0, 60.0) and hybrid_score >= STEP6_HYBRID_THR and stripe_count >= 3
    is_section = bool(cad_vote or sketch_vote or hybrid_vote)

    if cad_vote:
        mode_hint = "cad"
    elif sketch_vote:
        mode_hint = "sketch"
    elif hybrid_vote:
        mode_hint = "hybrid"
    else:
        mode_hint = "weak"

    return {
        "is_section": is_section,
        "diag_coverage": float(diag_coverage),
        "diag_components": int(comp_count),
        "diag_line_count": int(line_count),
        "dom_line_count": int(dom_line_count),
        "dom_ratio": float(dom_ratio),
        "dom_angle": float(dom_angle),
        "angle_std": float(angle_std),
        "fill_ratio": float(fill_ratio),
        "interior_share": float(interior_share),
        "stripe_count": int(stripe_count),
        "gap_cv": float(gap_cv),
        "median_gap_norm": float(median_gap_norm),
        "mean_len_ratio": float(mean_len_ratio),
        "hybrid_score": float(hybrid_score),
        "mode_hint": mode_hint,
        "target_angle": float(target_angle),
        "mask": selected_mask if debug else None
    }


def step6_run_only_selected3(clean_img, selected_boxes, debug=False):
    boxes = [tuple(map(int, b)) for b in selected_boxes if len(b) == 4 and step6_box_area(b) > 0]
    results, flags, scores = [], [], []
    for i, b in enumerate(boxes, start=1):
        roi = step6_crop_box(clean_img, b, pad_ratio=0.04)
        info = step6_extract_hatch_features(roi, debug=debug)
        score = STEP6_PER_PROJ if info["is_section"] else 0.0
        info["proj_index"] = i
        info["box"] = tuple(map(int, b))
        info["score"] = float(score)
        results.append(info)
        flags.append(bool(info["is_section"]))
        scores.append(float(score))
    return {"flags": flags, "scores": scores, "total_score": float(sum(scores)), "results": results}


__all__ = [
    'step6_odd',
    'step6_ensure_gray',
    'step6_clamp01',
    'step6_box_area',
    'step6_crop_box',
    'step6_auto_canny',
    'step6_cluster_positions',
    'step6_draw_preview',
    'step6_ink_mask_robust',
    'step6_extract_hatch_features',
    'step6_run_only_selected3',
]
