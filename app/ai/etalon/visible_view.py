"""Visible/isometric view comparison (24 points)."""

from __future__ import annotations

import cv2
import numpy as np

from app.ai.etalon.projections import box_area

def _bin_to_lines(thr_img: np.ndarray):
    return (((255 - thr_img) > 0).astype(np.uint8) * 255)


def _remove_small_components(mask: np.ndarray, min_pixels: int = 40):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_pixels:
            out[labels == i] = 255
    return out


def _component_angle_and_aspect(cnt):
    rect = cv2.minAreaRect(cnt)
    (_, _), (w, h), ang = rect
    if w < 1e-6 or h < 1e-6:
        return 0.0, 1.0
    if w < h:
        ang = ang + 90.0
    ang = ang % 180.0
    aspect = max(w, h) / max(min(w, h), 1e-6)
    return ang, aspect


def crop_box(img, box, pad=0):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1 - pad)); y1 = max(0, int(y1 - pad))
    x2 = min(w, int(x2 + pad)); y2 = min(h, int(y2 + pad))
    return img[y1:y2, x1:x2]


def build_roi_mask(shape, box, pad=18):
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1 - pad)); y1 = max(0, int(y1 - pad))
    x2 = min(w, int(x2 + pad)); y2 = min(h, int(y2 + pad))
    m = np.zeros((h, w), dtype=np.uint8)
    m[y1:y2, x1:x2] = 255
    return m


def skeletonize(mask: np.ndarray):
    img = (mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        done = (cv2.countNonZero(img) == 0)
    return skel


def auto_projection_boxes_from_etalon(ref_thr: np.ndarray, max_boxes=3):
    lines = _bin_to_lines(ref_thr)
    h, w = lines.shape[:2]
    work = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations=1)
    work = cv2.dilate(work, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
    work = _remove_small_components(work, min_pixels=max(350, int(0.0007 * h * w)))

    num, labels, stats, _ = cv2.connectedComponentsWithStats((work > 0).astype(np.uint8), connectivity=8)
    cands = []
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        box_area = bw * bh
        if area < max(350, int(0.0007 * h * w)): continue
        if bw < 0.08 * w and bh < 0.08 * h: continue
        score = float(area + 0.20 * box_area)
        cands.append((score, (x, y, x + bw, y + bh)))
    cands.sort(key=lambda z: z[0], reverse=True)

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / max(ua, 1)

    boxes = []
    for _, b in cands:
        if all(iou(b, k) < 0.35 for k in boxes):
            boxes.append(b)
        if len(boxes) >= max_boxes:
            break

    if not boxes:
        boxes = [(0, 0, w, h)]
    return boxes


def hough_angle_features(mask: np.ndarray):
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    segs = cv2.HoughLinesP(m, rho=1, theta=np.pi / 180, threshold=18,
                           minLineLength=max(14, int(0.06 * min(h, w))), maxLineGap=5)

    total_len = 0.0
    diag_len = 0.0
    long_diag_len = 0.0
    hv_len = 0.0
    seg_count = 0

    if segs is not None:
        for s in segs[:, 0]:
            x1, y1, x2, y2 = s
            dx = x2 - x1; dy = y2 - y1
            L = float(np.hypot(dx, dy))
            if L < 1:
                continue
            ang = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0
            total_len += L
            seg_count += 1

            is_hv = (ang <= 12) or (ang >= 168) or (78 <= ang <= 102)
            is_diag = (18 <= ang <= 72) or (108 <= ang <= 162)

            if is_hv:
                hv_len += L
            if is_diag:
                diag_len += L
                if L >= max(18, 0.10 * min(h, w)):
                    long_diag_len += L

    return {
        "total_len": float(total_len),
        "diag_ratio": float(diag_len / max(total_len, 1e-6)),
        "long_diag_ratio": float(long_diag_len / max(total_len, 1e-6)),
        "hv_ratio": float(hv_len / max(total_len, 1e-6)),
        "seg_count": int(seg_count),
    }


def choose_visible_box_from_etalon(ref_thr: np.ndarray, boxes):
    H, W = ref_thr.shape[:2]
    scored = []
    for b in boxes:
        crop = crop_box(ref_thr, b, pad=8)
        lines = _bin_to_lines(crop)
        lines = _remove_small_components(lines, min_pixels=20)
        feats = hough_angle_features(lines)
        x1, y1, x2, y2 = b
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area_ratio = (bw * bh) / max(H * W, 1)
        density = float((lines > 0).sum()) / max(bw * bh, 1)
        score = 1.8 * feats["long_diag_ratio"] + 1.0 * feats["diag_ratio"] + 0.45 * np.sqrt(area_ratio) + 0.20 * density
        scored.append({"box": b, "score": float(score), "area_ratio": float(area_ratio), "density": float(density), **feats})
    scored = sorted(scored, key=lambda z: z["score"], reverse=True)
    return scored[0]["box"], scored


def detect_text_like_mask(lines_mask: np.ndarray):
    num, labels, stats, _ = cv2.connectedComponentsWithStats((lines_mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(lines_mask)
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if 12 <= area <= 260 and bw <= 35 and bh <= 35:
            fill = area / max(bw * bh, 1)
            if 0.08 <= fill <= 0.75:
                out[labels == i] = 255
    return out


def detect_dashed_like_mask(lines_mask: np.ndarray):
    thin = cv2.erode(lines_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((thin > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(lines_mask)
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if not (6 <= area <= 160):
            continue
        max_dim = max(bw, bh)
        min_dim = max(min(bw, bh), 1)
        aspect = max_dim / min_dim
        if max_dim <= 30 and aspect >= 1.9:
            out[labels == i] = 255
    return out


def detect_short_diag_segments(lines_mask: np.ndarray):
    num, labels, stats, _ = cv2.connectedComponentsWithStats((lines_mask > 0).astype(np.uint8), connectivity=8)
    short_diag = np.zeros_like(lines_mask)

    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if not (8 <= area <= 150):
            continue
        comp = np.uint8(labels[y:y + bh, x:x + bw] == i) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        ang, aspect = _component_angle_and_aspect(cnts[0])
        is_diag = (20 <= ang <= 70) or (110 <= ang <= 160)
        max_dim = max(bw, bh)
        if is_diag and aspect >= 2.0 and 6 <= max_dim <= 34:
            short_diag[labels == i] = 255
    return short_diag


def detect_hatch_cluster_mask(lines_mask: np.ndarray):
    short_diag = detect_short_diag_segments(lines_mask)
    if int((short_diag > 0).sum()) == 0:
        return np.zeros_like(lines_mask), short_diag

    region = cv2.dilate(short_diag, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations=1)
    region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((region > 0).astype(np.uint8), connectivity=8)
    hatch_region = np.zeros_like(lines_mask)
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < 120:
            continue
        reg = np.uint8(labels == i) * 255
        inside_short = int(((short_diag > 0) & (reg > 0)).sum())
        density = inside_short / max(area, 1)
        if density >= 0.08:
            hatch_region[reg > 0] = 255
    hatch_lines = cv2.bitwise_and(short_diag, hatch_region)
    return hatch_lines, short_diag


def extract_structure_mask(binary_thr: np.ndarray, roi_mask=None):
    lines = _bin_to_lines(binary_thr)
    if roi_mask is not None:
        lines = cv2.bitwise_and(lines, roi_mask)

    text_mask = detect_text_like_mask(lines)
    dashed_mask = detect_dashed_like_mask(lines)
    hatch_mask, short_diag_mask = detect_hatch_cluster_mask(lines)

    remove_mask = cv2.bitwise_or(text_mask, dashed_mask)
    remove_mask = cv2.bitwise_or(remove_mask, hatch_mask)

    structure = lines.copy()
    structure[remove_mask > 0] = 0
    structure = cv2.morphologyEx(structure, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    structure = cv2.morphologyEx(structure, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    structure = _remove_small_components(structure, min_pixels=20)

    skel = skeletonize(structure)
    skel = _remove_small_components(skel, min_pixels=10)

    debug = {
        "text_px": int((text_mask > 0).sum()),
        "dashed_px": int((dashed_mask > 0).sum()),
        "short_diag_px": int((short_diag_mask > 0).sum()),
        "hatch_px": int((hatch_mask > 0).sum()),
        "structure_px": int((structure > 0).sum()),
        "skeleton_px": int((skel > 0).sum()),
    }
    return structure, skel, debug


def score_student_visible_stable_24(ref_skel: np.ndarray, student_skel: np.ndarray, tol_kernel=(13, 13)):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, tol_kernel)
    ref_band = cv2.dilate(ref_skel, k, iterations=1)
    student_band = cv2.dilate(student_skel, k, iterations=1)

    ref_bin = (ref_skel > 0)
    student_bin = (student_skel > 0)
    ref_area = int(ref_bin.sum())
    student_area = int(student_bin.sum())

    found = int((ref_bin & (student_band > 0)).sum())
    correct = int((student_bin & (ref_band > 0)).sum())

    completeness = found / max(ref_area, 1)
    correctness = correct / max(student_area, 1) if student_area > 0 else 0.0
    missing_ratio = 1.0 - completeness
    extra_ratio = 1.0 - correctness if student_area > 0 else 1.0

    inter = int(((ref_band > 0) & (student_band > 0)).sum())
    union = int(((ref_band > 0) | (student_band > 0)).sum())
    iou = inter / max(union, 1)

    area_ratio = student_area / max(ref_area, 1)
    area_balance = min(area_ratio, 1.0 / max(area_ratio, 1e-6))
    area_balance = float(np.clip(area_balance, 0.0, 1.0))

    ref_feats = hough_angle_features(ref_skel)
    student_feats = hough_angle_features(student_skel)

    zero_reason = None
    if student_area < max(80, int(0.15 * ref_area)):
        zero_reason = "student_visible_area_too_small"
    elif completeness < 0.18:
        zero_reason = "student_did_not_draw_visible_view"
    elif area_ratio < 0.20:
        zero_reason = "visible_view_area_ratio_too_small"
    elif ref_feats["diag_ratio"] >= 0.16 and student_feats["diag_ratio"] < 0.05:
        zero_reason = "student_has_no_visible_view_diagonal_structure"

    if zero_reason is not None:
        metrics = {
            "completeness": float(completeness), "correctness": float(correctness), "missing_ratio": float(missing_ratio),
            "extra_ratio": float(extra_ratio), "iou": float(iou), "area_ratio": float(area_ratio),
            "area_balance": float(area_balance), "ref_area": int(ref_area), "student_area": int(student_area),
            "ref_diag_ratio": float(ref_feats["diag_ratio"]), "student_diag_ratio": float(student_feats["diag_ratio"]),
            "zero_reason": zero_reason,
        }
        return 0, metrics

    f1 = (2.0 * completeness * correctness) / max(completeness + correctness, 1e-9)
    quality = 0.50 * f1 + 0.22 * completeness + 0.18 * correctness + 0.10 * np.sqrt(max(iou, 0.0))
    quality = float(np.clip(quality, 0.0, 1.0))
    raw = 24.0 * (quality ** 0.82)

    cap = 24
    if completeness < 0.40 or correctness < 0.40:
        cap = 6
    elif completeness < 0.58 or correctness < 0.58:
        cap = 12
    elif completeness < 0.72 or correctness < 0.72:
        cap = 18
    elif completeness < 0.84 or correctness < 0.84:
        cap = 21

    if completeness >= 0.90 and correctness >= 0.58 and iou >= 0.46 and area_balance >= 0.50:
        cap = max(cap, 20)
    if completeness >= 0.92 and correctness >= 0.60 and iou >= 0.50 and area_balance >= 0.55:
        cap = max(cap, 22)
    if completeness >= 0.95 and correctness >= 0.68 and iou >= 0.58 and area_balance >= 0.62:
        cap = 24

    score = int(np.clip(np.round(min(raw, cap)), 0, 24))
    metrics = {
        "completeness": float(completeness), "correctness": float(correctness), "missing_ratio": float(missing_ratio),
        "extra_ratio": float(extra_ratio), "f1": float(f1), "iou": float(iou), "quality": float(quality),
        "area_ratio": float(area_ratio), "area_balance": float(area_balance), "ref_area": int(ref_area),
        "student_area": int(student_area), "ref_diag_ratio": float(ref_feats["diag_ratio"]),
        "student_diag_ratio": float(student_feats["diag_ratio"]), "zero_reason": None,
    }
    return score, metrics


__all__ = [
    '_bin_to_lines',
    '_remove_small_components',
    '_component_angle_and_aspect',
    'crop_box',
    'build_roi_mask',
    'skeletonize',
    'auto_projection_boxes_from_etalon',
    'hough_angle_features',
    'choose_visible_box_from_etalon',
    'detect_text_like_mask',
    'detect_dashed_like_mask',
    'detect_short_diag_segments',
    'detect_hatch_cluster_mask',
    'extract_structure_mask',
    'score_student_visible_stable_24',
]
