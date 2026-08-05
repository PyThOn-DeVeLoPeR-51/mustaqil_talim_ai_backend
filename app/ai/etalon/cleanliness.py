"""Drawing cleanliness score (4 points)."""

from __future__ import annotations

import cv2
import numpy as np

from app.ai.etalon.projections import box_area
from app.ai.etalon.visible_view import _bin_to_lines, _remove_small_components

def _build_expected_region(ref_thr: np.ndarray):
    ref_lines = _bin_to_lines(ref_thr)
    h, w = ref_lines.shape[:2]

    region = cv2.morphologyEx(ref_lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)
    region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)), iterations=1)
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)), iterations=1)
    region = _remove_small_components(region, min_pixels=max(400, int(0.001 * h * w)))

    line_band = cv2.dilate(ref_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations=1)
    return region, line_band, ref_lines


def _extract_dirty_masks(ref_thr: np.ndarray, student_thr: np.ndarray):
    support_region, ref_band, ref_lines = _build_expected_region(ref_thr)
    student_lines = _bin_to_lines(student_thr)

    outside_noise = cv2.bitwise_and(student_lines, cv2.bitwise_not(support_region))
    outside_noise = _remove_small_components(outside_noise, min_pixels=4)

    inside_unexpected = cv2.bitwise_and(student_lines, cv2.bitwise_and(support_region, cv2.bitwise_not(ref_band)))
    num, labels, stats, _ = cv2.connectedComponentsWithStats((inside_unexpected > 0).astype(np.uint8), connectivity=8)

    blob_noise = np.zeros_like(student_lines)
    speckle_noise = np.zeros_like(student_lines)

    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < 3:
            continue
        box_area = max(bw * bh, 1)
        fill = area / box_area
        aspect = max(bw, bh) / max(min(bw, bh), 1)

        if area <= 18:
            speckle_noise[labels == i] = 255
            continue
        if area >= 20 and fill >= 0.28 and aspect <= 4.5:
            blob_noise[labels == i] = 255
            continue
        if 12 <= area <= 220 and aspect <= 8.0 and fill >= 0.10:
            blob_noise[labels == i] = 255

    num2, labels2, stats2, _ = cv2.connectedComponentsWithStats((outside_noise > 0).astype(np.uint8), connectivity=8)
    outside_speckles = np.zeros_like(student_lines)
    outside_lines = np.zeros_like(student_lines)

    for i in range(1, num2):
        area = stats2[i, cv2.CC_STAT_AREA]
        if area <= 16:
            outside_speckles[labels2 == i] = 255
        else:
            outside_lines[labels2 == i] = 255

    dirty_mask = cv2.bitwise_or(blob_noise, speckle_noise)
    dirty_mask = cv2.bitwise_or(dirty_mask, outside_lines)
    dirty_mask = cv2.bitwise_or(dirty_mask, outside_speckles)

    debug = {
        "support_region_px": int((support_region > 0).sum()),
        "ref_line_px": int((ref_lines > 0).sum()),
        "student_line_px": int((student_lines > 0).sum()),
        "outside_line_px": int((outside_lines > 0).sum()),
        "outside_speckle_px": int((outside_speckles > 0).sum()),
        "blob_px": int((blob_noise > 0).sum()),
        "speckle_px": int((speckle_noise > 0).sum()),
        "dirty_px": int((dirty_mask > 0).sum()),
    }

    masks = {
        "support_region": support_region,
        "ref_band": ref_band,
        "ref_lines": ref_lines,
        "student_lines": student_lines,
        "outside_lines": outside_lines,
        "outside_speckles": outside_speckles,
        "blob_noise": blob_noise,
        "speckle_noise": speckle_noise,
        "dirty_mask": dirty_mask,
    }
    return masks, debug


def _count_components(mask: np.ndarray, min_pixels: int = 1):
    num, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    cnt = 0
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_pixels:
            cnt += 1
    return cnt


def score_cleanliness_4(ref_thr: np.ndarray, student_thr: np.ndarray):
    masks, dbg = _extract_dirty_masks(ref_thr, student_thr)

    ref_line_px = max(dbg["ref_line_px"], 1)
    outside_line_px = dbg["outside_line_px"]
    outside_speckle_px = dbg["outside_speckle_px"]
    blob_px = dbg["blob_px"]
    speckle_px = dbg["speckle_px"]
    dirty_px = dbg["dirty_px"]

    outside_line_cnt = _count_components(masks["outside_lines"], min_pixels=6)
    outside_speckle_cnt = _count_components(masks["outside_speckles"], min_pixels=1)
    blob_cnt = _count_components(masks["blob_noise"], min_pixels=10)
    speckle_cnt = _count_components(masks["speckle_noise"], min_pixels=1)

    p_outside_lines = min(outside_line_px / (0.12 * ref_line_px), 1.0)
    p_blobs = min(blob_px / (0.10 * ref_line_px), 1.0)
    p_speckles = min((outside_speckle_cnt + speckle_cnt) / 45.0, 1.0)
    p_total_dirty = min(dirty_px / (0.18 * ref_line_px), 1.0)

    severity = 0.38 * p_outside_lines + 0.30 * p_blobs + 0.14 * p_speckles + 0.18 * p_total_dirty
    severity = float(np.clip(severity, 0.0, 1.0))
    cleanliness = 1.0 - severity

    if outside_line_px > 0.22 * ref_line_px or blob_px > 0.18 * ref_line_px:
        cap = 1
    elif dirty_px > 0.16 * ref_line_px:
        cap = 2
    elif dirty_px > 0.09 * ref_line_px or blob_cnt >= 6:
        cap = 3
    else:
        cap = 4

    raw = 4.0 * (cleanliness ** 0.85)
    score = int(np.clip(np.round(min(raw, cap)), 0, 4))

    metrics = {
        "cleanliness": float(cleanliness), "severity": float(severity), "outside_line_px": int(outside_line_px),
        "outside_speckle_px": int(outside_speckle_px), "blob_px": int(blob_px), "speckle_px": int(speckle_px),
        "dirty_px": int(dirty_px), "outside_line_cnt": int(outside_line_cnt),
        "outside_speckle_cnt": int(outside_speckle_cnt), "blob_cnt": int(blob_cnt), "speckle_cnt": int(speckle_cnt),
        "ref_line_px": int(ref_line_px), "student_line_px": int(dbg["student_line_px"]),
    }
    return score, metrics, masks, dbg


__all__ = [
    '_build_expected_region',
    '_extract_dirty_masks',
    '_count_components',
    'score_cleanliness_4',
]
