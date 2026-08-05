"""Title-block, drawing-zone, and projection-region discovery."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.ai.optional.config import CONFIG
from app.ai.optional.geometry import _smooth_1d, box_area, clip_box, merge_boxes_iter

def extract_hv_maps(bw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = bw.shape[:2]
    h_len = max(25, w // 20)
    v_len = max(25, h // 20)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    hmap = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    vmap = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
    hv = cv2.bitwise_or(hmap, vmap)
    return hmap, vmap, hv


def structure_mask_for_grouping(bw: np.ndarray, close_iter: int = 2, dilate_iter: int = 2) -> np.ndarray:
    small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(bw, cv2.MORPH_OPEN, small)
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    merged = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k1, iterations=close_iter)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    return cv2.dilate(merged, k2, iterations=dilate_iter)


def refine_title_block_box_safe(bundle: Dict[str, np.ndarray], box: Tuple[int, int, int, int], cfg: Dict[str, Any]) -> Tuple[int, int, int, int]:
    bw = bundle["bw"]
    H, W = bw.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    patch = bw[y1:y2, x1:x2]
    if patch.size == 0:
        return box
    ph, pw = patch.shape[:2]
    if ph < 20 or pw < 20:
        return box
    _, _, hv = extract_hv_maps(patch)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    hv = cv2.morphologyEx(hv, cv2.MORPH_CLOSE, k, iterations=1)
    row_occ = (hv > 0).mean(axis=1)
    col_occ = (hv > 0).mean(axis=0)
    row_s = _smooth_1d(row_occ, 9)
    col_s = _smooth_1d(col_occ, 9)
    row_thr = max(0.015, float(row_s.max()) * cfg["tb_safe_row_thr_ratio"])
    search_from = int(ph * 0.12)
    row_idx = np.where(row_s[search_from:] >= row_thr)[0]
    top_rel = search_from + int(row_idx[0]) if len(row_idx) > 0 else 0
    col_thr = max(0.015, float(col_s.max()) * cfg["tb_safe_col_thr_ratio"])
    col_idx = np.where(col_s >= col_thr)[0]
    left_rel = int(col_idx[0]) if len(col_idx) > 0 else 0
    pad = int(cfg["tb_safe_pad"])
    nx1 = x1 + max(0, left_rel - pad)
    ny1 = y1 + max(0, top_rel - pad)
    refined = clip_box((nx1, ny1, x2, y2), W, H)
    if refined is None:
        return box
    old_w, old_h = max(1, x2 - x1), max(1, y2 - y1)
    new_w, new_h = max(1, refined[2] - refined[0]), max(1, refined[3] - refined[1])
    if new_w < old_w * cfg["tb_safe_min_w_keep"] or new_h < old_h * cfg["tb_safe_min_h_keep"]:
        return box
    return refined


def find_title_block_candidate(bundle: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> Tuple[Optional[Tuple[int, int, int, int]], Dict[str, Any]]:
    bw = bundle["bw"]
    h, w = bw.shape[:2]
    y0 = int(h * (1.0 - cfg["tb_bottom_ratio"]))
    x0 = int(w * (1.0 - cfg["tb_right_ratio"]))
    roi = bw[y0:h, x0:w]
    _, _, hv = extract_hv_maps(roi)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    hv2 = cv2.morphologyEx(hv, cv2.MORPH_CLOSE, k, iterations=2)
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats((hv2 > 0).astype(np.uint8), connectivity=8)
    candidates = []
    full_area = h * w
    for i in range(1, nlabels):
        x, y, ww, hh, area = stats[i]
        if ww < 40 or hh < 25:
            continue
        box = (x0 + x, y0 + y, x0 + x + ww, y0 + y + hh)
        area_ratio = box_area(box) / (full_area + 1e-6)
        if area_ratio < cfg["tb_min_area_ratio"] or area_ratio > cfg["tb_max_area_ratio"]:
            continue
        patch = bw[box[1]:box[3], box[0]:box[2]]
        fill_ratio = float((patch > 0).mean())
        if fill_ratio < cfg["tb_min_fill_ratio"]:
            continue
        hv_patch = hv2[y:y + hh, x:x + ww]
        hv_fill = float((hv_patch > 0).mean())
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        right_bonus = cx / w
        bottom_bonus = cy / h
        h_ratio = hh / max(1.0, h)
        tall_penalty = max(0.0, h_ratio - 0.26)
        score = 2.5 * hv_fill + 0.8 * fill_ratio + 1.0 * right_bonus + 1.0 * bottom_bonus - 2.0 * tall_penalty
        candidates.append({
            "box": box,
            "score": float(score),
            "fill_ratio": float(fill_ratio),
            "hv_fill": float(hv_fill),
            "area_ratio": float(area_ratio),
            "h_ratio": float(h_ratio),
        })
    if not candidates:
        return None, {"found": False, "reason": "no_candidate"}
    best = sorted(candidates, key=lambda d: d["score"], reverse=True)[0]
    refined_box = refine_title_block_box_safe(bundle, best["box"], cfg)
    return refined_box, {
        "found": True,
        "score": float(round(best["score"], 4)),
        "fill_ratio": float(round(best["fill_ratio"], 4)),
        "hv_fill": float(round(best["hv_fill"], 4)),
        "area_ratio": float(round(best["area_ratio"], 4)),
        "h_ratio": float(round(best["h_ratio"], 4)),
    }


def subtract_box_from_region(region_box: Tuple[int, int, int, int], sub_box: Tuple[int, int, int, int], pad: int = 12) -> Tuple[int, int, int, int]:
    rx1, ry1, rx2, ry2 = region_box
    sx1, sy1, sx2, sy2 = sub_box
    top_region = (rx1, ry1, rx2, max(ry1, sy1 - pad))
    right_region = (rx1, ry1, max(rx1, sx1 - pad), ry2)
    return top_region if box_area(top_region) >= box_area(right_region) else right_region


def get_drawing_zone(bundle: Dict[str, np.ndarray], title_block_box: Optional[Tuple[int, int, int, int]], cfg: Dict[str, Any]) -> Tuple[int, int, int, int]:
    h, w = bundle["bw"].shape[:2]
    full_region = (0, 0, w, h)
    pad = cfg["draw_zone_pad"]
    if title_block_box is None:
        return (pad, pad, w - pad, h - pad)
    dz = subtract_box_from_region(full_region, title_block_box, pad=pad)
    dz = clip_box(dz, w, h)
    if dz is None or box_area(dz) < 1000:
        return (pad, pad, w - pad, h - pad)
    return dz


def estimate_region_orientation_type(edges_patch: np.ndarray) -> Dict[str, Any]:
    lines = cv2.HoughLinesP(
        edges_patch, 1, np.pi / 180, 40,
        minLineLength=max(20, min(edges_patch.shape[:2]) // 6), maxLineGap=10
    )
    if lines is None:
        return {"type": "unknown", "hv_ratio": 0.0, "diag_ratio": 0.0, "line_count": 0}
    hv_len = 0.0
    diag_len = 0.0
    total = 0.0
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 1:
            continue
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if ang > 90:
            ang = 180 - ang
        total += length
        if ang <= 15 or ang >= 75:
            hv_len += length
        elif 25 <= ang <= 65:
            diag_len += length
    hv_ratio = hv_len / (total + 1e-6)
    diag_ratio = diag_len / (total + 1e-6)
    return {
        "type": "diag_dominant" if diag_ratio > hv_ratio + 0.12 else "hv_dominant",
        "hv_ratio": float(hv_ratio),
        "diag_ratio": float(diag_ratio),
        "line_count": int(len(lines)),
    }


def find_projection_candidates(bundle: Dict[str, np.ndarray], drawing_zone_box: Tuple[int, int, int, int], cfg: Dict[str, Any]) -> Tuple[List[Tuple[int, int, int, int]], List[Dict[str, Any]], np.ndarray]:
    bw = bundle["bw"]
    edges = bundle["edges"]
    x1, y1, x2, y2 = drawing_zone_box
    zone_bw = bw[y1:y2, x1:x2]
    grouped = structure_mask_for_grouping(zone_bw, cfg["proj_close_iter"], cfg["proj_dilate_iter"])
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats((grouped > 0).astype(np.uint8), connectivity=8)
    zone_area = max(1, (y2 - y1) * (x2 - x1))
    boxes, infos = [], []
    for i in range(1, nlabels):
        xx, yy, ww, hh, area = stats[i]
        if ww < cfg["proj_min_w"] or hh < cfg["proj_min_h"]:
            continue
        box = (x1 + xx, y1 + yy, x1 + xx + ww, y1 + yy + hh)
        area_ratio = box_area(box) / zone_area
        if area_ratio < cfg["proj_min_area_ratio"] or area_ratio > cfg["proj_max_area_ratio"]:
            continue
        patch_bw = bw[box[1]:box[3], box[0]:box[2]]
        fill_ratio = float((patch_bw > 0).mean())
        if fill_ratio < 0.01:
            continue
        orient = estimate_region_orientation_type(edges[box[1]:box[3], box[0]:box[2]])
        boxes.append(box)
        infos.append({
            "box": box,
            "area_ratio": float(area_ratio),
            "fill_ratio": float(fill_ratio),
            "type": orient["type"],
            "hv_ratio": orient["hv_ratio"],
            "diag_ratio": orient["diag_ratio"],
            "line_count": orient["line_count"],
        })
    if not boxes:
        return [], [], grouped
    merged = merge_boxes_iter(boxes, cfg["merge_iou_thr"], cfg["merge_gap_px"])
    merged_infos = []
    for box in merged:
        patch_bw = bw[box[1]:box[3], box[0]:box[2]]
        fill_ratio = float((patch_bw > 0).mean())
        orient = estimate_region_orientation_type(edges[box[1]:box[3], box[0]:box[2]])
        merged_infos.append({
            "box": box,
            "area_ratio": float(box_area(box) / zone_area),
            "fill_ratio": float(fill_ratio),
            "type": orient["type"],
            "hv_ratio": orient["hv_ratio"],
            "diag_ratio": orient["diag_ratio"],
            "line_count": orient["line_count"],
        })
    merged_infos = sorted(merged_infos, key=lambda d: (d["box"][1], d["box"][0]))
    return [d["box"] for d in merged_infos], merged_infos, grouped


def discover_layout(bundle: Dict[str, np.ndarray], cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    rgb = bundle["rgb"]
    h, w = rgb.shape[:2]
    title_box, tb_info = find_title_block_candidate(bundle, cfg)
    drawing_zone = get_drawing_zone(bundle, title_box, cfg)
    proj_boxes, proj_infos, grouped = find_projection_candidates(bundle, drawing_zone, cfg)
    meta = {
        "image_shape": [int(h), int(w)],
        "title_block_found": bool(title_box is not None),
        "title_block_box": [int(v) for v in title_box] if title_box is not None else None,
        "title_block_info": tb_info,
        "drawing_zone": [int(v) for v in drawing_zone],
        "projection_count": int(len(proj_boxes)),
        "projection_types": [str(d["type"]) for d in proj_infos],
    }
    return {
        "title_box": title_box,
        "drawing_zone": drawing_zone,
        "projection_boxes": proj_boxes,
        "projection_infos": proj_infos,
        "grouped_mask": grouped,
        "meta": meta,
    }


__all__ = [
    'extract_hv_maps',
    'structure_mask_for_grouping',
    'refine_title_block_box_safe',
    'find_title_block_candidate',
    'subtract_box_from_region',
    'get_drawing_zone',
    'estimate_region_orientation_type',
    'find_projection_candidates',
    'discover_layout',
]
