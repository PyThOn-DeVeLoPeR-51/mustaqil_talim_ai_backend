"""Dimension presence and placement score (15 points)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.ai.optional.config import CONFIG
from app.ai.optional.geometry import _box_center, _intersects, _overlap_ratio, merge_boxes_simple

def _make_direction_bands(role_box: Tuple[int, int, int, int], W: int, H: int, cfg: Dict[str, Any] = CONFIG) -> Dict[str, Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [int(v) for v in role_box]
    rw, rh = max(1, x2 - x1), max(1, y2 - y1)
    band_h = max(12, int(round(rh * cfg["dim_band_thickness_ratio"])))
    band_w = max(12, int(round(rw * cfg["dim_band_thickness_ratio"])))
    reach_x = max(12, int(round(rw * cfg["dim_band_reach_ratio"])))
    reach_y = max(12, int(round(rh * cfg["dim_band_reach_ratio"])))
    bands = {
        "top": (max(0, x1 - reach_x), max(0, y1 - band_h), min(W, x2 + reach_x), y1),
        "bottom": (max(0, x1 - reach_x), y2, min(W, x2 + reach_x), min(H, y2 + band_h)),
        "left": (max(0, x1 - band_w), max(0, y1 - reach_y), x1, min(H, y2 + reach_y)),
        "right": (x2, max(0, y1 - reach_y), min(W, x2 + band_w), min(H, y2 + reach_y)),
    }
    out = {}
    for k, b in bands.items():
        if b[2] - b[0] >= 6 and b[3] - b[1] >= 6:
            out[k] = b
    return out


def _find_text_groups_in_band(patch_bw: np.ndarray, cfg: Dict[str, Any] = CONFIG) -> Tuple[List[Tuple[int, int, int, int]], List[Dict[str, Any]]]:
    if patch_bw.size == 0:
        return [], []
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw2 = cv2.dilate(patch_bw, k, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((bw2 > 0).astype(np.uint8), connectivity=8)
    atom_boxes, atom_infos = [], []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 8 or area > 2500 or w < 3 or h < 4 or w > 140 or h > 55:
            continue
        aspect = w / max(1.0, h)
        if h > 3.8 * w and area < 60:
            continue
        if aspect > 12 and area < 100:
            continue
        atom_boxes.append((x, y, x + w, y + h))
        atom_infos.append({"box": (x, y, x + w, y + h), "area": int(area), "aspect": float(aspect)})
    if not atom_boxes:
        return [], []
    grouped = merge_boxes_simple(atom_boxes, gap=cfg["dim_text_group_gap"])
    infos = []
    for g in grouped:
        count = 0
        total_area = 0
        for a, info in zip(atom_boxes, atom_infos):
            if _intersects(g, a):
                count += 1
                total_area += info["area"]
        infos.append({"box": g, "atom_count": int(count), "area_sum": int(total_area)})
    return grouped, infos


def _extract_dim_lines_in_band(patch_bw: np.ndarray, patch_edges: np.ndarray, cfg: Dict[str, Any] = CONFIG) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]], List[Dict[str, Any]]]:
    if patch_bw.size == 0:
        return np.zeros_like(patch_bw), [], []
    h, w = patch_bw.shape[:2]
    base = max(8, int(min(h, w) * cfg["dim_line_min_len_ratio"]))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (base, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, base))
    hmap = cv2.morphologyEx(patch_bw, cv2.MORPH_OPEN, hk)
    vmap = cv2.morphologyEx(patch_bw, cv2.MORPH_OPEN, vk)
    line_map = cv2.bitwise_or(hmap, vmap)
    hough_boxes = []
    lines = cv2.HoughLinesP(
        patch_edges, 1, np.pi / 180, 16,
        minLineLength=max(8, int(min(h, w) * cfg["dim_hough_min_len_ratio"])),
        maxLineGap=cfg["dim_hough_max_gap"]
    )
    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            ln = float(np.hypot(x2 - x1, y2 - y1))
            if ln < base:
                continue
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if ang > 90:
                ang = 180 - ang
            if ang <= 12 or ang >= 78:
                bx1 = max(0, min(x1, x2) - cfg["dim_line_box_pad"])
                by1 = max(0, min(y1, y2) - cfg["dim_line_box_pad"])
                bx2 = min(w, max(x1, x2) + cfg["dim_line_box_pad"] + 1)
                by2 = min(h, max(y1, y2) + cfg["dim_line_box_pad"] + 1)
                hough_boxes.append((bx1, by1, bx2, by2))
                cv2.line(line_map, (x1, y1), (x2, y2), 255, 1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((line_map > 0).astype(np.uint8), connectivity=8)
    morph_boxes = []
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if area < 10:
            continue
        thick = min(ww, hh)
        long_side = max(ww, hh)
        if thick > cfg["dim_line_max_thickness"] or long_side < base:
            continue
        morph_boxes.append((x, y, x + ww, y + hh))
    merged = merge_boxes_simple(morph_boxes + hough_boxes, gap=6)
    infos = []
    for b in merged:
        orient = "h" if (b[2] - b[0]) >= (b[3] - b[1]) else "v"
        infos.append({"box": b, "length": int(max(b[2] - b[0], b[3] - b[1])), "thickness": int(min(b[2] - b[0], b[3] - b[1])), "orientation": orient})
    return line_map, merged, infos


def discover_dimension_clusters_in_role(bundle: Dict[str, np.ndarray], role_box: Tuple[int, int, int, int], cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    bw = bundle["bw"]
    edges = bundle["edges"]
    H, W = bw.shape[:2]
    bands = _make_direction_bands(role_box, W, H, cfg)
    all_clusters = []
    per_band_debug = []
    outer_box = role_box
    if bands:
        xs1 = [b[0] for b in bands.values()]
        ys1 = [b[1] for b in bands.values()]
        xs2 = [b[2] for b in bands.values()]
        ys2 = [b[3] for b in bands.values()]
        outer_box = (min(xs1), min(ys1), max(xs2), max(ys2))
    total_text = 0
    total_lines = 0
    for band_name, b in bands.items():
        bx1, by1, bx2, by2 = b
        patch_bw = bw[by1:by2, bx1:bx2].copy()
        patch_edges = edges[by1:by2, bx1:bx2].copy()
        text_boxes, _ = _find_text_groups_in_band(patch_bw, cfg)
        _, line_boxes, _ = _extract_dim_lines_in_band(patch_bw, patch_edges, cfg)
        total_text += len(text_boxes)
        total_lines += len(line_boxes)
        merged_clusters = merge_boxes_simple(text_boxes + line_boxes, gap=14)
        band_clusters = []
        for cbox in merged_clusters:
            text_count = sum(1 for tb in text_boxes if _intersects(cbox, tb))
            line_count = sum(1 for lb in line_boxes if _intersects(cbox, lb))
            strong = ((text_count >= 1 and line_count >= 1) or (text_count >= 2) or (line_count >= 2))
            if not strong:
                continue
            gbox = (bx1 + cbox[0], by1 + cbox[1], bx1 + cbox[2], by1 + cbox[3])
            band_clusters.append({
                "box_global": gbox, "text_count": int(text_count), "line_count": int(line_count),
                "outside_ratio": 1.0, "area": int((cbox[2] - cbox[0]) * (cbox[3] - cbox[1])), "band": band_name
            })
        all_clusters.extend(band_clusters)
        per_band_debug.append({"band": band_name, "band_box": [int(v) for v in b], "text_count": int(len(text_boxes)), "line_count": int(len(line_boxes)), "strong_clusters": int(len(band_clusters))})
    strong_clusters = all_clusters
    dim_present = len(strong_clusters) > 0 or (total_text >= 2 and total_lines >= 1)
    quality = 0.0
    if dim_present:
        t_score = min(1.0, total_text / 8.0)
        l_score = min(1.0, total_lines / 8.0)
        c_score = min(1.0, len(strong_clusters) / 4.0)
        quality = 0.35 * t_score + 0.35 * l_score + 0.30 * c_score
    return {
        "role_box": tuple(map(int, role_box)),
        "outer_box": tuple(map(int, outer_box)),
        "text_count": int(total_text),
        "line_count": int(total_lines),
        "cluster_count": int(len(all_clusters)),
        "strong_cluster_count": int(len(strong_clusters)),
        "clusters": strong_clusters,
        "dim_present": bool(dim_present),
        "quality": float(quality),
        "per_band_debug": per_band_debug,
    }


def _classify_cluster_side(role_box: Tuple[int, int, int, int], cluster_box: Tuple[int, int, int, int], cfg: Dict[str, Any] = CONFIG) -> Optional[str]:
    rx1, ry1, rx2, ry2 = role_box
    ccx, ccy = _box_center(cluster_box)
    rw, rh = max(1, rx2 - rx1), max(1, ry2 - ry1)
    inner_x1 = rx1 + rw * cfg["dim_center_reject_ratio"] * 0.5
    inner_x2 = rx2 - rw * cfg["dim_center_reject_ratio"] * 0.5
    inner_y1 = ry1 + rh * cfg["dim_center_reject_ratio"] * 0.5
    inner_y2 = ry2 - rh * cfg["dim_center_reject_ratio"] * 0.5
    if inner_x1 <= ccx <= inner_x2 and inner_y1 <= ccy <= inner_y2:
        return None
    vals = {"left": rx1 - ccx, "right": ccx - rx2, "top": ry1 - ccy, "bottom": ccy - ry2}
    side = max(vals, key=vals.get)
    if side in ["left", "right"]:
        if abs(vals[side]) > rw * (1.0 + cfg["dim_far_reject_ratio"]):
            return None
    else:
        if abs(vals[side]) > rh * (1.0 + cfg["dim_far_reject_ratio"]):
            return None
    return side


def _cluster_strength(cluster: Dict[str, Any]) -> float:
    score = 0.0
    score += 1.2 * min(2, cluster["text_count"])
    score += 1.0 * min(2, cluster["line_count"])
    if cluster.get("band") in ["top", "bottom", "left", "right"]:
        score += 0.4
    return score


def score_dimensions(role_result: Dict[str, Any], layout_result: Dict[str, Any], bundle: Dict[str, np.ndarray], cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    roles = role_result["roles"]
    title_box = layout_result.get("title_box", None)
    projection_boxes = [tuple(info["box"]) for info in layout_result["projection_infos"]]
    ordered_roles = []
    for name in ["front", "top", "side"]:
        info = roles.get(name)
        if info is not None and "box" in info and info["box"] is not None:
            ordered_roles.append((name, tuple(info["box"])))
    extras = roles.get("extra_orthographic", [])
    for i, info in enumerate(extras, 1):
        if info is not None and "box" in info and info["box"] is not None:
            ordered_roles.append((f"extra_{i}", tuple(info["box"])))
    per_role = []
    for role_name, role_box in ordered_roles:
        det = discover_dimension_clusters_in_role(bundle, role_box, cfg)
        filtered_clusters = []
        for c in det["clusters"]:
            gbox = tuple(c["box_global"])
            if title_box is not None and _overlap_ratio(gbox, title_box) > cfg["dim_titleblock_overlap_reject"]:
                continue
            bad_overlap = False
            for pb in projection_boxes:
                if pb == role_box:
                    continue
                if _overlap_ratio(gbox, pb) > cfg["dim_other_view_overlap_reject"]:
                    bad_overlap = True
                    break
            if bad_overlap:
                continue
            side = _classify_cluster_side(role_box, gbox, cfg)
            if side is None:
                continue
            strength = _cluster_strength(c)
            if strength < cfg["dim_min_cluster_score"]:
                continue
            c2 = dict(c)
            c2["side_of_role"] = side
            c2["strength"] = float(strength)
            filtered_clusters.append(c2)
        det["clusters"] = filtered_clusters
        det["cluster_count"] = int(len(filtered_clusters))
        det["strong_cluster_count"] = int(len(filtered_clusters))
        det["dim_present"] = bool(len(filtered_clusters) > 0 or (det["text_count"] >= 2 and det["line_count"] >= 2))
        if det["dim_present"]:
            t_score = min(1.0, det["text_count"] / 8.0)
            l_score = min(1.0, det["line_count"] / 8.0)
            c_score = min(1.0, len(filtered_clusters) / 4.0)
            det["quality"] = float(0.35 * t_score + 0.35 * l_score + 0.30 * c_score)
        else:
            det["quality"] = 0.0
        det["role"] = role_name
        per_role.append(det)
    dim_roles = sum(1 for d in per_role if d["dim_present"])
    s_coverage = 6 if dim_roles >= 3 else 4 if dim_roles == 2 else 2 if dim_roles == 1 else 0
    total_text = sum(d["text_count"] for d in per_role)
    s_text = 4 if total_text >= 8 else 3 if total_text >= 5 else 2 if total_text >= 3 else 1 if total_text >= 1 else 0
    total_lines = sum(d["line_count"] for d in per_role)
    total_strong_clusters = sum(d["strong_cluster_count"] for d in per_role)
    cluster_combo = total_lines + 2 * total_strong_clusters
    s_line_cluster = 3 if cluster_combo >= 12 else 2 if cluster_combo >= 7 else 1 if cluster_combo >= 3 else 0
    outside_ratios = [c["outside_ratio"] for d in per_role for c in d["clusters"]]
    mean_outside = float(np.mean(outside_ratios)) if outside_ratios else 0.0
    s_place = 2 if mean_outside >= 0.82 else 1 if mean_outside >= 0.62 else 0
    total = int(min(15, s_coverage + s_text + s_line_cluster + s_place))
    errors, warnings = [], []
    if dim_roles == 0:
        errors.append("Asosiy ko‘rinishlar atrofida o‘lcham evidence topilmadi")
    elif dim_roles == 1:
        warnings.append("O‘lchamlar faqat bitta ko‘rinishda aniq ko‘rindi")
    if total_text == 0:
        warnings.append("O‘lcham matni/raqamlariga oid evidence juda kam")
    if total_lines == 0:
        warnings.append("Dimension line evidence juda kam")
    if total_strong_clusters > 0 and mean_outside < 0.62:
        warnings.append("Ba’zi dimension cluster’lar role ichiga juda yaqin tushgan")
    summary = {
        "criterion": "O‘lchamlar mavjudligi va joylashuvi",
        "score": int(total),
        "max_score": 15,
        "subscores": {
            "role_coverage": int(s_coverage),
            "text_evidence": int(s_text),
            "line_cluster_evidence": int(s_line_cluster),
            "placement_quality": int(s_place),
        },
        "counts": {
            "roles_checked": int(len(per_role)),
            "roles_with_dimension": int(dim_roles),
            "total_text_components": int(total_text),
            "total_line_components": int(total_lines),
            "total_strong_clusters": int(total_strong_clusters),
        },
        "mean_outside_ratio": float(round(mean_outside, 4)),
        "errors": errors,
        "warnings": warnings,
        "per_role": [{
            "role": d["role"],
            "role_box": [int(v) for v in d["role_box"]],
            "outer_box": [int(v) for v in d["outer_box"]],
            "text_count": int(d["text_count"]),
            "line_count": int(d["line_count"]),
            "cluster_count": int(d["cluster_count"]),
            "strong_cluster_count": int(d["strong_cluster_count"]),
            "dim_present": bool(d["dim_present"]),
            "quality": float(round(d["quality"], 4)),
            "clusters": [{
                "box_global": [int(v) for v in c["box_global"]],
                "text_count": int(c["text_count"]),
                "line_count": int(c["line_count"]),
                "outside_ratio": float(round(c["outside_ratio"], 4)),
                "area": int(c["area"]),
                "band": str(c.get("band", "")),
                "side_of_role": str(c.get("side_of_role", "")),
                "strength": float(round(c.get("strength", 0.0), 4)),
            } for c in d["clusters"]],
        } for d in per_role],
    }
    return {"score": int(total), "max_score": 15, "summary": summary}


__all__ = [
    '_make_direction_bands',
    '_find_text_groups_in_band',
    '_extract_dim_lines_in_band',
    'discover_dimension_clusters_in_role',
    '_classify_cluster_side',
    '_cluster_strength',
    'score_dimensions',
]
