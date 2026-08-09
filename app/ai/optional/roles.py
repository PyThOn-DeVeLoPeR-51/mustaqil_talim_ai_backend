"""Front, top, side, and isometric role assignment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.ai.optional.config import CONFIG
from app.ai.optional.geometry import box_area

def b_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def x_overlap_ratio(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    denom = max(1, min(a[2] - a[0], b[2] - b[0]))
    return inter / denom


def y_overlap_ratio(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    inter = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    denom = max(1, min(a[3] - a[1], b[3] - b[1]))
    return inter / denom


def center_dist(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay = b_center(a)
    bx, by = b_center(b)
    return float(np.hypot(ax - bx, ay - by))


def choose_isometric_candidate(proj_infos: List[Dict[str, Any]], drawing_zone_box: Tuple[int, int, int, int], cfg: Dict[str, Any]) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    if not proj_infos:
        return None, None
    zx1, zy1, zx2, zy2 = drawing_zone_box
    zarea = max(1, (zx2 - zx1) * (zy2 - zy1))
    candidates = []
    for i, info in enumerate(proj_infos):
        box = info["box"]
        area_ratio = box_area(box) / zarea
        diag_ratio = float(info.get("diag_ratio", 0.0))
        hv_ratio = float(info.get("hv_ratio", 0.0))
        dists = [center_dist(box, other["box"]) for j, other in enumerate(proj_infos) if i != j]
        far_score = float(np.mean(dists)) if dists else 0.0
        diag_bonus = diag_ratio - hv_ratio
        score = (
            cfg["iso_diag_bonus"] * diag_bonus +
            cfg["iso_area_bonus"] * area_ratio +
            cfg["iso_far_bonus"] * (far_score / (max(zx2 - zx1, zy2 - zy1) + 1e-6))
        )
        if info.get("type") == "diag_dominant":
            score += 0.35
        candidates.append({
            "index": i,
            "box": box,
            "score": float(score),
            "diag_ratio": diag_ratio,
            "hv_ratio": hv_ratio,
            "area_ratio": area_ratio,
            "far_score": far_score,
            "type": info.get("type", "unknown"),
        })
    best = sorted(candidates, key=lambda d: d["score"], reverse=True)[0]
    if best["diag_ratio"] < 0.12 and best["type"] != "diag_dominant":
        return None, None
    return best["index"], best


def choose_front_view(orth_infos: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    if not orth_infos:
        return None, None
    candidates = []
    for i, info in enumerate(orth_infos):
        box = info["box"]
        area = box_area(box)
        support = 0.0
        cx, cy = b_center(box)
        for j, other in enumerate(orth_infos):
            if i == j:
                continue
            ocx, ocy = b_center(other["box"])
            xov = x_overlap_ratio(box, other["box"])
            yov = y_overlap_ratio(box, other["box"])
            if xov >= cfg["assign_min_x_overlap"]:
                if ocy < cy:
                    support += 1.0
                elif ocy > cy:
                    support += 0.8
            if yov >= cfg["assign_min_y_overlap"]:
                if ocx < cx or ocx > cx:
                    support += 0.9
        score = cfg["front_area_bonus"] * area + cfg["front_neighbor_bonus"] * support * 10000.0
        candidates.append({"index": i, "box": box, "score": float(score), "support": float(support), "area": float(area)})
    best = sorted(candidates, key=lambda d: d["score"], reverse=True)[0]
    return best["index"], best


def assign_top_view(front_box: Optional[Tuple[int, int, int, int]], other_infos: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    if front_box is None:
        return None, None
    prefer_below = (cfg.get("projection_system", "first_angle") == "first_angle")
    fcx, fcy = b_center(front_box)
    cands = []
    for i, info in enumerate(other_infos):
        box = info["box"]
        cx, cy = b_center(box)
        xov = x_overlap_ratio(front_box, box)
        if xov < cfg["assign_min_x_overlap"]:
            continue
        dy = cy - fcy
        dx = abs(cx - fcx)
        direction_bonus = cfg["top_direction_bonus"] if ((prefer_below and dy > 0) or ((not prefer_below) and dy < 0)) else 0.15
        vertical_gap = abs(dy)
        score = 2.5 * xov + 1.5 * direction_bonus - 0.0025 * dx - 0.0015 * vertical_gap
        cands.append({
            "index": i, "box": box, "score": float(score), "x_overlap": float(xov), "dy": float(dy),
            "preferred_direction": bool((prefer_below and dy > 0) or ((not prefer_below) and dy < 0))
        })
    if not cands:
        return None, None
    best = sorted(cands, key=lambda d: d["score"], reverse=True)[0]
    return best["index"], best


def assign_side_view(front_box: Optional[Tuple[int, int, int, int]], other_infos: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    if front_box is None:
        return None, None
    fcx, fcy = b_center(front_box)
    cands = []
    for i, info in enumerate(other_infos):
        box = info["box"]
        cx, cy = b_center(box)
        yov = y_overlap_ratio(front_box, box)
        if yov < cfg["assign_min_y_overlap"]:
            continue
        dx = cx - fcx
        dy = abs(cy - fcy)
        if abs(dx) < 5:
            continue
        score = 2.5 * yov + 1.0 * cfg["side_direction_bonus"] - 0.0015 * dy - 0.0010 * abs(dx)
        cands.append({
            "index": i, "box": box, "score": float(score), "y_overlap": float(yov),
            "dx": float(dx), "side": "right" if dx > 0 else "left"
        })
    if not cands:
        return None, None
    best = sorted(cands, key=lambda d: d["score"], reverse=True)[0]
    return best["index"], best


def analyze_projection_roles(layout_result: Dict[str, Any], cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    proj_infos = layout_result["projection_infos"]
    drawing_zone = layout_result["drawing_zone"]
    if not proj_infos:
        return {
            "isometric": None,
            "orthographic": [],
            "roles": {},
            "meta": {"projection_count": 0, "orthographic_count": 0, "has_isometric": False},
            "debug": {},
        }
    iso_idx, iso_meta = choose_isometric_candidate(proj_infos, drawing_zone, cfg)
    orth_infos, iso_info = [], None
    for i, info in enumerate(proj_infos):
        if iso_idx is not None and i == iso_idx:
            iso_info = info
        else:
            orth_infos.append(info)
    front_idx, front_meta = choose_front_view(orth_infos, cfg)
    front_info = orth_infos[front_idx] if front_idx is not None else None
    remaining_for_top = [info for i, info in enumerate(orth_infos) if i != front_idx]
    top_idx_local, top_meta = assign_top_view(front_info["box"] if front_info else None, remaining_for_top, cfg)
    top_info = remaining_for_top[top_idx_local] if top_idx_local is not None else None
    remaining_for_side = []
    for info in orth_infos:
        if front_info is not None and info["box"] == front_info["box"]:
            continue
        if top_info is not None and info["box"] == top_info["box"]:
            continue
        remaining_for_side.append(info)
    side_idx_local, side_meta = assign_side_view(front_info["box"] if front_info else None, remaining_for_side, cfg)
    side_info = remaining_for_side[side_idx_local] if side_idx_local is not None else None
    used_boxes = set()
    for item in [front_info, top_info, side_info]:
        if item is not None:
            used_boxes.add(tuple(item["box"]))
    extra_orth = [info for info in orth_infos if tuple(info["box"]) not in used_boxes]
    meta = {
        "projection_count": int(len(proj_infos)),
        "orthographic_count": int(len(orth_infos)),
        "has_isometric": bool(iso_info is not None),
        "projection_system": str(cfg.get("projection_system", "unknown")),
        "isometric_box": [int(v) for v in iso_info["box"]] if iso_info is not None else None,
        "front_box": [int(v) for v in front_info["box"]] if front_info is not None else None,
        "top_box": [int(v) for v in top_info["box"]] if top_info is not None else None,
        "side_box": [int(v) for v in side_info["box"]] if side_info is not None else None,
        "extra_orthographic_count": int(len(extra_orth)),
    }
    return {
        "isometric": iso_info,
        "orthographic": orth_infos,
        "roles": {"isometric": iso_info, "front": front_info, "top": top_info, "side": side_info, "extra_orthographic": extra_orth},
        "meta": meta,
        "debug": {"iso_meta": iso_meta, "front_meta": front_meta, "top_meta": top_meta, "side_meta": side_meta},
    }


__all__ = [
    'b_center',
    'x_overlap_ratio',
    'y_overlap_ratio',
    'center_dist',
    'choose_isometric_candidate',
    'choose_front_view',
    'assign_top_view',
    'assign_side_view',
    'analyze_projection_roles',
]
