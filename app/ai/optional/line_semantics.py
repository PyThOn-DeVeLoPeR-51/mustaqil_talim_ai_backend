"""Line semantics score (15 points)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from app.ai.optional.config import CONFIG

def crop_role_arrays(bundle: Dict[str, np.ndarray], box: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = [int(v) for v in box]
    return (
        bundle["rgb"][y1:y2, x1:x2].copy(),
        bundle["gray_norm"][y1:y2, x1:x2].copy(),
        bundle["bw"][y1:y2, x1:x2].copy(),
        bundle["edges"][y1:y2, x1:x2].copy(),
    )


def acute_deg(angle_deg: float) -> float:
    a = abs(angle_deg) % 180.0
    return 180 - a if a > 90 else a


def line_len(x1: int, y1: int, x2: int, y2: int) -> float:
    return float(np.hypot(x2 - x1, y2 - y1))


def detect_visible_contours_in_patch(edges_patch: np.ndarray, cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    h, w = edges_patch.shape[:2]
    pmin = max(1, min(h, w))
    min_len = max(10, int(pmin * cfg["visible_min_len_ratio"]))
    lines = cv2.HoughLinesP(edges_patch, 1, np.pi / 180, cfg["visible_hough_threshold"], minLineLength=min_len, maxLineGap=8)
    if lines is None:
        return {"present": False, "count": 0, "total_length_ratio": 0.0}
    kept = []
    total_len = 0.0
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        ln = line_len(x1, y1, x2, y2)
        ang = acute_deg(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if ang <= 18 or ang >= 72:
            kept.append((x1, y1, x2, y2))
            total_len += ln
    total_length_ratio = total_len / max(1.0, pmin)
    present = len(kept) >= 4 and total_length_ratio >= 2.0
    return {"present": bool(present), "count": int(len(kept)), "total_length_ratio": float(total_length_ratio), "segments": kept}


def _cluster_axis_values(vals: List[float], tol: int = 5) -> List[List[float]]:
    vals = sorted(vals)
    groups: List[List[float]] = []
    for v in vals:
        if not groups:
            groups.append([v])
        else:
            if abs(v - np.mean(groups[-1])) <= tol:
                groups[-1].append(v)
            else:
                groups.append([v])
    return groups


def _group_short_hv_segments(lines: List[Tuple[int, int, int, int]], patch_shape: Tuple[int, int], cfg: Dict[str, Any] = CONFIG) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    h, w = patch_shape[:2]
    pmin = max(1, min(h, w))
    min_len = max(5, int(pmin * cfg["dash_min_len_ratio"]))
    max_len = max(min_len + 2, int(pmin * cfg["dash_max_len_ratio"]))
    hsegs, vsegs = [], []
    for line in lines:
        x1, y1, x2, y2 = line
        ln = line_len(x1, y1, x2, y2)
        if ln < min_len or ln > max_len:
            continue
        ang = acute_deg(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if ang <= 14:
            hsegs.append({"coord": (y1 + y2) / 2.0, "a1": min(x1, x2), "a2": max(x1, x2), "len": ln})
        elif ang >= 76:
            vsegs.append({"coord": (x1 + x2) / 2.0, "a1": min(y1, y2), "a2": max(y1, y2), "len": ln})
    return hsegs, vsegs


def _build_dashed_sequences(seg_list: List[Dict[str, Any]], axis_tol: int, patch_span: int, cfg: Dict[str, Any] = CONFIG) -> List[Dict[str, Any]]:
    if not seg_list:
        return []
    coords = [s["coord"] for s in seg_list]
    coord_groups = _cluster_axis_values(coords, tol=axis_tol)
    sequences = []
    for g in coord_groups:
        gmin, gmax = min(g), max(g)
        items = [s for s in seg_list if gmin - 1e-6 <= s["coord"] <= gmax + 1e-6]
        if len(items) < cfg["dash_min_segments"]:
            continue
        items = sorted(items, key=lambda d: (d["a1"], d["a2"]))
        intervals = []
        cur_s, cur_e, lengths = items[0]["a1"], items[0]["a2"], [items[0]["len"]]
        for it in items[1:]:
            if it["a1"] <= cur_e + 2:
                cur_e = max(cur_e, it["a2"])
                lengths.append(it["len"])
            else:
                intervals.append((cur_s, cur_e, float(np.mean(lengths))))
                cur_s, cur_e, lengths = it["a1"], it["a2"], [it["len"]]
        intervals.append((cur_s, cur_e, float(np.mean(lengths))))
        if len(intervals) < cfg["dash_min_segments"]:
            continue
        gaps = []
        for i in range(len(intervals) - 1):
            gap = intervals[i + 1][0] - intervals[i][1]
            if gap >= 0:
                gaps.append(gap)
        if len(gaps) < 2:
            continue
        mean_gap = float(np.mean(gaps))
        span = intervals[-1][1] - intervals[0][0]
        span_ratio = span / max(1.0, patch_span)
        if not (cfg["dash_gap_min"] <= mean_gap <= cfg["dash_gap_max"]):
            continue
        if span_ratio < cfg["dash_min_span_ratio"]:
            continue
        sequences.append({
            "coord_mean": float(np.mean(g)),
            "segment_count": int(len(intervals)),
            "span": float(span),
            "span_ratio": float(span_ratio),
            "mean_gap": float(mean_gap),
            "intervals": intervals,
        })
    return sequences


def detect_hidden_dashed_in_patch(edges_patch: np.ndarray, cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    h, w = edges_patch.shape[:2]
    pmin = max(1, min(h, w))
    lines = cv2.HoughLinesP(
        edges_patch, 1, np.pi / 180, cfg["dash_hough_threshold"],
        minLineLength=max(5, int(pmin * cfg["dash_min_len_ratio"])), maxLineGap=2
    )
    if lines is None:
        return {"present": False, "sequence_count": 0, "horizontal_sequences": [], "vertical_sequences": []}
    raw = [tuple(map(int, ln)) for ln in lines[:, 0]]
    hsegs, vsegs = _group_short_hv_segments(raw, edges_patch.shape, cfg)
    hseqs = _build_dashed_sequences(hsegs, cfg["dash_axis_tol"], w, cfg)
    vseqs = _build_dashed_sequences(vsegs, cfg["dash_axis_tol"], h, cfg)
    seq_count = len(hseqs) + len(vseqs)
    return {"present": bool(seq_count >= 1), "sequence_count": int(seq_count), "horizontal_sequences": hseqs, "vertical_sequences": vseqs}


def detect_centerline_from_dashed(hidden_result: Dict[str, Any], patch_shape: Tuple[int, int], cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    h, w = patch_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    near_x, near_y = w * cfg["center_near_ratio"], h * cfg["center_near_ratio"]
    hnear = [s for s in hidden_result["horizontal_sequences"] if abs(s["coord_mean"] - cy) <= near_y]
    vnear = [s for s in hidden_result["vertical_sequences"] if abs(s["coord_mean"] - cx) <= near_x]
    if hnear and vnear:
        return {"present": True, "strength": "strong", "horizontal_near_count": int(len(hnear)), "vertical_near_count": int(len(vnear))}
    for s in hnear:
        if s["span_ratio"] >= 0.24:
            return {"present": True, "strength": "weak", "horizontal_near_count": int(len(hnear)), "vertical_near_count": int(len(vnear))}
    for s in vnear:
        if s["span_ratio"] >= 0.24:
            return {"present": True, "strength": "weak", "horizontal_near_count": int(len(hnear)), "vertical_near_count": int(len(vnear))}
    return {"present": False, "strength": "none", "horizontal_near_count": int(len(hnear)), "vertical_near_count": int(len(vnear))}


def score_line_semantics(role_result: Dict[str, Any], bundle: Dict[str, np.ndarray], cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    roles = role_result["roles"]
    ordered_roles = []
    for name in ["front", "top", "side"]:
        info = roles.get(name)
        if info is not None and "box" in info and info["box"] is not None:
            ordered_roles.append((name, tuple(info["box"])))
    per_role = []
    for role_name, role_box in ordered_roles:
        _, _, _, edges_patch = crop_role_arrays(bundle, role_box)
        visible = detect_visible_contours_in_patch(edges_patch, cfg)
        hidden = detect_hidden_dashed_in_patch(edges_patch, cfg)
        center = detect_centerline_from_dashed(hidden, edges_patch.shape, cfg)
        per_role.append({"role": role_name, "role_box": tuple(map(int, role_box)), "visible": visible, "hidden": hidden, "center": center})
    visible_roles = sum(1 for d in per_role if d["visible"]["present"])
    hidden_roles = sum(1 for d in per_role if d["hidden"]["present"])
    center_roles = sum(1 for d in per_role if d["center"]["present"])
    strong_center_roles = sum(1 for d in per_role if d["center"]["strength"] == "strong")
    total_hidden_seq = sum(d["hidden"]["sequence_count"] for d in per_role)
    s_visible = 5 if visible_roles >= 3 else 3 if visible_roles == 2 else 1 if visible_roles == 1 else 0
    s_hidden = 4 if hidden_roles >= 2 or total_hidden_seq >= 3 else 2 if hidden_roles == 1 and total_hidden_seq >= 1 else 0
    s_center = 3 if strong_center_roles >= 1 else 2 if center_roles >= 1 else 0
    s_div = 3 if visible_roles >= 3 and hidden_roles >= 1 and center_roles >= 1 else 2 if visible_roles >= 2 and (hidden_roles >= 1 or center_roles >= 1) else 1 if visible_roles >= 1 else 0
    total = int(min(15, s_visible + s_hidden + s_center + s_div))
    errors, warnings = [], []
    if visible_roles == 0:
        errors.append("Asosiy kontur chiziqlari evidence topilmadi")
    elif visible_roles == 1:
        warnings.append("Kontur chiziqlari faqat bitta ko‘rinishda aniq ko‘rindi")
    if hidden_roles == 0:
        warnings.append("Shtrix/yashirin chiziqlar evidence juda kam")
    if center_roles == 0:
        warnings.append("Markaz chiziqlari evidence topilmadi yoki juda sust")
    if strong_center_roles == 0 and center_roles >= 1:
        warnings.append("Markaz chiziqlari kuchsiz evidence bilan topildi")
    summary = {
        "criterion": "Chiziq semantikasi va chizmachilik qoidalari",
        "score": int(total),
        "max_score": 15,
        "subscores": {
            "visible_contours": int(s_visible),
            "hidden_lines": int(s_hidden),
            "centerlines": int(s_center),
            "diversity_consistency": int(s_div),
        },
        "counts": {
            "roles_checked": int(len(per_role)),
            "roles_with_visible": int(visible_roles),
            "roles_with_hidden": int(hidden_roles),
            "roles_with_centerline": int(center_roles),
            "roles_with_strong_centerline": int(strong_center_roles),
            "total_hidden_sequences": int(total_hidden_seq),
        },
        "errors": errors,
        "warnings": warnings,
        "per_role": [{
            "role": d["role"],
            "role_box": [int(v) for v in d["role_box"]],
            "visible": {"present": bool(d["visible"]["present"]), "count": int(d["visible"]["count"]), "total_length_ratio": float(round(d["visible"]["total_length_ratio"], 4))},
            "hidden": {"present": bool(d["hidden"]["present"]), "sequence_count": int(d["hidden"]["sequence_count"]), "horizontal_sequences": int(len(d["hidden"]["horizontal_sequences"])), "vertical_sequences": int(len(d["hidden"]["vertical_sequences"]))},
            "center": {"present": bool(d["center"]["present"]), "strength": str(d["center"]["strength"]), "horizontal_near_count": int(d["center"]["horizontal_near_count"]), "vertical_near_count": int(d["center"]["vertical_near_count"])}
        } for d in per_role],
    }
    return {"score": int(total), "max_score": 15, "summary": summary}


__all__ = [
    'crop_role_arrays',
    'acute_deg',
    'line_len',
    'detect_visible_contours_in_patch',
    '_cluster_axis_values',
    '_group_short_hv_segments',
    '_build_dashed_sequences',
    'detect_hidden_dashed_in_patch',
    'detect_centerline_from_dashed',
    'score_line_semantics',
]
