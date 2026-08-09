"""Section and hatching score (10 points)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.ai.optional.config import CONFIG
from app.ai.optional.normalization import rotate_bound

def acute_angle_deg(angle_deg: float) -> float:
    a = abs(angle_deg) % 180.0
    if a > 90:
        a = 180 - a
    return a


def weighted_mean_std(values, weights) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if len(values) == 0 or weights.sum() <= 1e-6:
        return 0.0, 999.0
    mean = float(np.sum(values * weights) / (np.sum(weights) + 1e-6))
    var = float(np.sum(weights * (values - mean) ** 2) / (np.sum(weights) + 1e-6))
    return mean, math.sqrt(max(0.0, var))


def rotate_bound_gray(image: np.ndarray, angle_deg: float) -> np.ndarray:
    return rotate_bound(image, angle_deg, border_value=0)


def estimate_spacing_cv_from_mask(line_mask: np.ndarray, dominant_angle: float) -> Tuple[Optional[float], List[float]]:
    if line_mask is None or line_mask.size == 0 or (line_mask > 0).sum() == 0:
        return None, []
    rot = rotate_bound_gray((line_mask > 0).astype(np.uint8) * 255, -dominant_angle)
    row_occ = (rot > 0).mean(axis=1)
    thr = max(0.002, float(row_occ.max()) * 0.25)
    idx = np.where(row_occ >= thr)[0]
    if len(idx) == 0:
        return None, []
    splits = np.where(np.diff(idx) > 1)[0]
    groups = [g for g in np.split(idx, splits + 1) if len(g) >= 1]
    if len(groups) < 3:
        return None, []
    centers = np.array([(g[0] + g[-1]) / 2.0 for g in groups], dtype=np.float32)
    gaps = np.diff(centers)
    if len(gaps) < 2:
        return None, gaps.tolist()
    mean_gap = float(np.mean(gaps))
    std_gap = float(np.std(gaps))
    if mean_gap <= 1e-6:
        return None, gaps.tolist()
    return float(std_gap / mean_gap), gaps.tolist()


def _trim_mask_bbox(mask: np.ndarray, pad: int = 2) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    sub = mask[y1:y2, x1:x2]
    row_occ = (sub > 0).mean(axis=1)
    col_occ = (sub > 0).mean(axis=0)
    rthr = max(0.01, float(row_occ.max()) * 0.18)
    cthr = max(0.01, float(col_occ.max()) * 0.18)
    ys2 = np.where(row_occ >= rthr)[0]
    xs2 = np.where(col_occ >= cthr)[0]
    if len(xs2) == 0 or len(ys2) == 0:
        return (x1, y1, x2, y2)
    nx1 = x1 + max(0, int(xs2[0]) - pad)
    nx2 = x1 + min(sub.shape[1], int(xs2[-1]) + 1 + pad)
    ny1 = y1 + max(0, int(ys2[0]) - pad)
    ny2 = y1 + min(sub.shape[0], int(ys2[-1]) + 1 + pad)
    return (int(nx1), int(ny1), int(nx2), int(ny2))


def _detect_hatch_single(patch_bw: np.ndarray, patch_edges: np.ndarray, cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    h, w = patch_bw.shape[:2]
    patch_min = max(1, min(h, w))
    all_lines = []
    for threshold, min_part in [(18, 14), (12, 18)]:
        lines = cv2.HoughLinesP(
            patch_edges, 1, np.pi / 180, threshold,
            minLineLength=max(6, patch_min // min_part), maxLineGap=10
        )
        if lines is not None:
            all_lines.extend(lines[:, 0].tolist())
    if len(all_lines) == 0:
        return {"present": False, "reason": "no_lines", "line_count": 0}
    diag_angles, diag_lengths, diag_segments = [], [], []
    total_diag_length = 0.0
    for line in all_lines:
        x1, y1, x2, y2 = line
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 3:
            continue
        ang = acute_angle_deg(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if cfg["hatch_angle_min"] <= ang <= cfg["hatch_angle_max"]:
            diag_angles.append(float(ang))
            diag_lengths.append(length)
            diag_segments.append((x1, y1, x2, y2, ang, length))
            total_diag_length += length
    if len(diag_segments) < cfg["hatch_min_lines"]:
        return {"present": False, "reason": "few_diag_lines", "line_count": int(len(diag_segments))}
    bins = np.arange(cfg["hatch_angle_min"], cfg["hatch_angle_max"] + 1, 1)
    hist = np.zeros(len(bins), dtype=np.float32)
    for ang, ln in zip(diag_angles, diag_lengths):
        idx = int(np.argmin(np.abs(bins - ang)))
        hist[idx] += float(ln)
    dominant_angle = float(bins[int(np.argmax(hist))])
    kept, kept_angles, kept_lengths = [], [], []
    for (x1, y1, x2, y2, ang, length) in diag_segments:
        if abs(ang - dominant_angle) <= cfg["hatch_keep_tol_deg"]:
            kept.append((x1, y1, x2, y2))
            kept_angles.append(float(ang))
            kept_lengths.append(float(length))
    if len(kept) < cfg["hatch_min_lines"]:
        return {"present": False, "reason": "few_parallel_lines", "line_count": int(len(kept)), "dominant_angle": float(dominant_angle)}
    kept_total_length = float(np.sum(kept_lengths))
    parallel_ratio = kept_total_length / (total_diag_length + 1e-6)
    length_ratio = kept_total_length / max(1.0, float(patch_min))
    line_mask = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in kept:
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    line_mask = cv2.dilate(line_mask, k, iterations=int(cfg["hatch_mask_dilate"]))
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, k, iterations=1)
    hatch_bbox = _trim_mask_bbox(line_mask, 2)
    if hatch_bbox is None:
        return {"present": False, "reason": "no_hatch_bbox", "line_count": int(len(kept)), "dominant_angle": float(dominant_angle)}
    x1, y1, x2, y2 = hatch_bbox
    sub_mask = line_mask[y1:y2, x1:x2]
    coverage = float((sub_mask > 0).mean())
    mean_ang, std_ang = weighted_mean_std(kept_angles, kept_lengths)
    spacing_cv, _ = estimate_spacing_cv_from_mask(sub_mask, dominant_angle)
    angle_score = max(0.0, 1.0 - min(std_ang, 15.0) / 15.0)
    spacing_score = 0.40 if spacing_cv is None else max(0.0, 1.0 - min(spacing_cv, 1.1) / 1.1)
    if coverage < cfg["hatch_min_coverage"]:
        coverage_score = 0.0
    elif coverage > cfg["hatch_max_coverage"]:
        coverage_score = 0.35
    else:
        mid = 0.12
        coverage_score = max(0.0, 1.0 - abs(coverage - mid) / max(mid, 1e-6))
    strength_score = min(1.0, 0.5 * parallel_ratio + 0.5 * min(1.0, length_ratio / 1.6))
    quality = 0.30 * strength_score + 0.30 * angle_score + 0.25 * spacing_score + 0.15 * coverage_score
    present = (
        len(kept) >= cfg["hatch_min_lines"] and parallel_ratio >= cfg["hatch_min_parallel_ratio"] and
        length_ratio >= cfg["hatch_min_total_length_ratio"] and coverage >= cfg["hatch_min_coverage"]
    )
    return {
        "present": bool(present),
        "reason": "ok" if present else "weak_hatch_pattern",
        "line_count": int(len(kept)),
        "dominant_angle": float(dominant_angle),
        "angle_mean": float(mean_ang),
        "angle_std": float(std_ang),
        "parallel_ratio": float(parallel_ratio),
        "length_ratio": float(length_ratio),
        "coverage": float(coverage),
        "spacing_cv": None if spacing_cv is None else float(spacing_cv),
        "quality": float(quality),
        "local_bbox": [int(v) for v in hatch_bbox],
        "line_mask": line_mask,
    }


def detect_hatch_in_patch(patch_bw: np.ndarray, patch_edges: np.ndarray, cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    h, w = patch_bw.shape[:2]
    trials = []
    for scale in [1.0, 1.8]:
        if scale == 1.0:
            bw_s, ed_s = patch_bw, patch_edges
        else:
            nw = max(8, int(round(w * scale)))
            nh = max(8, int(round(h * scale)))
            bw_s = cv2.resize(patch_bw, (nw, nh), interpolation=cv2.INTER_NEAREST)
            ed_s = cv2.resize(patch_edges, (nw, nh), interpolation=cv2.INTER_NEAREST)
        det = _detect_hatch_single(bw_s, ed_s, cfg)
        det["scale"] = float(scale)
        if det.get("line_mask") is not None and scale != 1.0:
            back_mask = cv2.resize(det["line_mask"], (w, h), interpolation=cv2.INTER_NEAREST)
            det["line_mask"] = back_mask
            bbox = _trim_mask_bbox(back_mask, 1)
            det["local_bbox"] = [int(v) for v in bbox] if bbox is not None else det.get("local_bbox")
        trials.append(det)

    def rank_key(d: Dict[str, Any]):
        return (
            1 if d.get("present", False) else 0,
            float(d.get("quality", 0.0)),
            int(d.get("line_count", 0)),
            float(d.get("parallel_ratio", 0.0) or 0.0),
            float(d.get("length_ratio", 0.0) or 0.0),
        )

    return sorted(trials, key=rank_key, reverse=True)[0]


def crop_role_patch(bundle: Dict[str, np.ndarray], box: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = [int(v) for v in box]
    return (
        bundle["rgb"][y1:y2, x1:x2].copy(),
        bundle["bw"][y1:y2, x1:x2].copy(),
        bundle["edges"][y1:y2, x1:x2].copy(),
    )


def gather_hatch_candidates(role_result: Dict[str, Any], bundle: Dict[str, np.ndarray], cfg: Dict[str, Any] = CONFIG) -> List[Dict[str, Any]]:
    roles = role_result["roles"]
    ordered = []
    for name in ["front", "top", "side", "isometric"]:
        info = roles.get(name)
        if info is not None and "box" in info and info["box"] is not None:
            ordered.append((name, info["box"]))
    extras = roles.get("extra_orthographic", [])
    for i, info in enumerate(extras, 1):
        if info is not None and "box" in info and info["box"] is not None:
            ordered.append((f"extra_{i}", info["box"]))
    results = []
    for role_name, box in ordered:
        _, bw_patch, edges_patch = crop_role_patch(bundle, box)
        det = detect_hatch_in_patch(bw_patch, edges_patch, cfg)
        results.append({
            "role": role_name,
            "box": [int(v) for v in box],
            "present": bool(det["present"]),
            "reason": det.get("reason", ""),
            "quality": float(det.get("quality", 0.0)) if "quality" in det else 0.0,
            "line_count": int(det.get("line_count", 0)),
            "dominant_angle": None if det.get("dominant_angle") is None else float(det.get("dominant_angle")),
            "angle_std": None if det.get("angle_std") is None else float(det.get("angle_std")),
            "parallel_ratio": None if det.get("parallel_ratio") is None else float(det.get("parallel_ratio")),
            "length_ratio": None if det.get("length_ratio") is None else float(det.get("length_ratio")),
            "coverage": None if det.get("coverage") is None else float(det.get("coverage")),
            "spacing_cv": None if det.get("spacing_cv") is None else float(det.get("spacing_cv")),
            "local_bbox": det.get("local_bbox", None),
        })
    return results


def score_section_hatching(role_result: Dict[str, Any], bundle: Dict[str, np.ndarray], cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    candidates = gather_hatch_candidates(role_result, bundle, cfg)
    present = [c for c in candidates if c["present"]]
    warnings, errors = [], []
    if len(present) == 0:
        warnings.append("Qirqim yoki shtrixovka topilmadi. Keyingi modul qirqim talab qilingan-qilinmaganini tekshiradi.")
        summary = {
            "criterion": "Qirqim va shtrixovka sifati",
            "score": 0,
            "max_score": 10,
            "applicable": False,
            "detected_hatch_count": 0,
            "errors": [],
            "warnings": warnings,
            "candidates": candidates,
        }
        return {"score": 0, "max_score": 10, "summary": summary}
    best = sorted(present, key=lambda d: d["quality"], reverse=True)[0]
    s_presence = 3 if best["quality"] >= 0.55 else 2 if best["quality"] >= 0.35 else 1
    angle_std = 99.0 if best["angle_std"] is None else float(best["angle_std"])
    if angle_std <= 3.5:
        s_angle = 3
    elif angle_std <= 6.5:
        s_angle = 2
    elif angle_std <= 10.0:
        s_angle = 1
    else:
        s_angle = 0
        errors.append("Shtrixovka chiziqlari burchagi yetarli darajada bir xil emas")
    spacing_cv = None if best["spacing_cv"] is None else float(best["spacing_cv"])
    if spacing_cv is None:
        s_spacing = 1
        warnings.append("Shtrixovka chiziqlari oralig‘i muntazamligi to‘liq baholanmadi")
    elif spacing_cv <= 0.35:
        s_spacing = 2
    elif spacing_cv <= 0.70:
        s_spacing = 1
    else:
        s_spacing = 0
        errors.append("Shtrixovka chiziqlari oralig‘i notekis")
    coverage = 0.0 if best["coverage"] is None else float(best["coverage"])
    parallel_ratio = 0.0 if best["parallel_ratio"] is None else float(best["parallel_ratio"])
    region_quality_score = 0
    if 0.004 <= coverage <= 0.24:
        region_quality_score += 1
    else:
        warnings.append("Shtrixovka qamrovi juda kichik yoki juda katta ko‘rinadi")
    if parallel_ratio >= 0.70:
        region_quality_score += 1
    elif parallel_ratio < cfg["hatch_min_parallel_ratio"]:
        errors.append("Shtrixovka diagonallari yetarli darajada parallel emas")
    total = int(min(10, s_presence + s_angle + s_spacing + region_quality_score))
    summary = {
        "criterion": "Qirqim va shtrixovka sifati",
        "score": int(total),
        "max_score": 10,
        "applicable": True,
        "detected_hatch_count": int(len(present)),
        "best_role": str(best["role"]),
        "best_quality": float(round(best["quality"], 4)),
        "best_dominant_angle": None if best["dominant_angle"] is None else float(round(best["dominant_angle"], 4)),
        "subscores": {
            "presence_confidence": int(s_presence),
            "angle_consistency": int(s_angle),
            "spacing_consistency": int(s_spacing),
            "region_quality": int(region_quality_score),
        },
        "errors": errors,
        "warnings": warnings,
        "candidates": candidates,
    }
    return {"score": int(total), "max_score": 10, "summary": summary}


__all__ = [
    'acute_angle_deg',
    'weighted_mean_std',
    'rotate_bound_gray',
    'estimate_spacing_cv_from_mask',
    '_trim_mask_bbox',
    '_detect_hatch_single',
    'detect_hatch_in_patch',
    'crop_role_patch',
    'gather_hatch_candidates',
    'score_section_hatching',
]
