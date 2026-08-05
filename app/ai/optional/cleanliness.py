"""Drawing cleanliness and readability score (10 points)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np

from app.ai.optional.config import CONFIG

def _pad_box(box: Tuple[int, int, int, int], pad: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    return (max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad))


def _detect_frame_like_mask(bundle: Dict[str, np.ndarray], cfg: Dict[str, Any] = CONFIG) -> np.ndarray:
    bw = bundle["bw"]
    h, w = bw.shape[:2]
    h_len = max(40, w // 4)
    v_len = max(40, h // 4)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    hmap = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    vmap = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
    hv = cv2.bitwise_or(hmap, vmap)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((hv > 0).astype(np.uint8), connectivity=8)
    edge_margin_x = int(w * cfg["clean_frame_edge_margin_ratio"])
    edge_margin_y = int(h * cfg["clean_frame_edge_margin_ratio"])
    frame_mask = np.zeros_like(bw, dtype=np.uint8)
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        near_edge = (x <= edge_margin_x or y <= edge_margin_y or (x + ww) >= (w - edge_margin_x) or (y + hh) >= (h - edge_margin_y))
        long_enough = ((ww / max(1, w) >= cfg["clean_frame_min_span_ratio"]) or (hh / max(1, h) >= cfg["clean_frame_min_span_ratio"]))
        if near_edge and long_enough:
            frame_mask[y:y + hh, x:x + ww] = 255
    if cfg["clean_frame_dilate"] > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        frame_mask = cv2.dilate(frame_mask, k, iterations=int(cfg["clean_frame_dilate"]))
    return frame_mask


def _make_protected_mask(bundle: Dict[str, np.ndarray], layout_result: Dict[str, Any], pad: int = 10) -> np.ndarray:
    h, w = bundle["bw"].shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for info in layout_result.get("projection_infos", []):
        bx = _pad_box(info["box"], pad, w, h)
        mask[bx[1]:bx[3], bx[0]:bx[2]] = 255
    tbox = layout_result.get("title_box", None)
    if tbox is not None:
        bx = _pad_box(tbox, pad, w, h)
        mask[bx[1]:bx[3], bx[0]:bx[2]] = 255
    frame_mask = _detect_frame_like_mask(bundle, CONFIG)
    return cv2.bitwise_or(mask, frame_mask)


def _detect_stray_regions(bundle: Dict[str, np.ndarray], layout_result: Dict[str, Any], cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    bw = bundle["bw"]
    protected = _make_protected_mask(bundle, layout_result, cfg["clean_protect_pad"])
    outside = cv2.bitwise_and(bw, bw, mask=(255 - protected))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    outside2 = cv2.morphologyEx(outside, cv2.MORPH_OPEN, k, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((outside2 > 0).astype(np.uint8), connectivity=8)
    boxes, areas = [], []
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if area < cfg["clean_min_stray_area"] or area > cfg["clean_max_stray_area"]:
            continue
        boxes.append((int(x), int(y), int(x + ww), int(y + hh)))
        areas.append(int(area))
    stray_ratio = float((outside2 > 0).sum() / max(1, outside2.size))
    return {"boxes": boxes, "count": int(len(boxes)), "areas": areas, "stray_ratio": float(stray_ratio)}


def _measure_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _measure_contrast(gray: np.ndarray) -> float:
    return float(np.percentile(gray, 95) - np.percentile(gray, 5))


def _measure_border_occupancy(bundle: Dict[str, np.ndarray], cfg: Dict[str, Any] = CONFIG) -> float:
    bw = bundle["bw"]
    h, w = bw.shape[:2]
    b = int(cfg["clean_border_band"])
    frame_mask = _detect_frame_like_mask(bundle, cfg)
    bw2 = cv2.bitwise_and(bw, bw, mask=(255 - frame_mask))
    border_pixels = np.concatenate([
        bw2[:b, :].reshape(-1),
        bw2[h - b:h, :].reshape(-1),
        bw2[:, :b].reshape(-1),
        bw2[:, w - b:w].reshape(-1),
    ])
    return float((border_pixels > 0).mean())


def score_cleanliness(bundle: Dict[str, np.ndarray], layout_result: Dict[str, Any], cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    gray = bundle["gray_norm"]
    sharpness = _measure_sharpness(gray)
    contrast = _measure_contrast(gray)
    border_occ = _measure_border_occupancy(bundle, cfg)
    stray = _detect_stray_regions(bundle, layout_result, cfg)
    s_sharp = 3 if sharpness >= cfg["clean_sharp_good"] else 2 if sharpness >= cfg["clean_sharp_mid"] else 1 if sharpness >= cfg["clean_sharp_low"] else 0
    s_contrast = 2 if contrast >= cfg["clean_contrast_good"] else 1 if contrast >= cfg["clean_contrast_mid"] else 0
    sr = stray["stray_ratio"]
    s_bg = 3 if sr <= cfg["clean_stray_ratio_good"] and stray["count"] <= 1 else 2 if sr <= cfg["clean_stray_ratio_mid"] and stray["count"] <= 4 else 1 if sr <= cfg["clean_stray_ratio_bad"] else 0
    s_edge = 2 if border_occ <= cfg["clean_border_occ_good"] else 1 if border_occ <= cfg["clean_border_occ_mid"] else 0
    total = int(min(10, s_sharp + s_contrast + s_bg + s_edge))
    warnings = []
    if s_sharp <= 1:
        warnings.append("Rasmning aniqligi past yoki biroz xira ko‘rinadi")
    if s_contrast == 0:
        warnings.append("Kontrast past, chiziqlar ajralishi sust bo‘lishi mumkin")
    if s_bg <= 1 and stray["count"] > 0:
        warnings.append("Asosiy zonalardan tashqarida ortiqcha shovqin/kir belgilari mavjud")
    if s_edge == 0:
        warnings.append("Chizma chetga juda yaqin yoki crop juda tig‘iz")
    summary = {
        "criterion": "Chizma tozaligi va o‘qilishi",
        "score": int(total),
        "max_score": 10,
        "subscores": {"sharpness": int(s_sharp), "contrast": int(s_contrast), "background_cleanliness": int(s_bg), "edge_crop_cleanliness": int(s_edge)},
        "metrics": {
            "sharpness_laplacian_var": float(round(sharpness, 4)),
            "contrast_p95_p5": float(round(contrast, 4)),
            "border_occupancy": float(round(border_occ, 6)),
            "stray_ratio": float(round(stray["stray_ratio"], 6)),
            "stray_region_count": int(stray["count"]),
        },
        "errors": [],
        "warnings": warnings,
        "stray_regions": [{"box": [int(v) for v in b], "area": int(a)} for b, a in zip(stray["boxes"], stray["areas"])],
    }
    return {"score": int(total), "max_score": 10, "summary": summary}


__all__ = [
    '_pad_box',
    '_detect_frame_like_mask',
    '_make_protected_mask',
    '_detect_stray_regions',
    '_measure_sharpness',
    '_measure_contrast',
    '_measure_border_occupancy',
    'score_cleanliness',
]
