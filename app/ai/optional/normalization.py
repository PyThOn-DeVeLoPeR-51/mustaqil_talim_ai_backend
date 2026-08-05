"""Page perspective correction, deskew, and sheet normalization."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from app.ai.optional.io import normalize_gray, preprocess_bundle, rgb_to_gray

def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB), 10)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB), 10)
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def rotate_bound(image: np.ndarray, angle_deg: float, border_value=(255, 255, 255)) -> np.ndarray:
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy
    return cv2.warpAffine(
        image, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border_value
    )


def find_page_quad(rgb: np.ndarray, min_area_ratio: float = 0.20) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    h, w = rgb.shape[:2]
    total_area = h * w
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 40, 140)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:20]:
        area = cv2.contourArea(cnt)
        if area < total_area * min_area_ratio:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32), {"method": "approx_quad", "area_ratio": float(area / total_area)}
    if contours:
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        if area >= total_area * min_area_ratio:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(np.float32)
            return box, {"method": "min_area_rect", "area_ratio": float(area / total_area)}
    return None, {"method": "not_found", "area_ratio": 0.0}


def normalize_line_angle(angle_deg: float) -> float:
    while angle_deg <= -90:
        angle_deg += 180
    while angle_deg > 90:
        angle_deg -= 180
    if angle_deg > 45:
        angle_deg -= 90
    elif angle_deg < -45:
        angle_deg += 90
    return angle_deg


def estimate_skew_angle(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 120, minLineLength=max(gray.shape[:2]) // 8, maxLineGap=20)
    if lines is None:
        return 0.0
    angles, lengths = [], []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = normalize_line_angle(math.degrees(math.atan2(dy, dx)))
        length = math.hypot(dx, dy)
        if abs(angle) <= 20:
            angles.append(angle)
            lengths.append(length)
    if not angles:
        return 0.0
    angles = np.array(angles, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.float32)
    return float(np.sum(angles * lengths) / (np.sum(lengths) + 1e-6))


def content_bbox_from_bw(bw: np.ndarray, min_row_occ: float = 0.003, min_col_occ: float = 0.003, pad: int = 24) -> Tuple[int, int, int, int]:
    h, w = bw.shape[:2]
    row_occ = (bw > 0).mean(axis=1)
    col_occ = (bw > 0).mean(axis=0)
    ys = np.where(row_occ > min_row_occ)[0]
    xs = np.where(col_occ > min_col_occ)[0]
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, w, h)
    x1, x2 = xs[0], xs[-1]
    y1, y2 = ys[0], ys[-1]
    return (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad))


def crop_rgb_by_bbox(rgb: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return rgb[y1:y2, x1:x2].copy()


def normalize_sheet(rgb: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    h0, w0 = rgb.shape[:2]
    quad, quad_info = find_page_quad(rgb, cfg["page_min_area_ratio"])
    if quad is not None:
        warped = four_point_transform(rgb, quad)
        page_found = True
    else:
        warped = rgb.copy()
        page_found = False
    warped_gray = rgb_to_gray(warped)
    warped_gray_norm = normalize_gray(warped_gray)
    skew_angle = estimate_skew_angle(warped_gray_norm)
    if cfg["deskew_min_abs_angle"] <= abs(skew_angle) <= cfg["deskew_max_abs_angle"]:
        deskewed = rotate_bound(warped, -skew_angle)
        applied_angle = -skew_angle
    else:
        deskewed = warped.copy()
        applied_angle = 0.0
    tmp_bundle = preprocess_bundle(deskewed, cfg)
    bbox = content_bbox_from_bw(tmp_bundle["bw"], cfg["content_min_row_occ"], cfg["content_min_col_occ"], cfg["content_pad"])
    final_rgb = crop_rgb_by_bbox(deskewed, bbox)
    final_bundle = preprocess_bundle(final_rgb, cfg)
    return {
        "warped": warped,
        "deskewed": deskewed,
        "final_rgb": final_rgb,
        "final_bundle": final_bundle,
        "meta": {
            "input_shape": [h0, w0],
            "page_found": bool(page_found),
            "page_method": quad_info["method"],
            "page_area_ratio": round(float(quad_info["area_ratio"]), 4),
            "applied_rotation_deg": round(float(applied_angle), 4),
            "content_bbox": [int(v) for v in bbox],
            "final_shape": list(final_rgb.shape[:2]),
        },
    }


__all__ = [
    'order_points',
    'four_point_transform',
    'rotate_bound',
    'find_page_quad',
    'normalize_line_angle',
    'estimate_skew_angle',
    'content_bbox_from_bw',
    'crop_rgb_by_bbox',
    'normalize_sheet',
]
