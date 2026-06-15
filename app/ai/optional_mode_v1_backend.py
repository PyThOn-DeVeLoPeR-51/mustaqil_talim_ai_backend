
"""
optional_mode_v1.py

V1 heuristic evaluator for "ixtiyoriy rejim" (optional mode) of the Mustaqil ta'lim AI platform.

Notes
-----
- This is a speed-first prototype intended for backend integration.
- It works reasonably on many drawings, but layout detection can still be brittle on some unseen/challenging sheets.
- Recommended backend behavior: surface `warnings` and `confidence_label` to the teacher/student UI.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image


CONFIG: Dict[str, Any] = {
    # loader / preprocess
    "pdf_dpi": 300,
    "max_side": 2200,
    "adaptive_block_size": 31,
    "adaptive_c": 12,
    "canny1": 50,
    "canny2": 150,

    # normalize sheet
    "page_min_area_ratio": 0.20,
    "deskew_max_abs_angle": 12.0,
    "deskew_min_abs_angle": 0.25,
    "content_min_row_occ": 0.003,
    "content_min_col_occ": 0.003,
    "content_pad": 24,

    # title block / layout
    "tb_bottom_ratio": 0.42,
    "tb_right_ratio": 0.45,
    "tb_min_area_ratio": 0.010,
    "tb_max_area_ratio": 0.30,
    "tb_min_fill_ratio": 0.03,
    "proj_min_area_ratio": 0.004,
    "proj_max_area_ratio": 0.55,
    "proj_min_w": 70,
    "proj_min_h": 70,
    "proj_dilate_iter": 2,
    "proj_close_iter": 2,
    "merge_iou_thr": 0.18,
    "merge_gap_px": 28,
    "draw_zone_pad": 12,

    # safe title refinement
    "tb_safe_row_thr_ratio": 0.18,
    "tb_safe_col_thr_ratio": 0.16,
    "tb_safe_pad": 6,
    "tb_safe_min_h_keep": 0.38,
    "tb_safe_min_w_keep": 0.55,

    # projection roles
    "iso_diag_bonus": 1.6,
    "iso_area_bonus": 0.9,
    "iso_far_bonus": 0.6,
    "front_area_bonus": 1.2,
    "front_neighbor_bonus": 1.0,
    "assign_min_x_overlap": 0.18,
    "assign_min_y_overlap": 0.18,
    "projection_system": "first_angle",
    "top_direction_bonus": 1.2,
    "side_direction_bonus": 0.8,

    # scorer 1
    "score_views_max": 6,
    "score_iso_max": 2,
    "score_role_max": 3,
    "score_arrangement_max": 4,
    "required_orthographic_views": 3,
    "require_isometric": True,

    # hatching
    "score_hatch_max": 10,
    "hatch_angle_min": 18,
    "hatch_angle_max": 75,
    "hatch_keep_tol_deg": 10.0,
    "hatch_min_lines": 3,
    "hatch_min_parallel_ratio": 0.34,
    "hatch_min_total_length_ratio": 0.45,
    "hatch_min_coverage": 0.0012,
    "hatch_max_coverage": 0.35,
    "hatch_min_component_area": 40,
    "hatch_mask_dilate": 2,

    # dimensions
    "score_dimension_max": 15,
    "dim_text_min_area": 12,
    "dim_text_max_area": 1200,
    "dim_text_min_h": 5,
    "dim_text_max_h": 40,
    "dim_text_min_w": 3,
    "dim_text_max_w": 80,
    "dim_line_min_len_ratio": 0.10,
    "dim_line_max_thickness": 5,
    "dim_cluster_gap": 18,
    "dim_cluster_min_area": 40,
    "dim_outside_min_ratio": 0.65,
    "dim_band_thickness_ratio": 0.18,
    "dim_band_reach_ratio": 0.30,
    "dim_text_group_gap": 8,
    "dim_line_box_pad": 2,
    "dim_hough_min_len_ratio": 0.08,
    "dim_hough_max_gap": 8,
    "dim_center_reject_ratio": 0.55,
    "dim_far_reject_ratio": 0.42,
    "dim_titleblock_overlap_reject": 0.10,
    "dim_other_view_overlap_reject": 0.08,
    "dim_min_cluster_score": 1.0,

    # line semantics
    "score_line_semantics_max": 15,
    "visible_min_len_ratio": 0.16,
    "visible_hough_threshold": 26,
    "dash_min_len_ratio": 0.04,
    "dash_max_len_ratio": 0.30,
    "dash_hough_threshold": 10,
    "dash_axis_tol": 5,
    "dash_min_segments": 3,
    "dash_min_span_ratio": 0.16,
    "dash_gap_min": 2,
    "dash_gap_max": 26,
    "center_near_ratio": 0.22,

    # cleanliness
    "score_cleanliness_max": 10,
    "clean_protect_pad": 10,
    "clean_min_stray_area": 20,
    "clean_max_stray_area": 12000,
    "clean_border_band": 10,
    "clean_sharp_good": 180.0,
    "clean_sharp_mid": 80.0,
    "clean_sharp_low": 35.0,
    "clean_contrast_good": 110.0,
    "clean_contrast_mid": 70.0,
    "clean_stray_ratio_good": 0.0025,
    "clean_stray_ratio_mid": 0.0075,
    "clean_stray_ratio_bad": 0.0180,
    "clean_border_occ_good": 0.010,
    "clean_border_occ_mid": 0.030,
    "clean_frame_edge_margin_ratio": 0.08,
    "clean_frame_min_span_ratio": 0.55,
    "clean_frame_dilate": 2,

    # task compliance
    "score_task_compliance_max": 10,
    "task_default_required_views": 3,
    "task_default_requires_dimensions": True,
    "task_default_requires_isometric": False,
    "task_default_requires_section": False,
}


def _json_default(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def clip_box(box: Tuple[int, int, int, int], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(w, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def box_area(box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter + 1e-6
    return inter / union


def merge_two_boxes(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def boxes_close(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], gap_px: int = 28) -> bool:
    return not (
        a[2] + gap_px < b[0] or
        b[2] + gap_px < a[0] or
        a[3] + gap_px < b[1] or
        b[3] + gap_px < a[1]
    )


def merge_boxes_iter(boxes: List[Tuple[int, int, int, int]], iou_thr: float = 0.18, gap_px: int = 28) -> List[Tuple[int, int, int, int]]:
    boxes = [tuple(map(int, b)) for b in boxes]
    changed = True
    while changed:
        changed = False
        new_boxes: List[Tuple[int, int, int, int]] = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            cur = boxes[i]
            used[i] = True
            merged = True
            while merged:
                merged = False
                for j in range(len(boxes)):
                    if used[j]:
                        continue
                    if box_iou(cur, boxes[j]) > iou_thr or boxes_close(cur, boxes[j], gap_px):
                        cur = merge_two_boxes(cur, boxes[j])
                        used[j] = True
                        merged = True
                        changed = True
            new_boxes.append(cur)
        boxes = new_boxes
    return boxes


def merge_boxes_simple(boxes: List[Tuple[int, int, int, int]], gap: int = 18) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    boxes = [tuple(map(int, b)) for b in boxes]
    changed = True
    while changed:
        changed = False
        new_boxes = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            cur = boxes[i]
            used[i] = True
            merged = True
            while merged:
                merged = False
                for j in range(len(boxes)):
                    if used[j]:
                        continue
                    if not (
                        cur[2] + gap < boxes[j][0] or boxes[j][2] + gap < cur[0] or
                        cur[3] + gap < boxes[j][1] or boxes[j][3] + gap < cur[1]
                    ):
                        cur = merge_two_boxes(cur, boxes[j])
                        used[j] = True
                        merged = True
                        changed = True
            new_boxes.append(cur)
        boxes = new_boxes
    return boxes


def _smooth_1d(arr: np.ndarray, k: int = 9) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(arr, kernel, mode="same")


def _intersects(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _box_intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _overlap_ratio(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    inter = _box_intersection_area(a, b)
    denom = max(1, min(box_area(a), box_area(b)))
    return inter / denom


def _box_center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------
def resize_keep_ratio(img: np.ndarray, max_side: int = 2200) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale == 1.0:
        return img
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def pdf_bytes_to_rgb(pdf_bytes: bytes, dpi: int = 300, page_index: int = 0) -> np.ndarray:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_index]
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


def image_bytes_to_rgb(image_bytes: bytes) -> np.ndarray:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_img)


def load_bytes(file_bytes: bytes, filename: str, pdf_dpi: int = 300, max_side: int = 2200) -> Tuple[np.ndarray, Dict[str, Any]]:
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        rgb = pdf_bytes_to_rgb(file_bytes, dpi=pdf_dpi)
    elif ext in ["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"]:
        rgb = image_bytes_to_rgb(file_bytes)
    else:
        raise ValueError(f"Qo‘llab-quvvatlanmaydigan format: {ext}")
    rgb = resize_keep_ratio(rgb, max_side=max_side)
    meta = {"filename": filename, "ext": ext, "shape": list(rgb.shape)}
    return rgb, meta


def load_path(input_path: str | Path, pdf_dpi: int = 300, max_side: int = 2200) -> Tuple[np.ndarray, Dict[str, Any]]:
    p = Path(input_path)
    file_bytes = p.read_bytes()
    return load_bytes(file_bytes, p.name, pdf_dpi=pdf_dpi, max_side=max_side)


# ---------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------
def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def normalize_gray(gray: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray)


def adaptive_bin(gray: np.ndarray, block_size: int = 31, c: int = 12) -> np.ndarray:
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c
    )


def edge_map(gray: np.ndarray, canny1: int = 50, canny2: int = 150) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, canny1, canny2)


def preprocess_bundle(rgb: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    gray = rgb_to_gray(rgb)
    gray_norm = normalize_gray(gray)
    bw = adaptive_bin(gray_norm, cfg["adaptive_block_size"], cfg["adaptive_c"])
    edges = edge_map(gray_norm, cfg["canny1"], cfg["canny2"])
    return {"rgb": rgb, "gray": gray, "gray_norm": gray_norm, "bw": bw, "edges": edges}


# ---------------------------------------------------------------------
# Normalize sheet
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Layout discovery
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Role analysis
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Scorer 1 — completeness / arrangement
# ---------------------------------------------------------------------
def _safe_box(info: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int, int, int]]:
    return info["box"] if info is not None and "box" in info and info["box"] is not None else None


def _center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def score_view_count(orthographic_count: int, required_count: int = 3, max_score: int = 6) -> Tuple[int, List[str]]:
    if orthographic_count >= required_count:
        return max_score, []
    if orthographic_count == 2:
        return 4, ["Bitta asosiy proyeksiya yetishmaydi"]
    if orthographic_count == 1:
        return 2, ["Kamida ikkita asosiy proyeksiya yetishmaydi"]
    return 0, ["Asosiy proyeksiyalar topilmadi"]


def score_isometric_presence(has_isometric: bool, required: bool = True, max_score: int = 2) -> Tuple[int, List[str]]:
    if not required:
        return max_score, []
    return (max_score, []) if has_isometric else (0, ["Yaqqol tasvir topilmadi"])


def score_role_presence(front_box, top_box, side_box, max_score: int = 3) -> Tuple[int, List[str]]:
    score = 0
    errors = []
    if front_box is not None:
        score += 1
    else:
        errors.append("FRONT view aniqlanmadi")
    if top_box is not None:
        score += 1
    else:
        errors.append("TOP view aniqlanmadi")
    if side_box is not None:
        score += 1
    else:
        errors.append("SIDE view aniqlanmadi")
    return score, errors


def score_arrangement(front_box, top_box, side_box, system: str = "first_angle", max_score: int = 4) -> Tuple[int, List[str], Dict[str, bool]]:
    score = 0
    errors: List[str] = []
    checks: Dict[str, bool] = {}
    if front_box is None:
        return 0, ["Arrangement tekshirish uchun FRONT yo‘q"], checks
    fcx, fcy = _center(front_box)
    if top_box is not None:
        _, tcy = _center(top_box)
        xov = x_overlap_ratio(front_box, top_box)
        cond_dir = tcy > fcy if system == "first_angle" else tcy < fcy
        cond_align = xov >= 0.18
        if cond_dir:
            score += 1
        else:
            errors.append("TOP view FRONT'ga nisbatan noto‘g‘ri tomonda")
        if cond_align:
            score += 1
        else:
            errors.append("TOP view FRONT bilan vertikal o‘qda yetarli mos emas")
        checks["top_direction_ok"] = bool(cond_dir)
        checks["top_alignment_ok"] = bool(cond_align)
    else:
        errors.append("TOP view yo‘qligi sabab arrangement tekshirilmadi")
        checks["top_direction_ok"] = False
        checks["top_alignment_ok"] = False
    if side_box is not None:
        scx, _ = _center(side_box)
        yov = y_overlap_ratio(front_box, side_box)
        cond_side = abs(scx - fcx) > 5
        cond_align = yov >= 0.18
        if cond_side:
            score += 1
        else:
            errors.append("SIDE view FRONT'ga nisbatan chap/o‘ngda joylashmagan")
        if cond_align:
            score += 1
        else:
            errors.append("SIDE view FRONT bilan gorizontal o‘qda yetarli mos emas")
        checks["side_direction_ok"] = bool(cond_side)
        checks["side_alignment_ok"] = bool(cond_align)
    else:
        errors.append("SIDE view yo‘qligi sabab arrangement tekshirilmadi")
        checks["side_direction_ok"] = False
        checks["side_alignment_ok"] = False
    return min(score, max_score), errors, checks


def score_orthographic_completeness(role_result: Dict[str, Any], cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    roles = role_result["roles"]
    iso_box = _safe_box(roles.get("isometric"))
    front_box = _safe_box(roles.get("front"))
    top_box = _safe_box(roles.get("top"))
    side_box = _safe_box(roles.get("side"))
    extra_orth = roles.get("extra_orthographic", [])
    orthographic_count = len(role_result.get("orthographic", []))
    has_isometric = iso_box is not None
    s1, e1 = score_view_count(orthographic_count, cfg["required_orthographic_views"], cfg["score_views_max"])
    s2, e2 = score_isometric_presence(has_isometric, cfg["require_isometric"], cfg["score_iso_max"])
    s3, e3 = score_role_presence(front_box, top_box, side_box, cfg["score_role_max"])
    s4, e4, checks = score_arrangement(front_box, top_box, side_box, cfg["projection_system"], cfg["score_arrangement_max"])
    total = int(s1 + s2 + s3 + s4)
    warnings = [f"{len(extra_orth)} ta ortiqcha orthographic region mavjud yoki roli aniqlanmagan"] if len(extra_orth) > 0 else []
    summary = {
        "criterion": "Proyeksiyalar to‘liqligi va joylashuvi",
        "score": int(total),
        "max_score": 15,
        "subscores": {"view_count": int(s1), "isometric_presence": int(s2), "role_presence": int(s3), "arrangement": int(s4)},
        "counts": {"orthographic_count": int(orthographic_count), "extra_orthographic_count": int(len(extra_orth)), "has_isometric": bool(has_isometric)},
        "checks": checks,
        "errors": e1 + e2 + e3 + e4,
        "warnings": warnings,
    }
    return {"score": int(total), "max_score": 15, "summary": summary}


# ---------------------------------------------------------------------
# Scorer 2 — hatching
# ---------------------------------------------------------------------
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
        warnings.append("Hatch/qirqim topilmadi. Keyin task compliance moduli qirqim talab qilingan-qilinmaganini tekshiradi.")
        summary = {
            "criterion": "Section / hatching quality",
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
        errors.append("Hatch chiziqlari burchagi yetarli darajada bir xil emas")
    spacing_cv = None if best["spacing_cv"] is None else float(best["spacing_cv"])
    if spacing_cv is None:
        s_spacing = 1
        warnings.append("Hatch spacing regularity to‘liq baholanmadi")
    elif spacing_cv <= 0.35:
        s_spacing = 2
    elif spacing_cv <= 0.70:
        s_spacing = 1
    else:
        s_spacing = 0
        errors.append("Hatch chiziqlari oralig‘i notekis")
    coverage = 0.0 if best["coverage"] is None else float(best["coverage"])
    parallel_ratio = 0.0 if best["parallel_ratio"] is None else float(best["parallel_ratio"])
    region_quality_score = 0
    if 0.004 <= coverage <= 0.24:
        region_quality_score += 1
    else:
        warnings.append("Hatch coverage juda kichik yoki juda katta ko‘rinadi")
    if parallel_ratio >= 0.70:
        region_quality_score += 1
    elif parallel_ratio < cfg["hatch_min_parallel_ratio"]:
        errors.append("Hatch diagonallari yetarli darajada parallel emas")
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


# ---------------------------------------------------------------------
# Scorer 3 — dimensions
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Scorer 4 — line semantics
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Scorer 5 — cleanliness
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Scorer 6 — task compliance
# ---------------------------------------------------------------------
def parse_task_requirements(task_text: str, cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    text = (task_text or "").strip().lower()
    req = {
        "required_views": None,
        "requires_isometric": None,
        "requires_section": None,
        "requires_dimensions": None,
        "raw_text": task_text,
    }
    patterns = [
        r'(\d+)\s*(?:ta\s*)?(?:proyeksiya|proeksiya|ko[\'‘’`]?rinish|korinish|вид|вида|views?)',
        r'(?:proyeksiya|proeksiya|ko[\'‘’`]?rinish|korinish|вид|views?)\s*(\d+)',
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            try:
                req["required_views"] = int(m.group(1))
                break
            except Exception:
                pass
    if any(k in text for k in ["yaqqol tasvir", "yaqqol", "aksonometriya", "axonometry", "isometric", "izometrik", "аксонометр", "изометр"]):
        req["requires_isometric"] = True
    if any(k in text for k in ["qirqim", "kesim", "section", "sectional", "razrez", "разрез", "сечение"]):
        req["requires_section"] = True
    if any(k in text for k in ["o'lcham", "o‘lcham", "olcham", "razmer", "размер", "dimension", "dimensions"]):
        req["requires_dimensions"] = True
    if req["required_views"] is None:
        req["required_views"] = int(cfg["task_default_required_views"])
    if req["requires_isometric"] is None:
        req["requires_isometric"] = bool(cfg["task_default_requires_isometric"])
    if req["requires_section"] is None:
        req["requires_section"] = bool(cfg["task_default_requires_section"])
    if req["requires_dimensions"] is None:
        req["requires_dimensions"] = bool(cfg["task_default_requires_dimensions"])
    return req


def score_task_compliance(task_text: str, role_result: Dict[str, Any], score2_result: Dict[str, Any], score3_result: Dict[str, Any], cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    req = parse_task_requirements(task_text, cfg)
    orth_count = int(len(role_result.get("orthographic", [])))
    has_isometric = bool(role_result.get("meta", {}).get("has_isometric", False))
    hatch_count = int(score2_result["summary"].get("detected_hatch_count", 0))
    has_section_evidence = hatch_count > 0
    dim_roles = int(score3_result["summary"]["counts"].get("roles_with_dimension", 0))
    has_dimension_evidence = dim_roles >= 1
    required_views = int(req["required_views"])
    if orth_count >= required_views:
        s_views, views_ok = 4, True
    elif orth_count == required_views - 1:
        s_views, views_ok = 2, False
    else:
        s_views, views_ok = 0, False
    if req["requires_isometric"]:
        s_iso, iso_ok = (2, True) if has_isometric else (0, False)
    else:
        s_iso, iso_ok = 2, True
    if req["requires_section"]:
        s_section, section_ok = (2, True) if has_section_evidence else (0, False)
    else:
        s_section, section_ok = 2, True
    if req["requires_dimensions"]:
        if has_dimension_evidence:
            s_dim = 2 if dim_roles >= 2 else 1
            dim_ok = True
        else:
            s_dim, dim_ok = 0, False
    else:
        s_dim, dim_ok = 2, True
    total = int(min(10, s_views + s_iso + s_section + s_dim))
    errors, warnings = [], []
    if not views_ok:
        errors.append(f"Talab qilingan proyeksiyalar soni yetarli emas: kerak={required_views}, topildi={orth_count}")
    if req["requires_isometric"] and not iso_ok:
        errors.append("Topshiriqda yaqqol tasvir talab qilingan, lekin topilmadi")
    if req["requires_section"] and not section_ok:
        errors.append("Topshiriqda qirqim/kesim talab qilingan, lekin evidence topilmadi")
    if req["requires_dimensions"] and not has_dimension_evidence:
        errors.append("Topshiriqda o‘lcham qo‘yish talab qilingan, lekin dimension evidence topilmadi")
    elif req["requires_dimensions"] and dim_roles == 1:
        warnings.append("O‘lcham evidence bor, lekin faqat bitta ko‘rinishda aniq topildi")
    summary = {
        "criterion": "Topshiriq talabiga moslik",
        "score": int(total),
        "max_score": 10,
        "requirements": {
            "required_views": int(req["required_views"]),
            "requires_isometric": bool(req["requires_isometric"]),
            "requires_section": bool(req["requires_section"]),
            "requires_dimensions": bool(req["requires_dimensions"]),
        },
        "evidence": {
            "orthographic_count": int(orth_count),
            "has_isometric": bool(has_isometric),
            "detected_hatch_count": int(hatch_count),
            "roles_with_dimension": int(dim_roles),
        },
        "checks": {
            "views_ok": bool(views_ok),
            "isometric_ok": bool(iso_ok),
            "section_ok": bool(section_ok),
            "dimensions_ok": bool(dim_ok),
        },
        "subscores": {"views": int(s_views), "isometric": int(s_iso), "section": int(s_section), "dimensions": int(s_dim)},
        "errors": errors,
        "warnings": warnings,
    }
    return {"score": int(total), "max_score": 10, "summary": summary}


# ---------------------------------------------------------------------
# Final aggregation
# ---------------------------------------------------------------------
def _band_label(score100: int) -> str:
    if score100 >= 90:
        return "A'lo"
    if score100 >= 80:
        return "Yaxshi"
    if score100 >= 70:
        return "Qoniqarli+"
    if score100 >= 60:
        return "Qoniqarli"
    return "Qoniqarsiz"


def _collect_messages(*summaries: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors, warnings = [], []
    for s in summaries:
        errors.extend(s.get("errors", []))
        warnings.extend(s.get("warnings", []))
    return list(dict.fromkeys(errors)), list(dict.fromkeys(warnings))


def _make_feedback(report: Dict[str, Any]) -> List[str]:
    fb = []
    mods = report["modules"]

    def ratio(m: Dict[str, Any]) -> float:
        return m["score"] / max(1, m["max_score"])

    if ratio(mods["completeness_arrangement"]) < 0.75:
        fb.append("Proyeksiyalar soni yoki joylashuvi bo‘yicha kamchiliklar bor.")
    if ratio(mods["section_hatching"]) < 0.65:
        fb.append("Qirqim/shtrixlash sifati yoki aniqlanishida muammo bor.")
    if ratio(mods["dimensions"]) < 0.70:
        fb.append("O‘lcham qo‘yish evidence sust yoki to‘liq emas.")
    if ratio(mods["line_semantics"]) < 0.70:
        fb.append("Chiziq turlari evidence yetarli emas.")
    if ratio(mods["cleanliness"]) < 0.70:
        fb.append("Chizma sifati, tozaligi yoki crop holati yaxshilanishi kerak.")
    if ratio(mods["task_compliance"]) < 0.80:
        fb.append("Topshiriq talablariga to‘liq moslik kuzatilmadi.")
    if not fb:
        fb.append("Chizma topshiriq va GOST/ESKD mezonlariga yaxshi mos keldi.")
    return fb


def assess_layout_reliability(layout_result: Dict[str, Any], role_result: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    score = 1.0

    projection_count = int(layout_result["meta"].get("projection_count", 0))
    orthographic_count = int(role_result["meta"].get("orthographic_count", 0))
    has_isometric = bool(role_result["meta"].get("has_isometric", False))
    title_found = bool(layout_result["meta"].get("title_block_found", False))

    if projection_count <= 1:
        warnings.append("Projection count juda kam topildi; layout ishonchliligi past bo‘lishi mumkin.")
        score -= 0.35
    if projection_count >= 6:
        warnings.append("Projection count juda ko‘p topildi; ortiqcha box yoki merge xatosi bo‘lishi mumkin.")
        score -= 0.20
    if orthographic_count == 0:
        warnings.append("Orthographic role ajratilmadi.")
        score -= 0.35
    elif orthographic_count > 3:
        warnings.append("Orthographic role soni odatdagidan ko‘p.")
        score -= 0.15
    if not has_isometric:
        warnings.append("Yaqqol tasvir aniqlanmadi.")
        score -= 0.10
    if not title_found:
        warnings.append("Title block topilmadi.")
        score -= 0.05

    proj_boxes = layout_result.get("projection_boxes", [])
    if proj_boxes:
        areas = [box_area(tuple(b)) for b in proj_boxes]
        total_proj = sum(areas)
        if total_proj > 0:
            largest_ratio = max(areas) / total_proj
            if largest_ratio >= 0.72:
                warnings.append("Bitta projection box juda katta; bir nechta ko‘rinish merge bo‘lgan bo‘lishi mumkin.")
                score -= 0.20

    score = max(0.0, min(1.0, score))
    label = "high" if score >= 0.80 else "medium" if score >= 0.60 else "low"
    return {"confidence_score": round(score, 4), "confidence_label": label, "warnings": warnings}


def build_final_report(
    layout_result: Dict[str, Any],
    role_result: Dict[str, Any],
    score1_result: Dict[str, Any],
    score2_result: Dict[str, Any],
    score3_result: Dict[str, Any],
    score4_result: Dict[str, Any],
    score5_result: Dict[str, Any],
    score6_result: Dict[str, Any],
) -> Dict[str, Any]:
    modules = {
        "completeness_arrangement": {"score": int(score1_result["score"]), "max_score": int(score1_result["max_score"]), "summary": score1_result["summary"]},
        "section_hatching": {"score": int(score2_result["score"]), "max_score": int(score2_result["max_score"]), "summary": score2_result["summary"]},
        "dimensions": {"score": int(score3_result["score"]), "max_score": int(score3_result["max_score"]), "summary": score3_result["summary"]},
        "line_semantics": {"score": int(score4_result["score"]), "max_score": int(score4_result["max_score"]), "summary": score4_result["summary"]},
        "cleanliness": {"score": int(score5_result["score"]), "max_score": int(score5_result["max_score"]), "summary": score5_result["summary"]},
        "task_compliance": {"score": int(score6_result["score"]), "max_score": int(score6_result["max_score"]), "summary": score6_result["summary"]},
    }
    raw_total = sum(v["score"] for v in modules.values())
    raw_max = sum(v["max_score"] for v in modules.values())
    final_score_100 = int(round((raw_total / max(1, raw_max)) * 100))
    errors, warnings = _collect_messages(
        score1_result["summary"], score2_result["summary"], score3_result["summary"],
        score4_result["summary"], score5_result["summary"], score6_result["summary"]
    )
    reliability = assess_layout_reliability(layout_result, role_result)
    warnings.extend(reliability["warnings"])
    warnings = list(dict.fromkeys(warnings))
    report = {
        "mode": "optional",
        "scoring_version": "v1-prototype",
        "raw_total": int(raw_total),
        "raw_max": int(raw_max),
        "final_score_100": int(final_score_100),
        "grade_label": _band_label(final_score_100),
        "confidence_score": reliability["confidence_score"],
        "confidence_label": reliability["confidence_label"],
        "layout_meta": layout_result.get("meta", {}),
        "role_meta": role_result.get("meta", {}),
        "modules": modules,
        "errors": errors,
        "warnings": warnings,
    }
    report["feedback"] = _make_feedback(report)
    return report


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def run_optional_mode_core(rgb: np.ndarray, task_text: str = "", cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    base_bundle = preprocess_bundle(rgb, cfg)
    norm_result = normalize_sheet(base_bundle["rgb"], cfg)
    bundle = norm_result["final_bundle"]
    layout_result = discover_layout(bundle, cfg)
    role_result = analyze_projection_roles(layout_result, cfg)
    score1_result = score_orthographic_completeness(role_result, cfg)
    score2_result = score_section_hatching(role_result, bundle, cfg)
    score3_result = score_dimensions(role_result, layout_result, bundle, cfg)
    score4_result = score_line_semantics(role_result, bundle, cfg)
    score5_result = score_cleanliness(bundle, layout_result, cfg)
    score6_result = score_task_compliance(task_text, role_result, score2_result, score3_result, cfg)
    final_report = build_final_report(
        layout_result, role_result, score1_result, score2_result,
        score3_result, score4_result, score5_result, score6_result
    )
    return {
        "bundle": bundle,
        "norm_result": norm_result,
        "layout_result": layout_result,
        "role_result": role_result,
        "score1_result": score1_result,
        "score2_result": score2_result,
        "score3_result": score3_result,
        "score4_result": score4_result,
        "score5_result": score5_result,
        "score6_result": score6_result,
        "final_report": final_report,
    }


def analyze_optional_mode_bytes(file_bytes: bytes, filename: str, task_text: str = "", cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    rgb, file_meta = load_bytes(file_bytes, filename, cfg["pdf_dpi"], cfg["max_side"])
    outputs = run_optional_mode_core(rgb=rgb, task_text=task_text, cfg=cfg)
    outputs["file_meta"] = file_meta
    return outputs


def analyze_optional_mode_file(input_path: str | Path, task_text: str = "", cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    rgb, file_meta = load_path(input_path, cfg["pdf_dpi"], cfg["max_side"])
    outputs = run_optional_mode_core(rgb=rgb, task_text=task_text, cfg=cfg)
    outputs["file_meta"] = file_meta
    return outputs


def save_optional_artifacts(outputs: Dict[str, Any], out_dir: str | Path) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "final_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(outputs["final_report"], f, ensure_ascii=False, indent=2, default=_json_default)

    images = {}
    image_items = {
        "normalized_sheet.png": outputs["norm_result"]["final_rgb"],
    }
    for name, arr in image_items.items():
        path = out_dir / name
        Image.fromarray(arr).save(path)
        images[name] = str(path)

    return {"json_path": str(json_path), **images}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run optional mode heuristic evaluator.")
    parser.add_argument("input_path", help="Path to student drawing (pdf/jpg/png/...)")
    parser.add_argument("--task-text", default="", help="Teacher task text")
    parser.add_argument("--out-dir", default="optional_mode_outputs", help="Directory to save outputs")
    args = parser.parse_args()

    outputs = analyze_optional_mode_file(args.input_path, task_text=args.task_text)
    saved = save_optional_artifacts(outputs, args.out_dir)
    print(json.dumps(outputs["final_report"], ensure_ascii=False, indent=2, default=_json_default))
    print("\nSaved:", json.dumps(saved, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------
# Backend entrypoint
# ---------------------------------------------------------------------
def _backend_json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _backend_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_backend_json_safe(v) for v in value]
    return value


def _build_backend_table(final_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, module in final_report.get("modules", {}).items():
        summary = module.get("summary", {}) if isinstance(module, dict) else {}
        rows.append({
            "criterion": str(summary.get("criterion", key)),
            "score": float(module.get("score", 0)),
            "max_score": float(module.get("max_score", 0)),
            "comment": "; ".join(summary.get("errors", [])[:2] or summary.get("warnings", [])[:2]),
        })
    return rows


def _save_optional_overlay(outputs: Dict[str, Any], output_dir: str | Path, stem: str) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = outputs["norm_result"]["final_rgb"].copy()

    layout_result = outputs.get("layout_result", {})
    role_result = outputs.get("role_result", {})

    # Draw detected projection boxes.
    for box in layout_result.get("projection_boxes", []) or []:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 180, 0), 3)

    # Draw main roles with labels.
    roles = (role_result.get("roles") or {}) if isinstance(role_result, dict) else {}
    labels = {"front": "FRONT", "top": "TOP", "side": "SIDE", "isometric": "ISO"}
    for role_name, label in labels.items():
        info = roles.get(role_name)
        if isinstance(info, dict) and info.get("box") is not None:
            x1, y1, x2, y2 = [int(v) for v in info["box"]]
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(rgb, label, (x1 + 5, max(22, y1 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    path = out_dir / f"optional_{stem}_overlay.png"
    Image.fromarray(rgb.astype(np.uint8)).save(path)
    return str(path)


def evaluate_optional(student_path: str, output_dir: str = "app/uploads/results", task_text: str = "") -> dict:
    """
    Backend entrypoint for optional/ixtiyoriy mode.
    Returns: total_score, details, overlay_path, table_json.
    """
    outputs = analyze_optional_mode_file(student_path, task_text=task_text)
    final_report = outputs["final_report"]

    stem = Path(student_path).stem
    overlay_path = _save_optional_overlay(outputs, output_dir=output_dir, stem=stem)
    saved = save_optional_artifacts(outputs, Path(output_dir) / f"optional_{stem}_artifacts")

    details = {
        **final_report,
        "file_meta": outputs.get("file_meta", {}),
        "artifacts": saved,
        "overlay_path": overlay_path,
    }

    return {
        "total_score": int(final_report.get("final_score_100", 0)),
        "details": _backend_json_safe(details),
        "overlay_path": overlay_path,
        "table_json": _backend_json_safe(_build_backend_table(final_report)),
    }

