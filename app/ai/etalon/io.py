"""Input loading, normalization, alignment, and global match metrics."""

from __future__ import annotations

import io

import cv2
import fitz
import numpy as np
from PIL import Image

def read_image_from_upload(uploaded_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(uploaded_bytes)) as image:
        return np.array(image.convert("RGB"))


def read_first_page_pdf_as_image(pdf_bytes: bytes, zoom: float = 2.5) -> np.ndarray:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        if document.page_count < 1:
            raise ValueError("PDF sahifalari topilmadi.")
        page = document.load_page(0)
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        png_bytes = pixmap.tobytes("png")
    with Image.open(io.BytesIO(png_bytes)) as image:
        return np.array(image.convert("RGB"))


def normalize_for_cv(rgb: np.ndarray):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
    )
    return gray, thr


def load_any_to_thr(fname: str, data: bytes, zoom=2.5):
    if fname.lower().endswith(".pdf"):
        rgb = read_first_page_pdf_as_image(data, zoom=zoom)
    else:
        rgb = read_image_from_upload(data)
    _, thr = normalize_for_cv(rgb)
    return rgb, thr


def bin_to_lines(thr_img: np.ndarray):
    return (((255 - thr_img) > 0).astype(np.uint8) * 255)


def crop_to_drawing(binary_thr: np.ndarray, pad: int = 40):
    inv = 255 - binary_thr
    inv = cv2.medianBlur(inv, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=1)
    ys, xs = np.where(inv > 0)
    if len(xs) == 0 or len(ys) == 0:
        return binary_thr
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    h, w = binary_thr.shape[:2]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)
    return binary_thr[y1:y2 + 1, x1:x2 + 1]


def resize_to_same(a: np.ndarray, b: np.ndarray, max_side: int = 1200):
    def scale(img):
        h, w = img.shape[:2]
        s = max(h, w)
        if s <= max_side:
            return img
        r = max_side / s
        return cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)

    a2 = scale(a); b2 = scale(b)
    ha, wa = a2.shape[:2]; hb, wb = b2.shape[:2]
    H = max(ha, hb); W = max(wa, wb)

    def pad_to(img, H, W, bg=255):
        h, w = img.shape[:2]
        out = np.full((H, W), bg, dtype=img.dtype)
        out[:h, :w] = img
        return out

    return pad_to(a2, H, W), pad_to(b2, H, W)


def ecc_align(et_thr: np.ndarray, st_thr: np.ndarray, iters=900, eps=1e-6):
    et = (255 - et_thr).astype(np.float32) / 255.0
    st = (255 - st_thr).astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        _, warp = cv2.findTransformECC(et, st, warp, cv2.MOTION_AFFINE, criteria, None, 1)
    except cv2.error:
        return st_thr, None
    h, w = et_thr.shape[:2]
    aligned = cv2.warpAffine(st_thr, warp, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    return aligned, warp


def xor_diff(et_thr: np.ndarray, st_thr: np.ndarray, band_kernel=(7, 7), noise_kernel=(5, 5)):
    et_lines = bin_to_lines(et_thr)
    st_lines = bin_to_lines(st_thr)
    k_band = cv2.getStructuringElement(cv2.MORPH_RECT, band_kernel)
    et_d = cv2.dilate(et_lines, k_band, iterations=1)
    st_d = cv2.dilate(st_lines, k_band, iterations=1)
    diff = cv2.bitwise_xor(et_d, st_d)
    k_noise = cv2.getStructuringElement(cv2.MORPH_RECT, noise_kernel)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, k_noise, iterations=1)
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, k_noise, iterations=1)
    return diff, et_lines, st_lines


def remove_outer_border(mask: np.ndarray, border_px: int = 60):
    m = mask.copy()
    h, w = m.shape[:2]
    m[:border_px, :] = 0
    m[h - border_px:, :] = 0
    m[:, :border_px] = 0
    m[:, w - border_px:] = 0
    return m


def remove_small_components(mask: np.ndarray, min_pixels: int = 300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_pixels:
            out[labels == i] = 255
    return out


def compute_match_metrics(et_thr: np.ndarray, st_thr: np.ndarray, diff_mask: np.ndarray):
    et = bin_to_lines(et_thr)
    st = bin_to_lines(st_thr)
    diff = (diff_mask > 0).astype(np.uint8) * 255

    union = ((et > 0) | (st > 0))
    union_area = max(int(union.sum()), 1)
    diff_area = int((diff > 0).sum())
    similarity = 1.0 - (diff_area / union_area)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    et_band = cv2.dilate(et, k, iterations=1)
    st_band = cv2.dilate(st, k, iterations=1)

    st_inside = cv2.bitwise_and(st, et_band)
    st_area = max(int((st > 0).sum()), 1)
    coverage = int((st_inside > 0).sum()) / st_area

    et_missing = cv2.bitwise_and(et, cv2.bitwise_not(st_band))
    et_area = max(int((et > 0).sum()), 1)
    missing_ratio = int((et_missing > 0).sum()) / et_area

    st_extra = cv2.bitwise_and(st, cv2.bitwise_not(et_band))
    extra_ratio = int((st_extra > 0).sum()) / st_area

    return {
        "similarity": float(similarity),
        "coverage": float(coverage),
        "missing_ratio": float(missing_ratio),
        "extra_ratio": float(extra_ratio),
    }


__all__ = [
    'read_image_from_upload',
    'read_first_page_pdf_as_image',
    'normalize_for_cv',
    'load_any_to_thr',
    'bin_to_lines',
    'crop_to_drawing',
    'resize_to_same',
    'ecc_align',
    'xor_diff',
    'remove_outer_border',
    'remove_small_components',
    'compute_match_metrics',
]
