import io
import math
import json
import uuid
import numpy as np
import cv2
import fitz
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# ============================================================
# MEGA-CELL 1
# Mustaqil ta'lim chizma baholash platformasi
# ETALON MODE ONLY
# - Helperlar
# - Baholash modullari
# - 100 ballik tizim
# ============================================================

import io
import math
import json
import uuid
import numpy as np
import cv2
import fitz
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------
# CONFIG
# ----------------------------
SHOW_DEBUG = False
MAX_SIDE = 1200
PDF_ZOOM = 2.5

# ============================================================
# IO / PREPROCESS
# ============================================================
def read_image_from_upload(uploaded_bytes: bytes):
    img = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
    return np.array(img)

def read_first_page_pdf_as_image(pdf_bytes: bytes, zoom: float = 2.5):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return np.array(img)

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

# ============================================================
# /3 - RAMKA + BURCHAK SHTAMPI
# IMPORTANT:
# Agar sizda Step-1 uchun alohida final detector bo'lsa,
# shu funksiyaning ichini almashtirasiz.
# Hozircha bu blok mavjud match metriclar asosidagi vaqtinchalik
# placeholder sifatida qoldirildi, faqat kriteriy nomi to'g'rilandi.
# ============================================================
def score_frame_titleblock_3(metrics: dict):
    sim = float(metrics.get("similarity", 0.0))
    cov = float(metrics.get("coverage", 0.0))
    miss = float(metrics.get("missing_ratio", 1.0))
    extra = float(metrics.get("extra_ratio", 1.0))

    if sim < 0.12 or cov < 0.12:
        pts = 0
    else:
        q = (
            0.45 * sim +
            0.25 * cov +
            0.20 * (1.0 - miss) +
            0.10 * (1.0 - min(extra, 1.0))
        )
        pts = int(np.clip(round(q * 3), 0, 3))

    dbg = {
        "similarity": round(sim, 4),
        "coverage": round(cov, 4),
        "missing_ratio": round(miss, 4),
        "extra_ratio": round(extra, 4),
        "note": "placeholder_logic_until_real_step1_detector_is_inserted",
    }
    return pts, dbg

# ============================================================
# /6 - PLACEMENT
# ============================================================
def content_bbox(lines_255: np.ndarray, pad=20):
    ys, xs = np.where(lines_255 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    H, W = lines_255.shape[:2]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W - 1, x2 + pad); y2 = min(H - 1, y2 + pad)
    return (x1, y1, x2, y2)

def bbox_center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def angle_hist(lines_255, bins=18):
    edges = cv2.Canny(lines_255, 50, 150, apertureSize=3)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=10)
    if segs is None:
        return np.ones(bins) / bins
    angs = []
    for x1, y1, x2, y2 in segs[:, 0]:
        dx = x2 - x1; dy = y2 - y1
        L = np.hypot(dx, dy)
        if L < 40:
            continue
        a = (np.degrees(np.arctan2(dy, dx)) % 180.0)
        angs.append(a)
    if not angs:
        return np.ones(bins) / bins
    h, _ = np.histogram(angs, bins=bins, range=(0, 180))
    h = h.astype(np.float32)
    h = h / max(h.sum(), 1.0)
    return h

def cos_sim(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))

def score_placement_6(et_r: np.ndarray, st_aligned: np.ndarray):
    et_lines = bin_to_lines(et_r)
    st_lines = bin_to_lines(st_aligned)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    et_band = cv2.dilate(et_lines, k, iterations=1)
    st_inside = cv2.bitwise_and(st_lines, et_band)
    coverage = float((st_inside > 0).sum() / max((st_lines > 0).sum(), 1))

    eb = content_bbox(et_lines)
    sb = content_bbox(st_lines)
    H, W = et_lines.shape[:2]

    if eb is None or sb is None:
        return 0, {
            "coverage": round(coverage, 4),
            "shift_norm": None,
            "angle_sim": None,
            "reason": "content bbox not found",
        }

    ecx, ecy = bbox_center(eb)
    scx, scy = bbox_center(sb)
    shift = np.hypot(scx - ecx, scy - ecy)
    shift_norm = float(shift / max(W, H))
    shift_q = float(np.clip(1.0 - (shift_norm / 0.12), 0.0, 1.0))

    et_h = angle_hist(et_lines, bins=18)
    st_h = angle_hist(st_lines, bins=18)
    angle_sim = cos_sim(et_h, st_h)

    placement_score01 = (0.50 * coverage + 0.35 * shift_q + 0.15 * angle_sim)
    placement_pts = int(np.clip(round(placement_score01 * 6), 0, 6))

    dbg = {
        "coverage": round(coverage, 4),
        "shift_px": round(float(shift), 4),
        "shift_norm": round(shift_norm, 4),
        "shift_q": round(shift_q, 4),
        "angle_sim": round(angle_sim, 4),
        "placement_score01": round(float(placement_score01), 4),
    }
    return placement_pts, dbg

# ============================================================
# /8 - CHIZIQ TURLARI
# ============================================================
def dashed_like_count(lines_255: np.ndarray):
    m = (lines_255 > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m = cv2.erode(m, k, iterations=1)

    h_long = cv2.getStructuringElement(cv2.MORPH_RECT, (61, 1))
    v_long = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 61))
    long_h = cv2.morphologyEx(m, cv2.MORPH_OPEN, h_long, iterations=1)
    long_v = cv2.morphologyEx(m, cv2.MORPH_OPEN, v_long, iterations=1)
    m[cv2.bitwise_or(long_h, long_v) > 0] = 0

    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    cnt = 0
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if 10 <= area <= 220 and 2 <= w <= 45 and 2 <= h <= 45:
            cnt += 1
    return int(cnt), m

def stroke_width_stats(binary_thr: np.ndarray):
    inv = (255 - binary_thr)
    inv = (inv > 0).astype(np.uint8)
    if inv.sum() < 800:
        return None
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    vals = dist[inv > 0]
    p25 = float(np.percentile(vals, 25))
    p50 = float(np.percentile(vals, 50))
    p75 = float(np.percentile(vals, 75))
    p90 = float(np.percentile(vals, 90))
    r9050 = float(p90 / max(p50, 1e-6))
    r7525 = float(p75 / max(p25, 1e-6))
    return {"p50": p50, "r9050": r9050, "r7525": r7525}

def score_line_types_8(et_r: np.ndarray, st_aligned: np.ndarray):
    et_lines = bin_to_lines(et_r)
    st_lines = bin_to_lines(st_aligned)

    et_w = stroke_width_stats(et_r)
    st_w = stroke_width_stats(st_aligned)

    et_dash, et_dash_mask = dashed_like_count(et_lines)
    st_dash, st_dash_mask = dashed_like_count(st_lines)

    if et_w is None or st_w is None:
        return 0, {
            "reason": "too few strokes for width analysis",
            "et_dash": et_dash,
            "st_dash": st_dash,
        }, et_dash_mask, st_dash_mask

    div_et = 0.6 * et_w["r9050"] + 0.4 * et_w["r7525"]
    div_st = 0.6 * st_w["r9050"] + 0.4 * st_w["r7525"]

    dash_ratio = float((st_dash + 1) / (et_dash + 1))
    div_ratio = float(div_st / max(div_et, 1e-6))

    ABS_DASH_MIN = 70
    ABS_DASH_RATIO = 2.2

    gated = (st_dash >= ABS_DASH_MIN) and (dash_ratio >= ABS_DASH_RATIO)

    if gated:
        dbg = {
            "gated": True,
            "reason": "Absolute noisy dashed signature => 0",
            "et_dash": et_dash,
            "st_dash": st_dash,
            "dash_ratio": round(dash_ratio, 4),
            "div_et": round(div_et, 4),
            "div_st": round(div_st, 4),
            "div_ratio": round(div_ratio, 4),
        }
        return 0, dbg, et_dash_mask, st_dash_mask

    t_ratio = float(st_w["p50"] / max(et_w["p50"], 1e-6))
    t_ratio = float(max(t_ratio, 1.0 / t_ratio))
    thick_match_q = float(np.clip(1.0 - (t_ratio - 1.0) / 0.80, 0.0, 1.0))

    div_st_q = float(np.clip((div_st - 1.05) / (1.35 - 1.05), 0.0, 1.0))
    div_et_q = float(np.clip((div_et - 1.05) / (1.35 - 1.05), 0.0, 1.0))
    diversity_q = float(np.clip((1.0 - div_et_q) + div_et_q * div_st_q, 0.0, 1.0))

    if et_dash < 20:
        dash_q = 1.0
    else:
        dash_q = float(np.clip(st_dash / max(et_dash, 1), 0.0, 1.0))
        if st_dash < 8:
            dash_q = 0.0

    score01 = 0.45 * thick_match_q + 0.35 * diversity_q + 0.20 * dash_q
    line_pts = int(np.clip(round(score01 * 8), 0, 8))

    dbg = {
        "gated": False,
        "t_ratio_sym": round(t_ratio, 4),
        "thick_match_q": round(thick_match_q, 4),
        "div_et": round(div_et, 4),
        "div_st": round(div_st, 4),
        "div_ratio": round(div_ratio, 4),
        "et_dash": et_dash,
        "st_dash": st_dash,
        "dash_ratio": round(dash_ratio, 4),
        "dash_q": round(dash_q, 4),
        "diversity_q": round(diversity_q, 4),
        "score01": round(score01, 4),
    }
    return line_pts, dbg, et_dash_mask, st_dash_mask

# ============================================================
# /12 - O'LCHAM QO'YISH
# ============================================================
def remove_long_axes(lines_255: np.ndarray):
    m = lines_255.copy()
    h_long = cv2.getStructuringElement(cv2.MORPH_RECT, (81, 1))
    v_long = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 81))
    long_h = cv2.morphologyEx(m, cv2.MORPH_OPEN, h_long, iterations=1)
    long_v = cv2.morphologyEx(m, cv2.MORPH_OPEN, v_long, iterations=1)
    long_mask = cv2.bitwise_or(long_h, long_v)
    out = m.copy()
    out[long_mask > 0] = 0
    return out, long_mask

def small_component_stats(mask_255: np.ndarray):
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_255 > 0).astype(np.uint8), connectivity=8)
    small_cnt = 0
    small_area_sum = 0
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if 12 <= area <= 450 and 3 <= w <= 60 and 3 <= h <= 60:
            small_cnt += 1
            small_area_sum += area
    return small_cnt, small_area_sum

def dimension_like_mask(lines_255: np.ndarray):
    short, long_mask = remove_long_axes(lines_255)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    short = cv2.morphologyEx(short, cv2.MORPH_OPEN, k, iterations=1)
    short = cv2.morphologyEx(short, cv2.MORPH_CLOSE, k, iterations=1)
    return short, long_mask

def line_long_ratio(lines_255: np.ndarray):
    edges = cv2.Canny(lines_255, 50, 150, apertureSize=3)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=10)
    if segs is None:
        return 0.0
    lengths = []
    for x1, y1, x2, y2 in segs[:, 0]:
        L = float(np.hypot(x2 - x1, y2 - y1))
        if L >= 10:
            lengths.append(L)
    if not lengths:
        return 0.0
    lengths = np.array(lengths, dtype=np.float32)
    H, W = lines_255.shape[:2]
    Lthr = max(120.0, 0.10 * max(H, W))
    return float((lengths >= Lthr).mean())

def score_dimension_12(et_r: np.ndarray, st_aligned: np.ndarray, metrics: dict):
    et_lines = bin_to_lines(et_r)
    st_lines = bin_to_lines(st_aligned)

    sim = max(0.0, min(float(metrics.get("similarity", 0.0)), 1.0))
    miss = float(metrics.get("missing_ratio", 1.0))
    cov = float(metrics.get("coverage", 0.0))
    extra = float(metrics.get("extra_ratio", 0.0))

    et_dash, _ = dashed_like_count(et_lines)
    st_dash, _ = dashed_like_count(st_lines)
    dash_ratio = float((st_dash + 1) / (et_dash + 1))
    st_long = line_long_ratio(st_lines)

    hard_bad = (cov < 0.18 and sim < 0.28)
    ABS_DASH_MIN = 90
    ABS_DASH_RATIO = 2.6
    CAD_LONG_OK = 0.12
    noise_bad = (st_dash >= ABS_DASH_MIN) and (dash_ratio >= ABS_DASH_RATIO) and ((st_long < CAD_LONG_OK) or (sim < 0.15))
    extra_bad = (extra > 0.55 and miss > 0.40 and st_long < CAD_LONG_OK)
    noise_override = (sim <= 0.01) and (st_dash >= 70) and (dash_ratio >= 2.2)

    if hard_bad or noise_bad or extra_bad or noise_override:
        why = []
        if hard_bad: why.append("hard_bad")
        if noise_bad: why.append("noise_bad")
        if extra_bad: why.append("extra_bad")
        if noise_override: why.append("noise_override")
        dbg = {
            "mode": "GATED_TO_ZERO",
            "why": " + ".join(why),
            "similarity": round(sim, 4),
            "missing_ratio": round(miss, 4),
            "coverage": round(cov, 4),
            "extra_ratio": round(extra, 4),
            "et_dash": et_dash,
            "st_dash": st_dash,
            "dash_ratio": round(dash_ratio, 4),
            "st_long_ratio": round(st_long, 4),
        }
        return 0, dbg, np.zeros_like(et_lines), np.zeros_like(st_lines)

    et_dim, _ = dimension_like_mask(et_lines)
    st_dim, _ = dimension_like_mask(st_lines)

    et_dim_area = int((et_dim > 0).sum())
    st_dim_area = int((st_dim > 0).sum())
    et_small_cnt, _ = small_component_stats(et_dim)
    st_small_cnt, _ = small_component_stats(st_dim)

    if et_dim_area < 2000 and et_small_cnt < 12:
        pts = 12 if (st_dim_area > 1500 or st_small_cnt >= 10) else 8
        dbg = {
            "mode": "etalonda_dim_kam (soft)",
            "similarity": round(sim, 4),
            "missing_ratio": round(miss, 4),
            "coverage": round(cov, 4),
            "extra_ratio": round(extra, 4),
            "et_dim_area": et_dim_area,
            "st_dim_area": st_dim_area,
            "et_small_cnt": et_small_cnt,
            "st_small_cnt": st_small_cnt,
            "et_dash": et_dash,
            "st_dash": st_dash,
            "dash_ratio": round(dash_ratio, 4),
            "st_long_ratio": round(st_long, 4),
        }
        return pts, dbg, et_dim, st_dim

    area_ratio = float(np.clip(st_dim_area / max(et_dim_area, 1), 0.0, 1.2))
    cnt_ratio = float(np.clip(st_small_cnt / max(et_small_cnt, 1), 0.0, 1.2))
    score01 = 0.40 * min(area_ratio, 1.0) + 0.60 * min(cnt_ratio, 1.0)
    pts = int(np.clip(round(score01 * 12), 0, 12))

    dbg = {
        "mode": "ratio",
        "similarity": round(sim, 4),
        "missing_ratio": round(miss, 4),
        "coverage": round(cov, 4),
        "extra_ratio": round(extra, 4),
        "et_dim_area": et_dim_area,
        "st_dim_area": st_dim_area,
        "et_small_cnt": et_small_cnt,
        "st_small_cnt": st_small_cnt,
        "area_ratio": round(area_ratio, 4),
        "cnt_ratio": round(cnt_ratio, 4),
        "score01": round(score01, 4),
        "et_dash": et_dash,
        "st_dash": st_dash,
        "dash_ratio": round(dash_ratio, 4),
        "st_long_ratio": round(st_long, 4),
    }
    return pts, dbg, et_dim, st_dim

# ============================================================
# /18 - PROYEKSIYALAR
# ============================================================
def remove_border_long_lines_only(top_255: np.ndarray, band_ratio=0.06):
    H, W = top_255.shape[:2]
    band = int(min(H, W) * band_ratio)

    out = top_255.copy()
    long_mask = np.zeros_like(top_255)

    h_long = cv2.getStructuringElement(cv2.MORPH_RECT, (121, 1))
    v_long = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 121))

    regions = {
        "top":    (slice(0, band), slice(0, W)),
        "bottom": (slice(H - band, H), slice(0, W)),
        "left":   (slice(0, H), slice(0, band)),
        "right":  (slice(0, H), slice(W - band, W)),
    }

    for _, (ys, xs) in regions.items():
        roi = out[ys, xs]
        lh = cv2.morphologyEx(roi, cv2.MORPH_OPEN, h_long, iterations=1)
        lv = cv2.morphologyEx(roi, cv2.MORPH_OPEN, v_long, iterations=1)
        lm = cv2.bitwise_or(lh, lv)

        long_mask[ys, xs] = cv2.bitwise_or(long_mask[ys, xs], lm)

        roi2 = roi.copy()
        roi2[lm > 0] = 0
        out[ys, xs] = roi2

    return out, long_mask

def remove_tiny_components(mask_255: np.ndarray, min_area=20):
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_255 > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask_255)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out

def moving_average_1d(arr, k):
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(arr.astype(np.float32), kernel, mode="same")

def binary_close_1d(active_bool, gap):
    x = (active_bool.astype(np.uint8) * 255)[None, :]
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(gap)), 1))
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, ker, iterations=1)
    return (x[0] > 0)

def find_runs(active_bool, min_len):
    runs = []
    in_run = False
    s = 0
    for i, v in enumerate(active_bool):
        if v and not in_run:
            s = i
            in_run = True
        elif not v and in_run:
            e = i
            if e - s >= min_len:
                runs.append((s, e))
            in_run = False
    if in_run:
        e = len(active_bool)
        if e - s >= min_len:
            runs.append((s, e))
    return runs

def build_top_ycut_candidates(lines_255, base_ratios=(0.52, 0.58, 0.64, 0.70)):
    H, W = lines_255.shape[:2]
    candidates = set(int(H * r) for r in base_ratios)

    y0 = int(H * 0.62)
    prof = (lines_255[:y0, :] > 0).sum(axis=1).astype(np.float32)
    prof_s = moving_average_1d(prof, max(9, int(H * 0.02)))

    lo = int(y0 * 0.35)
    hi = int(y0 * 0.92)
    if hi > lo + 5:
        seg = prof_s[lo:hi]
        idx = int(np.argmin(seg)) + lo
        soft_ycut = max(int(H * 0.50), idx)
        candidates.add(int(soft_ycut))

    candidates = sorted(c for c in candidates if int(H * 0.45) <= c <= int(H * 0.78))
    return candidates

def split_run_by_valleys(roi_255, global_x1, min_subrun_w=20, valley_rel=0.45, smooth_k=11):
    H, W = roi_255.shape[:2]
    if W <= min_subrun_w * 2:
        return [(global_x1, global_x1 + W)]

    xprof = (roi_255 > 0).sum(axis=0).astype(np.float32)
    xprof_s = moving_average_1d(xprof, smooth_k)
    mx = float(xprof_s.max()) if len(xprof_s) else 0.0
    if mx <= 0:
        return []

    valley_thr = valley_rel * mx
    low = xprof_s <= valley_thr

    cuts = []
    in_low = False
    s = 0
    for i, v in enumerate(low):
        if v and not in_low:
            s = i
            in_low = True
        elif not v and in_low:
            e = i
            if e - s >= max(4, int(W * 0.025)):
                cuts.append((s, e))
            in_low = False
    if in_low:
        e = len(low)
        if e - s >= max(4, int(W * 0.025)):
            cuts.append((s, e))

    if not cuts:
        return [(global_x1, global_x1 + W)]

    segments = []
    prev = 0
    for cs, ce in cuts:
        if cs - prev >= min_subrun_w:
            segments.append((prev, cs))
        prev = ce
    if W - prev >= min_subrun_w:
        segments.append((prev, W))

    if len(segments) <= 1:
        return [(global_x1, global_x1 + W)]

    return [(global_x1 + a, global_x1 + b) for a, b in segments]

def build_box_from_xrun(top_clean, x1, x2):
    roi = top_clean[:, x1:x2]
    ys, xs = np.where(roi > 0)
    if len(xs) == 0:
        return None
    bx1 = int(x1 + xs.min())
    bx2 = int(x1 + xs.max() + 1)
    by1 = int(ys.min())
    by2 = int(ys.max() + 1)
    return [bx1, by1, bx2, by2]

def split_box_by_col_occupancy(top_clean, box, min_seg_w=18, valley_rel=0.20, smooth_k=9):
    x1, y1, x2, y2 = box
    roi = top_clean[y1:y2, x1:x2]
    H, W = roi.shape[:2]
    if W <= min_seg_w * 2:
        return [box]

    scan_h = max(20, int(H * 0.88))
    scan = roi[:scan_h, :]
    occ = (scan > 0).mean(axis=0).astype(np.float32)
    occ_s = moving_average_1d(occ, smooth_k)

    mx = float(occ_s.max()) if len(occ_s) else 0.0
    if mx <= 0:
        return [box]

    low = occ_s <= max(0.01, valley_rel * mx)
    cuts = []
    in_low = False
    s = 0
    for i, v in enumerate(low):
        if v and not in_low:
            s = i
            in_low = True
        elif not v and in_low:
            e = i
            if e - s >= max(4, int(W * 0.025)):
                cuts.append((s, e))
            in_low = False
    if in_low:
        e = len(low)
        if e - s >= max(4, int(W * 0.025)):
            cuts.append((s, e))

    if not cuts:
        return [box]

    segs = []
    prev = 0
    for cs, ce in cuts:
        if cs - prev >= min_seg_w:
            segs.append((prev, cs))
        prev = ce
    if W - prev >= min_seg_w:
        segs.append((prev, W))

    if len(segs) <= 1:
        return [box]

    out = []
    for a, b in segs:
        sub = build_box_from_xrun(top_clean, x1 + a, x1 + b)
        if sub is not None:
            out.append(sub)
    return out if out else [box]

def split_box_by_component_groups(top_clean, box, min_seg_w=18, xgap_ratio=0.07):
    x1, y1, x2, y2 = box
    roi = top_clean[y1:y2, x1:x2]
    H, W = roi.shape[:2]
    if W <= min_seg_w * 2 or H < 20:
        return [box]

    scan_h = max(20, int(H * 0.92))
    scan = roi[:scan_h, :].copy()

    vk = max(9, int(scan_h * 0.18))
    vker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    vmask = cv2.morphologyEx(scan, cv2.MORPH_OPEN, vker, iterations=1)

    hk = max(9, int(W * 0.10))
    hker = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    hmask = cv2.morphologyEx(scan, cv2.MORPH_OPEN, hker, iterations=1)

    hlong_k = max(15, int(W * 0.24))
    hlong = cv2.morphologyEx(
        scan,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (hlong_k, 1)),
        iterations=1
    )

    anchor = cv2.bitwise_or(vmask, hmask)
    anchor[hlong > 0] = 0
    anchor = cv2.morphologyEx(anchor, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    anchor = remove_tiny_components(anchor, min_area=max(10, int(0.0005 * W * H)))

    num, labels, stats, _ = cv2.connectedComponentsWithStats((anchor > 0).astype(np.uint8), connectivity=8)

    comps = []
    min_cc_area = max(10, int(0.0006 * W * H))
    min_cc_w = max(3, int(W * 0.015))
    min_cc_h = max(8, int(H * 0.10))

    for i in range(1, num):
        cx = stats[i, cv2.CC_STAT_LEFT]
        cy = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        ca = stats[i, cv2.CC_STAT_AREA]

        if cw >= int(W * 0.45) and ch <= int(H * 0.16):
            continue
        if ca < min_cc_area:
            continue
        if cw < min_cc_w:
            continue
        if ch < min_cc_h and ca < int(2.0 * min_cc_area):
            continue

        comps.append({
            "x1": int(cx), "x2": int(cx + cw), "y1": int(cy), "y2": int(cy + ch), "area": int(ca)
        })

    if len(comps) <= 1:
        return [box]

    comps = sorted(comps, key=lambda c: c["x1"])
    xgap = max(8, int(W * xgap_ratio))

    groups = []
    cur = [comps[0]]
    for c in comps[1:]:
        prev = cur[-1]
        gap = c["x1"] - prev["x2"]
        if gap >= xgap:
            groups.append(cur)
            cur = [c]
        else:
            cur.append(c)
    groups.append(cur)

    if len(groups) <= 1:
        return [box]

    out = []
    for g in groups:
        gx1 = min(t["x1"] for t in g)
        gx2 = max(t["x2"] for t in g)
        sub = build_box_from_xrun(top_clean, x1 + gx1, x1 + gx2)
        if sub is not None:
            bw = sub[2] - sub[0]
            if bw >= min_seg_w:
                out.append(sub)

    return out if len(out) >= 2 else [box]

def refine_box_full_extent(top_clean, box, xpad_ratio=0.04, ypad_ratio=0.04, min_col_occ=0.008, min_row_occ=0.008):
    H, W = top_clean.shape[:2]
    x1, y1, x2, y2 = box

    xpad = max(10, int(W * xpad_ratio))
    ypad = max(10, int(H * ypad_ratio))
    rx1 = max(0, x1 - xpad)
    rx2 = min(W, x2 + xpad)
    ry1 = max(0, y1 - ypad)
    ry2 = min(H, y2 + ypad)

    roi = top_clean[ry1:ry2, rx1:rx2]
    if roi.size == 0 or (roi > 0).sum() == 0:
        return box

    col_occ = (roi > 0).mean(axis=0).astype(np.float32)
    row_occ = (roi > 0).mean(axis=1).astype(np.float32)
    col_occ_s = moving_average_1d(col_occ, max(5, int(roi.shape[1] * 0.05)))
    row_occ_s = moving_average_1d(row_occ, max(5, int(roi.shape[0] * 0.05)))

    col_idx = np.where(col_occ_s >= min_col_occ)[0]
    row_idx = np.where(row_occ_s >= min_row_occ)[0]

    if len(col_idx) == 0 or len(row_idx) == 0:
        ys, xs = np.where(roi > 0)
        if len(xs) == 0:
            return box
        nx1 = rx1 + int(xs.min())
        nx2 = rx1 + int(xs.max()) + 1
        ny1 = ry1 + int(ys.min())
        ny2 = ry1 + int(ys.max()) + 1
        return [nx1, ny1, nx2, ny2]

    nx1 = rx1 + int(col_idx.min())
    nx2 = rx1 + int(col_idx.max()) + 1
    ny1 = ry1 + int(row_idx.min())
    ny2 = ry1 + int(row_idx.max()) + 1

    nx1 = max(0, nx1 - 4); ny1 = max(0, ny1 - 4)
    nx2 = min(W, nx2 + 4); ny2 = min(H, ny2 + 4)
    return [nx1, ny1, nx2, ny2]

def refine_box_on_full_image(full_lines_255, box, xpad_ratio=0.01, ypad_ratio=0.10, min_row_occ=0.004,
                             max_up_expand_ratio=0.02, max_down_expand_ratio=0.18):
    H, W = full_lines_255.shape[:2]
    x1, y1, x2, y2 = box

    xpad = max(2, int(W * xpad_ratio))
    ypad = max(12, int(H * ypad_ratio))
    rx1 = max(0, x1 - xpad)
    rx2 = min(W, x2 + xpad)
    ry1 = max(0, y1 - ypad)
    ry2 = min(H, y2 + ypad)

    roi = full_lines_255[ry1:ry2, rx1:rx2]
    if roi.size == 0 or (roi > 0).sum() == 0:
        return box

    row_occ = (roi > 0).mean(axis=1).astype(np.float32)
    row_occ_s = moving_average_1d(row_occ, max(5, int(roi.shape[0] * 0.04)))
    row_idx = np.where(row_occ_s >= min_row_occ)[0]

    if len(row_idx) == 0:
        ys, xs = np.where(roi > 0)
        if len(xs) == 0:
            return box
        ny1 = ry1 + int(ys.min())
        ny2 = ry1 + int(ys.max()) + 1
    else:
        ny1 = ry1 + int(row_idx.min())
        ny2 = ry1 + int(row_idx.max()) + 1

    ny1 = max(0, ny1 - 3)
    ny2 = min(H, ny2 + 3)

    max_up_expand = max(4, int(H * max_up_expand_ratio))
    max_down_expand = max(12, int(H * max_down_expand_ratio))
    ny1 = max(ny1, y1 - max_up_expand)
    ny2 = min(max(ny2, y2), y2 + max_down_expand)

    nx1 = max(0, x1 - 1)
    nx2 = min(W, x2 + 1)
    return [nx1, ny1, nx2, ny2]

def split_box_by_row_occupancy(top_clean, box, min_seg_h=18, valley_rel=0.20, smooth_k=9):
    x1, y1, x2, y2 = box
    roi = top_clean[y1:y2, x1:x2]
    H, W = roi.shape[:2]
    if H <= min_seg_h * 2:
        return [box]

    occ = (roi > 0).mean(axis=1).astype(np.float32)
    occ_s = moving_average_1d(occ, smooth_k)
    mx = float(occ_s.max()) if len(occ_s) else 0.0
    if mx <= 0:
        return [box]

    low = occ_s <= max(0.01, valley_rel * mx)
    cuts = []
    in_low = False
    s = 0
    for i, v in enumerate(low):
        if v and not in_low:
            s = i
            in_low = True
        elif not v and in_low:
            e = i
            if e - s >= max(4, int(H * 0.025)):
                cuts.append((s, e))
            in_low = False
    if in_low:
        e = len(low)
        if e - s >= max(4, int(H * 0.025)):
            cuts.append((s, e))

    if not cuts:
        return [box]

    segs = []
    prev = 0
    for cs, ce in cuts:
        if cs - prev >= min_seg_h:
            segs.append((prev, cs))
        prev = ce
    if H - prev >= min_seg_h:
        segs.append((prev, H))

    if len(segs) <= 1:
        return [box]

    out = []
    for a, b in segs:
        sub_roi = top_clean[y1 + a:y1 + b, x1:x2]
        ys, xs = np.where(sub_roi > 0)
        if len(xs) == 0:
            continue
        bx1 = x1 + int(xs.min())
        bx2 = x1 + int(xs.max()) + 1
        by1 = y1 + a + int(ys.min())
        by2 = y1 + a + int(ys.max()) + 1
        out.append([bx1, by1, bx2, by2])
    return out if len(out) >= 2 else [box]

def orientation_features(roi_255):
    ys, xs = np.where(roi_255 > 0)
    if len(xs) == 0:
        return {"hv_line_ratio": 0.0, "hv_len": 0.0, "diag_len": 0.0, "hv_struct_ratio": 0.0, "total_pixels": 0}

    H, W = roi_255.shape[:2]
    total_pixels = int((roi_255 > 0).sum())

    hk = max(9, min(31, int(W * 0.12)))
    vk = max(9, min(31, int(H * 0.12)))
    hker = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    vker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

    h_part = cv2.morphologyEx(roi_255, cv2.MORPH_OPEN, hker, iterations=1)
    v_part = cv2.morphologyEx(roi_255, cv2.MORPH_OPEN, vker, iterations=1)
    hv_part = cv2.bitwise_or(h_part, v_part)
    hv_struct_ratio = float((hv_part > 0).sum() / max(1, total_pixels))

    min_len = max(12, int(min(H, W) * 0.18))
    lines = cv2.HoughLinesP(roi_255, 1, np.pi / 180, threshold=18, minLineLength=min_len, maxLineGap=6)

    hv_len = 0.0
    diag_len = 0.0
    if lines is not None:
        for ln in lines[:, 0]:
            x1, y1, x2, y2 = ln
            dx = x2 - x1
            dy = y2 - y1
            L = float(np.hypot(dx, dy))
            ang = abs(np.degrees(np.arctan2(dy, dx))) % 180.0
            if ang > 90:
                ang = 180 - ang
            if ang <= 12 or abs(ang - 90) <= 12:
                hv_len += L
            elif 18 <= ang <= 72:
                diag_len += L

    hv_line_ratio = float(hv_len / max(1e-6, hv_len + diag_len))
    return {
        "hv_line_ratio": hv_line_ratio,
        "hv_len": hv_len,
        "diag_len": diag_len,
        "hv_struct_ratio": hv_struct_ratio,
        "total_pixels": total_pixels
    }

def fallback_single_projection_box(top_clean, min_box_w, min_box_h, min_area, hv_struct_thr_soft=0.14, hv_line_thr_soft=0.34):
    H, W = top_clean.shape[:2]
    mask = (top_clean > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best_box = None
    best_score = -1e9
    best_dbg = None

    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < max(20, int(min_area * 0.45)): continue
        if w < max(10, int(min_box_w * 0.70)): continue
        if h < max(12, int(min_box_h * 0.70)): continue
        if w >= int(W * 0.55) and h <= int(H * 0.12): continue

        roi = np.zeros((h, w), dtype=np.uint8)
        comp = (labels[y:y + h, x:x + w] == i)
        roi[comp] = 255

        feat = orientation_features(roi)
        hvs = float(feat["hv_struct_ratio"])
        hvl = float(feat["hv_line_ratio"])
        hv_len = float(feat["hv_len"])
        diag_len = float(feat["diag_len"])

        soft_ok = (
            (hvs >= hv_struct_thr_soft and hvl >= hv_line_thr_soft)
            or (hvs >= 0.22 and hv_len >= diag_len * 0.80)
            or (hvl >= 0.55 and area >= max(40, int(min_area * 0.65)))
        )
        if not soft_ok:
            continue

        score = (
            3.5 * hvs + 2.5 * hvl + 0.35 * np.log1p(area) +
            0.15 * min(h / max(1.0, w), 3.0) -
            0.20 * (diag_len / max(1.0, hv_len + diag_len))
        )

        box = [int(x), int(y), int(x + w), int(y + h)]
        box = refine_box_full_extent(top_clean, box, xpad_ratio=0.04, ypad_ratio=0.04, min_col_occ=0.006, min_row_occ=0.006)

        if score > best_score:
            best_score = score
            best_box = box
            best_dbg = {
                "box": box,
                "score": round(float(score), 3),
                "hv_struct_ratio": round(hvs, 3),
                "hv_line_ratio": round(hvl, 3),
                "hv_len": round(hv_len, 1),
                "diag_len": round(diag_len, 1),
                "area": int(area),
                "reason": "single_projection_fallback",
            }

    return best_box, best_dbg

def is_projection_box(top_clean, box, min_box_w, min_box_h, min_area, hv_struct_thr, hv_line_thr):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    area = int((top_clean[y1:y2, x1:x2] > 0).sum())

    if bw < min_box_w or bh < min_box_h or area < min_area:
        return False, {"box": box, "w": bw, "h": bh, "area": area, "keep": False, "reason": "size"}

    roi_box = top_clean[y1:y2, x1:x2]
    feat = orientation_features(roi_box)
    proj_like = (
        (feat["hv_struct_ratio"] >= hv_struct_thr and feat["hv_line_ratio"] >= hv_line_thr)
        or (feat["hv_struct_ratio"] >= 0.48 and feat["hv_len"] >= feat["diag_len"] * 1.15)
    )

    dbg = {
        "box": box,
        "w": bw, "h": bh, "area": area,
        "hv_struct_ratio": round(feat["hv_struct_ratio"], 3),
        "hv_line_ratio": round(feat["hv_line_ratio"], 3),
        "hv_len": round(feat["hv_len"], 1),
        "diag_len": round(feat["diag_len"], 1),
        "keep": bool(proj_like),
        "reason": "projection" if proj_like else "visual/other"
    }
    return proj_like, dbg

def box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = box_area(a) + box_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union

def dedupe_boxes(boxes, iou_thr=0.45):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: box_area(b), reverse=True)
    keep = []
    for b in boxes:
        ok = True
        for kb in keep:
            if box_iou(b, kb) >= iou_thr:
                ok = False
                break
        if ok:
            keep.append(b)
    return sorted(keep, key=lambda b: b[0])

def draw_boxes_on_rgb(gray_or_rgb, boxes, color=(0, 255, 0), thickness=2):
    if len(gray_or_rgb.shape) == 2:
        out = cv2.cvtColor(gray_or_rgb, cv2.COLOR_GRAY2RGB)
    else:
        out = gray_or_rgb.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out

def build_visual_debug_region(lines_255, display_h, mx_ratio=0.03, my_ratio=0.02, band_ratio=0.06):
    H, W = lines_255.shape[:2]
    display_h = min(H, int(display_h))

    vis_raw = lines_255[:display_h, :].copy()
    vis_clean, vis_long_mask = remove_border_long_lines_only(vis_raw, band_ratio=band_ratio)
    mx = int(W * mx_ratio)
    my = int(display_h * my_ratio)

    vis_clean[:my, :] = 0
    vis_clean[:, :mx] = 0
    vis_clean[:, W - mx:] = 0
    vis_clean = remove_tiny_components(vis_clean, min_area=max(18, int(0.00008 * vis_clean.size)))
    vis_clean = cv2.morphologyEx(vis_clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    vis_candidate = cv2.cvtColor(vis_clean, cv2.COLOR_GRAY2RGB)
    return vis_raw, vis_clean, vis_long_mask, vis_candidate

def detect_boxes_on_fixed_top(
    lines_255: np.ndarray,
    ycut: int,
    mx_ratio=0.03,
    my_ratio=0.02,
    band_ratio=0.06,
    profile_thr_rel=0.17,
    min_run_ratio=0.040,
    gap_close_ratio=0.014,
    min_box_w_ratio=0.05,
    min_box_h_ratio=0.08,
    min_area_ratio=0.0018,
    hv_struct_thr=0.26,
    hv_line_thr=0.50,
):
    H, W = lines_255.shape[:2]
    top_raw = lines_255[:ycut, :].copy()
    top_clean, long_mask = remove_border_long_lines_only(top_raw, band_ratio=band_ratio)

    mx = int(W * mx_ratio)
    my = int(ycut * my_ratio)
    top_clean[:my, :] = 0
    top_clean[:, :mx] = 0
    top_clean[:, W - mx:] = 0

    top_clean = remove_tiny_components(top_clean, min_area=max(18, int(0.00008 * top_clean.size)))
    top_clean = cv2.morphologyEx(top_clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    xprof = (top_clean > 0).sum(axis=0).astype(np.float32)
    sm_k = max(9, int(W * 0.02))
    xprof_s = moving_average_1d(xprof, sm_k)

    thr = max(4.0, float(profile_thr_rel * xprof_s.max()))
    active = xprof_s >= thr
    active = binary_close_1d(active, gap=max(5, int(W * gap_close_ratio)))
    runs = find_runs(active, min_len=max(12, int(W * min_run_ratio)))

    boxes = []
    group_debug = []

    min_box_w = max(16, int(W * min_box_w_ratio))
    min_box_h = max(16, int(ycut * min_box_h_ratio))
    min_area = max(120, int(W * ycut * min_area_ratio))

    for (rx1, rx2) in runs:
        outer_roi = top_clean[:, rx1:rx2]
        if (outer_roi > 0).sum() == 0:
            continue

        run_w = rx2 - rx1
        subruns = split_run_by_valleys(outer_roi, global_x1=rx1, min_subrun_w=max(18, int(W * 0.05)), valley_rel=0.45, smooth_k=max(9, int(W * 0.012)))
        if len(subruns) == 1 and run_w >= int(W * 0.22):
            subruns = split_run_by_valleys(outer_roi, global_x1=rx1, min_subrun_w=max(12, int(W * 0.035)), valley_rel=0.68, smooth_k=max(5, int(W * 0.006)))

        for (sx1, sx2) in subruns:
            box = build_box_from_xrun(top_clean, sx1, sx2)
            if box is None:
                continue
            ok, dbg_one = is_projection_box(
                top_clean, box,
                min_box_w=min_box_w, min_box_h=min_box_h, min_area=min_area,
                hv_struct_thr=hv_struct_thr, hv_line_thr=hv_line_thr
            )
            group_debug.append(dbg_one)
            if ok:
                boxes.append(box)

    boxes = dedupe_boxes(boxes, iou_thr=0.45)
    forced_boxes = []
    forced_debug = []

    for b in boxes:
        bw = b[2] - b[0]
        bh = b[3] - b[1]
        parts = [b]

        if bh >= int(ycut * 0.22):
            parts_y = []
            for p in parts:
                sp = split_box_by_row_occupancy(top_clean, p, min_seg_h=max(16, int(ycut * 0.07)), valley_rel=0.20, smooth_k=max(5, int(ycut * 0.015)))
                parts_y.extend(sp)
            parts = parts_y if len(parts_y) > 0 else parts

        parts_x = []
        for p in parts:
            pbw = p[2] - p[0]
            if pbw >= int(W * 0.16):
                sp = split_box_by_component_groups(top_clean, p, min_seg_w=max(14, int(W * 0.040)), xgap_ratio=0.06)
                if len(sp) == 1:
                    sp = split_box_by_col_occupancy(top_clean, p, min_seg_w=max(14, int(W * 0.040)), valley_rel=0.20, smooth_k=max(5, int(W * 0.008)))
                parts_x.extend(sp)
            else:
                parts_x.append(p)
        parts = parts_x if len(parts_x) > 0 else parts

        refined_parts = []
        for p in parts:
            rp = refine_box_full_extent(top_clean, p, xpad_ratio=0.05, ypad_ratio=0.05, min_col_occ=0.008, min_row_occ=0.008)
            refined_parts.append(rp)

        for pb in refined_parts:
            ok, dbg_one = is_projection_box(
                top_clean, pb,
                min_box_w=min_box_w, min_box_h=min_box_h, min_area=min_area,
                hv_struct_thr=hv_struct_thr, hv_line_thr=hv_line_thr
            )
            forced_debug.append(dbg_one)
            if ok:
                forced_boxes.append(pb)

    boxes = dedupe_boxes(forced_boxes, iou_thr=0.28)
    if len(boxes) > 3:
        boxes = sorted(boxes, key=lambda b: box_area(b), reverse=True)[:3]
        boxes = sorted(boxes, key=lambda b: b[0])

    group_debug.extend(forced_debug)
    boxes = [refine_box_full_extent(top_clean, b, xpad_ratio=0.06, ypad_ratio=0.06, min_col_occ=0.006, min_row_occ=0.006) for b in boxes]
    boxes = dedupe_boxes(boxes, iou_thr=0.28)

    if len(boxes) == 0:
        fb_box, fb_dbg = fallback_single_projection_box(
            top_clean,
            min_box_w=min_box_w, min_box_h=min_box_h, min_area=min_area,
            hv_struct_thr_soft=0.14, hv_line_thr_soft=0.34
        )
        if fb_box is not None:
            boxes = [fb_box]
            group_debug.append(fb_dbg)

    candidate_vis_final = cv2.cvtColor(top_clean, cv2.COLOR_GRAY2RGB)
    for b in boxes:
        cv2.rectangle(candidate_vis_final, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

    dbg = {
        "ycut": int(ycut),
        "thr": round(float(thr), 2),
        "runs": runs,
        "valid_boxes": boxes,
        "max_box_w_ratio": round(max([(b[2] - b[0]) / W for b in boxes], default=0.0), 3),
        "fallback_used": bool(len(boxes) == 1 and any(isinstance(g, dict) and g.get("reason") == "single_projection_fallback" for g in group_debug)),
        "groups": group_debug,
    }

    return len(boxes), dbg, top_raw, top_clean, long_mask, candidate_vis_final, boxes, ycut, xprof_s

def count_projection_groups_final(
    lines_255: np.ndarray,
    top_ratio=0.60,
    mx_ratio=0.03,
    my_ratio=0.02,
    band_ratio=0.06,
    profile_thr_rel=0.17,
    min_run_ratio=0.040,
    gap_close_ratio=0.014,
    min_box_w_ratio=0.05,
    min_box_h_ratio=0.08,
    min_area_ratio=0.0018,
    hv_struct_thr=0.26,
    hv_line_thr=0.50,
):
    H, W = lines_255.shape[:2]
    ycut_candidates = build_top_ycut_candidates(lines_255, base_ratios=(0.52, 0.58, 0.64, 0.70))
    all_results = []

    for ycut in ycut_candidates:
        cnt, dbg, top_raw, top_clean, long_mask, candidate_vis, boxes, ycut_used, xprof_s = detect_boxes_on_fixed_top(
            lines_255, ycut=ycut, mx_ratio=mx_ratio, my_ratio=my_ratio, band_ratio=band_ratio,
            profile_thr_rel=profile_thr_rel, min_run_ratio=min_run_ratio, gap_close_ratio=gap_close_ratio,
            min_box_w_ratio=min_box_w_ratio, min_box_h_ratio=min_box_h_ratio, min_area_ratio=min_area_ratio,
            hv_struct_thr=hv_struct_thr, hv_line_thr=hv_line_thr,
        )

        widths = [(b[2] - b[0]) for b in boxes]
        heights = [(b[3] - b[1]) for b in boxes]
        max_w_ratio = (max(widths) / W) if widths else 1.0
        max_h_ratio = (max(heights) / max(1, ycut_used)) if heights else 1.0

        spread_score_x = 0.0
        spread_score_y = 0.0
        if len(boxes) >= 2:
            xs = [((b[0] + b[2]) / 2.0) for b in boxes]
            ys = [((b[1] + b[3]) / 2.0) for b in boxes]
            spread_score_x = float(np.std(xs)) / max(1.0, W)
            spread_score_y = float(np.std(ys)) / max(1.0, ycut_used)

        wide_single_penalty = 0.0
        if cnt == 1 and max_w_ratio > 0.18:
            wide_single_penalty += 60.0 + 120.0 * (max_w_ratio - 0.18)

        tall_single_penalty = 0.0
        if cnt == 1 and max_h_ratio > 0.30:
            tall_single_penalty += 70.0 + 110.0 * (max_h_ratio - 0.30)

        score = (
            140.0 * cnt +
            30.0 * spread_score_x +
            40.0 * spread_score_y -
            30.0 * max(0.0, max_w_ratio - 0.16) -
            wide_single_penalty - tall_single_penalty
        )

        all_results.append({
            "score": float(score),
            "cnt": int(cnt),
            "dbg": dbg,
            "top_raw": top_raw,
            "top_clean": top_clean,
            "long_mask": long_mask,
            "candidate_vis": candidate_vis,
            "boxes": boxes,
            "ycut": int(ycut_used),
            "xprof_s": xprof_s,
        })

    best = sorted(all_results, key=lambda r: (r["score"], r["cnt"]), reverse=True)[0]
    dbg = dict(best["dbg"])
    dbg["candidate_ycuts"] = ycut_candidates
    dbg["all_scores"] = [
        {"ycut": int(r["ycut"]), "cnt": int(r["cnt"]), "score": round(float(r["score"]), 3), "valid_boxes": r["boxes"]}
        for r in all_results
    ]

    return (
        int(best["cnt"]), dbg, best["top_raw"], best["top_clean"], best["long_mask"],
        best["candidate_vis"], best["boxes"], int(best["ycut"]), best["xprof_s"]
    )

def score_projections_18(st_aligned: np.ndarray, metrics: dict, expected_proj=3):
    st_lines = bin_to_lines(st_aligned)
    sim = max(0.0, min(float(metrics.get("similarity", 0.0)), 1.0))
    cov = float(metrics.get("coverage", 0.0))
    miss = float(metrics.get("missing_ratio", 1.0))

    hard_bad = (cov < 0.18 and sim < 0.28)
    et_dash = 0
    st_dash, _ = dashed_like_count(st_lines)
    dash_ratio = float((st_dash + 1) / (et_dash + 1))
    noise_override = (sim <= 0.02) and (st_dash >= 70) and (dash_ratio >= 2.2)

    if hard_bad or noise_override:
        return {
            "score": 0,
            "student_projections": 0,
            "boxes": [],
            "boxes_vis": [],
            "top_clean": np.zeros_like(st_lines),
            "top_raw": np.zeros_like(st_lines),
            "long_mask": np.zeros_like(st_lines),
            "candidate_vis": cv2.cvtColor(st_lines, cv2.COLOR_GRAY2RGB),
            "ycut": 0,
            "xprof_s": np.array([]),
            "debug": {
                "sim": round(sim, 3),
                "cov": round(cov, 3),
                "miss": round(miss, 3),
                "hard_bad": hard_bad,
                "st_dash": st_dash,
                "dash_ratio": round(dash_ratio, 2),
                "noise_override": noise_override,
            },
        }

    st_proj, dbg_col, top_raw, top_clean, long_mask, candidate_vis, boxes, ycut, xprof_s = count_projection_groups_final(
        st_lines,
        top_ratio=0.60,
        mx_ratio=0.03,
        my_ratio=0.02,
        band_ratio=0.06,
        profile_thr_rel=0.17,
        min_run_ratio=0.040,
        gap_close_ratio=0.014,
        min_box_w_ratio=0.05,
        min_box_h_ratio=0.08,
        min_area_ratio=0.0018,
        hv_struct_thr=0.26,
        hv_line_thr=0.50,
    )

    boxes_vis = boxes.copy()
    if len(boxes_vis) == 1:
        boxes_vis = [refine_box_on_full_image(
            st_lines,
            boxes_vis[0],
            xpad_ratio=0.01,
            ypad_ratio=0.10,
            min_row_occ=0.004,
            max_up_expand_ratio=0.02,
            max_down_expand_ratio=0.18
        )]
        boxes_vis = dedupe_boxes(boxes_vis, iou_thr=0.28)

    st_proj = int(np.clip(st_proj, 0, expected_proj))
    proj_pts = int(6 * st_proj)

    return {
        "score": proj_pts,
        "student_projections": st_proj,
        "boxes": boxes,
        "boxes_vis": boxes_vis,
        "top_clean": top_clean,
        "top_raw": top_raw,
        "long_mask": long_mask,
        "candidate_vis": candidate_vis,
        "ycut": ycut,
        "xprof_s": xprof_s,
        "debug": dbg_col,
    }

# ============================================================
# /10 - PROYEKSIYALAR ICHIDA QIRQIM
# ============================================================
STEP6_TOTAL_SCORE = 10.0
STEP6_PER_PROJ = STEP6_TOTAL_SCORE / 3.0
STEP6_HYBRID_THR = 5.2

def step6_odd(x: int) -> int:
    return x if x % 2 == 1 else x + 1

def step6_ensure_gray(img):
    if img is None:
        raise ValueError("Input image is None")
    if len(img.shape) == 2:
        gray = img.copy()
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray

def step6_clamp01(x):
    return max(0.0, min(1.0, float(x)))

def step6_box_area(b):
    x1, y1, x2, y2 = map(int, b)
    return max(0, x2 - x1) * max(0, y2 - y1)

def step6_crop_box(img, box, pad_ratio=0.04):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    px = max(4, int((x2 - x1) * pad_ratio))
    py = max(4, int((y2 - y1) * pad_ratio))
    return img[max(0, y1 - py):min(H, y2 + py), max(0, x1 - px):min(W, x2 + px)].copy()

def step6_auto_canny(img, sigma=0.33):
    v = float(np.median(img))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    if upper <= lower:
        lower, upper = 30, 120
    return cv2.Canny(img, lower, upper, L2gradient=True)

def step6_cluster_positions(vals, min_gap):
    if len(vals) == 0:
        return []
    vals = sorted([float(v) for v in vals])
    clusters = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - np.mean(clusters[-1])) <= min_gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(np.mean(c)) for c in clusters]

def step6_draw_preview(clean_img, boxes, step6_out=None):
    if len(clean_img.shape) == 2:
        vis = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2BGR)
    else:
        vis = clean_img.copy()

    boxes = [tuple(map(int, b)) for b in boxes]
    for i, b in enumerate(boxes, start=1):
        x1, y1, x2, y2 = map(int, b)
        color = (255, 140, 0)
        label = f"P{i}"
        if step6_out is not None and i <= len(step6_out["results"]):
            ok = step6_out["results"][i - 1]["is_section"]
            sc = step6_out["results"][i - 1]["score"]
            color = (0, 180, 0) if ok else (0, 0, 255)
            label = f"P{i} | {'QIRQIM' if ok else 'NO'} | {sc:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return vis

def step6_ink_mask_robust(gray_or_bgr):
    gray = step6_ensure_gray(gray_or_bgr)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    block = step6_odd(max(21, int(min(gray.shape[:2]) * 0.10)))
    bw2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 7)

    def to_ink(bw):
        if np.mean(bw) < 127:
            bw = cv2.bitwise_not(bw)
        ink = cv2.bitwise_not(bw)
        ink = cv2.medianBlur(ink, 3)
        return ink

    ink1 = to_ink(bw1)
    ink2 = to_ink(bw2)

    area = gray.shape[0] * gray.shape[1]
    fill1 = float(np.count_nonzero(ink1)) / max(area, 1.0)
    fill2 = float(np.count_nonzero(ink2)) / max(area, 1.0)
    target = 0.08

    def score_fill(fill):
        if 0.006 <= fill <= 0.30:
            return 1.0 - abs(fill - target)
        return -abs(fill - target)

    ink = ink1 if score_fill(fill1) >= score_fill(fill2) else ink2
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    return ink

def step6_extract_hatch_features(roi_img, debug=False):
    gray = step6_ensure_gray(roi_img)
    h, w = gray.shape[:2]
    area = float(h * w)
    if min(h, w) < 26:
        return {"is_section": False, "diag_coverage": 0.0, "diag_components": 0, "diag_line_count": 0,
                "dom_line_count": 0, "dom_ratio": 0.0, "dom_angle": 0.0, "angle_std": 999.0,
                "fill_ratio": 0.0, "interior_share": 0.0, "stripe_count": 0, "gap_cv": 999.0,
                "median_gap_norm": 0.0, "mean_len_ratio": 0.0, "hybrid_score": -999.0,
                "mode_hint": "none", "target_angle": 0.0, "mask": None}

    inner = np.zeros_like(gray, dtype=np.uint8)
    mx = max(4, int(w * 0.10)); my = max(4, int(h * 0.10))
    inner[my:h - my, mx:w - mx] = 255

    ink = step6_ink_mask_robust(gray)
    ink = cv2.bitwise_and(ink, inner)
    fill_ratio = float(np.count_nonzero(ink)) / max(area, 1.0)
    ink_nonzero = max(np.count_nonzero(ink), 1)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    edges = step6_auto_canny(g)
    edges = cv2.bitwise_and(edges, inner)
    base = min(h, w)

    hk = step6_odd(max(9, int(base * 0.12)))
    hker = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    vker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, hk))
    long_h = cv2.morphologyEx(edges, cv2.MORPH_OPEN, hker)
    long_v = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vker)
    edges_work = cv2.subtract(edges, cv2.bitwise_or(long_h, long_v))
    edges_work = cv2.morphologyEx(edges_work, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    lines = cv2.HoughLinesP(edges_work, rho=1, theta=np.pi / 180,
                            threshold=max(10, int(base * 0.07)),
                            minLineLength=max(10, int(base * 0.08)),
                            maxLineGap=max(2, int(base * 0.02)))

    allowed_targets = [45.0, 60.0]
    angle_tol = 11.0
    candidates = []
    total_candidate_len = {45.0: 0.0, 60.0: 0.0}
    total_candidate_cnt = {45.0: 0, 60.0: 0}

    if lines is not None:
        for l in lines[:, 0]:
            x1, y1, x2, y2 = map(int, l)
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < max(10, int(base * 0.07)):
                continue
            ang = math.degrees(math.atan2(dy, dx)) % 180.0
            if ang > 90:
                ang = 180.0 - ang
            nearest = min(allowed_targets, key=lambda t: abs(ang - t))
            if abs(ang - nearest) <= angle_tol:
                mxp = 0.5 * (x1 + x2)
                myp = 0.5 * (y1 + y2)
                candidates.append((nearest, ang, length, mxp, myp, x1, y1, x2, y2))
                total_candidate_len[nearest] += length
                total_candidate_cnt[nearest] += 1

    if len(candidates) == 0:
        return {"is_section": False, "diag_coverage": 0.0, "diag_components": 0, "diag_line_count": 0,
                "dom_line_count": 0, "dom_ratio": 0.0, "dom_angle": 0.0, "angle_std": 999.0,
                "fill_ratio": float(fill_ratio), "interior_share": 0.0, "stripe_count": 0, "gap_cv": 999.0,
                "median_gap_norm": 0.0, "mean_len_ratio": 0.0, "hybrid_score": -999.0,
                "mode_hint": "weak", "target_angle": 0.0, "mask": edges_work if debug else None}

    target_angle = max(allowed_targets, key=lambda t: (total_candidate_cnt[t], total_candidate_len[t]))
    selected = [c for c in candidates if c[0] == target_angle]

    line_count = len(candidates)
    dom_line_count = len(selected)
    dom_ratio = float(dom_line_count) / max(line_count, 1)

    selected_angles = [c[1] for c in selected]
    selected_lengths = [c[2] for c in selected]
    selected_mid = [(c[3], c[4]) for c in selected]

    dom_angle = float(np.median(selected_angles)) if len(selected_angles) > 0 else 0.0
    angle_std = float(np.std(selected_angles)) if len(selected_angles) > 1 else 0.0
    mean_len_ratio = float(np.mean(selected_lengths)) / max(math.hypot(w, h), 1.0) if len(selected_lengths) > 0 else 0.0

    rad = math.radians(target_angle)
    nx, ny = -math.sin(rad), math.cos(rad)
    positions = [(mxp * nx + myp * ny) for (mxp, myp) in selected_mid]
    min_gap = max(3.0, base * 0.020)
    stripe_centers = step6_cluster_positions(positions, min_gap=min_gap)
    stripe_count = len(stripe_centers)

    if len(stripe_centers) >= 2:
        gaps = np.diff(sorted(stripe_centers))
        median_gap_norm = float(np.median(gaps)) / max(base, 1.0)
        gap_cv = float(np.std(gaps) / max(np.mean(gaps), 1e-6)) if len(gaps) > 1 else 0.0
    else:
        median_gap_norm = 0.0
        gap_cv = 999.0

    selected_mask = np.zeros_like(edges_work)
    for (_, _, _, _, _, x1, y1, x2, y2) in selected:
        cv2.line(selected_mask, (x1, y1), (x2, y2), 255, 1)

    diag_coverage = float(np.count_nonzero(selected_mask)) / max(area, 1.0)
    interior_share = float(np.count_nonzero(selected_mask)) / max(ink_nonzero, 1)

    ncc, lbl, st, _ = cv2.connectedComponentsWithStats(selected_mask, connectivity=8)
    comp_count = 0
    for i in range(1, ncc):
        x, y, ww, hh, aa = st[i]
        if aa >= max(6, int(area * 0.00010)) and max(ww, hh) >= max(5, int(base * 0.04)):
            comp_count += 1

    hybrid_score = 0.0
    if stripe_count >= 3: hybrid_score += 2.0
    if stripe_count >= 4: hybrid_score += 1.5
    if stripe_count >= 5: hybrid_score += 0.8
    hybrid_score += 2.2 * step6_clamp01((dom_ratio - 0.55) / 0.30)
    hybrid_score += 1.6 * step6_clamp01((12.0 - angle_std) / 12.0)
    hybrid_score += 1.2 * step6_clamp01((dom_line_count - 4) / 5.0)
    hybrid_score += 0.8 * step6_clamp01((interior_share - 0.02) / 0.08)
    if 0.008 <= median_gap_norm <= 0.22: hybrid_score += 0.8
    if gap_cv <= 1.25: hybrid_score += 0.7
    if stripe_count < 3: hybrid_score -= 4.0
    if dom_line_count < 4: hybrid_score -= 2.0
    if dom_ratio < 0.60: hybrid_score -= 1.6
    if mean_len_ratio > 0.80: hybrid_score -= 1.2

    cad_vote = (
        target_angle in (45.0, 60.0) and stripe_count >= 4 and dom_line_count >= 5 and dom_ratio >= 0.72 and
        angle_std <= 8.5 and gap_cv <= 1.25 and mean_len_ratio >= 0.06 and mean_len_ratio <= 0.75 and interior_share >= 0.02
    )
    sketch_vote = (
        target_angle in (45.0, 60.0) and stripe_count >= 3 and dom_line_count >= 4 and dom_ratio >= 0.62 and
        angle_std <= 12.0 and gap_cv <= 1.60 and mean_len_ratio >= 0.05 and mean_len_ratio <= 0.78 and interior_share >= 0.015
    )
    hybrid_vote = target_angle in (45.0, 60.0) and hybrid_score >= STEP6_HYBRID_THR and stripe_count >= 3
    is_section = bool(cad_vote or sketch_vote or hybrid_vote)

    if cad_vote:
        mode_hint = "cad"
    elif sketch_vote:
        mode_hint = "sketch"
    elif hybrid_vote:
        mode_hint = "hybrid"
    else:
        mode_hint = "weak"

    return {
        "is_section": is_section,
        "diag_coverage": float(diag_coverage),
        "diag_components": int(comp_count),
        "diag_line_count": int(line_count),
        "dom_line_count": int(dom_line_count),
        "dom_ratio": float(dom_ratio),
        "dom_angle": float(dom_angle),
        "angle_std": float(angle_std),
        "fill_ratio": float(fill_ratio),
        "interior_share": float(interior_share),
        "stripe_count": int(stripe_count),
        "gap_cv": float(gap_cv),
        "median_gap_norm": float(median_gap_norm),
        "mean_len_ratio": float(mean_len_ratio),
        "hybrid_score": float(hybrid_score),
        "mode_hint": mode_hint,
        "target_angle": float(target_angle),
        "mask": selected_mask if debug else None
    }

def step6_run_only_selected3(clean_img, selected_boxes, debug=False):
    boxes = [tuple(map(int, b)) for b in selected_boxes if len(b) == 4 and step6_box_area(b) > 0]
    results, flags, scores = [], [], []
    for i, b in enumerate(boxes, start=1):
        roi = step6_crop_box(clean_img, b, pad_ratio=0.04)
        info = step6_extract_hatch_features(roi, debug=debug)
        score = STEP6_PER_PROJ if info["is_section"] else 0.0
        info["proj_index"] = i
        info["box"] = tuple(map(int, b))
        info["score"] = float(score)
        results.append(info)
        flags.append(bool(info["is_section"]))
        scores.append(float(score))
    return {"flags": flags, "scores": scores, "total_score": float(sum(scores)), "results": results}

# ============================================================
# /24 - YAQQOL TASVIR TO'G'RILIGI
# ============================================================
def _bin_to_lines(thr_img: np.ndarray):
    return (((255 - thr_img) > 0).astype(np.uint8) * 255)

def _remove_small_components(mask: np.ndarray, min_pixels: int = 40):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_pixels:
            out[labels == i] = 255
    return out

def _component_angle_and_aspect(cnt):
    rect = cv2.minAreaRect(cnt)
    (_, _), (w, h), ang = rect
    if w < 1e-6 or h < 1e-6:
        return 0.0, 1.0
    if w < h:
        ang = ang + 90.0
    ang = ang % 180.0
    aspect = max(w, h) / max(min(w, h), 1e-6)
    return ang, aspect

def crop_box(img, box, pad=0):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1 - pad)); y1 = max(0, int(y1 - pad))
    x2 = min(w, int(x2 + pad)); y2 = min(h, int(y2 + pad))
    return img[y1:y2, x1:x2]

def build_roi_mask(shape, box, pad=18):
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1 - pad)); y1 = max(0, int(y1 - pad))
    x2 = min(w, int(x2 + pad)); y2 = min(h, int(y2 + pad))
    m = np.zeros((h, w), dtype=np.uint8)
    m[y1:y2, x1:x2] = 255
    return m

def skeletonize(mask: np.ndarray):
    img = (mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        done = (cv2.countNonZero(img) == 0)
    return skel

def auto_projection_boxes_from_etalon(ref_thr: np.ndarray, max_boxes=3):
    lines = _bin_to_lines(ref_thr)
    h, w = lines.shape[:2]
    work = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations=1)
    work = cv2.dilate(work, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
    work = _remove_small_components(work, min_pixels=max(350, int(0.0007 * h * w)))

    num, labels, stats, _ = cv2.connectedComponentsWithStats((work > 0).astype(np.uint8), connectivity=8)
    cands = []
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        box_area = bw * bh
        if area < max(350, int(0.0007 * h * w)): continue
        if bw < 0.08 * w and bh < 0.08 * h: continue
        score = float(area + 0.20 * box_area)
        cands.append((score, (x, y, x + bw, y + bh)))
    cands.sort(key=lambda z: z[0], reverse=True)

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / max(ua, 1)

    boxes = []
    for _, b in cands:
        if all(iou(b, k) < 0.35 for k in boxes):
            boxes.append(b)
        if len(boxes) >= max_boxes:
            break

    if not boxes:
        boxes = [(0, 0, w, h)]
    return boxes

def hough_angle_features(mask: np.ndarray):
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    segs = cv2.HoughLinesP(m, rho=1, theta=np.pi / 180, threshold=18,
                           minLineLength=max(14, int(0.06 * min(h, w))), maxLineGap=5)

    total_len = 0.0
    diag_len = 0.0
    long_diag_len = 0.0
    hv_len = 0.0
    seg_count = 0

    if segs is not None:
        for s in segs[:, 0]:
            x1, y1, x2, y2 = s
            dx = x2 - x1; dy = y2 - y1
            L = float(np.hypot(dx, dy))
            if L < 1:
                continue
            ang = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0
            total_len += L
            seg_count += 1

            is_hv = (ang <= 12) or (ang >= 168) or (78 <= ang <= 102)
            is_diag = (18 <= ang <= 72) or (108 <= ang <= 162)

            if is_hv:
                hv_len += L
            if is_diag:
                diag_len += L
                if L >= max(18, 0.10 * min(h, w)):
                    long_diag_len += L

    return {
        "total_len": float(total_len),
        "diag_ratio": float(diag_len / max(total_len, 1e-6)),
        "long_diag_ratio": float(long_diag_len / max(total_len, 1e-6)),
        "hv_ratio": float(hv_len / max(total_len, 1e-6)),
        "seg_count": int(seg_count),
    }

def choose_visible_box_from_etalon(ref_thr: np.ndarray, boxes):
    H, W = ref_thr.shape[:2]
    scored = []
    for b in boxes:
        crop = crop_box(ref_thr, b, pad=8)
        lines = _bin_to_lines(crop)
        lines = _remove_small_components(lines, min_pixels=20)
        feats = hough_angle_features(lines)
        x1, y1, x2, y2 = b
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area_ratio = (bw * bh) / max(H * W, 1)
        density = float((lines > 0).sum()) / max(bw * bh, 1)
        score = 1.8 * feats["long_diag_ratio"] + 1.0 * feats["diag_ratio"] + 0.45 * np.sqrt(area_ratio) + 0.20 * density
        scored.append({"box": b, "score": float(score), "area_ratio": float(area_ratio), "density": float(density), **feats})
    scored = sorted(scored, key=lambda z: z["score"], reverse=True)
    return scored[0]["box"], scored

def detect_text_like_mask(lines_mask: np.ndarray):
    num, labels, stats, _ = cv2.connectedComponentsWithStats((lines_mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(lines_mask)
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if 12 <= area <= 260 and bw <= 35 and bh <= 35:
            fill = area / max(bw * bh, 1)
            if 0.08 <= fill <= 0.75:
                out[labels == i] = 255
    return out

def detect_dashed_like_mask(lines_mask: np.ndarray):
    thin = cv2.erode(lines_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((thin > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(lines_mask)
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if not (6 <= area <= 160):
            continue
        max_dim = max(bw, bh)
        min_dim = max(min(bw, bh), 1)
        aspect = max_dim / min_dim
        if max_dim <= 30 and aspect >= 1.9:
            out[labels == i] = 255
    return out

def detect_short_diag_segments(lines_mask: np.ndarray):
    num, labels, stats, _ = cv2.connectedComponentsWithStats((lines_mask > 0).astype(np.uint8), connectivity=8)
    short_diag = np.zeros_like(lines_mask)

    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if not (8 <= area <= 150):
            continue
        comp = np.uint8(labels[y:y + bh, x:x + bw] == i) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        ang, aspect = _component_angle_and_aspect(cnts[0])
        is_diag = (20 <= ang <= 70) or (110 <= ang <= 160)
        max_dim = max(bw, bh)
        if is_diag and aspect >= 2.0 and 6 <= max_dim <= 34:
            short_diag[labels == i] = 255
    return short_diag

def detect_hatch_cluster_mask(lines_mask: np.ndarray):
    short_diag = detect_short_diag_segments(lines_mask)
    if int((short_diag > 0).sum()) == 0:
        return np.zeros_like(lines_mask), short_diag

    region = cv2.dilate(short_diag, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations=1)
    region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((region > 0).astype(np.uint8), connectivity=8)
    hatch_region = np.zeros_like(lines_mask)
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < 120:
            continue
        reg = np.uint8(labels == i) * 255
        inside_short = int(((short_diag > 0) & (reg > 0)).sum())
        density = inside_short / max(area, 1)
        if density >= 0.08:
            hatch_region[reg > 0] = 255
    hatch_lines = cv2.bitwise_and(short_diag, hatch_region)
    return hatch_lines, short_diag

def extract_structure_mask(binary_thr: np.ndarray, roi_mask=None):
    lines = _bin_to_lines(binary_thr)
    if roi_mask is not None:
        lines = cv2.bitwise_and(lines, roi_mask)

    text_mask = detect_text_like_mask(lines)
    dashed_mask = detect_dashed_like_mask(lines)
    hatch_mask, short_diag_mask = detect_hatch_cluster_mask(lines)

    remove_mask = cv2.bitwise_or(text_mask, dashed_mask)
    remove_mask = cv2.bitwise_or(remove_mask, hatch_mask)

    structure = lines.copy()
    structure[remove_mask > 0] = 0
    structure = cv2.morphologyEx(structure, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    structure = cv2.morphologyEx(structure, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    structure = _remove_small_components(structure, min_pixels=20)

    skel = skeletonize(structure)
    skel = _remove_small_components(skel, min_pixels=10)

    debug = {
        "text_px": int((text_mask > 0).sum()),
        "dashed_px": int((dashed_mask > 0).sum()),
        "short_diag_px": int((short_diag_mask > 0).sum()),
        "hatch_px": int((hatch_mask > 0).sum()),
        "structure_px": int((structure > 0).sum()),
        "skeleton_px": int((skel > 0).sum()),
    }
    return structure, skel, debug

def score_student_visible_stable_24(ref_skel: np.ndarray, student_skel: np.ndarray, tol_kernel=(13, 13)):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, tol_kernel)
    ref_band = cv2.dilate(ref_skel, k, iterations=1)
    student_band = cv2.dilate(student_skel, k, iterations=1)

    ref_bin = (ref_skel > 0)
    student_bin = (student_skel > 0)
    ref_area = int(ref_bin.sum())
    student_area = int(student_bin.sum())

    found = int((ref_bin & (student_band > 0)).sum())
    correct = int((student_bin & (ref_band > 0)).sum())

    completeness = found / max(ref_area, 1)
    correctness = correct / max(student_area, 1) if student_area > 0 else 0.0
    missing_ratio = 1.0 - completeness
    extra_ratio = 1.0 - correctness if student_area > 0 else 1.0

    inter = int(((ref_band > 0) & (student_band > 0)).sum())
    union = int(((ref_band > 0) | (student_band > 0)).sum())
    iou = inter / max(union, 1)

    area_ratio = student_area / max(ref_area, 1)
    area_balance = min(area_ratio, 1.0 / max(area_ratio, 1e-6))
    area_balance = float(np.clip(area_balance, 0.0, 1.0))

    ref_feats = hough_angle_features(ref_skel)
    student_feats = hough_angle_features(student_skel)

    zero_reason = None
    if student_area < max(80, int(0.15 * ref_area)):
        zero_reason = "student_visible_area_too_small"
    elif completeness < 0.18:
        zero_reason = "student_did_not_draw_visible_view"
    elif area_ratio < 0.20:
        zero_reason = "visible_view_area_ratio_too_small"
    elif ref_feats["diag_ratio"] >= 0.16 and student_feats["diag_ratio"] < 0.05:
        zero_reason = "student_has_no_visible_view_diagonal_structure"

    if zero_reason is not None:
        metrics = {
            "completeness": float(completeness), "correctness": float(correctness), "missing_ratio": float(missing_ratio),
            "extra_ratio": float(extra_ratio), "iou": float(iou), "area_ratio": float(area_ratio),
            "area_balance": float(area_balance), "ref_area": int(ref_area), "student_area": int(student_area),
            "ref_diag_ratio": float(ref_feats["diag_ratio"]), "student_diag_ratio": float(student_feats["diag_ratio"]),
            "zero_reason": zero_reason,
        }
        return 0, metrics

    f1 = (2.0 * completeness * correctness) / max(completeness + correctness, 1e-9)
    quality = 0.50 * f1 + 0.22 * completeness + 0.18 * correctness + 0.10 * np.sqrt(max(iou, 0.0))
    quality = float(np.clip(quality, 0.0, 1.0))
    raw = 24.0 * (quality ** 0.82)

    cap = 24
    if completeness < 0.40 or correctness < 0.40:
        cap = 6
    elif completeness < 0.58 or correctness < 0.58:
        cap = 12
    elif completeness < 0.72 or correctness < 0.72:
        cap = 18
    elif completeness < 0.84 or correctness < 0.84:
        cap = 21

    if completeness >= 0.90 and correctness >= 0.58 and iou >= 0.46 and area_balance >= 0.50:
        cap = max(cap, 20)
    if completeness >= 0.92 and correctness >= 0.60 and iou >= 0.50 and area_balance >= 0.55:
        cap = max(cap, 22)
    if completeness >= 0.95 and correctness >= 0.68 and iou >= 0.58 and area_balance >= 0.62:
        cap = 24

    score = int(np.clip(np.round(min(raw, cap)), 0, 24))
    metrics = {
        "completeness": float(completeness), "correctness": float(correctness), "missing_ratio": float(missing_ratio),
        "extra_ratio": float(extra_ratio), "f1": float(f1), "iou": float(iou), "quality": float(quality),
        "area_ratio": float(area_ratio), "area_balance": float(area_balance), "ref_area": int(ref_area),
        "student_area": int(student_area), "ref_diag_ratio": float(ref_feats["diag_ratio"]),
        "student_diag_ratio": float(student_feats["diag_ratio"]), "zero_reason": None,
    }
    return score, metrics

# ============================================================
# /15 - YAQQOL TASVIRDA QIRQIM
# ============================================================
def _axial_angle_diff(a_deg: float, b_deg: float) -> float:
    d = abs(a_deg - b_deg) % 180.0
    return min(d, 180.0 - d)

def _nearest_allowed_hatch_angle(ang_deg: float):
    allowed = [45.0, 60.0, 120.0, 135.0]
    best = min(allowed, key=lambda t: _axial_angle_diff(ang_deg, t))
    return best, _axial_angle_diff(ang_deg, best)

def is_box_valid_for_current_pair(ref_thr: np.ndarray, box, min_line_px=250):
    if box is None or len(box) != 4:
        return False
    h, w = ref_thr.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x2 <= x1 or y2 <= y1:
        return False
    roi = crop_box(ref_thr, box, pad=8)
    lines = _bin_to_lines(roi)
    line_px = int((lines > 0).sum())
    return line_px >= min_line_px

def detect_short_diag_segments_visible_section(lines_mask: np.ndarray):
    lines = (lines_mask > 0).astype(np.uint8) * 255
    h, w = lines.shape[:2]
    cc_mask = np.zeros_like(lines)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)

    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if not (6 <= area <= 260):
            continue
        comp = np.uint8(labels[y:y + bh, x:x + bw] == i) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        ang, aspect = _component_angle_and_aspect(cnts[0])
        max_dim = max(bw, bh)
        allowed = [30, 45, 60, 120, 135, 150]
        is_diag_family = min(_axial_angle_diff(ang, t) for t in allowed) <= 12
        if is_diag_family and aspect >= 1.5 and 6 <= max_dim <= 90:
            cc_mask[labels == i] = 255

    hough_mask = np.zeros_like(lines)
    segs = cv2.HoughLinesP(lines, rho=1, theta=np.pi / 180, threshold=10,
                           minLineLength=max(8, int(0.035 * min(h, w))), maxLineGap=3)
    if segs is not None:
        allowed = [30, 45, 60, 120, 135, 150]
        for s in segs[:, 0]:
            x1, y1, x2, y2 = map(int, s)
            dx = x2 - x1; dy = y2 - y1
            L = float(np.hypot(dx, dy))
            if L < 6:
                continue
            ang = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0
            if min(_axial_angle_diff(ang, t) for t in allowed) <= 10:
                if L <= max(22, int(0.42 * max(h, w))):
                    cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 1)

    out = cv2.bitwise_or(cc_mask, hough_mask)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    out = _remove_small_components(out, min_pixels=6)
    return out

def _collect_parallel_hatch_lines(lines_mask: np.ndarray):
    lines = (lines_mask > 0).astype(np.uint8) * 255
    h, w = lines.shape[:2]
    segs = cv2.HoughLinesP(lines, rho=1, theta=np.pi / 180, threshold=10,
                           minLineLength=max(8, int(0.03 * min(h, w))), maxLineGap=3)
    hatch_lines = np.zeros_like(lines)

    if segs is None:
        debug = {"candidate_segments": 0, "accepted_segments": 0, "accepted_families": 0, "offset_bins": 0}
        return hatch_lines, debug

    rows = []
    for s in segs[:, 0]:
        x1, y1, x2, y2 = map(int, s)
        dx = x2 - x1; dy = y2 - y1
        L = float(np.hypot(dx, dy))
        if L < 6:
            continue
        ang = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0
        target, diff = _nearest_allowed_hatch_angle(ang)
        if diff > 8.5:
            continue
        if L > max(26, int(0.33 * max(h, w))):
            continue

        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)
        nr = np.deg2rad((target + 90.0) % 180.0)
        rho = mx * np.cos(nr) + my * np.sin(nr)

        rows.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "L": L, "target": target, "rho": rho})

    if not rows:
        debug = {"candidate_segments": 0, "accepted_segments": 0, "accepted_families": 0, "offset_bins": 0}
        return hatch_lines, debug

    bin_size = max(6.0, 0.018 * min(h, w))
    accepted_segments = 0
    accepted_families = 0
    total_bins = 0

    for fam in [45.0, 60.0, 120.0, 135.0]:
        fam_rows = [r for r in rows if abs(r["target"] - fam) < 1e-6]
        if len(fam_rows) < 4:
            continue

        bins = {}
        for r in fam_rows:
            k = int(np.round(r["rho"] / bin_size))
            bins.setdefault(k, []).append(r)

        rich_bins = []
        for k, items in bins.items():
            total_len = sum(z["L"] for z in items)
            if len(items) >= 1 and total_len >= 8:
                rich_bins.append((k, items))
        if len(rich_bins) < 4:
            continue

        lengths = [z["L"] for z in fam_rows]
        median_len = float(np.median(lengths))
        p85_len = float(np.percentile(lengths, 85))
        if median_len > 0.16 * max(h, w):
            continue
        if p85_len > 0.28 * max(h, w):
            continue

        accepted_families += 1
        total_bins += len(rich_bins)
        for _, items in rich_bins:
            for r in items:
                cv2.line(hatch_lines, (r["x1"], r["y1"]), (r["x2"], r["y2"]), 255, 1)
                accepted_segments += 1

    hatch_lines = cv2.morphologyEx(hatch_lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    hatch_lines = _remove_small_components(hatch_lines, min_pixels=6)

    debug = {
        "candidate_segments": int(len(rows)),
        "accepted_segments": int(accepted_segments),
        "accepted_families": int(accepted_families),
        "offset_bins": int(total_bins),
    }
    return hatch_lines, debug

def detect_hatch_region(binary_thr: np.ndarray, roi_mask=None):
    lines = _bin_to_lines(binary_thr)
    if roi_mask is not None:
        lines = cv2.bitwise_and(lines, roi_mask)

    hatch_lines, line_debug = _collect_parallel_hatch_lines(lines)
    if int((hatch_lines > 0).sum()) == 0:
        empty = np.zeros_like(lines)
        debug = {"short_diag_px": 0, "hatch_region_px": 0, "hatch_line_px": 0, "cluster_count": 0, **line_debug}
        return empty, empty, debug

    hatch_region = cv2.dilate(hatch_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)
    hatch_region = cv2.morphologyEx(hatch_region, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)), iterations=1)
    hatch_region = cv2.dilate(hatch_region, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((hatch_region > 0).astype(np.uint8), connectivity=8)
    filtered_region = np.zeros_like(hatch_region)
    cluster_count = 0

    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < 120:
            continue
        reg = np.uint8(labels == i) * 255
        inside_lines = cv2.bitwise_and(hatch_lines, reg)
        line_px = int((inside_lines > 0).sum())
        density = line_px / max(area, 1)
        if density >= 0.018:
            filtered_region[reg > 0] = 255
            cluster_count += 1

    filtered_region = cv2.morphologyEx(filtered_region, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)
    filtered_region = _remove_small_components(filtered_region, min_pixels=90)

    debug = {
        "short_diag_px": int((hatch_lines > 0).sum()),
        "hatch_region_px": int((filtered_region > 0).sum()),
        "hatch_line_px": int((hatch_lines > 0).sum()),
        "cluster_count": int(cluster_count),
        **line_debug
    }
    return filtered_region, hatch_lines, debug

def score_visible_section_15(ref_hatch_region: np.ndarray, student_hatch_region: np.ndarray):
    ref_bin = (ref_hatch_region > 0)
    st_bin = (student_hatch_region > 0)
    ref_area = int(ref_bin.sum())
    st_area = int(st_bin.sum())

    if ref_area < 120:
        metrics = {
            "presence_recall": 0.0, "presence_precision": 0.0 if st_area == 0 else float(min(1.0, st_area / 120.0)),
            "iou": 0.0, "area_ratio": float(st_area / max(ref_area, 1)), "area_balance": 0.0,
            "ref_area": int(ref_area), "student_area": int(st_area),
            "zero_reason": "reference_section_region_not_found_or_invalid_roi",
        }
        return 0, metrics

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    ref_band = cv2.dilate(ref_hatch_region, k, iterations=1)
    st_band = cv2.dilate(student_hatch_region, k, iterations=1)

    ref_found = int((ref_bin & (st_band > 0)).sum())
    st_correct = int((st_bin & (ref_band > 0)).sum()) if st_area > 0 else 0

    presence_recall = ref_found / max(ref_area, 1)
    presence_precision = st_correct / max(st_area, 1) if st_area > 0 else 0.0
    inter = int(((ref_band > 0) & (st_band > 0)).sum())
    union = int(((ref_band > 0) | (st_band > 0)).sum())
    iou = inter / max(union, 1)

    area_ratio = st_area / max(ref_area, 1)
    area_balance = min(area_ratio, 1.0 / max(area_ratio, 1e-6))
    area_balance = float(np.clip(area_balance, 0.0, 1.0))

    zero_reason = None
    if st_area < max(100, int(0.12 * ref_area)):
        zero_reason = "student_has_no_section_region"
    elif presence_recall < 0.20:
        zero_reason = "student_section_not_in_correct_place"

    if zero_reason is not None:
        metrics = {
            "presence_recall": float(presence_recall), "presence_precision": float(presence_precision),
            "iou": float(iou), "area_ratio": float(area_ratio), "area_balance": float(area_balance),
            "ref_area": int(ref_area), "student_area": int(st_area), "zero_reason": zero_reason,
        }
        return 0, metrics

    f1 = (2.0 * presence_recall * presence_precision) / max(presence_recall + presence_precision, 1e-9)
    quality = 0.46 * f1 + 0.26 * presence_recall + 0.16 * presence_precision + 0.12 * np.sqrt(max(iou, 0.0))
    quality = float(np.clip(quality, 0.0, 1.0))
    raw = 15.0 * (quality ** 0.82)

    cap = 15
    if presence_recall < 0.40 or presence_precision < 0.35:
        cap = 5
    elif presence_recall < 0.60 or presence_precision < 0.45:
        cap = 9
    elif presence_recall < 0.78 or presence_precision < 0.55:
        cap = 12
    elif presence_recall < 0.88 or presence_precision < 0.65:
        cap = 14

    if presence_recall >= 0.90 and presence_precision >= 0.58 and iou >= 0.48 and area_balance >= 0.45:
        cap = max(cap, 13)
    if presence_recall >= 0.93 and presence_precision >= 0.64 and iou >= 0.54 and area_balance >= 0.50:
        cap = max(cap, 15)

    score = int(np.clip(np.round(min(raw, cap)), 0, 15))
    metrics = {
        "presence_recall": float(presence_recall), "presence_precision": float(presence_precision), "f1": float(f1),
        "iou": float(iou), "quality": float(quality), "area_ratio": float(area_ratio), "area_balance": float(area_balance),
        "ref_area": int(ref_area), "student_area": int(st_area), "zero_reason": None,
    }
    return score, metrics

# ============================================================
# /4 - CHIZMA TOZALIGI
# ============================================================
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

# ============================================================
# FINAL TABLE HELPERS
# ============================================================
def build_score_table(score_rows):
    df = pd.DataFrame(score_rows, columns=["Kriteriy", "Ball", "Maksimal"])
    df["Foiz"] = (df["Ball"] / df["Maksimal"] * 100).round(2)
    return df

def overall_grade_label(total_score):
    if total_score >= 86:
        return "A'lo"
    if total_score >= 71:
        return "Yaxshi"
    if total_score >= 56:
        return "Qoniqarli"
    return "Qoniqarsiz"

def to_native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(v) for v in value]
    return value


def _safe_cell(value, max_len=180):
    value = to_native(value)
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (dict, list, tuple)):
        s = str(value)
        return s if len(s) <= max_len else s[: max_len - 3] + "..."
    return value


def dict_to_rows(payload, prefix=""):
    rows = []
    payload = to_native(payload)
    if not isinstance(payload, dict):
        return [(prefix or "qiymat", _safe_cell(payload))]

    for k, v in payload.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            rows.extend(dict_to_rows(v, key))
        else:
            rows.append((key, _safe_cell(v)))
    return rows


def print_block(title, payload):
    print("\n" + "=" * 78)
    print(title)
    if isinstance(payload, dict):
        rows = dict_to_rows(payload)
        df = pd.DataFrame(rows, columns=["Maydon", "Qiymat"])
        display(df)
    elif isinstance(payload, pd.DataFrame):
        display(payload)
    else:
        print(_safe_cell(payload))
    print("=" * 78)


# ============================================================


# ============================================================
# BACKEND ENTRYPOINT
# ============================================================
def _backend_json_safe(value):
    """Convert numpy / tuple / pandas values into JSON-safe Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _backend_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_backend_json_safe(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value


def _save_rgb_image(arr: np.ndarray, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)
    return str(path)


def _file_to_thr(path: str | Path):
    p = Path(path)
    return load_any_to_thr(p.name, p.read_bytes(), zoom=PDF_ZOOM)


def evaluate_etalon(reference_path: str, student_path: str, output_dir: str = "app/uploads/results") -> dict:
    """
    Backend entrypoint for etalon mode.

    Parameters
    ----------
    reference_path: str
        Teacher's reference drawing path.
    student_path: str
        Student's submitted drawing path.
    output_dir: str
        Directory where result artifacts will be saved.

    Returns
    -------
    dict with keys: total_score, details, overlay_path, table_json.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"etalon_{Path(student_path).stem}_{uuid.uuid4().hex[:8]}" if 'uuid' in globals() else f"etalon_{Path(student_path).stem}"

    # base load
    et_rgb, et_thr = _file_to_thr(reference_path)
    st_rgb, st_thr = _file_to_thr(student_path)

    et_c = crop_to_drawing(et_thr, pad=40)
    st_c = crop_to_drawing(st_thr, pad=40)
    et_r, st_r = resize_to_same(et_c, st_c, max_side=MAX_SIDE)

    st_aligned, warp = ecc_align(et_r, st_r, iters=900)
    diff_xor, et_lines, st_lines = xor_diff(et_r, st_aligned)
    diff_focus = remove_small_components(remove_outer_border(diff_xor, 60), 300)

    student_rgb_base = cv2.cvtColor(st_aligned, cv2.COLOR_GRAY2RGB)
    overlay_diff = student_rgb_base.copy()
    overlay_diff[diff_focus > 0] = (255, 0, 0)

    metrics = compute_match_metrics(et_r, st_aligned, diff_focus)

    # /3 Global match
    frame3, frame3_dbg = score_frame_titleblock_3(metrics)

    # /6 Placement
    placement6, placement_dbg = score_placement_6(et_r, st_aligned)

    # /8 Line types
    line8, line8_dbg, et_dash_mask, st_dash_mask = score_line_types_8(et_r, st_aligned)

    # /12 Dimensioning
    dim12, dim12_dbg, et_dim_mask, st_dim_mask = score_dimension_12(et_r, st_aligned, metrics)

    # /18 Projections
    proj_pack = score_projections_18(st_aligned, metrics, expected_proj=3)
    proj18 = int(proj_pack["score"])
    student_projections = int(proj_pack["student_projections"])
    projection_boxes = proj_pack["boxes_vis"].copy()
    top_clean = proj_pack["top_clean"]
    proj_dbg = proj_pack["debug"]

    # /10 Projection sections
    if len(projection_boxes) > 0 and top_clean is not None and top_clean.size > 0:
        step6_out = step6_run_only_selected3(clean_img=top_clean, selected_boxes=projection_boxes, debug=False)
        step6_total_score = float(step6_out["total_score"])
    else:
        step6_out = {"flags": [], "scores": [], "total_score": 0.0, "results": []}
        step6_total_score = 0.0

    # /24 Visible view correctness
    candidate_boxes = auto_projection_boxes_from_etalon(et_r, max_boxes=3)
    visible_box, visible_box_debug = choose_visible_box_from_etalon(et_r, candidate_boxes)
    step7_roi = build_roi_mask(et_r.shape, visible_box, pad=18)

    ref_structure, ref_skel, ref_dbg = extract_structure_mask(et_r, roi_mask=step7_roi)
    student_structure, student_skel, student_dbg = extract_structure_mask(st_aligned, roi_mask=step7_roi)
    step7_score, step7_metrics = score_student_visible_stable_24(ref_skel, student_skel, tol_kernel=(13, 13))

    k_dbg = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    ref_band_dbg = cv2.dilate(ref_skel, k_dbg, iterations=1)
    student_band_dbg = cv2.dilate(student_skel, k_dbg, iterations=1)

    missing_mask_step7 = np.zeros_like(ref_skel)
    missing_mask_step7[(ref_skel > 0) & (student_band_dbg == 0)] = 255
    extra_mask_step7 = np.zeros_like(student_skel)
    extra_mask_step7[(student_skel > 0) & (ref_band_dbg == 0)] = 255

    student_rgb = cv2.cvtColor(st_aligned, cv2.COLOR_GRAY2RGB)
    overlay_step7 = student_rgb.copy()
    overlay_step7[missing_mask_step7 > 0] = (0, 0, 255)
    overlay_step7[extra_mask_step7 > 0] = (255, 0, 0)
    overlay_box_step7 = overlay_step7.copy()
    x1, y1, x2, y2 = visible_box
    cv2.rectangle(overlay_box_step7, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    step7_result = {
        "criterion": "Yaqqol tasvir to'g'riligi",
        "score": int(step7_score),
        "max_score": 24,
        "visible_box": tuple(map(int, visible_box)),
        "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in step7_metrics.items()},
        "ref_debug": ref_dbg,
        "student_debug": student_dbg,
        "candidate_boxes_debug": visible_box_debug,
    }

    # /15 Visible section
    visible_box_step8 = None
    if isinstance(step7_result, dict) and "visible_box" in step7_result:
        candidate_box = tuple(step7_result["visible_box"])
        if is_box_valid_for_current_pair(et_r, candidate_box, min_line_px=250):
            visible_box_step8 = candidate_box

    if visible_box_step8 is None:
        candidate_boxes_step8 = auto_projection_boxes_from_etalon(et_r, max_boxes=3)
        visible_box_step8, _ = choose_visible_box_from_etalon(et_r, candidate_boxes_step8)

    step8_roi = build_roi_mask(et_r.shape, visible_box_step8, pad=18)
    ref_hatch_region, ref_hatch_lines, ref_step8_dbg = detect_hatch_region(et_r, roi_mask=step8_roi)
    student_hatch_region, student_hatch_lines, student_step8_dbg = detect_hatch_region(st_aligned, roi_mask=step8_roi)
    step8_score, step8_metrics = score_visible_section_15(ref_hatch_region, student_hatch_region)

    student_rgb_step8 = cv2.cvtColor(st_aligned, cv2.COLOR_GRAY2RGB)
    overlay_step8 = student_rgb_step8.copy()
    k_dbg = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    ref_band_dbg = cv2.dilate(ref_hatch_region, k_dbg, iterations=1)
    student_band_dbg = cv2.dilate(student_hatch_region, k_dbg, iterations=1)

    missing_mask_step8 = np.zeros_like(ref_hatch_region)
    missing_mask_step8[(ref_hatch_region > 0) & (student_band_dbg == 0)] = 255
    extra_mask_step8 = np.zeros_like(student_hatch_region)
    extra_mask_step8[(student_hatch_region > 0) & (ref_band_dbg == 0)] = 255

    overlay_step8[missing_mask_step8 > 0] = (0, 0, 255)
    overlay_step8[extra_mask_step8 > 0] = (255, 0, 0)
    overlay_box_step8 = overlay_step8.copy()
    x1, y1, x2, y2 = visible_box_step8
    cv2.rectangle(overlay_box_step8, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    step8_result = {
        "criterion": "Yaqqol tasvirda qirqim bajarilganligi",
        "score": int(step8_score),
        "max_score": 15,
        "visible_box": tuple(map(int, visible_box_step8)),
        "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in step8_metrics.items()},
        "ref_debug": ref_step8_dbg,
        "student_debug": student_step8_dbg,
    }

    # /4 Cleanliness
    step9_score, step9_metrics, step9_masks, step9_debug = score_cleanliness_4(et_r, st_aligned)
    student_rgb_step9 = cv2.cvtColor(st_aligned, cv2.COLOR_GRAY2RGB)
    overlay_step9 = student_rgb_step9.copy()
    overlay_step9[step9_masks["dirty_mask"] > 0] = (255, 0, 0)

    step9_result = {
        "criterion": "Chizma tozaligi",
        "score": int(step9_score),
        "max_score": 4,
        "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in step9_metrics.items()},
        "debug": step9_debug,
    }

    score_rows = [
        ("Ramka + burchak shtampi", frame3, 3),
        ("Chizmani to'g'ri joylashtirish", placement6, 6),
        ("Chiziq turlari", line8, 8),
        ("O'lcham qo'yish", dim12, 12),
        ("Proyeksiyalar soni", proj18, 18),
        ("Proyeksiyalarda qirqim bajarilganligi", round(step6_total_score, 2), 10),
        ("Yaqqol tasvir to'g'riligi", step7_score, 24),
        ("Yaqqol tasvirda qirqim bajarilganligi", step8_score, 15),
        ("Chizma tozaligi", step9_score, 4),
    ]
    score_table = build_score_table(score_rows)
    total_score = float(score_table["Ball"].sum())
    grade_label = overall_grade_label(total_score)

    overlay_path = _save_rgb_image(overlay_diff, out_dir / f"{run_id}_diff_overlay.png")
    visible_overlay_path = _save_rgb_image(overlay_box_step7, out_dir / f"{run_id}_visible_overlay.png")
    section_overlay_path = _save_rgb_image(overlay_box_step8, out_dir / f"{run_id}_section_overlay.png")
    clean_overlay_path = _save_rgb_image(overlay_step9, out_dir / f"{run_id}_cleanliness_overlay.png")

    table_json = [
        {
            "criterion": str(row["Kriteriy"]),
            "score": float(row["Ball"]),
            "max_score": float(row["Maksimal"]),
            "percent": float(row["Foiz"]),
        }
        for _, row in score_table.iterrows()
    ]

    details = {
        "mode": "etalon",
        "reference_file": str(reference_path),
        "student_file": str(student_path),
        "grade_label": grade_label,
        "student_projections": student_projections,
        "projection_boxes": projection_boxes,
        "visible_box": tuple(map(int, visible_box_step8)) if visible_box_step8 is not None else None,
        "base_metrics": {k: round(v, 4) for k, v in metrics.items()},
        "modules": {
            "frame_titleblock_3": {"score": int(frame3), "max_score": 3, "debug": frame3_dbg},
            "placement_6": {"score": int(placement6), "max_score": 6, "debug": placement_dbg},
            "line_types_8": {"score": int(line8), "max_score": 8, "debug": line8_dbg},
            "dimension_12": {"score": int(dim12), "max_score": 12, "debug": dim12_dbg},
            "projections_18": {"score": int(proj18), "max_score": 18, "debug": proj_dbg},
            "projection_sections_10": {
                "score": round(float(step6_total_score), 4),
                "max_score": 10,
                "flags": step6_out.get("flags", []),
                "per_proj_scores": [round(float(x), 4) for x in step6_out.get("scores", [])],
            },
            "visible_view_24": step7_result,
            "visible_section_15": step8_result,
            "cleanliness_4": step9_result,
        },
        "artifacts": {
            "diff_overlay": overlay_path,
            "visible_overlay": visible_overlay_path,
            "section_overlay": section_overlay_path,
            "cleanliness_overlay": clean_overlay_path,
        },
    }

    return {
        "total_score": round(total_score, 2),
        "details": _backend_json_safe(details),
        "overlay_path": overlay_path,
        "table_json": _backend_json_safe(table_json),
    }

