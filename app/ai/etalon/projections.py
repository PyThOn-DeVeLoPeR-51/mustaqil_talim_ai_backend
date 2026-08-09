"""Projection detection and score (18 points)."""

from __future__ import annotations

import cv2
import numpy as np

from app.ai.etalon.io import bin_to_lines
from app.ai.etalon.line_types import dashed_like_count

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


def _box_contains(outer, inner, center_required=True, area_ratio=0.70):
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    if center_required:
        cx = (ix1 + ix2) / 2.0
        cy = (iy1 + iy2) / 2.0
        center_inside = ox1 <= cx <= ox2 and oy1 <= cy <= oy2
    else:
        center_inside = True
    inter_x1 = max(ox1, ix1)
    inter_y1 = max(oy1, iy1)
    inter_x2 = min(ox2, ix2)
    inter_y2 = min(oy2, iy2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    return center_inside and inter >= area_ratio * max(1, box_area(inner))


def _looks_like_title_block(box, image_shape):
    H, W = image_shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    bottom_right_table = (
        cx >= 0.58 * W and cy >= 0.72 * H and bw >= 0.18 * W and bh >= 0.08 * H
    )
    touches_bottom_right = (
        x1 >= 0.50 * W and y2 >= 0.90 * H and bw >= 0.16 * W
    )
    very_low_table = (
        y1 >= 0.72 * H and bw >= 0.20 * W and bh >= 0.06 * H
    )
    return bool(bottom_right_table or touches_bottom_right or very_low_table)


def _prepare_component_projection_mask(lines_255, mx_ratio=0.03, my_ratio=0.02, band_ratio=0.06):
    """Build a full-page mask for projection-component detection.

    The legacy projection counter mainly scans a top band. Real submissions often place
    the third orthographic view lower on the sheet, or connect views with long layout
    lines. This mask keeps the original binary evidence but removes page-border lines
    and tiny noise before full-page connected-component analysis.
    """
    H, W = lines_255.shape[:2]
    clean, long_mask = remove_border_long_lines_only(lines_255.copy(), band_ratio=band_ratio)

    mx = int(W * mx_ratio)
    my = int(H * my_ratio)
    clean[:my, :] = 0
    clean[:, :mx] = 0
    clean[:, W - mx:] = 0

    clean = remove_tiny_components(clean, min_area=max(18, int(0.00006 * clean.size)))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return clean, long_mask


def _remove_nested_projection_boxes(candidates):
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda c: box_area(c["box"]), reverse=True)
    keep = []
    for cand in candidates:
        box = cand["box"]
        nested = False
        for kept in keep:
            kept_box = kept["box"]
            if _box_contains(kept_box, box, center_required=True, area_ratio=0.62) and box_area(box) <= 0.55 * box_area(kept_box):
                nested = True
                break
        if not nested:
            keep.append(cand)
    return keep


def component_projection_candidates_full_page(
    lines_255: np.ndarray,
    expected_proj=3,
    mx_ratio=0.03,
    my_ratio=0.02,
    band_ratio=0.06,
):
    """Detect orthographic projection candidates using full-page components.

    This detector is intentionally conservative: it supplements the legacy top-band
    projection counter when it finds more orthographic-looking boxes. It filters out
    title-block tables, dimension-only strokes, nested inner geometry and diagonal
    axonometric views, while preserving the locked /18 scoring formula.
    """
    H, W = lines_255.shape[:2]
    clean, long_mask = _prepare_component_projection_mask(
        lines_255, mx_ratio=mx_ratio, my_ratio=my_ratio, band_ratio=band_ratio
    )

    mask = (clean > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    min_w = max(24, int(W * 0.045))
    min_h = max(24, int(H * 0.045))
    min_area = max(140, int(W * H * 0.00075))
    candidates = []
    debug_components = []

    for i in range(1, num):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        box = [x, y, x + w, y + h]

        reason = "candidate"
        keep = True
        if area < min_area or w < min_w or h < min_h:
            keep = False
            reason = "size"
        elif w > 0.88 * W and h > 0.80 * H:
            keep = False
            reason = "sheet_border"
        elif _looks_like_title_block(box, clean.shape):
            keep = False
            reason = "title_block"
        elif h <= max(16, int(H * 0.035)) or w <= max(12, int(W * 0.014)):
            keep = False
            reason = "dimension_or_axis_line"

        feat = {"hv_struct_ratio": 0.0, "hv_line_ratio": 0.0, "hv_len": 0.0, "diag_len": 0.0, "total_pixels": 0}
        refined = box
        score = -1e9
        if keep:
            roi = np.zeros((h, w), dtype=np.uint8)
            roi[labels[y:y + h, x:x + w] == i] = 255
            feat = orientation_features(roi)
            hvs = float(feat["hv_struct_ratio"])
            hvl = float(feat["hv_line_ratio"])
            hv_len = float(feat["hv_len"])
            diag_len = float(feat["diag_len"])

            diagonal_view = (diag_len > hv_len * 0.80 and hvl < 0.62 and hvs < 0.70)
            orthographic_like = (
                (hvs >= 0.55 and hvl >= 0.62)
                or (hvs >= 0.78 and hvl >= 0.48)
                or (hvl >= 0.82 and hv_len >= max(60.0, diag_len * 1.10))
            )
            if diagonal_view:
                keep = False
                reason = "diagonal_visual_view"
            elif not orthographic_like:
                keep = False
                reason = "not_orthographic_like"
            else:
                refined = refine_box_full_extent(
                    clean,
                    box,
                    xpad_ratio=0.035,
                    ypad_ratio=0.035,
                    min_col_occ=0.005,
                    min_row_occ=0.005,
                )
                rw = refined[2] - refined[0]
                rh = refined[3] - refined[1]
                refined_area = box_area(refined)
                area_norm = min(1.8, area / max(1.0, W * H * 0.006))
                position_penalty = 0.35 if _looks_like_title_block(refined, clean.shape) else 0.0
                score = (
                    2.4 * hvs +
                    2.0 * hvl +
                    0.9 * area_norm +
                    0.25 * min(rw / max(1.0, W * 0.16), 1.5) +
                    0.20 * min(rh / max(1.0, H * 0.16), 1.5) -
                    position_penalty
                )

        item = {
            "box": list(map(int, refined)),
            "raw_box": list(map(int, box)),
            "area": int(area),
            "score": round(float(score), 3),
            "hv_struct_ratio": round(float(feat.get("hv_struct_ratio", 0.0)), 3),
            "hv_line_ratio": round(float(feat.get("hv_line_ratio", 0.0)), 3),
            "hv_len": round(float(feat.get("hv_len", 0.0)), 1),
            "diag_len": round(float(feat.get("diag_len", 0.0)), 1),
            "keep": bool(keep),
            "reason": reason,
        }
        debug_components.append(item)
        if keep:
            candidates.append(item)

    candidates = _remove_nested_projection_boxes(candidates)
    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)

    selected = []
    for cand in candidates:
        box = cand["box"]
        if any(box_iou(box, kept["box"]) >= 0.35 for kept in selected):
            continue
        selected.append(cand)
        if len(selected) >= expected_proj:
            break

    selected_boxes = [c["box"] for c in selected]
    selected_boxes = sorted(dedupe_boxes(selected_boxes, iou_thr=0.35), key=lambda b: (b[1], b[0]))

    candidate_vis = cv2.cvtColor(clean, cv2.COLOR_GRAY2RGB)
    for b in selected_boxes:
        cv2.rectangle(candidate_vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)

    dbg = {
        "detector": "full_page_component_v2",
        "valid_boxes": selected_boxes,
        "component_count": int(num - 1),
        "selected_count": int(len(selected_boxes)),
        "components": debug_components,
    }
    return len(selected_boxes), dbg, clean, long_mask, candidate_vis, selected_boxes


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

    comp_cnt, comp_dbg, comp_clean, comp_long_mask, comp_candidate_vis, comp_boxes = component_projection_candidates_full_page(
        lines_255,
        expected_proj=3,
        mx_ratio=mx_ratio,
        my_ratio=my_ratio,
        band_ratio=band_ratio,
    )

    use_component = False
    if comp_cnt > int(best["cnt"]):
        use_component = True
    elif int(best["cnt"]) == 1 and comp_cnt >= 2:
        use_component = True

    if use_component:
        selected = {
            "score": float(150.0 * comp_cnt),
            "cnt": int(comp_cnt),
            "dbg": comp_dbg,
            "top_raw": lines_255.copy(),
            "top_clean": comp_clean,
            "long_mask": comp_long_mask,
            "candidate_vis": comp_candidate_vis,
            "boxes": comp_boxes,
            "ycut": int(lines_255.shape[0]),
            "xprof_s": np.array([]),
        }
    else:
        selected = best

    dbg = dict(selected["dbg"])
    dbg["candidate_ycuts"] = ycut_candidates
    dbg["all_scores"] = [
        {"ycut": int(r["ycut"]), "cnt": int(r["cnt"]), "score": round(float(r["score"]), 3), "valid_boxes": r["boxes"]}
        for r in all_results
    ]
    dbg["component_detector"] = {
        "used": bool(use_component),
        "cnt": int(comp_cnt),
        "valid_boxes": comp_boxes,
        "debug": comp_dbg,
    }

    return (
        int(selected["cnt"]), dbg, selected["top_raw"], selected["top_clean"], selected["long_mask"],
        selected["candidate_vis"], selected["boxes"], int(selected["ycut"]), selected["xprof_s"]
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


__all__ = [
    'remove_border_long_lines_only',
    'remove_tiny_components',
    'moving_average_1d',
    'binary_close_1d',
    'find_runs',
    'build_top_ycut_candidates',
    'split_run_by_valleys',
    'build_box_from_xrun',
    'split_box_by_col_occupancy',
    'split_box_by_component_groups',
    'refine_box_full_extent',
    'refine_box_on_full_image',
    'split_box_by_row_occupancy',
    'orientation_features',
    'fallback_single_projection_box',
    'is_projection_box',
    'box_area',
    'box_iou',
    'dedupe_boxes',
    'draw_boxes_on_rgb',
    'build_visual_debug_region',
    'component_projection_candidates_full_page',
    'detect_boxes_on_fixed_top',
    'count_projection_groups_final',
    'score_projections_18',
]
