"""Bounding-box and geometry helpers."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

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


__all__ = [
    'clip_box',
    'box_area',
    'box_iou',
    'merge_two_boxes',
    'boxes_close',
    'merge_boxes_iter',
    'merge_boxes_simple',
    '_smooth_1d',
    '_intersects',
    '_box_intersection_area',
    '_overlap_ratio',
    '_box_center',
]
