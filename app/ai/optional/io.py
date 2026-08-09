"""Drawing loading and preprocessing."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import fitz
import numpy as np
from PIL import Image

def resize_keep_ratio(img: np.ndarray, max_side: int = 2200) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale == 1.0:
        return img
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def pdf_bytes_to_rgb(pdf_bytes: bytes, dpi: int = 300, page_index: int = 0) -> np.ndarray:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        if document.page_count < 1:
            raise ValueError("PDF sahifalari topilmadi.")
        if not 0 <= page_index < document.page_count:
            raise ValueError("PDF page_index diapazondan tashqarida.")
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        ).copy()
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def image_bytes_to_rgb(image_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as image:
        return np.array(image.convert("RGB"))


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


__all__ = [
    'resize_keep_ratio',
    'pdf_bytes_to_rgb',
    'image_bytes_to_rgb',
    'load_bytes',
    'load_path',
    'rgb_to_gray',
    'normalize_gray',
    'adaptive_bin',
    'edge_map',
    'preprocess_bundle',
]
