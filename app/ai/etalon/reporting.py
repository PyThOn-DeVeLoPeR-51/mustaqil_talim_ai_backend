"""Result table, JSON conversion, and artifact helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.ai.etalon.config import PDF_ZOOM
from app.ai.etalon.io import load_any_to_thr

def build_score_table(score_rows):
    """Build the legacy score table without the heavy pandas dependency."""
    table = []
    for criterion, score, maximum in score_rows:
        score_value = float(score)
        maximum_value = float(maximum)
        percent = round((score_value / maximum_value * 100.0), 2) if maximum_value else 0.0
        table.append({
            "Kriteriy": str(criterion),
            "Ball": score_value,
            "Maksimal": maximum_value,
            "Foiz": percent,
        })
    return table


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
    """Small CLI/debug printer kept independent from notebook-only display APIs."""
    print("\n" + "=" * 78)
    print(title)
    if isinstance(payload, dict):
        for field, value in dict_to_rows(payload):
            print(f"{field}: {value}")
    elif isinstance(payload, list):
        for item in payload:
            print(_safe_cell(item))
    else:
        print(_safe_cell(payload))
    print("=" * 78)


def _backend_json_safe(value):
    """Convert numpy and tuple values into JSON-safe Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _backend_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_backend_json_safe(v) for v in value]
    return value


def _save_rgb_image(arr: np.ndarray, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)
    return str(path)


def _file_to_thr(path: str | Path):
    p = Path(path)
    return load_any_to_thr(p.name, p.read_bytes(), zoom=PDF_ZOOM)


__all__ = [
    'build_score_table',
    'overall_grade_label',
    'to_native',
    '_safe_cell',
    'dict_to_rows',
    'print_block',
    '_backend_json_safe',
    '_save_rgb_image',
    '_file_to_thr',
]
