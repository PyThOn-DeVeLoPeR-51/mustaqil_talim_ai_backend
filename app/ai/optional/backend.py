"""Backend result contract and overlay generation."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image

from app.ai.optional.pipeline import analyze_optional_mode_file, save_optional_artifacts

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
    run_id = f"{stem}_{uuid.uuid4().hex[:8]}"
    overlay_path = _save_optional_overlay(outputs, output_dir=output_dir, stem=run_id)
    saved = save_optional_artifacts(outputs, Path(output_dir) / f"optional_{run_id}_artifacts")

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


__all__ = [
    '_backend_json_safe',
    '_build_backend_table',
    '_save_optional_overlay',
    'evaluate_optional',
]
