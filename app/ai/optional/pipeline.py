"""Optional evaluator orchestration and artifact saving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

from app.ai.optional.cleanliness import score_cleanliness
from app.ai.optional.completeness import score_orthographic_completeness
from app.ai.optional.config import CONFIG
from app.ai.optional.dimensions import score_dimensions
from app.ai.optional.hatching import score_section_hatching
from app.ai.optional.io import load_bytes, load_path, preprocess_bundle
from app.ai.optional.layout import discover_layout
from app.ai.optional.line_semantics import score_line_semantics
from app.ai.optional.normalization import normalize_sheet
from app.ai.optional.reporting import build_final_report
from app.ai.optional.roles import analyze_projection_roles
from app.ai.optional.serialization import _json_default
from app.ai.optional.task_requirements import score_task_compliance

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


__all__ = [
    'run_optional_mode_core',
    'analyze_optional_mode_bytes',
    'analyze_optional_mode_file',
    'save_optional_artifacts',
]
