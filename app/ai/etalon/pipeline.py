"""Etalon evaluation orchestration preserving the original 100-point rubric."""

from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np

from app.ai.etalon.cleanliness import score_cleanliness_4
from app.ai.etalon.config import MAX_SIDE
from app.ai.etalon.dimensions import score_dimension_12
from app.ai.etalon.frame import score_frame_titleblock_3
from app.ai.etalon.io import (
    compute_match_metrics,
    crop_to_drawing,
    ecc_align,
    remove_outer_border,
    remove_small_components,
    resize_to_same,
    xor_diff,
)
from app.ai.etalon.line_types import score_line_types_8
from app.ai.etalon.placement import score_placement_6
from app.ai.etalon.projection_sections import step6_run_only_selected3
from app.ai.etalon.projections import score_projections_18
from app.ai.etalon.reporting import (
    _backend_json_safe,
    _file_to_thr,
    _save_rgb_image,
    build_score_table,
    overall_grade_label,
)
from app.ai.etalon.visible_section import detect_hatch_region, is_box_valid_for_current_pair, score_visible_section_15
from app.ai.etalon.visible_view import (
    auto_projection_boxes_from_etalon,
    build_roi_mask,
    choose_visible_box_from_etalon,
    extract_structure_mask,
    score_student_visible_stable_24,
)

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
    run_id = f"etalon_{Path(student_path).stem}_{uuid.uuid4().hex[:8]}"

    # base load
    et_thr = _file_to_thr(reference_path)[1]
    st_thr = _file_to_thr(student_path)[1]

    et_c = crop_to_drawing(et_thr, pad=40)
    st_c = crop_to_drawing(st_thr, pad=40)
    et_r, st_r = resize_to_same(et_c, st_c, max_side=MAX_SIDE)

    st_aligned, _warp = ecc_align(et_r, st_r, iters=900)
    diff_xor = xor_diff(et_r, st_aligned)[0]
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
    line8_result = score_line_types_8(et_r, st_aligned)
    line8, line8_dbg = line8_result[:2]
    del line8_result

    # /12 Dimensioning
    dim12_result = score_dimension_12(et_r, st_aligned, metrics)
    dim12, dim12_dbg = dim12_result[:2]
    del dim12_result

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

    _, ref_skel, ref_dbg = extract_structure_mask(et_r, roi_mask=step7_roi)
    _, student_skel, student_dbg = extract_structure_mask(st_aligned, roi_mask=step7_roi)
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
    ref_hatch_region, _, ref_step8_dbg = detect_hatch_region(et_r, roi_mask=step8_roi)
    student_hatch_region, _, student_step8_dbg = detect_hatch_region(st_aligned, roi_mask=step8_roi)
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
    total_score = float(sum(row["Ball"] for row in score_table))
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
        for row in score_table
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


__all__ = [
    'evaluate_etalon',
]
