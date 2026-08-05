"""Optional evaluator configuration. Criterion weights remain unchanged."""

from __future__ import annotations

from typing import Any, Dict

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

__all__ = ["CONFIG"]
