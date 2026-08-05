"""Final 75-to-100 score conversion, confidence, and feedback."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.ai.optional.geometry import box_area

def _band_label(score100: int) -> str:
    if score100 >= 90:
        return "A'lo"
    if score100 >= 80:
        return "Yaxshi"
    if score100 >= 70:
        return "Qoniqarli+"
    if score100 >= 60:
        return "Qoniqarli"
    return "Qoniqarsiz"


def _collect_messages(*summaries: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors, warnings = [], []
    for s in summaries:
        errors.extend(s.get("errors", []))
        warnings.extend(s.get("warnings", []))
    return list(dict.fromkeys(errors)), list(dict.fromkeys(warnings))


def _make_feedback(report: Dict[str, Any]) -> List[str]:
    fb = []
    mods = report["modules"]

    def ratio(m: Dict[str, Any]) -> float:
        return m["score"] / max(1, m["max_score"])

    if ratio(mods["completeness_arrangement"]) < 0.75:
        fb.append("Proyeksiyalar soni yoki joylashuvi bo‘yicha kamchiliklar bor.")
    if ratio(mods["section_hatching"]) < 0.65:
        fb.append("Qirqim/shtrixlash sifati yoki aniqlanishida muammo bor.")
    if ratio(mods["dimensions"]) < 0.70:
        fb.append("O‘lcham qo‘yish evidence sust yoki to‘liq emas.")
    if ratio(mods["line_semantics"]) < 0.70:
        fb.append("Chiziq turlari evidence yetarli emas.")
    if ratio(mods["cleanliness"]) < 0.70:
        fb.append("Chizma sifati, tozaligi yoki crop holati yaxshilanishi kerak.")
    if ratio(mods["task_compliance"]) < 0.80:
        fb.append("Topshiriq talablariga to‘liq moslik kuzatilmadi.")
    if not fb:
        fb.append("Chizma topshiriq va GOST/ESKD mezonlariga yaxshi mos keldi.")
    return fb


def assess_layout_reliability(layout_result: Dict[str, Any], role_result: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    score = 1.0

    projection_count = int(layout_result["meta"].get("projection_count", 0))
    orthographic_count = int(role_result["meta"].get("orthographic_count", 0))
    has_isometric = bool(role_result["meta"].get("has_isometric", False))
    title_found = bool(layout_result["meta"].get("title_block_found", False))

    if projection_count <= 1:
        warnings.append("Projection count juda kam topildi; layout ishonchliligi past bo‘lishi mumkin.")
        score -= 0.35
    if projection_count >= 6:
        warnings.append("Projection count juda ko‘p topildi; ortiqcha box yoki merge xatosi bo‘lishi mumkin.")
        score -= 0.20
    if orthographic_count == 0:
        warnings.append("Orthographic role ajratilmadi.")
        score -= 0.35
    elif orthographic_count > 3:
        warnings.append("Orthographic role soni odatdagidan ko‘p.")
        score -= 0.15
    if not has_isometric:
        warnings.append("Yaqqol tasvir aniqlanmadi.")
        score -= 0.10
    if not title_found:
        warnings.append("Title block topilmadi.")
        score -= 0.05

    proj_boxes = layout_result.get("projection_boxes", [])
    if proj_boxes:
        areas = [box_area(tuple(b)) for b in proj_boxes]
        total_proj = sum(areas)
        if total_proj > 0:
            largest_ratio = max(areas) / total_proj
            if largest_ratio >= 0.72:
                warnings.append("Bitta projection box juda katta; bir nechta ko‘rinish merge bo‘lgan bo‘lishi mumkin.")
                score -= 0.20

    score = max(0.0, min(1.0, score))
    label = "high" if score >= 0.80 else "medium" if score >= 0.60 else "low"
    return {"confidence_score": round(score, 4), "confidence_label": label, "warnings": warnings}


def build_final_report(
    layout_result: Dict[str, Any],
    role_result: Dict[str, Any],
    score1_result: Dict[str, Any],
    score2_result: Dict[str, Any],
    score3_result: Dict[str, Any],
    score4_result: Dict[str, Any],
    score5_result: Dict[str, Any],
    score6_result: Dict[str, Any],
) -> Dict[str, Any]:
    modules = {
        "completeness_arrangement": {"score": int(score1_result["score"]), "max_score": int(score1_result["max_score"]), "summary": score1_result["summary"]},
        "section_hatching": {"score": int(score2_result["score"]), "max_score": int(score2_result["max_score"]), "summary": score2_result["summary"]},
        "dimensions": {"score": int(score3_result["score"]), "max_score": int(score3_result["max_score"]), "summary": score3_result["summary"]},
        "line_semantics": {"score": int(score4_result["score"]), "max_score": int(score4_result["max_score"]), "summary": score4_result["summary"]},
        "cleanliness": {"score": int(score5_result["score"]), "max_score": int(score5_result["max_score"]), "summary": score5_result["summary"]},
        "task_compliance": {"score": int(score6_result["score"]), "max_score": int(score6_result["max_score"]), "summary": score6_result["summary"]},
    }
    raw_total = sum(v["score"] for v in modules.values())
    raw_max = sum(v["max_score"] for v in modules.values())
    final_score_100 = int(round((raw_total / max(1, raw_max)) * 100))
    errors, warnings = _collect_messages(
        score1_result["summary"], score2_result["summary"], score3_result["summary"],
        score4_result["summary"], score5_result["summary"], score6_result["summary"]
    )
    reliability = assess_layout_reliability(layout_result, role_result)
    warnings.extend(reliability["warnings"])
    warnings = list(dict.fromkeys(warnings))
    report = {
        "mode": "optional",
        "scoring_version": "v1-prototype",
        "raw_total": int(raw_total),
        "raw_max": int(raw_max),
        "final_score_100": int(final_score_100),
        "grade_label": _band_label(final_score_100),
        "confidence_score": reliability["confidence_score"],
        "confidence_label": reliability["confidence_label"],
        "layout_meta": layout_result.get("meta", {}),
        "role_meta": role_result.get("meta", {}),
        "modules": modules,
        "errors": errors,
        "warnings": warnings,
    }
    report["feedback"] = _make_feedback(report)
    return report


__all__ = [
    '_band_label',
    '_collect_messages',
    '_make_feedback',
    'assess_layout_reliability',
    'build_final_report',
]
