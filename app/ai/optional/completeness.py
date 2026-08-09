"""Projection completeness and arrangement score (15 points)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.ai.optional.config import CONFIG
from app.ai.optional.roles import x_overlap_ratio, y_overlap_ratio

def _safe_box(info: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int, int, int]]:
    return info["box"] if info is not None and "box" in info and info["box"] is not None else None


def _center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def score_view_count(orthographic_count: int, required_count: int = 3, max_score: int = 6) -> Tuple[int, List[str]]:
    if orthographic_count >= required_count:
        return max_score, []
    if orthographic_count == 2:
        return 4, ["Bitta asosiy proyeksiya yetishmaydi"]
    if orthographic_count == 1:
        return 2, ["Kamida ikkita asosiy proyeksiya yetishmaydi"]
    return 0, ["Asosiy proyeksiyalar topilmadi"]


def score_isometric_presence(has_isometric: bool, required: bool = True, max_score: int = 2) -> Tuple[int, List[str]]:
    if not required:
        return max_score, []
    return (max_score, []) if has_isometric else (0, ["Yaqqol tasvir topilmadi"])


def score_role_presence(front_box, top_box, side_box, max_score: int = 3) -> Tuple[int, List[str]]:
    score = 0
    errors = []
    if front_box is not None:
        score += 1
    else:
        errors.append("Old ko‘rinish aniqlanmadi")
    if top_box is not None:
        score += 1
    else:
        errors.append("Ustki ko‘rinish aniqlanmadi")
    if side_box is not None:
        score += 1
    else:
        errors.append("Profil ko‘rinish aniqlanmadi")
    return score, errors


def score_arrangement(front_box, top_box, side_box, system: str = "first_angle", max_score: int = 4) -> Tuple[int, List[str], Dict[str, bool]]:
    score = 0
    errors: List[str] = []
    checks: Dict[str, bool] = {}
    if front_box is None:
        return 0, ["Joylashuvni tekshirish uchun old ko‘rinish yo‘q"], checks
    fcx, fcy = _center(front_box)
    if top_box is not None:
        _, tcy = _center(top_box)
        xov = x_overlap_ratio(front_box, top_box)
        cond_dir = tcy > fcy if system == "first_angle" else tcy < fcy
        cond_align = xov >= 0.18
        if cond_dir:
            score += 1
        else:
            errors.append("Ustki ko‘rinish old ko‘rinishga nisbatan noto‘g‘ri tomonda")
        if cond_align:
            score += 1
        else:
            errors.append("Ustki ko‘rinish old ko‘rinish bilan vertikal o‘qda yetarli mos emas")
        checks["top_direction_ok"] = bool(cond_dir)
        checks["top_alignment_ok"] = bool(cond_align)
    else:
        errors.append("Ustki ko‘rinish yo‘qligi sabab joylashuv tekshirilmadi")
        checks["top_direction_ok"] = False
        checks["top_alignment_ok"] = False
    if side_box is not None:
        scx, _ = _center(side_box)
        yov = y_overlap_ratio(front_box, side_box)
        cond_side = abs(scx - fcx) > 5
        cond_align = yov >= 0.18
        if cond_side:
            score += 1
        else:
            errors.append("Profil ko‘rinish old ko‘rinishga nisbatan chap/o‘ng tomonda joylashmagan")
        if cond_align:
            score += 1
        else:
            errors.append("Profil ko‘rinish old ko‘rinish bilan gorizontal o‘qda yetarli mos emas")
        checks["side_direction_ok"] = bool(cond_side)
        checks["side_alignment_ok"] = bool(cond_align)
    else:
        errors.append("Profil ko‘rinish yo‘qligi sabab joylashuv tekshirilmadi")
        checks["side_direction_ok"] = False
        checks["side_alignment_ok"] = False
    return min(score, max_score), errors, checks


def score_orthographic_completeness(role_result: Dict[str, Any], cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    roles = role_result["roles"]
    iso_box = _safe_box(roles.get("isometric"))
    front_box = _safe_box(roles.get("front"))
    top_box = _safe_box(roles.get("top"))
    side_box = _safe_box(roles.get("side"))
    extra_orth = roles.get("extra_orthographic", [])
    orthographic_count = len(role_result.get("orthographic", []))
    has_isometric = iso_box is not None
    s1, e1 = score_view_count(orthographic_count, cfg["required_orthographic_views"], cfg["score_views_max"])
    s2, e2 = score_isometric_presence(has_isometric, cfg["require_isometric"], cfg["score_iso_max"])
    s3, e3 = score_role_presence(front_box, top_box, side_box, cfg["score_role_max"])
    s4, e4, checks = score_arrangement(front_box, top_box, side_box, cfg["projection_system"], cfg["score_arrangement_max"])
    total = int(s1 + s2 + s3 + s4)
    warnings = [f"{len(extra_orth)} ta ortiqcha asosiy proyeksiya sohasi mavjud yoki roli aniqlanmagan"] if len(extra_orth) > 0 else []
    summary = {
        "criterion": "Proyeksiyalar to‘liqligi va joylashuvi",
        "score": int(total),
        "max_score": 15,
        "subscores": {"view_count": int(s1), "isometric_presence": int(s2), "role_presence": int(s3), "arrangement": int(s4)},
        "counts": {"orthographic_count": int(orthographic_count), "extra_orthographic_count": int(len(extra_orth)), "has_isometric": bool(has_isometric)},
        "checks": checks,
        "errors": e1 + e2 + e3 + e4,
        "warnings": warnings,
    }
    return {"score": int(total), "max_score": 15, "summary": summary}


__all__ = [
    '_safe_box',
    '_center',
    'score_view_count',
    'score_isometric_presence',
    'score_role_presence',
    'score_arrangement',
    'score_orthographic_completeness',
]
