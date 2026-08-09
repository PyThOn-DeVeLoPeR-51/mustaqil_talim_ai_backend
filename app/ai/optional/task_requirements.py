"""Task requirement parsing and compliance score (10 points)."""

from __future__ import annotations

import re
from typing import Any, Dict

from app.ai.optional.config import CONFIG

def parse_task_requirements(task_text: str, cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    text = (task_text or "").strip().lower()
    req = {
        "required_views": None,
        "requires_isometric": None,
        "requires_section": None,
        "requires_dimensions": None,
        "raw_text": task_text,
    }
    patterns = [
        r'(\d+)\s*(?:ta\s*)?(?:proyeksiya|proeksiya|ko[\'‘’`]?rinish|korinish|вид|вида|views?)',
        r'(?:proyeksiya|proeksiya|ko[\'‘’`]?rinish|korinish|вид|views?)\s*(\d+)',
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            try:
                req["required_views"] = int(m.group(1))
                break
            except Exception:
                pass
    if any(k in text for k in ["yaqqol tasvir", "yaqqol", "aksonometriya", "axonometry", "isometric", "izometrik", "аксонометр", "изометр"]):
        req["requires_isometric"] = True
    if any(k in text for k in ["qirqim", "kesim", "section", "sectional", "razrez", "разрез", "сечение"]):
        req["requires_section"] = True
    if any(k in text for k in ["o'lcham", "o‘lcham", "olcham", "razmer", "размер", "dimension", "dimensions"]):
        req["requires_dimensions"] = True
    if req["required_views"] is None:
        req["required_views"] = int(cfg["task_default_required_views"])
    if req["requires_isometric"] is None:
        req["requires_isometric"] = bool(cfg["task_default_requires_isometric"])
    if req["requires_section"] is None:
        req["requires_section"] = bool(cfg["task_default_requires_section"])
    if req["requires_dimensions"] is None:
        req["requires_dimensions"] = bool(cfg["task_default_requires_dimensions"])
    return req


def score_task_compliance(task_text: str, role_result: Dict[str, Any], score2_result: Dict[str, Any], score3_result: Dict[str, Any], cfg: Dict[str, Any] = CONFIG) -> Dict[str, Any]:
    req = parse_task_requirements(task_text, cfg)
    orth_count = int(len(role_result.get("orthographic", [])))
    has_isometric = bool(role_result.get("meta", {}).get("has_isometric", False))
    hatch_count = int(score2_result["summary"].get("detected_hatch_count", 0))
    has_section_evidence = hatch_count > 0
    dim_roles = int(score3_result["summary"]["counts"].get("roles_with_dimension", 0))
    has_dimension_evidence = dim_roles >= 1
    required_views = int(req["required_views"])
    if orth_count >= required_views:
        s_views, views_ok = 4, True
    elif orth_count == required_views - 1:
        s_views, views_ok = 2, False
    else:
        s_views, views_ok = 0, False
    if req["requires_isometric"]:
        s_iso, iso_ok = (2, True) if has_isometric else (0, False)
    else:
        s_iso, iso_ok = 2, True
    if req["requires_section"]:
        s_section, section_ok = (2, True) if has_section_evidence else (0, False)
    else:
        s_section, section_ok = 2, True
    if req["requires_dimensions"]:
        if has_dimension_evidence:
            s_dim = 2 if dim_roles >= 2 else 1
            dim_ok = True
        else:
            s_dim, dim_ok = 0, False
    else:
        s_dim, dim_ok = 2, True
    total = int(min(10, s_views + s_iso + s_section + s_dim))
    errors, warnings = [], []
    if not views_ok:
        errors.append(f"Talab qilingan proyeksiyalar soni yetarli emas: kerak={required_views}, topildi={orth_count}")
    if req["requires_isometric"] and not iso_ok:
        errors.append("Topshiriqda yaqqol tasvir talab qilingan, lekin topilmadi")
    if req["requires_section"] and not section_ok:
        errors.append("Topshiriqda qirqim/kesim talab qilingan, lekin belgi topilmadi")
    if req["requires_dimensions"] and not has_dimension_evidence:
        errors.append("Topshiriqda o‘lcham qo‘yish talab qilingan, lekin o‘lcham qo‘yish belgisi topilmadi")
    elif req["requires_dimensions"] and dim_roles == 1:
        warnings.append("O‘lcham qo‘yish belgisi bor, lekin faqat bitta ko‘rinishda aniq topildi")
    summary = {
        "criterion": "Topshiriq talabiga moslik",
        "score": int(total),
        "max_score": 10,
        "requirements": {
            "required_views": int(req["required_views"]),
            "requires_isometric": bool(req["requires_isometric"]),
            "requires_section": bool(req["requires_section"]),
            "requires_dimensions": bool(req["requires_dimensions"]),
        },
        "evidence": {
            "orthographic_count": int(orth_count),
            "has_isometric": bool(has_isometric),
            "detected_hatch_count": int(hatch_count),
            "roles_with_dimension": int(dim_roles),
        },
        "checks": {
            "views_ok": bool(views_ok),
            "isometric_ok": bool(iso_ok),
            "section_ok": bool(section_ok),
            "dimensions_ok": bool(dim_ok),
        },
        "subscores": {"views": int(s_views), "isometric": int(s_iso), "section": int(s_section), "dimensions": int(s_dim)},
        "errors": errors,
        "warnings": warnings,
    }
    return {"score": int(total), "max_score": 10, "summary": summary}


__all__ = [
    'parse_task_requirements',
    'score_task_compliance',
]
