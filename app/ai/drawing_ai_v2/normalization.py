"""Result normalization and scoring-contract validation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.ai.drawing_ai_v2.contracts import EvaluationMode, EvaluationResult
from app.ai.drawing_ai_v2.exceptions import DrawingAIResultError
from app.ai.drawing_ai_v2.rubrics import ETALON_RUBRIC, OPTIONAL_RUBRIC


_SCORE_KEYS = (
    "total_score",
    "score",
    "final_score",
    "overall_score",
    "total_ball",
    "ball",
)


def _coerce_table(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        raise DrawingAIResultError("table_json list formatida bo‘lishi kerak.")
    return [row for row in value if isinstance(row, dict)]


def normalize_result(raw_result: dict[str, Any]) -> EvaluationResult:
    if not isinstance(raw_result, dict):
        raise DrawingAIResultError("AI natijasi dict formatida bo‘lishi kerak.")

    total_score: Any = None
    for key in _SCORE_KEYS:
        if raw_result.get(key) is not None:
            total_score = raw_result[key]
            break
    if total_score is None:
        raise DrawingAIResultError("AI natijasida total_score topilmadi.")

    try:
        score = float(total_score)
    except (TypeError, ValueError) as exc:
        raise DrawingAIResultError("total_score son bo‘lishi kerak.") from exc

    if not 0.0 <= score <= 100.0:
        raise DrawingAIResultError("total_score 0 va 100 oralig‘ida bo‘lishi kerak.")

    details = (
        raw_result.get("details")
        or raw_result.get("ai_json_result")
        or raw_result.get("json_result")
        or {}
    )
    if not isinstance(details, dict):
        raise DrawingAIResultError("details dict formatida bo‘lishi kerak.")

    overlay_path = (
        raw_result.get("overlay_path")
        or raw_result.get("overlay")
        or raw_result.get("overlay_file")
    )
    table_json = _coerce_table(
        raw_result.get("table_json")
        or raw_result.get("table")
        or raw_result.get("rows")
        or []
    )

    return EvaluationResult(
        total_score=score,
        details=details,
        overlay_path=str(overlay_path) if overlay_path else None,
        table_json=table_json,
    )


def validate_locked_rubric(
    mode: EvaluationMode,
    table_json: Iterable[dict[str, Any]],
) -> None:
    expected = ETALON_RUBRIC if mode is EvaluationMode.ETALON else OPTIONAL_RUBRIC
    rows = list(table_json)
    if len(rows) != len(expected):
        raise DrawingAIResultError(
            f"Rubrika qatorlari soni o‘zgargan: {len(rows)} != {len(expected)}."
        )

    for index, (row, criterion) in enumerate(zip(rows, expected, strict=True), start=1):
        label = str(row.get("criterion", ""))
        try:
            max_score = float(row.get("max_score"))
        except (TypeError, ValueError) as exc:
            raise DrawingAIResultError(
                f"{index}-rubrika qatorida max_score noto‘g‘ri."
            ) from exc

        if label != criterion.label or max_score != criterion.max_score:
            raise DrawingAIResultError(
                "Baholash mezoni o‘zgargan: "
                f"kutilgan=({criterion.label!r}, {criterion.max_score}), "
                f"olingan=({label!r}, {max_score})."
            )

def validate_score_consistency(
    mode: EvaluationMode,
    result: EvaluationResult,
) -> None:
    """Ensure criterion scores and the reported total use the locked formula."""
    raw_total = 0.0
    for index, row in enumerate(result.table_json, start=1):
        try:
            score = float(row.get("score"))
            maximum = float(row.get("max_score"))
        except (TypeError, ValueError) as exc:
            raise DrawingAIResultError(
                f"{index}-rubrika qatorida score yoki max_score noto‘g‘ri."
            ) from exc
        if not 0.0 <= score <= maximum:
            raise DrawingAIResultError(
                f"{index}-rubrika balli diapazondan tashqarida: {score}/{maximum}."
            )
        raw_total += score

    if mode is EvaluationMode.ETALON:
        expected_total = round(raw_total, 2)
        if abs(result.total_score - expected_total) > 0.01:
            raise DrawingAIResultError(
                f"Etalon total_score rubrika yig‘indisiga mos emas: "
                f"{result.total_score} != {expected_total}."
            )
        return

    expected_total = float(int(round((raw_total / 75.0) * 100.0)))
    if abs(result.total_score - expected_total) > 0.01:
        raise DrawingAIResultError(
            f"Optional total_score 75→100 formulasiga mos emas: "
            f"{result.total_score} != {expected_total}."
        )

