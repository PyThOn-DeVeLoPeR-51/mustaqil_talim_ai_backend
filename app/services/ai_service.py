from pathlib import Path
from typing import Any

from fastapi import HTTPException, status


RESULTS_DIR = Path("app/uploads/results")


def normalize_ai_result(result: dict[str, Any]) -> dict[str, Any]:
    """
    Etalon va ixtiyoriy rejimdan qaytgan natijani bitta formatga keltiradi.
    """

    if not isinstance(result, dict):
        raise ValueError("AI result dict formatida bo‘lishi kerak.")

    score_keys = [
        "total_score",
        "score",
        "final_score",
        "overall_score",
        "total_ball",
        "ball",
    ]

    total_score = None

    for key in score_keys:
        if key in result and result[key] is not None:
            total_score = result[key]
            break

    if total_score is None:
        raise ValueError("AI natijasida total_score topilmadi.")

    try:
        total_score = float(total_score)
    except (TypeError, ValueError):
        raise ValueError("total_score son bo‘lishi kerak.")

    details = (
        result.get("details")
        or result.get("ai_json_result")
        or result.get("json_result")
        or {}
    )

    overlay_path = (
        result.get("overlay_path")
        or result.get("overlay")
        or result.get("overlay_file")
    )

    table_json = (
        result.get("table_json")
        or result.get("table")
        or result.get("rows")
        or []
    )

    if isinstance(table_json, dict):
        table_json = [table_json]

    if table_json is None:
        table_json = []

    if not isinstance(table_json, list):
        table_json = []

    return {
        "total_score": total_score,
        "ai_json_result": details,
        "overlay_path": overlay_path,
        "table_json": table_json,
    }


def evaluate_submission_with_ai(
    mode: str,
    student_file_path: str,
    reference_file_path: str | None = None,
) -> dict[str, Any]:
    """
    Submission uchun AI baholashni ishga tushiradi.
    mode == etalon   -> evaluate_etalon()
    mode == optional -> evaluate_optional()
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if mode == "etalon":
            if not reference_file_path:
                raise ValueError("Etalon rejim uchun reference_file_path kerak.")

            from app.ai.etalon_mode_final_backend import evaluate_etalon

            raw_result = evaluate_etalon(
                reference_path=reference_file_path,
                student_path=student_file_path,
                output_dir=str(RESULTS_DIR),
            )

        elif mode == "optional":
            from app.ai.optional_mode_v1_backend import evaluate_optional

            raw_result = evaluate_optional(
                student_path=student_file_path,
                output_dir=str(RESULTS_DIR),
            )

        else:
            raise ValueError("mode faqat 'etalon' yoki 'optional' bo‘lishi mumkin.")

        return normalize_ai_result(raw_result)

    except ImportError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI fayl yoki funksiya import qilinmadi: {error}",
        )

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI baholashda xatolik: {error}",
        )