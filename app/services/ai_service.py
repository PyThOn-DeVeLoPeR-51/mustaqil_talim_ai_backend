"""Application service for Drawing AI evaluations.

The public function is intentionally kept backward compatible with the original
submission flow.  The implementation delegates to the typed Drawing AI v2
engine, which validates inputs and locks the existing scoring rubrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

from app.ai.drawing_ai_v2 import (
    DrawingEvaluationEngine,
    EvaluationMode,
    EvaluationRequest,
)
from app.ai.drawing_ai_v2.exceptions import (
    DrawingAIConfigurationError,
    DrawingAIError,
    DrawingAIValidationError,
)
from app.ai.drawing_ai_v2.normalization import normalize_result


RESULTS_DIR = Path("app/uploads/results")
_ENGINE = DrawingEvaluationEngine()


def normalize_ai_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy evaluator output to the existing submission contract."""

    return normalize_result(result).to_dict()


def evaluate_submission_with_ai(
    mode: str,
    student_file_path: str,
    reference_file_path: str | None = None,
    task_text: str = "",
) -> dict[str, Any]:
    """Evaluate a student drawing without changing the existing score criteria.

    Parameters
    ----------
    mode:
        ``etalon`` or ``optional``.
    student_file_path:
        Saved student drawing path.
    reference_file_path:
        Required only for etalon mode.
    task_text:
        Teacher task description for optional-mode requirement parsing.  The
        criterion weight and scoring formula remain unchanged.
    """

    try:
        evaluation_mode = EvaluationMode(mode)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="mode faqat 'etalon' yoki 'optional' bo‘lishi mumkin.",
        ) from exc

    request = EvaluationRequest(
        mode=evaluation_mode,
        student_path=Path(student_file_path),
        reference_path=Path(reference_file_path) if reference_file_path else None,
        output_dir=RESULTS_DIR,
        task_text=task_text or "",
    )

    try:
        return _ENGINE.evaluate(request).to_dict()
    except DrawingAIValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Drawing AI input xatosi: {exc}",
        ) from exc
    except DrawingAIConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Drawing AI konfiguratsiya xatosi: {exc}",
        ) from exc
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI modul yoki dependency import qilinmadi: {exc}",
        ) from exc
    except DrawingAIError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drawing AI natija xatosi: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI baholashda kutilmagan xatolik: {exc}",
        ) from exc
