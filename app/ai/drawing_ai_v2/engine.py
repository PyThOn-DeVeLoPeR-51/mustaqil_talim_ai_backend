"""Production-facing orchestration layer for Drawing AI v2."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.ai.drawing_ai_v2.adapters import (
    EtalonEvaluatorAdapter,
    EvaluatorAdapter,
    OptionalEvaluatorAdapter,
)
from app.ai.drawing_ai_v2.contracts import (
    EvaluationMode,
    EvaluationRequest,
    EvaluationResult,
)
from app.ai.drawing_ai_v2.exceptions import DrawingAIConfigurationError
from app.ai.drawing_ai_v2.normalization import (
    normalize_result,
    validate_locked_rubric,
    validate_score_consistency,
)
from app.ai.drawing_ai_v2.validation import ValidationLimits, validate_drawing_file


ENGINE_VERSION = "drawing-ai-v2.0.0"


class DrawingEvaluationEngine:
    """Validate, execute, normalize, and version Drawing AI evaluations."""

    def __init__(
        self,
        *,
        limits: ValidationLimits = ValidationLimits(),
        adapters: dict[EvaluationMode, EvaluatorAdapter] | None = None,
    ) -> None:
        self._limits = limits
        self._adapters: dict[EvaluationMode, EvaluatorAdapter] = adapters or {
            EvaluationMode.ETALON: EtalonEvaluatorAdapter(),
            EvaluationMode.OPTIONAL: OptionalEvaluatorAdapter(),
        }

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        student_path = validate_drawing_file(request.student_path, self._limits)
        reference_path: Path | None = None
        if request.mode is EvaluationMode.ETALON:
            if request.reference_path is None:
                raise DrawingAIConfigurationError(
                    "Etalon rejim uchun reference_file_path kerak."
                )
            reference_path = validate_drawing_file(request.reference_path, self._limits)

        adapter = self._adapters.get(request.mode)
        if adapter is None:
            raise DrawingAIConfigurationError(
                f"Qo‘llab-quvvatlanmaydigan evaluation mode: {request.mode}."
            )

        request.output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        raw_result = adapter.evaluate(request)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)

        result = normalize_result(raw_result)
        validate_locked_rubric(request.mode, result.table_json)
        validate_score_consistency(request.mode, result)

        scoring_version = (
            "etalon-v1-locked"
            if request.mode is EvaluationMode.ETALON
            else str(result.details.get("scoring_version") or "optional-v1-locked")
        )
        metadata: dict[str, Any] = {
            "engine_version": ENGINE_VERSION,
            "mode": request.mode.value,
            "criteria_locked": True,
            "scoring_version": scoring_version,
            "elapsed_ms": elapsed_ms,
            "student_file": student_path.name,
            "reference_file": reference_path.name if reference_path else None,
            "task_text_applied": bool(request.task_text.strip())
            if request.mode is EvaluationMode.OPTIONAL
            else False,
        }
        details = dict(result.details)
        details["drawing_ai_v2"] = metadata
        result.details = details
        return result
