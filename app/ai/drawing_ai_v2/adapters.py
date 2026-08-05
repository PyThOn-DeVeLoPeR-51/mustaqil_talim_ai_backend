"""Lazy adapters around the current etalon and optional heuristic cores."""

from __future__ import annotations

from typing import Any, Protocol

from app.ai.drawing_ai_v2.contracts import EvaluationMode, EvaluationRequest
from app.ai.drawing_ai_v2.exceptions import DrawingAIConfigurationError


class EvaluatorAdapter(Protocol):
    mode: EvaluationMode

    def evaluate(self, request: EvaluationRequest) -> dict[str, Any]: ...


class EtalonEvaluatorAdapter:
    mode = EvaluationMode.ETALON

    def evaluate(self, request: EvaluationRequest) -> dict[str, Any]:
        if request.reference_path is None:
            raise DrawingAIConfigurationError(
                "Etalon rejim uchun reference_file_path kerak."
            )

        from app.ai.etalon.pipeline import evaluate_etalon

        return evaluate_etalon(
            reference_path=str(request.reference_path),
            student_path=str(request.student_path),
            output_dir=str(request.output_dir),
        )


class OptionalEvaluatorAdapter:
    mode = EvaluationMode.OPTIONAL

    def evaluate(self, request: EvaluationRequest) -> dict[str, Any]:
        from app.ai.optional.backend import evaluate_optional

        return evaluate_optional(
            student_path=str(request.student_path),
            output_dir=str(request.output_dir),
            task_text=request.task_text,
        )
