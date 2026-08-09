from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.ai.drawing_ai_v2 import (
    DrawingEvaluationEngine,
    EvaluationMode,
    EvaluationRequest,
    EvaluationResult,
)
from app.ai.drawing_ai_v2.exceptions import DrawingAIResultError, DrawingAIValidationError
from app.ai.drawing_ai_v2.normalization import normalize_result, validate_score_consistency
from app.ai.drawing_ai_v2.rubrics import (
    ETALON_MAX_SCORE,
    ETALON_RUBRIC,
    OPTIONAL_RAW_MAX_SCORE,
    OPTIONAL_RUBRIC,
)
from app.ai.drawing_ai_v2.validation import validate_drawing_file
from app.services import ai_service


class DrawingAIV2ContractTestCase(unittest.TestCase):
    def test_rubric_weights_are_locked(self) -> None:
        self.assertEqual(ETALON_MAX_SCORE, 100.0)
        self.assertEqual(OPTIONAL_RAW_MAX_SCORE, 75.0)
        self.assertEqual([item.max_score for item in ETALON_RUBRIC], [3, 6, 8, 12, 18, 10, 24, 15, 4])
        self.assertEqual([item.max_score for item in OPTIONAL_RUBRIC], [15, 10, 15, 15, 10, 10])

    def test_normalize_result_keeps_existing_submission_contract(self) -> None:
        result = normalize_result(
            {
                "final_score": 82,
                "details": {"mode": "optional"},
                "overlay": "result.png",
                "table": {"criterion": "x", "score": 1, "max_score": 1},
            }
        )
        self.assertEqual(result.total_score, 82.0)
        self.assertEqual(result.overlay_path, "result.png")
        self.assertEqual(len(result.table_json), 1)
        self.assertEqual(
            result.to_dict().keys(),
            {"total_score", "ai_json_result", "overlay_path", "table_json"},
        )

    def test_score_formula_guard_rejects_inconsistent_total(self) -> None:
        result = EvaluationResult(
            total_score=100.0,
            details={},
            table_json=[
                {
                    "criterion": item.label,
                    "score": 0.0,
                    "max_score": item.max_score,
                }
                for item in OPTIONAL_RUBRIC
            ],
        )
        with self.assertRaises(DrawingAIResultError):
            validate_score_consistency(EvaluationMode.OPTIONAL, result)

    def test_validation_rejects_extension_spoofing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_png = Path(directory) / "drawing.png"
            fake_png.write_bytes(b"not-a-real-png")
            with self.assertRaises(DrawingAIValidationError):
                validate_drawing_file(fake_png)

    def test_validation_accepts_real_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "drawing.png"
            Image.new("RGB", (64, 64), "white").save(image_path)
            self.assertEqual(validate_drawing_file(image_path), image_path)

    def test_service_forwards_optional_task_text(self) -> None:
        class FakeEngine:
            request: EvaluationRequest | None = None

            def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
                self.request = request
                return EvaluationResult(
                    total_score=90.0,
                    details={"mode": "optional"},
                    overlay_path=None,
                    table_json=[],
                )

        fake_engine = FakeEngine()
        with patch.object(ai_service, "_ENGINE", fake_engine):
            result = ai_service.evaluate_submission_with_ai(
                mode="optional",
                student_file_path="student.png",
                task_text="3 ta proyeksiya va o‘lchamlar bo‘lsin",
            )

        self.assertEqual(result["total_score"], 90.0)
        self.assertIsNotNone(fake_engine.request)
        self.assertEqual(
            fake_engine.request.task_text,
            "3 ta proyeksiya va o‘lchamlar bo‘lsin",
        )


class _FakeOptionalAdapter:
    mode = EvaluationMode.OPTIONAL

    def evaluate(self, request: EvaluationRequest) -> dict:
        return {
            "total_score": 80,
            "details": {"mode": "optional"},
            "overlay_path": None,
            "table_json": [
                {
                    "criterion": item.label,
                    "score": score,
                    "max_score": item.max_score,
                }
                for item, score in zip(
                    OPTIONAL_RUBRIC,
                    [15, 10, 15, 15, 5, 0],
                    strict=True,
                )
            ],
        }


class DrawingAIV2EngineTestCase(unittest.TestCase):
    def test_engine_adds_versioned_runtime_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "drawing.png"
            Image.new("RGB", (64, 64), "white").save(image_path)
            engine = DrawingEvaluationEngine(
                adapters={EvaluationMode.OPTIONAL: _FakeOptionalAdapter()}
            )
            result = engine.evaluate(
                EvaluationRequest(
                    mode=EvaluationMode.OPTIONAL,
                    student_path=image_path,
                    output_dir=root / "results",
                    task_text="o‘lchamlar talab qilinadi",
                )
            )

        metadata = result.details["drawing_ai_v2"]
        self.assertTrue(metadata["criteria_locked"])
        self.assertTrue(metadata["task_text_applied"])
        self.assertEqual(metadata["mode"], "optional")
        self.assertEqual(metadata["scoring_version"], "optional-v1-locked")
        self.assertIn("elapsed_ms", metadata)


if __name__ == "__main__":
    unittest.main()
