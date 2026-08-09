from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.ai.etalon_mode_final_backend import evaluate_etalon


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "drawing_ai"
REFERENCE = FIXTURE_DIR / "reference.png"


class EtalonProjectionDetectionAccuracyTestCase(unittest.TestCase):
    def _evaluate(self, sample: str) -> dict:
        with tempfile.TemporaryDirectory(prefix="etalon-projection-test-") as directory:
            return evaluate_etalon(str(REFERENCE), str(FIXTURE_DIR / sample), directory)

    @staticmethod
    def _criterion_score(result: dict, criterion: str) -> float:
        for row in result["table_json"]:
            if row["criterion"] == criterion:
                return float(row["score"])
        raise AssertionError(f"Criterion not found: {criterion}")

    def test_identical_fixture_detects_three_orthographic_projections(self) -> None:
        result = self._evaluate("identical.png")
        self.assertEqual(result["details"]["student_projections"], 3)
        self.assertEqual(self._criterion_score(result, "Proyeksiyalar soni"), 18.0)
        self.assertTrue(
            result["details"]["modules"]["projections_18"]["debug"]["component_detector"]["used"]
        )

    def test_missing_views_fixture_detects_two_remaining_projections(self) -> None:
        result = self._evaluate("missing_views.png")
        self.assertEqual(result["details"]["student_projections"], 2)
        self.assertEqual(self._criterion_score(result, "Proyeksiyalar soni"), 12.0)

    def test_blank_frame_still_detects_zero_projections(self) -> None:
        result = self._evaluate("blank_frame.png")
        self.assertEqual(result["details"]["student_projections"], 0)
        self.assertEqual(self._criterion_score(result, "Proyeksiyalar soni"), 0.0)


if __name__ == "__main__":
    unittest.main()
