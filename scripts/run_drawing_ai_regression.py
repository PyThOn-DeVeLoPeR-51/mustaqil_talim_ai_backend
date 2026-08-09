"""Run the locked Drawing AI score regression suite.

Usage:
    python scripts/run_drawing_ai_regression.py
    python scripts/run_drawing_ai_regression.py --update-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.etalon_mode_final_backend import evaluate_etalon  # noqa: E402
from app.ai.optional_mode_v1_backend import evaluate_optional  # noqa: E402


FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "drawing_ai"
BASELINE_PATH = FIXTURE_DIR / "baseline_results.json"
SAMPLES = (
    "identical.png",
    "shifted.png",
    "missing_views.png",
    "noisy.png",
    "blank_frame.png",
)


def _stable_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": row["mode"],
        "name": row["name"],
        "score": row["score"],
        "table": row["table"],
        **({"confidence": row.get("confidence")} if row["mode"] == "optional" else {}),
    }


def run_suite(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference = FIXTURE_DIR / "reference.png"

    for sample in SAMPLES:
        started = time.perf_counter()
        result = evaluate_etalon(
            str(reference),
            str(FIXTURE_DIR / sample),
            str(output_dir / "etalon"),
        )
        rows.append(
            {
                "mode": "etalon",
                "name": sample,
                "score": result["total_score"],
                "table": result["table_json"],
                "elapsed": time.perf_counter() - started,
            }
        )

    for sample in SAMPLES:
        started = time.perf_counter()
        result = evaluate_optional(
            str(FIXTURE_DIR / sample),
            str(output_dir / "optional"),
        )
        rows.append(
            {
                "mode": "optional",
                "name": sample,
                "score": result["total_score"],
                "table": result["table_json"],
                "confidence": result["details"].get("confidence_label"),
                "elapsed": time.perf_counter() - started,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="drawing-ai-regression-") as directory:
        current = run_suite(Path(directory))

    if args.update_baseline:
        BASELINE_PATH.write_text(
            json.dumps(current, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Baseline updated: {BASELINE_PATH}")
        return 0

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    differences = []
    for expected, actual in zip(baseline, current, strict=True):
        if _stable_payload(expected) != _stable_payload(actual):
            differences.append(
                {
                    "expected": _stable_payload(expected),
                    "actual": _stable_payload(actual),
                }
            )

    summary = [
        {
            "mode": row["mode"],
            "sample": row["name"],
            "score": row["score"],
            "elapsed_s": round(row["elapsed"], 4),
        }
        for row in current
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if differences:
        print("\nREGRESSION DETECTED")
        print(json.dumps(differences, ensure_ascii=False, indent=2))
        return 1

    print("\nDrawing AI score regression: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
