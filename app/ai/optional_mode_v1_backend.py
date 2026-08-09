"""Backward-compatible facade for the modular optional evaluator.

New code should import ``evaluate_optional`` from ``app.ai.optional.backend``.
Existing helper imports from this module remain supported.
"""

from app.ai.optional.config import *
from app.ai.optional.serialization import *
from app.ai.optional.geometry import *
from app.ai.optional.io import *
from app.ai.optional.normalization import *
from app.ai.optional.layout import *
from app.ai.optional.roles import *
from app.ai.optional.completeness import *
from app.ai.optional.hatching import *
from app.ai.optional.dimensions import *
from app.ai.optional.line_semantics import *
from app.ai.optional.cleanliness import *
from app.ai.optional.task_requirements import *
from app.ai.optional.reporting import *
from app.ai.optional.pipeline import *
from app.ai.optional.backend import *


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run optional mode heuristic evaluator.")
    parser.add_argument("input_path", help="Path to student drawing (pdf/jpg/png/...)")
    parser.add_argument("--task-text", default="", help="Teacher task text")
    parser.add_argument("--out-dir", default="optional_mode_outputs", help="Directory to save outputs")
    args = parser.parse_args()

    outputs = analyze_optional_mode_file(args.input_path, task_text=args.task_text)
    saved = save_optional_artifacts(outputs, args.out_dir)
    print(json.dumps(outputs["final_report"], ensure_ascii=False, indent=2, default=_json_default))
    print("\nSaved:", json.dumps(saved, ensure_ascii=False, indent=2))
