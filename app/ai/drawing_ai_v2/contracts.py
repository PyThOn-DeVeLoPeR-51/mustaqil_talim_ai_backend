"""Typed contracts shared by Drawing AI adapters and services."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class EvaluationMode(StrEnum):
    ETALON = "etalon"
    OPTIONAL = "optional"


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    mode: EvaluationMode
    student_path: Path
    output_dir: Path
    reference_path: Path | None = None
    task_text: str = ""


@dataclass(slots=True)
class EvaluationResult:
    total_score: float
    details: dict[str, Any] = field(default_factory=dict)
    overlay_path: str | None = None
    table_json: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_score": self.total_score,
            "ai_json_result": self.details,
            "overlay_path": self.overlay_path,
            "table_json": self.table_json,
        }
