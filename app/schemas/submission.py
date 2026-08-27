from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


SubmissionStatus = Literal["pending", "evaluated", "failed"]
EvaluationStatus = Literal["queued", "evaluating", "evaluated", "failed"]
SubmissionMode = Literal["etalon", "optional"]


class SubmissionRead(BaseModel):
    id: int
    task_id: int
    student_id: int
    attempt_number: int
    mode: SubmissionMode
    uploaded_file_path: str

    total_score: float | None = None
    ai_json_result: dict[str, Any] | None = None
    overlay_path: str | None = None
    table_json: list[dict[str, Any]] | None = None

    status: SubmissionStatus
    evaluation_status: EvaluationStatus
    evaluation_progress_percent: int = 0
    evaluation_attempts: int = 0
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
