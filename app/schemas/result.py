from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


SubmissionStatus = Literal["pending", "evaluated", "failed"]
SubmissionMode = Literal["etalon", "optional"]


class ResultRead(BaseModel):
    id: int

    task_id: int
    task_title: str | None = None

    student_id: int
    student_full_name: str | None = None

    attempt_number: int
    mode: SubmissionMode

    uploaded_file_path: str
    uploaded_file_url: str | None = None

    total_score: float | None = None
    ai_json_result: dict[str, Any] | None = None

    overlay_path: str | None = None
    overlay_url: str | None = None

    table_json: list[dict[str, Any]] | None = None

    status: SubmissionStatus
    created_at: datetime