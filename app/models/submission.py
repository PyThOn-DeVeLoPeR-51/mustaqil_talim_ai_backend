from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base
from app.models.drawing_job import DrawingEvaluationJob


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    task_id: Mapped[int] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # "etalon" yoki "optional"
    mode: Mapped[str] = mapped_column(String(20), nullable=False)

    uploaded_file_path: Mapped[str] = mapped_column(String(500), nullable=False)

    # AI keyingi bosqichda shu joylarga yozadi
    total_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ai_json_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    overlay_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    table_json: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)

    # pending | evaluated | failed
    status: Mapped[str] = mapped_column(String(30), default="pending", nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


    evaluation_job: Mapped[DrawingEvaluationJob | None] = relationship(
        "DrawingEvaluationJob",
        back_populates="submission",
        uselist=False,
        lazy="selectin",
    )

    @property
    def evaluation_status(self) -> str:
        job = self.evaluation_job
        if job is not None:
            if job.status == "pending":
                return "queued"
            if job.status == "running":
                return "evaluating"
            if job.status == "succeeded":
                return "evaluated"
            if job.status in {"failed", "cancelled"}:
                return "failed"
        if self.status == "evaluated":
            return "evaluated"
        if self.status == "failed":
            return "failed"
        return "queued"

    @property
    def evaluation_progress_percent(self) -> int:
        job = self.evaluation_job
        if job is not None:
            return max(0, min(int(job.progress_percent or 0), 100))
        return 100 if self.status in {"evaluated", "failed"} else 0

    @property
    def evaluation_attempts(self) -> int:
        job = self.evaluation_job
        return int(job.attempts or 0) if job is not None else 0

    __table_args__ = (
        UniqueConstraint(
            "task_id",
            "student_id",
            "attempt_number",
            name="uq_submission_task_student_attempt",
        ),
    )