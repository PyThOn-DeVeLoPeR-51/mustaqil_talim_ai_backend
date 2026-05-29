from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


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

    __table_args__ = (
        UniqueConstraint(
            "task_id",
            "student_id",
            "attempt_number",
            name="uq_submission_task_student_attempt",
        ),
    )