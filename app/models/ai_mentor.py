from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class AIMentorDiagnosticQuestion(Base):
    """AI Mentor diagnostikasida ishlatiladigan savolning versiyalangan modeli."""

    __tablename__ = "ai_mentor_diagnostic_questions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    question_code: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    help_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    # single_choice | multiple_choice | short_text | long_text | number | boolean
    answer_type: Mapped[str] = mapped_column(String(30), nullable=False)

    # Tanlovli savollar uchun variantlar va zarur qo‘shimcha konfiguratsiya.
    options_json: Mapped[list[Any] | dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    sort_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_required: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "question_code",
            "version",
            name="uq_ai_diag_question_code_version",
        ),
        CheckConstraint("version >= 1", name="ck_ai_diag_question_version"),
        CheckConstraint("sort_order >= 0", name="ck_ai_diag_question_sort"),
        CheckConstraint(
            "answer_type IN ('single_choice', 'multiple_choice', 'short_text', "
            "'long_text', 'number', 'boolean')",
            name="ck_ai_diag_question_answer_type",
        ),
    )


class AIMentorDiagnosticSession(Base):
    """Bitta talabaning diagnostika topshirish jarayoni."""

    __tablename__ = "ai_mentor_diagnostic_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # in_progress | completed | cancelled
    status: Mapped[str] = mapped_column(
        String(30),
        default="in_progress",
        nullable=False,
        index=True,
    )

    analysis_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    analysis_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "student_id",
            "version",
            name="uq_ai_diag_session_student_version",
        ),
        CheckConstraint("version >= 1", name="ck_ai_diag_session_version"),
        CheckConstraint(
            "status IN ('in_progress', 'completed', 'cancelled')",
            name="ck_ai_diag_session_status",
        ),
    )


class AIMentorDiagnosticAnswer(Base):
    """Diagnostika sessiyasidagi talaba javobi."""

    __tablename__ = "ai_mentor_diagnostic_answers"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    session_id: Mapped[int] = mapped_column(
        ForeignKey("ai_mentor_diagnostic_sessions.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    question_id: Mapped[int] = mapped_column(
        ForeignKey("ai_mentor_diagnostic_questions.id", ondelete="RESTRICT"),
        index=True,
        nullable=False,
    )

    # Oddiy matn/son qiymatlari uchun tez o‘qiladigan ko‘rinish.
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Ko‘p tanlovli yoki tuzilmali javoblar uchun moslashuvchan ko‘rinish.
    answer_json: Mapped[Any | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "question_id",
            name="uq_ai_diag_answer_session_question",
        ),
    )


class AIMentorPlan(Base):
    """Talaba uchun yaratilgan shaxsiy 4 haftalik rejaning bosh yozuvi."""

    __tablename__ = "ai_mentor_plans"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    diagnostic_session_id: Mapped[int | None] = mapped_column(
        ForeignKey("ai_mentor_diagnostic_sessions.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )

    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # draft | active | completed | archived
    status: Mapped[str] = mapped_column(
        String(30),
        default="draft",
        nullable=False,
        index=True,
    )

    # mock | llm | manual
    generation_source: Mapped[str] = mapped_column(
        String(30),
        default="mock",
        nullable=False,
    )
    generation_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "student_id",
            "version",
            name="uq_ai_plan_student_version",
        ),
        CheckConstraint("version >= 1", name="ck_ai_plan_version"),
        CheckConstraint(
            "status IN ('draft', 'active', 'completed', 'archived')",
            name="ck_ai_plan_status",
        ),
        CheckConstraint(
            "generation_source IN ('mock', 'llm', 'manual')",
            name="ck_ai_plan_generation_source",
        ),
        CheckConstraint(
            "end_date IS NULL OR start_date IS NULL OR end_date >= start_date",
            name="ck_ai_plan_date_range",
        ),
    )


class AIMentorPlanWeek(Base):
    """4 haftalik rejaning bitta haftasi."""

    __tablename__ = "ai_mentor_plan_weeks"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    plan_id: Mapped[int] = mapped_column(
        ForeignKey("ai_mentor_plans.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    week_number: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    goal: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_outcome: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "plan_id",
            "week_number",
            name="uq_ai_plan_week_plan_number",
        ),
        CheckConstraint(
            "week_number BETWEEN 1 AND 4",
            name="ck_ai_plan_week_number",
        ),
    )


class AIMentorPlanItem(Base):
    """Reja haftasi ichidagi alohida o‘quv faoliyati."""

    __tablename__ = "ai_mentor_plan_items"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    plan_week_id: Mapped[int] = mapped_column(
        ForeignKey("ai_mentor_plan_weeks.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    item_order: Mapped[int] = mapped_column(Integer, nullable=False)
    day_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    activity_type: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )
    estimated_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resources_json: Mapped[list[Any] | dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    # pending | in_progress | completed | skipped
    status: Mapped[str] = mapped_column(
        String(30),
        default="pending",
        nullable=False,
        index=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "plan_week_id",
            "item_order",
            name="uq_ai_plan_item_week_order",
        ),
        CheckConstraint("item_order >= 1", name="ck_ai_plan_item_order"),
        CheckConstraint(
            "day_number IS NULL OR day_number BETWEEN 1 AND 7",
            name="ck_ai_plan_item_day",
        ),
        CheckConstraint(
            "estimated_minutes IS NULL OR estimated_minutes > 0",
            name="ck_ai_plan_item_duration",
        ),
        CheckConstraint(
            "status IN ('pending', 'in_progress', 'completed', 'skipped')",
            name="ck_ai_plan_item_status",
        ),
    )


class AIMentorChatSession(Base):
    """Talabaning AI Mentor bilan alohida suhbat sessiyasi."""

    __tablename__ = "ai_mentor_chat_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    plan_id: Mapped[int | None] = mapped_column(
        ForeignKey("ai_mentor_plans.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )

    title: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # active | closed | archived
    status: Mapped[str] = mapped_column(
        String(30),
        default="active",
        nullable=False,
        index=True,
    )
    context_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    last_message_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'closed', 'archived')",
            name="ck_ai_chat_session_status",
        ),
    )


class AIMentorChatMessage(Base):
    """AI Mentor chat sessiyasidagi bitta xabar."""

    __tablename__ = "ai_mentor_chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    session_id: Mapped[int] = mapped_column(
        ForeignKey("ai_mentor_chat_sessions.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # system | user | assistant
    role: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "sequence_number",
            name="uq_ai_chat_message_session_sequence",
        ),
        CheckConstraint(
            "sequence_number >= 1",
            name="ck_ai_chat_message_sequence",
        ),
        CheckConstraint(
            "role IN ('system', 'user', 'assistant')",
            name="ck_ai_chat_message_role",
        ),
        CheckConstraint(
            "token_count IS NULL OR token_count >= 0",
            name="ck_ai_chat_message_tokens",
        ),
    )
