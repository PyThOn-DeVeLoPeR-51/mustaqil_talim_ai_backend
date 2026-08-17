"""AI Mentor provider usage audit va per-student Groq quota."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.models.ai_mentor import AIMentorLLMUsage


def _utc_day_bounds(now: datetime | None = None) -> tuple[datetime, datetime]:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    start = current.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def groq_chat_success_count_today(db: Session, student_id: int) -> int:
    start, end = _utc_day_bounds()
    return (db.query(AIMentorLLMUsage).filter(
        AIMentorLLMUsage.student_id == student_id,
        AIMentorLLMUsage.feature == "chat",
        AIMentorLLMUsage.provider == "groq",
        AIMentorLLMUsage.status == "success",
        AIMentorLLMUsage.created_at >= start,
        AIMentorLLMUsage.created_at < end,
    ).count())


def student_has_groq_chat_quota(db: Session, student_id: int, daily_limit: int) -> bool:
    if daily_limit <= 0:
        return True
    return groq_chat_success_count_today(db, student_id) < daily_limit


def record_llm_usage(db: Session, *, student_id: int, feature: str, provider: str, status: str,
                     chat_session_id: int | None = None, model_name: str | None = None,
                     input_tokens: int | None = None, output_tokens: int | None = None,
                     total_tokens: int | None = None, request_id: str | None = None,
                     error_code: str | None = None, fallback_from_provider: str | None = None,
                     metadata_json: dict[str, Any] | None = None) -> AIMentorLLMUsage:
    row = AIMentorLLMUsage(student_id=student_id, chat_session_id=chat_session_id, feature=feature,
        provider=provider, model_name=model_name, status=status, input_tokens=input_tokens,
        output_tokens=output_tokens, total_tokens=total_tokens, request_id=request_id,
        error_code=error_code, fallback_from_provider=fallback_from_provider, metadata_json=metadata_json)
    db.add(row); db.commit(); db.refresh(row); return row
