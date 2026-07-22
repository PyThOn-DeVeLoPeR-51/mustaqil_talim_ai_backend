"""AI Mentor servis qatlamlari uchun umumiy yordamchi funksiyalar."""

from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def http_error(status_code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail)


def next_student_version(db: Session, model: type, student_id: int) -> int:
    latest_version = (
        db.query(func.max(model.version))
        .filter(model.student_id == student_id)
        .scalar()
    )
    return int(latest_version or 0) + 1
