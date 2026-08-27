from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import os
import socket
import time
import uuid
from typing import Callable

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import SessionLocal
from app.models.drawing_job import DrawingEvaluationJob
from app.models.submission import Submission
from app.models.task import Task
from app.services.submission_service import _evaluate_submission_from_storage
from app.storage import delete_storage_prefix


logger = logging.getLogger("app.drawing.jobs")

ACTIVE_JOB_STATUSES = {"pending", "running"}
TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def enqueue_drawing_evaluation_job(
    db: Session,
    submission: Submission,
    *,
    max_attempts: int | None = None,
    commit: bool = True,
) -> DrawingEvaluationJob:
    """Submission uchun bitta durable Drawing AI job yaratadi."""

    existing = (
        db.query(DrawingEvaluationJob)
        .filter(DrawingEvaluationJob.submission_id == submission.id)
        .first()
    )
    if existing is not None:
        return existing

    job = DrawingEvaluationJob(
        submission_id=submission.id,
        status="pending",
        progress_percent=0,
        attempts=0,
        max_attempts=max_attempts or settings.DRAWING_JOB_MAX_ATTEMPTS,
        available_at=utcnow(),
    )
    db.add(job)
    if commit:
        db.commit()
        db.refresh(job)
    else:
        db.flush()

    logger.info(
        "Drawing AI job enqueued",
        extra={
            "drawing_job_id": job.id,
            "submission_id": submission.id,
            "task_id": submission.task_id,
            "student_id": submission.student_id,
        },
    )
    return job


def claim_next_drawing_job(
    db: Session,
    worker_id: str,
) -> DrawingEvaluationJob | None:
    query = (
        db.query(DrawingEvaluationJob)
        .filter(
            DrawingEvaluationJob.status == "pending",
            DrawingEvaluationJob.available_at <= utcnow(),
        )
        .order_by(
            DrawingEvaluationJob.available_at.asc(),
            DrawingEvaluationJob.id.asc(),
        )
    )
    if db.bind is not None and db.bind.dialect.name == "postgresql":
        query = query.with_for_update(skip_locked=True)

    job = query.first()
    if job is None:
        db.rollback()
        return None

    now = utcnow()
    job.status = "running"
    job.attempts += 1
    job.progress_percent = max(int(job.progress_percent or 0), 5)
    job.locked_at = now
    job.locked_by = worker_id
    job.started_at = job.started_at or now
    job.finished_at = None
    job.error_message = None
    db.commit()
    db.refresh(job)
    return job


def _set_job_progress(
    db: Session,
    job: DrawingEvaluationJob,
    progress: int,
) -> None:
    job.progress_percent = max(0, min(int(progress), 100))
    job.locked_at = utcnow()
    db.commit()


def _mark_job_success(
    db: Session,
    job: DrawingEvaluationJob,
) -> DrawingEvaluationJob:
    now = utcnow()
    job.status = "succeeded"
    job.progress_percent = 100
    job.error_message = None
    job.finished_at = now
    job.locked_at = None
    job.locked_by = None
    db.commit()
    db.refresh(job)
    return job


def _mark_job_failure(
    db: Session,
    job: DrawingEvaluationJob,
    submission: Submission | None,
    exc: Exception,
) -> DrawingEvaluationJob:
    now = utcnow()
    message = (str(exc) or exc.__class__.__name__)[:4000]

    # Qisman yozilgan artifactlar keyingi urinishga aralashmasin.
    if submission is not None:
        try:
            delete_storage_prefix(f"results/submissions/{submission.id}")
        except Exception:
            logger.warning(
                "Drawing AI failed artifact cleanup failed",
                extra={"submission_id": submission.id},
                exc_info=True,
            )

    job.error_message = message
    job.progress_percent = 0
    job.locked_at = None
    job.locked_by = None

    if job.attempts < job.max_attempts:
        delay = settings.DRAWING_JOB_RETRY_BASE_SECONDS * (
            2 ** max(job.attempts - 1, 0)
        )
        job.status = "pending"
        job.available_at = now + timedelta(seconds=delay)
        job.finished_at = None
        if submission is not None:
            submission.status = "pending"
            submission.ai_json_result = None
    else:
        job.status = "failed"
        job.finished_at = now
        if submission is not None:
            submission.status = "failed"
            submission.ai_json_result = {"error": message}

    db.commit()
    db.refresh(job)

    logger.exception(
        "Drawing AI job failed",
        extra={
            "drawing_job_id": job.id,
            "submission_id": job.submission_id,
            "attempts": job.attempts,
            "job_status": job.status,
        },
    )
    return job


def execute_claimed_drawing_job(
    db: Session,
    job: DrawingEvaluationJob,
) -> DrawingEvaluationJob:
    started = time.perf_counter()
    submission: Submission | None = None

    try:
        submission = (
            db.query(Submission)
            .filter(Submission.id == job.submission_id)
            .first()
        )
        if submission is None:
            raise RuntimeError("Drawing AI jobga tegishli submission topilmadi.")

        # Migration eski pending rowni seed qilgan paytda old backend requesti
        # baholashni allaqachon tugatgan bo'lishi mumkin. Bunday submissionni
        # ikkinchi marta AI'dan o'tkazmaymiz.
        if submission.status == "evaluated":
            return _mark_job_success(db, job)

        task = db.query(Task).filter(Task.id == submission.task_id).first()
        if task is None:
            raise RuntimeError("Drawing AI jobga tegishli topshiriq topilmadi.")

        _set_job_progress(db, job, 10)

        # Mavjud Drawing AI algoritmi o'zgarmaydi: worker faqat qachon
        # ishga tushishini boshqaradi.
        ai_result = _evaluate_submission_from_storage(
            submission=submission,
            task=task,
        )
        _set_job_progress(db, job, 90)

        submission.total_score = ai_result["total_score"]
        submission.ai_json_result = ai_result["ai_json_result"]
        submission.overlay_path = ai_result["overlay_path"]
        submission.table_json = ai_result["table_json"]
        submission.status = "evaluated"

        # Submission natijasi va job=succeeded bitta transactionda commit bo'ladi.
        # Process shu nuqtadan oldin uzilsa, stale recovery jobni xavfsiz qayta oladi.
        succeeded = _mark_job_success(db, job)
        logger.info(
            "Drawing AI job succeeded",
            extra={
                "drawing_job_id": job.id,
                "submission_id": job.submission_id,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return succeeded

    except Exception as exc:
        db.rollback()
        refreshed_job = (
            db.query(DrawingEvaluationJob)
            .filter(DrawingEvaluationJob.id == job.id)
            .first()
        )
        if refreshed_job is None:
            raise

        refreshed_submission = (
            db.query(Submission)
            .filter(Submission.id == refreshed_job.submission_id)
            .first()
        )
        return _mark_job_failure(
            db,
            refreshed_job,
            refreshed_submission,
            exc,
        )


def recover_stale_drawing_jobs(db: Session) -> int:
    cutoff = utcnow() - timedelta(minutes=settings.DRAWING_JOB_STALE_MINUTES)
    jobs = (
        db.query(DrawingEvaluationJob)
        .filter(
            DrawingEvaluationJob.status == "running",
            DrawingEvaluationJob.locked_at.is_not(None),
            DrawingEvaluationJob.locked_at < cutoff,
        )
        .all()
    )

    for job in jobs:
        submission = (
            db.query(Submission)
            .filter(Submission.id == job.submission_id)
            .first()
        )

        # Crash/OOM paytida DB final commit bo'lmagan bo'lsa, R2'da qisman
        # qolgan result artifactlarini keyingi urinishdan oldin tozalaymiz.
        try:
            delete_storage_prefix(f"results/submissions/{job.submission_id}")
        except Exception:
            logger.warning(
                "Stale Drawing AI artifact cleanup failed",
                extra={"submission_id": job.submission_id},
                exc_info=True,
            )

        if job.attempts >= job.max_attempts:
            job.status = "failed"
            job.finished_at = utcnow()
            job.error_message = (
                "Worker uzilganidan keyin Drawing AI job stale holatda qoldi."
            )
            if submission is not None:
                submission.status = "failed"
                submission.ai_json_result = {"error": job.error_message}
        else:
            job.status = "pending"
            job.available_at = utcnow()
            job.error_message = "Stale Drawing AI job qayta navbatga qo‘yildi."
            if submission is not None:
                submission.status = "pending"

        job.progress_percent = 0
        job.locked_at = None
        job.locked_by = None

    if jobs:
        db.commit()
    return len(jobs)


def run_next_drawing_job_once(
    *,
    session_factory: Callable[[], Session] = SessionLocal,
    worker_id: str | None = None,
) -> int | None:
    resolved_worker_id = worker_id or default_worker_id()
    with session_factory() as db:
        job = claim_next_drawing_job(db, resolved_worker_id)
        if job is None:
            return None
        job_id = job.id
        execute_claimed_drawing_job(db, job)
        return job_id
