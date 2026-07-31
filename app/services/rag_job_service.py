from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import os
import socket
import time
import uuid
from typing import Callable

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import SessionLocal
from app.models.rag import RAGDocument, RAGProcessingJob
from app.models.teacher import Teacher
from app.rag.embeddings import EmbeddingProvider
from app.services.rag_embedding_service import embed_teacher_document
from app.services.rag_service import (
    get_teacher_document_or_404,
    process_rag_document,
)


logger = logging.getLogger("app.rag.jobs")
ACTIVE_JOB_STATUSES = {"pending", "running"}
TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}
ALLOWED_JOB_TYPES = {"ingest", "reprocess", "embed"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def enqueue_rag_job(
    db: Session,
    teacher: Teacher,
    document: RAGDocument,
    *,
    job_type: str,
    auto_embed: bool = False,
    max_attempts: int | None = None,
) -> RAGProcessingJob:
    if document.teacher_id != teacher.id:
        raise HTTPException(status_code=404, detail="RAG hujjati topilmadi.")
    if job_type not in ALLOWED_JOB_TYPES:
        raise HTTPException(status_code=422, detail="Noto‘g‘ri RAG job turi.")

    existing = (
        db.query(RAGProcessingJob)
        .filter(
            RAGProcessingJob.document_id == document.id,
            RAGProcessingJob.job_type == job_type,
            RAGProcessingJob.status.in_(ACTIVE_JOB_STATUSES),
        )
        .order_by(RAGProcessingJob.id.desc())
        .first()
    )
    if existing is not None:
        return existing

    job = RAGProcessingJob(
        document_id=document.id,
        teacher_id=teacher.id,
        job_type=job_type,
        status="pending",
        progress_percent=0,
        attempts=0,
        max_attempts=max_attempts or settings.RAG_JOB_MAX_ATTEMPTS,
        available_at=utcnow(),
        payload_json={"auto_embed": bool(auto_embed)},
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    logger.info(
        "RAG job enqueued",
        extra={
            "rag_job_id": job.id,
            "document_id": document.id,
            "teacher_id": teacher.id,
            "job_type": job_type,
        },
    )
    return job


def get_teacher_job_or_404(
    db: Session, teacher: Teacher, job_id: int
) -> RAGProcessingJob:
    job = (
        db.query(RAGProcessingJob)
        .filter(
            RAGProcessingJob.id == job_id,
            RAGProcessingJob.teacher_id == teacher.id,
        )
        .first()
    )
    if job is None:
        raise HTTPException(status_code=404, detail="RAG processing job topilmadi.")
    return job


def list_teacher_jobs(
    db: Session,
    teacher: Teacher,
    *,
    job_status: str | None = None,
    document_id: int | None = None,
    limit: int = 100,
) -> list[RAGProcessingJob]:
    query = db.query(RAGProcessingJob).filter(
        RAGProcessingJob.teacher_id == teacher.id
    )
    if job_status is not None:
        if job_status not in ACTIVE_JOB_STATUSES | TERMINAL_JOB_STATUSES:
            raise HTTPException(status_code=422, detail="Noto‘g‘ri RAG job status.")
        query = query.filter(RAGProcessingJob.status == job_status)
    if document_id is not None:
        query = query.filter(RAGProcessingJob.document_id == document_id)
    return query.order_by(RAGProcessingJob.id.desc()).limit(min(limit, 200)).all()


def retry_teacher_job(
    db: Session, teacher: Teacher, job_id: int
) -> RAGProcessingJob:
    job = get_teacher_job_or_404(db, teacher, job_id)
    if job.status != "failed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Faqat failed holatidagi job qayta navbatga qo‘yiladi.",
        )
    job.status = "pending"
    job.progress_percent = 0
    job.attempts = 0
    job.error_message = None
    job.result_json = None
    job.available_at = utcnow()
    job.locked_at = None
    job.locked_by = None
    job.started_at = None
    job.finished_at = None
    db.commit()
    db.refresh(job)
    return job


def recover_stale_jobs(db: Session) -> int:
    cutoff = utcnow() - timedelta(minutes=settings.RAG_JOB_STALE_MINUTES)
    jobs = (
        db.query(RAGProcessingJob)
        .filter(
            RAGProcessingJob.status == "running",
            RAGProcessingJob.locked_at.is_not(None),
            RAGProcessingJob.locked_at < cutoff,
        )
        .all()
    )
    for job in jobs:
        if job.attempts >= job.max_attempts:
            job.status = "failed"
            job.finished_at = utcnow()
            job.error_message = "Worker uzilganidan keyin job stale holatda qoldi."
        else:
            job.status = "pending"
            job.available_at = utcnow()
            job.error_message = "Stale job qayta navbatga qo‘yildi."
        job.locked_at = None
        job.locked_by = None
    if jobs:
        db.commit()
    return len(jobs)


def claim_next_rag_job(db: Session, worker_id: str) -> RAGProcessingJob | None:
    query = (
        db.query(RAGProcessingJob)
        .filter(
            RAGProcessingJob.status == "pending",
            RAGProcessingJob.available_at <= utcnow(),
        )
        .order_by(RAGProcessingJob.available_at.asc(), RAGProcessingJob.id.asc())
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
    job.progress_percent = max(job.progress_percent, 1)
    job.locked_at = now
    job.locked_by = worker_id
    job.started_at = job.started_at or now
    job.finished_at = None
    db.commit()
    db.refresh(job)
    return job


def _set_job_progress(db: Session, job: RAGProcessingJob, progress: int) -> None:
    job.progress_percent = max(0, min(int(progress), 100))
    job.locked_at = utcnow()
    db.commit()


def _mark_job_success(
    db: Session,
    job: RAGProcessingJob,
    result: dict,
) -> RAGProcessingJob:
    now = utcnow()
    job.status = "succeeded"
    job.progress_percent = 100
    job.result_json = result
    job.error_message = None
    job.finished_at = now
    job.locked_at = None
    job.locked_by = None
    db.commit()
    db.refresh(job)
    return job


def _mark_job_failure(
    db: Session,
    job: RAGProcessingJob,
    exc: Exception,
) -> RAGProcessingJob:
    now = utcnow()
    message = (str(exc) or exc.__class__.__name__)[:4000]
    job.error_message = message
    job.progress_percent = 0
    job.locked_at = None
    job.locked_by = None

    if job.attempts < job.max_attempts:
        delay = settings.RAG_JOB_RETRY_BASE_SECONDS * (2 ** max(job.attempts - 1, 0))
        job.status = "pending"
        job.available_at = now + timedelta(seconds=delay)
        job.finished_at = None
    else:
        job.status = "failed"
        job.finished_at = now
    db.commit()
    db.refresh(job)
    logger.exception(
        "RAG job failed",
        extra={
            "rag_job_id": job.id,
            "document_id": job.document_id,
            "job_type": job.job_type,
            "attempts": job.attempts,
            "job_status": job.status,
        },
    )
    return job


def execute_claimed_rag_job(
    db: Session,
    job: RAGProcessingJob,
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> RAGProcessingJob:
    started = time.perf_counter()
    try:
        document = db.query(RAGDocument).filter(RAGDocument.id == job.document_id).first()
        teacher = db.query(Teacher).filter(Teacher.id == job.teacher_id).first()
        if document is None or teacher is None:
            raise RuntimeError("Jobga tegishli document yoki teacher topilmadi.")

        _set_job_progress(db, job, 10)
        result: dict = {
            "document_id": document.id,
            "job_type": job.job_type,
        }

        if job.job_type in {"ingest", "reprocess"}:
            document = process_rag_document(db, document)
            _set_job_progress(db, job, 80)
            result.update(
                {
                    "chunk_count": document.chunk_count,
                    "document_status": document.status,
                }
            )
            if bool((job.payload_json or {}).get("auto_embed", False)):
                next_job = enqueue_rag_job(
                    db,
                    teacher,
                    document,
                    job_type="embed",
                    auto_embed=False,
                )
                result["next_job_id"] = next_job.id
        elif job.job_type == "embed":
            embedding_result = embed_teacher_document(
                db,
                teacher,
                document.id,
                provider=embedding_provider,
            )
            _set_job_progress(db, job, 95)
            result.update(embedding_result.model_dump())
        else:
            raise RuntimeError(f"Qo‘llab-quvvatlanmaydigan job turi: {job.job_type}")

        result["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
        succeeded = _mark_job_success(db, job, result)
        logger.info(
            "RAG job succeeded",
            extra={
                "rag_job_id": job.id,
                "document_id": job.document_id,
                "job_type": job.job_type,
                "duration_ms": result["duration_ms"],
            },
        )
        return succeeded
    except Exception as exc:
        db.rollback()
        refreshed = db.query(RAGProcessingJob).filter(RAGProcessingJob.id == job.id).first()
        if refreshed is None:
            raise
        return _mark_job_failure(db, refreshed, exc)


def run_next_rag_job_once(
    *,
    session_factory: Callable[[], Session] = SessionLocal,
    worker_id: str | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> int | None:
    resolved_worker_id = worker_id or default_worker_id()
    with session_factory() as db:
        job = claim_next_rag_job(db, resolved_worker_id)
        if job is None:
            return None
        job_id = job.id
        execute_claimed_rag_job(db, job, embedding_provider=embedding_provider)
        return job_id


def get_teacher_monitoring_summary_values(
    db: Session, teacher: Teacher
) -> dict:
    from app.services.rag_service import get_teacher_storage_usage_values

    documents = (
        db.query(RAGDocument)
        .filter(RAGDocument.teacher_id == teacher.id)
        .all()
    )
    document_statuses = {key: 0 for key in ("uploaded", "processing", "ready", "failed", "archived")}
    embedding_statuses = {key: 0 for key in ("not_started", "partial", "ready")}
    for document in documents:
        document_statuses[document.status] = document_statuses.get(document.status, 0) + 1
        embedding_statuses[document.embedding_status] = (
            embedding_statuses.get(document.embedding_status, 0) + 1
        )

    job_rows = (
        db.query(RAGProcessingJob.status, func.count(RAGProcessingJob.id))
        .filter(RAGProcessingJob.teacher_id == teacher.id)
        .group_by(RAGProcessingJob.status)
        .all()
    )
    jobs_by_status = {key: 0 for key in ("pending", "running", "succeeded", "failed", "cancelled")}
    for job_status, count in job_rows:
        jobs_by_status[str(job_status)] = int(count)

    return {
        "documents_total": len(documents),
        "documents_by_status": document_statuses,
        "embedding_by_status": embedding_statuses,
        "jobs_total": sum(jobs_by_status.values()),
        "jobs_by_status": jobs_by_status,
        "storage": get_teacher_storage_usage_values(db, teacher.id),
        "worker_enabled": settings.RAG_BACKGROUND_WORKER_ENABLED,
        "worker_poll_seconds": settings.RAG_WORKER_POLL_SECONDS,
    }
