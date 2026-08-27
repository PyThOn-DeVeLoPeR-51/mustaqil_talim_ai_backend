from __future__ import annotations

import importlib.util
from io import BytesIO
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

if "pgvector" not in sys.modules and importlib.util.find_spec("pgvector") is None:
    from sqlalchemy import JSON
    from sqlalchemy.types import TypeDecorator

    pgvector_module = types.ModuleType("pgvector")
    pgvector_psycopg2 = types.ModuleType("pgvector.psycopg2")
    pgvector_sqlalchemy = types.ModuleType("pgvector.sqlalchemy")

    def register_vector(connection):
        return None

    class VECTOR(TypeDecorator):
        impl = JSON
        cache_ok = True

        def __init__(self, dim=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim = dim
            self.dimensions = dim

    pgvector_psycopg2.register_vector = register_vector
    pgvector_sqlalchemy.VECTOR = VECTOR
    sys.modules["pgvector"] = pgvector_module
    sys.modules["pgvector.psycopg2"] = pgvector_psycopg2
    sys.modules["pgvector.sqlalchemy"] = pgvector_sqlalchemy

from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.base import Base
from app.models.drawing_job import DrawingEvaluationJob
from app.models.student import Student
from app.models.submission import Submission
from app.models.task import Task, TaskAssignment
from app.models.teacher import Teacher
from app.services.drawing_job_service import (
    claim_next_drawing_job,
    recover_stale_drawing_jobs,
    run_next_drawing_job_once,
)
from app.services.submission_service import create_submission_for_student
from app.storage import delete_storage_object
from app.storage.service import reset_storage_backend


class DrawingJobQueueTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.db: Session = self.factory()

        teacher = Teacher(
            first_name="Queue",
            last_name="Teacher",
            email="drawing-queue-teacher@example.com",
            password_hash="hash",
        )
        self.db.add(teacher)
        self.db.flush()
        self.student = Student(
            teacher_id=teacher.id,
            full_name="Queue Student",
            login="drawing_queue_student",
            password_hash="hash",
            is_active=True,
        )
        self.db.add(self.student)
        self.db.flush()
        self.task = Task(
            teacher_id=teacher.id,
            title="Queue task",
            description="Queue integration",
            mode="optional",
            deadline=datetime.now(timezone.utc) + timedelta(days=1),
            is_active=True,
        )
        self.db.add(self.task)
        self.db.flush()
        self.db.add(TaskAssignment(task_id=self.task.id, student_id=self.student.id))
        self.db.commit()

        self.old_provider = settings.STORAGE_PROVIDER
        self.old_upload_dir = settings.LOCAL_UPLOAD_DIR
        self.tempdir = tempfile.TemporaryDirectory()
        settings.STORAGE_PROVIDER = "local"
        settings.LOCAL_UPLOAD_DIR = str(Path(self.tempdir.name) / "uploads")
        reset_storage_backend()

    def tearDown(self) -> None:
        settings.STORAGE_PROVIDER = self.old_provider
        settings.LOCAL_UPLOAD_DIR = self.old_upload_dir
        reset_storage_backend()
        self.tempdir.cleanup()
        self.db.close()
        self.engine.dispose()

    def _create_submission(self) -> Submission:
        return create_submission_for_student(
            db=self.db,
            student=self.student,
            task_id=self.task.id,
            drawing_file=UploadFile(
                filename="drawing.png",
                file=BytesIO(b"drawing-bytes"),
            ),
        )

    def test_failed_job_is_retried_then_succeeds(self) -> None:
        submission = self._create_submission()
        old_retry = settings.DRAWING_JOB_RETRY_BASE_SECONDS
        settings.DRAWING_JOB_RETRY_BASE_SECONDS = 0

        calls = {"count": 0}

        def flaky_evaluate(**kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("temporary evaluator failure")
            return {
                "total_score": 77.0,
                "ai_json_result": {"ok": True},
                "overlay_path": None,
                "table_json": [],
            }

        try:
            with patch(
                "app.services.submission_service.evaluate_submission_with_ai",
                side_effect=flaky_evaluate,
            ):
                run_next_drawing_job_once(
                    session_factory=self.factory,
                    worker_id="retry-worker",
                )
                self.db.expire_all()
                job = (
                    self.db.query(DrawingEvaluationJob)
                    .filter(DrawingEvaluationJob.submission_id == submission.id)
                    .one()
                )
                current = self.db.get(Submission, submission.id)
                self.assertEqual(job.status, "pending")
                self.assertEqual(job.attempts, 1)
                self.assertEqual(current.status, "pending")

                run_next_drawing_job_once(
                    session_factory=self.factory,
                    worker_id="retry-worker",
                )

            self.db.expire_all()
            job = (
                self.db.query(DrawingEvaluationJob)
                .filter(DrawingEvaluationJob.submission_id == submission.id)
                .one()
            )
            current = self.db.get(Submission, submission.id)
            self.assertEqual(job.status, "succeeded")
            self.assertEqual(job.attempts, 2)
            self.assertEqual(current.status, "evaluated")
            self.assertEqual(current.total_score, 77.0)
        finally:
            settings.DRAWING_JOB_RETRY_BASE_SECONDS = old_retry

    def test_140_submissions_are_queued_without_running_ai_inline(self) -> None:
        # 140 xil student bir vaqtda yuborgan holatning DB queue qismi:
        # requestlar faqat submission + job yaratadi, evaluator inline ishlamaydi.
        teacher_id = self.student.teacher_id

        students = [self.student]
        for index in range(1, 140):
            student = Student(
                teacher_id=teacher_id,
                full_name=f"Load Student {index}",
                login=f"drawing_load_{index}",
                password_hash="hash",
                is_active=True,
            )
            self.db.add(student)
            self.db.flush()
            self.db.add(
                TaskAssignment(
                    task_id=self.task.id,
                    student_id=student.id,
                )
            )
            students.append(student)
        self.db.commit()

        with patch(
            "app.services.submission_service.evaluate_submission_with_ai"
        ) as evaluate_mock:
            for student in students:
                create_submission_for_student(
                    db=self.db,
                    student=student,
                    task_id=self.task.id,
                    uploaded_file_path=(
                        f"submissions/load/student-{student.id}/drawing.png"
                    ),
                )

        evaluate_mock.assert_not_called()
        self.assertEqual(self.db.query(Submission).count(), 140)
        self.assertEqual(self.db.query(DrawingEvaluationJob).count(), 140)
        self.assertEqual(
            self.db.query(DrawingEvaluationJob)
            .filter(DrawingEvaluationJob.status == "pending")
            .count(),
            140,
        )


    def test_missing_storage_object_fails_without_retry(self) -> None:
        submission = self._create_submission()
        delete_storage_object(submission.uploaded_file_path)

        run_next_drawing_job_once(
            session_factory=self.factory,
            worker_id="missing-object-worker",
        )

        self.db.expire_all()
        job = (
            self.db.query(DrawingEvaluationJob)
            .filter(DrawingEvaluationJob.submission_id == submission.id)
            .one()
        )
        current = self.db.get(Submission, submission.id)

        self.assertEqual(job.status, "failed")
        self.assertEqual(job.attempts, 1)
        self.assertEqual(current.status, "failed")
        self.assertIn("topilmadi", str(job.error_message).lower())

    def test_stale_running_job_is_requeued(self) -> None:
        submission = self._create_submission()
        job = claim_next_drawing_job(self.db, "stale-worker")
        self.assertIsNotNone(job)
        job.locked_at = datetime.now(timezone.utc) - timedelta(
            minutes=settings.DRAWING_JOB_STALE_MINUTES + 1
        )
        self.db.commit()

        recovered = recover_stale_drawing_jobs(self.db)
        self.assertEqual(recovered, 1)

        self.db.expire_all()
        job = (
            self.db.query(DrawingEvaluationJob)
            .filter(DrawingEvaluationJob.submission_id == submission.id)
            .one()
        )
        self.assertEqual(job.status, "pending")
        self.assertIsNone(job.locked_at)


if __name__ == "__main__":
    unittest.main()
