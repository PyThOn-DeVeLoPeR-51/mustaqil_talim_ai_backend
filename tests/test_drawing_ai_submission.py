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

# This isolated integration test can run even when the optional pgvector Python
# package is not installed in the test runner. Production still uses the real
# dependency declared in requirements.txt.
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

from fastapi import HTTPException, UploadFile
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
from app.schemas.submission import SubmissionRead
from app.services.drawing_job_service import run_next_drawing_job_once
from app.services.submission_service import create_submission_for_student
from app.storage import storage_exists
from app.storage.service import reset_storage_backend


class DrawingAISubmissionIntegrationTestCase(unittest.TestCase):
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
            first_name="Drawing",
            last_name="Teacher",
            email="drawing-teacher@example.com",
            password_hash="hash",
        )
        self.db.add(teacher)
        self.db.flush()
        self.student = Student(
            teacher_id=teacher.id,
            full_name="Drawing Student",
            login="drawing_student",
            password_hash="hash",
            is_active=True,
        )
        self.db.add(self.student)
        self.db.flush()
        self.task = Task(
            teacher_id=teacher.id,
            title="Optional drawing",
            description="3 ta proyeksiya va o‘lchamlar bo‘lishi kerak",
            mode="optional",
            deadline=datetime.now(timezone.utc) + timedelta(days=1),
            is_active=True,
        )
        self.db.add(self.task)
        self.db.flush()
        self.db.add(TaskAssignment(task_id=self.task.id, student_id=self.student.id))
        self.db.commit()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def _upload(self, name: str = "drawing.png") -> UploadFile:
        return UploadFile(filename=name, file=BytesIO(b"drawing-bytes"))

    def test_submission_is_queued_then_worker_persists_ai_overlay(self) -> None:
        original_provider = settings.STORAGE_PROVIDER
        original_upload_dir = settings.LOCAL_UPLOAD_DIR
        try:
            with tempfile.TemporaryDirectory() as directory:
                settings.STORAGE_PROVIDER = "local"
                settings.LOCAL_UPLOAD_DIR = str(Path(directory) / "uploads")
                reset_storage_backend()

                def fake_evaluate(**kwargs):
                    output_dir = Path(kwargs["output_dir"])
                    overlay = output_dir / "overlay.png"
                    overlay.write_bytes(b"overlay-bytes")
                    return {
                        "total_score": 88.0,
                        "ai_json_result": {
                            "student_file": kwargs["student_file_path"],
                            "artifacts": {"overlay": str(overlay)},
                        },
                        "overlay_path": str(overlay),
                        "table_json": [],
                    }

                with patch(
                    "app.services.submission_service.evaluate_submission_with_ai",
                    side_effect=fake_evaluate,
                ) as evaluate_mock:
                    submission = create_submission_for_student(
                        db=self.db,
                        student=self.student,
                        task_id=self.task.id,
                        drawing_file=self._upload(),
                    )

                    self.assertEqual(submission.status, "pending")
                    self.assertEqual(submission.evaluation_status, "queued")
                    self.assertIsNone(submission.overlay_path)
                    response = SubmissionRead.model_validate(submission)
                    self.assertEqual(response.evaluation_status, "queued")
                    self.assertEqual(response.evaluation_progress_percent, 0)
                    evaluate_mock.assert_not_called()

                    job = (
                        self.db.query(DrawingEvaluationJob)
                        .filter(DrawingEvaluationJob.submission_id == submission.id)
                        .one()
                    )
                    self.assertEqual(job.status, "pending")

                    processed_id = run_next_drawing_job_once(
                        session_factory=self.factory,
                        worker_id="test-worker",
                    )
                    self.assertEqual(processed_id, job.id)

                self.db.expire_all()
                submission = (
                    self.db.query(Submission)
                    .filter(Submission.id == submission.id)
                    .one()
                )
                job = (
                    self.db.query(DrawingEvaluationJob)
                    .filter(DrawingEvaluationJob.submission_id == submission.id)
                    .one()
                )

                self.assertEqual(submission.status, "evaluated")
                self.assertEqual(submission.evaluation_status, "evaluated")
                self.assertEqual(job.status, "succeeded")
                self.assertEqual(job.progress_percent, 100)
                self.assertTrue(submission.uploaded_file_path.startswith("submissions/"))
                self.assertTrue(
                    submission.overlay_path.startswith(
                        f"results/submissions/{submission.id}/"
                    )
                )
                self.assertTrue(storage_exists(submission.uploaded_file_path))
                self.assertTrue(storage_exists(submission.overlay_path))
                self.assertEqual(
                    submission.ai_json_result["student_file"],
                    submission.uploaded_file_path,
                )
                self.assertEqual(
                    submission.ai_json_result["artifacts"]["overlay"],
                    submission.overlay_path,
                )
        finally:
            settings.STORAGE_PROVIDER = original_provider
            settings.LOCAL_UPLOAD_DIR = original_upload_dir
            reset_storage_backend()

    def test_teacher_description_reaches_worker_evaluator(self) -> None:
        original_provider = settings.STORAGE_PROVIDER
        original_upload_dir = settings.LOCAL_UPLOAD_DIR
        fake_result = {
            "total_score": 80.0,
            "ai_json_result": {"mode": "optional"},
            "overlay_path": None,
            "table_json": [],
        }
        try:
            with tempfile.TemporaryDirectory() as directory:
                settings.STORAGE_PROVIDER = "local"
                settings.LOCAL_UPLOAD_DIR = str(Path(directory) / "uploads")
                reset_storage_backend()

                with patch(
                    "app.services.submission_service.evaluate_submission_with_ai",
                    return_value=fake_result,
                ) as evaluate_mock:
                    submission = create_submission_for_student(
                        db=self.db,
                        student=self.student,
                        task_id=self.task.id,
                        drawing_file=self._upload(),
                    )
                    run_next_drawing_job_once(
                        session_factory=self.factory,
                        worker_id="test-worker",
                    )

                kwargs = evaluate_mock.call_args.kwargs
                self.assertEqual(kwargs["mode"], "optional")
                self.assertEqual(
                    kwargs["task_text"],
                    "3 ta proyeksiya va o‘lchamlar bo‘lishi kerak",
                )
                self.assertIsNone(kwargs["reference_file_path"])

                self.db.refresh(submission)
                self.assertEqual(submission.status, "evaluated")
                self.assertEqual(submission.total_score, 80.0)
        finally:
            settings.STORAGE_PROVIDER = original_provider
            settings.LOCAL_UPLOAD_DIR = original_upload_dir
            reset_storage_backend()

    def test_second_attempt_is_blocked_while_first_is_queued(self) -> None:
        original_provider = settings.STORAGE_PROVIDER
        original_upload_dir = settings.LOCAL_UPLOAD_DIR
        try:
            with tempfile.TemporaryDirectory() as directory:
                settings.STORAGE_PROVIDER = "local"
                settings.LOCAL_UPLOAD_DIR = str(Path(directory) / "uploads")
                reset_storage_backend()

                first = create_submission_for_student(
                    db=self.db,
                    student=self.student,
                    task_id=self.task.id,
                    drawing_file=self._upload("first.png"),
                )
                self.assertEqual(first.status, "pending")

                with self.assertRaises(HTTPException) as ctx:
                    create_submission_for_student(
                        db=self.db,
                        student=self.student,
                        task_id=self.task.id,
                        drawing_file=self._upload("second.png"),
                    )
                self.assertEqual(ctx.exception.status_code, 409)
                self.assertEqual(
                    self.db.query(Submission)
                    .filter(
                        Submission.student_id == self.student.id,
                        Submission.task_id == self.task.id,
                    )
                    .count(),
                    1,
                )
        finally:
            settings.STORAGE_PROVIDER = original_provider
            settings.LOCAL_UPLOAD_DIR = original_upload_dir
            reset_storage_backend()


if __name__ == "__main__":
    unittest.main()
