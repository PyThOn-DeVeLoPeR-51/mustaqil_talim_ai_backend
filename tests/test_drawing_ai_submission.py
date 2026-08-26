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
if importlib.util.find_spec("pgvector") is None:
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
from app.models.student import Student
from app.models.task import Task, TaskAssignment
from app.models.teacher import Teacher
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
        factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.db: Session = factory()

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

    def test_managed_storage_persists_submission_and_ai_overlay(self) -> None:
        original_provider = settings.STORAGE_PROVIDER
        original_upload_dir = settings.LOCAL_UPLOAD_DIR
        try:
            with tempfile.TemporaryDirectory() as directory:
                settings.STORAGE_PROVIDER = "local"
                settings.LOCAL_UPLOAD_DIR = str(Path(directory) / "uploads")
                reset_storage_backend()

                drawing_file = UploadFile(
                    filename="drawing.png",
                    file=BytesIO(b"drawing-bytes"),
                )

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
                ):
                    submission = create_submission_for_student(
                        db=self.db,
                        student=self.student,
                        task_id=self.task.id,
                        drawing_file=drawing_file,
                    )

                self.assertTrue(submission.uploaded_file_path.startswith("submissions/"))
                self.assertTrue(submission.overlay_path.startswith(f"results/submissions/{submission.id}/"))
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

    def test_teacher_description_reaches_optional_evaluator(self) -> None:
        fake_result = {
            "total_score": 80.0,
            "ai_json_result": {"mode": "optional"},
            "overlay_path": None,
            "table_json": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            drawing_path = Path(directory) / "drawing.png"
            drawing_path.write_bytes(b"test-placeholder")
            with patch(
                "app.services.submission_service.evaluate_submission_with_ai",
                return_value=fake_result,
            ) as evaluate_mock:
                submission = create_submission_for_student(
                    db=self.db,
                    student=self.student,
                    task_id=self.task.id,
                    uploaded_file_path=str(drawing_path),
                )

        evaluate_mock.assert_called_once_with(
            mode="optional",
            student_file_path=str(drawing_path),
            reference_file_path=None,
            task_text="3 ta proyeksiya va o‘lchamlar bo‘lishi kerak",
        )
        self.assertEqual(submission.status, "evaluated")
        self.assertEqual(submission.total_score, 80.0)


if __name__ == "__main__":
    unittest.main()
