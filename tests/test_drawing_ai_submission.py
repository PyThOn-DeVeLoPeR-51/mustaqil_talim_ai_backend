from __future__ import annotations

import importlib.util
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

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.student import Student
from app.models.task import Task, TaskAssignment
from app.models.teacher import Teacher
from app.services.submission_service import create_submission_for_student


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
