from __future__ import annotations

import os
import unittest
from datetime import date

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from sqlalchemy import create_engine
from fastapi import HTTPException
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.ai_mentor import AIMentorDiagnosticQuestion
from app.models.student import Student
from app.models.teacher import Teacher
from app.schemas.ai_mentor import (
    AIMentorChatSessionCreate,
    AIMentorDiagnosticAnswerCreate,
    AIMentorDiagnosticAnswersSubmit,
    AIMentorPlanItemProgressUpdate,
)
from app.services.ai_mentor_service import (
    create_chat_session,
    create_mock_plan,
    seed_diagnostic_questions,
    get_student_plan_or_404,
    send_mock_chat_message,
    start_diagnostic_session,
    submit_diagnostic_answers,
    update_plan_item_progress,
)


class AIMentorServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.db: Session = session_factory()

        teacher = Teacher(
            first_name="Test",
            last_name="Teacher",
            email="teacher@example.com",
            password_hash="hash",
        )
        self.db.add(teacher)
        self.db.flush()

        self.student = Student(
            teacher_id=teacher.id,
            full_name="Test Student",
            login="test_student",
            password_hash="hash",
            is_active=True,
        )
        self.db.add(self.student)
        self.db.commit()
        self.db.refresh(self.student)

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def _complete_diagnostic(self):
        seed_diagnostic_questions(self.db)
        session = start_diagnostic_session(self.db, self.student)
        questions = {
            question.question_code: question
            for question in self.db.query(AIMentorDiagnosticQuestion).all()
        }

        payload = AIMentorDiagnosticAnswersSubmit(
            answers=[
                AIMentorDiagnosticAnswerCreate(
                    question_id=questions["primary_goal"].id,
                    answer_json="deep_learning",
                ),
                AIMentorDiagnosticAnswerCreate(
                    question_id=questions["current_level"].id,
                    answer_json="intermediate",
                ),
                AIMentorDiagnosticAnswerCreate(
                    question_id=questions["difficult_areas"].id,
                    answer_json=["time_management", "practice"],
                ),
                AIMentorDiagnosticAnswerCreate(
                    question_id=questions["weekly_hours"].id,
                    answer_json=6,
                ),
                AIMentorDiagnosticAnswerCreate(
                    question_id=questions["learning_formats"].id,
                    answer_json=["practice", "video"],
                ),
                AIMentorDiagnosticAnswerCreate(
                    question_id=questions["study_days"].id,
                    answer_json=["monday", "wednesday", "saturday"],
                ),
                AIMentorDiagnosticAnswerCreate(
                    question_id=questions["session_minutes"].id,
                    answer_json=30,
                ),
                AIMentorDiagnosticAnswerCreate(
                    question_id=questions["motivation_level"].id,
                    answer_json=8,
                ),
                AIMentorDiagnosticAnswerCreate(
                    question_id=questions["focus_topic"].id,
                    answer_text="Muhandislik grafikasi",
                ),
            ]
        )
        return submit_diagnostic_answers(
            self.db,
            self.student,
            session.id,
            payload,
        )

    def test_seed_is_idempotent(self) -> None:
        first = seed_diagnostic_questions(self.db)
        second = seed_diagnostic_questions(self.db)

        self.assertEqual(first["created"], 10)
        self.assertEqual(second["created"], 0)
        self.assertEqual(second["unchanged"], 10)

    def test_full_mock_flow(self) -> None:
        diagnostic = self._complete_diagnostic()
        self.assertEqual(diagnostic.status, "completed")
        self.assertEqual(len(diagnostic.answers), 9)

        plan_response = create_mock_plan(
            self.db,
            self.student,
            diagnostic_session_id=diagnostic.id,
            start_date_value=date(2026, 7, 22),
        )
        self.assertEqual(len(plan_response.plan.weeks), 4)
        self.assertEqual(plan_response.progress.total_items, 12)

        first_item = plan_response.plan.weeks[0].items[0]
        updated_item = update_plan_item_progress(
            self.db,
            self.student,
            first_item.id,
            AIMentorPlanItemProgressUpdate(status="completed"),
        )
        self.assertEqual(updated_item.status, "completed")
        self.assertIsNotNone(updated_item.completed_at)

        chat_session = create_chat_session(
            self.db,
            self.student,
            AIMentorChatSessionCreate(plan_id=plan_response.plan.id),
        )
        response = send_mock_chat_message(
            self.db,
            self.student,
            chat_session.id,
            "Rejam bo‘yicha keyingi vazifa qaysi?",
        )
        self.assertEqual(response.user_message.sequence_number, 1)
        self.assertEqual(response.assistant_message.sequence_number, 2)
        self.assertIn("12 ta vazifa", response.assistant_message.content)

    def test_latest_active_question_version_is_used(self) -> None:
        seed_diagnostic_questions(self.db)
        original = (
            self.db.query(AIMentorDiagnosticQuestion)
            .filter(AIMentorDiagnosticQuestion.question_code == "primary_goal")
            .one()
        )
        self.db.add(
            AIMentorDiagnosticQuestion(
                question_code="primary_goal",
                version=2,
                question_text="Yangilangan asosiy maqsad savoli",
                category="goal",
                answer_type="single_choice",
                options_json=original.options_json,
                sort_order=original.sort_order,
                is_required=True,
                is_active=True,
            )
        )
        self.db.commit()

        from app.services.ai_mentor_service import get_active_diagnostic_questions

        questions = get_active_diagnostic_questions(self.db)
        primary_goal_versions = [
            question.version
            for question in questions
            if question.question_code == "primary_goal"
        ]
        self.assertEqual(primary_goal_versions, [2])

    def test_student_cannot_access_another_students_plan(self) -> None:
        diagnostic = self._complete_diagnostic()
        plan_response = create_mock_plan(
            self.db,
            self.student,
            diagnostic_session_id=diagnostic.id,
        )

        other_student = Student(
            teacher_id=self.student.teacher_id,
            full_name="Other Student",
            login="other_student",
            password_hash="hash",
            is_active=True,
        )
        self.db.add(other_student)
        self.db.commit()
        self.db.refresh(other_student)

        with self.assertRaises(HTTPException) as error_context:
            get_student_plan_or_404(
                self.db,
                other_student,
                plan_response.plan.id,
            )

        self.assertEqual(error_context.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
