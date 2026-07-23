from __future__ import annotations

from app.core.config import settings

import importlib.util
import os
import sys
import types
import unittest

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

# Izolyatsiyalangan sinov muhitida production auth kutubxonalari bo‘lmasa,
# router importini tekshirish uchun minimal test stublari ishlatiladi.
if importlib.util.find_spec("jose") is None:
    jose_module = types.ModuleType("jose")

    class JWTError(Exception):
        pass

    class JWTStub:
        @staticmethod
        def decode(*args, **kwargs):
            return {}

        @staticmethod
        def encode(*args, **kwargs):
            return "test-token"

    jose_module.JWTError = JWTError
    jose_module.jwt = JWTStub()
    sys.modules["jose"] = jose_module

if importlib.util.find_spec("passlib") is None:
    passlib_module = types.ModuleType("passlib")
    passlib_context_module = types.ModuleType("passlib.context")
    passlib_hash_module = types.ModuleType("passlib.hash")

    class CryptContext:
        def __init__(self, *args, **kwargs):
            pass

        def hash(self, password: str) -> str:
            return f"hashed:{password}"

        def verify(self, password: str, password_hash: str) -> bool:
            return password_hash == f"hashed:{password}"

    class BcryptSHA256:
        @staticmethod
        def hash(password: str) -> str:
            return f"hashed:{password}"

        @staticmethod
        def verify(password: str, password_hash: str) -> bool:
            return password_hash == f"hashed:{password}"

    passlib_context_module.CryptContext = CryptContext
    passlib_hash_module.bcrypt_sha256 = BcryptSHA256()
    passlib_module.context = passlib_context_module
    passlib_module.hash = passlib_hash_module
    sys.modules["passlib"] = passlib_module
    sys.modules["passlib.context"] = passlib_context_module
    sys.modules["passlib.hash"] = passlib_hash_module

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.v1.endpoints.ai_mentor import router
from app.db.base import Base
from app.db.database import get_db
from app.models.student import Student
from app.models.teacher import Teacher
from app.services.ai_mentor_service import seed_diagnostic_questions
from app.services.auth_service import get_current_student


class AIMentorAPITestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.original_llm_provider = settings.LLM_PROVIDER
        self.original_llm_fallback = settings.LLM_FALLBACK_TO_MOCK

        settings.LLM_PROVIDER = "mock"
        settings.LLM_FALLBACK_TO_MOCK = True

        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.db: Session = session_factory()

        teacher = Teacher(
            first_name="API",
            last_name="Teacher",
            email="api-teacher@example.com",
            password_hash="hash",
        )
        self.db.add(teacher)
        self.db.flush()

        self.student = Student(
            teacher_id=teacher.id,
            full_name="API Student",
            login="api_student",
            password_hash="hash",
            is_active=True,
        )
        self.db.add(self.student)
        self.db.commit()
        self.db.refresh(self.student)
        seed_diagnostic_questions(self.db)

        app = FastAPI()
        app.include_router(router, prefix="/ai-mentor")

        def override_get_db():
            yield self.db

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_current_student] = lambda: self.student
        self.app = app
        self.client = TestClient(app)

    def tearDown(self) -> None:
        settings.LLM_PROVIDER = self.original_llm_provider
        settings.LLM_FALLBACK_TO_MOCK = self.original_llm_fallback

        self.client.close()
        self.db.close()
        self.engine.dispose()

    def _answer_payload(self, questions: list[dict]) -> dict:
        by_code = {question["question_code"]: question for question in questions}
        return {
            "answers": [
                {
                    "question_id": by_code["primary_goal"]["id"],
                    "answer_json": "deep_learning",
                },
                {
                    "question_id": by_code["current_level"]["id"],
                    "answer_json": "intermediate",
                },
                {
                    "question_id": by_code["difficult_areas"]["id"],
                    "answer_json": ["time_management", "practice"],
                },
                {
                    "question_id": by_code["weekly_hours"]["id"],
                    "answer_json": 6,
                },
                {
                    "question_id": by_code["learning_formats"]["id"],
                    "answer_json": ["practice", "video"],
                },
                {
                    "question_id": by_code["study_days"]["id"],
                    "answer_json": ["monday", "wednesday", "saturday"],
                },
                {
                    "question_id": by_code["session_minutes"]["id"],
                    "answer_json": 30,
                },
                {
                    "question_id": by_code["motivation_level"]["id"],
                    "answer_json": 8,
                },
                {
                    "question_id": by_code["focus_topic"]["id"],
                    "answer_text": "Muhandislik grafikasi",
                },
            ]
        }

    def test_full_mock_api_flow(self) -> None:
        status_response = self.client.get("/ai-mentor/llm/status")
        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.json()["provider"], "mock")
        self.assertTrue(status_response.json()["configured"])

        questions_response = self.client.get("/ai-mentor/diagnostic/questions")
        self.assertEqual(questions_response.status_code, 200)
        questions = questions_response.json()
        self.assertEqual(len(questions), 10)

        session_response = self.client.post(
            "/ai-mentor/diagnostic/sessions"
        )
        self.assertEqual(session_response.status_code, 201)
        diagnostic_session_id = session_response.json()["id"]

        completed_response = self.client.post(
            f"/ai-mentor/diagnostic/sessions/{diagnostic_session_id}/answers",
            json=self._answer_payload(questions),
        )
        self.assertEqual(completed_response.status_code, 200)
        self.assertEqual(completed_response.json()["status"], "completed")
        self.assertEqual(len(completed_response.json()["answers"]), 9)

        latest_response = self.client.get(
            "/ai-mentor/diagnostic/sessions/latest"
        )
        self.assertEqual(latest_response.status_code, 200)
        self.assertEqual(latest_response.json()["id"], diagnostic_session_id)

        plan_response = self.client.post(
            "/ai-mentor/plans/generate",
            json={
                "diagnostic_session_id": diagnostic_session_id,
                "start_date": "2026-07-22",
            },
        )
        self.assertEqual(plan_response.status_code, 201)
        plan_data = plan_response.json()
        self.assertEqual(len(plan_data["plan"]["weeks"]), 4)
        self.assertEqual(plan_data["progress"]["total_items"], 12)
        plan_id = plan_data["plan"]["id"]
        first_item_id = plan_data["plan"]["weeks"][0]["items"][0]["id"]

        progress_response = self.client.patch(
            f"/ai-mentor/plan-items/{first_item_id}/progress",
            json={"status": "completed"},
        )
        self.assertEqual(progress_response.status_code, 200)
        self.assertEqual(progress_response.json()["status"], "completed")

        current_plan_response = self.client.get("/ai-mentor/plans/current")
        self.assertEqual(current_plan_response.status_code, 200)
        self.assertEqual(current_plan_response.json()["plan"]["id"], plan_id)
        self.assertEqual(
            current_plan_response.json()["progress"]["completed_items"],
            1,
        )

        chat_session_response = self.client.post(
            "/ai-mentor/chat/sessions",
            json={"plan_id": plan_id, "title": "Reja bo‘yicha yordam"},
        )
        self.assertEqual(chat_session_response.status_code, 201)
        chat_session_id = chat_session_response.json()["id"]

        chat_response = self.client.post(
            f"/ai-mentor/chat/sessions/{chat_session_id}/messages",
            json={"content": "Rejam bo‘yicha keyingi vazifa qaysi?"},
        )
        self.assertEqual(chat_response.status_code, 201)
        self.assertEqual(chat_response.json()["user_message"]["sequence_number"], 1)
        self.assertEqual(
            chat_response.json()["assistant_message"]["sequence_number"],
            2,
        )
        self.assertIn(
            "12 ta vazifa",
            chat_response.json()["assistant_message"]["content"],
        )

        chat_detail_response = self.client.get(
            f"/ai-mentor/chat/sessions/{chat_session_id}"
        )
        self.assertEqual(chat_detail_response.status_code, 200)
        self.assertEqual(len(chat_detail_response.json()["messages"]), 2)

        close_chat_response = self.client.patch(
            f"/ai-mentor/chat/sessions/{chat_session_id}",
            json={"status": "closed"},
        )
        self.assertEqual(close_chat_response.status_code, 200)
        self.assertEqual(close_chat_response.json()["status"], "closed")

        rejected_message_response = self.client.post(
            f"/ai-mentor/chat/sessions/{chat_session_id}/messages",
            json={"content": "Yana bir savol"},
        )
        self.assertEqual(rejected_message_response.status_code, 409)

        self.assertEqual(
            len(self.client.get("/ai-mentor/diagnostic/sessions").json()),
            1,
        )
        self.assertEqual(len(self.client.get("/ai-mentor/plans").json()), 1)
        self.assertEqual(
            len(self.client.get("/ai-mentor/chat/sessions").json()),
            1,
        )

    def test_student_scope_is_enforced_at_api_level(self) -> None:
        questions = self.client.get("/ai-mentor/diagnostic/questions").json()
        session_id = self.client.post(
            "/ai-mentor/diagnostic/sessions",
            json={},
        ).json()["id"]
        self.client.post(
            f"/ai-mentor/diagnostic/sessions/{session_id}/answers",
            json=self._answer_payload(questions),
        )
        plan_id = self.client.post(
            "/ai-mentor/plans/mock",
            json={"diagnostic_session_id": session_id},
        ).json()["plan"]["id"]

        other_student = Student(
            teacher_id=self.student.teacher_id,
            full_name="Other API Student",
            login="other_api_student",
            password_hash="hash",
            is_active=True,
        )
        self.db.add(other_student)
        self.db.commit()
        self.db.refresh(other_student)
        self.app.dependency_overrides[get_current_student] = lambda: other_student

        response = self.client.get(f"/ai-mentor/plans/{plan_id}")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
