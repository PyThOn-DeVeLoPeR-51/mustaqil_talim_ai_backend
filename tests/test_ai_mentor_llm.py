from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from datetime import date
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.base import Base
from app.llm.contracts import (
    DiagnosticAnalysisOutput,
    LLMCallMetadata,
    LLMProviderError,
    PlanGenerationOutput,
    PlanItemOutput,
    PlanWeekOutput,
    StructuredLLMResult,
    TextLLMResult,
)
from app.models.ai_mentor import AIMentorDiagnosticQuestion
from app.models.student import Student
from app.models.teacher import Teacher
from app.schemas.ai_mentor import (
    AIMentorChatSessionCreate,
    AIMentorDiagnosticAnswerCreate,
    AIMentorDiagnosticAnswersSubmit,
)
from app.services.ai_mentor_service import (
    create_chat_session,
    create_generated_plan,
    seed_diagnostic_questions,
    send_chat_message,
    stream_chat_message,
    start_diagnostic_session,
    submit_diagnostic_answers,
)


class FakeLLMProvider:
    provider_name = "openai"
    model_name = "fake-test-model"

    @staticmethod
    def _metadata(total_tokens: int) -> LLMCallMetadata:
        return LLMCallMetadata(
            provider="openai",
            model="fake-test-model",
            input_tokens=total_tokens - 20,
            output_tokens=20,
            total_tokens=total_tokens,
            request_id="fake-request-id",
        )

    def analyze_diagnostic(self, context):
        del context
        return StructuredLLMResult(
            output=DiagnosticAnalysisOutput(
                summary=(
                    "Talaba muhandislik grafikasi bo‘yicha muntazam amaliyotga tayyor, "
                    "ammo vaqtni rejalashtirishni kuchaytirishi kerak."
                ),
                risk_level="low",
                strengths=["Motivatsiya yuqori", "Amaliy mashqlarga qiziqadi"],
                improvement_areas=["Vaqtni boshqarish"],
                recommended_strategies=[
                    "Haftalik jadval tuzish",
                    "Har mashg‘ulot oxirida o‘zini tekshirish",
                ],
                focus_topic="Muhandislik grafikasi",
                weekly_hours=6,
                session_minutes=30,
            ),
            metadata=self._metadata(120),
        )

    def generate_plan(self, context):
        del context
        weeks = []
        for week_number in range(1, 5):
            items = [
                PlanItemOutput(
                    item_order=item_order,
                    day_number=(item_order - 1) * 2 + 1,
                    title=f"{week_number}-hafta {item_order}-vazifa",
                    description=(
                        "Muhandislik grafikasi bo‘yicha aniq amaliy topshiriqni "
                        "bosqichma-bosqich bajaring va natijani tekshiring."
                    ),
                    activity_type="practice",
                    estimated_minutes=30,
                    resources=["Kurs materiali"],
                )
                for item_order in range(1, 4)
            ]
            weeks.append(
                PlanWeekOutput(
                    week_number=week_number,
                    title=f"{week_number}-hafta rejasi",
                    goal="Nazariya va amaliyotni izchil mustahkamlash.",
                    description="Hafta davomida uchta kichik va o‘lchanadigan vazifa bajariladi.",
                    expected_outcome="Talaba haftalik mavzuni mustaqil qo‘llay oladi.",
                    items=items,
                )
            )
        return StructuredLLMResult(
            output=PlanGenerationOutput(
                title="LLM yaratgan 4 haftalik reja",
                summary="Diagnostika natijasiga mos shaxsiy o‘quv rejasi.",
                weeks=weeks,
            ),
            metadata=self._metadata(450),
        )

    def chat_reply(self, context):
        del context
        return TextLLMResult(
            text=(
                "Bugun birinchi bajarilmagan vazifani tanlang, 30 daqiqalik taymer "
                "qo‘ying va oxirida natijani mezonlar bo‘yicha tekshiring."
            ),
            metadata=self._metadata(80),
        )


class CapturingLLMProvider(FakeLLMProvider):
    def __init__(self) -> None:
        self.last_chat_context = None

    def chat_reply(self, context):
        self.last_chat_context = context
        return super().chat_reply(context)


class FailingLLMProvider(FakeLLMProvider):
    def analyze_diagnostic(self, context):
        del context
        raise LLMProviderError("provider_request_failed", "test failure")




class FailingStreamingLLMProvider(FakeLLMProvider):
    provider_name = "groq"
    model_name = "openai/gpt-oss-120b"

    def chat_reply_stream(self, context):
        del context
        yield "Boshlang‘ich partial javob"
        raise LLMProviderError("provider_request_failed", "stream failure")

class AIMentorLLMTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.original_provider = settings.LLM_PROVIDER
        self.original_fallback = settings.LLM_FALLBACK_TO_MOCK
        self.original_rag_chat_enabled = settings.RAG_CHAT_ENABLED
        settings.LLM_PROVIDER = "openai"
        settings.LLM_FALLBACK_TO_MOCK = False
        settings.RAG_CHAT_ENABLED = True

        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.db: Session = session_factory()

        teacher = Teacher(
            first_name="LLM",
            last_name="Teacher",
            email="llm-teacher@example.com",
            password_hash="hash",
        )
        self.db.add(teacher)
        self.db.flush()
        self.student = Student(
            teacher_id=teacher.id,
            full_name="LLM Student",
            login="llm_student",
            password_hash="hash",
            is_active=True,
            direction="Muhandislik grafikasi",
        )
        self.db.add(self.student)
        self.db.commit()
        self.db.refresh(self.student)

    def tearDown(self) -> None:
        settings.LLM_PROVIDER = self.original_provider
        settings.LLM_FALLBACK_TO_MOCK = self.original_fallback
        settings.RAG_CHAT_ENABLED = self.original_rag_chat_enabled
        self.db.close()
        self.engine.dispose()

    def _answer_payload(self) -> AIMentorDiagnosticAnswersSubmit:
        seed_diagnostic_questions(self.db)
        questions = {
            question.question_code: question
            for question in self.db.query(AIMentorDiagnosticQuestion).all()
        }
        return AIMentorDiagnosticAnswersSubmit(
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

    def test_full_llm_flow_without_network(self) -> None:
        provider = FakeLLMProvider()
        seed_diagnostic_questions(self.db)
        diagnostic_session = start_diagnostic_session(self.db, self.student)

        with patch(
            "app.services.ai_mentor_diagnostic_service.get_ai_mentor_provider",
            return_value=provider,
        ):
            diagnostic = submit_diagnostic_answers(
                self.db,
                self.student,
                diagnostic_session.id,
                self._answer_payload(),
            )

        self.assertEqual(diagnostic.analysis_json["provider"], "openai")
        self.assertEqual(
            diagnostic.analysis_json["llm_metadata"]["total_tokens"],
            120,
        )

        with patch(
            "app.services.ai_mentor_plan_service.get_ai_mentor_provider",
            return_value=provider,
        ):
            plan_response = create_generated_plan(
                self.db,
                self.student,
                diagnostic_session_id=diagnostic.id,
                start_date_value=date(2026, 7, 22),
            )

        self.assertEqual(plan_response.plan.generation_source, "llm")
        self.assertEqual(len(plan_response.plan.weeks), 4)
        self.assertEqual(plan_response.progress.total_items, 12)
        self.assertEqual(
            plan_response.plan.generation_metadata["total_tokens"],
            450,
        )

        chat_session = create_chat_session(
            self.db,
            self.student,
            AIMentorChatSessionCreate(plan_id=plan_response.plan.id),
        )
        with patch(
            "app.services.ai_mentor_chat_service.get_ai_mentor_provider",
            return_value=provider,
        ):
            response = send_chat_message(
                self.db,
                self.student,
                chat_session.id,
                "Bugungi vazifani qanday boshlayman?",
            )

        self.assertEqual(response.assistant_message.model_name, "fake-test-model")
        self.assertEqual(response.assistant_message.token_count, 80)
        self.assertEqual(
            response.assistant_message.metadata_json["provider"],
            "openai",
        )

    def test_chat_includes_rag_context_and_persists_source_metadata(self) -> None:
        provider = CapturingLLMProvider()
        chat_session = create_chat_session(
            self.db,
            self.student,
            AIMentorChatSessionCreate(),
        )

        fake_search = SimpleNamespace(
            query="Kredit-modul tizimida mustaqil ta'limning ahamiyati nimada?",
            model="intfloat/multilingual-e5-small",
            hits=[
                SimpleNamespace(
                    score=0.8421,
                    document=SimpleNamespace(
                        id=7,
                        title="Kredit-modul bo‘yicha ma'ruza",
                        original_filename="lecture.docx",
                        task_id=None,
                    ),
                    chunk=SimpleNamespace(
                        id=71,
                        chunk_index=3,
                        section_title="Mustaqil ta'lim",
                        page_number_start=None,
                        page_number_end=None,
                        content=(
                            "Kredit-modul tizimida mustaqil ta'lim talabaning "
                            "individual o‘quv yuklamasini rejalashtirishga xizmat qiladi."
                        ),
                    ),
                )
            ],
        )

        with (
            patch(
                "app.services.ai_mentor_chat_service.student_has_searchable_embeddings",
                return_value=True,
            ),
            patch(
                "app.services.ai_mentor_chat_service.semantic_search_student_documents",
                return_value=fake_search,
            ),
            patch(
                "app.services.ai_mentor_chat_service.get_ai_mentor_provider",
                return_value=provider,
            ),
        ):
            response = send_chat_message(
                self.db,
                self.student,
                chat_session.id,
                "Mustaqil ta'limning o‘rni qanday?",
            )

        self.assertIsNotNone(provider.last_chat_context)
        kb = provider.last_chat_context["knowledge_base"]
        self.assertEqual(kb["status"], "ready")
        self.assertEqual(len(kb["sources"]), 1)
        self.assertEqual(kb["sources"][0]["source_id"], 1)
        self.assertIn("Kredit-modul", kb["sources"][0]["content"] )

        rag_metadata = response.assistant_message.metadata_json["rag"]
        self.assertTrue(rag_metadata["used_for_answer"])
        self.assertEqual(rag_metadata["source_count"], 1)
        self.assertEqual(rag_metadata["sources"][0]["document_id"], 7)
        self.assertEqual(rag_metadata["sources"][0]["chunk_id"], 71)
        self.assertNotIn("content", rag_metadata["sources"][0])
        self.assertIn("excerpt", rag_metadata["sources"][0])

    def test_streaming_chat_replaces_partial_with_mock_on_provider_failure(self) -> None:
        settings.LLM_PROVIDER = "groq"
        settings.LLM_FALLBACK_TO_MOCK = True
        chat_session = create_chat_session(
            self.db,
            self.student,
            AIMentorChatSessionCreate(),
        )

        with patch(
            "app.services.ai_mentor_chat_service.get_ai_mentor_provider",
            return_value=FailingStreamingLLMProvider(),
        ):
            events = list(
                stream_chat_message(
                    self.db,
                    self.student,
                    chat_session.id,
                    "Vaqtni rejalashtirishga yordam bering",
                )
            )

        stream_text = "".join(events)
        self.assertIn("event: fallback", stream_text)
        self.assertIn('"replace": true', stream_text)
        self.assertIn("event: done", stream_text)

        from app.services.ai_mentor_chat_service import build_chat_session_detail

        detail = build_chat_session_detail(self.db, chat_session)
        self.assertEqual(len(detail.messages), 2)
        assistant = detail.messages[-1]
        self.assertEqual(assistant.model_name, "mock-ai-mentor-v1")
        self.assertEqual(assistant.metadata_json["provider"], "mock")
        self.assertEqual(
            assistant.metadata_json["fallback_from_provider"],
            "groq",
        )

    def test_diagnostic_falls_back_to_mock(self) -> None:
        settings.LLM_FALLBACK_TO_MOCK = True
        seed_diagnostic_questions(self.db)
        diagnostic_session = start_diagnostic_session(self.db, self.student)

        with patch(
            "app.services.ai_mentor_diagnostic_service.get_ai_mentor_provider",
            return_value=FailingLLMProvider(),
        ):
            diagnostic = submit_diagnostic_answers(
                self.db,
                self.student,
                diagnostic_session.id,
                self._answer_payload(),
            )

        self.assertEqual(diagnostic.analysis_json["provider"], "mock")
        self.assertEqual(
            diagnostic.analysis_json["fallback"]["reason"],
            "provider_request_failed",
        )


if __name__ == "__main__":
    unittest.main()
