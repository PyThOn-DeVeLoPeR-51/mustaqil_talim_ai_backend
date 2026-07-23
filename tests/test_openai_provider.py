from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from app.llm.contracts import DiagnosticAnalysisOutput
from app.llm.openai_provider import OpenAIAIMentorProvider


class FakeUsage:
    input_tokens = 40
    output_tokens = 20
    total_tokens = 60


class FakeResponses:
    def __init__(self) -> None:
        self.last_parse_kwargs = None
        self.last_create_kwargs = None

    def parse(self, **kwargs):
        self.last_parse_kwargs = kwargs
        return types.SimpleNamespace(
            output_parsed=DiagnosticAnalysisOutput(
                summary=(
                    "Talaba mustaqil ta'limni reja asosida davom ettirishi va "
                    "amaliy mashqlarni muntazam bajarishi kerak."
                ),
                risk_level="medium",
                strengths=["Maqsadi aniq"],
                improvement_areas=["Vaqtni boshqarish"],
                recommended_strategies=[
                    "Haftalik reja tuzish",
                    "Mashg‘ulot yakunida o‘zini tekshirish",
                ],
                focus_topic="Muhandislik grafikasi",
                weekly_hours=4,
                session_minutes=30,
            ),
            usage=FakeUsage(),
            id="response-structured",
        )

    def create(self, **kwargs):
        self.last_create_kwargs = kwargs
        return types.SimpleNamespace(
            output_text="Birinchi vazifani 30 daqiqalik kichik bosqichdan boshlang.",
            usage=FakeUsage(),
            id="response-text",
        )


class FakeOpenAI:
    last_instance = None

    def __init__(self, **kwargs) -> None:
        self.client_kwargs = kwargs
        self.responses = FakeResponses()
        FakeOpenAI.last_instance = self


class OpenAIProviderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        module = types.ModuleType("openai")
        module.OpenAI = FakeOpenAI
        self.module_patch = patch.dict(sys.modules, {"openai": module})
        self.module_patch.start()

    def tearDown(self) -> None:
        self.module_patch.stop()

    def test_structured_and_text_calls_are_mapped(self) -> None:
        provider = OpenAIAIMentorProvider(
            api_key="test-key",
            model="test-model",
            timeout_seconds=15,
            max_retries=1,
            max_output_tokens=1000,
        )

        diagnostic = provider.analyze_diagnostic({"answers": {}})
        chat = provider.chat_reply({"student_message": "Test"})

        self.assertEqual(diagnostic.output.risk_level, "medium")
        self.assertEqual(diagnostic.metadata.total_tokens, 60)
        self.assertEqual(chat.metadata.request_id, "response-text")
        self.assertIn("30 daqiqalik", chat.text)

        client = FakeOpenAI.last_instance
        self.assertEqual(client.client_kwargs["timeout"], 15)
        self.assertEqual(client.client_kwargs["max_retries"], 1)
        self.assertEqual(
            client.responses.last_parse_kwargs["text_format"],
            DiagnosticAnalysisOutput,
        )
        self.assertEqual(
            client.responses.last_create_kwargs["max_output_tokens"],
            1000,
        )


if __name__ == "__main__":
    unittest.main()
