from __future__ import annotations

import json
import sys
import types
import unittest
from unittest.mock import patch

from app.llm.contracts import DiagnosticAnalysisOutput, PlanGenerationOutput
from app.llm.groq_provider import GROQ_OPENAI_BASE_URL, GroqAIMentorProvider


class FakeUsage:
    prompt_tokens = 55
    completion_tokens = 25
    total_tokens = 80


class FakeChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            chunks = [
                types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            delta=types.SimpleNamespace(content="Birinchi ")
                        )
                    ],
                    usage=None,
                    id="groq-stream-1",
                    model="openai/gpt-oss-120b",
                ),
                types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            delta=types.SimpleNamespace(content="vazifani bajaring.")
                        )
                    ],
                    usage=FakeUsage(),
                    id="groq-stream-1",
                    model="openai/gpt-oss-120b",
                ),
            ]
            return iter(chunks)
        if kwargs.get("response_format"):
            content = json.dumps(
                {
                    "summary": (
                        "Talaba mustaqil ta'limni izchil davom ettirishga tayyor va "
                        "amaliy mashqlarni reja asosida bajarishi foydali bo‘ladi."
                    ),
                    "risk_level": "medium",
                    "strengths": ["Maqsadi aniq"],
                    "improvement_areas": ["Vaqtni boshqarish"],
                    "recommended_strategies": [
                        "Haftalik reja tuzish",
                        "Mashg‘ulot yakunida o‘zini tekshirish",
                    ],
                    "focus_topic": "Muhandislik grafikasi",
                    "weekly_hours": 4,
                    "session_minutes": 30,
                },
                ensure_ascii=False,
            )
        else:
            content = "Birinchi vazifani kichik bosqichga ajratib, 30 daqiqa ishlang."

        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=content)
                )
            ],
            usage=FakeUsage(),
            id=f"groq-response-{len(self.calls)}",
        )


class FakeOpenAI:
    last_instance = None

    def __init__(self, **kwargs) -> None:
        self.client_kwargs = kwargs
        self.chat = types.SimpleNamespace(completions=FakeChatCompletions())
        FakeOpenAI.last_instance = self


class StrictFailThenJsonObjectCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response_format = kwargs.get("response_format") or {}
        if response_format.get("type") == "json_schema":
            raise RuntimeError("400 json_validate_failed")

        content = json.dumps(
            {
                "title": "4 haftalik shaxsiy mustaqil ta'lim rejasi",
                "summary": "Diagnostika asosida bosqichma-bosqich tuzilgan individual reja.",
                "weeks": [
                    {
                        "week_number": week,
                        "title": f"{week}-hafta",
                        "goal": "Mavzuni nazariya va amaliyot orqali izchil o‘zlashtirish.",
                        "description": "Hafta davomida qisqa nazariya, mashq va o‘zini tekshirish bajariladi.",
                        "expected_outcome": "Talaba mavzuni tushuntirib, mustaqil mashq bajara oladi.",
                        "items": [
                            {
                                "item_order": item,
                                "day_number": min(7, item * 2),
                                "title": f"{week}.{item} vazifa",
                                "description": "Mavzu bo‘yicha aniq va o‘lchanadigan mustaqil mashqni bajaring.",
                                "activity_type": "practice",
                                "estimated_minutes": 30,
                                "resources": ["Kurs materiali"],
                            }
                            for item in (1, 2, 3)
                        ],
                    }
                    for week in (1, 2, 3, 4)
                ],
            },
            ensure_ascii=False,
        )
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))],
            usage=FakeUsage(),
            id="groq-json-fallback",
        )


class StrictFailThenJsonObjectOpenAI:
    last_instance = None

    def __init__(self, **kwargs) -> None:
        self.client_kwargs = kwargs
        self.chat = types.SimpleNamespace(
            completions=StrictFailThenJsonObjectCompletions()
        )
        StrictFailThenJsonObjectOpenAI.last_instance = self


class GroqProviderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        module = types.ModuleType("openai")
        module.OpenAI = FakeOpenAI
        self.module_patch = patch.dict(sys.modules, {"openai": module})
        self.module_patch.start()

    def tearDown(self) -> None:
        self.module_patch.stop()

    def test_structured_and_text_calls_are_mapped(self) -> None:
        provider = GroqAIMentorProvider(
            api_key="gsk_test",
            model="openai/gpt-oss-120b",
            timeout_seconds=20,
            max_retries=2,
            max_output_tokens=1200,
            reasoning_effort="medium",
        )

        diagnostic = provider.analyze_diagnostic({"answers": {}})
        chat = provider.chat_reply({"student_message": "Test"})

        self.assertIsInstance(diagnostic.output, DiagnosticAnalysisOutput)
        self.assertEqual(diagnostic.metadata.provider, "groq")
        self.assertEqual(diagnostic.metadata.total_tokens, 80)
        self.assertEqual(chat.metadata.request_id, "groq-response-2")
        self.assertIn("30 daqiqa", chat.text)

        client = FakeOpenAI.last_instance
        self.assertEqual(client.client_kwargs["base_url"], GROQ_OPENAI_BASE_URL)
        self.assertEqual(client.client_kwargs["timeout"], 20)
        self.assertEqual(client.client_kwargs["max_retries"], 2)

        structured_call = client.chat.completions.calls[0]
        self.assertEqual(structured_call["model"], "openai/gpt-oss-120b")
        self.assertEqual(structured_call["reasoning_effort"], "medium")
        self.assertEqual(structured_call["max_completion_tokens"], 1200)
        self.assertTrue(
            structured_call["response_format"]["json_schema"]["strict"]
        )
        schema = structured_call["response_format"]["json_schema"]["schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))


    def test_streaming_chat_yields_deltas_and_metadata(self) -> None:
        provider = GroqAIMentorProvider(
            api_key="gsk_test",
            model="openai/gpt-oss-120b",
            timeout_seconds=20,
            max_retries=2,
            max_output_tokens=1200,
            reasoning_effort="medium",
        )

        stream = provider.chat_reply_stream({"student_message": "Test"})
        deltas = []
        while True:
            try:
                deltas.append(next(stream))
            except StopIteration as stop:
                metadata = stop.value
                break

        self.assertEqual("".join(deltas), "Birinchi vazifani bajaring.")
        self.assertEqual(metadata.provider, "groq")
        self.assertEqual(metadata.model, "openai/gpt-oss-120b")
        self.assertEqual(metadata.total_tokens, 80)
        self.assertEqual(metadata.request_id, "groq-stream-1")
        call = FakeOpenAI.last_instance.chat.completions.calls[-1]
        self.assertTrue(call["stream"])
        self.assertEqual(call["reasoning_effort"], "medium")

    def test_structured_output_retries_with_real_json_object_mode(self) -> None:
        module = types.ModuleType("openai")
        module.OpenAI = StrictFailThenJsonObjectOpenAI
        with patch.dict(sys.modules, {"openai": module}):
            provider = GroqAIMentorProvider(
                api_key="gsk_test",
                model="openai/gpt-oss-120b",
                timeout_seconds=20,
                max_retries=2,
                max_output_tokens=3000,
                reasoning_effort="medium",
            )
            result = provider.generate_plan({"student": {"id": 1}})

        self.assertIsInstance(result.output, PlanGenerationOutput)
        self.assertEqual(len(result.output.weeks), 4)
        self.assertEqual(result.metadata.provider, "groq")
        calls = StrictFailThenJsonObjectOpenAI.last_instance.chat.completions.calls
        self.assertEqual(calls[0]["response_format"]["type"], "json_schema")
        self.assertEqual(calls[1]["response_format"], {"type": "json_object"})
        self.assertEqual(calls[1]["reasoning_effort"], "low")

    def test_invalid_reasoning_effort_is_rejected(self) -> None:
        with self.assertRaisesRegex(Exception, "GROQ_REASONING_EFFORT"):
            GroqAIMentorProvider(
                api_key="gsk_test",
                model="openai/gpt-oss-120b",
                timeout_seconds=20,
                max_retries=2,
                max_output_tokens=1200,
                reasoning_effort="turbo",
            )


if __name__ == "__main__":
    unittest.main()
