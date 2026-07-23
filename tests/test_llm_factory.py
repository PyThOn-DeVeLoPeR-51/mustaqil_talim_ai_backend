from __future__ import annotations

import unittest
from unittest.mock import patch

from app.core.config import settings
from app.llm.factory import get_llm_provider_status


class LLMFactoryStatusTestCase(unittest.TestCase):
    def test_groq_status_reports_model_and_configuration(self) -> None:
        with (
            patch.object(settings, "LLM_PROVIDER", "groq"),
            patch.object(settings, "GROQ_API_KEY", "gsk_test"),
            patch.object(settings, "GROQ_MODEL", "openai/gpt-oss-120b"),
        ):
            status = get_llm_provider_status()

        self.assertEqual(status["provider"], "groq")
        self.assertEqual(status["model"], "openai/gpt-oss-120b")
        self.assertTrue(status["configured"])


if __name__ == "__main__":
    unittest.main()
