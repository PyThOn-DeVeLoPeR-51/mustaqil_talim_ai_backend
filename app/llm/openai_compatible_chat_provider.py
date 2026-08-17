"""Gemini/OpenRouter uchun OpenAI-compatible chat provider."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

from app.llm.contracts import (
    AIMentorLLMProvider, DiagnosticAnalysisOutput, LLMCallMetadata, LLMProviderError,
    PlanGenerationOutput, StructuredLLMResult, TextLLMResult,
)
from app.llm.error_utils import normalize_provider_error
from app.llm.prompts import CHAT_SYSTEM_PROMPT


class OpenAICompatibleChatProvider(AIMentorLLMProvider):
    def __init__(self, *, provider_name: str, api_key: str, model: str, base_url: str,
                 timeout_seconds: float, max_retries: int, max_output_tokens: int,
                 default_headers: dict[str, str] | None = None) -> None:
        if not api_key.strip():
            raise LLMProviderError("missing_api_key", f"{provider_name} API kaliti kiritilmagan.")
        if not model.strip():
            raise LLMProviderError("missing_model", f"{provider_name} modeli kiritilmagan.")
        try:
            from openai import OpenAI
        except Exception as exc:
            raise LLMProviderError("missing_dependency", "OpenAI Python kutubxonasi o'rnatilmagan.") from exc
        self.provider_name = provider_name.strip().casefold()
        self.model_name = model.strip()
        self.max_output_tokens = max_output_tokens
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds,
                              max_retries=max_retries, default_headers=default_headers or None)

    @staticmethod
    def _metadata(response: Any, provider: str, model: str) -> LLMCallMetadata:
        usage = getattr(response, "usage", None)
        return LLMCallMetadata(
            provider=provider, model=getattr(response, "model", None) or model,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            total_tokens=getattr(usage, "total_tokens", None) if usage else None,
            request_id=getattr(response, "_request_id", None) or getattr(response, "id", None),
        )

    def analyze_diagnostic(self, context: dict[str, Any]) -> StructuredLLMResult[DiagnosticAnalysisOutput]:
        raise LLMProviderError("unsupported_operation", "Bu provider diagnostika uchun ishlatilmaydi.")

    def generate_plan(self, context: dict[str, Any]) -> StructuredLLMResult[PlanGenerationOutput]:
        raise LLMProviderError("unsupported_operation", "Bu provider reja uchun ishlatilmaydi.")

    def _messages(self, context: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False, default=str)},
        ]

    def chat_reply(self, context: dict[str, Any]) -> TextLLMResult:
        try:
            kwargs = {
                "model": self.model_name,
                "messages": self._messages(context),
                "max_tokens": self.max_output_tokens,
            }

            if self.provider_name == "gemini":
                kwargs["reasoning_effort"] = "low"

            response = self._client.chat.completions.create(**kwargs)

            choices = getattr(response, "choices", None) or []
            text = str(getattr(choices[0].message, "content", "") or "").strip() if choices else ""
            if not text:
                raise LLMProviderError("empty_text_output", f"{self.provider_name} chat javobi bo'sh qaytdi.")
            return TextLLMResult(text=text, metadata=self._metadata(response, self.provider_name, self.model_name))
        except LLMProviderError:
            raise
        except Exception as exc:
            raise normalize_provider_error(exc, self.provider_name) from exc

    def chat_reply_stream(self, context: dict[str, Any]) -> Generator[str, None, LLMCallMetadata]:
        try:
            kwargs = {
                "model": self.model_name,
                "messages": self._messages(context),
                "max_tokens": self.max_output_tokens,
                "stream": True,
            }

            if self.provider_name == "gemini":
                kwargs["reasoning_effort"] = "low"

            stream = self._client.chat.completions.create(**kwargs)

            saw = False
            request_id = None
            response_model = self.model_name
            for chunk in stream:
                request_id = getattr(chunk, "_request_id", None) or getattr(chunk, "id", None) or request_id
                response_model = getattr(chunk, "model", None) or response_model
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta is not None else None
                if content:
                    saw = True
                    yield str(content)
            if not saw:
                raise LLMProviderError("empty_text_output", f"{self.provider_name} streaming javobi bo'sh qaytdi.")
            return LLMCallMetadata(provider=self.provider_name, model=response_model, request_id=request_id)
        except LLMProviderError:
            raise
        except Exception as exc:
            raise normalize_provider_error(exc, self.provider_name) from exc
