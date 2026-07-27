"""OpenAI Responses API asosidagi AI Mentor provider implementatsiyasi."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.contracts import (
    AIMentorLLMProvider,
    DiagnosticAnalysisOutput,
    LLMCallMetadata,
    LLMProviderError,
    PlanGenerationOutput,
    StructuredLLMResult,
    TextLLMResult,
)
from app.llm.prompts import (
    CHAT_SYSTEM_PROMPT,
    DIAGNOSTIC_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
)


OutputT = TypeVar("OutputT", bound=BaseModel)


class OpenAIAIMentorProvider(AIMentorLLMProvider):
    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        max_output_tokens: int,
        base_url: str | None = None,
    ) -> None:
        if not api_key.strip():
            raise LLMProviderError("missing_api_key", "OpenAI API kaliti kiritilmagan.")
        if not model.strip():
            raise LLMProviderError("missing_model", "OpenAI model nomi kiritilmagan.")

        try:
            from openai import OpenAI
        except (ImportError, AttributeError) as exc:
            raise LLMProviderError(
                "sdk_unavailable",
                "OpenAI Python kutubxonasi o‘rnatilmagan yoki noto‘g‘ri versiyada.",
            ) from exc

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout_seconds,
            "max_retries": max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = OpenAI(**client_kwargs)
        self.model_name = model
        self.max_output_tokens = max_output_tokens

    @staticmethod
    def _metadata(response: Any, provider: str, model: str) -> LLMCallMetadata:
        usage = getattr(response, "usage", None)
        return LLMCallMetadata(
            provider=provider,
            model=model,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            request_id=(
                getattr(response, "_request_id", None)
                or getattr(response, "id", None)
            ),
        )

    def _structured_response(
        self,
        *,
        instructions: str,
        context: dict[str, Any],
        output_type: type[OutputT],
    ) -> StructuredLLMResult[OutputT]:
        try:
            response = self._client.responses.parse(
                model=self.model_name,
                instructions=instructions,
                input=json.dumps(context, ensure_ascii=False, default=str),
                text_format=output_type,
                max_output_tokens=self.max_output_tokens,
            )
            parsed = getattr(response, "output_parsed", None)
            if parsed is None:
                raise LLMProviderError(
                    "empty_structured_output",
                    "LLM strukturalangan javob qaytarmadi.",
                )
            return StructuredLLMResult(
                output=parsed,
                metadata=self._metadata(
                    response,
                    self.provider_name,
                    self.model_name,
                ),
            )
        except LLMProviderError:
            raise
        except ValidationError as exc:
            raise LLMProviderError(
                "invalid_structured_output",
                "LLM javobi kutilgan JSON sxemasiga mos kelmadi.",
            ) from exc
        except Exception as exc:
            raise LLMProviderError(
                "provider_request_failed",
                "LLM xizmatiga so‘rov yuborishda xatolik yuz berdi.",
            ) from exc

    def analyze_diagnostic(
        self,
        context: dict[str, Any],
    ) -> StructuredLLMResult[DiagnosticAnalysisOutput]:
        return self._structured_response(
            instructions=DIAGNOSTIC_SYSTEM_PROMPT,
            context=context,
            output_type=DiagnosticAnalysisOutput,
        )

    def generate_plan(
        self,
        context: dict[str, Any],
    ) -> StructuredLLMResult[PlanGenerationOutput]:
        return self._structured_response(
            instructions=PLAN_SYSTEM_PROMPT,
            context=context,
            output_type=PlanGenerationOutput,
        )

    def chat_reply(self, context: dict[str, Any]) -> TextLLMResult:
        try:
            response = self._client.responses.create(
                model=self.model_name,
                instructions=CHAT_SYSTEM_PROMPT,
                input=json.dumps(context, ensure_ascii=False, default=str),
                max_output_tokens=self.max_output_tokens,
            )
            text = str(getattr(response, "output_text", "") or "").strip()
            if not text:
                raise LLMProviderError(
                    "empty_text_output",
                    "LLM chat javobi bo‘sh qaytdi.",
                )
            return TextLLMResult(
                text=text,
                metadata=self._metadata(
                    response,
                    self.provider_name,
                    self.model_name,
                ),
            )
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                "provider_request_failed",
                "LLM xizmatiga so‘rov yuborishda xatolik yuz berdi.",
            ) from exc

    def chat_reply_stream(
        self,
        context: dict[str, Any],
    ) -> Generator[str, None, LLMCallMetadata]:
        """OpenAI provider uchun mos streaming kontrakti.

        Hozir productionda Groq haqiqiy token streaming beradi. OpenAI provider
        esa mavjud barqaror Responses chaqiruvini ishlatib, natijani kichik
        bo‘laklarda uzatadi. Bu provider protokolini to‘liq saqlaydi va kelajakda
        Responses streamingga xavfsiz o‘tish imkonini beradi.
        """

        result = self.chat_reply(context)
        text = result.text
        for start in range(0, len(text), 48):
            yield text[start : start + 48]
        return result.metadata
