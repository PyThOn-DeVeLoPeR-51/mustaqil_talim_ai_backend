"""Groq OpenAI-compatible API asosidagi AI Mentor provider implementatsiyasi.

Groq OpenAI SDK bilan mos HTTP API taqdim etadi. Shu sababli loyiha qo‘shimcha
SDK o‘rnatmasdan mavjud ``openai`` paketidan foydalanadi, lekin Groq'ning
``https://api.groq.com/openai/v1`` base URL manziliga ulanadi.
"""

from __future__ import annotations

import json
import logging
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
from app.llm.error_utils import normalize_provider_error
from app.llm.prompts import (
    CHAT_SYSTEM_PROMPT,
    DIAGNOSTIC_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
)


OutputT = TypeVar("OutputT", bound=BaseModel)

GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
logger = logging.getLogger(__name__)


def _strict_json_schema(model_type: type[BaseModel]) -> dict[str, Any]:
    """Pydantic sxemasini Groq strict Structured Outputs talabiga moslaydi.

    Groq strict mode barcha object maydonlarini ``required`` va
    ``additionalProperties=false`` ko‘rinishida kutadi. Pydantic default
    qiymatli maydonlarni optional deb chiqarishi mumkin; provider esa model
    generatsiyasi uchun ularni ham majburiy qiladi.
    """

    schema = model_type.model_json_schema()

    def normalize(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                properties = node.get("properties")
                if isinstance(properties, dict):
                    node["required"] = list(properties.keys())
                node["additionalProperties"] = False

            for value in node.values():
                normalize(value)
        elif isinstance(node, list):
            for value in node:
                normalize(value)

    normalize(schema)
    return schema


class GroqAIMentorProvider(AIMentorLLMProvider):
    """Groq'dagi modelni diagnostika, reja va chat uchun ishlatadi."""

    provider_name = "groq"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        max_output_tokens: int,
        reasoning_effort: str = "medium",
    ) -> None:
        if not api_key.strip():
            raise LLMProviderError("missing_api_key", "Groq API kaliti kiritilmagan.")
        if not model.strip():
            raise LLMProviderError("missing_model", "Groq model nomi kiritilmagan.")

        normalized_reasoning = reasoning_effort.strip().casefold()
        if normalized_reasoning not in {"low", "medium", "high"}:
            raise LLMProviderError(
                "invalid_reasoning_effort",
                "GROQ_REASONING_EFFORT low, medium yoki high bo‘lishi kerak.",
            )

        try:
            from openai import OpenAI
        except (ImportError, AttributeError) as exc:
            raise LLMProviderError(
                "sdk_unavailable",
                "OpenAI Python kutubxonasi o‘rnatilmagan yoki noto‘g‘ri versiyada.",
            ) from exc

        self._client = OpenAI(
            api_key=api_key,
            base_url=GROQ_OPENAI_BASE_URL,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self.model_name = model.strip()
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = normalized_reasoning

    @staticmethod
    def _metadata(response: Any, provider: str, model: str) -> LLMCallMetadata:
        usage = getattr(response, "usage", None)
        input_tokens = None
        output_tokens = None
        total_tokens = None
        if usage is not None:
            # Chat Completions nomlari Groq/OpenAI'da prompt/completion ko‘rinishida.
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)

        return LLMCallMetadata(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            request_id=(
                getattr(response, "_request_id", None)
                or getattr(response, "id", None)
            ),
        )

    def _json_object_response(
        self,
        *,
        instructions: str,
        context: dict[str, Any],
        output_type: type[OutputT],
        strict_error: Exception | None = None,
    ) -> StructuredLLMResult[OutputT]:
        """Strict Structured Outputs ishlamasa haqiqiy Groq bilan JSON rejimida qayta urinadi."""

        schema_text = json.dumps(
            output_type.model_json_schema(),
            ensure_ascii=False,
            default=str,
        )
        fallback_instructions = (
            f"{instructions}\n\n"
            "Javob faqat bitta JSON object bo‘lsin. Markdown, izoh yoki JSON tashqarisida "
            "hech qanday matn yozmang. Quyidagi JSON Schema mazmuniga qat'iy amal qiling. "
            "Ayniqsa haftalar va vazifalar sonini promptdagi talab bo‘yicha to‘liq bering.\n"
            f"JSON_SCHEMA={schema_text}"
        )

        last_error: Exception | None = strict_error
        for attempt in range(2):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": fallback_instructions},
                        {
                            "role": "user",
                            "content": json.dumps(
                                context,
                                ensure_ascii=False,
                                default=str,
                            ),
                        },
                    ],
                    response_format={"type": "json_object"},
                    # Structured fallbackda tokenni javobning o‘ziga ko‘proq qoldiramiz.
                    reasoning_effort="low",
                    max_completion_tokens=self.max_output_tokens,
                )
                choices = getattr(response, "choices", None) or []
                if not choices:
                    raise ValueError("Groq JSON object javobi bo‘sh.")

                content = str(
                    getattr(choices[0].message, "content", "") or ""
                ).strip()
                if not content:
                    raise ValueError("Groq JSON object javobi bo‘sh.")

                parsed = output_type.model_validate_json(content)
                return StructuredLLMResult(
                    output=parsed,
                    metadata=self._metadata(
                        response,
                        self.provider_name,
                        self.model_name,
                    ),
                )
            except Exception as exc:
                normalized = normalize_provider_error(exc, "Groq")
                if normalized.code in {"rate_limit", "authentication_error", "timeout", "provider_unavailable"}:
                    raise normalized from exc
                last_error = exc
                logger.warning(
                    "Groq JSON object fallback attempt %s failed: %s: %s",
                    attempt + 1,
                    type(exc).__name__,
                    exc,
                )

        if last_error is not None:
            raise normalize_provider_error(last_error, "Groq") from last_error
        raise LLMProviderError("provider_request_failed", "Groq strukturalangan javobni yaratolmadi.")

    def _structured_response(
        self,
        *,
        instructions: str,
        context: dict[str, Any],
        output_type: type[OutputT],
        schema_name: str,
    ) -> StructuredLLMResult[OutputT]:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": instructions},
                    {
                        "role": "user",
                        "content": json.dumps(
                            context,
                            ensure_ascii=False,
                            default=str,
                        ),
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": _strict_json_schema(output_type),
                    },
                },
                reasoning_effort=self.reasoning_effort,
                max_completion_tokens=self.max_output_tokens,
            )
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise ValueError("Groq strukturalangan javob qaytarmadi.")

            content = str(getattr(choices[0].message, "content", "") or "").strip()
            if not content:
                raise ValueError("Groq strukturalangan javob qaytarmadi.")

            parsed = output_type.model_validate_json(content)
            return StructuredLLMResult(
                output=parsed,
                metadata=self._metadata(
                    response,
                    self.provider_name,
                    self.model_name,
                ),
            )
        except Exception as exc:
            normalized = normalize_provider_error(exc, "Groq")
            if normalized.code in {"rate_limit", "authentication_error", "timeout", "provider_unavailable"}:
                raise normalized from exc
            # Groq strict Structured Outputs ba'zan 400/json_validate_failed qaytarishi
            # mumkin. Mock'ka tushishdan oldin haqiqiy Groq JSON Object Mode bilan
            # qayta urinib ko‘ramiz.
            logger.warning(
                "Groq strict structured output failed; trying JSON object mode: %s: %s",
                type(exc).__name__,
                exc,
            )
            return self._json_object_response(
                instructions=instructions,
                context=context,
                output_type=output_type,
                strict_error=exc,
            )

    def analyze_diagnostic(
        self,
        context: dict[str, Any],
    ) -> StructuredLLMResult[DiagnosticAnalysisOutput]:
        return self._structured_response(
            instructions=DIAGNOSTIC_SYSTEM_PROMPT,
            context=context,
            output_type=DiagnosticAnalysisOutput,
            schema_name="ai_mentor_diagnostic_analysis",
        )

    def generate_plan(
        self,
        context: dict[str, Any],
    ) -> StructuredLLMResult[PlanGenerationOutput]:
        return self._structured_response(
            instructions=PLAN_SYSTEM_PROMPT,
            context=context,
            output_type=PlanGenerationOutput,
            schema_name="ai_mentor_four_week_plan",
        )

    def chat_reply(self, context: dict[str, Any]) -> TextLLMResult:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            context,
                            ensure_ascii=False,
                            default=str,
                        ),
                    },
                ],
                reasoning_effort=self.reasoning_effort,
                max_completion_tokens=self.max_output_tokens,
            )
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise LLMProviderError(
                    "empty_text_output",
                    "Groq chat javobi bo‘sh qaytdi.",
                )

            text = str(getattr(choices[0].message, "content", "") or "").strip()
            if not text:
                raise LLMProviderError(
                    "empty_text_output",
                    "Groq chat javobi bo‘sh qaytdi.",
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
            raise normalize_provider_error(exc, "Groq") from exc

    def chat_reply_stream(
        self,
        context: dict[str, Any],
    ) -> Generator[str, None, LLMCallMetadata]:
        """Groq chat javobini delta ko‘rinishida uzatadi.

        Groq Chat Completions API ``stream=True`` bo‘lganda iterator qaytaradi.
        Generatorning ``return`` qiymati yakuniy provider metadata bo‘lib, service
        qatlamida assistant xabari bilan birga bazaga saqlanadi. Streaming
        javobda usage kelmasa token maydonlari ``None`` bo‘lib qolishi mumkin;
        provider/model/request_id baribir saqlanadi.
        """

        try:
            stream = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            context,
                            ensure_ascii=False,
                            default=str,
                        ),
                    },
                ],
                reasoning_effort=self.reasoning_effort,
                max_completion_tokens=self.max_output_tokens,
                stream=True,
            )

            saw_content = False
            request_id: str | None = None
            response_model = self.model_name
            input_tokens: int | None = None
            output_tokens: int | None = None
            total_tokens: int | None = None

            for chunk in stream:
                request_id = (
                    getattr(chunk, "_request_id", None)
                    or getattr(chunk, "id", None)
                    or request_id
                )
                response_model = getattr(chunk, "model", None) or response_model

                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    input_tokens = getattr(usage, "prompt_tokens", None)
                    output_tokens = getattr(usage, "completion_tokens", None)
                    total_tokens = getattr(usage, "total_tokens", None)

                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue

                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta is not None else None
                if content:
                    saw_content = True
                    yield str(content)

            if not saw_content:
                raise LLMProviderError(
                    "empty_text_output",
                    "Groq streaming chat javobi bo‘sh qaytdi.",
                )

            return LLMCallMetadata(
                provider=self.provider_name,
                model=response_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                request_id=request_id,
            )
        except LLMProviderError:
            raise
        except Exception as exc:
            raise normalize_provider_error(exc, "Groq") from exc
