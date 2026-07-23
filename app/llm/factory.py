"""Sozlamaga qarab AI Mentor LLM providerini yaratadi."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.core.config import settings
from app.llm.contracts import AIMentorLLMProvider, LLMProviderError


@lru_cache(maxsize=4)
def _cached_openai_provider(
    api_key: str,
    model: str,
    timeout_seconds: float,
    max_retries: int,
    max_output_tokens: int,
    base_url: str | None,
) -> AIMentorLLMProvider:
    from app.llm.openai_provider import OpenAIAIMentorProvider

    return OpenAIAIMentorProvider(
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_output_tokens=max_output_tokens,
        base_url=base_url,
    )


@lru_cache(maxsize=4)
def _cached_groq_provider(
    api_key: str,
    model: str,
    timeout_seconds: float,
    max_retries: int,
    max_output_tokens: int,
    reasoning_effort: str,
) -> AIMentorLLMProvider:
    from app.llm.groq_provider import GroqAIMentorProvider

    return GroqAIMentorProvider(
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )


def get_ai_mentor_provider() -> AIMentorLLMProvider | None:
    provider_name = settings.LLM_PROVIDER.strip().casefold()

    if provider_name == "mock":
        return None

    if provider_name == "openai":
        return _cached_openai_provider(
            settings.OPENAI_API_KEY or "",
            settings.OPENAI_MODEL,
            settings.LLM_TIMEOUT_SECONDS,
            settings.LLM_MAX_RETRIES,
            settings.LLM_MAX_OUTPUT_TOKENS,
            settings.OPENAI_BASE_URL or None,
        )

    if provider_name == "groq":
        return _cached_groq_provider(
            settings.GROQ_API_KEY or "",
            settings.GROQ_MODEL,
            settings.LLM_TIMEOUT_SECONDS,
            settings.LLM_MAX_RETRIES,
            settings.LLM_MAX_OUTPUT_TOKENS,
            settings.GROQ_REASONING_EFFORT,
        )

    raise LLMProviderError(
        "unsupported_provider",
        f"Qo‘llab-quvvatlanmaydigan LLM provider: {settings.LLM_PROVIDER}",
    )


def get_llm_provider_status() -> dict[str, Any]:
    provider_name = settings.LLM_PROVIDER.strip().casefold()
    configured = True
    model: str | None = None

    if provider_name == "openai":
        configured = bool(settings.OPENAI_API_KEY and settings.OPENAI_MODEL)
        model = settings.OPENAI_MODEL
    elif provider_name == "groq":
        configured = bool(settings.GROQ_API_KEY and settings.GROQ_MODEL)
        model = settings.GROQ_MODEL
    elif provider_name != "mock":
        configured = False

    return {
        "provider": provider_name,
        "model": model,
        "configured": configured,
        "fallback_to_mock": settings.LLM_FALLBACK_TO_MOCK,
    }
