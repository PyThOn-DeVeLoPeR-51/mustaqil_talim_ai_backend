"""AI Mentor featurelari uchun LLM providerlarini yaratadi."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.core.config import settings
from app.llm.contracts import AIMentorLLMProvider, LLMProviderError


@lru_cache(maxsize=4)
def _cached_openai_provider(api_key: str, model: str, timeout_seconds: float, max_retries: int,
                            max_output_tokens: int, base_url: str | None) -> AIMentorLLMProvider:
    from app.llm.openai_provider import OpenAIAIMentorProvider
    return OpenAIAIMentorProvider(api_key=api_key, model=model, timeout_seconds=timeout_seconds,
                                  max_retries=max_retries, max_output_tokens=max_output_tokens, base_url=base_url)


@lru_cache(maxsize=4)
def _cached_groq_provider(api_key: str, model: str, timeout_seconds: float, max_retries: int,
                          max_output_tokens: int, reasoning_effort: str) -> AIMentorLLMProvider:
    from app.llm.groq_provider import GroqAIMentorProvider
    return GroqAIMentorProvider(api_key=api_key, model=model, timeout_seconds=timeout_seconds,
                                max_retries=max_retries, max_output_tokens=max_output_tokens,
                                reasoning_effort=reasoning_effort)


@lru_cache(maxsize=8)
def _cached_compatible_chat_provider(provider_name: str, api_key: str, model: str, base_url: str,
                                     timeout_seconds: float, max_retries: int, max_output_tokens: int,
                                     site_url: str | None, app_name: str | None) -> AIMentorLLMProvider:
    from app.llm.openai_compatible_chat_provider import OpenAICompatibleChatProvider
    headers: dict[str, str] = {}
    if provider_name == "openrouter":
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name
    return OpenAICompatibleChatProvider(provider_name=provider_name, api_key=api_key, model=model,
                                         base_url=base_url, timeout_seconds=timeout_seconds,
                                         max_retries=max_retries, max_output_tokens=max_output_tokens,
                                         default_headers=headers or None)


def _groq_provider() -> AIMentorLLMProvider:
    return _cached_groq_provider(settings.GROQ_API_KEY or "", settings.GROQ_MODEL,
        settings.LLM_TIMEOUT_SECONDS, settings.LLM_MAX_RETRIES, settings.LLM_MAX_OUTPUT_TOKENS,
        settings.GROQ_REASONING_EFFORT)


def _provider_by_name(provider_name: str) -> AIMentorLLMProvider | None:
    name = provider_name.strip().casefold()
    if name in {"", "none", "disabled", "mock"}:
        return None
    if name == "groq":
        return _groq_provider()
    if name == "openai":
        return _cached_openai_provider(settings.OPENAI_API_KEY or "", settings.OPENAI_MODEL,
            settings.LLM_TIMEOUT_SECONDS, settings.LLM_MAX_RETRIES, settings.LLM_MAX_OUTPUT_TOKENS,
            settings.OPENAI_BASE_URL or None)
    if name == "gemini":
        return _cached_compatible_chat_provider("gemini", settings.GEMINI_API_KEY or "",
            settings.GEMINI_MODEL, settings.GEMINI_BASE_URL, settings.LLM_TIMEOUT_SECONDS,
            settings.LLM_MAX_RETRIES, settings.LLM_MAX_OUTPUT_TOKENS, None, None)
    if name == "openrouter":
        return _cached_compatible_chat_provider("openrouter", settings.OPENROUTER_API_KEY or "",
            settings.OPENROUTER_MODEL, settings.OPENROUTER_BASE_URL, settings.LLM_TIMEOUT_SECONDS,
            settings.LLM_MAX_RETRIES, settings.LLM_MAX_OUTPUT_TOKENS, settings.OPENROUTER_SITE_URL,
            settings.OPENROUTER_APP_NAME)
    raise LLMProviderError("unsupported_provider", f"Qo'llab-quvvatlanmaydigan LLM provider: {provider_name}")


def get_ai_mentor_provider() -> AIMentorLLMProvider | None:
    """Legacy diagnostika routingini saqlaydi."""
    return _provider_by_name(settings.LLM_PROVIDER)


def get_ai_mentor_plan_provider() -> AIMentorLLMProvider:
    """4 haftalik reja uchun qat'iy Groq provider."""
    return _groq_provider()


def get_ai_mentor_chat_primary_provider() -> AIMentorLLMProvider | None:
    return _provider_by_name(settings.AI_MENTOR_CHAT_PRIMARY_PROVIDER)


def get_ai_mentor_chat_fallback_provider() -> AIMentorLLMProvider | None:
    return _provider_by_name(settings.AI_MENTOR_CHAT_FALLBACK_PROVIDER)


def get_llm_provider_status() -> dict[str, Any]:
    provider_name = settings.LLM_PROVIDER.strip().casefold()
    configured = True
    model: str | None = None
    if provider_name == "openai":
        configured = bool(settings.OPENAI_API_KEY and settings.OPENAI_MODEL); model = settings.OPENAI_MODEL
    elif provider_name == "groq":
        configured = bool(settings.GROQ_API_KEY and settings.GROQ_MODEL); model = settings.GROQ_MODEL
    elif provider_name != "mock":
        configured = False
    return {"provider": provider_name, "model": model, "configured": configured,
            "fallback_to_mock": settings.LLM_FALLBACK_TO_MOCK}
