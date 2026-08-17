"""Provider SDK xatolarini AI Mentor uchun xavfsiz kodlarga normalizatsiya qiladi."""

from __future__ import annotations

from app.llm.contracts import LLMProviderError


def normalize_provider_error(exc: Exception, provider_label: str) -> LLMProviderError:
    if isinstance(exc, LLMProviderError):
        return exc

    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)

    name = type(exc).__name__.casefold()
    message = str(exc).casefold()

    if status_code == 429 or "ratelimit" in name or "rate limit" in message or "rate_limit" in message:
        return LLMProviderError("rate_limit", f"{provider_label} rate-limit berdi.")
    if status_code in {401, 403} or "authentication" in name or "permission" in name:
        return LLMProviderError("authentication_error", f"{provider_label} autentifikatsiyasi muvaffaqiyatsiz.")
    if status_code == 408 or "timeout" in name or "timed out" in message:
        return LLMProviderError("timeout", f"{provider_label} so'rovi vaqtidan oshdi.")
    if (isinstance(status_code, int) and status_code >= 500) or "connection" in name:
        return LLMProviderError("provider_unavailable", f"{provider_label} vaqtincha ishlamayapti.")
    return LLMProviderError("provider_request_failed", f"{provider_label} so'rovida xatolik yuz berdi.")
