"""AI Mentor chat sessiyalari, xabarlar tarixi va LLM/mock javoblari."""

import json
import logging
import re
from collections.abc import Generator, Iterator
from typing import Any

from fastapi import status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.llm.contracts import LLMCallMetadata, LLMProviderError
from app.llm.factory import (
    get_ai_mentor_chat_fallback_provider,
    get_ai_mentor_chat_primary_provider,
)
from app.models.ai_mentor import (
    AIMentorChatMessage,
    AIMentorChatSession,
    AIMentorPlanItem,
    AIMentorPlanWeek,
)
from app.models.student import Student
from app.schemas.ai_mentor import (
    AIMentorChatMessageCreate,
    AIMentorChatResponse,
    AIMentorChatSessionCreate,
    AIMentorChatSessionDetail,
    AIMentorChatSessionRead,
    AIMentorChatSessionUpdate,
)
from app.services.ai_mentor_common import http_error, utcnow
from app.services.ai_mentor_diagnostic_service import (
    get_latest_completed_diagnostic_session,
)
from app.services.ai_mentor_plan_service import (
    calculate_plan_progress,
    get_active_or_latest_plan,
    get_student_plan_or_404,
)
from app.services.ai_mentor_usage_service import (
    record_llm_usage,
    student_has_groq_chat_quota,
)
from app.services.rag_embedding_service import (
    semantic_search_student_documents,
    student_has_searchable_embeddings,
)


logger = logging.getLogger(__name__)


_RAG_EXCERPT_MAX_CHARS = 500
_RAG_BOUNDARY_MIN_RATIO = 0.60


def _truncate_rag_text(
    value: str,
    max_chars: int,
    *,
    add_ellipsis: bool = True,
) -> str:
    """Matnni gap/paragraf/so‘z chegarasida xavfsiz qisqartiradi."""

    text = (value or "").strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    available_chars = max_chars - 1 if add_ellipsis and max_chars > 1 else max_chars
    candidate = text[:available_chars].rstrip()
    minimum_boundary = max(1, int(available_chars * _RAG_BOUNDARY_MIN_RATIO))

    cut_at: int | None = None

    paragraph_boundary = candidate.rfind("\n\n", minimum_boundary)
    if paragraph_boundary >= minimum_boundary:
        cut_at = paragraph_boundary

    sentence_boundaries = [
        match.end()
        for match in re.finditer(r"[.!?…](?:[\"'”’)]*)", candidate)
        if match.end() >= minimum_boundary
    ]
    if sentence_boundaries:
        sentence_cut = sentence_boundaries[-1]
        if cut_at is None or sentence_cut > cut_at:
            cut_at = sentence_cut

    if cut_at is None:
        whitespace_boundary = max(
            candidate.rfind(" ", minimum_boundary),
            candidate.rfind("\n", minimum_boundary),
            candidate.rfind("\t", minimum_boundary),
        )
        if whitespace_boundary >= minimum_boundary:
            cut_at = whitespace_boundary

    if cut_at is None:
        cut_at = len(candidate)

    truncated = candidate[:cut_at].rstrip()
    if not add_ellipsis:
        return truncated
    return f"{truncated}…"


def create_chat_session(
    db: Session,
    student: Student,
    payload: AIMentorChatSessionCreate,
) -> AIMentorChatSession:
    if payload.plan_id is not None:
        get_student_plan_or_404(db, student, payload.plan_id)

    session = AIMentorChatSession(
        student_id=student.id,
        plan_id=payload.plan_id,
        title=payload.title or "AI Mentor bilan suhbat",
        status="active",
        context_json=payload.context_json,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_student_chat_session_or_404(
    db: Session,
    student: Student,
    session_id: int,
) -> AIMentorChatSession:
    chat_session = (
        db.query(AIMentorChatSession)
        .filter(
            AIMentorChatSession.id == session_id,
            AIMentorChatSession.student_id == student.id,
        )
        .first()
    )

    if chat_session is None:
        raise http_error(status.HTTP_404_NOT_FOUND, "Chat sessiyasi topilmadi.")

    return chat_session


def _next_message_sequence(db: Session, session_id: int) -> int:
    latest_sequence = (
        db.query(func.max(AIMentorChatMessage.sequence_number))
        .filter(AIMentorChatMessage.session_id == session_id)
        .scalar()
    )
    return int(latest_sequence or 0) + 1


def _add_chat_message_without_commit(
    db: Session,
    chat_session: AIMentorChatSession,
    payload: AIMentorChatMessageCreate,
) -> AIMentorChatMessage:
    message = AIMentorChatMessage(
        session_id=chat_session.id,
        sequence_number=_next_message_sequence(db, chat_session.id),
        role=payload.role,
        content=payload.content,
        model_name=payload.model_name,
        token_count=payload.token_count,
        metadata_json=payload.metadata_json,
    )
    db.add(message)
    db.flush()
    chat_session.last_message_at = utcnow()
    return message


def build_chat_session_detail(
    db: Session,
    chat_session: AIMentorChatSession,
) -> AIMentorChatSessionDetail:
    messages = (
        db.query(AIMentorChatMessage)
        .filter(AIMentorChatMessage.session_id == chat_session.id)
        .order_by(AIMentorChatMessage.sequence_number.asc())
        .all()
    )

    session_data = AIMentorChatSessionRead.model_validate(chat_session).model_dump()
    return AIMentorChatSessionDetail(
        **session_data,
        messages=messages,
    )


def _mock_chat_reply(
    db: Session,
    student: Student,
    content: str,
) -> str:
    normalized = content.casefold()
    plan = get_active_or_latest_plan(db, student)

    if any(keyword in normalized for keyword in ("reja", "hafta", "vazifa")):
        if plan is None:
            return (
                "Siz uchun hali 4 haftalik reja yaratilmagan. Avval diagnostik savollarga "
                "javob bering, so‘ng shaxsiy reja yaratamiz."
            )

        progress = calculate_plan_progress(db, plan.id)
        return (
            f"Sizning “{plan.title}” rejangizda {progress.total_items} ta vazifa bor. "
            f"Hozirgacha {progress.completed_items} tasi bajarilgan "
            f"({progress.progress_percent}%). Keyingi bajarilmagan vazifadan davom eting."
        )

    if any(keyword in normalized for keyword in ("vaqt", "ulgur", "jadval")):
        return (
            "Vazifani kichik qismlarga ajrating: avval 5 daqiqa maqsadni aniqlang, "
            "keyin 20–30 daqiqa asosiy ishni bajaring va oxirida 5 daqiqa natijani "
            "tekshiring. Rejadagi bitta vazifani tanlab, shu siklni bugun bajaring."
        )

    if any(keyword in normalized for keyword in ("qiyin", "tushun", "xato")):
        return (
            "Qiyin bo‘lgan joyni aniq bitta savolga aylantiring. Avval tayanch qoida yoki "
            "namunani ko‘ring, keyin o‘xshash kichik mashq bajaring va xatoni sababiga "
            "ko‘ra yozib chiqing."
        )

    if any(keyword in normalized for keyword in ("motiv", "xohish", "charch")):
        return (
            "Bugun katta natija talab qilmang. Rejangizdagi eng qisqa vazifani tanlab, "
            "faqat boshlashga e’tibor qarating. Bajarilgach holatini “completed” qilib "
            "belgilash keyingi qadamni ko‘rishni osonlashtiradi."
        )

    return (
        "Savolingiz qabul qilindi. Hozirgi mock AI rejimida men reja, vaqtni boshqarish, "
        "qiyin mavzuni kichik bosqichlarga ajratish va o‘quv progressi bo‘yicha yordam "
        "beraman. Muammoni aniq mavzu yoki vazifa bilan yozing."
    )


def _chat_history_context(
    db: Session,
    session_id: int,
) -> list[dict[str, Any]]:
    messages = (
        db.query(AIMentorChatMessage)
        .filter(AIMentorChatMessage.session_id == session_id)
        .order_by(AIMentorChatMessage.sequence_number.desc())
        .limit(settings.LLM_MAX_CHAT_HISTORY)
        .all()
    )
    messages.reverse()
    return [
        {
            "role": message.role,
            "content": message.content,
            "sequence_number": message.sequence_number,
        }
        for message in messages
    ]


def _plan_context(
    db: Session,
    student: Student,
    chat_session: AIMentorChatSession,
) -> dict[str, Any] | None:
    if chat_session.plan_id is not None:
        plan = get_student_plan_or_404(db, student, chat_session.plan_id)
    else:
        plan = get_active_or_latest_plan(db, student)

    if plan is None:
        return None

    progress = calculate_plan_progress(db, plan.id)
    weeks = (
        db.query(AIMentorPlanWeek)
        .filter(AIMentorPlanWeek.plan_id == plan.id)
        .order_by(AIMentorPlanWeek.week_number.asc())
        .all()
    )
    week_data: list[dict[str, Any]] = []
    for week in weeks:
        items = (
            db.query(AIMentorPlanItem)
            .filter(AIMentorPlanItem.plan_week_id == week.id)
            .order_by(AIMentorPlanItem.item_order.asc())
            .all()
        )
        week_data.append(
            {
                "week_number": week.week_number,
                "title": week.title,
                "goal": week.goal,
                "items": [
                    {
                        "title": item.title,
                        "description": item.description,
                        "status": item.status,
                        "estimated_minutes": item.estimated_minutes,
                    }
                    for item in items
                ],
            }
        )

    return {
        "title": plan.title,
        "summary": plan.summary,
        "generation_source": plan.generation_source,
        "progress": progress.model_dump(),
        "weeks": week_data,
    }


def _rag_chat_context(
    db: Session,
    student: Student,
    content: str,
) -> dict[str, Any]:
    """Student savoli uchun o‘qituvchi materiallaridan xavfsiz RAG kontekst yaratadi.

    RAG xatosi AI Mentor chatni yiqitmaydi. Embedding/model yoki pgvector vaqtincha
    ishlamasa, chat oddiy LLM kontekstida davom etadi va metadata'da sabab saqlanadi.
    """

    if not settings.RAG_CHAT_ENABLED:
        return {"enabled": False, "status": "disabled", "sources": []}
    if settings.AI_MENTOR_CHAT_PRIMARY_PROVIDER.strip().casefold() == "mock":
        return {"enabled": True, "status": "llm_mock", "sources": []}

    try:
        if not student_has_searchable_embeddings(db, student):
            return {"enabled": True, "status": "no_embedded_sources", "sources": []}

        result = semantic_search_student_documents(
            db,
            student,
            query=content,
            top_k=settings.RAG_CHAT_TOP_K,
            min_score=settings.RAG_CHAT_MIN_SCORE,
        )
    except Exception as exc:  # RAG chat uchun graceful degradation
        logger.warning("AI Mentor RAG retrieval skipped: %s: %s", type(exc).__name__, exc)
        return {
            "enabled": True,
            "status": "retrieval_failed",
            "sources": [],
            "error_type": type(exc).__name__,
        }

    sources: list[dict[str, Any]] = []
    remaining_chars = max(int(settings.RAG_CHAT_MAX_CONTEXT_CHARS), 0)
    for hit in result.hits:
        if remaining_chars <= 0:
            break

        content_text = (hit.chunk.content or "").strip()
        if not content_text:
            continue
        if len(content_text) > remaining_chars:
            content_text = _truncate_rag_text(
                content_text,
                remaining_chars,
                add_ellipsis=True,
            )
        remaining_chars -= len(content_text)

        sources.append(
            {
                "source_id": len(sources) + 1,
                "document_id": hit.document.id,
                "document_title": hit.document.title,
                "original_filename": hit.document.original_filename,
                "task_id": hit.document.task_id,
                "chunk_id": hit.chunk.id,
                "chunk_index": hit.chunk.chunk_index,
                "section_title": hit.chunk.section_title,
                "page_number_start": hit.chunk.page_number_start,
                "page_number_end": hit.chunk.page_number_end,
                "score": hit.score,
                "content": content_text,
            }
        )

    return {
        "enabled": True,
        "status": "ready" if sources else "no_relevant_sources",
        "query": result.query,
        "embedding_model": result.model,
        "sources": sources,
    }


def _rag_metadata_from_context(
    context: dict[str, Any],
    *,
    used_for_answer: bool,
) -> dict[str, Any]:
    knowledge_base = context.get("knowledge_base") or {}
    metadata_sources: list[dict[str, Any]] = []
    for source in knowledge_base.get("sources", []):
        metadata_sources.append(
            {
                "source_id": source.get("source_id"),
                "document_id": source.get("document_id"),
                "document_title": source.get("document_title"),
                "original_filename": source.get("original_filename"),
                "task_id": source.get("task_id"),
                "chunk_id": source.get("chunk_id"),
                "chunk_index": source.get("chunk_index"),
                "section_title": source.get("section_title"),
                "page_number_start": source.get("page_number_start"),
                "page_number_end": source.get("page_number_end"),
                "score": source.get("score"),
                "excerpt": _truncate_rag_text(
                    str(source.get("content") or ""),
                    _RAG_EXCERPT_MAX_CHARS,
                    add_ellipsis=True,
                ),
            }
        )

    return {
        "enabled": bool(knowledge_base.get("enabled")),
        "status": knowledge_base.get("status", "unknown"),
        "used_for_answer": bool(used_for_answer and metadata_sources),
        "source_count": len(metadata_sources),
        "embedding_model": knowledge_base.get("embedding_model"),
        "sources": metadata_sources,
    }


def _llm_chat_context(
    db: Session,
    student: Student,
    chat_session: AIMentorChatSession,
    content: str,
) -> dict[str, Any]:
    diagnostic = get_latest_completed_diagnostic_session(db, student)
    return {
        "student_profile": {
            "university": student.university,
            "direction": student.direction,
            "stage": student.stage,
            "group_name": student.group_name,
        },
        "diagnostic": (
            {
                "summary": diagnostic.analysis_summary,
                "analysis": diagnostic.analysis_json or {},
            }
            if diagnostic is not None
            else None
        ),
        "plan": _plan_context(db, student, chat_session),
        "chat_context": chat_session.context_json or {},
        "recent_messages": _chat_history_context(db, chat_session.id),
        "knowledge_base": _rag_chat_context(db, student, content),
        "student_message": content,
    }


def _persist_chat_exchange(
    db: Session,
    chat_session: AIMentorChatSession,
    *,
    user_content: str,
    assistant_content: str,
    model_name: str,
    token_count: int | None,
    metadata: dict[str, Any],
) -> AIMentorChatResponse:
    user_message = _add_chat_message_without_commit(
        db,
        chat_session,
        AIMentorChatMessageCreate(role="user", content=user_content),
    )
    assistant_message = _add_chat_message_without_commit(
        db,
        chat_session,
        AIMentorChatMessageCreate(
            role="assistant",
            content=assistant_content,
            model_name=model_name,
            token_count=token_count,
            metadata_json=metadata,
        ),
    )

    db.commit()
    db.refresh(chat_session)
    db.refresh(user_message)
    db.refresh(assistant_message)

    return AIMentorChatResponse(
        session=AIMentorChatSessionRead.model_validate(chat_session),
        user_message=user_message,
        assistant_message=assistant_message,
    )


def _sse_event(event: str, data: dict[str, Any]) -> str:
    """Frontend uchun bitta Server-Sent Event yozuvini yaratadi."""

    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _text_chunks(text: str, chunk_size: int = 48) -> Iterator[str]:
    """Mock/non-streaming provider javobini kichik delta'larga ajratadi."""

    if chunk_size < 1:
        chunk_size = 48
    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]


def _persist_stream_user_message(
    db: Session,
    chat_session: AIMentorChatSession,
    content: str,
) -> AIMentorChatMessage:
    """Streaming boshlanishidan oldin user xabarini bazaga mustahkam saqlaydi."""

    user_message = _add_chat_message_without_commit(
        db,
        chat_session,
        AIMentorChatMessageCreate(role="user", content=content),
    )
    db.commit()
    db.refresh(chat_session)
    db.refresh(user_message)
    return user_message


def _persist_stream_assistant_message(
    db: Session,
    chat_session: AIMentorChatSession,
    *,
    content: str,
    model_name: str,
    token_count: int | None,
    metadata: dict[str, Any],
) -> AIMentorChatMessage:
    assistant_message = _add_chat_message_without_commit(
        db,
        chat_session,
        AIMentorChatMessageCreate(
            role="assistant",
            content=content,
            model_name=model_name,
            token_count=token_count,
            metadata_json=metadata,
        ),
    )
    db.commit()
    db.refresh(chat_session)
    db.refresh(assistant_message)
    return assistant_message


def _consume_provider_stream(
    provider: Any,
    context: dict[str, Any],
) -> Generator[str, None, LLMCallMetadata]:
    """Provider streamingni yagona kontraktga keltiradi.

    Groq haqiqiy delta streaming beradi. Eski yoki boshqa providerda streaming
    metodi bo‘lmasa, uning to‘liq javobi kichik bo‘laklarga ajratiladi.
    """

    stream_method = getattr(provider, "chat_reply_stream", None)
    if callable(stream_method):
        stream = stream_method(context)
        while True:
            try:
                delta = next(stream)
            except StopIteration as stop:
                metadata = stop.value
                if isinstance(metadata, LLMCallMetadata):
                    return metadata
                return LLMCallMetadata(
                    provider=getattr(provider, "provider_name", "unknown"),
                    model=getattr(provider, "model_name", "unknown"),
                )
            if delta:
                yield str(delta)

    result = provider.chat_reply(context)
    for delta in _text_chunks(result.text):
        yield delta
    return result.metadata


def _usage_status_for_error(error_code: str) -> str:
    return "rate_limited" if error_code == "rate_limit" else "error"


def _provider_attempt_entry(
    provider: str,
    status_value: str,
    *,
    model: str | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {"provider": provider, "status": status_value}
    if model:
        data["model"] = model
    if error_code:
        data["error_code"] = error_code
    return data


def _fallback_unavailable_error() -> Exception:
    return http_error(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "AI Mentor fallback LLM xizmati sozlanmagan yoki vaqtincha mavjud emas.",
    )


def _get_chat_fallback_provider_or_raise() -> Any:
    try:
        provider = get_ai_mentor_chat_fallback_provider()
    except LLMProviderError as exc:
        raise _fallback_unavailable_error() from exc
    if provider is None:
        raise _fallback_unavailable_error()
    return provider


def _record_success_usage(
    db: Session,
    student: Student,
    chat_session: AIMentorChatSession,
    metadata: LLMCallMetadata,
    *,
    fallback_from_provider: str | None = None,
) -> None:
    record_llm_usage(
        db,
        student_id=student.id,
        chat_session_id=chat_session.id,
        feature="chat",
        provider=metadata.provider,
        model_name=metadata.model,
        status="success",
        input_tokens=metadata.input_tokens,
        output_tokens=metadata.output_tokens,
        total_tokens=metadata.total_tokens,
        request_id=metadata.request_id,
        fallback_from_provider=fallback_from_provider,
    )


def _record_error_usage(
    db: Session,
    student: Student,
    chat_session: AIMentorChatSession,
    *,
    provider: str,
    model_name: str | None,
    error_code: str,
    fallback_from_provider: str | None = None,
) -> None:
    record_llm_usage(
        db,
        student_id=student.id,
        chat_session_id=chat_session.id,
        feature="chat",
        provider=provider,
        model_name=model_name,
        status=_usage_status_for_error(error_code),
        error_code=error_code,
        fallback_from_provider=fallback_from_provider,
    )


def stream_chat_message(
    db: Session,
    student: Student,
    session_id: int,
    content: str,
) -> Iterator[str]:
    """Streamingni tayyorlaydi va HTTP status xatolarini headerlardan oldin beradi."""

    chat_session = get_student_chat_session_or_404(db, student, session_id)
    if chat_session.status != "active":
        raise http_error(
            status.HTTP_409_CONFLICT,
            "Yopilgan yoki arxivlangan chatga xabar yuborib bo‘lmaydi.",
        )

    # User xabari history'ga ikki marta tushmasligi uchun context avval olinadi.
    context = _llm_chat_context(db, student, chat_session, content)
    user_message = _persist_stream_user_message(db, chat_session, content)
    return _stream_chat_message_events(
        db,
        student,
        chat_session,
        content,
        context,
        user_message,
    )


def _stream_chat_message_events(
    db: Session,
    student: Student,
    chat_session: AIMentorChatSession,
    content: str,
    context: dict[str, Any],
    user_message: AIMentorChatMessage,
) -> Iterator[str]:
    """SSE chat: Groq primary, faqat quota/429 holatida secondary fallback."""

    primary_name = settings.AI_MENTOR_CHAT_PRIMARY_PROVIDER.strip().casefold()
    fallback_name = settings.AI_MENTOR_CHAT_FALLBACK_PROVIDER.strip().casefold()
    assistant_parts: list[str] = []
    assistant_persisted = False
    provider_attempts: list[dict[str, Any]] = []
    active_provider: Any | None = None
    fallback_reason: str | None = None

    def persist_partial(reason: str) -> None:
        nonlocal assistant_persisted
        partial = "".join(assistant_parts).strip()
        if not partial or assistant_persisted:
            return
        _persist_stream_assistant_message(
            db,
            chat_session,
            content=partial,
            model_name=(getattr(active_provider, "model_name", None) or "unknown"),
            token_count=None,
            metadata={
                "provider": getattr(active_provider, "provider_name", primary_name or "unknown"),
                "stream": True,
                "stream_interrupted": True,
                "interruption_reason": reason,
                "provider_attempts": provider_attempts,
                "rag": _rag_metadata_from_context(context, used_for_answer=True),
            },
        )
        assistant_persisted = True

    try:
        # Test/local mock remains possible only when explicitly configured as chat primary.
        if primary_name == "mock":
            yield _sse_event(
                "start",
                {
                    "session_id": chat_session.id,
                    "user_message": {
                        "id": user_message.id,
                        "session_id": user_message.session_id,
                        "sequence_number": user_message.sequence_number,
                        "role": user_message.role,
                        "content": user_message.content,
                        "created_at": user_message.created_at.isoformat(),
                    },
                    "provider": "mock",
                    "model": "mock-ai-mentor-v1",
                    "rag": _rag_metadata_from_context(context, used_for_answer=False),
                },
            )
            mock_text = _mock_chat_reply(db, student, content)
            for delta in _text_chunks(mock_text):
                assistant_parts.append(delta)
                yield _sse_event("delta", {"delta": delta})
            metadata = {
                "provider": "mock",
                "stream": True,
                "provider_attempts": [{"provider": "mock", "status": "success"}],
                "rag": _rag_metadata_from_context(context, used_for_answer=False),
            }
            model_name = "mock-ai-mentor-v1"
            token_count = None
        else:
            use_fallback_immediately = False
            if primary_name == "groq" and not student_has_groq_chat_quota(
                db,
                student.id,
                settings.AI_MENTOR_GROQ_CHAT_DAILY_LIMIT,
            ):
                use_fallback_immediately = True
                fallback_reason = "student_groq_quota_exhausted"
                provider_attempts.append(
                    _provider_attempt_entry("groq", "skipped_quota", error_code=fallback_reason)
                )
                record_llm_usage(
                    db,
                    student_id=student.id,
                    chat_session_id=chat_session.id,
                    feature="chat",
                    provider="groq",
                    model_name=settings.GROQ_MODEL,
                    status="skipped_quota",
                    error_code=fallback_reason,
                )
                active_provider = _get_chat_fallback_provider_or_raise()
            else:
                try:
                    active_provider = get_ai_mentor_chat_primary_provider()
                except LLMProviderError as exc:
                    _record_error_usage(
                        db,
                        student,
                        chat_session,
                        provider=primary_name or "unknown",
                        model_name=None,
                        error_code=exc.code,
                    )
                    raise
                if active_provider is None:
                    raise LLMProviderError("missing_provider", "AI Mentor chat primary provider sozlanmagan.")

            yield _sse_event(
                "start",
                {
                    "session_id": chat_session.id,
                    "user_message": {
                        "id": user_message.id,
                        "session_id": user_message.session_id,
                        "sequence_number": user_message.sequence_number,
                        "role": user_message.role,
                        "content": user_message.content,
                        "created_at": user_message.created_at.isoformat(),
                    },
                    "provider": getattr(active_provider, "provider_name", "unknown"),
                    "model": getattr(active_provider, "model_name", "unknown"),
                    "rag": _rag_metadata_from_context(context, used_for_answer=False),
                },
            )

            if use_fallback_immediately:
                yield _sse_event(
                    "fallback",
                    {
                        "from_provider": "groq",
                        "to_provider": getattr(active_provider, "provider_name", fallback_name),
                        "reason": fallback_reason,
                        "replace": False,
                    },
                )

            try:
                stream = _consume_provider_stream(active_provider, context)
                while True:
                    try:
                        delta = next(stream)
                    except StopIteration as stop:
                        llm_metadata = stop.value
                        break
                    assistant_parts.append(delta)
                    yield _sse_event("delta", {"delta": delta})

                if not assistant_parts:
                    raise LLMProviderError("empty_text_output", "AI Mentor streaming javobi bo'sh qaytdi.")
                _record_success_usage(
                    db,
                    student,
                    chat_session,
                    llm_metadata,
                    fallback_from_provider="groq" if use_fallback_immediately else None,
                )
                provider_attempts.append(
                    _provider_attempt_entry(
                        llm_metadata.provider,
                        "success",
                        model=llm_metadata.model,
                    )
                )
            except LLMProviderError as exc:
                failed_name = getattr(active_provider, "provider_name", primary_name or "unknown")
                failed_model = getattr(active_provider, "model_name", None)
                _record_error_usage(
                    db,
                    student,
                    chat_session,
                    provider=failed_name,
                    model_name=failed_model,
                    error_code=exc.code,
                    fallback_from_provider="groq" if use_fallback_immediately else None,
                )
                provider_attempts.append(
                    _provider_attempt_entry(failed_name, _usage_status_for_error(exc.code), model=failed_model, error_code=exc.code)
                )

                # Fallback providerning o'zi xato bersa yoki Groq 429 bo'lmasa boshqa fallback yo'q.
                if use_fallback_immediately or primary_name != "groq" or exc.code != "rate_limit":
                    persist_partial(exc.code)
                    yield _sse_event(
                        "error",
                        {
                            "code": exc.code,
                            "message": "AI Mentor LLM xizmati vaqtincha javob bera olmadi.",
                            "partial_saved": bool(assistant_parts),
                        },
                    )
                    return

                fallback_reason = "rate_limit"
                replace_existing = bool(assistant_parts)
                assistant_parts.clear()
                try:
                    active_provider = _get_chat_fallback_provider_or_raise()
                except Exception:
                    yield _sse_event(
                        "error",
                        {
                            "code": "fallback_unavailable",
                            "message": "AI Mentor fallback LLM xizmati mavjud emas.",
                            "partial_saved": False,
                        },
                    )
                    return

                yield _sse_event(
                    "fallback",
                    {
                        "from_provider": "groq",
                        "to_provider": getattr(active_provider, "provider_name", fallback_name),
                        "reason": fallback_reason,
                        "replace": replace_existing,
                    },
                )

                try:
                    fallback_stream = _consume_provider_stream(active_provider, context)
                    while True:
                        try:
                            delta = next(fallback_stream)
                        except StopIteration as stop:
                            llm_metadata = stop.value
                            break
                        assistant_parts.append(delta)
                        yield _sse_event("delta", {"delta": delta})
                    if not assistant_parts:
                        raise LLMProviderError("empty_text_output", "Fallback streaming javobi bo'sh qaytdi.")
                    _record_success_usage(db, student, chat_session, llm_metadata, fallback_from_provider="groq")
                    provider_attempts.append(
                        _provider_attempt_entry(llm_metadata.provider, "success", model=llm_metadata.model)
                    )
                except LLMProviderError as fallback_exc:
                    fallback_provider_name = getattr(active_provider, "provider_name", fallback_name or "unknown")
                    fallback_model = getattr(active_provider, "model_name", None)
                    _record_error_usage(
                        db,
                        student,
                        chat_session,
                        provider=fallback_provider_name,
                        model_name=fallback_model,
                        error_code=fallback_exc.code,
                        fallback_from_provider="groq",
                    )
                    provider_attempts.append(
                        _provider_attempt_entry(
                            fallback_provider_name,
                            _usage_status_for_error(fallback_exc.code),
                            model=fallback_model,
                            error_code=fallback_exc.code,
                        )
                    )
                    yield _sse_event(
                        "error",
                        {
                            "code": fallback_exc.code,
                            "message": "AI Mentor fallback LLM xizmati vaqtincha javob bera olmadi.",
                            "partial_saved": False,
                        },
                    )
                    return

            metadata = llm_metadata.as_dict()
            metadata["stream"] = True
            metadata["rag"] = _rag_metadata_from_context(context, used_for_answer=True)
            metadata["provider_attempts"] = provider_attempts
            if fallback_reason:
                metadata["fallback_from_provider"] = "groq"
                metadata["fallback_reason"] = fallback_reason
            model_name = llm_metadata.model
            token_count = llm_metadata.total_tokens

        assistant_content = "".join(assistant_parts).strip()
        if not assistant_content:
            yield _sse_event(
                "error",
                {"code": "empty_text_output", "message": "AI Mentor bo'sh javob qaytardi.", "partial_saved": False},
            )
            return

        assistant_message = _persist_stream_assistant_message(
            db,
            chat_session,
            content=assistant_content,
            model_name=model_name,
            token_count=token_count,
            metadata=metadata,
        )
        assistant_persisted = True
        yield _sse_event(
            "done",
            {
                "session_id": chat_session.id,
                "assistant_message": {
                    "id": assistant_message.id,
                    "session_id": assistant_message.session_id,
                    "sequence_number": assistant_message.sequence_number,
                    "role": assistant_message.role,
                    "content": assistant_message.content,
                    "model_name": assistant_message.model_name,
                    "token_count": assistant_message.token_count,
                    "metadata_json": assistant_message.metadata_json,
                    "created_at": assistant_message.created_at.isoformat(),
                },
            },
        )
    except GeneratorExit:
        persist_partial("client_disconnected")
        raise
    except LLMProviderError as exc:
        logger.warning("Chat stream provider initialization failed: %s", exc.code)
        yield _sse_event(
            "error",
            {
                "code": exc.code,
                "message": "AI Mentor LLM xizmati vaqtincha javob bera olmadi.",
                "partial_saved": False,
            },
        )
    except Exception as exc:
        logger.warning("Chat stream fallback initialization failed: %s: %s", type(exc).__name__, exc)
        yield _sse_event(
            "error",
            {
                "code": "fallback_unavailable",
                "message": "AI Mentor fallback LLM xizmati mavjud emas.",
                "partial_saved": False,
            },
        )


def send_mock_chat_message(
    db: Session,
    student: Student,
    session_id: int,
    content: str,
) -> AIMentorChatResponse:
    chat_session = get_student_chat_session_or_404(db, student, session_id)

    if chat_session.status != "active":
        raise http_error(
            status.HTTP_409_CONFLICT,
            "Yopilgan yoki arxivlangan chatga xabar yuborib bo‘lmaydi.",
        )

    return _persist_chat_exchange(
        db,
        chat_session,
        user_content=content,
        assistant_content=_mock_chat_reply(db, student, content),
        model_name="mock-ai-mentor-v1",
        token_count=None,
        metadata={"provider": "mock"},
    )


def send_chat_message(
    db: Session,
    student: Student,
    session_id: int,
    content: str,
) -> AIMentorChatResponse:
    """Groq primary; per-student quota yoki Groq 429 bo'lsa secondary fallback."""

    chat_session = get_student_chat_session_or_404(db, student, session_id)
    if chat_session.status != "active":
        raise http_error(
            status.HTTP_409_CONFLICT,
            "Yopilgan yoki arxivlangan chatga xabar yuborib bo'lmaydi.",
        )

    primary_name = settings.AI_MENTOR_CHAT_PRIMARY_PROVIDER.strip().casefold()
    if primary_name == "mock":
        return send_mock_chat_message(db, student, session_id, content)

    context = _llm_chat_context(db, student, chat_session, content)
    provider_attempts: list[dict[str, Any]] = []
    fallback_reason: str | None = None
    fallback_from: str | None = None

    use_fallback = False
    if primary_name == "groq" and not student_has_groq_chat_quota(
        db,
        student.id,
        settings.AI_MENTOR_GROQ_CHAT_DAILY_LIMIT,
    ):
        use_fallback = True
        fallback_reason = "student_groq_quota_exhausted"
        fallback_from = "groq"
        provider_attempts.append(_provider_attempt_entry("groq", "skipped_quota", error_code=fallback_reason))
        record_llm_usage(
            db,
            student_id=student.id,
            chat_session_id=chat_session.id,
            feature="chat",
            provider="groq",
            model_name=settings.GROQ_MODEL,
            status="skipped_quota",
            error_code=fallback_reason,
        )
        provider = _get_chat_fallback_provider_or_raise()
    else:
        try:
            provider = get_ai_mentor_chat_primary_provider()
        except LLMProviderError as exc:
            _record_error_usage(
                db,
                student,
                chat_session,
                provider=primary_name or "unknown",
                model_name=None,
                error_code=exc.code,
            )
            raise http_error(status.HTTP_503_SERVICE_UNAVAILABLE, "AI Mentor LLM xizmati vaqtincha javob bera olmadi.") from exc
        if provider is None:
            raise http_error(status.HTTP_503_SERVICE_UNAVAILABLE, "AI Mentor chat primary provider sozlanmagan.")

    try:
        result = provider.chat_reply(context)
    except LLMProviderError as exc:
        failed_name = getattr(provider, "provider_name", primary_name or "unknown")
        failed_model = getattr(provider, "model_name", None)
        _record_error_usage(
            db,
            student,
            chat_session,
            provider=failed_name,
            model_name=failed_model,
            error_code=exc.code,
            fallback_from_provider=fallback_from,
        )
        provider_attempts.append(
            _provider_attempt_entry(failed_name, _usage_status_for_error(exc.code), model=failed_model, error_code=exc.code)
        )

        # Quota sabab fallbackga kirgan bo'lsak, secondary xatosidan keyin boshqa provider yo'q.
        if use_fallback or primary_name != "groq" or exc.code != "rate_limit":
            raise http_error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "AI Mentor LLM xizmati vaqtincha javob bera olmadi.",
            ) from exc

        fallback_reason = "rate_limit"
        fallback_from = "groq"
        try:
            provider = _get_chat_fallback_provider_or_raise()
            result = provider.chat_reply(context)
        except LLMProviderError as fallback_exc:
            fallback_provider_name = getattr(provider, "provider_name", settings.AI_MENTOR_CHAT_FALLBACK_PROVIDER)
            fallback_model = getattr(provider, "model_name", None)
            _record_error_usage(
                db,
                student,
                chat_session,
                provider=fallback_provider_name,
                model_name=fallback_model,
                error_code=fallback_exc.code,
                fallback_from_provider="groq",
            )
            provider_attempts.append(
                _provider_attempt_entry(
                    fallback_provider_name,
                    _usage_status_for_error(fallback_exc.code),
                    model=fallback_model,
                    error_code=fallback_exc.code,
                )
            )
            raise http_error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "AI Mentor fallback LLM xizmati vaqtincha javob bera olmadi.",
            ) from fallback_exc

    _record_success_usage(db, student, chat_session, result.metadata, fallback_from_provider=fallback_from)
    provider_attempts.append(
        _provider_attempt_entry(result.metadata.provider, "success", model=result.metadata.model)
    )
    metadata = result.metadata.as_dict()
    metadata["rag"] = _rag_metadata_from_context(context, used_for_answer=True)
    metadata["provider_attempts"] = provider_attempts
    if fallback_reason:
        metadata["fallback_from_provider"] = "groq"
        metadata["fallback_reason"] = fallback_reason

    return _persist_chat_exchange(
        db,
        chat_session,
        user_content=content,
        assistant_content=result.text,
        model_name=result.metadata.model,
        token_count=result.metadata.total_tokens,
        metadata=metadata,
    )


def get_student_chat_sessions(
    db: Session,
    student: Student,
) -> list[AIMentorChatSession]:
    """Talabaning chat sessiyalarini so‘nggi faollik bo‘yicha qaytaradi."""

    return (
        db.query(AIMentorChatSession)
        .filter(AIMentorChatSession.student_id == student.id)
        .order_by(
            func.coalesce(
                AIMentorChatSession.last_message_at,
                AIMentorChatSession.created_at,
            ).desc()
        )
        .all()
    )


def update_chat_session(
    db: Session,
    student: Student,
    session_id: int,
    payload: AIMentorChatSessionUpdate,
) -> AIMentorChatSession:
    """Chat nomi, holati yoki kontekstini yangilaydi."""

    chat_session = get_student_chat_session_or_404(
        db,
        student,
        session_id,
    )

    for field_name, value in payload.model_dump(exclude_unset=True).items():
        setattr(chat_session, field_name, value)

    db.commit()
    db.refresh(chat_session)
    return chat_session
