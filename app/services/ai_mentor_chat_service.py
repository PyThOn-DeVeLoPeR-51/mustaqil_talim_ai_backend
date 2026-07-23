"""AI Mentor chat sessiyalari, xabarlar tarixi va LLM/mock javoblari."""

import logging
from typing import Any

from fastapi import status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.llm.contracts import LLMProviderError
from app.llm.factory import get_ai_mentor_provider
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


logger = logging.getLogger(__name__)


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
    """Sozlangan provider orqali chat javobi yaratadi, zarur bo‘lsa mock'ka qaytadi."""

    chat_session = get_student_chat_session_or_404(db, student, session_id)
    if chat_session.status != "active":
        raise http_error(
            status.HTTP_409_CONFLICT,
            "Yopilgan yoki arxivlangan chatga xabar yuborib bo‘lmaydi.",
        )

    if settings.LLM_PROVIDER.strip().casefold() == "mock":
        return send_mock_chat_message(db, student, session_id, content)

    try:
        provider = get_ai_mentor_provider()
        if provider is None:
            return send_mock_chat_message(db, student, session_id, content)
        result = provider.chat_reply(
            _llm_chat_context(db, student, chat_session, content)
        )
        assistant_content = result.text
        model_name = result.metadata.model
        token_count = result.metadata.total_tokens
        metadata = result.metadata.as_dict()
    except LLMProviderError as exc:
        logger.warning("Chat LLM fallback: %s", exc.code)
        if not settings.LLM_FALLBACK_TO_MOCK:
            raise http_error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "AI Mentor LLM xizmati vaqtincha javob bera olmadi.",
            ) from exc
        assistant_content = _mock_chat_reply(db, student, content)
        model_name = "mock-ai-mentor-v1"
        token_count = None
        metadata = {
            "provider": "mock",
            "fallback_from_provider": settings.LLM_PROVIDER,
            "fallback_reason": exc.code,
        }

    return _persist_chat_exchange(
        db,
        chat_session,
        user_content=content,
        assistant_content=assistant_content,
        model_name=model_name,
        token_count=token_count,
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
