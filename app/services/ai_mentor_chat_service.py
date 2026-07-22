"""AI Mentor chat sessiyalari, xabarlar tarixi va mock javoblari."""

from fastapi import status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.ai_mentor import AIMentorChatMessage, AIMentorChatSession
from app.models.student import Student
from app.schemas.ai_mentor import (
    AIMentorChatMessageCreate,
    AIMentorChatResponse,
    AIMentorChatSessionCreate,
    AIMentorChatSessionDetail,
    AIMentorChatSessionRead,
)
from app.services.ai_mentor_common import http_error, utcnow
from app.services.ai_mentor_plan_service import (
    calculate_plan_progress,
    get_active_or_latest_plan,
    get_student_plan_or_404,
)


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

    user_message = _add_chat_message_without_commit(
        db,
        chat_session,
        AIMentorChatMessageCreate(role="user", content=content),
    )
    assistant_message = _add_chat_message_without_commit(
        db,
        chat_session,
        AIMentorChatMessageCreate(
            role="assistant",
            content=_mock_chat_reply(db, student, content),
            model_name="mock-ai-mentor-v1",
            metadata_json={"provider": "mock"},
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
