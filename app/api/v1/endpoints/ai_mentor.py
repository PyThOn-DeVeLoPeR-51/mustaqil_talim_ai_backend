"""Talaba uchun AI Mentor diagnostika, reja va chat API endpointlari."""

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.student import Student
from app.schemas.ai_mentor import (
    AIMentorChatRequest,
    AIMentorChatResponse,
    AIMentorChatSessionCreate,
    AIMentorChatSessionDetail,
    AIMentorChatSessionRead,
    AIMentorChatSessionUpdate,
    AIMentorDiagnosticAnswersSubmit,
    AIMentorDiagnosticQuestionRead,
    AIMentorDiagnosticSessionCreate,
    AIMentorDiagnosticSessionDetail,
    AIMentorDiagnosticSessionRead,
    AIMentorLLMStatus,
    AIMentorPlanDetailResponse,
    AIMentorPlanGenerateCreate,
    AIMentorPlanItemProgressUpdate,
    AIMentorPlanItemRead,
    AIMentorPlanRead,
)
from app.services.ai_mentor_service import (
    build_chat_session_detail,
    build_diagnostic_session_detail,
    build_plan_detail_response,
    create_chat_session,
    create_generated_plan,
    get_active_diagnostic_questions,
    get_active_or_latest_plan,
    get_latest_completed_diagnostic_session,
    get_llm_provider_status,
    get_student_chat_session_or_404,
    get_student_chat_sessions,
    get_student_diagnostic_session_or_404,
    get_student_diagnostic_sessions,
    get_student_plan_or_404,
    get_student_plans,
    send_chat_message as send_chat_message_service,
    start_diagnostic_session,
    stream_chat_message,
    submit_diagnostic_answers,
    update_chat_session,
    update_plan_item_progress,
)
from app.services.auth_service import get_current_ai_mentor_student


router = APIRouter()


@router.get(
    "/llm/status",
    response_model=AIMentorLLMStatus,
)
def get_llm_status(
    current_student: Student = Depends(get_current_ai_mentor_student),
):
    del current_student
    return get_llm_provider_status()


# ---------------------------------------------------------------------------
# Diagnostika
# ---------------------------------------------------------------------------


@router.get(
    "/diagnostic/questions",
    response_model=list[AIMentorDiagnosticQuestionRead],
)
def get_diagnostic_questions(
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_ai_mentor_student),
):
    del current_student  # Endpoint faqat student tokeni bilan ochilishini ta'minlaydi.
    return get_active_diagnostic_questions(db)


@router.post(
    "/diagnostic/sessions",
    response_model=AIMentorDiagnosticSessionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_diagnostic_session(
    _payload: AIMentorDiagnosticSessionCreate | None = None,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_ai_mentor_student),
):
    return start_diagnostic_session(db, current_student)


@router.get(
    "/diagnostic/sessions",
    response_model=list[AIMentorDiagnosticSessionRead],
)
def get_diagnostic_sessions(
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_ai_mentor_student),
):
    return get_student_diagnostic_sessions(db, current_student)


@router.get(
    "/diagnostic/sessions/latest",
    response_model=AIMentorDiagnosticSessionDetail | None,
)
def get_latest_diagnostic_session(
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_ai_mentor_student),
):
    diagnostic_session = get_latest_completed_diagnostic_session(
        db,
        current_student,
    )
    if diagnostic_session is None:
        return None
    return build_diagnostic_session_detail(db, diagnostic_session)


@router.get(
    "/diagnostic/sessions/{session_id}",
    response_model=AIMentorDiagnosticSessionDetail,
)
def get_diagnostic_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_ai_mentor_student),
):
    diagnostic_session = get_student_diagnostic_session_or_404(
        db,
        current_student,
        session_id,
    )
    return build_diagnostic_session_detail(db, diagnostic_session)


@router.post(
    "/diagnostic/sessions/{session_id}/answers",
    response_model=AIMentorDiagnosticSessionDetail,
)
def submit_answers(
    session_id: int,
    payload: AIMentorDiagnosticAnswersSubmit,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_ai_mentor_student),
):
    return submit_diagnostic_answers(
        db,
        current_student,
        session_id,
        payload,
    )


# ---------------------------------------------------------------------------
# 4 haftalik reja
# ---------------------------------------------------------------------------


@router.post(
    "/plans/generate",
    response_model=AIMentorPlanDetailResponse,
    status_code=status.HTTP_201_CREATED,
)
def generate_plan(
    payload: AIMentorPlanGenerateCreate,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_ai_mentor_student),
):
    return create_generated_plan(
        db,
        current_student,
        diagnostic_session_id=payload.diagnostic_session_id,
        start_date_value=payload.start_date,
    )


@router.get(
    "/plans",
    response_model=list[AIMentorPlanRead],
)
def get_plans(
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_ai_mentor_student),
):
    return get_student_plans(db, current_student)


@router.get(
    "/plans/current",
    response_model=AIMentorPlanDetailResponse | None,
)
def get_current_plan(
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_ai_mentor_student),
):
    plan = get_active_or_latest_plan(db, current_student)
    if plan is None:
        return None
    return build_plan_detail_response(db, plan)


@router.get(
    "/plans/{plan_id}",
    response_model=AIMentorPlanDetailResponse,
)
def get_plan(
    plan_id: int,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_ai_mentor_student),
):
    plan = get_student_plan_or_404(db, current_student, plan_id)
    return build_plan_detail_response(db, plan)


@router.patch(
    "/plan-items/{item_id}/progress",
    response_model=AIMentorPlanItemRead,
)
def update_item_progress(
    item_id: int,
    payload: AIMentorPlanItemProgressUpdate,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_ai_mentor_student),
):
    return update_plan_item_progress(
        db,
        current_student,
        item_id,
        payload,
    )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


@router.post(
    "/chat/sessions",
    response_model=AIMentorChatSessionRead,
    status_code=status.HTTP_201_CREATED,
)
def start_chat_session(
    payload: AIMentorChatSessionCreate,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_ai_mentor_student),
):
    return create_chat_session(db, current_student, payload)


@router.get(
    "/chat/sessions",
    response_model=list[AIMentorChatSessionRead],
)
def get_chat_sessions(
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_ai_mentor_student),
):
    return get_student_chat_sessions(db, current_student)


@router.get(
    "/chat/sessions/{session_id}",
    response_model=AIMentorChatSessionDetail,
)
def get_chat_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_ai_mentor_student),
):
    chat_session = get_student_chat_session_or_404(
        db,
        current_student,
        session_id,
    )
    return build_chat_session_detail(db, chat_session)


@router.patch(
    "/chat/sessions/{session_id}",
    response_model=AIMentorChatSessionRead,
)
def patch_chat_session(
    session_id: int,
    payload: AIMentorChatSessionUpdate,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_ai_mentor_student),
):
    return update_chat_session(
        db,
        current_student,
        session_id,
        payload,
    )


@router.post(
    "/chat/sessions/{session_id}/messages",
    response_model=AIMentorChatResponse,
    status_code=status.HTTP_201_CREATED,
)
def send_chat_message_endpoint(
    session_id: int,
    payload: AIMentorChatRequest,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_ai_mentor_student),
):
    return send_chat_message_service(
        db,
        current_student,
        session_id,
        payload.content,
    )


@router.post(
    "/chat/sessions/{session_id}/messages/stream",
    status_code=status.HTTP_200_OK,
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {
                "text/event-stream": {
                    "example": (
                        "event: delta\n"
                        'data: {"delta":"Salom"}\n\n'
                    )
                }
            },
            "description": "AI Mentor javobini Server-Sent Events orqali uzatadi.",
        }
    },
)
def stream_chat_message_endpoint(
    session_id: int,
    payload: AIMentorChatRequest,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_ai_mentor_student),
):
    event_stream = stream_chat_message(
        db,
        current_student,
        session_id,
        payload.content,
    )
    return StreamingResponse(
        event_stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
