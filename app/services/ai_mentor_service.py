"""AI Mentor servislarining yagona import nuqtasi.

Routerlar barqaror va sodda importlardan foydalanishi uchun diagnostika, reja va
chat servislarining ommaviy funksiyalari shu modul orqali qayta eksport qilinadi.
"""

from app.services.ai_mentor_chat_service import (
    build_chat_session_detail,
    create_chat_session,
    get_student_chat_session_or_404,
    send_mock_chat_message,
)
from app.services.ai_mentor_diagnostic_service import (
    build_diagnostic_session_detail,
    get_active_diagnostic_questions,
    get_latest_completed_diagnostic_session,
    get_student_diagnostic_session_or_404,
    seed_diagnostic_questions,
    start_diagnostic_session,
    submit_diagnostic_answers,
)
from app.services.ai_mentor_plan_service import (
    build_plan_detail_response,
    calculate_plan_progress,
    create_mock_plan,
    create_plan_from_payload,
    get_active_or_latest_plan,
    get_student_plan_or_404,
    update_plan_item_progress,
)

__all__ = [
    "build_chat_session_detail",
    "build_diagnostic_session_detail",
    "build_plan_detail_response",
    "calculate_plan_progress",
    "create_chat_session",
    "create_mock_plan",
    "create_plan_from_payload",
    "get_active_diagnostic_questions",
    "get_active_or_latest_plan",
    "get_latest_completed_diagnostic_session",
    "get_student_chat_session_or_404",
    "get_student_diagnostic_session_or_404",
    "get_student_plan_or_404",
    "seed_diagnostic_questions",
    "send_mock_chat_message",
    "start_diagnostic_session",
    "submit_diagnostic_answers",
    "update_plan_item_progress",
]
