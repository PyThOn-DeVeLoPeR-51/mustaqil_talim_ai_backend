"""AI Mentor diagnostik savollari, sessiyalari va mock tahlili."""

from typing import Any

from fastapi import status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.data.ai_mentor_diagnostic_questions import (
    AI_MENTOR_DIAGNOSTIC_QUESTIONS,
)
from app.models.ai_mentor import (
    AIMentorDiagnosticAnswer,
    AIMentorDiagnosticQuestion,
    AIMentorDiagnosticSession,
)
from app.models.student import Student
from app.schemas.ai_mentor import (
    AIMentorDiagnosticAnswerCreate,
    AIMentorDiagnosticAnswerDetail,
    AIMentorDiagnosticAnswersSubmit,
    AIMentorDiagnosticQuestionRead,
    AIMentorDiagnosticSessionDetail,
)
from app.services.ai_mentor_common import (
    http_error,
    next_student_version,
    utcnow,
)


def _option_items(options_json: Any) -> list[Any]:
    if isinstance(options_json, list):
        return options_json

    if isinstance(options_json, dict):
        nested_options = options_json.get("options")
        if isinstance(nested_options, list):
            return nested_options

    return []


def _option_value(item: Any) -> Any:
    if not isinstance(item, dict):
        return item

    for key in ("value", "id", "code", "label"):
        if key in item:
            return item[key]

    return None


def _option_label_map(options_json: Any) -> dict[str, str]:
    result: dict[str, str] = {}

    for item in _option_items(options_json):
        value = _option_value(item)
        if value is None:
            continue

        label = item.get("label", value) if isinstance(item, dict) else value
        result[str(value)] = str(label)

    return result


def _allowed_option_values(options_json: Any) -> set[str]:
    return set(_option_label_map(options_json))


def _answer_raw_value(answer: AIMentorDiagnosticAnswerCreate) -> Any:
    if answer.answer_json is not None:
        return answer.answer_json
    return answer.answer_text


def _normalize_diagnostic_answer(
    question: AIMentorDiagnosticQuestion,
    answer: AIMentorDiagnosticAnswerCreate,
) -> tuple[str | None, Any | None]:
    """Javob turini savol konfiguratsiyasiga mos tekshiradi va normallashtiradi."""

    raw_value = _answer_raw_value(answer)

    if question.answer_type == "single_choice":
        if isinstance(raw_value, (list, dict)) or raw_value is None:
            raise http_error(
                status.HTTP_400_BAD_REQUEST,
                f"“{question.question_text}” savoliga bitta variant yuborilishi kerak.",
            )

        allowed_values = _allowed_option_values(question.options_json)
        if allowed_values and str(raw_value) not in allowed_values:
            raise http_error(
                status.HTTP_400_BAD_REQUEST,
                f"“{question.question_text}” savoliga ruxsat etilmagan variant yuborildi.",
            )

        return str(raw_value), raw_value

    if question.answer_type == "multiple_choice":
        if not isinstance(raw_value, list) or not raw_value:
            raise http_error(
                status.HTTP_400_BAD_REQUEST,
                f"“{question.question_text}” savoliga kamida bitta variant tanlang.",
            )

        normalized_values = list(dict.fromkeys(str(value) for value in raw_value))
        allowed_values = _allowed_option_values(question.options_json)
        invalid_values = set(normalized_values) - allowed_values

        if allowed_values and invalid_values:
            raise http_error(
                status.HTTP_400_BAD_REQUEST,
                f"“{question.question_text}” savolida noto‘g‘ri variantlar bor: "
                f"{sorted(invalid_values)}.",
            )

        return ", ".join(normalized_values), normalized_values

    if question.answer_type == "number":
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            raise http_error(
                status.HTTP_400_BAD_REQUEST,
                f"“{question.question_text}” savoliga son yuborilishi kerak.",
            ) from None

        configuration = (
            question.options_json if isinstance(question.options_json, dict) else {}
        )
        minimum = configuration.get("min")
        maximum = configuration.get("max")

        if minimum is not None and numeric_value < float(minimum):
            raise http_error(
                status.HTTP_400_BAD_REQUEST,
                f"“{question.question_text}” javobi {minimum} dan kichik bo‘lishi mumkin emas.",
            )

        if maximum is not None and numeric_value > float(maximum):
            raise http_error(
                status.HTTP_400_BAD_REQUEST,
                f"“{question.question_text}” javobi {maximum} dan katta bo‘lishi mumkin emas.",
            )

        normalized_number: int | float
        if numeric_value.is_integer():
            normalized_number = int(numeric_value)
        else:
            normalized_number = numeric_value

        return str(normalized_number), normalized_number

    if question.answer_type == "boolean":
        if isinstance(raw_value, bool):
            boolean_value = raw_value
        elif isinstance(raw_value, str) and raw_value.strip().lower() in {
            "true",
            "1",
            "yes",
            "ha",
        }:
            boolean_value = True
        elif isinstance(raw_value, str) and raw_value.strip().lower() in {
            "false",
            "0",
            "no",
            "yo‘q",
            "yo'q",
        }:
            boolean_value = False
        else:
            raise http_error(
                status.HTTP_400_BAD_REQUEST,
                f"“{question.question_text}” savoliga ha yoki yo‘q qiymati yuborilishi kerak.",
            )

        return "true" if boolean_value else "false", boolean_value

    if isinstance(raw_value, (dict, list)) or raw_value is None:
        raise http_error(
            status.HTTP_400_BAD_REQUEST,
            f"“{question.question_text}” savoliga matnli javob yuborilishi kerak.",
        )

    text_value = str(raw_value).strip()
    if not text_value:
        raise http_error(
            status.HTTP_400_BAD_REQUEST,
            f"“{question.question_text}” savoliga bo‘sh javob yuborib bo‘lmaydi.",
        )

    return text_value, None


# ---------------------------------------------------------------------------
# Diagnostik savollar seed'i
# ---------------------------------------------------------------------------


def seed_diagnostic_questions(db: Session) -> dict[str, int]:
    """Boshlang‘ich savollarni idempotent tarzda yaratadi yoki yangilaydi."""

    created = 0
    updated = 0
    unchanged = 0

    for payload in AI_MENTOR_DIAGNOSTIC_QUESTIONS:
        question = (
            db.query(AIMentorDiagnosticQuestion)
            .filter(
                AIMentorDiagnosticQuestion.question_code
                == payload["question_code"],
                AIMentorDiagnosticQuestion.version == payload["version"],
            )
            .first()
        )

        if question is None:
            db.add(AIMentorDiagnosticQuestion(**payload))
            created += 1
            continue

        changed = False
        for field_name, value in payload.items():
            if getattr(question, field_name) != value:
                setattr(question, field_name, value)
                changed = True

        if changed:
            updated += 1
        else:
            unchanged += 1

    db.commit()

    return {
        "created": created,
        "updated": updated,
        "unchanged": unchanged,
        "total": len(AI_MENTOR_DIAGNOSTIC_QUESTIONS),
    }


def get_active_diagnostic_questions(
    db: Session,
) -> list[AIMentorDiagnosticQuestion]:
    """Har bir ``question_code`` uchun eng so‘nggi faol versiyani qaytaradi."""

    latest_versions = (
        db.query(
            AIMentorDiagnosticQuestion.question_code.label("question_code"),
            func.max(AIMentorDiagnosticQuestion.version).label("version"),
        )
        .filter(AIMentorDiagnosticQuestion.is_active.is_(True))
        .group_by(AIMentorDiagnosticQuestion.question_code)
        .subquery()
    )

    return (
        db.query(AIMentorDiagnosticQuestion)
        .join(
            latest_versions,
            (
                AIMentorDiagnosticQuestion.question_code
                == latest_versions.c.question_code
            )
            & (AIMentorDiagnosticQuestion.version == latest_versions.c.version),
        )
        .filter(AIMentorDiagnosticQuestion.is_active.is_(True))
        .order_by(
            AIMentorDiagnosticQuestion.sort_order.asc(),
            AIMentorDiagnosticQuestion.id.asc(),
        )
        .all()
    )


# ---------------------------------------------------------------------------
# Diagnostika sessiyasi va javoblar
# ---------------------------------------------------------------------------


def start_diagnostic_session(
    db: Session,
    student: Student,
) -> AIMentorDiagnosticSession:
    """Yakunlanmagan sessiya bo‘lsa qaytaradi, aks holda yangisini yaratadi."""

    existing_session = (
        db.query(AIMentorDiagnosticSession)
        .filter(
            AIMentorDiagnosticSession.student_id == student.id,
            AIMentorDiagnosticSession.status == "in_progress",
        )
        .order_by(AIMentorDiagnosticSession.version.desc())
        .first()
    )

    if existing_session is not None:
        return existing_session

    if not get_active_diagnostic_questions(db):
        raise http_error(
            status.HTTP_409_CONFLICT,
            "Diagnostik savollar hali bazaga kiritilmagan.",
        )

    session = AIMentorDiagnosticSession(
        student_id=student.id,
        version=next_student_version(db, AIMentorDiagnosticSession, student.id),
        status="in_progress",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_student_diagnostic_session_or_404(
    db: Session,
    student: Student,
    session_id: int,
) -> AIMentorDiagnosticSession:
    session = (
        db.query(AIMentorDiagnosticSession)
        .filter(
            AIMentorDiagnosticSession.id == session_id,
            AIMentorDiagnosticSession.student_id == student.id,
        )
        .first()
    )

    if session is None:
        raise http_error(
            status.HTTP_404_NOT_FOUND,
            "Diagnostika sessiyasi topilmadi.",
        )

    return session


def _session_answer_rows(
    db: Session,
    session_id: int,
) -> list[tuple[AIMentorDiagnosticAnswer, AIMentorDiagnosticQuestion]]:
    return (
        db.query(AIMentorDiagnosticAnswer, AIMentorDiagnosticQuestion)
        .join(
            AIMentorDiagnosticQuestion,
            AIMentorDiagnosticQuestion.id == AIMentorDiagnosticAnswer.question_id,
        )
        .filter(AIMentorDiagnosticAnswer.session_id == session_id)
        .order_by(
            AIMentorDiagnosticQuestion.sort_order.asc(),
            AIMentorDiagnosticAnswer.id.asc(),
        )
        .all()
    )


def build_diagnostic_session_detail(
    db: Session,
    session: AIMentorDiagnosticSession,
) -> AIMentorDiagnosticSessionDetail:
    answer_details: list[AIMentorDiagnosticAnswerDetail] = []

    for answer, question in _session_answer_rows(db, session.id):
        answer_details.append(
            AIMentorDiagnosticAnswerDetail(
                id=answer.id,
                session_id=answer.session_id,
                question_id=answer.question_id,
                answer_text=answer.answer_text,
                answer_json=answer.answer_json,
                created_at=answer.created_at,
                updated_at=answer.updated_at,
                question=AIMentorDiagnosticQuestionRead.model_validate(question),
            )
        )

    session_data = {
        "id": session.id,
        "student_id": session.student_id,
        "version": session.version,
        "status": session.status,
        "analysis_summary": session.analysis_summary,
        "analysis_json": session.analysis_json,
        "started_at": session.started_at,
        "completed_at": session.completed_at,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "answers": answer_details,
    }
    return AIMentorDiagnosticSessionDetail(**session_data)


def _answer_value_for_analysis(answer: AIMentorDiagnosticAnswer) -> Any:
    if answer.answer_json is not None:
        return answer.answer_json
    return answer.answer_text


def _diagnostic_value_map(
    db: Session,
    session_id: int,
) -> tuple[dict[str, Any], dict[str, AIMentorDiagnosticQuestion]]:
    values: dict[str, Any] = {}
    questions: dict[str, AIMentorDiagnosticQuestion] = {}

    for answer, question in _session_answer_rows(db, session_id):
        values[question.question_code] = _answer_value_for_analysis(answer)
        questions[question.question_code] = question

    return values, questions


def _display_answer(
    question: AIMentorDiagnosticQuestion | None,
    value: Any,
) -> Any:
    if question is None:
        return value

    labels = _option_label_map(question.options_json)
    if isinstance(value, list):
        return [labels.get(str(item), str(item)) for item in value]
    return labels.get(str(value), value)


def _build_mock_diagnostic_analysis(
    values: dict[str, Any],
    questions: dict[str, AIMentorDiagnosticQuestion],
) -> tuple[str, dict[str, Any]]:
    weekly_hours = int(float(values.get("weekly_hours", 4) or 4))
    motivation = int(float(values.get("motivation_level", 5) or 5))
    session_minutes = int(float(values.get("session_minutes", 30) or 30))

    difficult_areas = values.get("difficult_areas") or []
    learning_formats = values.get("learning_formats") or []
    study_days = values.get("study_days") or []

    if weekly_hours <= 2 or motivation <= 3:
        risk_level = "high"
    elif weekly_hours <= 4 or motivation <= 6:
        risk_level = "medium"
    else:
        risk_level = "low"

    goal_label = _display_answer(
        questions.get("primary_goal"), values.get("primary_goal")
    )
    level_label = _display_answer(
        questions.get("current_level"), values.get("current_level")
    )
    focus_topic = str(values.get("focus_topic") or "tanlangan fan mavzulari")

    summary = (
        f"Talabaning asosiy maqsadi — {goal_label or 'mustaqil ta’limni yaxshilash'}. "
        f"Hozirgi tayyorgarlik darajasi: {level_label or 'aniqlanmagan'}. "
        f"Haftasiga taxminan {weekly_hours} soat ajratishi mumkin; asosiy yo‘nalish — "
        f"{focus_topic}. Reja {session_minutes} daqiqalik kichik mashg‘ulotlar, "
        "muntazam amaliyot va haftalik o‘zini baholash asosida tuziladi."
    )

    analysis = {
        "provider": "mock",
        "analysis_version": "1.0",
        "primary_goal": values.get("primary_goal"),
        "primary_goal_label": goal_label,
        "current_level": values.get("current_level"),
        "current_level_label": level_label,
        "focus_topic": focus_topic,
        "weekly_hours": weekly_hours,
        "session_minutes": session_minutes,
        "motivation_level": motivation,
        "difficult_areas": difficult_areas,
        "difficult_area_labels": _display_answer(
            questions.get("difficult_areas"), difficult_areas
        ),
        "learning_formats": learning_formats,
        "learning_format_labels": _display_answer(
            questions.get("learning_formats"), learning_formats
        ),
        "study_days": study_days,
        "study_day_labels": _display_answer(
            questions.get("study_days"), study_days
        ),
        "mentor_expectation": values.get("mentor_expectation"),
        "risk_level": risk_level,
    }
    return summary, analysis


def submit_diagnostic_answers(
    db: Session,
    student: Student,
    session_id: int,
    payload: AIMentorDiagnosticAnswersSubmit,
) -> AIMentorDiagnosticSessionDetail:
    session = get_student_diagnostic_session_or_404(db, student, session_id)

    if session.status != "in_progress":
        raise http_error(
            status.HTTP_409_CONFLICT,
            "Faqat davom etayotgan diagnostika sessiyasiga javob yuborish mumkin.",
        )

    active_questions = get_active_diagnostic_questions(db)
    questions_by_id = {question.id: question for question in active_questions}
    submitted_question_ids = {answer.question_id for answer in payload.answers}

    unknown_ids = submitted_question_ids - set(questions_by_id)
    if unknown_ids:
        raise http_error(
            status.HTTP_400_BAD_REQUEST,
            f"Faol bo‘lmagan yoki mavjud bo‘lmagan savollar yuborildi: {sorted(unknown_ids)}.",
        )

    required_ids = {
        question.id for question in active_questions if question.is_required
    }
    missing_required_ids = required_ids - submitted_question_ids
    if missing_required_ids:
        missing_questions = [
            questions_by_id[question_id].question_text
            for question_id in sorted(missing_required_ids)
        ]
        raise http_error(
            status.HTTP_400_BAD_REQUEST,
            "Majburiy savollarga javob yetishmaydi: " + "; ".join(missing_questions),
        )

    existing_answers = {
        answer.question_id: answer
        for answer in db.query(AIMentorDiagnosticAnswer)
        .filter(AIMentorDiagnosticAnswer.session_id == session.id)
        .all()
    }

    for answer_payload in payload.answers:
        question = questions_by_id[answer_payload.question_id]
        answer_text, answer_json = _normalize_diagnostic_answer(
            question,
            answer_payload,
        )

        answer = existing_answers.get(question.id)
        if answer is None:
            answer = AIMentorDiagnosticAnswer(
                session_id=session.id,
                question_id=question.id,
            )
            db.add(answer)

        answer.answer_text = answer_text
        answer.answer_json = answer_json

    db.flush()

    values, questions = _diagnostic_value_map(db, session.id)
    analysis_summary, analysis_json = _build_mock_diagnostic_analysis(
        values,
        questions,
    )

    session.status = "completed"
    session.analysis_summary = analysis_summary
    session.analysis_json = analysis_json
    session.completed_at = utcnow()

    db.commit()
    db.refresh(session)
    return build_diagnostic_session_detail(db, session)


def get_latest_completed_diagnostic_session(
    db: Session,
    student: Student,
) -> AIMentorDiagnosticSession | None:
    return (
        db.query(AIMentorDiagnosticSession)
        .filter(
            AIMentorDiagnosticSession.student_id == student.id,
            AIMentorDiagnosticSession.status == "completed",
        )
        .order_by(AIMentorDiagnosticSession.version.desc())
        .first()
    )
