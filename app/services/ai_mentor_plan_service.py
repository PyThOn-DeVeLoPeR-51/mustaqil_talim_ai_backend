"""AI Mentorning 4 haftalik reja va progress biznes mantiqi."""

import logging
from datetime import date, timedelta
from typing import Any

from fastapi import status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.llm.contracts import LLMProviderError, StructuredLLMResult, PlanGenerationOutput
from app.llm.factory import get_ai_mentor_provider
from app.models.ai_mentor import (
    AIMentorDiagnosticSession,
    AIMentorPlan,
    AIMentorPlanItem,
    AIMentorPlanWeek,
)
from app.models.student import Student
from app.schemas.ai_mentor import (
    AIMentorPlanCreate,
    AIMentorPlanDetail,
    AIMentorPlanDetailResponse,
    AIMentorPlanItemCreate,
    AIMentorPlanItemProgressUpdate,
    AIMentorPlanItemRead,
    AIMentorPlanProgress,
    AIMentorPlanRead,
    AIMentorPlanWeekCreate,
    AIMentorPlanWeekRead,
)
from app.services.ai_mentor_common import (
    http_error,
    next_student_version,
    utcnow,
)
from app.services.ai_mentor_diagnostic_service import (
    get_latest_completed_diagnostic_session,
    get_student_diagnostic_session_or_404,
)


logger = logging.getLogger(__name__)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _build_mock_plan_payload(
    session: AIMentorDiagnosticSession,
    start_date_value: date,
) -> AIMentorPlanCreate:
    analysis = session.analysis_json or {}
    focus_topic = str(analysis.get("focus_topic") or "tanlangan fan mavzulari")
    level_label = str(analysis.get("current_level_label") or "moslashtirilgan")
    goal_label = str(
        analysis.get("primary_goal_label") or "mustaqil ta’lim samaradorligini oshirish"
    )
    session_minutes = min(
        90,
        max(15, _safe_int(analysis.get("session_minutes"), 30)),
    )

    difficult_areas = set(analysis.get("difficult_areas") or [])
    learning_formats = set(analysis.get("learning_formats") or [])

    if "video" in learning_formats:
        resource_hint = "Qisqa video tushuntirish yoki vizual namuna bilan ishlang."
    elif "reading" in learning_formats:
        resource_hint = "Asosiy qo‘llanma va qisqa konspekt bilan ishlang."
    else:
        resource_hint = "Mavzu bo‘yicha mos o‘quv manbasini tanlang."

    if "time_management" in difficult_areas:
        planning_hint = "Mashg‘ulotni kalendarga oldindan kiriting va taymerdan foydalaning."
    else:
        planning_hint = "Mashg‘ulot boshlanishida aniq kichik natijani belgilang."

    weeks = [
        AIMentorPlanWeekCreate(
            week_number=1,
            title="Boshlang‘ich tayyorgarlik va yo‘nalishni aniqlash",
            goal=f"{focus_topic} bo‘yicha tayanch bilimlarni tartibga solish.",
            description=f"{level_label} darajaga mos boshlang‘ich hafta.",
            expected_outcome="Asosiy tushunchalar ro‘yxati va shaxsiy o‘quv jadvali tayyor bo‘ladi.",
            items=[
                AIMentorPlanItemCreate(
                    item_order=1,
                    day_number=1,
                    title="Maqsad va jadvalni aniqlashtirish",
                    description=planning_hint,
                    activity_type="planning",
                    estimated_minutes=session_minutes,
                ),
                AIMentorPlanItemCreate(
                    item_order=2,
                    day_number=3,
                    title="Tayanch tushunchalarni o‘rganish",
                    description=f"{focus_topic} bo‘yicha asosiy atamalarni yozib chiqing. {resource_hint}",
                    activity_type="theory",
                    estimated_minutes=session_minutes,
                ),
                AIMentorPlanItemCreate(
                    item_order=3,
                    day_number=6,
                    title="Boshlang‘ich o‘zini tekshirish",
                    description="O‘rganilgan tushunchalarni savollar yoki kichik test orqali tekshiring.",
                    activity_type="self_assessment",
                    estimated_minutes=session_minutes,
                ),
            ],
        ),
        AIMentorPlanWeekCreate(
            week_number=2,
            title="Tushunchalarni mustahkamlash",
            goal=f"{focus_topic} bo‘yicha nazariya va misollar o‘rtasidagi bog‘lanishni tushunish.",
            description="Qisqa nazariy bloklar amaliy mashqlar bilan birlashtiriladi.",
            expected_outcome="Talaba asosiy qoidalarni misollarda qo‘llay oladi.",
            items=[
                AIMentorPlanItemCreate(
                    item_order=1,
                    day_number=1,
                    title="Nazariy blokni takrorlash",
                    description=resource_hint,
                    activity_type="theory",
                    estimated_minutes=session_minutes,
                ),
                AIMentorPlanItemCreate(
                    item_order=2,
                    day_number=3,
                    title="Yo‘naltirilgan amaliy mashq",
                    description="Namuna asosida kamida bitta amaliy topshiriqni bosqichma-bosqich bajaring.",
                    activity_type="practice",
                    estimated_minutes=session_minutes,
                ),
                AIMentorPlanItemCreate(
                    item_order=3,
                    day_number=6,
                    title="Xatolar tahlili",
                    description="Mashqdagi xatolarni aniqlang va ularning sababini qisqa yozing.",
                    activity_type="reflection",
                    estimated_minutes=session_minutes,
                ),
            ],
        ),
        AIMentorPlanWeekCreate(
            week_number=3,
            title="Mustaqil qo‘llash",
            goal=f"{focus_topic} bo‘yicha vazifalarni kamroq yordam bilan bajarish.",
            description="Murakkablik asta-sekin oshiriladi va mustaqil qaror qabul qilish rag‘batlantiriladi.",
            expected_outcome="Talaba o‘xshash topshiriqlarni mustaqil bajarish strategiyasiga ega bo‘ladi.",
            items=[
                AIMentorPlanItemCreate(
                    item_order=1,
                    day_number=1,
                    title="Mustaqil amaliy topshiriq",
                    description="Yangi misol yoki topshiriqni tayyor yechimsiz bajarishga harakat qiling.",
                    activity_type="practice",
                    estimated_minutes=session_minutes,
                ),
                AIMentorPlanItemCreate(
                    item_order=2,
                    day_number=4,
                    title="Natijani mezonlar bo‘yicha tekshirish",
                    description="Natijani topshiriq talabi va baholash mezonlari bilan solishtiring.",
                    activity_type="self_assessment",
                    estimated_minutes=session_minutes,
                ),
                AIMentorPlanItemCreate(
                    item_order=3,
                    day_number=6,
                    title="Qiyin nuqtalarni qayta ishlash",
                    description="Eng ko‘p xato qilingan bitta jihat bo‘yicha qo‘shimcha mashq bajaring.",
                    activity_type="remediation",
                    estimated_minutes=session_minutes,
                ),
            ],
        ),
        AIMentorPlanWeekCreate(
            week_number=4,
            title="Yakuniy mustahkamlash va refleksiya",
            goal=f"{goal_label} natijasini baholash va keyingi o‘quv qadamlarini belgilash.",
            description="To‘rt haftalik faoliyat umumlashtiriladi.",
            expected_outcome="Talaba erishilgan natijalar, qolgan qiyinchiliklar va keyingi maqsadlarini aniq belgilaydi.",
            items=[
                AIMentorPlanItemCreate(
                    item_order=1,
                    day_number=1,
                    title="Umumlashtiruvchi topshiriq",
                    description=f"{focus_topic} bo‘yicha asosiy bilim va ko‘nikmalarni birlashtiruvchi vazifani bajaring.",
                    activity_type="capstone",
                    estimated_minutes=session_minutes,
                ),
                AIMentorPlanItemCreate(
                    item_order=2,
                    day_number=4,
                    title="Yakuniy o‘zini baholash",
                    description="Boshlang‘ich holat bilan hozirgi natijani solishtirib, kuchli va zaif jihatlarni yozing.",
                    activity_type="self_assessment",
                    estimated_minutes=session_minutes,
                ),
                AIMentorPlanItemCreate(
                    item_order=3,
                    day_number=7,
                    title="Keyingi 4 hafta uchun yo‘nalish",
                    description="Keyingi davr uchun bitta aniq maqsad va uchta amaliy qadam belgilang.",
                    activity_type="planning",
                    estimated_minutes=session_minutes,
                ),
            ],
        ),
    ]

    return AIMentorPlanCreate(
        diagnostic_session_id=session.id,
        title="4 haftalik shaxsiy mustaqil ta’lim rejasi",
        summary=session.analysis_summary,
        status="active",
        generation_source="mock",
        generation_metadata={
            "provider": "mock",
            "strategy_version": "1.0",
            "diagnostic_session_version": session.version,
        },
        start_date=start_date_value,
        end_date=start_date_value + timedelta(days=27),
        weeks=weeks,
    )


def create_plan_from_payload(
    db: Session,
    student: Student,
    payload: AIMentorPlanCreate,
) -> AIMentorPlan:
    if payload.diagnostic_session_id is not None:
        diagnostic_session = get_student_diagnostic_session_or_404(
            db,
            student,
            payload.diagnostic_session_id,
        )
        if diagnostic_session.status != "completed":
            raise http_error(
                status.HTTP_409_CONFLICT,
                "Reja yaratish uchun diagnostika yakunlangan bo‘lishi kerak.",
            )

    if payload.status == "active":
        (
            db.query(AIMentorPlan)
            .filter(
                AIMentorPlan.student_id == student.id,
                AIMentorPlan.status == "active",
            )
            .update({AIMentorPlan.status: "archived"}, synchronize_session=False)
        )

    plan = AIMentorPlan(
        student_id=student.id,
        diagnostic_session_id=payload.diagnostic_session_id,
        version=next_student_version(db, AIMentorPlan, student.id),
        title=payload.title,
        summary=payload.summary,
        status=payload.status,
        generation_source=payload.generation_source,
        generation_metadata=payload.generation_metadata,
        start_date=payload.start_date,
        end_date=payload.end_date,
    )
    db.add(plan)
    db.flush()

    for week_payload in sorted(payload.weeks, key=lambda item: item.week_number):
        week = AIMentorPlanWeek(
            plan_id=plan.id,
            week_number=week_payload.week_number,
            title=week_payload.title,
            goal=week_payload.goal,
            description=week_payload.description,
            expected_outcome=week_payload.expected_outcome,
        )
        db.add(week)
        db.flush()

        for item_payload in sorted(
            week_payload.items,
            key=lambda item: item.item_order,
        ):
            db.add(
                AIMentorPlanItem(
                    plan_week_id=week.id,
                    item_order=item_payload.item_order,
                    day_number=item_payload.day_number,
                    title=item_payload.title,
                    description=item_payload.description,
                    activity_type=item_payload.activity_type,
                    estimated_minutes=item_payload.estimated_minutes,
                    resources_json=item_payload.resources_json,
                    status="pending",
                )
            )

    db.commit()
    db.refresh(plan)
    return plan


def _resolve_completed_diagnostic_session(
    db: Session,
    student: Student,
    diagnostic_session_id: int | None,
) -> AIMentorDiagnosticSession:
    if diagnostic_session_id is None:
        session = get_latest_completed_diagnostic_session(db, student)
        if session is None:
            raise http_error(
                status.HTTP_409_CONFLICT,
                "Avval diagnostikani yakunlash kerak.",
            )
        return session

    session = get_student_diagnostic_session_or_404(
        db,
        student,
        diagnostic_session_id,
    )
    if session.status != "completed":
        raise http_error(
            status.HTTP_409_CONFLICT,
            "Diagnostika hali yakunlanmagan.",
        )
    return session


def create_mock_plan(
    db: Session,
    student: Student,
    diagnostic_session_id: int | None = None,
    start_date_value: date | None = None,
) -> AIMentorPlanDetailResponse:
    session = _resolve_completed_diagnostic_session(
        db,
        student,
        diagnostic_session_id,
    )
    payload = _build_mock_plan_payload(
        session,
        start_date_value or date.today(),
    )
    plan = create_plan_from_payload(db, student, payload)
    return build_plan_detail_response(db, plan)


def _llm_plan_context(
    student: Student,
    session: AIMentorDiagnosticSession,
) -> dict[str, Any]:
    return {
        "student_profile": {
            "university": student.university,
            "direction": student.direction,
            "stage": student.stage,
            "group_name": student.group_name,
        },
        "diagnostic_summary": session.analysis_summary,
        "diagnostic_analysis": session.analysis_json or {},
        "constraints": {
            "weeks": 4,
            "items_per_week": 3,
            "language": "uzbek",
            "progression": "oddiydan murakkabga",
        },
    }


def _plan_payload_from_llm(
    result: StructuredLLMResult[PlanGenerationOutput],
    session: AIMentorDiagnosticSession,
    start_date_value: date,
) -> AIMentorPlanCreate:
    weeks = [
        AIMentorPlanWeekCreate(
            week_number=week.week_number,
            title=week.title,
            goal=week.goal,
            description=week.description,
            expected_outcome=week.expected_outcome,
            items=[
                AIMentorPlanItemCreate(
                    item_order=item.item_order,
                    day_number=item.day_number,
                    title=item.title,
                    description=item.description,
                    activity_type=item.activity_type,
                    estimated_minutes=item.estimated_minutes,
                    resources_json=item.resources or None,
                )
                for item in week.items
            ],
        )
        for week in result.output.weeks
    ]

    return AIMentorPlanCreate(
        diagnostic_session_id=session.id,
        title=result.output.title,
        summary=result.output.summary,
        status="active",
        generation_source="llm",
        generation_metadata={
            **result.metadata.as_dict(),
            "diagnostic_session_version": session.version,
            "strategy_version": "llm-1.0",
        },
        start_date=start_date_value,
        end_date=start_date_value + timedelta(days=27),
        weeks=weeks,
    )


def create_generated_plan(
    db: Session,
    student: Student,
    diagnostic_session_id: int | None = None,
    start_date_value: date | None = None,
) -> AIMentorPlanDetailResponse:
    """Sozlangan LLM provider orqali reja yaratadi, zarur bo‘lsa mock'ka qaytadi."""

    if settings.LLM_PROVIDER.strip().casefold() == "mock":
        return create_mock_plan(
            db,
            student,
            diagnostic_session_id=diagnostic_session_id,
            start_date_value=start_date_value,
        )

    session = _resolve_completed_diagnostic_session(
        db,
        student,
        diagnostic_session_id,
    )
    plan_start_date = start_date_value or date.today()

    try:
        provider = get_ai_mentor_provider()
        if provider is None:
            return create_mock_plan(
                db,
                student,
                diagnostic_session_id=session.id,
                start_date_value=plan_start_date,
            )
        result = provider.generate_plan(_llm_plan_context(student, session))
        payload = _plan_payload_from_llm(result, session, plan_start_date)
    except LLMProviderError as exc:
        logger.warning("Plan LLM fallback: %s", exc.code)
        if not settings.LLM_FALLBACK_TO_MOCK:
            raise http_error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "AI Mentor LLM xizmati vaqtincha reja yarata olmadi.",
            ) from exc

        payload = _build_mock_plan_payload(session, plan_start_date)
        payload.generation_metadata = {
            **(payload.generation_metadata or {}),
            "fallback_from_provider": settings.LLM_PROVIDER,
            "fallback_reason": exc.code,
        }

    plan = create_plan_from_payload(db, student, payload)
    return build_plan_detail_response(db, plan)


def get_student_plan_or_404(
    db: Session,
    student: Student,
    plan_id: int,
) -> AIMentorPlan:
    plan = (
        db.query(AIMentorPlan)
        .filter(
            AIMentorPlan.id == plan_id,
            AIMentorPlan.student_id == student.id,
        )
        .first()
    )

    if plan is None:
        raise http_error(status.HTTP_404_NOT_FOUND, "AI Mentor rejasi topilmadi.")

    return plan


def get_active_or_latest_plan(
    db: Session,
    student: Student,
) -> AIMentorPlan | None:
    active_plan = (
        db.query(AIMentorPlan)
        .filter(
            AIMentorPlan.student_id == student.id,
            AIMentorPlan.status == "active",
        )
        .order_by(AIMentorPlan.version.desc())
        .first()
    )
    if active_plan is not None:
        return active_plan

    return (
        db.query(AIMentorPlan)
        .filter(AIMentorPlan.student_id == student.id)
        .order_by(AIMentorPlan.version.desc())
        .first()
    )


def _plan_week_rows(
    db: Session,
    plan_id: int,
) -> list[AIMentorPlanWeek]:
    return (
        db.query(AIMentorPlanWeek)
        .filter(AIMentorPlanWeek.plan_id == plan_id)
        .order_by(AIMentorPlanWeek.week_number.asc())
        .all()
    )


def _week_item_rows(
    db: Session,
    week_id: int,
) -> list[AIMentorPlanItem]:
    return (
        db.query(AIMentorPlanItem)
        .filter(AIMentorPlanItem.plan_week_id == week_id)
        .order_by(AIMentorPlanItem.item_order.asc())
        .all()
    )


def calculate_plan_progress(
    db: Session,
    plan_id: int,
) -> AIMentorPlanProgress:
    rows = (
        db.query(AIMentorPlanItem.status, func.count(AIMentorPlanItem.id))
        .join(
            AIMentorPlanWeek,
            AIMentorPlanWeek.id == AIMentorPlanItem.plan_week_id,
        )
        .filter(AIMentorPlanWeek.plan_id == plan_id)
        .group_by(AIMentorPlanItem.status)
        .all()
    )
    counts = {item_status: count for item_status, count in rows}
    total_items = sum(counts.values())
    completed_items = counts.get("completed", 0)

    progress_percent = (
        round(completed_items / total_items * 100, 2) if total_items else 0.0
    )

    return AIMentorPlanProgress(
        total_items=total_items,
        completed_items=completed_items,
        in_progress_items=counts.get("in_progress", 0),
        skipped_items=counts.get("skipped", 0),
        progress_percent=progress_percent,
    )


def build_plan_detail_response(
    db: Session,
    plan: AIMentorPlan,
) -> AIMentorPlanDetailResponse:
    weeks: list[AIMentorPlanWeekRead] = []

    for week in _plan_week_rows(db, plan.id):
        items = [
            AIMentorPlanItemRead.model_validate(item)
            for item in _week_item_rows(db, week.id)
        ]
        weeks.append(
            AIMentorPlanWeekRead(
                id=week.id,
                plan_id=week.plan_id,
                week_number=week.week_number,
                title=week.title,
                goal=week.goal,
                description=week.description,
                expected_outcome=week.expected_outcome,
                created_at=week.created_at,
                updated_at=week.updated_at,
                items=items,
            )
        )

    plan_data = AIMentorPlanRead.model_validate(plan).model_dump()
    detail = AIMentorPlanDetail(**plan_data, weeks=weeks)

    return AIMentorPlanDetailResponse(
        plan=detail,
        progress=calculate_plan_progress(db, plan.id),
    )


def update_plan_item_progress(
    db: Session,
    student: Student,
    item_id: int,
    payload: AIMentorPlanItemProgressUpdate,
) -> AIMentorPlanItem:
    item = (
        db.query(AIMentorPlanItem)
        .join(
            AIMentorPlanWeek,
            AIMentorPlanWeek.id == AIMentorPlanItem.plan_week_id,
        )
        .join(AIMentorPlan, AIMentorPlan.id == AIMentorPlanWeek.plan_id)
        .filter(
            AIMentorPlanItem.id == item_id,
            AIMentorPlan.student_id == student.id,
        )
        .first()
    )

    if item is None:
        raise http_error(status.HTTP_404_NOT_FOUND, "Reja vazifasi topilmadi.")

    item.status = payload.status
    item.completed_at = utcnow() if payload.status == "completed" else None
    db.commit()
    db.refresh(item)
    return item


def get_student_plans(
    db: Session,
    student: Student,
) -> list[AIMentorPlan]:
    """Talabaning reja versiyalarini eng yangisidan boshlab qaytaradi."""

    return (
        db.query(AIMentorPlan)
        .filter(AIMentorPlan.student_id == student.id)
        .order_by(AIMentorPlan.version.desc())
        .all()
    )
