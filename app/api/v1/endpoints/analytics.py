from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.teacher import Teacher
from app.schemas.analytics import TeacherAnalyticsRead
from app.services.analytics_service import get_teacher_analytics
from app.services.auth_service import get_current_teacher


router = APIRouter()


ExperimentGroupFilter = Literal["experimental", "control"]
TaskModeFilter = Literal["etalon", "optional", "ixtiyoriy"]
AssessmentStageFilter = Literal[
    "pretest",
    "intermediate",
    "posttest",
]


@router.get(
    "/teacher",
    response_model=TeacherAnalyticsRead,
)
def teacher_analytics(
    group_name: str | None = Query(
        default=None,
        max_length=100,
    ),
    student_id: int | None = Query(
        default=None,
        ge=1,
    ),
    experiment_group: ExperimentGroupFilter | None = Query(
        default=None,
    ),
    mode: TaskModeFilter | None = Query(
        default=None,
    ),
    topic: str | None = Query(
        default=None,
        max_length=255,
    ),
    week_number: int | None = Query(
        default=None,
        ge=1,
        le=52,
    ),
    assessment_stage: AssessmentStageFilter | None = Query(
        default=None,
    ),
    academic_period: str | None = Query(
        default=None,
        max_length=100,
    ),
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return get_teacher_analytics(
        db=db,
        teacher=current_teacher,
        group_name=group_name,
        student_id=student_id,
        experiment_group=experiment_group,
        mode=mode,
        topic=topic,
        week_number=week_number,
        assessment_stage=assessment_stage,
        academic_period=academic_period,
    )