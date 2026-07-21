from typing import Literal

from pydantic import BaseModel, Field


ExperimentGroup = Literal["experimental", "control"]
TaskMode = Literal["etalon", "optional"]
AssessmentStage = Literal["pretest", "intermediate", "posttest"]


class AnalyticsSummary(BaseModel):
    student_count: int = 0
    evaluated_student_count: int = 0
    evaluated_submission_count: int = 0

    initial_average: float | None = None
    final_average: float | None = None
    growth: float | None = None

    second_attempt_growth: float | None = None
    success_rate: float | None = None


class AnalyticsProgress(BaseModel):
    labels: list[str] = Field(
        default_factory=lambda: [
            "Boshlang‘ich",
            "1-hafta",
            "2-hafta",
            "3-hafta",
            "4-hafta",
            "Yakuniy",
        ]
    )
    values: list[float | None] = Field(default_factory=list)


class AnalyticsGroupComparisonItem(BaseModel):
    label: str
    before: float | None = None
    after: float | None = None
    count: int = 0


class AnalyticsDistributionItem(BaseModel):
    key: Literal["high", "good", "satisfactory", "low"]
    label: str
    value: int = 0


class AnalyticsModeComparisonItem(BaseModel):
    mode: TaskMode
    label: str
    average: float | None = None
    evaluated_student_count: int = 0
    evaluated_result_count: int = 0


class AnalyticsCriteria(BaseModel):
    labels: list[str] = Field(
        default_factory=lambda: [
            "Proyeksiya",
            "O‘lcham",
            "Chiziqlar",
            "Aniqlik",
            "Standart",
        ]
    )
    values: list[float | None] = Field(default_factory=list)


class AnalyticsHeatmapRow(BaseModel):
    student_id: int
    name: str
    group_name: str | None = None
    values: list[float | None] = Field(default_factory=list)


class AnalyticsStudentOption(BaseModel):
    id: int
    full_name: str
    group_name: str | None = None
    experiment_group: ExperimentGroup | None = None


class AnalyticsFilterOptions(BaseModel):
    groups: list[str] = Field(default_factory=list)
    students: list[AnalyticsStudentOption] = Field(default_factory=list)

    experiment_groups: list[ExperimentGroup] = Field(default_factory=list)
    modes: list[TaskMode] = Field(default_factory=list)

    topics: list[str] = Field(default_factory=list)
    week_numbers: list[int] = Field(default_factory=list)
    assessment_stages: list[AssessmentStage] = Field(default_factory=list)
    academic_periods: list[str] = Field(default_factory=list)


class TeacherAnalyticsRead(BaseModel):
    summary: AnalyticsSummary
    progress: AnalyticsProgress

    group_comparison: list[AnalyticsGroupComparisonItem] = Field(
        default_factory=list
    )

    distribution: list[AnalyticsDistributionItem] = Field(
        default_factory=list
    )

    mode_comparison: list[AnalyticsModeComparisonItem] = Field(
        default_factory=list
    )

    criteria: AnalyticsCriteria

    heatmap: list[AnalyticsHeatmapRow] = Field(
        default_factory=list
    )

    filters: AnalyticsFilterOptions