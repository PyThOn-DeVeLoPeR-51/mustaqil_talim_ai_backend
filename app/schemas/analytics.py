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


class AnalyticsDescriptiveStats(BaseModel):
    count: int = 0
    mean: float | None = None
    median: float | None = None
    standard_deviation: float | None = None
    minimum: float | None = None
    maximum: float | None = None


class AnalyticsPrePostStatistics(BaseModel):
    paired_count: int = 0
    pretest: AnalyticsDescriptiveStats = Field(default_factory=AnalyticsDescriptiveStats)
    posttest: AnalyticsDescriptiveStats = Field(default_factory=AnalyticsDescriptiveStats)
    difference: AnalyticsDescriptiveStats = Field(default_factory=AnalyticsDescriptiveStats)
    mean_difference: float | None = None
    percent_growth: float | None = None
    cohen_dz: float | None = None
    confidence_interval_95_low: float | None = None
    confidence_interval_95_high: float | None = None
    t_value: float | None = None
    degrees_of_freedom: int | None = None
    p_value: float | None = None
    p_value_note: str = "p-value juftlangan t-test uchun taxminiy hisoblanadi. Yakuniy maqolada statistik paket bilan qayta tekshirish tavsiya etiladi."
    interpretation: str | None = None


class AnalyticsExperimentGroupStats(BaseModel):
    group: ExperimentGroup
    label: str
    student_count: int = 0
    paired_count: int = 0
    pretest: AnalyticsDescriptiveStats = Field(default_factory=AnalyticsDescriptiveStats)
    posttest: AnalyticsDescriptiveStats = Field(default_factory=AnalyticsDescriptiveStats)
    mean_growth: float | None = None
    percent_growth: float | None = None


class AnalyticsBetweenGroupStatistics(BaseModel):
    experimental_count: int = 0
    control_count: int = 0
    experimental_growth: float | None = None
    control_growth: float | None = None
    growth_difference: float | None = None
    cohen_d: float | None = None
    t_value: float | None = None
    degrees_of_freedom: float | None = None
    p_value: float | None = None
    p_value_note: str = "p-value Welch t-test uchun taxminiy hisoblanadi. Yakuniy maqolada statistik paket bilan qayta tekshirish tavsiya etiladi."
    interpretation: str | None = None


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


class AnalyticsRubricProfile(BaseModel):
    mode: TaskMode
    label: str
    labels: list[str] = Field(default_factory=list)
    values: list[float | None] = Field(default_factory=list)


class AnalyticsHeatmapRow(BaseModel):
    student_id: int
    name: str
    group_name: str | None = None
    values: list[float | None] = Field(default_factory=list)


class AnalyticsExportRow(BaseModel):
    student_id: int
    name: str
    group_name: str | None = None
    experiment_group: ExperimentGroup | None = None
    pretest: float | None = None
    week_1: float | None = None
    week_2: float | None = None
    week_3: float | None = None
    week_4: float | None = None
    posttest: float | None = None
    average: float | None = None
    growth: float | None = None
    success: bool | None = None


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

    descriptive_statistics: AnalyticsDescriptiveStats = Field(
        default_factory=AnalyticsDescriptiveStats
    )
    pre_post_statistics: AnalyticsPrePostStatistics = Field(
        default_factory=AnalyticsPrePostStatistics
    )
    group_statistics: list[AnalyticsExperimentGroupStats] = Field(
        default_factory=list
    )
    between_group_statistics: AnalyticsBetweenGroupStatistics = Field(
        default_factory=AnalyticsBetweenGroupStatistics
    )
    rubric_profiles: list[AnalyticsRubricProfile] = Field(default_factory=list)
    export_rows: list[AnalyticsExportRow] = Field(default_factory=list)

    filters: AnalyticsFilterOptions
