from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DiagnosticAnswerType = Literal[
    "single_choice",
    "multiple_choice",
    "short_text",
    "long_text",
    "number",
    "boolean",
]
DiagnosticSessionStatus = Literal["in_progress", "completed", "cancelled"]
PlanStatus = Literal["draft", "active", "completed", "archived"]
PlanGenerationSource = Literal["mock", "llm", "manual"]
PlanItemStatus = Literal["pending", "in_progress", "completed", "skipped"]
ChatSessionStatus = Literal["active", "closed", "archived"]
ChatMessageRole = Literal["system", "user", "assistant"]

JsonObject = dict[str, Any]
JsonCollection = list[Any] | JsonObject


class AIMentorSchema(BaseModel):
    """AI Mentor ORM response schema'lari uchun umumiy konfiguratsiya."""

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Diagnostik savollar
# ---------------------------------------------------------------------------


class AIMentorDiagnosticQuestionBase(BaseModel):
    question_code: str = Field(min_length=1, max_length=100)
    version: int = Field(default=1, ge=1)
    question_text: str = Field(min_length=1)
    help_text: str | None = None
    category: str | None = Field(default=None, max_length=100)
    answer_type: DiagnosticAnswerType
    options_json: JsonCollection | None = None
    sort_order: int = Field(default=0, ge=0)
    is_required: bool = True
    is_active: bool = True


class AIMentorDiagnosticQuestionCreate(AIMentorDiagnosticQuestionBase):
    @model_validator(mode="after")
    def validate_choice_options(self) -> "AIMentorDiagnosticQuestionCreate":
        if self.answer_type in {"single_choice", "multiple_choice"}:
            if self.options_json is None or len(self.options_json) == 0:
                raise ValueError(
                    "Tanlovli diagnostik savol uchun options_json bo‘sh bo‘lmasligi kerak."
                )
        return self


class AIMentorDiagnosticQuestionUpdate(BaseModel):
    question_text: str | None = Field(default=None, min_length=1)
    help_text: str | None = None
    category: str | None = Field(default=None, max_length=100)
    answer_type: DiagnosticAnswerType | None = None
    options_json: JsonCollection | None = None
    sort_order: int | None = Field(default=None, ge=0)
    is_required: bool | None = None
    is_active: bool | None = None


class AIMentorDiagnosticQuestionRead(AIMentorSchema):
    id: int
    question_code: str
    version: int
    question_text: str
    help_text: str | None = None
    category: str | None = None
    answer_type: DiagnosticAnswerType
    options_json: JsonCollection | None = None
    sort_order: int
    is_required: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Diagnostika sessiyasi va javoblar
# ---------------------------------------------------------------------------


class AIMentorDiagnosticSessionCreate(BaseModel):
    """Talaba uchun yangi diagnostika sessiyasini boshlash payload'i."""

    # Hozircha bo‘sh. Keyinchalik diagnostika shabloni yoki til qo‘shilishi mumkin.
    pass


class AIMentorDiagnosticAnswerCreate(BaseModel):
    question_id: int = Field(gt=0)
    answer_text: str | None = Field(default=None, min_length=1)
    answer_json: Any | None = None

    @model_validator(mode="after")
    def validate_answer_value(self) -> "AIMentorDiagnosticAnswerCreate":
        if isinstance(self.answer_text, str):
            self.answer_text = self.answer_text.strip() or None

        if self.answer_text is None and self.answer_json is None:
            raise ValueError("Diagnostik javob qiymati bo‘sh bo‘lmasligi kerak.")
        return self


class AIMentorDiagnosticAnswersSubmit(BaseModel):
    answers: list[AIMentorDiagnosticAnswerCreate] = Field(
        min_length=1,
        max_length=100,
    )

    @model_validator(mode="after")
    def validate_unique_questions(self) -> "AIMentorDiagnosticAnswersSubmit":
        question_ids = [answer.question_id for answer in self.answers]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("Bitta savol uchun payload ichida takroriy javob yuborilgan.")
        return self


class AIMentorDiagnosticAnswerRead(AIMentorSchema):
    id: int
    session_id: int
    question_id: int
    answer_text: str | None = None
    answer_json: Any | None = None
    created_at: datetime
    updated_at: datetime


class AIMentorDiagnosticAnswerDetail(AIMentorDiagnosticAnswerRead):
    question: AIMentorDiagnosticQuestionRead | None = None


class AIMentorDiagnosticSessionUpdate(BaseModel):
    status: DiagnosticSessionStatus | None = None
    analysis_summary: str | None = None
    analysis_json: JsonObject | None = None


class AIMentorDiagnosticSessionRead(AIMentorSchema):
    id: int
    student_id: int
    version: int
    status: DiagnosticSessionStatus
    analysis_summary: str | None = None
    analysis_json: JsonObject | None = None
    started_at: datetime
    completed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class AIMentorDiagnosticSessionDetail(AIMentorDiagnosticSessionRead):
    answers: list[AIMentorDiagnosticAnswerDetail] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 4 haftalik individual reja
# ---------------------------------------------------------------------------


class AIMentorPlanItemBase(BaseModel):
    item_order: int = Field(ge=1)
    day_number: int | None = Field(default=None, ge=1, le=7)
    title: str = Field(min_length=1, max_length=255)
    description: str | None = None
    activity_type: str | None = Field(default=None, max_length=50)
    estimated_minutes: int | None = Field(default=None, gt=0)
    resources_json: JsonCollection | None = None


class AIMentorPlanItemCreate(AIMentorPlanItemBase):
    pass


class AIMentorPlanItemUpdate(BaseModel):
    day_number: int | None = Field(default=None, ge=1, le=7)
    title: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    activity_type: str | None = Field(default=None, max_length=50)
    estimated_minutes: int | None = Field(default=None, gt=0)
    resources_json: JsonCollection | None = None


class AIMentorPlanItemProgressUpdate(BaseModel):
    status: PlanItemStatus


class AIMentorPlanItemRead(AIMentorSchema):
    id: int
    plan_week_id: int
    item_order: int
    day_number: int | None = None
    title: str
    description: str | None = None
    activity_type: str | None = None
    estimated_minutes: int | None = None
    resources_json: JsonCollection | None = None
    status: PlanItemStatus
    completed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class AIMentorPlanWeekBase(BaseModel):
    week_number: int = Field(ge=1, le=4)
    title: str = Field(min_length=1, max_length=255)
    goal: str | None = None
    description: str | None = None
    expected_outcome: str | None = None


class AIMentorPlanWeekCreate(AIMentorPlanWeekBase):
    items: list[AIMentorPlanItemCreate] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_unique_item_order(self) -> "AIMentorPlanWeekCreate":
        item_orders = [item.item_order for item in self.items]
        if len(item_orders) != len(set(item_orders)):
            raise ValueError("Hafta ichida item_order qiymatlari takrorlanmasligi kerak.")
        return self


class AIMentorPlanWeekUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    goal: str | None = None
    description: str | None = None
    expected_outcome: str | None = None


class AIMentorPlanWeekRead(AIMentorSchema):
    id: int
    plan_id: int
    week_number: int
    title: str
    goal: str | None = None
    description: str | None = None
    expected_outcome: str | None = None
    created_at: datetime
    updated_at: datetime
    items: list[AIMentorPlanItemRead] = Field(default_factory=list)


class AIMentorPlanCreate(BaseModel):
    diagnostic_session_id: int | None = Field(default=None, gt=0)
    title: str = Field(min_length=1, max_length=255)
    summary: str | None = None
    status: PlanStatus = "draft"
    generation_source: PlanGenerationSource = "mock"
    generation_metadata: JsonObject | None = None
    start_date: date | None = None
    end_date: date | None = None
    weeks: list[AIMentorPlanWeekCreate] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_plan(self) -> "AIMentorPlanCreate":
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.end_date < self.start_date
        ):
            raise ValueError("Reja yakun sanasi boshlanish sanasidan oldin bo‘lishi mumkin emas.")

        week_numbers = [week.week_number for week in self.weeks]
        if sorted(week_numbers) != [1, 2, 3, 4]:
            raise ValueError("Reja aynan 1, 2, 3 va 4-haftalardan iborat bo‘lishi kerak.")
        return self


class AIMentorMockPlanCreate(BaseModel):
    """Yakunlangan diagnostika asosida mock reja yaratish payload'i."""

    diagnostic_session_id: int | None = Field(default=None, gt=0)
    start_date: date | None = None


class AIMentorPlanUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    summary: str | None = None
    status: PlanStatus | None = None
    generation_metadata: JsonObject | None = None
    start_date: date | None = None
    end_date: date | None = None

    @model_validator(mode="after")
    def validate_date_range(self) -> "AIMentorPlanUpdate":
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.end_date < self.start_date
        ):
            raise ValueError("Reja yakun sanasi boshlanish sanasidan oldin bo‘lishi mumkin emas.")
        return self


class AIMentorPlanRead(AIMentorSchema):
    id: int
    student_id: int
    diagnostic_session_id: int | None = None
    version: int
    title: str
    summary: str | None = None
    status: PlanStatus
    generation_source: PlanGenerationSource
    generation_metadata: JsonObject | None = None
    start_date: date | None = None
    end_date: date | None = None
    created_at: datetime
    updated_at: datetime


class AIMentorPlanDetail(AIMentorPlanRead):
    weeks: list[AIMentorPlanWeekRead] = Field(default_factory=list)


class AIMentorPlanProgress(BaseModel):
    total_items: int = Field(default=0, ge=0)
    completed_items: int = Field(default=0, ge=0)
    in_progress_items: int = Field(default=0, ge=0)
    skipped_items: int = Field(default=0, ge=0)
    progress_percent: float = Field(default=0, ge=0, le=100)


class AIMentorPlanDetailResponse(BaseModel):
    plan: AIMentorPlanDetail
    progress: AIMentorPlanProgress


# ---------------------------------------------------------------------------
# Chat sessiyalari va xabarlar
# ---------------------------------------------------------------------------


class AIMentorChatSessionCreate(BaseModel):
    plan_id: int | None = Field(default=None, gt=0)
    title: str | None = Field(default=None, min_length=1, max_length=255)
    context_json: JsonObject | None = None


class AIMentorChatSessionUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    status: ChatSessionStatus | None = None
    context_json: JsonObject | None = None


class AIMentorChatMessageCreate(BaseModel):
    """Service qatlamida user/system/assistant xabarlarini saqlash uchun."""

    role: ChatMessageRole
    content: str = Field(min_length=1)
    model_name: str | None = Field(default=None, max_length=100)
    token_count: int | None = Field(default=None, ge=0)
    metadata_json: JsonObject | None = None

    @model_validator(mode="after")
    def strip_content(self) -> "AIMentorChatMessageCreate":
        self.content = self.content.strip()
        if not self.content:
            raise ValueError("Chat xabari bo‘sh bo‘lmasligi kerak.")
        return self


class AIMentorChatRequest(BaseModel):
    """Talaba frontendidan AI Mentor chatiga yuboriladigan oddiy so‘rov."""

    content: str = Field(min_length=1, max_length=20_000)

    @model_validator(mode="after")
    def strip_content(self) -> "AIMentorChatRequest":
        self.content = self.content.strip()
        if not self.content:
            raise ValueError("Chat xabari bo‘sh bo‘lmasligi kerak.")
        return self


class AIMentorChatMessageRead(AIMentorSchema):
    id: int
    session_id: int
    sequence_number: int
    role: ChatMessageRole
    content: str
    model_name: str | None = None
    token_count: int | None = None
    metadata_json: JsonObject | None = None
    created_at: datetime


class AIMentorChatSessionRead(AIMentorSchema):
    id: int
    student_id: int
    plan_id: int | None = None
    title: str | None = None
    status: ChatSessionStatus
    context_json: JsonObject | None = None
    last_message_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class AIMentorChatSessionDetail(AIMentorChatSessionRead):
    messages: list[AIMentorChatMessageRead] = Field(default_factory=list)


class AIMentorChatResponse(BaseModel):
    session: AIMentorChatSessionRead
    user_message: AIMentorChatMessageRead
    assistant_message: AIMentorChatMessageRead
