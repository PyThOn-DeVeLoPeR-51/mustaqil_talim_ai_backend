"""AI Mentor LLM providerlari uchun qat'iy kontraktlar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, Field, model_validator


class DiagnosticAnalysisOutput(BaseModel):
    """LLM diagnostik tahlilining strukturalangan natijasi."""

    summary: str = Field(min_length=30, max_length=3000)
    risk_level: Literal["low", "medium", "high"]
    strengths: list[str] = Field(min_length=1, max_length=6)
    improvement_areas: list[str] = Field(min_length=1, max_length=6)
    recommended_strategies: list[str] = Field(min_length=2, max_length=8)
    focus_topic: str = Field(min_length=1, max_length=500)
    weekly_hours: int = Field(ge=1, le=60)
    session_minutes: int = Field(ge=10, le=180)


class PlanItemOutput(BaseModel):
    item_order: int = Field(ge=1, le=3)
    day_number: int = Field(ge=1, le=7)
    title: str = Field(min_length=3, max_length=255)
    description: str = Field(min_length=10, max_length=2000)
    activity_type: str = Field(min_length=2, max_length=50)
    estimated_minutes: int = Field(ge=10, le=180)
    resources: list[str] = Field(default_factory=list, max_length=5)


class PlanWeekOutput(BaseModel):
    week_number: int = Field(ge=1, le=4)
    title: str = Field(min_length=3, max_length=255)
    goal: str = Field(min_length=10, max_length=1500)
    description: str = Field(min_length=10, max_length=2000)
    expected_outcome: str = Field(min_length=10, max_length=1500)
    items: list[PlanItemOutput] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_item_order(self) -> "PlanWeekOutput":
        if sorted(item.item_order for item in self.items) != [1, 2, 3]:
            raise ValueError("Har bir haftada item_order 1, 2 va 3 bo‘lishi kerak.")
        return self


class PlanGenerationOutput(BaseModel):
    """LLM yaratadigan aynan 4 haftalik reja."""

    title: str = Field(min_length=3, max_length=255)
    summary: str = Field(min_length=20, max_length=3000)
    weeks: list[PlanWeekOutput] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_week_numbers(self) -> "PlanGenerationOutput":
        if sorted(week.week_number for week in self.weeks) != [1, 2, 3, 4]:
            raise ValueError("Reja aynan 1, 2, 3 va 4-haftalardan iborat bo‘lishi kerak.")
        return self


@dataclass(frozen=True, slots=True)
class LLMCallMetadata:
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    request_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "request_id": self.request_id,
        }


OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class StructuredLLMResult(Generic[OutputT]):
    output: OutputT
    metadata: LLMCallMetadata


@dataclass(frozen=True, slots=True)
class TextLLMResult:
    text: str
    metadata: LLMCallMetadata


class LLMProviderError(RuntimeError):
    """Provider xatosini foydalanuvchiga xavfsiz kod bilan uzatadi."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


class AIMentorLLMProvider(Protocol):
    provider_name: str
    model_name: str

    def analyze_diagnostic(
        self,
        context: dict[str, Any],
    ) -> StructuredLLMResult[DiagnosticAnalysisOutput]: ...

    def generate_plan(
        self,
        context: dict[str, Any],
    ) -> StructuredLLMResult[PlanGenerationOutput]: ...

    def chat_reply(self, context: dict[str, Any]) -> TextLLMResult: ...
