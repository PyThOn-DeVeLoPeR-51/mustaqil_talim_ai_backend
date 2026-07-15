from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ExperimentGroup = Literal["experimental", "control"]


class StudentCreate(BaseModel):
    full_name: str = Field(min_length=3, max_length=255)
    university: str | None = Field(default=None, max_length=255)
    direction: str | None = Field(default=None, max_length=255)
    stage: str | None = Field(default=None, max_length=100)

    group_name: str | None = Field(default=None, max_length=100)
    experiment_group: ExperimentGroup | None = None
    cohort_year: int | None = Field(default=None, ge=2000, le=2100)

    # Agar teacher login/parolni o‘zi bermasa, backend avtomatik yaratadi.
    login: str | None = Field(default=None, min_length=3, max_length=100)
    password: str | None = Field(default=None, min_length=6, max_length=100)


class StudentUpdate(BaseModel):
    full_name: str | None = Field(default=None, min_length=3, max_length=255)
    university: str | None = Field(default=None, max_length=255)
    direction: str | None = Field(default=None, max_length=255)
    stage: str | None = Field(default=None, max_length=100)
    group_name: str | None = Field(default=None, max_length=100)
    experiment_group: ExperimentGroup | None = None
    cohort_year: int | None = Field(default=None, ge=2000, le=2100)
    is_active: bool | None = None


class StudentRead(BaseModel):
    id: int
    teacher_id: int
    full_name: str
    university: str | None = None
    direction: str | None = None
    stage: str | None = None
    group_name: str | None = None
    experiment_group: ExperimentGroup | None = None
    cohort_year: int | None = None
    login: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class StudentCreateResponse(BaseModel):
    student: StudentRead
    login: str
    temporary_password: str