from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


TaskMode = Literal["etalon", "optional"]


class TaskRead(BaseModel):
    id: int
    teacher_id: int
    title: str
    description: str | None = None
    mode: TaskMode
    reference_file_path: str | None = None
    instruction_file_path: str | None = None
    deadline: datetime | None = None
    is_active: bool
    created_at: datetime

    assigned_student_ids: list[int] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class TaskUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=3, max_length=255)
    description: str | None = None
    mode: TaskMode | None = None
    deadline: datetime | None = None
    is_active: bool | None = None


class TaskAssignmentUpdate(BaseModel):
    student_ids: list[int] = Field(default_factory=list)