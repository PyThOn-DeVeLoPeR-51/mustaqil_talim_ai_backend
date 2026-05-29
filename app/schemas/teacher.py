from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class TeacherCreate(BaseModel):
    first_name: str = Field(min_length=2, max_length=100)
    last_name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    position: str | None = Field(default=None, max_length=150)
    university: str | None = Field(default=None, max_length=255)


class TeacherRead(BaseModel):
    id: int
    first_name: str
    last_name: str
    email: EmailStr
    position: str | None = None
    university: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)