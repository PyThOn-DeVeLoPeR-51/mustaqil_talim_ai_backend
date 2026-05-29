from pydantic import BaseModel, EmailStr, Field

from app.schemas.teacher import TeacherRead
from app.schemas.student import StudentRead


class TeacherLogin(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=100)


class StudentLogin(BaseModel):
    login: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=6, max_length=100)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TeacherLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    teacher: TeacherRead


class StudentLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    student: StudentRead