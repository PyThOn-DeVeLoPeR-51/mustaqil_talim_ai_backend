from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import create_access_token
from app.db.database import get_db
from app.models.student import Student
from app.models.teacher import Teacher
from app.schemas.auth import (
    StudentLogin,
    StudentLoginResponse,
    TeacherLogin,
    TeacherLoginResponse,
)
from app.schemas.student import StudentRead
from app.schemas.teacher import TeacherCreate, TeacherRead
from app.services.auth_service import (
    authenticate_student,
    authenticate_teacher,
    create_teacher,
    get_current_student,
    get_current_teacher,
)


router = APIRouter()


@router.post(
    "/teacher/register",
    response_model=TeacherRead,
    status_code=status.HTTP_201_CREATED,
)
def register_teacher(
    teacher_data: TeacherCreate,
    db: Session = Depends(get_db),
):
    teacher = create_teacher(db=db, teacher_data=teacher_data)
    return teacher


@router.post(
    "/teacher/login",
    response_model=TeacherLoginResponse,
)
def login_teacher(
    login_data: TeacherLogin,
    db: Session = Depends(get_db),
):
    teacher = authenticate_teacher(
        db=db,
        email=login_data.email,
        password=login_data.password,
    )

    if not teacher:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email yoki parol noto‘g‘ri.",
        )

    access_token = create_access_token(
        subject=str(teacher.id),
        role="teacher",
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "teacher": teacher,
    }


@router.get("/teacher/me", response_model=TeacherRead)
def get_teacher_me(
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return current_teacher


@router.post(
    "/student/login",
    response_model=StudentLoginResponse,
)
def login_student(
    login_data: StudentLogin,
    db: Session = Depends(get_db),
):
    student = authenticate_student(
        db=db,
        login=login_data.login,
        password=login_data.password,
    )

    if not student:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login yoki parol noto‘g‘ri.",
        )

    access_token = create_access_token(
        subject=str(student.id),
        role="student",
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "student": student,
    }


@router.get("/student/me", response_model=StudentRead)
def get_student_me(
    current_student: Student = Depends(get_current_student),
):
    return current_student