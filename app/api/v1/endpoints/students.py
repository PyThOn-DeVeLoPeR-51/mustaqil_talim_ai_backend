from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.teacher import Teacher
from app.schemas.student import (
    StudentCreate,
    StudentCreateResponse,
    StudentRead,
    StudentUpdate,
)
from app.services.auth_service import get_current_teacher
from app.services.student_service import (
    create_student_for_teacher,
    delete_teacher_student,
    get_students_for_teacher,
    get_teacher_student_or_404,
    update_teacher_student,
)


router = APIRouter()


@router.post(
    "",
    response_model=StudentCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_student(
    student_data: StudentCreate,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    student, plain_password = create_student_for_teacher(
        db=db,
        teacher=current_teacher,
        student_data=student_data,
    )

    return {
        "student": student,
        "login": student.login,
        "temporary_password": plain_password,
    }


@router.get("", response_model=list[StudentRead])
def get_my_students(
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return get_students_for_teacher(
        db=db,
        teacher=current_teacher,
    )


@router.get("/{student_id}", response_model=StudentRead)
def get_my_student(
    student_id: int,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return get_teacher_student_or_404(
        db=db,
        teacher=current_teacher,
        student_id=student_id,
    )


@router.patch("/{student_id}", response_model=StudentRead)
def update_my_student(
    student_id: int,
    student_data: StudentUpdate,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return update_teacher_student(
        db=db,
        teacher=current_teacher,
        student_id=student_id,
        student_data=student_data,
    )


@router.delete("/{student_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_my_student(
    student_id: int,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    delete_teacher_student(
        db=db,
        teacher=current_teacher,
        student_id=student_id,
    )