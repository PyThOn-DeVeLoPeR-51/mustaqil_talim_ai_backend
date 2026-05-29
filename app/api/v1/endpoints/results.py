from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.student import Student
from app.models.teacher import Teacher
from app.schemas.result import ResultRead
from app.services.auth_service import get_current_student, get_current_teacher
from app.services.result_service import (
    get_student_result_by_id,
    get_student_results,
    get_student_task_results,
    get_teacher_result_by_id,
    get_teacher_results,
    get_teacher_student_results,
    get_teacher_task_results,
)


router = APIRouter()


@router.get("/teacher", response_model=list[ResultRead])
def teacher_results(
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return get_teacher_results(
        db=db,
        teacher=current_teacher,
    )


@router.get("/teacher/task/{task_id}", response_model=list[ResultRead])
def teacher_task_results(
    task_id: int,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return get_teacher_task_results(
        db=db,
        teacher=current_teacher,
        task_id=task_id,
    )


@router.get("/teacher/student/{student_id}", response_model=list[ResultRead])
def teacher_student_results(
    student_id: int,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return get_teacher_student_results(
        db=db,
        teacher=current_teacher,
        student_id=student_id,
    )


@router.get("/teacher/{submission_id}", response_model=ResultRead)
def teacher_result_detail(
    submission_id: int,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return get_teacher_result_by_id(
        db=db,
        teacher=current_teacher,
        submission_id=submission_id,
    )


@router.get("/student/my", response_model=list[ResultRead])
def student_results(
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    return get_student_results(
        db=db,
        student=current_student,
    )


@router.get("/student/task/{task_id}", response_model=list[ResultRead])
def student_task_results(
    task_id: int,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    return get_student_task_results(
        db=db,
        student=current_student,
        task_id=task_id,
    )


@router.get("/student/{submission_id}", response_model=ResultRead)
def student_result_detail(
    submission_id: int,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    return get_student_result_by_id(
        db=db,
        student=current_student,
        submission_id=submission_id,
    )