from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.student import Student
from app.models.teacher import Teacher
from app.schemas.submission import SubmissionRead
from app.services.auth_service import get_current_student, get_current_teacher
from app.services.submission_service import (
    create_submission_for_student,
    get_student_submission_or_404,
    get_student_submissions,
    get_student_task_submissions,
    get_teacher_submission_or_404,
    get_teacher_submissions,
    get_teacher_task_submissions,
    save_submission_file,
)


router = APIRouter()


@router.post(
    "",
    response_model=SubmissionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_submission(
    task_id: int = Form(...),
    drawing_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    uploaded_file_path = save_submission_file(drawing_file)

    return create_submission_for_student(
        db=db,
        student=current_student,
        task_id=task_id,
        uploaded_file_path=uploaded_file_path,
    )


@router.get("/my", response_model=list[SubmissionRead])
def get_my_submissions(
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    return get_student_submissions(
        db=db,
        student=current_student,
    )


@router.get("/my/task/{task_id}", response_model=list[SubmissionRead])
def get_my_task_submissions(
    task_id: int,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    return get_student_task_submissions(
        db=db,
        student=current_student,
        task_id=task_id,
    )


@router.get("/my/{submission_id}", response_model=SubmissionRead)
def get_my_submission(
    submission_id: int,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    return get_student_submission_or_404(
        db=db,
        student=current_student,
        submission_id=submission_id,
    )


@router.get("/teacher", response_model=list[SubmissionRead])
def get_submissions_for_teacher(
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return get_teacher_submissions(
        db=db,
        teacher=current_teacher,
    )


@router.get("/teacher/task/{task_id}", response_model=list[SubmissionRead])
def get_task_submissions_for_teacher(
    task_id: int,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return get_teacher_task_submissions(
        db=db,
        teacher=current_teacher,
        task_id=task_id,
    )


@router.get("/teacher/{submission_id}", response_model=SubmissionRead)
def get_submission_for_teacher(
    submission_id: int,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return get_teacher_submission_or_404(
        db=db,
        teacher=current_teacher,
        submission_id=submission_id,
    )