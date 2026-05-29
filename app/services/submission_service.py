import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.models.student import Student
from app.models.submission import Submission
from app.models.task import Task, TaskAssignment
from app.models.teacher import Teacher

from app.services.ai_service import evaluate_submission_with_ai


ALLOWED_SUBMISSION_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}


def save_submission_file(
    file: UploadFile,
    upload_dir: str = "app/uploads/submissions",
) -> str:
    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_SUBMISSION_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Faqat JPG, JPEG, PNG yoki PDF fayl yuklash mumkin.",
        )

    Path(upload_dir).mkdir(parents=True, exist_ok=True)

    safe_filename = f"{uuid.uuid4().hex}{extension}"
    file_path = Path(upload_dir) / safe_filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return str(file_path)


def get_assigned_task_for_student_or_404(
    db: Session,
    student: Student,
    task_id: int,
) -> Task:
    task = (
        db.query(Task)
        .join(TaskAssignment, TaskAssignment.task_id == Task.id)
        .filter(
            Task.id == task_id,
            TaskAssignment.student_id == student.id,
            Task.is_active == True,  # noqa: E712
        )
        .first()
    )

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bu topshiriq sizga biriktirilmagan yoki faol emas.",
        )

    return task


def check_task_deadline(task: Task) -> None:
    if task.deadline is None:
        return

    now = datetime.now(timezone.utc)

    deadline = task.deadline
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)

    if now > deadline:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Topshiriq muddati tugagan. Endi submission yuborib bo‘lmaydi.",
        )


def get_next_attempt_number(
    db: Session,
    student_id: int,
    task_id: int,
) -> int:
    attempts_count = (
        db.query(Submission)
        .filter(
            Submission.student_id == student_id,
            Submission.task_id == task_id,
        )
        .count()
    )

    if attempts_count >= 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bu topshiriq uchun 2 ta urinishdan foydalanib bo‘lgansiz.",
        )

    return attempts_count + 1


def create_submission_for_student(
    db: Session,
    student: Student,
    task_id: int,
    uploaded_file_path: str,
) -> Submission:
    task = get_assigned_task_for_student_or_404(
        db=db,
        student=student,
        task_id=task_id,
    )

    check_task_deadline(task)

    attempt_number = get_next_attempt_number(
        db=db,
        student_id=student.id,
        task_id=task.id,
    )

    submission = Submission(
        task_id=task.id,
        student_id=student.id,
        attempt_number=attempt_number,
        mode=task.mode,
        uploaded_file_path=uploaded_file_path,
        total_score=None,
        ai_json_result=None,
        overlay_path=None,
        table_json=None,
        status="pending",
    )

    db.add(submission)
    db.commit()
    db.refresh(submission)

    try:
        ai_result = evaluate_submission_with_ai(
            mode=task.mode,
            student_file_path=uploaded_file_path,
            reference_file_path=task.reference_file_path,
        )

        submission.total_score = ai_result["total_score"]
        submission.ai_json_result = ai_result["ai_json_result"]
        submission.overlay_path = ai_result["overlay_path"]
        submission.table_json = ai_result["table_json"]
        submission.status = "evaluated"

    except Exception as error:
        submission.status = "failed"
        submission.ai_json_result = {
            "error": str(error),
        }

    db.commit()
    db.refresh(submission)

    return submission


def get_student_submissions(
    db: Session,
    student: Student,
) -> list[Submission]:
    return (
        db.query(Submission)
        .filter(Submission.student_id == student.id)
        .order_by(Submission.id.desc())
        .all()
    )


def get_student_task_submissions(
    db: Session,
    student: Student,
    task_id: int,
) -> list[Submission]:
    return (
        db.query(Submission)
        .filter(
            Submission.student_id == student.id,
            Submission.task_id == task_id,
        )
        .order_by(Submission.attempt_number.asc())
        .all()
    )


def get_teacher_submissions(
    db: Session,
    teacher: Teacher,
) -> list[Submission]:
    return (
        db.query(Submission)
        .join(Task, Task.id == Submission.task_id)
        .filter(Task.teacher_id == teacher.id)
        .order_by(Submission.id.desc())
        .all()
    )


def get_teacher_task_submissions(
    db: Session,
    teacher: Teacher,
    task_id: int,
) -> list[Submission]:
    task = (
        db.query(Task)
        .filter(
            Task.id == task_id,
            Task.teacher_id == teacher.id,
        )
        .first()
    )

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Topshiriq topilmadi.",
        )

    return (
        db.query(Submission)
        .filter(Submission.task_id == task.id)
        .order_by(Submission.student_id.asc(), Submission.attempt_number.asc())
        .all()
    )


def get_teacher_submission_or_404(
    db: Session,
    teacher: Teacher,
    submission_id: int,
) -> Submission:
    submission = (
        db.query(Submission)
        .join(Task, Task.id == Submission.task_id)
        .filter(
            Submission.id == submission_id,
            Task.teacher_id == teacher.id,
        )
        .first()
    )

    if not submission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Submission topilmadi.",
        )

    return submission


def get_student_submission_or_404(
    db: Session,
    student: Student,
    submission_id: int,
) -> Submission:
    submission = (
        db.query(Submission)
        .filter(
            Submission.id == submission_id,
            Submission.student_id == student.id,
        )
        .first()
    )

    if not submission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Submission topilmadi.",
        )

    return submission