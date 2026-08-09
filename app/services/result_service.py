from copy import deepcopy

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.files import ensure_file_preview, to_upload_url
from app.models.student import Student
from app.models.submission import Submission
from app.models.task import Task
from app.models.teacher import Teacher



def _student_attempt_count(db: Session, task_id: int, student_id: int) -> int:
    return (
        db.query(Submission.id)
        .filter(
            Submission.task_id == task_id,
            Submission.student_id == student_id,
        )
        .count()
    )


def _reference_visible_to_student(db: Session, task_id: int, student_id: int, max_attempts: int = 2) -> bool:
    return _student_attempt_count(db=db, task_id=task_id, student_id=student_id) >= max_attempts


def _hide_reference_from_ai_json(ai_json_result):
    if not isinstance(ai_json_result, dict):
        return ai_json_result

    cleaned = deepcopy(ai_json_result)
    for key in (
        "reference_file",
        "reference_file_path",
        "reference_path",
        "etalon_file",
        "etalon_file_path",
    ):
        cleaned.pop(key, None)

    metadata = cleaned.get("drawing_ai_v2")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        metadata["reference_file"] = None
        cleaned["drawing_ai_v2"] = metadata

    return cleaned


def build_student_result_response(
    db: Session,
    submission: Submission,
    task: Task | None = None,
    student: Student | None = None,
) -> dict:
    response = build_result_response(
        submission=submission,
        task=task,
        student=student,
    )

    if task is None or not _reference_visible_to_student(
        db=db,
        task_id=submission.task_id,
        student_id=submission.student_id,
    ):
        response["ai_json_result"] = _hide_reference_from_ai_json(response.get("ai_json_result"))

    return response

def build_result_response(
    submission: Submission,
    task: Task | None = None,
    student: Student | None = None,
) -> dict:
    uploaded_preview_path = ensure_file_preview(
        submission.uploaded_file_path
    )

    return {
        "id": submission.id,

        "task_id": submission.task_id,
        "task_title": task.title if task else None,

        "student_id": submission.student_id,
        "student_full_name": student.full_name if student else None,

        "attempt_number": submission.attempt_number,
        "mode": submission.mode,

        "uploaded_file_path": submission.uploaded_file_path,
        "uploaded_file_url": to_upload_url(submission.uploaded_file_path),
        "uploaded_preview_url": to_upload_url(uploaded_preview_path),

        "total_score": submission.total_score,
        "ai_json_result": submission.ai_json_result,

        "overlay_path": submission.overlay_path,
        "overlay_url": to_upload_url(submission.overlay_path),

        "table_json": submission.table_json,

        "status": submission.status,
        "created_at": submission.created_at,
    }


def get_teacher_results(
    db: Session,
    teacher: Teacher,
) -> list[dict]:
    rows = (
        db.query(Submission, Task, Student)
        .join(Task, Task.id == Submission.task_id)
        .join(Student, Student.id == Submission.student_id)
        .filter(Task.teacher_id == teacher.id)
        .order_by(Submission.created_at.desc())
        .all()
    )

    return [
        build_result_response(
            submission=submission,
            task=task,
            student=student,
        )
        for submission, task, student in rows
    ]


def get_teacher_task_results(
    db: Session,
    teacher: Teacher,
    task_id: int,
) -> list[dict]:
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

    rows = (
        db.query(Submission, Task, Student)
        .join(Task, Task.id == Submission.task_id)
        .join(Student, Student.id == Submission.student_id)
        .filter(
            Task.teacher_id == teacher.id,
            Task.id == task_id,
        )
        .order_by(Submission.student_id.asc(), Submission.attempt_number.asc())
        .all()
    )

    return [
        build_result_response(
            submission=submission,
            task=task_obj,
            student=student,
        )
        for submission, task_obj, student in rows
    ]


def get_teacher_student_results(
    db: Session,
    teacher: Teacher,
    student_id: int,
) -> list[dict]:
    student = (
        db.query(Student)
        .filter(
            Student.id == student_id,
            Student.teacher_id == teacher.id,
        )
        .first()
    )

    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Talaba topilmadi.",
        )

    rows = (
        db.query(Submission, Task, Student)
        .join(Task, Task.id == Submission.task_id)
        .join(Student, Student.id == Submission.student_id)
        .filter(
            Task.teacher_id == teacher.id,
            Submission.student_id == student_id,
        )
        .order_by(Submission.created_at.desc())
        .all()
    )

    return [
        build_result_response(
            submission=submission,
            task=task,
            student=student_obj,
        )
        for submission, task, student_obj in rows
    ]


def get_teacher_result_by_id(
    db: Session,
    teacher: Teacher,
    submission_id: int,
) -> dict:
    row = (
        db.query(Submission, Task, Student)
        .join(Task, Task.id == Submission.task_id)
        .join(Student, Student.id == Submission.student_id)
        .filter(
            Submission.id == submission_id,
            Task.teacher_id == teacher.id,
        )
        .first()
    )

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Natija topilmadi.",
        )

    submission, task, student = row

    return build_result_response(
        submission=submission,
        task=task,
        student=student,
    )


def get_student_results(
    db: Session,
    student: Student,
) -> list[dict]:
    rows = (
        db.query(Submission, Task, Student)
        .join(Task, Task.id == Submission.task_id)
        .join(Student, Student.id == Submission.student_id)
        .filter(Submission.student_id == student.id)
        .order_by(Submission.created_at.desc())
        .all()
    )

    return [
        build_student_result_response(
            db=db,
            submission=submission,
            task=task,
            student=student_obj,
        )
        for submission, task, student_obj in rows
    ]


def get_student_task_results(
    db: Session,
    student: Student,
    task_id: int,
) -> list[dict]:
    rows = (
        db.query(Submission, Task, Student)
        .join(Task, Task.id == Submission.task_id)
        .join(Student, Student.id == Submission.student_id)
        .filter(
            Submission.student_id == student.id,
            Submission.task_id == task_id,
        )
        .order_by(Submission.attempt_number.asc())
        .all()
    )

    return [
        build_student_result_response(
            db=db,
            submission=submission,
            task=task,
            student=student_obj,
        )
        for submission, task, student_obj in rows
    ]


def get_student_result_by_id(
    db: Session,
    student: Student,
    submission_id: int,
) -> dict:
    row = (
        db.query(Submission, Task, Student)
        .join(Task, Task.id == Submission.task_id)
        .join(Student, Student.id == Submission.student_id)
        .filter(
            Submission.id == submission_id,
            Submission.student_id == student.id,
        )
        .first()
    )

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Natija topilmadi.",
        )

    submission, task, student_obj = row

    return build_student_result_response(
        db=db,
        submission=submission,
        task=task,
        student=student_obj,
    )