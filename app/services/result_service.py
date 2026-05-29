from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.files import to_upload_url
from app.models.student import Student
from app.models.submission import Submission
from app.models.task import Task
from app.models.teacher import Teacher


def build_result_response(
    submission: Submission,
    task: Task | None = None,
    student: Student | None = None,
) -> dict:
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
        build_result_response(
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
        build_result_response(
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

    return build_result_response(
        submission=submission,
        task=task,
        student=student_obj,
    )