from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.models.student import Student
from app.models.submission import Submission
from app.models.task import Task, TaskAssignment
from app.models.teacher import Teacher
from app.services.ai_service import evaluate_submission_with_ai
from app.storage import (
    delete_storage_object,
    delete_storage_prefix,
    materialize_storage_file,
    persist_local_file,
    persist_upload_file,
)


ALLOWED_SUBMISSION_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}


def save_submission_file(
    file: UploadFile,
    *,
    student_id: int,
    task_id: int,
    attempt_number: int,
) -> str:
    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_SUBMISSION_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Faqat JPG, JPEG, PNG yoki PDF fayl yuklash mumkin.",
        )

    key = (
        f"submissions/students/{student_id}/tasks/{task_id}/"
        f"attempt-{attempt_number}/{uuid.uuid4().hex}{extension}"
    )
    return persist_upload_file(
        file.file,
        key,
        content_type=file.content_type,
    )


def _normalize_path_string(value: str) -> str:
    return value.replace("\\", "/")


def _rewrite_artifact_paths(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return mapping.get(_normalize_path_string(value), value)
    if isinstance(value, dict):
        return {key: _rewrite_artifact_paths(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_artifact_paths(item, mapping) for item in value]
    return value


def _persist_ai_outputs(
    ai_result: dict[str, Any],
    *,
    output_dir: Path,
    submission_id: int,
    input_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    prefix = f"results/submissions/{submission_id}"
    mapping: dict[str, str] = {
        _normalize_path_string(key): value
        for key, value in (input_mapping or {}).items()
    }

    try:
        for artifact in sorted(output_dir.rglob("*")):
            if not artifact.is_file():
                continue
            relative = artifact.relative_to(output_dir).as_posix()
            key = persist_local_file(artifact, f"{prefix}/{relative}")
            mapping[_normalize_path_string(str(artifact))] = key
            try:
                mapping[_normalize_path_string(str(artifact.resolve()))] = key
            except OSError:
                pass
    except Exception:
        try:
            delete_storage_prefix(prefix)
        except Exception:
            pass
        raise

    persisted = dict(ai_result)
    persisted["ai_json_result"] = _rewrite_artifact_paths(
        persisted.get("ai_json_result"),
        mapping,
    )
    persisted["table_json"] = _rewrite_artifact_paths(
        persisted.get("table_json"),
        mapping,
    )
    overlay = persisted.get("overlay_path")
    if isinstance(overlay, str):
        persisted["overlay_path"] = mapping.get(
            _normalize_path_string(overlay),
            overlay,
        )
    return persisted


def _evaluate_submission_from_storage(
    *,
    submission: Submission,
    task: Task,
) -> dict[str, Any]:
    with ExitStack() as stack:
        student_path = stack.enter_context(
            materialize_storage_file(submission.uploaded_file_path)
        )
        reference_path: Path | None = None
        if task.reference_file_path:
            reference_path = stack.enter_context(
                materialize_storage_file(task.reference_file_path)
            )

        with tempfile.TemporaryDirectory(
            prefix=f"drawing-ai-{submission.id}-"
        ) as temp_dir:
            output_dir = Path(temp_dir) / "results"
            output_dir.mkdir(parents=True, exist_ok=True)
            ai_result = evaluate_submission_with_ai(
                mode=task.mode,
                student_file_path=str(student_path),
                reference_file_path=(str(reference_path) if reference_path else None),
                task_text=task.description or "",
                output_dir=output_dir,
            )
            input_mapping = {
                str(student_path): submission.uploaded_file_path,
            }
            if reference_path is not None and task.reference_file_path:
                input_mapping[str(reference_path)] = task.reference_file_path

            persisted = _persist_ai_outputs(
                ai_result,
                output_dir=output_dir,
                submission_id=submission.id,
                input_mapping=input_mapping,
            )
            metadata = persisted.get("ai_json_result")
            if isinstance(metadata, dict):
                drawing_meta = metadata.get("drawing_ai_v2")
                if isinstance(drawing_meta, dict):
                    drawing_meta = dict(drawing_meta)
                    drawing_meta["student_file"] = submission.uploaded_file_path
                    drawing_meta["reference_file"] = task.reference_file_path
                    metadata = dict(metadata)
                    metadata["drawing_ai_v2"] = drawing_meta
                    persisted["ai_json_result"] = metadata
            return persisted


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
    uploaded_file_path: str | None = None,
    drawing_file: UploadFile | None = None,
) -> Submission:
    # Validate assignment/deadline/attempt BEFORE writing to object storage.
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

    managed_storage = drawing_file is not None
    if managed_storage:
        stored_file_path = save_submission_file(
            drawing_file,
            student_id=student.id,
            task_id=task.id,
            attempt_number=attempt_number,
        )
    elif uploaded_file_path:
        # Backward compatibility for service-level tests and legacy callers.
        stored_file_path = uploaded_file_path
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="drawing_file yuborilishi kerak.",
        )

    submission = Submission(
        task_id=task.id,
        student_id=student.id,
        attempt_number=attempt_number,
        mode=task.mode,
        uploaded_file_path=stored_file_path,
        total_score=None,
        ai_json_result=None,
        overlay_path=None,
        table_json=None,
        status="pending",
    )

    try:
        db.add(submission)
        db.commit()
        db.refresh(submission)
    except Exception:
        db.rollback()
        if managed_storage:
            try:
                delete_storage_object(stored_file_path)
            except Exception:
                pass
        raise

    try:
        if managed_storage:
            ai_result = _evaluate_submission_from_storage(
                submission=submission,
                task=task,
            )
        else:
            ai_result = evaluate_submission_with_ai(
                mode=task.mode,
                student_file_path=stored_file_path,
                reference_file_path=task.reference_file_path,
                task_text=task.description or "",
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