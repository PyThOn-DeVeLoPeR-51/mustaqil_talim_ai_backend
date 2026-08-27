import logging
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.models.student import Student
from app.models.submission import Submission
from app.models.task import Task, TaskAssignment
from app.models.teacher import Teacher
from app.schemas.task import TaskUpdate
from app.storage import (
    delete_storage_object,
    delete_storage_prefix,
    persist_upload_file,
    storage_read_url,
)


ALLOWED_REFERENCE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
logger = logging.getLogger("app.task.service")


def save_task_file(file: UploadFile, *, teacher_id: int, kind: str) -> str:
    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_REFERENCE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Faqat JPG, JPEG, PNG yoki PDF fayl yuklash mumkin.",
        )

    if kind not in {"references", "instructions"}:
        raise ValueError("Noto‘g‘ri task storage turi.")

    key = f"tasks/teachers/{teacher_id}/{kind}/{uuid.uuid4().hex}{extension}"
    return persist_upload_file(
        file.file,
        key,
        content_type=file.content_type,
    )


def save_reference_file(file: UploadFile, *, teacher_id: int) -> str:
    return save_task_file(file=file, teacher_id=teacher_id, kind="references")


def save_instruction_file(file: UploadFile, *, teacher_id: int) -> str:
    return save_task_file(file=file, teacher_id=teacher_id, kind="instructions")


def parse_student_ids(raw_student_ids: str | None) -> list[int]:
    if not raw_student_ids:
        return []

    ids: list[int] = []

    for item in raw_student_ids.split(","):
        item = item.strip()

        if not item:
            continue

        if not item.isdigit():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="assigned_student_ids faqat raqamlardan iborat bo‘lishi kerak. Masalan: 1,2,3",
            )

        ids.append(int(item))

    return list(dict.fromkeys(ids))


def ensure_teacher_owns_students(
    db: Session,
    teacher_id: int,
    student_ids: list[int],
) -> None:
    if not student_ids:
        return

    existing_ids = (
        db.query(Student.id)
        .filter(
            Student.teacher_id == teacher_id,
            Student.id.in_(student_ids),
        )
        .all()
    )

    existing_ids_set = {item[0] for item in existing_ids}
    requested_ids_set = set(student_ids)

    missing_ids = requested_ids_set - existing_ids_set

    if missing_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Bu student_id lar sizga tegishli emas yoki mavjud emas: {sorted(missing_ids)}",
        )


def validate_task_create_inputs(
    db: Session,
    teacher: Teacher,
    *,
    mode: str,
    assessment_stage: str | None,
    has_reference: bool,
    assigned_student_ids: list[int],
) -> None:
    if mode not in {"etalon", "optional"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="mode faqat 'etalon' yoki 'optional' bo‘lishi mumkin.",
        )

    if mode == "etalon" and not has_reference:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Etalon rejim uchun reference_file yuklash majburiy.",
        )

    if assessment_stage not in {None, "pretest", "intermediate", "posttest"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "assessment_stage faqat 'pretest', "
                "'intermediate' yoki 'posttest' bo‘lishi mumkin."
            ),
        )

    ensure_teacher_owns_students(
        db=db,
        teacher_id=teacher.id,
        student_ids=assigned_student_ids,
    )


def build_task_response(db: Session, task: Task, include_reference: bool = True) -> dict:
    assigned_student_ids = (
        db.query(TaskAssignment.student_id)
        .filter(TaskAssignment.task_id == task.id)
        .all()
    )

    reference_file_path = task.reference_file_path if include_reference else None
    instruction_file_path = task.instruction_file_path

    return {
        "id": task.id,
        "teacher_id": task.teacher_id,
        "title": task.title,
        "description": task.description,
        "topic": task.topic,
        "week_number": task.week_number,
        "assessment_stage": task.assessment_stage,
        "academic_period": task.academic_period,
        "mode": task.mode,
        "reference_file_path": reference_file_path,
        "reference_file_url": storage_read_url(reference_file_path),
        "instruction_file_path": instruction_file_path,
        "instruction_file_url": storage_read_url(instruction_file_path),
        "deadline": task.deadline,
        "is_active": task.is_active,
        "created_at": task.created_at,
        "assigned_student_ids": [item[0] for item in assigned_student_ids],
    }


def create_task_for_teacher(
    db: Session,
    teacher: Teacher,
    title: str,
    description: str | None,
    topic: str | None,
    week_number: int | None,
    assessment_stage: str | None,
    academic_period: str | None,
    mode: str,
    deadline,
    reference_file_path: str | None,
    instruction_file_path: str | None,
    assigned_student_ids: list[int],
) -> dict:
    validate_task_create_inputs(
        db,
        teacher,
        mode=mode,
        assessment_stage=assessment_stage,
        has_reference=bool(reference_file_path),
        assigned_student_ids=assigned_student_ids,
    )

    task = Task(
        teacher_id=teacher.id,
        title=title,
        description=description,
        topic=topic,
        week_number=week_number,
        assessment_stage=assessment_stage,
        academic_period=academic_period,
        mode=mode,
        reference_file_path=reference_file_path,
        instruction_file_path=instruction_file_path,
        deadline=deadline,
        is_active=True,
    )

    db.add(task)
    db.commit()
    db.refresh(task)

    for student_id in assigned_student_ids:
        assignment = TaskAssignment(
            task_id=task.id,
            student_id=student_id,
        )
        db.add(assignment)

    db.commit()
    db.refresh(task)

    return build_task_response(db=db, task=task)


def get_teacher_tasks(db: Session, teacher: Teacher) -> list[dict]:
    tasks = (
        db.query(Task)
        .filter(Task.teacher_id == teacher.id)
        .order_by(Task.id.desc())
        .all()
    )

    return [build_task_response(db=db, task=task) for task in tasks]


def get_teacher_task_or_404(
    db: Session,
    teacher: Teacher,
    task_id: int,
) -> Task:
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

    return task


def get_teacher_task_response_or_404(
    db: Session,
    teacher: Teacher,
    task_id: int,
) -> dict:
    task = get_teacher_task_or_404(
        db=db,
        teacher=teacher,
        task_id=task_id,
    )

    return build_task_response(db=db, task=task)


def update_teacher_task(
    db: Session,
    teacher: Teacher,
    task_id: int,
    task_data: TaskUpdate,
) -> dict:
    task = get_teacher_task_or_404(
        db=db,
        teacher=teacher,
        task_id=task_id,
    )

    update_data = task_data.model_dump(exclude_unset=True)

    new_mode = update_data.get("mode")

    if new_mode == "etalon" and not task.reference_file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Topshiriqni etalon rejimga o‘tkazish uchun avval reference file kerak.",
        )

    for key, value in update_data.items():
        setattr(task, key, value)

    db.commit()
    db.refresh(task)

    return build_task_response(db=db, task=task)


def replace_task_assignments(
    db: Session,
    teacher: Teacher,
    task_id: int,
    student_ids: list[int],
) -> dict:
    task = get_teacher_task_or_404(
        db=db,
        teacher=teacher,
        task_id=task_id,
    )

    student_ids = list(dict.fromkeys(student_ids))

    ensure_teacher_owns_students(
        db=db,
        teacher_id=teacher.id,
        student_ids=student_ids,
    )

    db.query(TaskAssignment).filter(TaskAssignment.task_id == task.id).delete()

    for student_id in student_ids:
        assignment = TaskAssignment(
            task_id=task.id,
            student_id=student_id,
        )
        db.add(assignment)

    db.commit()
    db.refresh(task)

    return build_task_response(db=db, task=task)


def delete_teacher_task(
    db: Session,
    teacher: Teacher,
    task_id: int,
) -> None:
    task = get_teacher_task_or_404(
        db=db,
        teacher=teacher,
        task_id=task_id,
    )

    submissions = (
        db.query(Submission.id, Submission.uploaded_file_path, Submission.overlay_path)
        .filter(Submission.task_id == task.id)
        .all()
    )
    object_paths = [task.reference_file_path, task.instruction_file_path]
    object_paths.extend(item.uploaded_file_path for item in submissions)
    object_paths.extend(item.overlay_path for item in submissions)
    submission_ids = [item.id for item in submissions]

    db.delete(task)
    db.commit()

    # Storage cleanup is best-effort: DB deletion must not be rolled back by an
    # external object-store outage. Orphan cleanup can be retried later.
    for object_path in object_paths:
        try:
            delete_storage_object(object_path)
        except Exception as exc:  # pragma: no cover - external storage failure
            logger.warning("Task storage object cleanup failed: %s", exc)
    for submission_id in submission_ids:
        try:
            delete_storage_prefix(f"results/submissions/{submission_id}")
        except Exception as exc:  # pragma: no cover - external storage failure
            logger.warning("Task result prefix cleanup failed: %s", exc)


def is_reference_visible_for_student(db: Session, task_id: int, student_id: int, max_attempts: int = 2) -> bool:
    """Etalon faylini talaba faqat barcha urinishlarni tugatgandan keyin ko‘radi."""

    attempt_count = (
        db.query(Submission.id)
        .filter(
            Submission.task_id == task_id,
            Submission.student_id == student_id,
            Submission.status.in_(("evaluated", "failed")),
        )
        .count()
    )
    return attempt_count >= max_attempts

def get_tasks_for_student(db: Session, student: Student) -> list[dict]:
    tasks = (
        db.query(Task)
        .join(TaskAssignment, TaskAssignment.task_id == Task.id)
        .filter(
            TaskAssignment.student_id == student.id,
            Task.is_active == True,  # noqa: E712
        )
        .order_by(Task.id.desc())
        .all()
    )

    return [
        build_task_response(
            db=db,
            task=task,
            include_reference=is_reference_visible_for_student(
                db=db,
                task_id=task.id,
                student_id=student.id,
            ),
        )
        for task in tasks
    ]