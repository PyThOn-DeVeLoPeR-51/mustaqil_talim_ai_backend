from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.student import Student
from app.models.teacher import Teacher
from app.schemas.task import TaskAssignmentUpdate, TaskRead, TaskUpdate
from app.services.auth_service import get_current_student, get_current_teacher
from app.services.task_service import (
    create_task_for_teacher,
    delete_teacher_task,
    get_tasks_for_student,
    get_teacher_task_response_or_404,
    get_teacher_tasks,
    parse_student_ids,
    replace_task_assignments,
    save_instruction_file,
    save_reference_file,
    update_teacher_task,
)


router = APIRouter()


@router.post(
    "",
    response_model=TaskRead,
    status_code=status.HTTP_201_CREATED,
)
def create_task(
    title: str = Form(...),
    mode: str = Form(...),
    description: str | None = Form(default=None),
    deadline: datetime | None = Form(default=None),
    assigned_student_ids: str | None = Form(default=None),
    reference_file: UploadFile | None = File(default=None),
    instruction_file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    reference_file_path = None
    instruction_file_path = None

    if reference_file:
        reference_file_path = save_reference_file(reference_file)

    if instruction_file:
        instruction_file_path = save_instruction_file(instruction_file)

    student_ids = parse_student_ids(assigned_student_ids)

    return create_task_for_teacher(
        db=db,
        teacher=current_teacher,
        title=title,
        description=description,
        mode=mode,
        deadline=deadline,
        reference_file_path=reference_file_path,
        instruction_file_path=instruction_file_path,
        assigned_student_ids=student_ids,
    )


@router.get("", response_model=list[TaskRead])
def get_my_tasks(
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return get_teacher_tasks(
        db=db,
        teacher=current_teacher,
    )


@router.get("/student/my", response_model=list[TaskRead])
def get_my_assigned_tasks(
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student),
):
    return get_tasks_for_student(
        db=db,
        student=current_student,
    )


@router.get("/{task_id}", response_model=TaskRead)
def get_my_task(
    task_id: int,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return get_teacher_task_response_or_404(
        db=db,
        teacher=current_teacher,
        task_id=task_id,
    )


@router.patch("/{task_id}", response_model=TaskRead)
def update_my_task(
    task_id: int,
    task_data: TaskUpdate,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return update_teacher_task(
        db=db,
        teacher=current_teacher,
        task_id=task_id,
        task_data=task_data,
    )


@router.post("/{task_id}/assign-students", response_model=TaskRead)
def assign_students_to_task(
    task_id: int,
    assignment_data: TaskAssignmentUpdate,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return replace_task_assignments(
        db=db,
        teacher=current_teacher,
        task_id=task_id,
        student_ids=assignment_data.student_ids,
    )


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_my_task(
    task_id: int,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    delete_teacher_task(
        db=db,
        teacher=current_teacher,
        task_id=task_id,
    )