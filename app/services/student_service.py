import secrets

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import get_password_hash
from app.models.student import Student
from app.models.teacher import Teacher
from app.schemas.student import StudentCreate, StudentUpdate


def generate_student_login(db: Session) -> str:
    while True:
        login = f"student_{secrets.token_hex(4)}"
        exists = db.query(Student).filter(Student.login == login).first()
        if not exists:
            return login


def generate_student_password() -> str:
    return secrets.token_urlsafe(8)


def get_student_by_id(db: Session, student_id: int) -> Student | None:
    return db.query(Student).filter(Student.id == student_id).first()


def get_student_by_login(db: Session, login: str) -> Student | None:
    return db.query(Student).filter(Student.login == login).first()


def create_student_for_teacher(
    db: Session,
    teacher: Teacher,
    student_data: StudentCreate,
) -> tuple[Student, str]:
    login = student_data.login or generate_student_login(db)
    plain_password = student_data.password or generate_student_password()

    existing_student = get_student_by_login(db, login)

    if existing_student:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bu login band. Boshqa login kiriting.",
        )

    student = Student(
        teacher_id=teacher.id,
        full_name=student_data.full_name,
        university=student_data.university,
        direction=student_data.direction,
        stage=student_data.stage,
        login=login,
        password_hash=get_password_hash(plain_password),
        is_active=True,
    )

    db.add(student)
    db.commit()
    db.refresh(student)

    return student, plain_password


def get_students_for_teacher(
    db: Session,
    teacher: Teacher,
) -> list[Student]:
    return (
        db.query(Student)
        .filter(Student.teacher_id == teacher.id)
        .order_by(Student.id.desc())
        .all()
    )


def get_teacher_student_or_404(
    db: Session,
    teacher: Teacher,
    student_id: int,
) -> Student:
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

    return student


def update_teacher_student(
    db: Session,
    teacher: Teacher,
    student_id: int,
    student_data: StudentUpdate,
) -> Student:
    student = get_teacher_student_or_404(
        db=db,
        teacher=teacher,
        student_id=student_id,
    )

    update_data = student_data.model_dump(exclude_unset=True)

    for key, value in update_data.items():
        setattr(student, key, value)

    db.commit()
    db.refresh(student)

    return student


def delete_teacher_student(
    db: Session,
    teacher: Teacher,
    student_id: int,
) -> None:
    student = get_teacher_student_or_404(
        db=db,
        teacher=teacher,
        student_id=student_id,
    )

    db.delete(student)
    db.commit()