from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import get_password_hash, verify_password
from app.db.database import get_db
from app.models.student import Student
from app.models.teacher import Teacher
from app.schemas.teacher import TeacherCreate


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_PREFIX}/auth/teacher/login"
)


def get_teacher_by_email(db: Session, email: str) -> Teacher | None:
    return db.query(Teacher).filter(Teacher.email == email).first()


def get_teacher_by_id(db: Session, teacher_id: int) -> Teacher | None:
    return db.query(Teacher).filter(Teacher.id == teacher_id).first()


def get_student_by_id(db: Session, student_id: int) -> Student | None:
    return db.query(Student).filter(Student.id == student_id).first()


def get_student_by_login(db: Session, login: str) -> Student | None:
    return db.query(Student).filter(Student.login == login).first()


def create_teacher(db: Session, teacher_data: TeacherCreate) -> Teacher:
    existing_teacher = get_teacher_by_email(db, teacher_data.email)

    if existing_teacher:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bu email orqali o‘qituvchi avval ro‘yxatdan o‘tgan.",
        )

    teacher = Teacher(
        first_name=teacher_data.first_name,
        last_name=teacher_data.last_name,
        email=teacher_data.email,
        password_hash=get_password_hash(teacher_data.password),
        position=teacher_data.position,
        university=teacher_data.university,
    )

    db.add(teacher)
    db.commit()
    db.refresh(teacher)

    return teacher


def authenticate_teacher(
    db: Session,
    email: str,
    password: str,
) -> Teacher | None:
    teacher = get_teacher_by_email(db, email)

    if not teacher:
        return None

    if not verify_password(password, teacher.password_hash):
        return None

    return teacher


def authenticate_student(
    db: Session,
    login: str,
    password: str,
) -> Student | None:
    student = get_student_by_login(db, login)

    if not student:
        return None

    if not student.is_active:
        return None

    if not verify_password(password, student.password_hash):
        return None

    return student


def get_current_teacher(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Teacher:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token noto‘g‘ri yoki muddati tugagan.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )

        teacher_id: str | None = payload.get("sub")
        role: str | None = payload.get("role")

        if teacher_id is None or role != "teacher":
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    teacher = get_teacher_by_id(db, int(teacher_id))

    if teacher is None:
        raise credentials_exception

    return teacher


def get_current_student(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Student:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token noto‘g‘ri yoki muddati tugagan.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )

        student_id: str | None = payload.get("sub")
        role: str | None = payload.get("role")

        if student_id is None or role != "student":
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    student = get_student_by_id(db, int(student_id))

    if student is None or not student.is_active:
        raise credentials_exception

    return student