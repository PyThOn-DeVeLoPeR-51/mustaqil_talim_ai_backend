from datetime import datetime, timedelta, timezone

from jose import jwt
from passlib.hash import bcrypt_sha256

from app.core.config import settings


def get_password_hash(password: str) -> str:
    return bcrypt_sha256.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt_sha256.verify(plain_password, hashed_password)


def create_access_token(
    subject: str,
    role: str = "teacher",
    expires_delta: timedelta | None = None,
) -> str:
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    expire = datetime.now(timezone.utc) + expires_delta

    to_encode = {
        "sub": str(subject),
        "role": role,
        "exp": expire,
    }

    return jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )