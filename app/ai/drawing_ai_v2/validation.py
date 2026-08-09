"""Input validation and bounded resource checks for drawing evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image, UnidentifiedImageError

from app.ai.drawing_ai_v2.exceptions import DrawingAIValidationError


SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".pdf"})


@dataclass(frozen=True, slots=True)
class ValidationLimits:
    max_file_size_bytes: int = 25 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    max_pdf_pages: int = 20


DEFAULT_LIMITS = ValidationLimits()


def _validate_magic(path: Path, suffix: str) -> None:
    with path.open("rb") as file_handle:
        header = file_handle.read(16)
    if suffix == ".pdf" and not header.startswith(b"%PDF-"):
        raise DrawingAIValidationError("PDF fayl signaturasi noto‘g‘ri.")
    if suffix == ".png" and not header.startswith(b"\x89PNG\r\n\x1a\n"):
        raise DrawingAIValidationError("PNG fayl signaturasi noto‘g‘ri.")
    if suffix in {".jpg", ".jpeg"} and not header.startswith(b"\xff\xd8\xff"):
        raise DrawingAIValidationError("JPEG fayl signaturasi noto‘g‘ri.")


def validate_drawing_file(
    path: str | Path,
    limits: ValidationLimits = DEFAULT_LIMITS,
) -> Path:
    file_path = Path(path)
    if not file_path.is_file():
        raise DrawingAIValidationError(f"Fayl topilmadi: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DrawingAIValidationError(
            "Faqat JPG, JPEG, PNG yoki PDF fayllar qo‘llab-quvvatlanadi."
        )

    file_size = file_path.stat().st_size
    if file_size <= 0:
        raise DrawingAIValidationError("Bo‘sh faylni baholab bo‘lmaydi.")
    if file_size > limits.max_file_size_bytes:
        max_mb = limits.max_file_size_bytes // (1024 * 1024)
        raise DrawingAIValidationError(f"Fayl hajmi {max_mb} MB dan oshmasligi kerak.")

    _validate_magic(file_path, suffix)

    if suffix == ".pdf":
        try:
            with fitz.open(file_path) as document:
                if document.needs_pass:
                    raise DrawingAIValidationError("Parol bilan himoyalangan PDF qo‘llab-quvvatlanmaydi.")
                if document.page_count < 1:
                    raise DrawingAIValidationError("PDF sahifalari topilmadi.")
                if document.page_count > limits.max_pdf_pages:
                    raise DrawingAIValidationError(
                        f"PDF {limits.max_pdf_pages} sahifadan oshmasligi kerak."
                    )
        except DrawingAIValidationError:
            raise
        except Exception as exc:
            raise DrawingAIValidationError(f"PDF faylni o‘qib bo‘lmadi: {exc}") from exc
    else:
        try:
            with Image.open(file_path) as image:
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise DrawingAIValidationError("Rasm o‘lchami noto‘g‘ri.")
                if width * height > limits.max_image_pixels:
                    raise DrawingAIValidationError(
                        "Rasm o‘lchami juda katta; kichikroq rezolyutsiya yuklang."
                    )
                image.verify()
        except DrawingAIValidationError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise DrawingAIValidationError(f"Rasm faylni o‘qib bo‘lmadi: {exc}") from exc

    return file_path
