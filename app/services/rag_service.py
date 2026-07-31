from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
import shutil
import uuid
import zipfile

from fastapi import HTTPException, UploadFile, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.rag import RAGChunk, RAGDocument
from app.models.task import Task
from app.models.teacher import Teacher
from app.rag.chunker import chunk_blocks
from app.rag.extractors import DocumentExtractionError, extract_document


ALLOWED_RAG_EXTENSIONS = {".pdf": "pdf", ".docx": "docx"}
BUFFER_SIZE = 1024 * 1024
logger = logging.getLogger("app.rag.service")


class RAGProcessingError(RuntimeError):
    def __init__(self, message: str, document_id: int | None = None):
        super().__init__(message)
        self.document_id = document_id


@dataclass(frozen=True)
class StoredRAGFile:
    path: Path
    original_filename: str
    file_type: str
    mime_type: str | None
    size_bytes: int
    checksum_sha256: str


def get_teacher_storage_usage_values(db: Session, teacher_id: int) -> dict[str, int | float | None]:
    used_bytes, document_count = (
        db.query(
            func.coalesce(func.sum(RAGDocument.file_size_bytes), 0),
            func.count(RAGDocument.id),
        )
        .filter(
            RAGDocument.teacher_id == teacher_id,
            RAGDocument.status != "archived",
        )
        .one()
    )
    used = int(used_bytes or 0)
    count = int(document_count or 0)
    limit_bytes = (
        int(settings.RAG_TEACHER_STORAGE_LIMIT_MB * 1024 * 1024)
        if settings.RAG_TEACHER_STORAGE_LIMIT_MB > 0
        else None
    )
    document_limit = (
        int(settings.RAG_TEACHER_DOCUMENT_LIMIT)
        if settings.RAG_TEACHER_DOCUMENT_LIMIT > 0
        else None
    )
    remaining = max(limit_bytes - used, 0) if limit_bytes is not None else None
    usage_percent = (
        round((used / limit_bytes) * 100, 2)
        if limit_bytes and limit_bytes > 0
        else None
    )
    return {
        "teacher_id": teacher_id,
        "document_count": count,
        "document_limit": document_limit,
        "used_bytes": used,
        "limit_bytes": limit_bytes,
        "remaining_bytes": remaining,
        "usage_percent": usage_percent,
    }


def _validate_teacher_document_limit(db: Session, teacher_id: int) -> None:
    usage = get_teacher_storage_usage_values(db, teacher_id)
    limit = usage["document_limit"]
    if limit is not None and int(usage["document_count"]) >= int(limit):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"RAG hujjatlari limiti tugagan: {limit} ta.",
        )


def _validate_teacher_storage_quota(
    db: Session, teacher_id: int, incoming_size_bytes: int
) -> None:
    usage = get_teacher_storage_usage_values(db, teacher_id)
    limit = usage["limit_bytes"]
    if limit is not None and int(usage["used_bytes"]) + incoming_size_bytes > int(limit):
        raise HTTPException(
            status_code=507,
            detail={
                "message": "O‘qituvchi uchun RAG storage limiti tugagan.",
                "used_bytes": usage["used_bytes"],
                "incoming_bytes": incoming_size_bytes,
                "limit_bytes": limit,
            },
        )


def _validate_disk_free_space(storage_root: Path) -> None:
    storage_root.mkdir(parents=True, exist_ok=True)
    min_free = int(settings.RAG_STORAGE_MIN_FREE_MB * 1024 * 1024)
    if min_free <= 0:
        return
    free = shutil.disk_usage(storage_root).free
    if free < min_free:
        raise HTTPException(
            status_code=507,
            detail={
                "message": "Server diskida RAG fayllari uchun yetarli bo‘sh joy yo‘q.",
                "free_bytes": free,
                "required_free_bytes": min_free,
            },
        )


def _validate_task_ownership(db: Session, teacher: Teacher, task_id: int | None) -> None:
    if task_id is None:
        return
    task = (
        db.query(Task)
        .filter(Task.id == task_id, Task.teacher_id == teacher.id)
        .first()
    )
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Topshiriq topilmadi yoki sizga tegishli emas.",
        )


def _validate_file_signature(path: Path, file_type: str) -> None:
    if file_type == "pdf":
        with path.open("rb") as source:
            if source.read(5) != b"%PDF-":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Fayl kengaytmasi PDF, ammo fayl mazmuni haqiqiy PDF emas.",
                )
        return

    if file_type == "docx":
        try:
            if not zipfile.is_zipfile(path):
                raise ValueError
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                    raise ValueError
        except (OSError, zipfile.BadZipFile, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Fayl kengaytmasi DOCX, ammo fayl mazmuni haqiqiy DOCX emas.",
            ) from exc
        return

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Faqat PDF yoki DOCX fayl yuklash mumkin.",
    )


def save_rag_upload(file: UploadFile, teacher_id: int) -> StoredRAGFile:
    original_filename = Path(file.filename or "").name.strip()
    extension = Path(original_filename).suffix.lower()
    file_type = ALLOWED_RAG_EXTENSIONS.get(extension)
    if not original_filename or file_type is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Faqat PDF yoki DOCX fayl yuklash mumkin.",
        )

    max_bytes = int(settings.RAG_MAX_FILE_SIZE_MB * 1024 * 1024)
    storage_root = Path(settings.RAG_STORAGE_DIR)
    _validate_disk_free_space(storage_root)
    storage_dir = storage_root / str(teacher_id)
    storage_dir.mkdir(parents=True, exist_ok=True)
    final_path = storage_dir / f"{uuid.uuid4().hex}{extension}"
    temp_path = final_path.with_suffix(final_path.suffix + ".part")

    digest = hashlib.sha256()
    total = 0

    try:
        file.file.seek(0)
        with temp_path.open("wb") as target:
            while True:
                chunk = file.file.read(BUFFER_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"RAG fayli {settings.RAG_MAX_FILE_SIZE_MB} MB dan katta bo‘lishi mumkin emas.",
                    )
                digest.update(chunk)
                target.write(chunk)

        if total == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Bo‘sh fayl yuklab bo‘lmaydi.",
            )

        _validate_file_signature(temp_path, file_type)
        temp_path.replace(final_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise
    finally:
        try:
            file.file.seek(0)
        except Exception:
            pass

    return StoredRAGFile(
        path=final_path,
        original_filename=original_filename,
        file_type=file_type,
        mime_type=file.content_type,
        size_bytes=total,
        checksum_sha256=digest.hexdigest(),
    )


def get_teacher_document_or_404(
    db: Session,
    teacher: Teacher,
    document_id: int,
) -> RAGDocument:
    document = (
        db.query(RAGDocument)
        .filter(
            RAGDocument.id == document_id,
            RAGDocument.teacher_id == teacher.id,
        )
        .first()
    )
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="RAG hujjati topilmadi.",
        )
    return document


def _mark_processing_failed(db: Session, document_id: int, message: str) -> None:
    db.rollback()
    document = db.query(RAGDocument).filter(RAGDocument.id == document_id).first()
    if document is None:
        return
    document.status = "failed"
    document.processing_error = message[:4000]
    document.processed_at = datetime.now(timezone.utc)
    db.commit()


def process_rag_document(db: Session, document: RAGDocument) -> RAGDocument:
    document.status = "processing"
    document.processing_error = None
    db.commit()
    db.refresh(document)

    try:
        extracted = extract_document(document.stored_file_path, document.file_type)
        chunks = chunk_blocks(
            extracted.blocks,
            chunk_size=settings.RAG_CHUNK_SIZE_CHARS,
            overlap=settings.RAG_CHUNK_OVERLAP_CHARS,
        )
        if not chunks:
            raise DocumentExtractionError("Hujjatdan RAG uchun yetarli matn olinmadi.")

        db.query(RAGChunk).filter(RAGChunk.document_id == document.id).delete(
            synchronize_session=False
        )

        for chunk in chunks:
            db.add(
                RAGChunk(
                    document_id=document.id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    content_hash=chunk.content_hash,
                    section_title=chunk.section_title,
                    page_number_start=chunk.page_number_start,
                    page_number_end=chunk.page_number_end,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    token_count=None,
                    char_count=chunk.char_count,
                    embedding=None,
                    embedding_model=None,
                    embedding_dimensions=None,
                    metadata_json={"ingestion_version": 3, "chunking_strategy": "sentence_aware_overlap_v3"},
                )
            )

        metadata = dict(document.metadata_json or {})
        metadata.update(extracted.metadata)
        # Reprocess qilinganda eski embedding metadata endi haqiqiy bo‘lmaydi,
        # chunki chunklar qayta yaratiladi va embedding=None bo‘ladi.
        for key in (
            "embedding_provider",
            "embedding_model",
            "embedding_dimensions",
            "embedded_chunk_count",
            "embedded_at",
        ):
            metadata.pop(key, None)

        metadata.update(
            {
                "chunk_size_chars": settings.RAG_CHUNK_SIZE_CHARS,
                "chunk_overlap_chars": settings.RAG_CHUNK_OVERLAP_CHARS,
                "chunking_strategy": "sentence_aware_overlap_v3",
            }
        )

        document.status = "ready"
        document.page_count = extracted.page_count
        document.chunk_count = len(chunks)
        document.processing_error = None
        document.metadata_json = metadata
        document.processed_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(document)
        return document
    except Exception as exc:
        message = str(exc) or exc.__class__.__name__
        _mark_processing_failed(db, document.id, message)
        raise RAGProcessingError(message, document.id) from exc


def create_rag_document_record(
    db: Session,
    teacher: Teacher,
    *,
    title: str,
    task_id: int | None,
    file: UploadFile,
) -> RAGDocument:
    """Faylni xavfsiz saqlaydi va uploaded holatidagi DB yozuvini yaratadi."""

    clean_title = title.strip()
    if len(clean_title) < 3 or len(clean_title) > 255:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="title 3–255 belgi oralig‘ida bo‘lishi kerak.",
        )

    _validate_task_ownership(db, teacher, task_id)
    _validate_teacher_document_limit(db, teacher.id)
    stored = save_rag_upload(file, teacher.id)

    try:
        duplicate = (
            db.query(RAGDocument)
            .filter(
                RAGDocument.teacher_id == teacher.id,
                RAGDocument.task_id == task_id,
                RAGDocument.checksum_sha256 == stored.checksum_sha256,
                RAGDocument.status != "archived",
            )
            .first()
        )
        if duplicate is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Bu fayl avval yuklangan (document_id={duplicate.id}).",
            )

        _validate_teacher_storage_quota(db, teacher.id, stored.size_bytes)

        document = RAGDocument(
            teacher_id=teacher.id,
            task_id=task_id,
            title=clean_title,
            original_filename=stored.original_filename,
            stored_file_path=str(stored.path),
            file_type=stored.file_type,
            mime_type=stored.mime_type,
            file_size_bytes=stored.size_bytes,
            checksum_sha256=stored.checksum_sha256,
            status="uploaded",
            chunk_count=0,
            metadata_json={"storage": "private_local"},
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        logger.info(
            "RAG document stored",
            extra={
                "document_id": document.id,
                "teacher_id": teacher.id,
                "file_size_bytes": stored.size_bytes,
            },
        )
        return document
    except Exception:
        db.rollback()
        stored.path.unlink(missing_ok=True)
        raise


def create_and_process_rag_document(
    db: Session,
    teacher: Teacher,
    *,
    title: str,
    task_id: int | None,
    file: UploadFile,
) -> RAGDocument:
    document = create_rag_document_record(
        db, teacher, title=title, task_id=task_id, file=file
    )
    return process_rag_document(db, document)


def reprocess_teacher_document(
    db: Session,
    teacher: Teacher,
    document_id: int,
) -> RAGDocument:
    document = get_teacher_document_or_404(db, teacher, document_id)
    if not Path(document.stored_file_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Hujjatning saqlangan fayli topilmadi.",
        )
    return process_rag_document(db, document)


def update_teacher_document(
    db: Session,
    teacher: Teacher,
    document_id: int,
    *,
    title: str | None = None,
    task_id: int | None = None,
    update_title: bool = False,
    update_task_id: bool = False,
) -> RAGDocument:
    document = get_teacher_document_or_404(db, teacher, document_id)

    if update_title:
        clean_title = (title or "").strip()
        if len(clean_title) < 3 or len(clean_title) > 255:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="title 3–255 belgi oralig‘ida bo‘lishi kerak.",
            )
        document.title = clean_title

    if update_task_id:
        _validate_task_ownership(db, teacher, task_id)
        duplicate = (
            db.query(RAGDocument)
            .filter(
                RAGDocument.id != document.id,
                RAGDocument.teacher_id == teacher.id,
                RAGDocument.task_id == task_id,
                RAGDocument.checksum_sha256 == document.checksum_sha256,
                RAGDocument.status != "archived",
            )
            .first()
        )
        if duplicate is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Bu fayl tanlangan topshiriq uchun avval yuklangan (document_id={duplicate.id}).",
            )
        document.task_id = task_id

    db.commit()
    db.refresh(document)
    return document


def get_teacher_documents(
    db: Session,
    teacher: Teacher,
    *,
    task_id: int | None = None,
    document_status: str | None = None,
) -> list[RAGDocument]:
    query = db.query(RAGDocument).filter(RAGDocument.teacher_id == teacher.id)
    if task_id is not None:
        query = query.filter(RAGDocument.task_id == task_id)
    if document_status is not None:
        if document_status not in {"uploaded", "processing", "ready", "failed", "archived"}:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Noto‘g‘ri RAG document status.",
            )
        query = query.filter(RAGDocument.status == document_status)
    return query.order_by(RAGDocument.id.desc()).all()


def get_document_chunks(
    db: Session,
    teacher: Teacher,
    document_id: int,
) -> list[RAGChunk]:
    get_teacher_document_or_404(db, teacher, document_id)
    return (
        db.query(RAGChunk)
        .filter(RAGChunk.document_id == document_id)
        .order_by(RAGChunk.chunk_index.asc())
        .all()
    )


def delete_teacher_document(
    db: Session,
    teacher: Teacher,
    document_id: int,
) -> None:
    document = get_teacher_document_or_404(db, teacher, document_id)
    stored_path = Path(document.stored_file_path)
    db.delete(document)
    db.commit()
    stored_path.unlink(missing_ok=True)
    # teacher papkasi bo‘sh qolgan bo‘lsa tozalashga urinib ko‘ramiz.
    try:
        stored_path.parent.rmdir()
    except OSError:
        pass
