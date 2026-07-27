from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class RAGDocument(Base):
    """O‘qituvchi yuklagan va RAG uchun qayta ishlanadigan o‘quv materiali."""

    __tablename__ = "rag_documents"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    teacher_id: Mapped[int] = mapped_column(
        ForeignKey("teachers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    task_id: Mapped[int | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    title: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    stored_file_path: Mapped[str] = mapped_column(String(1000), nullable=False)

    # Hozirgi birinchi versiyada PDF va DOCX bilan boshlaymiz.
    file_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    mime_type: Mapped[str | None] = mapped_column(String(150), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    checksum_sha256: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    # uploaded | processing | ready | failed | archived
    status: Mapped[str] = mapped_column(
        String(30),
        default="uploaded",
        nullable=False,
        index=True,
    )

    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processing_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    __table_args__ = (
        CheckConstraint(
            "file_type IN ('pdf', 'docx')",
            name="ck_rag_document_file_type",
        ),
        CheckConstraint(
            "status IN ('uploaded', 'processing', 'ready', 'failed', 'archived')",
            name="ck_rag_document_status",
        ),
        CheckConstraint(
            "file_size_bytes IS NULL OR file_size_bytes >= 0",
            name="ck_rag_document_file_size",
        ),
        CheckConstraint(
            "page_count IS NULL OR page_count >= 0",
            name="ck_rag_document_page_count",
        ),
        CheckConstraint(
            "chunk_count >= 0",
            name="ck_rag_document_chunk_count",
        ),
    )


class RAGChunk(Base):
    """Hujjatdan ajratilgan qidiriladigan matn bo‘lagi va uning embeddingi."""

    __tablename__ = "rag_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    document_id: Mapped[int] = mapped_column(
        ForeignKey("rag_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    section_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    page_number_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_number_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # O‘lchamni hozir qotirmaymiz. Keyingi bosqichda embedding model tanlangach,
    # embedding_model va embedding_dimensions bilan birga saqlanadi.
    embedding: Mapped[list[float] | None] = mapped_column(VECTOR(), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )
    embedding_dimensions: Mapped[int | None] = mapped_column(Integer, nullable=True)

    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_rag_chunk_document_index",
        ),
        CheckConstraint("chunk_index >= 0", name="ck_rag_chunk_index"),
        CheckConstraint(
            "page_number_start IS NULL OR page_number_start >= 1",
            name="ck_rag_chunk_page_start",
        ),
        CheckConstraint(
            "page_number_end IS NULL OR page_number_end >= 1",
            name="ck_rag_chunk_page_end",
        ),
        CheckConstraint(
            "page_number_start IS NULL OR page_number_end IS NULL OR "
            "page_number_end >= page_number_start",
            name="ck_rag_chunk_page_range",
        ),
        CheckConstraint(
            "char_start IS NULL OR char_start >= 0",
            name="ck_rag_chunk_char_start",
        ),
        CheckConstraint(
            "char_end IS NULL OR char_end >= 0",
            name="ck_rag_chunk_char_end",
        ),
        CheckConstraint(
            "char_start IS NULL OR char_end IS NULL OR char_end >= char_start",
            name="ck_rag_chunk_char_range",
        ),
        CheckConstraint(
            "token_count IS NULL OR token_count >= 0",
            name="ck_rag_chunk_token_count",
        ),
        CheckConstraint(
            "char_count IS NULL OR char_count >= 0",
            name="ck_rag_chunk_char_count",
        ),
        CheckConstraint(
            "embedding_dimensions IS NULL OR embedding_dimensions > 0",
            name="ck_rag_chunk_embedding_dimensions",
        ),
    )
