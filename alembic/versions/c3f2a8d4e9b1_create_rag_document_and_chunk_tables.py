"""create rag document and chunk tables

Revision ID: c3f2a8d4e9b1
Revises: 8d4063b0ab9f
Create Date: 2026-07-24

"""
from typing import Sequence, Union

from alembic import op
from pgvector.sqlalchemy import VECTOR
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c3f2a8d4e9b1"
down_revision: Union[str, Sequence[str], None] = "8d4063b0ab9f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # pgvector PostgreSQL extensioni. Render PostgreSQL ham pgvector'ni qo‘llab-quvvatlaydi.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "rag_documents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("teacher_id", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("original_filename", sa.String(length=500), nullable=False),
        sa.Column("stored_file_path", sa.String(length=1000), nullable=False),
        sa.Column("file_type", sa.String(length=20), nullable=False),
        sa.Column("mime_type", sa.String(length=150), nullable=True),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("processing_error", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "file_type IN ('pdf', 'docx')",
            name="ck_rag_document_file_type",
        ),
        sa.CheckConstraint(
            "status IN ('uploaded', 'processing', 'ready', 'failed', 'archived')",
            name="ck_rag_document_status",
        ),
        sa.CheckConstraint(
            "file_size_bytes IS NULL OR file_size_bytes >= 0",
            name="ck_rag_document_file_size",
        ),
        sa.CheckConstraint(
            "page_count IS NULL OR page_count >= 0",
            name="ck_rag_document_page_count",
        ),
        sa.CheckConstraint("chunk_count >= 0", name="ck_rag_document_chunk_count"),
        sa.ForeignKeyConstraint(["teacher_id"], ["teachers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_rag_documents_id"), "rag_documents", ["id"], unique=False)
    op.create_index(
        op.f("ix_rag_documents_teacher_id"),
        "rag_documents",
        ["teacher_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_documents_task_id"),
        "rag_documents",
        ["task_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_documents_file_type"),
        "rag_documents",
        ["file_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_documents_checksum_sha256"),
        "rag_documents",
        ["checksum_sha256"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_documents_status"),
        "rag_documents",
        ["status"],
        unique=False,
    )

    op.create_table(
        "rag_chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("section_title", sa.String(length=500), nullable=True),
        sa.Column("page_number_start", sa.Integer(), nullable=True),
        sa.Column("page_number_end", sa.Integer(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=True),
        sa.Column("char_end", sa.Integer(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("char_count", sa.Integer(), nullable=True),
        # O‘lcham model tanlanguncha erkin. Keyin modelga mos HNSW indeks qo‘shiladi.
        sa.Column("embedding", VECTOR(), nullable=True),
        sa.Column("embedding_model", sa.String(length=255), nullable=True),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint("chunk_index >= 0", name="ck_rag_chunk_index"),
        sa.CheckConstraint(
            "page_number_start IS NULL OR page_number_start >= 1",
            name="ck_rag_chunk_page_start",
        ),
        sa.CheckConstraint(
            "page_number_end IS NULL OR page_number_end >= 1",
            name="ck_rag_chunk_page_end",
        ),
        sa.CheckConstraint(
            "page_number_start IS NULL OR page_number_end IS NULL OR "
            "page_number_end >= page_number_start",
            name="ck_rag_chunk_page_range",
        ),
        sa.CheckConstraint(
            "char_start IS NULL OR char_start >= 0",
            name="ck_rag_chunk_char_start",
        ),
        sa.CheckConstraint(
            "char_end IS NULL OR char_end >= 0",
            name="ck_rag_chunk_char_end",
        ),
        sa.CheckConstraint(
            "char_start IS NULL OR char_end IS NULL OR char_end >= char_start",
            name="ck_rag_chunk_char_range",
        ),
        sa.CheckConstraint(
            "token_count IS NULL OR token_count >= 0",
            name="ck_rag_chunk_token_count",
        ),
        sa.CheckConstraint(
            "char_count IS NULL OR char_count >= 0",
            name="ck_rag_chunk_char_count",
        ),
        sa.CheckConstraint(
            "embedding_dimensions IS NULL OR embedding_dimensions > 0",
            name="ck_rag_chunk_embedding_dimensions",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["rag_documents.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_rag_chunk_document_index",
        ),
    )
    op.create_index(op.f("ix_rag_chunks_id"), "rag_chunks", ["id"], unique=False)
    op.create_index(
        op.f("ix_rag_chunks_document_id"),
        "rag_chunks",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_chunks_content_hash"),
        "rag_chunks",
        ["content_hash"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_chunks_embedding_model"),
        "rag_chunks",
        ["embedding_model"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_rag_chunks_embedding_model"), table_name="rag_chunks")
    op.drop_index(op.f("ix_rag_chunks_content_hash"), table_name="rag_chunks")
    op.drop_index(op.f("ix_rag_chunks_document_id"), table_name="rag_chunks")
    op.drop_index(op.f("ix_rag_chunks_id"), table_name="rag_chunks")
    op.drop_table("rag_chunks")

    op.drop_index(op.f("ix_rag_documents_status"), table_name="rag_documents")
    op.drop_index(op.f("ix_rag_documents_checksum_sha256"), table_name="rag_documents")
    op.drop_index(op.f("ix_rag_documents_file_type"), table_name="rag_documents")
    op.drop_index(op.f("ix_rag_documents_task_id"), table_name="rag_documents")
    op.drop_index(op.f("ix_rag_documents_teacher_id"), table_name="rag_documents")
    op.drop_index(op.f("ix_rag_documents_id"), table_name="rag_documents")
    op.drop_table("rag_documents")

    # vector extensionini ataylab o‘chirmaymiz: kelajakda boshqa funksiyalar ham ishlatishi mumkin.
