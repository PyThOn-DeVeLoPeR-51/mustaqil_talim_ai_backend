"""add durable RAG processing jobs

Revision ID: e4b7c2d9a6f1
Revises: d8a4f9c1e2b7
Create Date: 2026-07-31
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e4b7c2d9a6f1"
down_revision: Union[str, Sequence[str], None] = "d8a4f9c1e2b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "rag_processing_jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("teacher_id", sa.Integer(), nullable=False),
        sa.Column("job_type", sa.String(length=30), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("progress_percent", sa.Integer(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=True),
        sa.Column("result_json", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("locked_by", sa.String(length=255), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.CheckConstraint(
            "job_type IN ('ingest', 'reprocess', 'embed')",
            name="ck_rag_processing_job_type",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_rag_processing_job_status",
        ),
        sa.CheckConstraint(
            "progress_percent >= 0 AND progress_percent <= 100",
            name="ck_rag_processing_job_progress",
        ),
        sa.CheckConstraint("attempts >= 0", name="ck_rag_processing_job_attempts"),
        sa.CheckConstraint("max_attempts >= 1", name="ck_rag_processing_job_max_attempts"),
        sa.ForeignKeyConstraint(
            ["document_id"], ["rag_documents.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["teacher_id"], ["teachers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_rag_processing_jobs_id"),
        "rag_processing_jobs",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_processing_jobs_document_id"),
        "rag_processing_jobs",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_processing_jobs_teacher_id"),
        "rag_processing_jobs",
        ["teacher_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_processing_jobs_job_type"),
        "rag_processing_jobs",
        ["job_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_processing_jobs_status"),
        "rag_processing_jobs",
        ["status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_rag_processing_jobs_available_at"),
        "rag_processing_jobs",
        ["available_at"],
        unique=False,
    )
    op.create_index(
        "ix_rag_processing_jobs_queue",
        "rag_processing_jobs",
        ["status", "available_at", "id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_rag_processing_jobs_queue", table_name="rag_processing_jobs")
    op.drop_index(op.f("ix_rag_processing_jobs_available_at"), table_name="rag_processing_jobs")
    op.drop_index(op.f("ix_rag_processing_jobs_status"), table_name="rag_processing_jobs")
    op.drop_index(op.f("ix_rag_processing_jobs_job_type"), table_name="rag_processing_jobs")
    op.drop_index(op.f("ix_rag_processing_jobs_teacher_id"), table_name="rag_processing_jobs")
    op.drop_index(op.f("ix_rag_processing_jobs_document_id"), table_name="rag_processing_jobs")
    op.drop_index(op.f("ix_rag_processing_jobs_id"), table_name="rag_processing_jobs")
    op.drop_table("rag_processing_jobs")
