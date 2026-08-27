"""add durable Drawing AI evaluation jobs

Revision ID: b7f1c2d3e4a5
Revises: a6c9f1d2e3b4
Create Date: 2026-08-27
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b7f1c2d3e4a5"
down_revision: Union[str, Sequence[str], None] = "a6c9f1d2e3b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "drawing_evaluation_jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("submission_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("progress_percent", sa.Integer(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("locked_by", sa.String(length=255), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_drawing_evaluation_job_status",
        ),
        sa.CheckConstraint(
            "progress_percent >= 0 AND progress_percent <= 100",
            name="ck_drawing_evaluation_job_progress",
        ),
        sa.CheckConstraint(
            "attempts >= 0",
            name="ck_drawing_evaluation_job_attempts",
        ),
        sa.CheckConstraint(
            "max_attempts >= 1",
            name="ck_drawing_evaluation_job_max_attempts",
        ),
        sa.ForeignKeyConstraint(
            ["submission_id"],
            ["submissions.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "submission_id",
            name="uq_drawing_evaluation_job_submission",
        ),
    )
    op.create_index(
        op.f("ix_drawing_evaluation_jobs_id"),
        "drawing_evaluation_jobs",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_drawing_evaluation_jobs_status"),
        "drawing_evaluation_jobs",
        ["status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_drawing_evaluation_jobs_available_at"),
        "drawing_evaluation_jobs",
        ["available_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_drawing_evaluation_jobs_submission_id"),
        "drawing_evaluation_jobs",
        ["submission_id"],
        unique=True,
    )

    # Deploy vaqtida oldindan pending bo'lib qolgan submissionlar ham queue'da
    # yo'qolib ketmasin.
    op.execute(
        """
        INSERT INTO drawing_evaluation_jobs
            (submission_id, status, progress_percent, attempts, max_attempts, available_at, created_at, updated_at)
        SELECT
            id, 'pending', 0, 0, 3, now(), now(), now()
        FROM submissions
        WHERE status = 'pending'
        ON CONFLICT (submission_id) DO NOTHING
        """
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_drawing_evaluation_jobs_submission_id"),
        table_name="drawing_evaluation_jobs",
    )
    op.drop_index(
        op.f("ix_drawing_evaluation_jobs_available_at"),
        table_name="drawing_evaluation_jobs",
    )
    op.drop_index(
        op.f("ix_drawing_evaluation_jobs_status"),
        table_name="drawing_evaluation_jobs",
    )
    op.drop_index(
        op.f("ix_drawing_evaluation_jobs_id"),
        table_name="drawing_evaluation_jobs",
    )
    op.drop_table("drawing_evaluation_jobs")
