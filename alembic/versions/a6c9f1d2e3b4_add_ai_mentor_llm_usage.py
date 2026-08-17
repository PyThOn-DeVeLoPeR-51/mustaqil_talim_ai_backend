"""add ai mentor llm usage

Revision ID: a6c9f1d2e3b4
Revises: e4b7c2d9a6f1
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a6c9f1d2e3b4"
down_revision: Union[str, Sequence[str], None] = "e4b7c2d9a6f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ai_mentor_llm_usage",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("student_id", sa.Integer(), nullable=False),
        sa.Column("chat_session_id", sa.Integer(), nullable=True),
        sa.Column("feature", sa.String(length=30), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("model_name", sa.String(length=150), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("request_id", sa.String(length=255), nullable=True),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("fallback_from_provider", sa.String(length=50), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("feature IN ('chat', 'plan', 'diagnostic')", name="ck_ai_llm_usage_feature"),
        sa.CheckConstraint("status IN ('success', 'error', 'rate_limited', 'skipped_quota')", name="ck_ai_llm_usage_status"),
        sa.CheckConstraint("input_tokens IS NULL OR input_tokens >= 0", name="ck_ai_llm_usage_input_tokens"),
        sa.CheckConstraint("output_tokens IS NULL OR output_tokens >= 0", name="ck_ai_llm_usage_output_tokens"),
        sa.CheckConstraint("total_tokens IS NULL OR total_tokens >= 0", name="ck_ai_llm_usage_total_tokens"),
        sa.ForeignKeyConstraint(["student_id"], ["students.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chat_session_id"], ["ai_mentor_chat_sessions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    for name, cols in [
        ("ix_ai_mentor_llm_usage_id", ["id"]), ("ix_ai_mentor_llm_usage_student_id", ["student_id"]),
        ("ix_ai_mentor_llm_usage_chat_session_id", ["chat_session_id"]), ("ix_ai_mentor_llm_usage_feature", ["feature"]),
        ("ix_ai_mentor_llm_usage_provider", ["provider"]), ("ix_ai_mentor_llm_usage_status", ["status"]),
        ("ix_ai_mentor_llm_usage_error_code", ["error_code"]), ("ix_ai_mentor_llm_usage_created_at", ["created_at"]),
    ]:
        op.create_index(name, "ai_mentor_llm_usage", cols, unique=False)
    op.create_index("ix_ai_llm_usage_student_provider_feature_created", "ai_mentor_llm_usage",
                    ["student_id", "provider", "feature", "created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_ai_llm_usage_student_provider_feature_created", table_name="ai_mentor_llm_usage")
    for name in ["ix_ai_mentor_llm_usage_created_at", "ix_ai_mentor_llm_usage_error_code",
                 "ix_ai_mentor_llm_usage_status", "ix_ai_mentor_llm_usage_provider",
                 "ix_ai_mentor_llm_usage_feature", "ix_ai_mentor_llm_usage_chat_session_id",
                 "ix_ai_mentor_llm_usage_student_id", "ix_ai_mentor_llm_usage_id"]:
        op.drop_index(name, table_name="ai_mentor_llm_usage")
    op.drop_table("ai_mentor_llm_usage")
