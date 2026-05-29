"""add instruction file path to tasks

Revision ID: b12c9b5a1d33
Revises: 787bbcaac536
Create Date: 2026-05-16 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b12c9b5a1d33"
down_revision: Union[str, None] = "787bbcaac536"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "tasks",
        sa.Column("instruction_file_path", sa.String(length=500), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("tasks", "instruction_file_path")
