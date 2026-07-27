"""add RAG multilingual E5 HNSW index

Revision ID: d8a4f9c1e2b7
Revises: c3f2a8d4e9b1
"""
from typing import Sequence, Union

from alembic import op


revision: str = "d8a4f9c1e2b7"
down_revision: Union[str, Sequence[str], None] = "c3f2a8d4e9b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


INDEX_NAME = "ix_rag_chunks_e5_small_embedding_hnsw"
MODEL_NAME = "intfloat/multilingual-e5-small"
DIMENSIONS = 384


def upgrade() -> None:
    # embedding ustuni VECTOR() bo‘lib qoladi: kelajakda boshqa o‘lchamli modelni
    # ham saqlash mumkin. HNSW esa aynan tanlangan model/dimension uchun partial
    # expression index sifatida quriladi.
    op.execute(
        f'''\n        CREATE INDEX {INDEX_NAME}\n        ON rag_chunks\n        USING hnsw ((embedding::vector({DIMENSIONS})) vector_cosine_ops)\n        WHERE embedding_model = '{MODEL_NAME}'\n          AND embedding_dimensions = {DIMENSIONS}\n          AND embedding IS NOT NULL\n        '''
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
