import unittest

from app.db.base import Base


class RAGModelMetadataTestCase(unittest.TestCase):
    def test_rag_tables_are_registered(self) -> None:
        self.assertIn("rag_documents", Base.metadata.tables)
        self.assertIn("rag_chunks", Base.metadata.tables)
        self.assertIn("rag_processing_jobs", Base.metadata.tables)

    def test_embedding_column_is_dimension_flexible_vector(self) -> None:
        chunk_table = Base.metadata.tables["rag_chunks"]
        embedding_type = chunk_table.c.embedding.type

        self.assertEqual(embedding_type.__class__.__name__.upper(), "VECTOR")
        # pgvector VECTOR() o‘lchami qotirilmagan bo‘lishi kerak.
        dimensions = getattr(embedding_type, "dim", None)
        if dimensions is None:
            dimensions = getattr(embedding_type, "dimensions", None)
        self.assertIsNone(dimensions)

    def test_document_chunk_unique_constraint_exists(self) -> None:
        chunk_table = Base.metadata.tables["rag_chunks"]
        constraint_names = {
            constraint.name for constraint in chunk_table.constraints if constraint.name
        }
        self.assertIn("uq_rag_chunk_document_index", constraint_names)

    def test_processing_job_constraints_exist(self) -> None:
        job_table = Base.metadata.tables["rag_processing_jobs"]
        constraint_names = {
            constraint.name for constraint in job_table.constraints if constraint.name
        }
        self.assertIn("ck_rag_processing_job_type", constraint_names)
        self.assertIn("ck_rag_processing_job_status", constraint_names)
        self.assertIn("ck_rag_processing_job_progress", constraint_names)


if __name__ == "__main__":
    unittest.main()
