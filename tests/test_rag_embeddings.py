from __future__ import annotations

import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.rag import RAGChunk, RAGDocument
from app.models.teacher import Teacher
from app.models.student import Student
from app.models.task import Task, TaskAssignment
from app.rag.embeddings import EmbeddingProviderStatus, LocalONNXE5EmbeddingProvider
from app.schemas.rag import RAGSemanticSearchRequest
from app.services.rag_embedding_service import (
    embed_teacher_document,
    student_has_searchable_embeddings,
)


class FakeEmbeddingProvider:
    provider_name = "fake"
    model_name = "fake-multilingual"
    dimensions = 4

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0, float(index)] for index, _ in enumerate(texts)]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]


class RAGEmbeddingServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.db: Session = factory()

        self.teacher = Teacher(
            first_name="Embed",
            last_name="Teacher",
            email="embed-teacher@example.com",
            password_hash="hash",
        )
        self.db.add(self.teacher)
        self.db.commit()
        self.db.refresh(self.teacher)

        document = RAGDocument(
            teacher_id=self.teacher.id,
            title="Test material",
            original_filename="test.docx",
            stored_file_path="private/test.docx",
            file_type="docx",
            status="ready",
            chunk_count=2,
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        self.document = document

        self.db.add_all(
            [
                RAGChunk(document_id=document.id, chunk_index=0, content="Kredit modul tizimi"),
                RAGChunk(document_id=document.id, chunk_index=1, content="Mustaqil ta'lim faoliyati"),
            ]
        )
        self.db.commit()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_document_chunks_receive_embeddings_and_metadata(self) -> None:
        result = embed_teacher_document(
            self.db,
            self.teacher,
            self.document.id,
            provider=FakeEmbeddingProvider(),
        )
        self.assertEqual(result.embedded_chunk_count, 2)
        self.assertEqual(result.dimensions, 4)
        self.db.refresh(self.document)
        self.assertEqual(self.document.embedding_status, "ready")
        self.assertEqual(self.document.embedded_chunk_count, 2)
        self.assertEqual(self.document.embedding_model, "fake-multilingual")
        self.assertEqual(self.document.embedding_dimensions, 4)

        chunks = (
            self.db.query(RAGChunk)
            .filter(RAGChunk.document_id == self.document.id)
            .order_by(RAGChunk.chunk_index)
            .all()
        )
        self.assertEqual(chunks[0].embedding_model, "fake-multilingual")
        self.assertEqual(chunks[0].embedding_dimensions, 4)
        self.assertEqual(len(chunks[0].embedding), 4)
        self.assertEqual(chunks[0].metadata_json["embedding_provider"], "fake")

    def test_search_request_validation(self) -> None:
        payload = RAGSemanticSearchRequest(query="Mustaqil ta'lim nima?", top_k=5)
        self.assertEqual(payload.top_k, 5)
        self.assertIsNone(payload.document_ids)
        self.assertIsNone(payload.min_score)
        example = RAGSemanticSearchRequest.model_json_schema()["example"]
        self.assertEqual(set(example), {"query", "top_k"})
        with self.assertRaises(Exception):
            RAGSemanticSearchRequest(query="x", top_k=5)


class RAGStudentAccessScopeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.db: Session = factory()

        self.teacher = Teacher(
            first_name="Scope",
            last_name="Teacher",
            email="scope-teacher@example.com",
            password_hash="hash",
        )
        self.other_teacher = Teacher(
            first_name="Other",
            last_name="Teacher",
            email="other-teacher@example.com",
            password_hash="hash",
        )
        self.db.add_all([self.teacher, self.other_teacher])
        self.db.flush()
        self.student = Student(
            teacher_id=self.teacher.id,
            full_name="Scope Student",
            login="scope_student",
            password_hash="hash",
            is_active=True,
        )
        self.db.add(self.student)
        self.db.commit()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def _embedded_document(self, *, teacher_id: int, task_id: int | None = None):
        document = RAGDocument(
            teacher_id=teacher_id,
            task_id=task_id,
            title="Scope material",
            original_filename="scope.docx",
            stored_file_path="private/scope.docx",
            file_type="docx",
            status="ready",
            chunk_count=1,
        )
        self.db.add(document)
        self.db.flush()
        self.db.add(
            RAGChunk(
                document_id=document.id,
                chunk_index=0,
                content="Scope material content",
                embedding=[0.1] * 384,
                embedding_model="intfloat/multilingual-e5-small",
                embedding_dimensions=384,
            )
        )
        self.db.commit()
        return document

    def test_general_teacher_material_is_searchable_for_student(self) -> None:
        self._embedded_document(teacher_id=self.teacher.id)
        self.assertTrue(student_has_searchable_embeddings(self.db, self.student))

    def test_task_material_requires_student_assignment(self) -> None:
        task = Task(
            teacher_id=self.teacher.id,
            title="Assigned task",
            mode="optional",
        )
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        self._embedded_document(teacher_id=self.teacher.id, task_id=task.id)

        self.assertFalse(student_has_searchable_embeddings(self.db, self.student))

        self.db.add(TaskAssignment(task_id=task.id, student_id=self.student.id))
        self.db.commit()
        self.assertTrue(student_has_searchable_embeddings(self.db, self.student))

    def test_other_teacher_material_is_not_searchable(self) -> None:
        self._embedded_document(teacher_id=self.other_teacher.id)
        self.assertFalse(student_has_searchable_embeddings(self.db, self.student))


class RAGLocalEmbeddingMathTestCase(unittest.TestCase):
    def test_average_pool_and_normalize(self) -> None:
        import numpy as np

        last_hidden = np.asarray(
            [[[1.0, 0.0], [3.0, 0.0], [100.0, 100.0]]],
            dtype=np.float32,
        )
        mask = np.asarray([[1, 1, 0]], dtype=np.int64)
        pooled = LocalONNXE5EmbeddingProvider._average_pool(last_hidden, mask)
        self.assertTrue(np.allclose(pooled, [[2.0, 0.0]]))
        normalized = LocalONNXE5EmbeddingProvider._normalize(pooled)
        self.assertTrue(np.allclose(normalized, [[1.0, 0.0]]))

    def test_provider_status_does_not_download_model(self) -> None:
        provider = LocalONNXE5EmbeddingProvider(
            model_name="intfloat/multilingual-e5-small",
            dimensions=384,
            cache_dir="app/rag_models",
        )
        status = provider.status()
        self.assertIsInstance(status, EmbeddingProviderStatus)
        self.assertFalse(status.loaded)
        self.assertEqual(status.dimensions, 384)


if __name__ == "__main__":
    unittest.main()
