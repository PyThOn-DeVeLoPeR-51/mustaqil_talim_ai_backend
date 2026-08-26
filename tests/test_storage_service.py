from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from app.core.config import settings
from app.storage.service import (
    canonical_storage_key,
    delete_storage_object,
    materialize_storage_file,
    persist_local_file,
    persist_upload_file,
    reset_storage_backend,
    storage_exists,
    storage_read_url,
)


class LocalStorageServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.original_provider = settings.STORAGE_PROVIDER
        self.original_upload_dir = settings.LOCAL_UPLOAD_DIR
        self.original_rag_dir = settings.RAG_STORAGE_DIR
        settings.STORAGE_PROVIDER = "local"
        settings.LOCAL_UPLOAD_DIR = str(Path(self.temp.name) / "uploads")
        settings.RAG_STORAGE_DIR = str(Path(self.temp.name) / "rag")
        reset_storage_backend()

    def tearDown(self) -> None:
        settings.STORAGE_PROVIDER = self.original_provider
        settings.LOCAL_UPLOAD_DIR = self.original_upload_dir
        settings.RAG_STORAGE_DIR = self.original_rag_dir
        reset_storage_backend()
        self.temp.cleanup()

    def test_canonicalizes_historical_paths(self) -> None:
        self.assertEqual(
            canonical_storage_key(r"app\uploads\tasks\abc.pdf"),
            "tasks/abc.pdf",
        )
        self.assertEqual(
            canonical_storage_key("app/rag_storage/7/book.docx"),
            "rag/7/book.docx",
        )

    def test_public_object_roundtrip_and_url(self) -> None:
        key = persist_upload_file(
            BytesIO(b"drawing"),
            "submissions/student-1/drawing.png",
            content_type="image/png",
        )
        self.assertTrue(storage_exists(key))
        self.assertEqual(
            storage_read_url(key),
            "/uploads/submissions/student-1/drawing.png",
        )
        with materialize_storage_file(key) as path:
            self.assertEqual(path.read_bytes(), b"drawing")

        delete_storage_object(key)
        self.assertFalse(storage_exists(key))

    def test_rag_object_stays_outside_public_upload_root(self) -> None:
        source = Path(self.temp.name) / "source.pdf"
        source.write_bytes(b"%PDF-test")
        key = persist_local_file(source, "rag/teachers/4/documents/a.pdf")
        self.assertTrue(storage_exists(key))
        self.assertIsNone(storage_read_url(key))
        self.assertTrue(
            (Path(settings.RAG_STORAGE_DIR) / "teachers/4/documents/a.pdf").is_file()
        )


if __name__ == "__main__":
    unittest.main()
