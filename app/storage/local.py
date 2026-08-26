from __future__ import annotations

import shutil
from pathlib import Path
from typing import BinaryIO

from app.core.config import settings
from app.storage.base import StorageBackend, StorageObjectNotFound


class LocalStorage(StorageBackend):
    """Development/test backend that keeps the historical local filesystem behavior."""

    @staticmethod
    def _path_for_key(key: str) -> Path:
        normalized = key.replace("\\", "/").lstrip("/")
        if normalized.startswith("rag/"):
            relative = normalized[len("rag/") :]
            return Path(settings.RAG_STORAGE_DIR) / relative
        return Path(settings.LOCAL_UPLOAD_DIR) / normalized

    def put_fileobj(
        self,
        fileobj: BinaryIO,
        key: str,
        *,
        content_type: str | None = None,
    ) -> str:
        target = self._path_for_key(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as buffer:
            shutil.copyfileobj(fileobj, buffer)
        return key

    def put_path(
        self,
        local_path: str | Path,
        key: str,
        *,
        content_type: str | None = None,
    ) -> str:
        source = Path(local_path)
        target = self._path_for_key(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return key

    def download_to_path(self, key: str, destination: str | Path) -> Path:
        source = self._path_for_key(key)
        if not source.is_file():
            raise StorageObjectNotFound(f"Storage fayli topilmadi: {key}")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        return destination

    def exists(self, key: str) -> bool:
        return self._path_for_key(key).is_file()

    def delete(self, key: str) -> None:
        path = self._path_for_key(key)
        path.unlink(missing_ok=True)
        parent = path.parent
        # Only clean empty directories inside our configured roots.
        for _ in range(4):
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    def delete_prefix(self, prefix: str) -> None:
        path = self._path_for_key(prefix.rstrip("/"))
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file():
            path.unlink(missing_ok=True)

    def generate_read_url(self, key: str, *, expires_seconds: int) -> str | None:
        normalized = key.replace("\\", "/").lstrip("/")
        if normalized.startswith("rag/"):
            # RAG originals are private and are not mounted as StaticFiles.
            return None
        return f"/uploads/{normalized}"

    def direct_path(self, key: str) -> Path:
        return self._path_for_key(key)
