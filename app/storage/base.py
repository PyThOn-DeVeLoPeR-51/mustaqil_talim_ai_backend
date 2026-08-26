from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO


class StorageError(RuntimeError):
    """Base storage error."""


class StorageConfigurationError(StorageError):
    """Raised when the selected storage backend is not configured."""


class StorageObjectNotFound(StorageError):
    """Raised when an object does not exist."""


class StorageBackend(ABC):
    @abstractmethod
    def put_fileobj(
        self,
        fileobj: BinaryIO,
        key: str,
        *,
        content_type: str | None = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def put_path(
        self,
        local_path: str | Path,
        key: str,
        *,
        content_type: str | None = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def download_to_path(self, key: str, destination: str | Path) -> Path:
        raise NotImplementedError

    @abstractmethod
    def exists(self, key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate_read_url(self, key: str, *, expires_seconds: int) -> str | None:
        raise NotImplementedError
