from __future__ import annotations

from contextlib import contextmanager
import mimetypes
from pathlib import Path
import tempfile
from typing import BinaryIO, Iterator

from app.core.config import settings
from app.storage.base import StorageBackend, StorageConfigurationError
from app.storage.local import LocalStorage
from app.storage.r2 import R2Storage


_BACKEND: StorageBackend | None = None
_BACKEND_PROVIDER: str | None = None


def canonical_storage_key(value: str) -> str:
    """Convert historical local paths to provider-independent object keys."""
    normalized = str(value).strip().replace("\\", "/")
    if not normalized:
        return normalized

    if normalized.startswith("http://") or normalized.startswith("https://"):
        return normalized

    normalized = normalized.lstrip("/")

    for marker in ("app/uploads/", "uploads/"):
        if normalized.startswith(marker):
            return normalized[len(marker) :]

    for marker in ("app/rag_storage/", "rag_storage/"):
        if normalized.startswith(marker):
            return "rag/" + normalized[len(marker) :]

    return normalized


def get_storage_backend() -> StorageBackend:
    global _BACKEND, _BACKEND_PROVIDER
    provider = settings.STORAGE_PROVIDER.strip().lower()
    if provider not in {"local", "r2"}:
        raise StorageConfigurationError(
            "STORAGE_PROVIDER faqat 'local' yoki 'r2' bo‘lishi mumkin."
        )

    if _BACKEND is not None and _BACKEND_PROVIDER == provider:
        return _BACKEND

    _BACKEND = LocalStorage() if provider == "local" else R2Storage()
    _BACKEND_PROVIDER = provider
    return _BACKEND


def reset_storage_backend() -> None:
    global _BACKEND, _BACKEND_PROVIDER
    _BACKEND = None
    _BACKEND_PROVIDER = None


def _legacy_existing_path(value: str | None) -> Path | None:
    if not value:
        return None
    raw = Path(str(value))
    try:
        if raw.is_file():
            return raw
    except OSError:
        return None
    return None


def _guess_content_type(key: str) -> str | None:
    content_type, _ = mimetypes.guess_type(key)
    return content_type


def persist_upload_file(
    fileobj: BinaryIO,
    key: str,
    *,
    content_type: str | None = None,
) -> str:
    key = canonical_storage_key(key)
    try:
        fileobj.seek(0)
    except Exception:
        pass
    try:
        return get_storage_backend().put_fileobj(
            fileobj,
            key,
            content_type=content_type or _guess_content_type(key),
        )
    finally:
        try:
            fileobj.seek(0)
        except Exception:
            pass


def persist_local_file(
    local_path: str | Path,
    key: str,
    *,
    content_type: str | None = None,
) -> str:
    key = canonical_storage_key(key)
    return get_storage_backend().put_path(
        local_path,
        key,
        content_type=content_type or _guess_content_type(key),
    )


def storage_exists(value: str | None) -> bool:
    if not value:
        return False
    legacy = _legacy_existing_path(value)
    if legacy is not None:
        return True
    key = canonical_storage_key(value)
    if key.startswith("http://") or key.startswith("https://"):
        return False
    return get_storage_backend().exists(key)


def delete_storage_object(value: str | None) -> None:
    if not value:
        return
    legacy = _legacy_existing_path(value)
    if legacy is not None:
        legacy.unlink(missing_ok=True)
        return
    key = canonical_storage_key(value)
    if key.startswith("http://") or key.startswith("https://"):
        return
    get_storage_backend().delete(key)


def delete_storage_prefix(prefix: str) -> None:
    key = canonical_storage_key(prefix).rstrip("/") + "/"
    get_storage_backend().delete_prefix(key)


def storage_read_url(value: str | None, *, expires_seconds: int | None = None) -> str | None:
    if not value:
        return None
    raw = str(value)
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    key = canonical_storage_key(raw)
    ttl = (
        int(expires_seconds)
        if expires_seconds is not None
        else int(settings.R2_PRESIGNED_GET_TTL_SECONDS)
    )
    return get_storage_backend().generate_read_url(key, expires_seconds=ttl)


@contextmanager
def materialize_storage_file(value: str) -> Iterator[Path]:
    """Expose a storage object as a local Path for OpenCV/PyMuPDF/etc."""
    legacy = _legacy_existing_path(value)
    if legacy is not None:
        yield legacy
        return

    key = canonical_storage_key(value)
    if key.startswith("http://") or key.startswith("https://"):
        raise StorageConfigurationError("Remote URL storage key sifatida ishlatilmaydi.")

    backend = get_storage_backend()
    if isinstance(backend, LocalStorage):
        direct = backend.direct_path(key)
        if not direct.is_file():
            backend.download_to_path(key, direct)
        yield direct
        return

    suffix = Path(key).suffix
    with tempfile.TemporaryDirectory(prefix="mustaqil-storage-") as temp_dir:
        destination = Path(temp_dir) / f"object{suffix}"
        backend.download_to_path(key, destination)
        yield destination
