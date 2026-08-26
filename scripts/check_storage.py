"""Safe Cloudflare R2 smoke test using values from .env.

The test writes a tiny temporary object, checks it, generates a signed read URL,
and deletes the object. It never prints credentials or the signed URL itself.
"""

from __future__ import annotations

from io import BytesIO
import uuid

from app.core.config import settings
from app.storage import (
    delete_storage_object,
    persist_upload_file,
    storage_exists,
    storage_read_url,
)


def main() -> int:
    if settings.STORAGE_PROVIDER.strip().lower() != "r2":
        raise SystemExit("STORAGE_PROVIDER=r2 qilib .env ni sozlang.")

    key = f"_healthchecks/storage-v1/{uuid.uuid4().hex}.txt"
    try:
        persist_upload_file(
            BytesIO(b"mustaqil-talim-ai storage-v1"),
            key,
            content_type="text/plain",
        )
        if not storage_exists(key):
            raise RuntimeError("Upload tugadi, ammo R2 object HEAD tekshiruvida topilmadi.")
        signed_url = storage_read_url(key)
        if not signed_url or "X-Amz-" not in signed_url:
            raise RuntimeError("Presigned GET URL yaratilmagan.")
        print("R2 upload: OK")
        print("R2 exists/HEAD: OK")
        print("R2 presigned GET URL: OK")
    finally:
        try:
            delete_storage_object(key)
            print("R2 cleanup: OK")
        except Exception as exc:
            print(f"R2 cleanup warning: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
