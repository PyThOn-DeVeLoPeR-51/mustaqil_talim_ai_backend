from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.core.config import settings
from app.storage.base import (
    StorageBackend,
    StorageConfigurationError,
    StorageObjectNotFound,
)


class R2Storage(StorageBackend):
    def __init__(self) -> None:
        missing = [
            name
            for name, value in (
                ("R2_ACCOUNT_ID", settings.R2_ACCOUNT_ID),
                ("R2_ACCESS_KEY_ID", settings.R2_ACCESS_KEY_ID),
                ("R2_SECRET_ACCESS_KEY", settings.R2_SECRET_ACCESS_KEY),
                ("R2_BUCKET_NAME", settings.R2_BUCKET_NAME),
            )
            if not value
        ]
        if missing:
            raise StorageConfigurationError(
                "R2 konfiguratsiyasi to‘liq emas: " + ", ".join(missing)
            )

        self.bucket = str(settings.R2_BUCKET_NAME)
        endpoint = f"https://{settings.R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )

    @staticmethod
    def _extra_args(content_type: str | None) -> dict[str, str] | None:
        if not content_type:
            return None
        return {
            "ContentType": content_type,
            "ContentDisposition": "inline",
        }

    def put_fileobj(
        self,
        fileobj: BinaryIO,
        key: str,
        *,
        content_type: str | None = None,
    ) -> str:
        extra = self._extra_args(content_type)
        kwargs = {"ExtraArgs": extra} if extra else {}
        self.client.upload_fileobj(fileobj, self.bucket, key, **kwargs)
        return key

    def put_path(
        self,
        local_path: str | Path,
        key: str,
        *,
        content_type: str | None = None,
    ) -> str:
        extra = self._extra_args(content_type)
        kwargs = {"ExtraArgs": extra} if extra else {}
        self.client.upload_file(str(local_path), self.bucket, key, **kwargs)
        return key

    def download_to_path(self, key: str, destination: str | Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download_file(self.bucket, key, str(destination))
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise StorageObjectNotFound(f"R2 fayli topilmadi: {key}") from exc
            raise
        return destination

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def delete_prefix(self, prefix: str) -> None:
        # Re-list from the beginning after each batch. This avoids relying on a
        # continuation token whose underlying keys were just deleted.
        while True:
            response = self.client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=1000,
            )
            contents = response.get("Contents") or []
            objects = [{"Key": item["Key"]} for item in contents if item.get("Key")]
            if not objects:
                break
            self.client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": objects, "Quiet": True},
            )
            if len(objects) < 1000:
                break

    def generate_read_url(self, key: str, *, expires_seconds: int) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=max(1, int(expires_seconds)),
        )
