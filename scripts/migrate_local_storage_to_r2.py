"""Migrate legacy local file paths in PostgreSQL to the configured R2 bucket.

Dry-run by default:
    python scripts/migrate_local_storage_to_r2.py

Apply changes:
    STORAGE_PROVIDER=r2 python scripts/migrate_local_storage_to_r2.py --apply

The script can only migrate files that still physically exist on the machine
where it is executed. Missing Render ephemeral files are reported and must be
re-uploaded from an external copy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.db.database import SessionLocal
from app.models.rag import RAGDocument
from app.models.submission import Submission
from app.models.task import Task
from app.storage import persist_local_file


@dataclass
class MigrationStats:
    migrated: int = 0
    missing: int = 0
    skipped: int = 0


def _is_legacy_local_path(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.replace("\\", "/")
    return normalized.startswith("app/uploads/") or normalized.startswith(
        "app/rag_storage/"
    ) or Path(value).is_file()


def _migrate_value(
    value: str | None,
    *,
    destination_key: str,
    apply: bool,
    stats: MigrationStats,
) -> str | None:
    if not value:
        return value
    if not _is_legacy_local_path(value):
        stats.skipped += 1
        return value

    source = Path(value)
    if not source.is_file():
        print(f"MISSING  {value}")
        stats.missing += 1
        return value

    print(f"{'MIGRATE' if apply else 'DRY-RUN'}  {value} -> {destination_key}")
    if apply:
        persist_local_file(source, destination_key)
        stats.migrated += 1
        return destination_key

    stats.migrated += 1
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Upload existing local files to R2 and update database paths.",
    )
    args = parser.parse_args()

    if args.apply and settings.STORAGE_PROVIDER.strip().lower() != "r2":
        parser.error("--apply uchun STORAGE_PROVIDER=r2 bo‘lishi kerak.")

    stats = MigrationStats()
    db = SessionLocal()
    try:
        for task in db.query(Task).order_by(Task.id.asc()).all():
            if task.reference_file_path:
                suffix = Path(task.reference_file_path).suffix.lower()
                task.reference_file_path = _migrate_value(
                    task.reference_file_path,
                    destination_key=(
                        f"tasks/legacy/task-{task.id}/references/reference{suffix}"
                    ),
                    apply=args.apply,
                    stats=stats,
                )
            if task.instruction_file_path:
                suffix = Path(task.instruction_file_path).suffix.lower()
                task.instruction_file_path = _migrate_value(
                    task.instruction_file_path,
                    destination_key=(
                        f"tasks/legacy/task-{task.id}/instructions/instruction{suffix}"
                    ),
                    apply=args.apply,
                    stats=stats,
                )

        for submission in db.query(Submission).order_by(Submission.id.asc()).all():
            suffix = Path(submission.uploaded_file_path).suffix.lower()
            submission.uploaded_file_path = _migrate_value(
                submission.uploaded_file_path,
                destination_key=(
                    f"submissions/legacy/submission-{submission.id}/drawing{suffix}"
                ),
                apply=args.apply,
                stats=stats,
            ) or submission.uploaded_file_path

            if submission.overlay_path:
                suffix = Path(submission.overlay_path).suffix.lower()
                submission.overlay_path = _migrate_value(
                    submission.overlay_path,
                    destination_key=(
                        f"results/submissions/{submission.id}/legacy/overlay{suffix}"
                    ),
                    apply=args.apply,
                    stats=stats,
                )

        for document in db.query(RAGDocument).order_by(RAGDocument.id.asc()).all():
            suffix = Path(document.stored_file_path).suffix.lower()
            document.stored_file_path = _migrate_value(
                document.stored_file_path,
                destination_key=(
                    f"rag/teachers/{document.teacher_id}/legacy/"
                    f"document-{document.id}{suffix}"
                ),
                apply=args.apply,
                stats=stats,
            ) or document.stored_file_path
            if args.apply and isinstance(document.metadata_json, dict):
                metadata = dict(document.metadata_json)
                metadata["storage"] = "r2"
                document.metadata_json = metadata

        if args.apply:
            db.commit()
        else:
            db.rollback()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(
        f"\n{mode} summary: candidates={stats.migrated}, "
        f"missing={stats.missing}, skipped={stats.skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
