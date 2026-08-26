"""Safely remove legacy local-storage-backed test data from the database.

Dry-run by default:
    python -m scripts.clean_legacy_storage_records

Apply deletion:
    python -m scripts.clean_legacy_storage_records --apply

Scope:
- Tasks whose reference/instruction path points to app/uploads or app/rag_storage.
- Submissions whose uploaded/overlay path points to app/uploads or app/rag_storage.
- RAG documents whose stored file path points to app/uploads or app/rag_storage.

Effects of deleting legacy tasks/documents rely on existing FK rules:
- task_assignments and submissions for deleted tasks: ON DELETE CASCADE
- rag_documents.task_id for deleted tasks: ON DELETE SET NULL
- rag_chunks and rag_processing_jobs for deleted documents: ON DELETE CASCADE

Teacher/student accounts and AI Mentor data are not deleted by this script.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from sqlalchemy import or_

from app.db.database import SessionLocal
from app.models.rag import RAGDocument, RAGChunk, RAGProcessingJob
from app.models.submission import Submission
from app.models.task import Task, TaskAssignment


LEGACY_PREFIXES = ("app/uploads/", "app/rag_storage/")


def _is_legacy_path(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.replace("\\", "/").strip()
    return normalized.startswith(LEGACY_PREFIXES)


@dataclass
class CleanupPlan:
    legacy_task_ids: list[int]
    direct_legacy_submission_ids: list[int]
    legacy_document_ids: list[int]
    task_assignment_ids: list[int]
    submissions_removed_via_tasks: list[int]
    rag_chunk_ids: list[int]
    rag_job_ids: list[int]



def _build_plan(db) -> CleanupPlan:
    tasks = db.query(Task).order_by(Task.id.asc()).all()
    legacy_task_ids = [
        task.id
        for task in tasks
        if _is_legacy_path(task.reference_file_path)
        or _is_legacy_path(task.instruction_file_path)
    ]

    submissions = db.query(Submission).order_by(Submission.id.asc()).all()
    direct_legacy_submission_ids = [
        submission.id
        for submission in submissions
        if _is_legacy_path(submission.uploaded_file_path)
        or _is_legacy_path(submission.overlay_path)
    ]

    documents = db.query(RAGDocument).order_by(RAGDocument.id.asc()).all()
    legacy_document_ids = [
        document.id
        for document in documents
        if _is_legacy_path(document.stored_file_path)
    ]

    if legacy_task_ids:
        task_assignment_ids = [
            row.id
            for row in db.query(TaskAssignment)
            .filter(TaskAssignment.task_id.in_(legacy_task_ids))
            .order_by(TaskAssignment.id.asc())
            .all()
        ]
        submissions_removed_via_tasks = [
            row.id
            for row in db.query(Submission)
            .filter(Submission.task_id.in_(legacy_task_ids))
            .order_by(Submission.id.asc())
            .all()
        ]
    else:
        task_assignment_ids = []
        submissions_removed_via_tasks = []

    if legacy_document_ids:
        rag_chunk_ids = [
            row.id
            for row in db.query(RAGChunk)
            .filter(RAGChunk.document_id.in_(legacy_document_ids))
            .order_by(RAGChunk.id.asc())
            .all()
        ]
        rag_job_ids = [
            row.id
            for row in db.query(RAGProcessingJob)
            .filter(RAGProcessingJob.document_id.in_(legacy_document_ids))
            .order_by(RAGProcessingJob.id.asc())
            .all()
        ]
    else:
        rag_chunk_ids = []
        rag_job_ids = []

    return CleanupPlan(
        legacy_task_ids=legacy_task_ids,
        direct_legacy_submission_ids=direct_legacy_submission_ids,
        legacy_document_ids=legacy_document_ids,
        task_assignment_ids=task_assignment_ids,
        submissions_removed_via_tasks=submissions_removed_via_tasks,
        rag_chunk_ids=rag_chunk_ids,
        rag_job_ids=rag_job_ids,
    )



def _fmt(ids: list[int]) -> str:
    return ", ".join(map(str, ids)) if ids else "-"



def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete legacy local-storage-backed test records. Dry-run by default."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the planned records from the database.",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        plan = _build_plan(db)

        print("Legacy cleanup plan")
        print("===================")
        print(f"Tasks to delete: {len(plan.legacy_task_ids)} [{_fmt(plan.legacy_task_ids)}]")
        print(
            "Direct legacy submissions: "
            f"{len(plan.direct_legacy_submission_ids)} "
            f"[{_fmt(plan.direct_legacy_submission_ids)}]"
        )
        print(
            "Submissions removed by task CASCADE: "
            f"{len(plan.submissions_removed_via_tasks)} "
            f"[{_fmt(plan.submissions_removed_via_tasks)}]"
        )
        print(
            "Task assignments removed by CASCADE: "
            f"{len(plan.task_assignment_ids)} [{_fmt(plan.task_assignment_ids)}]"
        )
        print(
            "RAG documents to delete: "
            f"{len(plan.legacy_document_ids)} [{_fmt(plan.legacy_document_ids)}]"
        )
        print(
            "RAG chunks removed by CASCADE: "
            f"{len(plan.rag_chunk_ids)} [{_fmt(plan.rag_chunk_ids)}]"
        )
        print(
            "RAG jobs removed by CASCADE: "
            f"{len(plan.rag_job_ids)} [{_fmt(plan.rag_job_ids)}]"
        )

        # Avoid double-deleting submissions already removed by deleting their task.
        cascade_submission_ids = set(plan.submissions_removed_via_tasks)
        standalone_submission_ids = [
            submission_id
            for submission_id in plan.direct_legacy_submission_ids
            if submission_id not in cascade_submission_ids
        ]

        print(
            "Standalone submissions to delete explicitly: "
            f"{len(standalone_submission_ids)} [{_fmt(standalone_submission_ids)}]"
        )

        if not args.apply:
            db.rollback()
            print("\nDRY-RUN only. Database was not changed.")
            print("If this plan is correct, run the same command with --apply.")
            return 0

        # Delete child-like standalone rows first, then parent objects.
        if standalone_submission_ids:
            db.query(Submission).filter(
                Submission.id.in_(standalone_submission_ids)
            ).delete(synchronize_session=False)

        if plan.legacy_document_ids:
            db.query(RAGDocument).filter(
                RAGDocument.id.in_(plan.legacy_document_ids)
            ).delete(synchronize_session=False)

        if plan.legacy_task_ids:
            db.query(Task).filter(Task.id.in_(plan.legacy_task_ids)).delete(
                synchronize_session=False
            )

        db.commit()

        # Verify that no legacy path remains in the three storage-backed models.
        remaining_tasks = [
            row.id
            for row in db.query(Task).all()
            if _is_legacy_path(row.reference_file_path)
            or _is_legacy_path(row.instruction_file_path)
        ]
        remaining_submissions = [
            row.id
            for row in db.query(Submission).all()
            if _is_legacy_path(row.uploaded_file_path)
            or _is_legacy_path(row.overlay_path)
        ]
        remaining_documents = [
            row.id
            for row in db.query(RAGDocument).all()
            if _is_legacy_path(row.stored_file_path)
        ]

        print("\nAPPLY complete.")
        print(f"Remaining legacy tasks: {len(remaining_tasks)} [{_fmt(remaining_tasks)}]")
        print(
            "Remaining legacy submissions: "
            f"{len(remaining_submissions)} [{_fmt(remaining_submissions)}]"
        )
        print(
            "Remaining legacy RAG documents: "
            f"{len(remaining_documents)} [{_fmt(remaining_documents)}]"
        )

        if remaining_tasks or remaining_submissions or remaining_documents:
            print("WARNING: Some legacy paths still remain. Do not delete local files yet.")
            return 2

        print("Legacy DB storage references are clean.")
        return 0

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
