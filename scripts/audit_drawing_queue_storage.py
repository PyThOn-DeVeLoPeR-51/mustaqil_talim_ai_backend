from __future__ import annotations

import argparse

import app.db.base  # noqa: F401 - register all SQLAlchemy models
from app.db.database import SessionLocal
from app.models.drawing_job import DrawingEvaluationJob
from app.models.submission import Submission
from app.storage import storage_exists


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Drawing AI queue jobs whose submission file is missing from storage."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Mark orphan jobs/submissions as failed. Without this flag the script is dry-run only.",
    )
    args = parser.parse_args()

    with SessionLocal() as db:
        rows = (
            db.query(DrawingEvaluationJob, Submission)
            .join(Submission, Submission.id == DrawingEvaluationJob.submission_id)
            .filter(DrawingEvaluationJob.status.in_(["pending", "running"]))
            .order_by(DrawingEvaluationJob.id.asc())
            .all()
        )

        missing = []
        for job, submission in rows:
            try:
                exists = storage_exists(submission.uploaded_file_path)
            except Exception as exc:
                print(f"CHECK ERROR job={job.id} submission={submission.id}: {exc}")
                continue
            if not exists:
                missing.append((job, submission))
                print(
                    "MISSING "
                    f"job={job.id} submission={submission.id} "
                    f"task={submission.task_id} student={submission.student_id} "
                    f"attempt={submission.attempt_number} "
                    f"key={submission.uploaded_file_path}"
                )

        print(f"\nActive jobs checked: {len(rows)}")
        print(f"Missing submission objects: {len(missing)}")

        if not args.apply:
            print("DRY-RUN only. Database was not changed.")
            return

        for job, submission in missing:
            message = f"Storage object topilmadi: {submission.uploaded_file_path}"
            job.status = "failed"
            job.progress_percent = 0
            job.error_message = message
            job.locked_at = None
            job.locked_by = None
            submission.status = "failed"
            submission.ai_json_result = {"error": message}

        db.commit()
        print(f"APPLY complete. Marked failed: {len(missing)}")


if __name__ == "__main__":
    main()
