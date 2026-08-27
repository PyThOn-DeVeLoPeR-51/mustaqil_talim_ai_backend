"""Dedicated Drawing AI worker process.

Usage:
    python -m scripts.run_drawing_worker

The DB-backed queue is shared with the in-process worker. For a dedicated worker
deployment, set DRAWING_BACKGROUND_WORKER_ENABLED=false on the web service.
"""
from __future__ import annotations

import logging
import time

from app.core.config import settings
from app.core.logging import configure_logging
from app.db.database import SessionLocal
from app.services.drawing_job_service import (
    default_worker_id,
    recover_stale_drawing_jobs,
    run_next_drawing_job_once,
)


def main() -> None:
    configure_logging()
    logger = logging.getLogger("app.drawing.worker.cli")
    worker_id = default_worker_id()

    with SessionLocal() as db:
        recovered = recover_stale_drawing_jobs(db)

    logger.info(
        "Dedicated Drawing AI worker started",
        extra={"worker_id": worker_id, "recovered_jobs": recovered},
    )

    last_recovery = time.monotonic()
    while True:
        if time.monotonic() - last_recovery >= 60:
            with SessionLocal() as db:
                recovered = recover_stale_drawing_jobs(db)
            if recovered:
                logger.warning(
                    "Recovered stale Drawing AI jobs",
                    extra={"worker_id": worker_id, "recovered_jobs": recovered},
                )
            last_recovery = time.monotonic()

        job_id = run_next_drawing_job_once(worker_id=worker_id)
        if job_id is None:
            time.sleep(max(settings.DRAWING_WORKER_POLL_SECONDS, 0.2))


if __name__ == "__main__":
    main()
