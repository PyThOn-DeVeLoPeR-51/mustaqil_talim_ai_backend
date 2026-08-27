from __future__ import annotations

import asyncio
import logging
import time

from app.core.config import settings
from app.db.database import SessionLocal
from app.services.drawing_job_service import (
    default_worker_id,
    recover_stale_drawing_jobs,
    run_next_drawing_job_once,
)


logger = logging.getLogger("app.drawing.worker")


class DrawingBackgroundWorker:
    """Bitta process ichida bir vaqtning o'zida bitta Drawing AI job ishlatadi."""

    def __init__(self) -> None:
        self.worker_id = default_worker_id()
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None:
            return

        with SessionLocal() as db:
            recovered = recover_stale_drawing_jobs(db)
        if recovered:
            logger.warning(
                "Recovered stale Drawing AI jobs",
                extra={"recovered_jobs": recovered, "worker_id": self.worker_id},
            )

        self._task = asyncio.create_task(
            self._run_loop(),
            name="drawing-background-worker",
        )
        logger.info(
            "Drawing AI background worker started",
            extra={"worker_id": self.worker_id},
        )

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            await self._task
            self._task = None
        logger.info(
            "Drawing AI background worker stopped",
            extra={"worker_id": self.worker_id},
        )

    async def _run_loop(self) -> None:
        last_recovery = time.monotonic()

        while not self._stop_event.is_set():
            try:
                # Crash/OOMdan keyin running holatda qolgan joblarni faqat startupda
                # emas, periodik ham tekshiramiz.
                if time.monotonic() - last_recovery >= 60:
                    with SessionLocal() as db:
                        recovered = recover_stale_drawing_jobs(db)
                    if recovered:
                        logger.warning(
                            "Recovered stale Drawing AI jobs",
                            extra={
                                "recovered_jobs": recovered,
                                "worker_id": self.worker_id,
                            },
                        )
                    last_recovery = time.monotonic()

                job_id = await asyncio.to_thread(
                    run_next_drawing_job_once,
                    worker_id=self.worker_id,
                )
                if job_id is None:
                    try:
                        await asyncio.wait_for(
                            self._stop_event.wait(),
                            timeout=max(settings.DRAWING_WORKER_POLL_SECONDS, 0.2),
                        )
                    except asyncio.TimeoutError:
                        pass
            except Exception:
                logger.exception(
                    "Drawing AI worker loop error",
                    extra={"worker_id": self.worker_id},
                )
                await asyncio.sleep(max(settings.DRAWING_WORKER_POLL_SECONDS, 1.0))
