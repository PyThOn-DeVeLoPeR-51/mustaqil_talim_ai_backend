from __future__ import annotations

import asyncio
import logging

from app.core.config import settings
from app.db.database import SessionLocal
from app.services.rag_job_service import (
    default_worker_id,
    recover_stale_jobs,
    run_next_rag_job_once,
)


logger = logging.getLogger("app.rag.worker")


class RAGBackgroundWorker:
    def __init__(self) -> None:
        self.worker_id = default_worker_id()
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None:
            return
        with SessionLocal() as db:
            recovered = recover_stale_jobs(db)
        if recovered:
            logger.warning(
                "Recovered stale RAG jobs",
                extra={"recovered_jobs": recovered, "worker_id": self.worker_id},
            )
        self._task = asyncio.create_task(self._run_loop(), name="rag-background-worker")
        logger.info("RAG background worker started", extra={"worker_id": self.worker_id})

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            await self._task
            self._task = None
        logger.info("RAG background worker stopped", extra={"worker_id": self.worker_id})

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job_id = await asyncio.to_thread(
                    run_next_rag_job_once,
                    worker_id=self.worker_id,
                )
                if job_id is None:
                    try:
                        await asyncio.wait_for(
                            self._stop_event.wait(),
                            timeout=max(settings.RAG_WORKER_POLL_SECONDS, 0.2),
                        )
                    except asyncio.TimeoutError:
                        pass
            except Exception:
                logger.exception("RAG worker loop error", extra={"worker_id": self.worker_id})
                await asyncio.sleep(max(settings.RAG_WORKER_POLL_SECONDS, 1.0))
