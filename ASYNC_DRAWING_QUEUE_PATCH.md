# Async Drawing AI Queue Patch (2026-08-27)

## What changed
- `POST /api/v1/submissions` no longer runs Drawing AI inline.
- Submission upload + `DrawingEvaluationJob` are persisted before HTTP 201 returns.
- One in-process Drawing AI worker consumes the durable PostgreSQL queue.
- Automatic retry and stale-job recovery are enabled.
- Existing Drawing AI detection/scoring code under `app/ai/` was not changed.
- Student reference visibility waits for terminal attempts (`evaluated`/`failed`).
- A second attempt is blocked while the previous attempt is still queued/running.

## New env values
```env
DRAWING_BACKGROUND_WORKER_ENABLED=true
DRAWING_WORKER_POLL_SECONDS=1
DRAWING_JOB_MAX_ATTEMPTS=3
DRAWING_JOB_RETRY_BASE_SECONDS=10
DRAWING_JOB_STALE_MINUTES=20
```

## Local upgrade
```powershell
alembic upgrade head
python -m pytest
```

Expected suite count for this ZIP: 77 tests.

Frontend build:
```powershell
npm run build
```

## Render deployment
The database migration must run before the new app starts.

Recommended Render Start Command:
```bash
alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Keep:
```env
DRAWING_BACKGROUND_WORKER_ENABLED=true
```
for a single web-service deployment.

## Dedicated worker later
Web:
```env
DRAWING_BACKGROUND_WORKER_ENABLED=false
```

Worker command:
```bash
python -m scripts.run_drawing_worker
```

The PostgreSQL queue uses `FOR UPDATE SKIP LOCKED`, so multiple worker instances can
consume different jobs safely.
