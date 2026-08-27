from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.api import api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.rag.worker import RAGBackgroundWorker
from app.drawing.worker import DrawingBackgroundWorker


configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_worker: RAGBackgroundWorker | None = None
    drawing_worker: DrawingBackgroundWorker | None = None

    if settings.RAG_BACKGROUND_WORKER_ENABLED:
        rag_worker = RAGBackgroundWorker()
        await rag_worker.start()
        app.state.rag_worker = rag_worker

    if settings.DRAWING_BACKGROUND_WORKER_ENABLED:
        drawing_worker = DrawingBackgroundWorker()
        await drawing_worker.start()
        app.state.drawing_worker = drawing_worker

    try:
        yield
    finally:
        if drawing_worker is not None:
            await drawing_worker.stop()
        if rag_worker is not None:
            await rag_worker.stop()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.1.107:3000",
        "https://mustaqil-talim-ai-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.STORAGE_PROVIDER.strip().lower() == "local":
    local_upload_dir = Path(settings.LOCAL_UPLOAD_DIR)
    local_upload_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/uploads",
        StaticFiles(directory=str(local_upload_dir)),
        name="uploads",
    )

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
def root():
    return {
        "message": "Mustaqil Ta'lim AI Platforma backend ishlayapti"
    }
