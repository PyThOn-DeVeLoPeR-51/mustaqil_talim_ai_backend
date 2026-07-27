from fastapi import APIRouter

from app.api.v1.endpoints import (
    ai_mentor,
    analytics,
    auth,
    results,
    rag,
    students,
    submissions,
    tasks,
)


api_router = APIRouter()

api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Auth"],
)

api_router.include_router(
    students.router,
    prefix="/students",
    tags=["Students"],
)

api_router.include_router(
    tasks.router,
    prefix="/tasks",
    tags=["Tasks"],
)

api_router.include_router(
    submissions.router,
    prefix="/submissions",
    tags=["Submissions"],
)

api_router.include_router(
    results.router,
    prefix="/results",
    tags=["Results"],
)

api_router.include_router(
    analytics.router,
    prefix="/analytics",
    tags=["Analytics"],
)
api_router.include_router(
    ai_mentor.router,
    prefix="/ai-mentor",
    tags=["AI Mentor"],
)

api_router.include_router(
    rag.router,
    prefix="/rag",
    tags=["RAG Knowledge Base"],
)
