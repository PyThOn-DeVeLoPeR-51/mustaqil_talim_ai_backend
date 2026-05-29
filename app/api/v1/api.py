from fastapi import APIRouter

from app.api.v1.endpoints import auth, results, students, submissions, tasks


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