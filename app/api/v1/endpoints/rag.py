from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.teacher import Teacher
from app.schemas.rag import (
    RAGChunkRead,
    RAGDocumentEmbeddingResult,
    RAGDocumentJobResponse,
    RAGDocumentRead,
    RAGDocumentUpdate,
    RAGEmbeddingStatusRead,
    RAGMonitoringSummaryRead,
    RAGProcessingJobRead,
    RAGSemanticSearchRequest,
    RAGSemanticSearchResponse,
    RAGStorageUsageRead,
)
from app.services.auth_service import get_current_teacher
from app.services.rag_embedding_service import (
    embed_teacher_document,
    get_embedding_status,
    semantic_search_teacher_documents,
)
from app.services.rag_job_service import (
    enqueue_rag_job,
    get_teacher_job_or_404,
    get_teacher_monitoring_summary_values,
    list_teacher_jobs,
    retry_teacher_job,
)
from app.services.rag_service import (
    RAGProcessingError,
    create_and_process_rag_document,
    create_rag_document_record,
    delete_teacher_document,
    get_document_chunks,
    get_teacher_document_or_404,
    get_teacher_documents,
    get_teacher_storage_usage_values,
    reprocess_teacher_document,
    update_teacher_document,
)


router = APIRouter()


@router.post(
    "/documents",
    response_model=RAGDocumentRead,
    status_code=status.HTTP_201_CREATED,
)
def upload_rag_document(
    title: str = Form(...),
    task_id: int | None = Form(default=None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    """Backward-compatible synchronous upload and chunking endpoint."""
    try:
        return create_and_process_rag_document(
            db,
            current_teacher,
            title=title,
            task_id=task_id,
            file=file,
        )
    except RAGProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": str(exc),
                "document_id": exc.document_id,
                "status": "failed",
            },
        ) from exc


@router.post(
    "/documents/background",
    response_model=RAGDocumentJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def upload_rag_document_background(
    title: str = Form(...),
    task_id: int | None = Form(default=None),
    auto_embed: bool = Form(default=True),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    """Faylni saqlab, ingestionni DB-backed background queue'ga qo‘yadi."""
    document = create_rag_document_record(
        db,
        current_teacher,
        title=title,
        task_id=task_id,
        file=file,
    )
    job = enqueue_rag_job(
        db,
        current_teacher,
        document,
        job_type="ingest",
        auto_embed=auto_embed,
    )
    return RAGDocumentJobResponse(document=document, job=job)


@router.get("/documents", response_model=list[RAGDocumentRead])
def list_rag_documents(
    task_id: int | None = None,
    document_status: str | None = None,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return get_teacher_documents(
        db,
        current_teacher,
        task_id=task_id,
        document_status=document_status,
    )


@router.get("/documents/{document_id}", response_model=RAGDocumentRead)
def get_rag_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return get_teacher_document_or_404(db, current_teacher, document_id)


@router.patch("/documents/{document_id}", response_model=RAGDocumentRead)
def update_rag_document(
    document_id: int,
    payload: RAGDocumentUpdate,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    fields = payload.model_fields_set
    if not fields:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Yangilash uchun kamida bitta maydon yuboring.",
        )
    return update_teacher_document(
        db,
        current_teacher,
        document_id,
        title=payload.title,
        task_id=payload.task_id,
        update_title="title" in fields,
        update_task_id="task_id" in fields,
    )


@router.get(
    "/documents/{document_id}/chunks",
    response_model=list[RAGChunkRead],
)
def list_rag_document_chunks(
    document_id: int,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return get_document_chunks(db, current_teacher, document_id)


@router.post(
    "/documents/{document_id}/reprocess",
    response_model=RAGDocumentRead,
)
def reprocess_rag_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    try:
        return reprocess_teacher_document(db, current_teacher, document_id)
    except RAGProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": str(exc),
                "document_id": exc.document_id,
                "status": "failed",
            },
        ) from exc


@router.post(
    "/documents/{document_id}/reprocess/background",
    response_model=RAGProcessingJobRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def reprocess_rag_document_background(
    document_id: int,
    auto_embed: bool = True,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    document = get_teacher_document_or_404(db, current_teacher, document_id)
    return enqueue_rag_job(
        db,
        current_teacher,
        document,
        job_type="reprocess",
        auto_embed=auto_embed,
    )


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_rag_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    delete_teacher_document(db, current_teacher, document_id)


@router.get("/embedding/status", response_model=RAGEmbeddingStatusRead)
def rag_embedding_status(
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return get_embedding_status()


@router.post(
    "/documents/{document_id}/embed",
    response_model=RAGDocumentEmbeddingResult,
)
def embed_rag_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return embed_teacher_document(db, current_teacher, document_id)


@router.post(
    "/documents/{document_id}/embed/background",
    response_model=RAGProcessingJobRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def embed_rag_document_background(
    document_id: int,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    document = get_teacher_document_or_404(db, current_teacher, document_id)
    if document.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Faqat ready holatidagi hujjat embedding queue'ga qo‘yiladi.",
        )
    return enqueue_rag_job(
        db,
        current_teacher,
        document,
        job_type="embed",
    )


@router.post("/search", response_model=RAGSemanticSearchResponse)
def semantic_search_rag_documents(
    payload: RAGSemanticSearchRequest,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return semantic_search_teacher_documents(
        db,
        current_teacher,
        query=payload.query,
        top_k=payload.top_k,
        task_id=payload.task_id,
        document_ids=payload.document_ids,
        min_score=payload.min_score,
    )


@router.get("/jobs", response_model=list[RAGProcessingJobRead])
def list_rag_processing_jobs(
    job_status: str | None = None,
    document_id: int | None = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return list_teacher_jobs(
        db,
        current_teacher,
        job_status=job_status,
        document_id=document_id,
        limit=limit,
    )


@router.get("/jobs/{job_id}", response_model=RAGProcessingJobRead)
def get_rag_processing_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return get_teacher_job_or_404(db, current_teacher, job_id)


@router.post("/jobs/{job_id}/retry", response_model=RAGProcessingJobRead)
def retry_rag_processing_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return retry_teacher_job(db, current_teacher, job_id)


@router.get("/storage/usage", response_model=RAGStorageUsageRead)
def rag_storage_usage(
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return get_teacher_storage_usage_values(db, current_teacher.id)


@router.get("/monitoring/summary", response_model=RAGMonitoringSummaryRead)
def rag_monitoring_summary(
    db: Session = Depends(get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
):
    return get_teacher_monitoring_summary_values(db, current_teacher)
