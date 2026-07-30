from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.teacher import Teacher
from app.schemas.rag import (
    RAGChunkRead,
    RAGDocumentEmbeddingResult,
    RAGDocumentRead,
    RAGDocumentUpdate,
    RAGEmbeddingStatusRead,
    RAGSemanticSearchRequest,
    RAGSemanticSearchResponse,
)
from app.services.auth_service import get_current_teacher
from app.services.rag_embedding_service import (
    embed_teacher_document,
    get_embedding_status,
    semantic_search_teacher_documents,
)
from app.services.rag_service import (
    RAGProcessingError,
    create_and_process_rag_document,
    delete_teacher_document,
    get_document_chunks,
    get_teacher_document_or_404,
    get_teacher_documents,
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
    # Auth talab qilinadi, lekin status uchun DB kerak emas.
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
