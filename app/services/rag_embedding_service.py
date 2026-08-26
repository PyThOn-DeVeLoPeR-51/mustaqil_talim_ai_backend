from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from fastapi import HTTPException, status
from pgvector.sqlalchemy import VECTOR
from sqlalchemy import cast, or_, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.rag import RAGChunk, RAGDocument
from app.models.student import Student
from app.models.task import TaskAssignment
from app.models.teacher import Teacher
from app.rag.embeddings import (
    EmbeddingProvider,
    EmbeddingProviderError,
    GeminiEmbeddingProvider,
    LocalONNXE5EmbeddingProvider,
    get_embedding_provider,
)
from app.schemas.rag import (
    RAGDocumentEmbeddingResult,
    RAGEmbeddingStatusRead,
    RAGSemanticSearchHit,
    RAGSemanticSearchResponse,
)
from app.services.rag_service import get_teacher_document_or_404


def get_embedding_status() -> RAGEmbeddingStatusRead:
    provider = get_embedding_provider()
    if isinstance(provider, (LocalONNXE5EmbeddingProvider, GeminiEmbeddingProvider)):
        provider_status = provider.status()
        return RAGEmbeddingStatusRead(**provider_status.__dict__)
    return RAGEmbeddingStatusRead(
        provider=provider.provider_name,
        model=provider.model_name,
        dimensions=provider.dimensions,
        loaded=True,
        cache_dir=settings.RAG_EMBEDDING_CACHE_DIR,
    )


def _validate_embedding_vectors(
    vectors: Sequence[Sequence[float]],
    expected_count: int,
    dimensions: int,
) -> None:
    if len(vectors) != expected_count:
        raise EmbeddingProviderError(
            f"Embedding soni mos emas: kutilgan={expected_count}, olingan={len(vectors)}"
        )
    for vector in vectors:
        if len(vector) != dimensions:
            raise EmbeddingProviderError(
                f"Embedding dimension mos emas: kutilgan={dimensions}, olingan={len(vector)}"
            )


def embed_teacher_document(
    db: Session,
    teacher: Teacher,
    document_id: int,
    *,
    provider: EmbeddingProvider | None = None,
) -> RAGDocumentEmbeddingResult:
    document = get_teacher_document_or_404(db, teacher, document_id)
    if document.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Faqat ready holatidagi RAG hujjatiga embedding yaratish mumkin.",
        )

    chunks = (
        db.query(RAGChunk)
        .filter(RAGChunk.document_id == document.id)
        .order_by(RAGChunk.chunk_index.asc())
        .all()
    )
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Hujjatda embedding yaratish uchun chunk mavjud emas.",
        )

    embedding_provider = provider or get_embedding_provider()
    try:
        vectors = embedding_provider.embed_documents([chunk.content for chunk in chunks])
        _validate_embedding_vectors(
            vectors,
            expected_count=len(chunks),
            dimensions=embedding_provider.dimensions,
        )
    except EmbeddingProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding modeli ishlamadi: {exc}",
        ) from exc

    now = datetime.now(timezone.utc)
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk.embedding = list(vector)
        chunk.embedding_model = embedding_provider.model_name
        chunk.embedding_dimensions = embedding_provider.dimensions
        metadata = dict(chunk.metadata_json or {})
        metadata.update(
            {
                "embedding_provider": embedding_provider.provider_name,
                "embedding_version": 1,
                "embedded_at": now.isoformat(),
            }
        )
        chunk.metadata_json = metadata

    document_metadata = dict(document.metadata_json or {})
    document_metadata.update(
        {
            "embedding_provider": embedding_provider.provider_name,
            "embedding_model": embedding_provider.model_name,
            "embedding_dimensions": embedding_provider.dimensions,
            "embedded_chunk_count": len(chunks),
            "embedded_at": now.isoformat(),
        }
    )
    document.metadata_json = document_metadata
    db.commit()

    return RAGDocumentEmbeddingResult(
        document_id=document.id,
        chunk_count=len(chunks),
        embedded_chunk_count=len(chunks),
        provider=embedding_provider.provider_name,
        model=embedding_provider.model_name,
        dimensions=embedding_provider.dimensions,
    )


def _search_documents(
    db: Session,
    *,
    teacher_id: int,
    query: str,
    top_k: int | None,
    task_id: int | None,
    document_ids: list[int] | None,
    min_score: float | None,
    provider: EmbeddingProvider | None,
    student_id: int | None = None,
) -> RAGSemanticSearchResponse:
    clean_query = query.strip()
    if len(clean_query) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Qidiruv matni juda qisqa.",
        )

    embedding_provider = provider or get_embedding_provider()
    requested_top_k = top_k or settings.RAG_SEARCH_TOP_K
    requested_top_k = min(requested_top_k, settings.RAG_SEARCH_MAX_TOP_K)

    try:
        query_vector = embedding_provider.embed_query(clean_query)
    except EmbeddingProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding modeli ishlamadi: {exc}",
        ) from exc

    if len(query_vector) != embedding_provider.dimensions:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query embedding dimension noto‘g‘ri.",
        )

    # RAGChunk.embedding VECTOR() o‘lchamsiz saqlanadi. Tanlangan model uchun
    # cast vector(384) qilamiz; migrationdagi HNSW expression index ham aynan
    # shu expression va model filteri bilan yaratiladi.
    vector_expr = cast(RAGChunk.embedding, VECTOR(embedding_provider.dimensions))
    distance_expr = vector_expr.cosine_distance(query_vector)

    query_db = (
        db.query(RAGChunk, RAGDocument, distance_expr.label("distance"))
        .join(RAGDocument, RAGDocument.id == RAGChunk.document_id)
        .filter(
            RAGDocument.teacher_id == teacher_id,
            RAGDocument.status == "ready",
            RAGChunk.embedding.is_not(None),
            RAGChunk.embedding_model == embedding_provider.model_name,
            RAGChunk.embedding_dimensions == embedding_provider.dimensions,
        )
    )

    # Student qidiruvida task_id=NULL materiallar umumiy bilim bazasi hisoblanadi.
    # Taskga bog‘langan material esa faqat shu studentga assignment mavjud bo‘lsa ko‘rinadi.
    if student_id is not None:
        assigned_task_ids = select(TaskAssignment.task_id).where(
            TaskAssignment.student_id == student_id
        )
        query_db = query_db.filter(
            or_(
                RAGDocument.task_id.is_(None),
                RAGDocument.task_id.in_(assigned_task_ids),
            )
        )

    if task_id is not None:
        query_db = query_db.filter(RAGDocument.task_id == task_id)
    if document_ids:
        query_db = query_db.filter(RAGDocument.id.in_(document_ids))

    # Thresholdni SQLga qattiq tiqmaymiz: score=1-distance ni Python tarafda
    # aniq hisoblaymiz. Top-k kichik bo‘lgani uchun bu sodda va tushunarli.
    fetch_limit = min(max(requested_top_k * 4, requested_top_k), 80)
    rows = query_db.order_by(distance_expr.asc()).limit(fetch_limit).all()

    hits: list[RAGSemanticSearchHit] = []
    for chunk, document, distance in rows:
        distance_value = float(distance)
        score = 1.0 - distance_value
        if min_score is not None and score < min_score:
            continue
        hits.append(
            RAGSemanticSearchHit(
                score=round(score, 6),
                distance=round(distance_value, 6),
                document=document,
                chunk=chunk,
            )
        )
        if len(hits) >= requested_top_k:
            break

    return RAGSemanticSearchResponse(
        query=clean_query,
        provider=embedding_provider.provider_name,
        model=embedding_provider.model_name,
        dimensions=embedding_provider.dimensions,
        top_k=requested_top_k,
        hits=hits,
    )


def semantic_search_teacher_documents(
    db: Session,
    teacher: Teacher,
    *,
    query: str,
    top_k: int | None = None,
    task_id: int | None = None,
    document_ids: list[int] | None = None,
    min_score: float | None = None,
    provider: EmbeddingProvider | None = None,
) -> RAGSemanticSearchResponse:
    return _search_documents(
        db,
        teacher_id=teacher.id,
        query=query,
        top_k=top_k,
        task_id=task_id,
        document_ids=document_ids,
        min_score=min_score,
        provider=provider,
    )


def student_has_searchable_embeddings(db: Session, student: Student) -> bool:
    """Student ko‘rishi mumkin bo‘lgan kamida bitta embedded chunk borligini tekshiradi.

    Bu tekshiruv embedding modelini yuklamaydi. Shuning uchun RAG materiallari hali
    tayyor bo‘lmasa AI Mentor chat odatdagi LLM rejimida tez davom etadi.
    """

    assigned_task_ids = select(TaskAssignment.task_id).where(
        TaskAssignment.student_id == student.id
    )
    row = (
        db.query(RAGChunk.id)
        .join(RAGDocument, RAGDocument.id == RAGChunk.document_id)
        .filter(
            RAGDocument.teacher_id == student.teacher_id,
            RAGDocument.status == "ready",
            or_(
                RAGDocument.task_id.is_(None),
                RAGDocument.task_id.in_(assigned_task_ids),
            ),
            RAGChunk.embedding.is_not(None),
            RAGChunk.embedding_model == settings.RAG_EMBEDDING_MODEL,
            RAGChunk.embedding_dimensions == settings.RAG_EMBEDDING_DIMENSIONS,
        )
        .first()
    )
    return row is not None


def semantic_search_student_documents(
    db: Session,
    student: Student,
    *,
    query: str,
    top_k: int | None = None,
    task_id: int | None = None,
    document_ids: list[int] | None = None,
    min_score: float | None = None,
    provider: EmbeddingProvider | None = None,
) -> RAGSemanticSearchResponse:
    """AI Mentor uchun studentga ruxsat etilgan o‘quv materiallarida qidiruv."""

    return _search_documents(
        db,
        teacher_id=student.teacher_id,
        student_id=student.id,
        query=query,
        top_k=top_k,
        task_id=task_id,
        document_ids=document_ids,
        min_score=min_score,
        provider=provider,
    )
