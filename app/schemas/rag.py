from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


RAGDocumentStatus = Literal["uploaded", "processing", "ready", "failed", "archived"]
RAGFileType = Literal["pdf", "docx"]


class RAGDocumentRead(BaseModel):
    id: int
    teacher_id: int
    task_id: int | None = None
    title: str
    original_filename: str
    file_type: RAGFileType
    mime_type: str | None = None
    file_size_bytes: int | None = None
    checksum_sha256: str | None = None
    status: RAGDocumentStatus
    page_count: int | None = None
    chunk_count: int
    processing_error: str | None = None
    metadata_json: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class RAGChunkRead(BaseModel):
    id: int
    document_id: int
    chunk_index: int
    content: str
    content_hash: str | None = None
    section_title: str | None = None
    page_number_start: int | None = None
    page_number_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    token_count: int | None = None
    char_count: int | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    metadata_json: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RAGDocumentWithChunks(BaseModel):
    document: RAGDocumentRead
    chunks: list[RAGChunkRead] = Field(default_factory=list)


class RAGEmbeddingStatusRead(BaseModel):
    provider: str
    model: str
    dimensions: int
    loaded: bool
    cache_dir: str


class RAGDocumentEmbeddingResult(BaseModel):
    document_id: int
    chunk_count: int
    embedded_chunk_count: int
    provider: str
    model: str
    dimensions: int


class RAGSemanticSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    task_id: int | None = None
    document_ids: list[int] | None = Field(default=None, max_length=50)
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)

    # Swagger should demonstrate the normal broad semantic-search request.
    # Optional filters are intentionally omitted so users do not accidentally
    # set min_score=1.0 (which effectively requires an exact vector match).
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Kredit-modul tizimida mustaqil ta'limning ahamiyati nimada?",
                "top_k": 5,
            }
        }
    )


class RAGSemanticSearchHit(BaseModel):
    score: float
    distance: float
    document: RAGDocumentRead
    chunk: RAGChunkRead


class RAGSemanticSearchResponse(BaseModel):
    query: str
    provider: str
    model: str
    dimensions: int
    top_k: int
    hits: list[RAGSemanticSearchHit] = Field(default_factory=list)
