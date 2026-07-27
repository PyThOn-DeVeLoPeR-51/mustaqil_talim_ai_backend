# RAG Stage 13 — Local multilingual embeddings + pgvector semantic search

## Model choice

Default: `intfloat/multilingual-e5-small` (384 dimensions).

Why this model for the first production-like RAG iteration:
- multilingual retrieval model;
- runs locally with no per-request API cost;
- ONNX Runtime path avoids PyTorch on Windows/Python 3.13;
- 512-token input fits the current ~1800-character chunks well;
- E5 uses `query: ` for questions and `passage: ` for document chunks.

The standard ONNX file is used for broad CPU compatibility. It is downloaded once from Hugging Face and cached under `app/rag_models/`.

## New endpoints

- `GET /api/v1/rag/embedding/status`
- `POST /api/v1/rag/documents/{document_id}/embed`
- `POST /api/v1/rag/search`

All endpoints require teacher authentication in this stage.

## Embedding flow

1. Upload/process PDF or DOCX as before.
2. Call `/documents/{id}/embed`.
3. The first call downloads the model once, then embeds every chunk.
4. `embedding`, `embedding_model`, `embedding_dimensions` are stored in PostgreSQL/pgvector.
5. `/search` embeds the query and orders chunks by cosine distance.

## HNSW

The `rag_chunks.embedding` column remains dimension-flexible `VECTOR()`.
Migration `d8a4f9c1e2b7` adds a partial expression HNSW index for the default model:

```sql
(embedding::vector(384)) vector_cosine_ops
```

filtered to `intfloat/multilingual-e5-small`. This keeps the schema open for future 768/1024-dimension embedding models.
