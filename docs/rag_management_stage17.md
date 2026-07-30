# RAG Management — Stage 17

## Backend additions

- `PATCH /api/v1/rag/documents/{document_id}` updates a material title and/or task binding.
- `RAGDocumentRead` now exposes explicit embedding management fields:
  - `embedded_chunk_count`
  - `embedding_status` (`not_started`, `partial`, `ready`)
  - `embedding_model`
  - `embedding_dimensions`
- Reprocess continues to clear stale embedding metadata, so the UI accurately returns to `not_started`.
- No Alembic migration is required; all new fields are computed from existing document metadata.

## Safety

- Task binding is checked against the current teacher.
- Rebinding a duplicate file to a task where the same checksum already exists returns `409`.
- Private storage paths remain excluded from responses.
