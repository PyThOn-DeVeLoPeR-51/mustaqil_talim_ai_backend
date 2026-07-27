# RAG chunking v3

Stage 13.1 improves retrieval readability and Swagger defaults.

- Chunk end boundaries prefer paragraphs and complete sentences.
- Overlap starts at a paragraph/sentence boundary instead of raw `end - overlap`.
- Very long single sentences fall back to whitespace boundaries.
- Reprocessed chunks use `ingestion_version=3` and `chunking_strategy=sentence_aware_overlap_v3`.
- Semantic search Swagger example contains only `query` and `top_k`; optional `document_ids`, `task_id`, and `min_score` remain unset unless intentionally supplied.
- `min_score=1.0` should not be used as a general default because it effectively requires an exact cosine match.

After installing this update, reprocess a document and then embed it again because reprocessing recreates chunks and clears their embeddings.
