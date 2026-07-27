# RAG Stage 12 — Document ingestion

Bu bosqich embedding va similarity searchdan oldingi ingestion pipeline'ni yaratadi.

## Oqim

1. O‘qituvchi JWT bilan PDF yoki DOCX yuklaydi.
2. Fayl `app/rag_storage/<teacher_id>/` ichiga UUID nom bilan private saqlanadi.
3. Extension bilan birga fayl signaturasi ham tekshiriladi.
4. SHA-256 checksum va fayl hajmi hisoblanadi.
5. `rag_documents` yozuvi yaratiladi va status `processing` ga o‘tadi.
6. PDF — PyMuPDF, DOCX — python-docx orqali matnga aylantiriladi.
7. Matn overlap bilan chunklarga bo‘linadi.
8. Chunklar `rag_chunks` ga yoziladi; embedding hozircha `NULL` qoladi.
9. Muvaffaqiyatli yakunda document status `ready` bo‘ladi.
10. Xato bo‘lsa status `failed` va `processing_error` saqlanadi.

## API

- `POST /api/v1/rag/documents` — multipart upload (`title`, optional `task_id`, `file`)
- `GET /api/v1/rag/documents` — o‘qituvchining hujjatlari
- `GET /api/v1/rag/documents/{id}` — bitta hujjat
- `GET /api/v1/rag/documents/{id}/chunks` — ajratilgan chunklar
- `POST /api/v1/rag/documents/{id}/reprocess` — qayta extract/chunk
- `DELETE /api/v1/rag/documents/{id}` — DB yozuvi va private faylni o‘chirish

## Xavfsizlik

- Faqat teacher JWT endpointlarga kira oladi.
- `task_id` berilsa, task aynan shu o‘qituvchiga tegishli bo‘lishi shart.
- Original filename filesystem nomi sifatida ishlatilmaydi.
- Fayl extensioniga ishonib qolmaymiz: PDF magic bytes va DOCX zip struktura tekshiriladi.
- Maksimal fayl hajmi default 25 MB.
- RAG storage `/uploads` ostida emas va FastAPI StaticFiles orqali public qilinmagan.
- Embedding vector yoki private disk path API response'ga chiqarilmaydi.

## Chunking defaults

- `RAG_CHUNK_SIZE_CHARS=1800`
- `RAG_CHUNK_OVERLAP_CHARS=250`

PDF chunklari sahifa chegarasidan tashqariga o‘tmaydi, shuning uchun keyinchalik source page ko‘rsatish oson. DOCX pagination barqaror bo‘lmagani uchun page raqamlari `NULL`, lekin heading/section title saqlanadi.

## Hozircha qilinmagan

- OCR
- embedding generation
- HNSW index
- similarity search
- chatga RAG context ulash

Ular keyingi bosqichlarda qo‘shiladi.
