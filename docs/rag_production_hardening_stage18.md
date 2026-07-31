# RAG Production Hardening — Stage 18

## Maqsad

RAG hujjatlarini yuklash, chunklash va embedding yaratishni HTTP request ichida uzoq kutmasdan, PostgreSQL’da saqlanadigan job queue orqali bajarish.

## Backward compatibility

Eski endpointlar saqlangan:

- `POST /api/v1/rag/documents` — synchronous upload + chunking
- `POST /api/v1/rag/documents/{id}/embed` — synchronous embedding
- `POST /api/v1/rag/documents/{id}/reprocess` — synchronous reprocess

Yangi production endpointlar:

- `POST /api/v1/rag/documents/background`
- `POST /api/v1/rag/documents/{id}/reprocess/background`
- `POST /api/v1/rag/documents/{id}/embed/background`
- `GET /api/v1/rag/jobs`
- `GET /api/v1/rag/jobs/{job_id}`
- `POST /api/v1/rag/jobs/{job_id}/retry`
- `GET /api/v1/rag/storage/usage`
- `GET /api/v1/rag/monitoring/summary`

## Job lifecycle

`pending → running → succeeded`

Xatolikda eksponensial retry:

`pending → running → pending ... → failed`

Joblar `rag_processing_jobs` jadvalida saqlanadi. Web process restart bo‘lsa pending joblar yo‘qolmaydi. `running` holatida uzoq qolgan stale joblar worker ishga tushganda qayta navbatga qo‘yiladi.

## Worker rejimlari

### In-process worker

Default:

```env
RAG_BACKGROUND_WORKER_ENABLED=true
```

Lokal va bitta web instance uchun qulay.

### Dedicated worker

Web service:

```env
RAG_BACKGROUND_WORKER_ENABLED=false
```

Alohida process:

```bash
python scripts/run_rag_worker.py
```

PostgreSQL `FOR UPDATE SKIP LOCKED` orqali bir nechta worker bir queue’dan xavfsiz job olishi mumkin.

## Auto embedding

Background uploadda `auto_embed=true` bo‘lsa ingestion job tugagach alohida embedding job navbatga qo‘yiladi. Birinchi job `result_json.next_job_id` orqali keyingi job ID’ni qaytaradi.

## Storage limitlari

```env
RAG_TEACHER_STORAGE_LIMIT_MB=1024
RAG_TEACHER_DOCUMENT_LIMIT=100
RAG_STORAGE_MIN_FREE_MB=512
```

- teacher storage kvotasi;
- teacher hujjatlar soni limiti;
- server diskida minimal bo‘sh joy;
- mavjud per-file `RAG_MAX_FILE_SIZE_MB` limiti.

## Monitoring

`GET /api/v1/rag/monitoring/summary` quyidagilarni qaytaradi:

- document statuslari;
- embedding statuslari;
- job statuslari;
- teacher storage usage;
- worker konfiguratsiyasi.

Job `result_json` ichida `duration_ms`, chunk yoki embedding natijalari saqlanadi. Logging text yoki JSON formatda yoqiladi:

```env
LOG_LEVEL=INFO
LOG_JSON=false
```

## Migration

Yangi Alembic head:

```text
e4b7c2d9a6f1
```

Qo‘llash:

```bash
alembic upgrade head
```
