# Storage v1 — Cloudflare R2

Production fayllari Render ephemeral filesystemida emas, private Cloudflare R2 bucketida saqlanadi.

## Environment

```env
STORAGE_PROVIDER=r2
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET_NAME=mustaqil-talim-ai
R2_PRESIGNED_GET_TTL_SECONDS=900
```

Local development uchun `STORAGE_PROVIDER=local` qoldiriladi. Bunda public task/submission/result fayllari `app/uploads`, RAG original hujjatlari esa `app/rag_storage` ichida saqlanadi.

## Object key layout

- `tasks/teachers/<teacher_id>/references/...`
- `tasks/teachers/<teacher_id>/instructions/...`
- `submissions/students/<student_id>/tasks/<task_id>/attempt-<n>/...`
- `results/submissions/<submission_id>/...`
- `rag/teachers/<teacher_id>/documents/...`

PostgreSQL URL emas, object key saqlaydi. API response qisqa muddatli presigned GET URL qaytaradi.

## Smoke test

`.env` to‘ldirilgach:

```bash
python scripts/check_storage.py
```

Kutilgan natija: upload, HEAD, presigned URL va cleanup — `OK`.

## Legacy migration

Avval dry-run:

```bash
python scripts/migrate_local_storage_to_r2.py
```

Mavjud lokal fayllarni R2 ga ko‘chirish va DB pathlarni object keyga almashtirish:

```bash
python scripts/migrate_local_storage_to_r2.py --apply
```

`--apply` faqat `STORAGE_PROVIDER=r2` bo‘lganda ishlaydi. Render restartida allaqachon yo‘qolgan fayllar `MISSING` sifatida ko‘rsatiladi va tashqi nusxadan qayta yuklanishi kerak.
