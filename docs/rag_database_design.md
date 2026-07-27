# RAG database design — Stage 1

## Maqsad

AI Mentor o‘qituvchi yuklagan PDF/DOCX o‘quv materiallariga tayangan holda javob berishi uchun PostgreSQL + pgvector asosiy saqlash qatlamini tayyorlash.

## Jadvallar

### `rag_documents`

Har bir yuklangan o‘quv materialining bosh yozuvi.

Muhim maydonlar:

- `teacher_id` — hujjat egasi;
- `task_id` — material aniq topshiriqqa bog‘langan bo‘lsa;
- `title`, `original_filename`, `stored_file_path`;
- `file_type` — hozircha `pdf` yoki `docx`;
- `checksum_sha256` — keyinchalik dublikatlarni aniqlash uchun;
- `status` — `uploaded`, `processing`, `ready`, `failed`, `archived`;
- `page_count`, `chunk_count`, `processing_error`;
- `metadata_json` — fan, mavzu va boshqa moslashuvchan metadata.

### `rag_chunks`

Hujjatdan ajratilgan matn bo‘laklari va embeddinglar.

Muhim maydonlar:

- `document_id`;
- `chunk_index`;
- `content` va `content_hash`;
- sahifa/bo‘lim/pozitsiya metadata;
- `embedding` — pgvector `vector` ustuni;
- `embedding_model`;
- `embedding_dimensions`.

## Nega `vector(n)` emas, `vector`?

Embedding model hali yakuniy tanlanmagan. pgvector o‘lchami ko‘rsatilmagan `vector` ustunida turli o‘lchamdagi vektorlarni saqlashga ruxsat beradi. Keyingi bosqichda bepul multilingual embedding modelini tanlaganimizdan so‘ng, aynan o‘sha model o‘lchami uchun HNSW indeks yaratamiz.

Bu qaror modelni keyin almashtirishni va eski embeddinglarni qayta indekslashni osonlashtiradi.

## Migration

Revision:

`c3f2a8d4e9b1`

Oldingi revision:

`8d4063b0ab9f`

Migration avval:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

ni bajaradi, keyin `rag_documents` va `rag_chunks` jadvallarini yaratadi.

`downgrade` vaqtida `vector` extension o‘chirilmaydi. Bu xavfsizroq, chunki extension boshqa funksiyalar tomonidan ham ishlatilishi mumkin.

## Keyingi bosqich

1. teacher-only PDF/DOCX upload API;
2. faylni xavfsiz saqlash va SHA-256 hisoblash;
3. PDF/DOCX matnini ajratish;
4. sahifani saqlagan holda chunklash;
5. bepul multilingual embedding modelini tanlash;
6. embeddinglarni yozish va HNSW qidiruv indeksini yaratish.
