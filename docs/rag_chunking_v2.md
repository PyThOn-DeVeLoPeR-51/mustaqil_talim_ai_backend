# RAG Chunking v2

## Muammo
DOCX extractor har bir paragrafni alohida `ExtractedBlock` sifatida qaytaradi. Stage 12 chunker esa har bir blockni mustaqil chunklagani sababli qisqa paragraflar deyarli 1:1 chunkga aylanardi (masalan, 164 paragraph -> 162 chunk).

## Yechim
`app/rag/chunker.py` endi chunklashdan oldin qo‘shni blocklarni mantiqiy region bo‘yicha birlashtiradi:

- PDF: sahifa chegaralari qat'iy saqlanadi; turli sahifalar qo‘shilmaydi.
- DOCX: bir xil `section_title`ga ega ketma-ket paragraflar bitta matn oqimiga birlashtiriladi.
- Keyin matn `RAG_CHUNK_SIZE_CHARS` (default 1800) va `RAG_CHUNK_OVERLAP_CHARS` (default 250) bo‘yicha sentence/paragraph chegaralarida bo‘linadi.
- Bo‘lim o‘zgarsa yangi chunk region boshlanadi.
- Chunk metadata `ingestion_version=2` va `chunking_strategy=merged_blocks_v2` bilan belgilanadi.

## Mavjud hujjatlarni yangilash
Yangi migration kerak emas. Oldin yuklangan hujjatni quyidagi endpoint bilan qayta ishlash kifoya:

`POST /api/v1/rag/documents/{document_id}/reprocess`

Eski chunklar o‘chirilib, v2 chunklar qayta yaratiladi.
