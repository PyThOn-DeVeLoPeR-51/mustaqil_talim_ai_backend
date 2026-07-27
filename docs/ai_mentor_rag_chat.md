# AI Mentor RAG Chat — Stage 14

## Maqsad

AI Mentor chatiga o‘qituvchi yuklagan va embedding yaratilgan RAG materiallarini avtomatik ulash.

## Oqim

1. Student savol yuboradi.
2. Backend student uchun ruxsat etilgan RAG materiallari borligini tekshiradi.
3. Savol `intfloat/multilingual-e5-small` orqali embeddingga aylantiriladi.
4. pgvector semantic search top-k chunklarni topadi.
5. Topilgan chunklar `knowledge_base.sources` sifatida GPT-OSS 120B kontekstiga beriladi.
6. LLM tegishli joylarda `[Manba 1]`, `[Manba 2]` kabi belgilardan foydalanishi mumkin.
7. Yakuniy assistant message `metadata_json.rag.sources` ichida manba metadata’lari bilan saqlanadi.
8. RAG ishlamasa chat oddiy LLM rejimida davom etadi.

## Student access scope

Student faqat:

- o‘z `teacher_id`iga tegishli `task_id = NULL` umumiy materiallarni;
- o‘ziga `task_assignments` orqali biriktirilgan task materiallarini

ko‘ra oladi.

Boshqa teacher materiallari yoki studentga biriktirilmagan task materiallari retrievalga kirmaydi.

## Sozlamalar

```env
RAG_CHAT_ENABLED=true
RAG_CHAT_TOP_K=5
RAG_CHAT_MIN_SCORE=0.45
RAG_CHAT_MAX_CONTEXT_CHARS=9000
```

## Chat metadata namunasi

```json
{
  "provider": "groq",
  "model": "openai/gpt-oss-120b",
  "stream": true,
  "rag": {
    "enabled": true,
    "status": "ready",
    "used_for_answer": true,
    "source_count": 2,
    "embedding_model": "intfloat/multilingual-e5-small",
    "sources": [
      {
        "source_id": 1,
        "document_id": 1,
        "document_title": "Muhandislik grafikasi 1-ma'ruza",
        "chunk_id": 163,
        "chunk_index": 0,
        "page_number_start": null,
        "page_number_end": null,
        "score": 0.82,
        "excerpt": "..."
      }
    ]
  }
}
```

`content`ning to‘liq nusxasi message metadata’ga qayta yozilmaydi. Faqat UI/manba preview uchun qisqa `excerpt` saqlanadi.

## Graceful fallback

- embedded RAG material yo‘q → oddiy LLM chat;
- semantic search natija topmasa → oddiy LLM chat;
- embedding/pgvector retrieval xato bersa → oddiy LLM chat;
- LLM provider xato bersa → mavjud mock fallback oqimi ishlaydi.

RAG retrieval xatosi asosiy AI Mentor chat xizmatini to‘xtatmaydi.
