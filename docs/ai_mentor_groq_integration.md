# AI Mentor — Groq integratsiyasi

## Maqsad

AI Mentor diagnostikasi, 4 haftalik reja generatsiyasi va chat javoblarini
Groq orqali `openai/gpt-oss-120b` modeliga ulash. Mock provider fallback sifatida
saqlanadi.

## Nega alohida `groq_provider.py`?

Loyiha provider-abstraksiyasini saqlaydi. Groq OpenAI-compatible API bergani
uchun mavjud `openai` Python SDK qayta ishlatiladi, ammo Groq base URL va Groq
modeli bilan. Bu qo‘shimcha `groq` dependency talab qilmaydi.

## `.env`

```env
LLM_PROVIDER=groq
LLM_FALLBACK_TO_MOCK=true
LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=2
LLM_MAX_OUTPUT_TOKENS=3000
LLM_MAX_CHAT_HISTORY=12

GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b
GROQ_REASONING_EFFORT=medium

OPENAI_API_KEY=
OPENAI_MODEL=gpt-5-mini
OPENAI_BASE_URL=
```

API key `.env` ichida qoladi va Git'ga commit qilinmaydi.

## Structured Outputs

Diagnostika va 4 haftalik reja `json_schema` + `strict=true` bilan olinadi va
keyin Pydantic model bilan yana validatsiya qilinadi. Chat esa oddiy matnli
javob qaytaradi.

## Tekshiruv

1. `python -m unittest discover -s tests -v`
2. Backendni qayta ishga tushiring.
3. Student JWT bilan `GET /api/v1/ai-mentor/llm/status` ni tekshiring.
4. Chat sessiyasiga yangi xabar yuboring. Javobdagi `model_name` qiymati
   `openai/gpt-oss-120b`, `metadata_json.provider` esa `groq` bo‘lishi kerak.

Agar Groq rate limit, timeout yoki boshqa provider xatosi qaytarsa va
`LLM_FALLBACK_TO_MOCK=true` bo‘lsa, AI Mentor avtomatik mock javobga qaytadi.
