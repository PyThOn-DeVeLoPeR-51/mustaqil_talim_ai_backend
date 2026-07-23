# AI Mentor LLM integratsiyasi

## Maqsad

Ushbu qatlam AI Mentorning diagnostik tahlili, 4 haftalik reja generatsiyasi va
chat javoblarini providerga bog‘liq bo‘lmagan servis orqali yaratadi. Lokal va
test muhitida `mock` rejimi saqlanadi; haqiqiy API xatosida ixtiyoriy fallback
ham mavjud.

## Arxitektura

```text
app/llm/
├── contracts.py        # Pydantic output sxemalari, metadata va Protocol
├── prompts.py          # diagnostika, reja va chat system promptlari
├── openai_provider.py  # OpenAI Responses API implementatsiyasi
└── factory.py          # .env bo‘yicha provider tanlash
```

Service qatlamlari:

- `ai_mentor_diagnostic_service.py` — javoblarni saqlaydi, LLM tahlilini
  `analysis_summary` va `analysis_json`ga yozadi;
- `ai_mentor_plan_service.py` — LLM natijasini aynan 4 hafta va 12 vazifali
  rejaga aylantiradi;
- `ai_mentor_chat_service.py` — diagnostika, reja, progress va oxirgi xabarlar
  kontekstidan foydalanadi;
- `ai_mentor_service.py` — routerlar uchun yagona import nuqtasi.

## Muhit sozlamalari

`.env` ichida:

```env
LLM_PROVIDER=openai
LLM_FALLBACK_TO_MOCK=true
LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=2
LLM_MAX_OUTPUT_TOKENS=3000
LLM_MAX_CHAT_HISTORY=12

OPENAI_API_KEY=YOUR_REAL_KEY
OPENAI_MODEL=gpt-5-mini
OPENAI_BASE_URL=
```

API kalitni `.env.example` yoki GitHubga yozmang.

`LLM_PROVIDER=mock` bo‘lsa barcha eski mock test va oqimlar o‘zgarishsiz ishlaydi.

## Endpointlar

### Provider holati

```http
GET /api/v1/ai-mentor/llm/status
```

Kalitning o‘zini qaytarmaydi. Faqat provider, model, konfiguratsiya holati va
fallback sozlamasini ko‘rsatadi.

### Provider orqali reja yaratish

```http
POST /api/v1/ai-mentor/plans/generate
```

Body:

```json
{
  "diagnostic_session_id": 1,
  "start_date": "2026-07-22"
}
```

`/plans/mock` eski endpoint sifatida saqlangan. Frontend haqiqiy providerga
o‘tganda `/plans/generate`dan foydalanadi.

### Diagnostika va chat

Diagnostika javoblarini yuborish endpointi provider sozlamasiga qarab avtomatik
LLM yoki mock tahlil qiladi. Chat xabar endpointi ham o‘zgarmaydi:

```http
POST /api/v1/ai-mentor/chat/sessions/{session_id}/messages
```

## Fallback

`LLM_FALLBACK_TO_MOCK=true` bo‘lsa timeout, noto‘g‘ri sxema, kalit yo‘qligi yoki
provider xatosida foydalanuvchi oqimi to‘xtamaydi. Metadata ichida:

```json
{
  "provider": "mock",
  "fallback_from_provider": "openai",
  "fallback_reason": "provider_request_failed"
}
```

kabi texnik belgi saqlanadi.

`LLM_FALLBACK_TO_MOCK=false` bo‘lsa provider xatosida API `503` qaytaradi.

## Saqlanadigan LLM metadata

- diagnostika: `analysis_json.llm_metadata`;
- reja: `generation_metadata`;
- chat: `model_name`, `token_count`, `metadata_json`.

Shuning uchun yangi database ustuni va Alembic migration talab qilinmadi.

## Lokal tekshiruv

```powershell
python -m pip install -r requirements.txt
python -m compileall app scripts tests
python -m unittest discover -s tests -v
uvicorn app.main:app --reload
```

Swaggerda avval:

```http
GET /api/v1/ai-mentor/llm/status
```

tekshiriladi. `provider=openai` va `configured=true` bo‘lsa real API sinoviga
o‘tiladi.
