# AI Mentor streaming chat

## Maqsad

Frontend AI Mentor javobini to‘liq tugashini kutmasdan, ChatGPT-uslubida bo‘lakma-bo‘lak ko‘rsatishi uchun SSE (Server-Sent Events) endpoint qo‘shildi.

## Endpoint

```text
POST /api/v1/ai-mentor/chat/sessions/{session_id}/messages/stream
Content-Type: application/json
Accept: text/event-stream
```

Request:

```json
{
  "content": "Bugungi vazifani qanday boshlayman?"
}
```

Response media type:

```text
text/event-stream
```

## Eventlar

### start

User xabari PostgreSQL bazaga saqlangach yuboriladi.

```text
event: start
data: {"session_id":1,"provider":"groq","model":"openai/gpt-oss-120b",...}
```

### delta

Modeldan kelgan navbatdagi matn bo‘lagi.

```text
event: delta
data: {"delta":"Bugun avval ..."}
```

### fallback

Provider ishlamay qolsa va `LLM_FALLBACK_TO_MOCK=true` bo‘lsa yuboriladi.

```text
event: fallback
data: {"from_provider":"groq","reason":"provider_request_failed","replace":true}
```

`replace=true` bo‘lsa frontend oldin ko‘rsatgan partial AI matnini tozalab, keyingi `delta` eventlaridan mock javobni yangidan yig‘ishi kerak.

### done

To‘liq assistant xabari bazaga saqlangach yuboriladi.

```text
event: done
data: {"session_id":1,"assistant_message":{...}}
```

### error

Fallback o‘chirilgan bo‘lsa yoki javobni davom ettirish imkonsiz bo‘lsa yuboriladi.

## Saqlash qoidasi

- User xabari stream boshlanishidan oldin DBga commit qilinadi.
- Muvaffaqiyatli LLM javobi to‘liq yig‘ilgach assistant xabari sifatida saqlanadi.
- Provider partial matndan keyin uzilib, mock fallback ishlasa DBga partial LLM emas, yakuniy mock javob saqlanadi.
- Client streamni uzib yuborsa mavjud partial assistant matn imkon qadar `stream_interrupted=true` metadata bilan saqlanadi.
- Eski `POST .../messages` endpoint saqlangan va backward-compatible.

## Frontend uchun tavsiya

Browser `EventSource` faqat GET bilan ishlaydi. Bu endpoint POST body va JWT talab qilgani uchun frontendda `fetch()` + `ReadableStream` orqali SSE matnini parse qilish kerak.

## Lokal test

```powershell
python -m unittest discover -s tests -v
```

Streaming bilan bog‘liq testlar:

- Groq provider delta streaming;
- SSE `start/delta/done` oqimi;
- provider partial javobdan keyin uzilganda mock fallback;
- yopilgan chat uchun streaming endpoint `409` qaytarishi;
- chat tarixi PostgreSQL/SQLAlchemy qatlamida saqlanishi.
