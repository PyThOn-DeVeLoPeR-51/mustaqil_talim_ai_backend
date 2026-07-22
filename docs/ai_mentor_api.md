# AI Mentor FastAPI endpointlari

Ushbu bosqich AI Mentorning diagnostika, 4 haftalik reja, progress va mock chat oqimini student JWT autentifikatsiyasi orqali API sifatida taqdim etadi.

## Router

Router `app/api/v1/endpoints/ai_mentor.py` faylida joylashgan va `app/api/v1/api.py` ichida quyidagi prefix bilan ulangan:

```text
/api/v1/ai-mentor
```

Barcha endpointlar `get_current_student` dependency'sidan foydalanadi. Talaba faqat o‘z diagnostikasi, rejalari, vazifalari va chat sessiyalariga kira oladi.

## Diagnostika endpointlari

```text
GET  /diagnostic/questions
POST /diagnostic/sessions
GET  /diagnostic/sessions
GET  /diagnostic/sessions/latest
GET  /diagnostic/sessions/{session_id}
POST /diagnostic/sessions/{session_id}/answers
```

Yangi sessiya yaratish so‘rovi body'siz yoki bo‘sh JSON bilan yuborilishi mumkin:

```json
{}
```

Javoblarni yakuniy yuborish namunasi:

```json
{
  "answers": [
    {
      "question_id": 1,
      "answer_json": "deep_learning"
    },
    {
      "question_id": 9,
      "answer_text": "Muhandislik grafikasi"
    }
  ]
}
```

## Reja endpointlari

```text
POST  /plans/mock
GET   /plans
GET   /plans/current
GET   /plans/{plan_id}
PATCH /plan-items/{item_id}/progress
```

Mock reja yaratish payload'i:

```json
{
  "diagnostic_session_id": 1,
  "start_date": "2026-07-22"
}
```

`diagnostic_session_id` yuborilmasa, talabaning eng so‘nggi yakunlangan diagnostikasi ishlatiladi. `start_date` yuborilmasa, serverning joriy sanasi olinadi.

Progress yangilash:

```json
{
  "status": "completed"
}
```

Qo‘llab-quvvatlanadigan holatlar:

```text
pending
in_progress
completed
skipped
```

## Chat endpointlari

```text
POST  /chat/sessions
GET   /chat/sessions
GET   /chat/sessions/{session_id}
PATCH /chat/sessions/{session_id}
POST  /chat/sessions/{session_id}/messages
```

Chat sessiyasi yaratish:

```json
{
  "plan_id": 1,
  "title": "Reja bo‘yicha yordam"
}
```

Mock AI'ga xabar yuborish:

```json
{
  "content": "Rejam bo‘yicha keyingi vazifa qaysi?"
}
```

Chatni yopish:

```json
{
  "status": "closed"
}
```

Yopilgan yoki arxivlangan chatga yangi xabar yuborilsa, backend `409 Conflict` qaytaradi.

## Testlar

API integratsiya testlari:

```text
tests/test_ai_mentor_api.py
```

Test qamrovi:

- diagnostik savollarni olish;
- sessiya yaratish va javoblarni yuborish;
- mock reja yaratish;
- vazifa progressini yangilash;
- chat sessiyasi va mock javob;
- chat tarixini qayta olish;
- yopilgan chatga xabar yuborishni rad etish;
- boshqa talabaning rejasiga kirishni rad etish.

Ishga tushirish:

```powershell
python -m unittest discover -s tests -v
```

Swagger UI orqali qo‘lda tekshirish:

```text
http://127.0.0.1:8000/docs
```
