# AI Mentor Pydantic schema'lari — 2-bosqich

## Fayl

`app/schemas/ai_mentor.py`

## Maqsad

AI Mentor endpointlari uchun kiruvchi payload va qaytuvchi response shakllarini qat'iy belgilash. Schema'lar database modelidan alohida bo'lib, frontenddan noto'g'ri yoki to'liq bo'lmagan ma'lumot kelishini endpointga yetmasdan oldin aniqlaydi.

## Asosiy schema guruhlari

### Diagnostik savollar

- `AIMentorDiagnosticQuestionCreate`
- `AIMentorDiagnosticQuestionUpdate`
- `AIMentorDiagnosticQuestionRead`

Tanlovli savollarda `options_json` bo'sh bo'lishiga yo'l qo'yilmaydi.

### Diagnostika sessiyasi va javoblar

- `AIMentorDiagnosticSessionCreate`
- `AIMentorDiagnosticAnswerCreate`
- `AIMentorDiagnosticAnswersSubmit`
- `AIMentorDiagnosticSessionRead`
- `AIMentorDiagnosticSessionDetail`

Bitta payload ichida bir savolga ikki marta javob yuborish cheklanadi. Har bir javobda `answer_text` yoki `answer_json` qiymatlaridan kamida bittasi mavjud bo'lishi shart.

### To'rt haftalik reja

- `AIMentorPlanCreate`
- `AIMentorPlanUpdate`
- `AIMentorPlanRead`
- `AIMentorPlanDetail`
- `AIMentorPlanProgress`
- `AIMentorPlanItemProgressUpdate`

Yangi reja aynan 4 ta hafta oladi va hafta raqamlari `1, 2, 3, 4` bo'lishi shart. Har bir haftadagi `item_order` qiymatlari noyob bo'lishi tekshiriladi.

### Chat

- `AIMentorChatSessionCreate`
- `AIMentorChatSessionUpdate`
- `AIMentorChatRequest`
- `AIMentorChatMessageCreate`
- `AIMentorChatSessionRead`
- `AIMentorChatSessionDetail`
- `AIMentorChatResponse`

Bo'sh yoki faqat probeldan iborat xabarlar qabul qilinmaydi.

## Keyingi bosqich

1. Diagnostik savollar uchun boshlang'ich seed ma'lumotlarini yaratish.
2. `ai_mentor_service.py` service qatlamini yozish.
3. Talaba faqat o'z diagnostika, reja va chat ma'lumotlariga kira olishini tekshirish.
4. Mock AI yordamida diagnostika tahlili va 4 haftalik reja generatsiyasini yaratish.
5. Router endpointlarini qo'shish.
