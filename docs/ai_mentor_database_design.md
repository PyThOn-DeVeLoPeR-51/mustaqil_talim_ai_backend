# AI Mentor database modeli — 1-bosqich

## Maqsad

AI Mentor ma'lumotlarini har bir talaba kesimida xavfsiz ajratish, diagnostikani qayta topshirish, 4 haftalik rejani qayta yaratish va chat tarixini keyinchalik LLM/RAG bilan ishlatishga tayyor holda saqlash.

## Jadvallar

### 1. `ai_mentor_diagnostic_questions`

Diagnostik savollar katalogi.

- `question_code` va `version` savolni versiyalash imkonini beradi.
- `answer_type` matn, son, bitta tanlov, ko'p tanlov va boshqa javob formatlarini belgilaydi.
- `options_json` tanlov variantlari yoki qo'shimcha konfiguratsiyani saqlaydi.
- Savol o'chirilmaydi; eskirgan savol `is_active=False` qilinadi.

### 2. `ai_mentor_diagnostic_sessions`

Talabaning bitta diagnostika jarayoni.

- Har bir talaba uchun sessiyalar `version` orqali ketma-ket saqlanadi.
- `analysis_summary` va `analysis_json` mock AI yoki keyinchalik LLM tahlilini saqlash uchun ajratilgan.
- Holatlar: `in_progress`, `completed`, `cancelled`.

### 3. `ai_mentor_diagnostic_answers`

Diagnostika sessiyasidagi har bir savolga berilgan javob.

- Bir sessiyada bir savol uchun faqat bitta javob bo'ladi.
- `answer_text` oddiy javoblar uchun.
- `answer_json` ko'p tanlovli va tuzilmali javoblar uchun.

### 4. `ai_mentor_plans`

Talabaning shaxsiy rejasining bosh yozuvi.

- Rejalar talaba kesimida versiyalanadi.
- Reja diagnostika sessiyasiga bog'lanishi mumkin.
- `generation_source`: `mock`, `llm`, `manual`.
- Holatlar: `draft`, `active`, `completed`, `archived`.

### 5. `ai_mentor_plan_weeks`

Rejaning 1–4-haftalari.

- Har bir rejada bir xil hafta raqami takrorlanmaydi.
- Database darajasida `week_number BETWEEN 1 AND 4` cheklovi mavjud.

### 6. `ai_mentor_plan_items`

Hafta ichidagi alohida mashg'ulot yoki vazifa.

- Mashg'ulot tartibi, kun raqami, davomiyligi va resurslari saqlanadi.
- Talabaning bajarish holati alohida yuritiladi.
- Holatlar: `pending`, `in_progress`, `completed`, `skipped`.

### 7. `ai_mentor_chat_sessions`

Talabaning AI Mentor bilan alohida chat sessiyasi.

- Sessiya kerak bo'lsa muayyan rejaga bog'lanadi.
- `context_json` RAG yoki LLM uchun qo'shimcha kontekstni saqlashi mumkin.
- Holatlar: `active`, `closed`, `archived`.

### 8. `ai_mentor_chat_messages`

Chat sessiyasidagi xabarlar tarixi.

- `sequence_number` xabarlarning qat'iy ketma-ketligini saqlaydi.
- Rollar: `system`, `user`, `assistant`.
- `model_name`, `token_count`, `metadata_json` keyinchalik LLM monitoringi uchun tayyorlangan.

## Asosiy bog'lanishlar

```text
students
  ├── ai_mentor_diagnostic_sessions
  │     └── ai_mentor_diagnostic_answers
  │             └── ai_mentor_diagnostic_questions
  ├── ai_mentor_plans
  │     └── ai_mentor_plan_weeks
  │             └── ai_mentor_plan_items
  └── ai_mentor_chat_sessions
        └── ai_mentor_chat_messages
```

## O'chirish qoidalari

- Talaba o'chirilsa, uning diagnostikasi, rejasi va chatlari `CASCADE` orqali o'chadi.
- Reja o'chirilsa, hafta va reja elementlari o'chadi.
- Chat sessiyasi o'chirilsa, barcha xabarlar o'chadi.
- Diagnostik savol tarixiy javobda ishlatilgan bo'lsa, uni fizik o'chirish cheklanadi (`RESTRICT`).
- Diagnostika sessiyasi yoki reja bilan yumshoq bog'langan yozuvlarda `SET NULL` ishlatiladi.

## Keyingi bosqich

1. Pydantic schema'lar yaratish.
2. Diagnostik savollar uchun boshlang'ich seed ro'yxatini aniqlash.
3. Service qatlamida talaba egaligini tekshirish va javoblarni upsert qilish.
4. Router endpointlarini qo'shish.
5. Modellar yakuniy tasdiqlangach Alembic migration yaratish.
