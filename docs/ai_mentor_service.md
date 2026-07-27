# AI Mentor service va seed qatlami

## Qo‘shilgan qismlar

- `app/data/ai_mentor_diagnostic_questions.py` — 10 ta versiyalangan diagnostik savol.
- `app/services/ai_mentor_common.py` — servislar uchun umumiy yordamchi funksiyalar.
- `app/services/ai_mentor_diagnostic_service.py` — seed, diagnostika sessiyasi,
  javob validatsiyasi va mock tahlil.
- `app/services/ai_mentor_plan_service.py` — 4 haftalik mock reja, reja versiyasi
  va vazifalar progressi.
- `app/services/ai_mentor_chat_service.py` — chat sessiyalari, xabarlar tarixi va
  mock AI javobi.
- `app/services/ai_mentor_service.py` — routerlar uchun yagona facade/import nuqtasi.
- `scripts/seed_ai_mentor_questions.py` — migrationdan keyin savollarni bazaga
  idempotent kiritish skripti.
- `tests/test_ai_mentor_service.py` — vaqtinchalik SQLite bazasida servis oqimi testi.

## Muhim xatti-harakatlar

1. Seed skripti takroran ishga tushirilsa dublikat yaratmaydi.
2. Har bir `question_code` uchun eng so‘nggi faol versiya ishlatiladi.
3. Har bir talaba uchun tugallanmagan diagnostika sessiyasi qayta ishlatiladi.
4. Faqat faol savollar qabul qilinadi va barcha majburiy savollar tekshiriladi.
5. Javob turi savolning `answer_type` qiymatiga mos validatsiya qilinadi.
6. Diagnostika yakunlanganda mock tahlil `analysis_summary` va `analysis_json`ga yoziladi.
7. Mock reja aynan 4 hafta va 12 ta vazifadan iborat bo‘ladi.
8. Yangi active reja yaratilganda oldingi active reja `archived` qilinadi.
9. Talaba faqat o‘z sessiyasi, rejasi, vazifasi va chatiga murojaat qila oladi.
10. Mock chat foydalanuvchi va assistant xabarlarini ketma-ket raqam bilan saqlaydi.

## Tekshirish

```bash
python -m compileall app scripts tests
python -m unittest discover -s tests -v
```

## Keyingi bosqich

Alembic migration yaratiladi va bazaga qo‘llanadi. Shundan keyin:

```bash
python scripts/seed_ai_mentor_questions.py
```

Keyingi qadamda FastAPI router endpointlari servis funksiyalariga ulanadi.
