# AI Mentor migration va diagnostik savollar seed'i

## Yangi Alembic revision

- Revision: `8d4063b0ab9f`
- Oldingi revision: `9ed5260c4037`
- Fayl: `alembic/versions/8d4063b0ab9f_create_ai_mentor_tables.py`

Migration quyidagi 8 ta jadvalni yaratadi:

1. `ai_mentor_diagnostic_questions`
2. `ai_mentor_diagnostic_sessions`
3. `ai_mentor_diagnostic_answers`
4. `ai_mentor_plans`
5. `ai_mentor_plan_weeks`
6. `ai_mentor_plan_items`
7. `ai_mentor_chat_sessions`
8. `ai_mentor_chat_messages`

## Lokal bazaga qo‘llash

Backend virtual muhiti faol bo‘lgan terminalda:

```powershell
alembic current
alembic heads
alembic upgrade head
```

Migrationdan keyin joriy revision:

```text
8d4063b0ab9f (head)
```

## Diagnostik savollarni bazaga yozish

Migration muvaffaqiyatli tugagandan keyin:

```powershell
python scripts/seed_ai_mentor_questions.py
```

Birinchi ishga tushirishdagi kutiladigan natija:

```text
AI Mentor savollari tayyor: created=10, updated=0, unchanged=0, total=10
```

Skriptni qayta ishga tushirish xavfsiz. Ikkinchi ishga tushirishda dublikat yaratilmaydi:

```text
AI Mentor savollari tayyor: created=0, updated=0, unchanged=10, total=10
```

## Tekshiruv

```powershell
python -c "from app.db.database import SessionLocal; from app.models.ai_mentor import AIMentorDiagnosticQuestion; db=SessionLocal(); print('Savollar:', db.query(AIMentorDiagnosticQuestion).count()); db.close()"
```

Kutiladigan natija:

```text
Savollar: 10
```

## Downgrade

Faqat migrationni bekor qilish zarur bo‘lsa:

```powershell
alembic downgrade 9ed5260c4037
```

Bu buyruq barcha AI Mentor jadvallari va ulardagi ma’lumotlarni o‘chiradi. Ishlayotgan bazada downgrade qilishdan oldin zaxira nusxa olish zarur.

## Git commit

```powershell
git add .
git commit -m "Add AI mentor database migration"
git push
```
