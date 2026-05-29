# Mustaqil Ta'lim AI Platforma — Backend

FastAPI backend for teacher/student workflow, task management, submissions, and AI-based drawing evaluation.

## 1. Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Environment

Create `.env` from `.env.example`:

```bash
copy .env.example .env
```

Edit `.env` and set your PostgreSQL password/port:

```env
DATABASE_URL=postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost:5433/mustaqil_talim_db
SECRET_KEY=change-this-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

## 3. Database

Create PostgreSQL database:

```sql
CREATE DATABASE mustaqil_talim_db;
```

Run migrations:

```bash
alembic upgrade head
```

## 4. Run

```bash
uvicorn app.main:app --reload
```

Open:

- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs

## 5. Main API areas

- `/api/v1/auth/*` — teacher/student login
- `/api/v1/students` — teacher student management
- `/api/v1/tasks` — task management
- `/api/v1/submissions` — student uploads
- `/api/v1/results` — evaluated results for teacher/student dashboards

## Notes

- `.env` is ignored by Git and must not be committed.
- Uploaded files and AI result images are ignored by Git.
- Keep `.gitkeep` files inside upload folders so empty directories remain in the repository.
