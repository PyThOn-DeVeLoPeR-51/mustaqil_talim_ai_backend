"""AI Mentor diagnostik savollarini bazaga kiritish skripti.

Ishga tushirish:
    python scripts/seed_ai_mentor_questions.py

Alembic migration bazaga qo‘llanmaguncha bu skriptni ishga tushirmang.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.database import SessionLocal
# Standalone scriptda barcha SQLAlchemy mapper/relationship lar querydan oldin
# ro'yxatdan o'tishi uchun markaziy model registry-ni yuklaymiz.
import app.db.base as _model_registry  # noqa: F401
from app.services.ai_mentor_service import seed_diagnostic_questions


def main() -> None:
    db = SessionLocal()
    try:
        result = seed_diagnostic_questions(db)
        print(
            "AI Mentor savollari tayyor: "
            f"created={result['created']}, "
            f"updated={result['updated']}, "
            f"unchanged={result['unchanged']}, "
            f"total={result['total']}"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
