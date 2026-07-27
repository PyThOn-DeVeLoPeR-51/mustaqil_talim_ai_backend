from pgvector.psycopg2 import register_vector
from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings


engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)


@event.listens_for(engine, "connect")
def register_pgvector_type(dbapi_connection, connection_record) -> None:
    """pgvector type'ni Psycopg2 connection uchun xavfsiz ro‘yxatdan o‘tkazadi.

    Alembic migrationdan oldin `vector` extension hali mavjud bo‘lmasligi mumkin.
    Shu sababli avval extension mavjudligini tekshiramiz va faqat keyin register qilamiz.
    """

    module_name = dbapi_connection.__class__.__module__
    if not module_name.startswith("psycopg2"):
        return

    cursor = dbapi_connection.cursor()
    try:
        cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
        )
        has_vector = bool(cursor.fetchone()[0])
    finally:
        cursor.close()

    if has_vector:
        register_vector(dbapi_connection)


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
