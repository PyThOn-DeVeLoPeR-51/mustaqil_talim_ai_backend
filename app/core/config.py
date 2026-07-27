from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Mustaqil Ta'lim AI Platforma"
    API_V1_PREFIX: str = "/api/v1"

    DATABASE_URL: str

    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    # AI Mentor LLM sozlamalari. Lokal va test muhitida xavfsiz default — mock.
    LLM_PROVIDER: str = "mock"
    LLM_FALLBACK_TO_MOCK: bool = True
    LLM_TIMEOUT_SECONDS: float = 30.0
    LLM_MAX_RETRIES: int = 2
    LLM_MAX_OUTPUT_TOKENS: int = 3000
    LLM_MAX_CHAT_HISTORY: int = 12

    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-5-mini"
    OPENAI_BASE_URL: str | None = None

    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_REASONING_EFFORT: str = "medium"

    # RAG document ingestion sozlamalari. Fayllar /uploads dan tashqarida
    # saqlanadi, shuning uchun public StaticFiles orqali ochilmaydi.
    RAG_STORAGE_DIR: str = "app/rag_storage"
    RAG_MAX_FILE_SIZE_MB: int = 25
    RAG_CHUNK_SIZE_CHARS: int = 1800
    RAG_CHUNK_OVERLAP_CHARS: int = 250

    # RAG embedding: PyTorch talab qilmaydigan lokal ONNX E5 modeli.
    # Default model 100 tilni qo‘llab-quvvatlaydi va 384-o‘lchamli vektor beradi.
    RAG_EMBEDDING_PROVIDER: str = "local_onnx_e5"
    RAG_EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    RAG_EMBEDDING_DIMENSIONS: int = 384
    RAG_EMBEDDING_ONNX_FILE: str = "onnx/model.onnx"
    RAG_EMBEDDING_CACHE_DIR: str = "app/rag_models"
    RAG_EMBEDDING_BATCH_SIZE: int = 16
    RAG_EMBEDDING_MAX_LENGTH: int = 512
    RAG_SEARCH_TOP_K: int = 5
    RAG_SEARCH_MAX_TOP_K: int = 20

    # AI Mentor chat uchun avtomatik RAG retrieval. Talaba faqat o‘z
    # o‘qituvchisining umumiy yoki o‘ziga biriktirilgan topshiriq materiallarini ko‘radi.
    RAG_CHAT_ENABLED: bool = True
    RAG_CHAT_TOP_K: int = 5
    RAG_CHAT_MIN_SCORE: float = 0.45
    RAG_CHAT_MAX_CONTEXT_CHARS: int = 9000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()