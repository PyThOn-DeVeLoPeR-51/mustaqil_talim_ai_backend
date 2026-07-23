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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()