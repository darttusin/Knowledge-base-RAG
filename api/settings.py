from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter=".",
        extra="ignore",
        case_sensitive=False,
    )

    CHROMADB_PATH: str
    CHROMADB_COLLECTION: str = "docs_fast"

    MODEL_NAME: str = "BAAI/bge-base-en-v1.5"

    GEMINI_API_KEY: str


settings = Settings()  # type: ignore
