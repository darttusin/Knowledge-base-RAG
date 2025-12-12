from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresSettings(BaseModel):
    USER: str
    PASSWORD: str
    HOST: str
    PORT: int = 5432
    DATABASE: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="_",
        extra="ignore",
        case_sensitive=False,
    )

    CHROMADB_PATH: str
    CHROMADB_COLLECTION: str = "docs_fast"

    MODEL_NAME: str = "BAAI/bge-base-en-v1.5"

    GEMINI_API_KEY: str

    POSTGRES: PostgresSettings

    JWT_SECRET_KEY: str = "super-secrey-key"
    JWT_ALGORITHM: str = "HS256"
    ADMIN_DELETE_TOKEN: str = "admin-delete-token"

    ADMIN_LOGIN: str = "admin"
    ADMIN_PASSWORD: str = "admin"


settings = Settings()  # type: ignore
