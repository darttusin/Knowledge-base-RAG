from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter=".",
        extra="ignore",
        case_sensitive=False,
    )

    API_URL: str = "http://localhost:8000/forward"
    API_HISTORY_URL: str = "http://localhost:8000/history"
    API_STATS_URL: str = "http://localhost:8000/stats"
    TGBOT_TOKEN: str
    ALLOWED_USERS: list[int]


settings = Settings()  # type: ignore
