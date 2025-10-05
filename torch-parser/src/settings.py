from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter=".",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    torch_url: str
    path_to_save: str


settings = Settings()  # type: ignore
