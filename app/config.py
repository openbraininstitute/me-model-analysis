"""Configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings class."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=True,
    )

    ENVIRONMENT: str = "dev"
    APP_NAME: str = "accounting-service"
    APP_VERSION: str | None = None
    APP_DEBUG: bool = False
    COMMIT_SHA: str | None = None

    UVICORN_PORT: int = 8000

    ALLOWED_ORIGIN: list[str] = ["*"]

    NEXUS_DELTA_BASE_URL: str = "https://staging.openbraininstitute.org/api/nexus/v1"

    APIGW_ENDPOINT: str | None = None
    APIGW_REGION: str | None = None
    APIGW_CONN_ID: str | None = None

    LOG_LEVEL: str = "DEBUG"
    LOG_FORMAT: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    LOG_SERIALIZE: bool = True
    LOG_BACKTRACE: bool = False
    LOG_DIAGNOSE: bool = False
    LOG_ENQUEUE: bool = False
    LOG_CATCH: bool = True
    LOG_STANDARD_LOGGER: dict[str, str] = {"root": "DEBUG"}


settings = Settings()
