"""Environment-based application settings."""

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApplicationEnvironment(StrEnum):
    """Supported application runtime environments."""

    LOCAL = "local"
    TEST = "test"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    """Supported application log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Validated runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SUPPORTOPS_",
        case_sensitive=False,
        extra="ignore",
    )

    environment: ApplicationEnvironment = ApplicationEnvironment.LOCAL
    application_name: str = Field(
        default="SupportOps AI Platform",
        min_length=1,
        max_length=100,
    )
    application_version: str = Field(
        default="0.1.0",
        min_length=1,
        max_length=32,
    )
    log_level: LogLevel = LogLevel.INFO

    api_host: str = Field(default="127.0.0.1", min_length=1)
    api_port: int = Field(default=8000, ge=1, le=65535)

    postgresql_url: PostgresDsn
    postgresql_pool_size: int = Field(default=5, ge=1, le=50)
    postgresql_max_overflow: int = Field(default=10, ge=0, le=100)
    postgresql_pool_timeout_seconds: float = Field(default=10.0, gt=0, le=60)

    worker_max_attempts: int = Field(default=3, ge=1, le=100)

    qdrant_url: str = Field(min_length=1)
    qdrant_api_key: str | None = None

    dependency_health_timeout_seconds: float = Field(default=2.0, gt=0, le=30)

    @field_validator("application_name", "application_version", "api_host", "qdrant_url")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        """Reject strings that contain only whitespace."""

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError("value must not be blank")

        return normalized_value

    @field_validator("qdrant_api_key")
    @classmethod
    def normalize_optional_secret(cls, value: str | None) -> str | None:
        """Normalize empty optional secrets to an absent value."""

        if value is None:
            return None

        normalized_value = value.strip()
        return normalized_value or None


@lru_cache
def get_settings() -> Settings:
    """Return the process-level cached settings instance."""

    return Settings()
