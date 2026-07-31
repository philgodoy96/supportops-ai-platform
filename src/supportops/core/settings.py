"""Environment-based application settings."""

from enum import StrEnum
from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, PostgresDsn, field_validator, model_validator
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

    worker_id: str | None = Field(default=None, max_length=128)
    worker_executor: Literal["deterministic-ticket-processing"] = "deterministic-ticket-processing"
    worker_poll_interval_seconds: float = Field(default=1.0, gt=0, le=60)
    worker_lease_seconds: float = Field(default=45.0, gt=0, le=3600)
    worker_execution_timeout_seconds: float = Field(default=30.0, gt=0, le=1800)
    worker_shutdown_grace_seconds: float = Field(default=10.0, ge=0, le=300)
    worker_max_attempts: int = Field(default=3, ge=1, le=100)
    worker_retry_base_seconds: float = Field(default=2.0, gt=0, le=3600)
    worker_retry_max_seconds: float = Field(default=60.0, gt=0, le=86400)

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

    @field_validator("worker_id")
    @classmethod
    def validate_worker_id(cls, value: str | None) -> str | None:
        """Reject empty or whitespace-padded worker identities."""

        if value is None:
            return None

        if not value:
            raise ValueError("worker_id must not be empty")

        if value != value.strip():
            raise ValueError("worker_id must not contain surrounding whitespace")

        return value

    @model_validator(mode="after")
    def validate_worker_timing_relationships(self) -> Self:
        """Reject worker timing relationships that cannot safely coexist."""

        if self.worker_lease_seconds < self.worker_execution_timeout_seconds + 5:
            raise ValueError(
                "worker_lease_seconds must exceed "
                "worker_execution_timeout_seconds by at least 5 seconds."
            )

        if self.worker_retry_max_seconds < self.worker_retry_base_seconds:
            raise ValueError(
                "worker_retry_max_seconds must not be smaller than worker_retry_base_seconds."
            )

        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-level cached settings instance."""

    return Settings()
