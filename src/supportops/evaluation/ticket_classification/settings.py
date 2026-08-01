"""Environment settings required only by classification evaluation."""

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TicketClassificationEvaluationSettings(BaseSettings):
    """Validated LLM configuration without infrastructure dependencies."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SUPPORTOPS_",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: SecretStr | None = None
    openai_model: str = Field(
        default="gpt-5-nano",
        min_length=1,
        max_length=128,
    )
    openai_base_url: str | None = Field(
        default=None,
        max_length=2048,
    )
    llm_request_timeout_seconds: float = Field(
        default=12.0,
        gt=0,
        le=300,
    )
    llm_transport_max_retries: int = Field(
        default=1,
        ge=0,
        le=2,
    )
    llm_max_repair_attempts: int = Field(
        default=1,
        ge=0,
        le=1,
    )

    @field_validator("openai_model")
    @classmethod
    def reject_blank_openai_model(
        cls,
        value: str,
    ) -> str:
        """Reject blank or padded model identifiers."""

        if value != value.strip():
            raise ValueError(
                "openai_model must not contain surrounding whitespace.",
            )

        if not value:
            raise ValueError(
                "openai_model must not be blank.",
            )

        return value

    @field_validator(
        "openai_api_key",
        mode="before",
    )
    @classmethod
    def normalize_openai_api_key(
        cls,
        value: object,
    ) -> object:
        """Normalize an optional key without exposing its value."""

        if value is None:
            return None

        if isinstance(value, SecretStr):
            return value

        if isinstance(value, str):
            normalized_value = value.strip()
            return normalized_value or None

        return value

    @field_validator("openai_base_url")
    @classmethod
    def normalize_openai_base_url(
        cls,
        value: str | None,
    ) -> str | None:
        """Normalize an optional OpenAI-compatible base URL."""

        if value is None:
            return None

        normalized_value = value.strip()

        return normalized_value or None
