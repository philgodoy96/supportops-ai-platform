"""Environment-based application settings."""

import re
from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    AnyHttpUrl,
    BeforeValidator,
    Field,
    PostgresDsn,
    SecretStr,
    TypeAdapter,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from supportops.observability.models import (
    ObservabilityCaptureMode,
    ObservabilityProvider,
)


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


class LLMProviderName(StrEnum):
    """Supported LLM provider adapters."""

    MOCK = "mock"
    OPENAI = "openai"


class EmbeddingProviderName(StrEnum):
    """Supported embedding provider adapters."""

    MOCK = "mock"
    OPENAI = "openai"


class AgentGraphDurability(StrEnum):
    """Supported controlled-agent checkpoint durability modes."""

    SYNC = "sync"


TicketProcessingWorkflowVersion = Literal[
    "deterministic-baseline-v1",
    "ticket-classification-v1",
    "controlled-support-v1",
    "human-approved-support-v1",
]

SupportWorkflowVersion = Literal["controlled-support-v1"]

_WORKER_EXECUTION_SAFETY_MARGIN_SECONDS = 15.0
_CONTROLLED_SUPPORT_LOGICAL_LLM_GENERATIONS = 6
_CHECKPOINT_URL_ERROR = (
    "agent_graph_checkpoint_database_url must be a valid PostgreSQL connection URL."
)
_LANGFUSE_ENVIRONMENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_LANGFUSE_RELEASE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_HTTP_URL_ADAPTER: TypeAdapter[AnyHttpUrl] = TypeAdapter(AnyHttpUrl)


def _validate_http_or_https_url(value: object) -> str:
    """Validate and normalize HTTP or HTTPS URLs without custom URL parsing."""

    if not isinstance(value, str):
        raise ValueError("URL value must be a string")

    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError("value must not be blank")

    validated_url = _HTTP_URL_ADAPTER.validate_python(normalized_value)
    return str(validated_url).rstrip("/")


HttpOrHttpsUrl = Annotated[str, BeforeValidator(_validate_http_or_https_url)]


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
    worker_poll_interval_seconds: float = Field(default=1.0, gt=0, le=60)
    worker_lease_seconds: float = Field(default=150.0, gt=0, le=3600)
    worker_execution_timeout_seconds: float = Field(default=135.0, gt=0, le=1800)
    worker_shutdown_grace_seconds: float = Field(default=10.0, ge=0, le=300)
    worker_max_retryable_failures: int = Field(default=3, ge=1, le=100)
    worker_retry_base_seconds: float = Field(default=2.0, gt=0, le=3600)
    worker_retry_max_seconds: float = Field(default=60.0, gt=0, le=86400)

    approval_ttl_seconds: float = Field(
        default=86400.0,
        gt=0,
        le=2592000,
    )
    approval_expiration_batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
    )

    ticket_processing_workflow_version: TicketProcessingWorkflowVersion = "controlled-support-v1"
    agent_graph_max_steps: int = Field(default=16, ge=8, le=64)
    agent_graph_max_tool_calls: int = Field(default=3, ge=1, le=10)
    agent_graph_max_decision_turns: int = Field(default=4, ge=2, le=11)
    agent_graph_tool_timeout_seconds: float = Field(default=15.0, gt=0, le=60)
    agent_graph_checkpoint_database_url: SecretStr | None = None
    agent_graph_durability: AgentGraphDurability = AgentGraphDurability.SYNC
    support_workflow_version: SupportWorkflowVersion = "controlled-support-v1"
    llm_provider: LLMProviderName = LLMProviderName.MOCK
    openai_api_key: SecretStr | None = None
    openai_model: str = Field(default="gpt-5-nano", min_length=1, max_length=128)
    openai_base_url: str | None = Field(default=None, max_length=2048)
    llm_request_timeout_seconds: float = Field(default=10.0, gt=0, le=300)
    llm_transport_max_retries: int = Field(default=1, ge=0, le=2)
    llm_max_repair_attempts: int = Field(default=1, ge=0, le=1)
    embedding_provider: EmbeddingProviderName = EmbeddingProviderName.MOCK
    embedding_model: str = Field(
        default="mock-hashing-embedding-v1",
        min_length=1,
        max_length=128,
    )
    embedding_dimensions: int = Field(default=64, ge=1, le=4096)
    embedding_request_timeout_seconds: float = Field(
        default=12.0,
        gt=0,
        le=300,
    )
    embedding_transport_max_retries: int = Field(
        default=1,
        ge=0,
        le=2,
    )

    ai_observability_provider: ObservabilityProvider = ObservabilityProvider.NOOP
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: HttpOrHttpsUrl = "https://cloud.langfuse.com"
    langfuse_environment: str = Field(default="local", min_length=1, max_length=64)
    langfuse_release: str | None = Field(default=None, max_length=128)
    langfuse_capture_mode: ObservabilityCaptureMode = ObservabilityCaptureMode.METADATA_ONLY
    langfuse_flush_at_attempt_end: bool = False
    langfuse_timeout_seconds: float = Field(default=5.0, gt=0, le=30)

    qdrant_url: str = Field(min_length=1)
    qdrant_api_key: str | None = None

    dependency_health_timeout_seconds: float = Field(default=2.0, gt=0, le=30)

    @field_validator(
        "application_name",
        "application_version",
        "api_host",
        "openai_model",
        "embedding_model",
        "qdrant_url",
    )
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        """Reject strings that contain only whitespace."""

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError("value must not be blank")

        return normalized_value

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def normalize_openai_api_key(cls, value: object) -> object:
        """Normalize optional OpenAI API keys without exposing secret values."""

        if value is None:
            return None

        if isinstance(value, SecretStr):
            return value

        if isinstance(value, str):
            normalized_value = value.strip()
            return normalized_value or None

        return value

    @field_validator("langfuse_public_key", "langfuse_secret_key", mode="before")
    @classmethod
    def normalize_langfuse_secret(cls, value: object) -> object:
        """Normalize optional Langfuse secrets without exposing secret values."""

        if value is None:
            return None

        if isinstance(value, SecretStr):
            normalized_value = value.get_secret_value().strip()
            return normalized_value or None

        if isinstance(value, str):
            normalized_value = value.strip()
            return normalized_value or None

        return value

    @field_validator("agent_graph_checkpoint_database_url", mode="before")
    @classmethod
    def normalize_agent_graph_checkpoint_database_url(cls, value: object) -> object:
        """Normalize optional checkpoint URLs without exposing secret values."""

        if value is None:
            return None

        if isinstance(value, SecretStr):
            return value

        if isinstance(value, str):
            normalized_value = value.strip()
            return normalized_value or None

        return value

    @field_validator("agent_graph_checkpoint_database_url")
    @classmethod
    def validate_agent_graph_checkpoint_database_url(
        cls,
        value: SecretStr | None,
    ) -> SecretStr | None:
        """Accept only PostgreSQL checkpoint URLs without disclosing credentials."""

        if value is None:
            return None

        parsed_url = urlsplit(value.get_secret_value())
        database_name = parsed_url.path.lstrip("/")

        if (
            parsed_url.scheme not in {"postgres", "postgresql"}
            or not parsed_url.hostname
            or not database_name
        ):
            raise ValueError(_CHECKPOINT_URL_ERROR)

        return value

    @field_validator("openai_base_url")
    @classmethod
    def normalize_openai_base_url(cls, value: str | None) -> str | None:
        """Normalize empty optional OpenAI base URLs to an absent value."""

        if value is None:
            return None

        normalized_value = value.strip()
        return normalized_value or None

    @field_validator("langfuse_environment")
    @classmethod
    def validate_langfuse_environment(cls, value: str) -> str:
        """Normalize and validate Langfuse environment labels."""

        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("langfuse_environment must not be blank")

        if not _LANGFUSE_ENVIRONMENT_PATTERN.fullmatch(normalized_value):
            raise ValueError(
                "langfuse_environment must start with an ASCII letter or digit and "
                "contain only ASCII letters, digits, periods, underscores, or hyphens"
            )

        return normalized_value

    @field_validator("langfuse_release")
    @classmethod
    def validate_langfuse_release(cls, value: str | None) -> str | None:
        """Normalize and validate optional Langfuse release labels."""

        if value is None:
            return None

        normalized_value = value.strip()
        if not normalized_value:
            return None

        if not _LANGFUSE_RELEASE_PATTERN.fullmatch(normalized_value):
            raise ValueError(
                "langfuse_release must start with an ASCII letter or digit and "
                "contain only ASCII letters, digits, periods, underscores, "
                "hyphens, or plus signs"
            )

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

        if (
            self.worker_lease_seconds
            < self.worker_execution_timeout_seconds + _WORKER_EXECUTION_SAFETY_MARGIN_SECONDS
        ):
            raise ValueError(
                "worker_lease_seconds must exceed "
                "worker_execution_timeout_seconds by at least 15 seconds."
            )

        if self.worker_retry_max_seconds < self.worker_retry_base_seconds:
            raise ValueError(
                "worker_retry_max_seconds must not be smaller than worker_retry_base_seconds."
            )

        openai_provider_configured = (
            self.llm_provider is LLMProviderName.OPENAI
            or self.embedding_provider is EmbeddingProviderName.OPENAI
        )
        if openai_provider_configured and self.openai_api_key is None:
            raise ValueError("openai_api_key is required when an OpenAI provider is configured.")

        maximum_logical_invocation_count = 1 + self.llm_max_repair_attempts
        logical_generation_count = (
            _CONTROLLED_SUPPORT_LOGICAL_LLM_GENERATIONS
            if self.ticket_processing_workflow_version
            in {
                "controlled-support-v1",
                "human-approved-support-v1",
            }
            else 1
        )
        logical_llm_budget_seconds = (
            logical_generation_count
            * self.llm_request_timeout_seconds
            * maximum_logical_invocation_count
        )
        if (
            self.worker_execution_timeout_seconds
            < logical_llm_budget_seconds + _WORKER_EXECUTION_SAFETY_MARGIN_SECONDS
        ):
            raise ValueError(
                "worker_execution_timeout_seconds must cover all configured LLM "
                "invocations plus the safety margin."
            )

        minimum_graph_steps = 6 + (2 * self.agent_graph_max_tool_calls)
        if self.agent_graph_max_steps < minimum_graph_steps:
            raise ValueError("agent_graph_max_steps must cover the configured maximum tool loop.")

        if self.agent_graph_max_decision_turns < self.agent_graph_max_tool_calls + 1:
            raise ValueError(
                "agent_graph_max_decision_turns must allow one terminal decision "
                "after all tool calls."
            )

        if self.agent_graph_tool_timeout_seconds >= self.worker_execution_timeout_seconds:
            raise ValueError(
                "agent_graph_tool_timeout_seconds must be smaller than "
                "worker_execution_timeout_seconds."
            )

        return self

    @model_validator(mode="after")
    def validate_langfuse_observability_configuration(self) -> Self:
        """Require Langfuse credentials only when the Langfuse provider is selected."""

        if self.ai_observability_provider is not ObservabilityProvider.LANGFUSE:
            return self

        missing_fields: list[str] = []
        if self.langfuse_public_key is None:
            missing_fields.append("langfuse_public_key")
        if self.langfuse_secret_key is None:
            missing_fields.append("langfuse_secret_key")

        if missing_fields:
            raise ValueError(
                f"{', '.join(missing_fields)} required when ai_observability_provider is langfuse."
            )

        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-level cached settings instance."""

    return Settings()
