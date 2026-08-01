"""Provider-independent contracts for structured LLM generation."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from pydantic import BaseModel


class LLMOperation(StrEnum):
    """Supported application-owned LLM operations."""

    TICKET_CLASSIFICATION = "ticket_classification"


@dataclass(frozen=True, slots=True)
class LLMTokenUsage:
    """Provider-reported token usage with unknown values preserved as null."""

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        token_fields = {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }
        for field_name, value in token_fields.items():
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative when reported.")

        if (
            self.cached_input_tokens is not None
            and self.input_tokens is not None
            and self.cached_input_tokens > self.input_tokens
        ):
            raise ValueError("cached_input_tokens cannot exceed input_tokens.")

        if (
            self.reasoning_tokens is not None
            and self.output_tokens is not None
            and self.reasoning_tokens > self.output_tokens
        ):
            raise ValueError("reasoning_tokens cannot exceed output_tokens.")

        if (
            self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens is not None
            and self.total_tokens != self.input_tokens + self.output_tokens
        ):
            raise ValueError("total_tokens must equal input_tokens plus output_tokens.")


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """Application-owned request supplied to one configured LLM provider."""

    operation: LLMOperation
    model: str
    instructions: str
    input: str
    output_schema: type[BaseModel]
    timeout_seconds: float
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_required_text(self.model, field_name="model")
        _validate_required_text(self.instructions, field_name="instructions")
        _validate_required_text(self.input, field_name="input")

        if not isinstance(self.output_schema, type) or not issubclass(
            self.output_schema,
            BaseModel,
        ):
            raise TypeError("output_schema must be a Pydantic BaseModel type.")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        normalized_metadata = dict(self.metadata)
        for key, value in normalized_metadata.items():
            _validate_required_text(key, field_name="metadata key")
            _validate_required_text(value, field_name=f"metadata[{key!r}]")

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(normalized_metadata),
        )


@dataclass(frozen=True, slots=True)
class LLMProviderResponse:
    """Successful provider result without SDK-specific response objects."""

    parsed_output: Mapping[str, object]
    provider: str
    model: str
    provider_request_id: str | None = None
    usage: LLMTokenUsage | None = None
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        _validate_required_text(self.provider, field_name="provider")
        _validate_required_text(self.model, field_name="model")
        _validate_optional_text(
            self.provider_request_id,
            field_name="provider_request_id",
        )
        _validate_optional_text(self.finish_reason, field_name="finish_reason")

        object.__setattr__(
            self,
            "parsed_output",
            MappingProxyType(dict(self.parsed_output)),
        )


class LLMProvider(Protocol):
    """Asynchronous provider adapter used by the application-owned gateway."""

    @property
    def provider_name(self) -> str:
        """Return the stable provider identifier used for provenance."""
        ...

    async def generate(self, request: LLMRequest) -> LLMProviderResponse:
        """Generate one structured result or raise an application-owned error."""
        ...

    async def close(self) -> None:
        """Release provider-owned process resources."""
        ...


def _validate_required_text(value: str, *, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")


def _validate_optional_text(value: str | None, *, field_name: str) -> None:
    if value is not None:
        _validate_required_text(value, field_name=field_name)
