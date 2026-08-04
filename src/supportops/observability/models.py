"""Application-owned models for AI observability."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type Metadata = Mapping[str, JsonValue]


class ObservabilityProvider(StrEnum):
    """Supported observability backends."""

    NOOP = "noop"
    LANGFUSE = "langfuse"


class ObservabilityCaptureMode(StrEnum):
    """Supported content-export policies."""

    METADATA_ONLY = "metadata_only"
    REDACTED_CONTENT = "redacted_content"


class ObservationType(StrEnum):
    """Application observation taxonomy."""

    AGENT = "agent"
    SPAN = "span"
    GENERATION = "generation"
    EMBEDDING = "embedding"
    RETRIEVER = "retriever"
    TOOL = "tool"
    CHAIN = "chain"
    EVALUATOR = "evaluator"
    EVENT = "event"


class ObservationStatus(StrEnum):
    """Normalized observation completion status."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"


class PricingStatus(StrEnum):
    """Whether an application pricing catalog resolved a cost."""

    KNOWN = "known"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class UsageDetails:
    """Mutually exclusive token-usage buckets.

    ``input_tokens`` represents uncached input when ``cached_input_tokens`` is
    also populated. This prevents cached input from being counted twice.
    """

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        values = {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }

        for name, value in values.items():
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

        component_values = tuple(
            value
            for value in (
                self.input_tokens,
                self.cached_input_tokens,
                self.output_tokens,
                self.reasoning_tokens,
            )
            if value is not None
        )

        if (
            self.total_tokens is not None
            and component_values
            and self.total_tokens != sum(component_values)
        ):
            raise ValueError("total_tokens must equal the sum of the populated token buckets")


@dataclass(frozen=True, slots=True)
class CostDetails:
    """Application-calculated estimated-cost buckets."""

    pricing_status: PricingStatus
    currency: str = "USD"
    input_cost: Decimal | None = None
    cached_input_cost: Decimal | None = None
    output_cost: Decimal | None = None
    reasoning_cost: Decimal | None = None
    total_cost: Decimal | None = None
    pricing_catalog_version: str | None = None

    def __post_init__(self) -> None:
        if (
            len(self.currency) != 3
            or not self.currency.isascii()
            or not self.currency.isalpha()
            or self.currency != self.currency.upper()
        ):
            raise ValueError("currency must be a three-letter uppercase ASCII code")

        values = {
            "input_cost": self.input_cost,
            "cached_input_cost": self.cached_input_cost,
            "output_cost": self.output_cost,
            "reasoning_cost": self.reasoning_cost,
            "total_cost": self.total_cost,
        }

        for name, value in values.items():
            if value is not None and value < Decimal("0"):
                raise ValueError(f"{name} must be non-negative")

        if self.pricing_status is PricingStatus.UNKNOWN and any(
            value is not None for value in values.values()
        ):
            raise ValueError("unknown pricing must not contain estimated cost values")

        component_values = tuple(
            value
            for value in (
                self.input_cost,
                self.cached_input_cost,
                self.output_cost,
                self.reasoning_cost,
            )
            if value is not None
        )

        if (
            self.total_cost is not None
            and component_values
            and self.total_cost != sum(component_values, start=Decimal("0"))
        ):
            raise ValueError("total_cost must equal the sum of the populated cost buckets")

        if self.pricing_catalog_version is not None and not self.pricing_catalog_version.strip():
            raise ValueError("pricing_catalog_version must not be blank")


@dataclass(frozen=True, slots=True)
class TraceAttributes:
    """Attributes required to start one logical trace."""

    trace_id: str
    name: str
    session_id: str | None = None
    metadata: Metadata = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_blank("trace_id", self.trace_id)
        _require_non_blank("name", self.name)

        if self.session_id is not None:
            _require_non_blank("session_id", self.session_id)

        for tag in self.tags:
            _require_non_blank("tag", tag)

        if len(set(self.tags)) != len(self.tags):
            raise ValueError("trace tags must be unique")


@dataclass(frozen=True, slots=True)
class ObservationAttributes:
    """Attributes required to start an observation."""

    name: str
    observation_type: ObservationType
    metadata: Metadata = field(default_factory=dict)
    input_data: JsonValue = None
    provider: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        _require_non_blank("name", self.name)

        if self.provider is not None:
            _require_non_blank("provider", self.provider)

        if self.model is not None:
            _require_non_blank("model", self.model)


@dataclass(frozen=True, slots=True)
class ObservationUpdate:
    """A bounded update applied before an observation ends."""

    status: ObservationStatus | None = None
    metadata: Metadata = field(default_factory=dict)
    output_data: JsonValue = None
    usage: UsageDetails | None = None
    cost: CostDetails | None = None
    error_code: str | None = None
    status_message: str | None = None

    def __post_init__(self) -> None:
        if self.error_code is not None:
            _require_non_blank("error_code", self.error_code)

        if self.status_message is not None:
            _require_non_blank("status_message", self.status_message)


@dataclass(frozen=True, slots=True)
class EventObservation:
    """A discrete event attached to the current trace context."""

    name: str
    metadata: Metadata = field(default_factory=dict)
    status: ObservationStatus = ObservationStatus.OK
    occurred_at: datetime | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        _require_non_blank("name", self.name)

        if self.error_code is not None:
            _require_non_blank("error_code", self.error_code)

        if self.occurred_at is not None and self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")


def _require_non_blank(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
