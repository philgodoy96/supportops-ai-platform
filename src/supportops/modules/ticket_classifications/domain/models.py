"""Durable ticket-classification and LLM invocation domain entities."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

from supportops.ai.gateway.contracts import LLMTokenUsage
from supportops.ai.gateway.errors import LLMErrorCode
from supportops.ai.gateway.results import LLMInvocationStatus
from supportops.ai.pricing.estimation import LLMCostEstimate
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketClassificationSchemaVersion,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)

TICKET_CLASSIFICATION_SUMMARY_MAX_LENGTH = 500
TICKET_CLASSIFICATION_PROVIDER_MAX_LENGTH = 64
TICKET_CLASSIFICATION_MODEL_MAX_LENGTH = 128
TICKET_CLASSIFICATION_PROMPT_ID_MAX_LENGTH = 128
TICKET_CLASSIFICATION_SCHEMA_VERSION_MAX_LENGTH = 128

LLM_INVOCATION_PROVIDER_REQUEST_ID_MAX_LENGTH = 255
LLM_INVOCATION_PRICING_CATALOG_VERSION_MAX_LENGTH = 128
LLM_INVOCATION_MONETARY_SCALE = 12
LLM_INVOCATION_MAX_COST_USD = Decimal(
    "99999999.999999999999",
)

_PROMPT_CONTENT_HASH_LENGTH = 64
_LOWERCASE_HEXADECIMAL_CHARACTERS = frozenset(
    "0123456789abcdef",
)


@dataclass(frozen=True, slots=True)
class TicketClassification:
    """One immutable accepted classification for a durable AgentRun."""

    id: UUID
    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    category: TicketCategory
    intent: TicketIntent
    urgency: TicketUrgency
    sentiment: TicketSentiment
    requires_human_review: bool
    summary: str
    schema_version: TicketClassificationSchemaVersion
    prompt_id: str
    prompt_version: int
    prompt_content_hash: str
    provider: str
    model: str
    accepted_llm_invocation_id: UUID
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _validate_classification_taxonomy(self)

        if type(self.requires_human_review) is not bool:
            raise ValueError(
                "requires_human_review must be a boolean.",
            )

        _validate_bounded_text(
            self.summary,
            field_name="summary",
            maximum_length=(TICKET_CLASSIFICATION_SUMMARY_MAX_LENGTH),
        )

        if self.schema_version != TICKET_CLASSIFICATION_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {TICKET_CLASSIFICATION_SCHEMA_VERSION}.",
            )

        _validate_bounded_identifier(
            self.prompt_id,
            field_name="prompt_id",
            maximum_length=(TICKET_CLASSIFICATION_PROMPT_ID_MAX_LENGTH),
        )

        if self.prompt_version <= 0:
            raise ValueError("prompt_version must be positive.")

        _validate_sha256_hash(
            self.prompt_content_hash,
            field_name="prompt_content_hash",
        )
        _validate_bounded_identifier(
            self.provider,
            field_name="provider",
            maximum_length=(TICKET_CLASSIFICATION_PROVIDER_MAX_LENGTH),
        )
        _validate_bounded_identifier(
            self.model,
            field_name="model",
            maximum_length=(TICKET_CLASSIFICATION_MODEL_MAX_LENGTH),
        )

        _validate_utc_timestamp(
            self.created_at,
            field_name="created_at",
        )
        _validate_utc_timestamp(
            self.updated_at,
            field_name="updated_at",
        )

        if self.updated_at != self.created_at:
            raise ValueError(
                "updated_at must equal created_at for an immutable classification.",
            )

    @classmethod
    def create(
        cls,
        *,
        workspace_id: UUID,
        ticket_id: UUID,
        agent_run_id: UUID,
        category: TicketCategory,
        intent: TicketIntent,
        urgency: TicketUrgency,
        sentiment: TicketSentiment,
        requires_human_review: bool,
        summary: str,
        schema_version: TicketClassificationSchemaVersion,
        prompt_id: str,
        prompt_version: int,
        prompt_content_hash: str,
        provider: str,
        model: str,
        accepted_llm_invocation_id: UUID,
        classification_id: UUID | None = None,
        now: datetime | None = None,
    ) -> "TicketClassification":
        """Create one immutable accepted classification."""

        created_at = now or datetime.now(UTC)

        return cls(
            id=classification_id or uuid4(),
            workspace_id=workspace_id,
            ticket_id=ticket_id,
            agent_run_id=agent_run_id,
            category=category,
            intent=intent,
            urgency=urgency,
            sentiment=sentiment,
            requires_human_review=requires_human_review,
            summary=summary.strip(),
            schema_version=schema_version,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            prompt_content_hash=prompt_content_hash,
            provider=provider,
            model=model,
            accepted_llm_invocation_id=accepted_llm_invocation_id,
            created_at=created_at,
            updated_at=created_at,
        )


@dataclass(frozen=True, slots=True)
class LLMInvocation:
    """Durable metadata for one logical provider invocation."""

    id: UUID
    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    agent_run_attempt_id: UUID
    invocation_sequence: int
    status: LLMInvocationStatus
    provider: str
    model: str
    provider_request_id: str | None
    prompt_id: str
    prompt_version: int
    prompt_content_hash: str
    schema_version: str
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None
    pricing_catalog_version: str
    pricing_found: bool
    estimated_input_cost_usd: Decimal | None
    estimated_cached_input_cost_usd: Decimal | None
    estimated_output_cost_usd: Decimal | None
    estimated_total_cost_usd: Decimal | None
    latency_ms: int
    error_code: LLMErrorCode | None
    created_at: datetime

    def __post_init__(self) -> None:
        if self.invocation_sequence <= 0:
            raise ValueError(
                "invocation_sequence must be positive.",
            )

        if not isinstance(
            self.status,
            LLMInvocationStatus,
        ):
            raise ValueError(
                "status must be a supported LLMInvocationStatus.",
            )

        _validate_bounded_identifier(
            self.provider,
            field_name="provider",
            maximum_length=(TICKET_CLASSIFICATION_PROVIDER_MAX_LENGTH),
        )
        _validate_bounded_identifier(
            self.model,
            field_name="model",
            maximum_length=(TICKET_CLASSIFICATION_MODEL_MAX_LENGTH),
        )
        _validate_optional_bounded_identifier(
            self.provider_request_id,
            field_name="provider_request_id",
            maximum_length=(LLM_INVOCATION_PROVIDER_REQUEST_ID_MAX_LENGTH),
        )
        _validate_bounded_identifier(
            self.prompt_id,
            field_name="prompt_id",
            maximum_length=(TICKET_CLASSIFICATION_PROMPT_ID_MAX_LENGTH),
        )

        if self.prompt_version <= 0:
            raise ValueError("prompt_version must be positive.")

        _validate_sha256_hash(
            self.prompt_content_hash,
            field_name="prompt_content_hash",
        )
        _validate_bounded_identifier(
            self.schema_version,
            field_name="schema_version",
            maximum_length=(TICKET_CLASSIFICATION_SCHEMA_VERSION_MAX_LENGTH),
        )

        _validate_token_usage(self)

        _validate_bounded_identifier(
            self.pricing_catalog_version,
            field_name="pricing_catalog_version",
            maximum_length=(LLM_INVOCATION_PRICING_CATALOG_VERSION_MAX_LENGTH),
        )

        if type(self.pricing_found) is not bool:
            raise ValueError(
                "pricing_found must be a boolean.",
            )

        _validate_cost_estimate(self)

        if self.latency_ms < 0:
            raise ValueError(
                "latency_ms must be non-negative.",
            )

        _validate_invocation_error_state(self)

        _validate_utc_timestamp(
            self.created_at,
            field_name="created_at",
        )

    @classmethod
    def create(
        cls,
        *,
        workspace_id: UUID,
        ticket_id: UUID,
        agent_run_id: UUID,
        agent_run_attempt_id: UUID,
        invocation_sequence: int,
        status: LLMInvocationStatus,
        provider: str,
        model: str,
        provider_request_id: str | None,
        prompt_id: str,
        prompt_version: int,
        prompt_content_hash: str,
        schema_version: str,
        input_tokens: int | None,
        cached_input_tokens: int | None,
        output_tokens: int | None,
        reasoning_tokens: int | None,
        total_tokens: int | None,
        pricing_catalog_version: str,
        pricing_found: bool,
        estimated_input_cost_usd: Decimal | None,
        estimated_cached_input_cost_usd: Decimal | None,
        estimated_output_cost_usd: Decimal | None,
        estimated_total_cost_usd: Decimal | None,
        latency_ms: int,
        error_code: LLMErrorCode | None,
        invocation_id: UUID | None = None,
        now: datetime | None = None,
    ) -> "LLMInvocation":
        """Create one durable logical invocation record."""

        return cls(
            id=invocation_id or uuid4(),
            workspace_id=workspace_id,
            ticket_id=ticket_id,
            agent_run_id=agent_run_id,
            agent_run_attempt_id=agent_run_attempt_id,
            invocation_sequence=invocation_sequence,
            status=status,
            provider=provider,
            model=model,
            provider_request_id=provider_request_id,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            prompt_content_hash=prompt_content_hash,
            schema_version=schema_version,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            pricing_catalog_version=pricing_catalog_version,
            pricing_found=pricing_found,
            estimated_input_cost_usd=estimated_input_cost_usd,
            estimated_cached_input_cost_usd=(estimated_cached_input_cost_usd),
            estimated_output_cost_usd=(estimated_output_cost_usd),
            estimated_total_cost_usd=(estimated_total_cost_usd),
            latency_ms=latency_ms,
            error_code=error_code,
            created_at=now or datetime.now(UTC),
        )


def _validate_classification_taxonomy(
    classification: TicketClassification,
) -> None:
    taxonomy_values = (
        (
            classification.category,
            TicketCategory,
            "category",
        ),
        (
            classification.intent,
            TicketIntent,
            "intent",
        ),
        (
            classification.urgency,
            TicketUrgency,
            "urgency",
        ),
        (
            classification.sentiment,
            TicketSentiment,
            "sentiment",
        ),
    )

    for value, expected_type, field_name in taxonomy_values:
        if not isinstance(value, expected_type):
            raise ValueError(
                f"{field_name} must use the supported taxonomy.",
            )


def _validate_token_usage(
    invocation: LLMInvocation,
) -> None:
    LLMTokenUsage(
        input_tokens=invocation.input_tokens,
        cached_input_tokens=invocation.cached_input_tokens,
        output_tokens=invocation.output_tokens,
        reasoning_tokens=invocation.reasoning_tokens,
        total_tokens=invocation.total_tokens,
    )


def _validate_cost_estimate(
    invocation: LLMInvocation,
) -> None:
    costs = (
        (
            invocation.estimated_input_cost_usd,
            "estimated_input_cost_usd",
        ),
        (
            invocation.estimated_cached_input_cost_usd,
            "estimated_cached_input_cost_usd",
        ),
        (
            invocation.estimated_output_cost_usd,
            "estimated_output_cost_usd",
        ),
        (
            invocation.estimated_total_cost_usd,
            "estimated_total_cost_usd",
        ),
    )

    for value, field_name in costs:
        _validate_optional_cost(
            value,
            field_name=field_name,
        )

    LLMCostEstimate(
        pricing_catalog_version=(invocation.pricing_catalog_version),
        pricing_found=invocation.pricing_found,
        estimated_input_cost_usd=(invocation.estimated_input_cost_usd),
        estimated_cached_input_cost_usd=(invocation.estimated_cached_input_cost_usd),
        estimated_output_cost_usd=(invocation.estimated_output_cost_usd),
        estimated_total_cost_usd=(invocation.estimated_total_cost_usd),
    )


def _validate_invocation_error_state(
    invocation: LLMInvocation,
) -> None:
    if invocation.error_code is not None and not isinstance(
        invocation.error_code,
        LLMErrorCode,
    ):
        raise ValueError(
            "error_code must be a supported LLMErrorCode.",
        )

    if invocation.status is LLMInvocationStatus.SUCCEEDED:
        if invocation.error_code is not None:
            raise ValueError(
                "Successful invocations cannot define an error_code.",
            )
        return

    if invocation.error_code is None:
        raise ValueError(
            "Failed invocations require an error_code.",
        )


def _validate_optional_cost(
    value: Decimal | None,
    *,
    field_name: str,
) -> None:
    if value is None:
        return

    if not isinstance(value, Decimal):
        raise TypeError(
            f"{field_name} must be a Decimal.",
        )

    if not value.is_finite():
        raise ValueError(
            f"{field_name} must be finite.",
        )

    if value < 0:
        raise ValueError(
            f"{field_name} must be non-negative.",
        )

    if value > LLM_INVOCATION_MAX_COST_USD:
        raise ValueError(
            f"{field_name} exceeds the supported maximum.",
        )

    exponent = value.as_tuple().exponent
    if isinstance(exponent, int) and exponent < (-LLM_INVOCATION_MONETARY_SCALE):
        raise ValueError(
            f"{field_name} exceeds the supported decimal scale.",
        )


def _validate_sha256_hash(
    value: str,
    *,
    field_name: str,
) -> None:
    if len(value) != _PROMPT_CONTENT_HASH_LENGTH:
        raise ValueError(
            f"{field_name} must be a lowercase SHA-256 hash.",
        )

    if any(character not in _LOWERCASE_HEXADECIMAL_CHARACTERS for character in value):
        raise ValueError(
            f"{field_name} must be a lowercase SHA-256 hash.",
        )


def _validate_bounded_identifier(
    value: str,
    *,
    field_name: str,
    maximum_length: int,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")

    if value != value.strip():
        raise ValueError(
            f"{field_name} must not contain surrounding whitespace.",
        )

    if len(value) > maximum_length:
        raise ValueError(
            f"{field_name} exceeds the maximum length.",
        )


def _validate_bounded_text(
    value: str,
    *,
    field_name: str,
    maximum_length: int,
) -> None:
    _validate_bounded_identifier(
        value,
        field_name=field_name,
        maximum_length=maximum_length,
    )


def _validate_optional_bounded_identifier(
    value: str | None,
    *,
    field_name: str,
    maximum_length: int,
) -> None:
    if value is None:
        return

    _validate_bounded_identifier(
        value,
        field_name=field_name,
        maximum_length=maximum_length,
    )


def _validate_utc_timestamp(
    value: datetime,
    *,
    field_name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(
            f"{field_name} must be a UTC-aware timestamp.",
        )
