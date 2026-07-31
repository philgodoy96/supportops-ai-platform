"""Fenced transition contracts for durable AgentRun execution."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from uuid import UUID

from supportops.modules.agent_runs.domain.models import (
    AGENT_RUN_ERROR_CODE_MAX_LENGTH,
    AGENT_RUN_ERROR_SUMMARY_MAX_LENGTH,
    AgentRunAttemptOutcome,
)


class AgentRunFailureDisposition(StrEnum):
    """Persisted run state selected after a failed attempt."""

    RETRY_SCHEDULED = "retry_scheduled"
    FAILED = "failed"


class AgentRunTransitionResult(StrEnum):
    """Outcome of a lease-token-fenced persistence transition."""

    APPLIED = "applied"
    LEASE_LOST = "lease_lost"


@dataclass(frozen=True, slots=True)
class CompleteAgentRunCommand:
    """Values required to complete an actively leased AgentRun."""

    agent_run_id: UUID
    lease_token: UUID
    finished_at: datetime

    def __post_init__(self) -> None:
        _validate_utc_timestamp(
            self.finished_at,
            field_name="finished_at",
        )


@dataclass(frozen=True, slots=True)
class FailAgentRunCommand:
    """Values required to persist a fenced failed execution attempt."""

    agent_run_id: UUID
    lease_token: UUID
    finished_at: datetime
    outcome: AgentRunAttemptOutcome
    disposition: AgentRunFailureDisposition
    error_code: str
    error_summary: str
    retry_available_at: datetime | None

    def __post_init__(self) -> None:
        _validate_utc_timestamp(
            self.finished_at,
            field_name="finished_at",
        )
        _validate_failure_outcome(self.outcome)
        _validate_error_text(
            self.error_code,
            field_name="error_code",
            maximum_length=AGENT_RUN_ERROR_CODE_MAX_LENGTH,
        )
        _validate_error_text(
            self.error_summary,
            field_name="error_summary",
            maximum_length=AGENT_RUN_ERROR_SUMMARY_MAX_LENGTH,
        )
        _validate_retry_schedule(self)


def _validate_failure_outcome(
    outcome: AgentRunAttemptOutcome,
) -> None:
    supported_outcomes = {
        AgentRunAttemptOutcome.RETRYABLE_FAILURE,
        AgentRunAttemptOutcome.TERMINAL_FAILURE,
        AgentRunAttemptOutcome.TIMED_OUT,
    }

    if outcome not in supported_outcomes:
        raise ValueError(
            "outcome must represent an executor failure.",
        )


def _validate_retry_schedule(
    command: FailAgentRunCommand,
) -> None:
    if command.disposition is AgentRunFailureDisposition.RETRY_SCHEDULED:
        if command.outcome is AgentRunAttemptOutcome.TERMINAL_FAILURE:
            raise ValueError(
                "Terminal failures cannot be retry-scheduled.",
            )

        if command.retry_available_at is None:
            raise ValueError(
                "retry_available_at is required for a retry.",
            )

        _validate_utc_timestamp(
            command.retry_available_at,
            field_name="retry_available_at",
        )

        if command.retry_available_at <= command.finished_at:
            raise ValueError(
                "retry_available_at must be later than finished_at.",
            )

        return

    if command.retry_available_at is not None:
        raise ValueError(
            "retry_available_at must be null for a terminal run.",
        )


def _validate_error_text(
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


def _validate_utc_timestamp(
    value: datetime,
    *,
    field_name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be a UTC-aware timestamp.")
