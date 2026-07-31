"""Contracts for recovering expired AgentRun leases."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from uuid import UUID

from supportops.modules.agent_runs.domain.models import AgentRun


class ExpiredAgentRunDisposition(StrEnum):
    """Persisted run state selected after expired-lease recovery."""

    RETRY_SCHEDULED = "retry_scheduled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RecoverExpiredAgentRunCommand:
    """Values required to recover one expired AgentRun ownership."""

    recovered_at: datetime
    retry_delay_seconds: float
    error_code: str
    error_summary: str

    def __post_init__(self) -> None:
        _validate_utc_timestamp(
            self.recovered_at,
            field_name="recovered_at",
        )

        if self.retry_delay_seconds <= 0:
            raise ValueError(
                "retry_delay_seconds must be greater than zero.",
            )

        _validate_error_text(
            self.error_code,
            field_name="error_code",
        )
        _validate_error_text(
            self.error_summary,
            field_name="error_summary",
        )


@dataclass(frozen=True, slots=True)
class RecoverExpiredAgentRunResult:
    """Result of one successfully persisted expired-lease recovery."""

    agent_run: AgentRun
    expired_lease_token: UUID
    disposition: ExpiredAgentRunDisposition

    def __post_init__(self) -> None:
        expected_status = self.disposition.value

        if self.agent_run.status.value != expected_status:
            raise ValueError(
                "Recovered AgentRun status must match the disposition.",
            )

        if self.agent_run.lease_owner is not None:
            raise ValueError(
                "Recovered AgentRun must not retain a lease owner.",
            )

        if self.agent_run.lease_token is not None:
            raise ValueError(
                "Recovered AgentRun must not retain a lease token.",
            )

        if self.agent_run.lease_expires_at is not None:
            raise ValueError(
                "Recovered AgentRun must not retain lease expiration.",
            )


def _validate_error_text(
    value: str,
    *,
    field_name: str,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")

    if value != value.strip():
        raise ValueError(
            f"{field_name} must not contain surrounding whitespace.",
        )


def _validate_utc_timestamp(
    value: datetime,
    *,
    field_name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be a UTC-aware timestamp.")
