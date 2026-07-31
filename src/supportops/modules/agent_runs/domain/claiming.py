"""Domain contracts for atomic AgentRun claiming."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

from supportops.modules.agent_runs.domain.models import (
    AGENT_RUN_LEASE_OWNER_MAX_LENGTH,
    AgentRun,
    AgentRunAttempt,
)


@dataclass(frozen=True, slots=True)
class ClaimAgentRunCommand:
    """Ownership values required for one atomic AgentRun claim."""

    worker_id: str
    lease_token: UUID
    execution_request_id: UUID
    claimed_at: datetime
    lease_expires_at: datetime

    def __post_init__(self) -> None:
        _validate_worker_id(self.worker_id)
        _validate_utc_timestamp(
            self.claimed_at,
            field_name="claimed_at",
        )
        _validate_utc_timestamp(
            self.lease_expires_at,
            field_name="lease_expires_at",
        )

        if self.lease_expires_at <= self.claimed_at:
            raise ValueError(
                "lease_expires_at must be later than claimed_at.",
            )


@dataclass(frozen=True, slots=True)
class AgentRunClaim:
    """AgentRun and attempt created by one successful atomic claim."""

    agent_run: AgentRun
    attempt: AgentRunAttempt

    def __post_init__(self) -> None:
        if self.agent_run.id != self.attempt.agent_run_id:
            raise ValueError(
                "Claimed AgentRun and attempt must reference the same run.",
            )

        if self.agent_run.attempt_count != self.attempt.attempt_number:
            raise ValueError(
                "Claimed AgentRun attempt count must match the attempt number.",
            )

        if self.agent_run.lease_token != self.attempt.lease_token:
            raise ValueError(
                "Claimed AgentRun and attempt must share the lease token.",
            )

        if (
            self.agent_run.lease_owner is not None
            and self.agent_run.lease_owner != self.attempt.worker_id
        ):
            raise ValueError(
                "Claimed AgentRun and attempt must share the worker ID.",
            )


def _validate_worker_id(worker_id: str) -> None:
    if not worker_id:
        raise ValueError("worker_id is required.")

    if worker_id != worker_id.strip():
        raise ValueError(
            "worker_id must not contain surrounding whitespace.",
        )

    if len(worker_id) > AGENT_RUN_LEASE_OWNER_MAX_LENGTH:
        raise ValueError("worker_id exceeds the maximum length.")


def _validate_utc_timestamp(
    value: datetime,
    *,
    field_name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be a UTC-aware timestamp.")
