"""Durable AgentRun domain entities and invariants."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import UUID, uuid4

AGENT_RUN_WORKFLOW_NAME_MAX_LENGTH = 64
AGENT_RUN_WORKFLOW_VERSION_MAX_LENGTH = 64
AGENT_RUN_TRIGGER_KEY_MAX_LENGTH = 64
AGENT_RUN_LEASE_OWNER_MAX_LENGTH = 128
AGENT_RUN_ERROR_CODE_MAX_LENGTH = 64
AGENT_RUN_ERROR_SUMMARY_MAX_LENGTH = 512
AGENT_RUN_ATTEMPT_WORKER_ID_MAX_LENGTH = 128

INITIAL_TICKET_PROCESSING_WORKFLOW_NAME = "ticket-processing"
DETERMINISTIC_BASELINE_WORKFLOW_VERSION = "deterministic-baseline-v1"
TICKET_CLASSIFICATION_WORKFLOW_VERSION = "ticket-classification-v1"
INITIAL_TICKET_PROCESSING_TRIGGER_KEY = "initial-ticket-processing"


class AgentRunStatus(StrEnum):
    """Durable AgentRun lifecycle states."""

    QUEUED = "queued"
    RUNNING = "running"
    RETRY_SCHEDULED = "retry_scheduled"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AgentRunAttemptOutcome(StrEnum):
    """Terminal outcomes for a claimed AgentRun attempt."""

    SUCCEEDED = "succeeded"
    AWAITING_APPROVAL = "awaiting_approval"
    RETRYABLE_FAILURE = "retryable_failure"
    TERMINAL_FAILURE = "terminal_failure"
    TIMED_OUT = "timed_out"
    LEASE_EXPIRED = "lease_expired"


@dataclass(frozen=True, slots=True)
class AgentRun:
    """Durable execution state for processing one workspace-owned ticket."""

    id: UUID
    workspace_id: UUID
    ticket_id: UUID
    workflow_name: str
    workflow_version: str
    trigger_key: str
    status: AgentRunStatus
    available_at: datetime | None
    attempt_count: int
    retryable_failure_count: int
    max_retryable_failures: int
    lease_owner: str | None
    lease_token: UUID | None
    lease_expires_at: datetime | None
    first_started_at: datetime | None
    completed_at: datetime | None
    last_error_code: str | None
    last_error_summary: str | None
    ingestion_request_id: UUID
    correlation_id: UUID
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _validate_bounded_identifier(
            self.workflow_name,
            field_name="workflow_name",
            maximum_length=AGENT_RUN_WORKFLOW_NAME_MAX_LENGTH,
        )
        _validate_bounded_identifier(
            self.workflow_version,
            field_name="workflow_version",
            maximum_length=AGENT_RUN_WORKFLOW_VERSION_MAX_LENGTH,
        )
        _validate_bounded_identifier(
            self.trigger_key,
            field_name="trigger_key",
            maximum_length=AGENT_RUN_TRIGGER_KEY_MAX_LENGTH,
        )
        _validate_agent_run_status(self.status)
        _validate_optional_utc_timestamp(
            self.available_at,
            field_name="available_at",
        )
        _validate_retry_budget(
            attempt_count=self.attempt_count,
            retryable_failure_count=self.retryable_failure_count,
            max_retryable_failures=self.max_retryable_failures,
        )
        _validate_optional_bounded_text(
            self.lease_owner,
            field_name="lease_owner",
            maximum_length=AGENT_RUN_LEASE_OWNER_MAX_LENGTH,
        )
        _validate_optional_utc_timestamp(
            self.lease_expires_at,
            field_name="lease_expires_at",
        )
        _validate_optional_utc_timestamp(
            self.first_started_at,
            field_name="first_started_at",
        )
        _validate_optional_utc_timestamp(
            self.completed_at,
            field_name="completed_at",
        )
        _validate_optional_bounded_text(
            self.last_error_code,
            field_name="last_error_code",
            maximum_length=AGENT_RUN_ERROR_CODE_MAX_LENGTH,
        )
        _validate_optional_bounded_text(
            self.last_error_summary,
            field_name="last_error_summary",
            maximum_length=AGENT_RUN_ERROR_SUMMARY_MAX_LENGTH,
        )
        _validate_utc_timestamp(
            self.created_at,
            field_name="created_at",
        )
        _validate_utc_timestamp(
            self.updated_at,
            field_name="updated_at",
        )
        _validate_agent_run_timestamps(self)
        _validate_started_attempt_state(self)
        _validate_lease_state(self)
        _validate_completion_state(self)
        _validate_error_state(self)

    @classmethod
    def create_initial(
        cls,
        *,
        workspace_id: UUID,
        ticket_id: UUID,
        ingestion_request_id: UUID,
        correlation_id: UUID,
        workflow_version: str,
        max_retryable_failures: int,
        agent_run_id: UUID | None = None,
        now: datetime | None = None,
    ) -> "AgentRun":
        """Create the initial durable processing run for a ticket."""

        created_at = now or datetime.now(UTC)
        return cls(
            id=agent_run_id or uuid4(),
            workspace_id=workspace_id,
            ticket_id=ticket_id,
            workflow_name=INITIAL_TICKET_PROCESSING_WORKFLOW_NAME,
            workflow_version=workflow_version,
            trigger_key=INITIAL_TICKET_PROCESSING_TRIGGER_KEY,
            status=AgentRunStatus.QUEUED,
            available_at=created_at,
            attempt_count=0,
            retryable_failure_count=0,
            max_retryable_failures=max_retryable_failures,
            lease_owner=None,
            lease_token=None,
            lease_expires_at=None,
            first_started_at=None,
            completed_at=None,
            last_error_code=None,
            last_error_summary=None,
            ingestion_request_id=ingestion_request_id,
            correlation_id=correlation_id,
            created_at=created_at,
            updated_at=created_at,
        )

    @property
    def is_terminal(self) -> bool:
        """Return whether the run has reached an immutable terminal state."""

        return self.status in {
            AgentRunStatus.SUCCEEDED,
            AgentRunStatus.FAILED,
        }

    @property
    def retryable_failures_remaining(self) -> int:
        """Return retryable failures that may still be consumed."""

        return self.max_retryable_failures - self.retryable_failure_count


@dataclass(frozen=True, slots=True)
class AgentRunAttempt:
    """Historical record of one claimed AgentRun execution attempt."""

    id: UUID
    agent_run_id: UUID
    attempt_number: int
    worker_id: str
    lease_token: UUID
    execution_request_id: UUID
    started_at: datetime
    finished_at: datetime | None
    outcome: AgentRunAttemptOutcome | None
    error_code: str | None
    error_summary: str | None

    def __post_init__(self) -> None:
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be at least one.")

        _validate_bounded_identifier(
            self.worker_id,
            field_name="worker_id",
            maximum_length=AGENT_RUN_ATTEMPT_WORKER_ID_MAX_LENGTH,
        )
        _validate_utc_timestamp(
            self.started_at,
            field_name="started_at",
        )
        _validate_optional_utc_timestamp(
            self.finished_at,
            field_name="finished_at",
        )
        _validate_optional_attempt_outcome(self.outcome)
        _validate_optional_bounded_text(
            self.error_code,
            field_name="error_code",
            maximum_length=AGENT_RUN_ERROR_CODE_MAX_LENGTH,
        )
        _validate_optional_bounded_text(
            self.error_summary,
            field_name="error_summary",
            maximum_length=AGENT_RUN_ERROR_SUMMARY_MAX_LENGTH,
        )
        _validate_attempt_completion_state(self)

    @classmethod
    def start(
        cls,
        *,
        agent_run_id: UUID,
        attempt_number: int,
        worker_id: str,
        lease_token: UUID,
        execution_request_id: UUID,
        attempt_id: UUID | None = None,
        now: datetime | None = None,
    ) -> "AgentRunAttempt":
        """Create an active attempt for a successfully claimed run."""

        started_at = now or datetime.now(UTC)
        return cls(
            id=attempt_id or uuid4(),
            agent_run_id=agent_run_id,
            attempt_number=attempt_number,
            worker_id=worker_id,
            lease_token=lease_token,
            execution_request_id=execution_request_id,
            started_at=started_at,
            finished_at=None,
            outcome=None,
            error_code=None,
            error_summary=None,
        )

    @property
    def is_finished(self) -> bool:
        """Return whether the attempt has a persisted terminal outcome."""

        return self.finished_at is not None


def _validate_agent_run_status(status: AgentRunStatus) -> None:
    if not isinstance(status, AgentRunStatus):
        raise ValueError("status must be a supported AgentRunStatus.")


def _validate_optional_attempt_outcome(
    outcome: AgentRunAttemptOutcome | None,
) -> None:
    if outcome is not None and not isinstance(
        outcome,
        AgentRunAttemptOutcome,
    ):
        raise ValueError(
            "outcome must be a supported AgentRunAttemptOutcome.",
        )


def _validate_retry_budget(
    *,
    attempt_count: int,
    retryable_failure_count: int,
    max_retryable_failures: int,
) -> None:
    if attempt_count < 0:
        raise ValueError("attempt_count must not be negative.")

    if retryable_failure_count < 0:
        raise ValueError("retryable_failure_count must not be negative.")

    if max_retryable_failures < 1:
        raise ValueError("max_retryable_failures must be at least one.")

    if retryable_failure_count > max_retryable_failures:
        raise ValueError(
            "retryable_failure_count must not exceed max_retryable_failures.",
        )


def _validate_agent_run_timestamps(run: AgentRun) -> None:
    if run.updated_at < run.created_at:
        raise ValueError(
            "updated_at must not be earlier than created_at.",
        )

    if run.first_started_at is not None and run.first_started_at < run.created_at:
        raise ValueError(
            "first_started_at must not be earlier than created_at.",
        )

    if run.completed_at is not None and run.completed_at < run.created_at:
        raise ValueError(
            "completed_at must not be earlier than created_at.",
        )

    if run.available_at is not None and run.available_at < run.created_at:
        raise ValueError(
            "available_at must not be earlier than created_at.",
        )

    if run.status is AgentRunStatus.WAITING_FOR_APPROVAL and run.available_at is not None:
        raise ValueError(
            "Waiting AgentRuns must not define available_at.",
        )

    if run.status is not AgentRunStatus.WAITING_FOR_APPROVAL and run.available_at is None:
        raise ValueError(
            "Non-waiting AgentRuns require available_at.",
        )


def _validate_started_attempt_state(run: AgentRun) -> None:
    if run.attempt_count == 0 and run.first_started_at is not None:
        raise ValueError(
            "first_started_at must be null before the first attempt.",
        )

    if run.attempt_count > 0 and run.first_started_at is None:
        raise ValueError(
            "first_started_at is required after an attempt has started.",
        )


def _validate_lease_state(run: AgentRun) -> None:
    lease_fields = (
        run.lease_owner,
        run.lease_token,
        run.lease_expires_at,
    )
    populated_count = sum(value is not None for value in lease_fields)

    if populated_count not in {0, len(lease_fields)}:
        raise ValueError(
            "Lease ownership fields must be populated or cleared together.",
        )

    has_lease = populated_count == len(lease_fields)

    if run.status is AgentRunStatus.RUNNING and not has_lease:
        raise ValueError(
            "Running AgentRuns require active lease ownership.",
        )

    if run.status is not AgentRunStatus.RUNNING and has_lease:
        raise ValueError(
            "Lease ownership is allowed only while an AgentRun is running.",
        )

    if run.status is AgentRunStatus.RUNNING and run.attempt_count < 1:
        raise ValueError(
            "Running AgentRuns require at least one started attempt.",
        )

    if (
        run.lease_expires_at is not None
        and run.first_started_at is not None
        and run.lease_expires_at <= run.first_started_at
    ):
        raise ValueError(
            "lease_expires_at must be later than first_started_at.",
        )


def _validate_completion_state(run: AgentRun) -> None:
    terminal_statuses = {
        AgentRunStatus.SUCCEEDED,
        AgentRunStatus.FAILED,
    }

    if run.status in terminal_statuses and run.completed_at is None:
        raise ValueError(
            "Terminal AgentRuns require completed_at.",
        )

    if run.status not in terminal_statuses and run.completed_at is not None:
        raise ValueError(
            "Non-terminal AgentRuns must not define completed_at.",
        )


def _validate_error_state(run: AgentRun) -> None:
    _validate_error_pair(
        error_code=run.last_error_code,
        error_summary=run.last_error_summary,
    )

    if (
        run.status
        in {
            AgentRunStatus.QUEUED,
            AgentRunStatus.SUCCEEDED,
        }
        and run.last_error_code is not None
    ):
        raise ValueError(
            "Queued and succeeded AgentRuns must not contain error details.",
        )

    if run.status is AgentRunStatus.WAITING_FOR_APPROVAL and run.last_error_code is not None:
        raise ValueError(
            "Waiting AgentRuns must not contain error details.",
        )

    if (
        run.status
        in {
            AgentRunStatus.RETRY_SCHEDULED,
            AgentRunStatus.FAILED,
        }
        and run.last_error_code is None
    ):
        raise ValueError(
            "Retry-scheduled and failed AgentRuns require error details.",
        )


def _validate_attempt_completion_state(
    attempt: AgentRunAttempt,
) -> None:
    is_finished = attempt.finished_at is not None
    has_outcome = attempt.outcome is not None

    if is_finished != has_outcome:
        raise ValueError(
            "Attempt outcome and finished_at must be populated together.",
        )

    if attempt.finished_at is not None and attempt.finished_at < attempt.started_at:
        raise ValueError(
            "finished_at must not be earlier than started_at.",
        )

    _validate_error_pair(
        error_code=attempt.error_code,
        error_summary=attempt.error_summary,
    )

    if attempt.outcome is None and attempt.error_code is not None:
        raise ValueError(
            "Active attempts must not contain error details.",
        )

    if attempt.outcome is AgentRunAttemptOutcome.SUCCEEDED and attempt.error_code is not None:
        raise ValueError(
            "Succeeded attempts must not contain error details.",
        )

    if (
        attempt.outcome is AgentRunAttemptOutcome.AWAITING_APPROVAL
        and attempt.error_code is not None
    ):
        raise ValueError(
            "Awaiting-approval attempts must not contain error details.",
        )

    failure_outcomes = {
        AgentRunAttemptOutcome.RETRYABLE_FAILURE,
        AgentRunAttemptOutcome.TERMINAL_FAILURE,
        AgentRunAttemptOutcome.TIMED_OUT,
        AgentRunAttemptOutcome.LEASE_EXPIRED,
    }
    if attempt.outcome in failure_outcomes and attempt.error_code is None:
        raise ValueError(
            "Failed attempts require error details.",
        )


def _validate_error_pair(
    *,
    error_code: str | None,
    error_summary: str | None,
) -> None:
    if (error_code is None) != (error_summary is None):
        raise ValueError(
            "Error code and summary must be populated or cleared together.",
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


def _validate_optional_bounded_text(
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


def _validate_optional_utc_timestamp(
    value: datetime | None,
    *,
    field_name: str,
) -> None:
    if value is None:
        return

    _validate_utc_timestamp(
        value,
        field_name=field_name,
    )


def _validate_utc_timestamp(
    value: datetime,
    *,
    field_name: str,
) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be a UTC-aware timestamp.")
