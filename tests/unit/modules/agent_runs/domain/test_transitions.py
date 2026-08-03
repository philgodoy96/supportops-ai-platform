"""Unit tests for fenced AgentRun transition contracts."""

from datetime import UTC, datetime, timedelta, timezone
from re import escape
from uuid import UUID

import pytest

from supportops.modules.agent_runs.domain.models import (
    AgentRunAttemptOutcome,
)
from supportops.modules.agent_runs.domain.transitions import (
    AgentRunApprovalRequeueResult,
    AgentRunFailureDisposition,
    AgentRunTransitionResult,
    CompleteAgentRunCommand,
    FailAgentRunCommand,
    RequeueWaitingAgentRunCommand,
)

_RUN_ID = UUID(
    "69184ef1-4d71-452e-8070-0b784c29368e",
)
_LEASE_TOKEN = UUID(
    "dd0ae456-3467-41db-93d1-a908f40e8365",
)
_FINISHED_AT = datetime(
    2026,
    7,
    31,
    18,
    0,
    tzinfo=UTC,
)


def create_failure_command(
    *,
    outcome: AgentRunAttemptOutcome = (AgentRunAttemptOutcome.RETRYABLE_FAILURE),
    disposition: AgentRunFailureDisposition = (AgentRunFailureDisposition.RETRY_SCHEDULED),
    error_code: str = "retryable_executor_failure",
    error_summary: str = ("The configured executor reported a retryable failure."),
    retry_available_at: datetime | None = None,
) -> FailAgentRunCommand:
    if retry_available_at is None and disposition is AgentRunFailureDisposition.RETRY_SCHEDULED:
        retry_available_at = _FINISHED_AT + timedelta(seconds=2)

    return FailAgentRunCommand(
        agent_run_id=_RUN_ID,
        lease_token=_LEASE_TOKEN,
        finished_at=_FINISHED_AT,
        outcome=outcome,
        disposition=disposition,
        error_code=error_code,
        error_summary=error_summary,
        retry_available_at=retry_available_at,
    )


def test_complete_command_preserves_fencing_values() -> None:
    command = CompleteAgentRunCommand(
        agent_run_id=_RUN_ID,
        lease_token=_LEASE_TOKEN,
        finished_at=_FINISHED_AT,
    )

    assert command.agent_run_id == _RUN_ID
    assert command.lease_token == _LEASE_TOKEN
    assert command.finished_at == _FINISHED_AT


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2026, 7, 31, 18, 0),
        datetime(
            2026,
            7,
            31,
            15,
            0,
            tzinfo=timezone(timedelta(hours=-3)),
        ),
    ],
)
def test_complete_command_requires_utc_finished_at(
    timestamp: datetime,
) -> None:
    with pytest.raises(
        ValueError,
        match=escape(
            "finished_at must be a UTC-aware timestamp.",
        ),
    ):
        CompleteAgentRunCommand(
            agent_run_id=_RUN_ID,
            lease_token=_LEASE_TOKEN,
            finished_at=timestamp,
        )


def test_retryable_failure_requires_future_availability() -> None:
    command = create_failure_command()

    assert command.disposition is AgentRunFailureDisposition.RETRY_SCHEDULED
    assert command.outcome is AgentRunAttemptOutcome.RETRYABLE_FAILURE
    assert command.retry_available_at == _FINISHED_AT + timedelta(seconds=2)


@pytest.mark.parametrize(
    "outcome",
    [
        AgentRunAttemptOutcome.SUCCEEDED,
        AgentRunAttemptOutcome.LEASE_EXPIRED,
    ],
)
def test_failure_command_rejects_non_executor_failure_outcome(
    outcome: AgentRunAttemptOutcome,
) -> None:
    with pytest.raises(
        ValueError,
        match=escape(
            "outcome must represent an executor failure.",
        ),
    ):
        create_failure_command(outcome=outcome)


def test_terminal_failure_cannot_be_retry_scheduled() -> None:
    with pytest.raises(
        ValueError,
        match=escape(
            "Terminal failures cannot be retry-scheduled.",
        ),
    ):
        create_failure_command(
            outcome=AgentRunAttemptOutcome.TERMINAL_FAILURE,
        )


def test_retry_disposition_requires_available_at() -> None:
    with pytest.raises(
        ValueError,
        match=escape(
            "retry_available_at is required for a retry.",
        ),
    ):
        FailAgentRunCommand(
            agent_run_id=_RUN_ID,
            lease_token=_LEASE_TOKEN,
            finished_at=_FINISHED_AT,
            outcome=AgentRunAttemptOutcome.TIMED_OUT,
            disposition=AgentRunFailureDisposition.RETRY_SCHEDULED,
            error_code="executor_timeout",
            error_summary=("The configured executor exceeded its execution timeout."),
            retry_available_at=None,
        )


@pytest.mark.parametrize(
    "retry_available_at",
    [
        _FINISHED_AT,
        _FINISHED_AT - timedelta(seconds=1),
    ],
)
def test_retry_availability_must_follow_finished_at(
    retry_available_at: datetime,
) -> None:
    with pytest.raises(
        ValueError,
        match=escape(
            "retry_available_at must be later than finished_at.",
        ),
    ):
        create_failure_command(
            retry_available_at=retry_available_at,
        )


def test_terminal_disposition_rejects_retry_availability() -> None:
    with pytest.raises(
        ValueError,
        match=escape(
            "retry_available_at must be null for a terminal run.",
        ),
    ):
        create_failure_command(
            disposition=AgentRunFailureDisposition.FAILED,
            retry_available_at=_FINISHED_AT + timedelta(seconds=2),
        )


def test_terminal_failure_accepts_terminal_disposition() -> None:
    command = create_failure_command(
        outcome=AgentRunAttemptOutcome.TERMINAL_FAILURE,
        disposition=AgentRunFailureDisposition.FAILED,
        error_code="terminal_executor_failure",
        error_summary=("The configured executor reported a terminal failure."),
        retry_available_at=None,
    )

    assert command.disposition is AgentRunFailureDisposition.FAILED
    assert command.outcome is AgentRunAttemptOutcome.TERMINAL_FAILURE
    assert command.retry_available_at is None


def test_exhausted_retryable_failure_accepts_terminal_disposition() -> None:
    command = create_failure_command(
        disposition=AgentRunFailureDisposition.FAILED,
        retry_available_at=None,
    )

    assert command.disposition is AgentRunFailureDisposition.FAILED
    assert command.outcome is AgentRunAttemptOutcome.RETRYABLE_FAILURE


@pytest.mark.parametrize(
    ("field_name", "value", "expected_message"),
    [
        (
            "error_code",
            "",
            "error_code is required.",
        ),
        (
            "error_code",
            " retryable_executor_failure",
            ("error_code must not contain surrounding whitespace."),
        ),
        (
            "error_code",
            "a" * 65,
            "error_code exceeds the maximum length.",
        ),
        (
            "error_summary",
            "",
            "error_summary is required.",
        ),
        (
            "error_summary",
            " unsafe summary ",
            ("error_summary must not contain surrounding whitespace."),
        ),
        (
            "error_summary",
            "a" * 513,
            "error_summary exceeds the maximum length.",
        ),
    ],
)
def test_failure_command_rejects_invalid_error_text(
    field_name: str,
    value: str,
    expected_message: str,
) -> None:
    error_code = "retryable_executor_failure"
    error_summary = "The configured executor reported a retryable failure."
    if field_name == "error_code":
        error_code = value
    else:
        error_summary = value

    with pytest.raises(
        ValueError,
        match=escape(expected_message),
    ):
        create_failure_command(
            error_code=error_code,
            error_summary=error_summary,
        )


def test_transition_results_are_explicit() -> None:
    assert AgentRunTransitionResult.APPLIED.value == "applied"
    assert AgentRunTransitionResult.LEASE_LOST.value == "lease_lost"
    assert AgentRunApprovalRequeueResult.APPLIED.value == "applied"
    assert AgentRunApprovalRequeueResult.STATE_CONFLICT.value == ("state_conflict")


def test_requeue_waiting_command_preserves_values() -> None:
    workspace_id = UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4")
    ticket_id = UUID("38bb60fe-d2ea-4615-b499-91aa45069019")
    command = RequeueWaitingAgentRunCommand(
        workspace_id=workspace_id,
        ticket_id=ticket_id,
        agent_run_id=_RUN_ID,
        requeued_at=_FINISHED_AT,
    )

    assert command.workspace_id == workspace_id
    assert command.ticket_id == ticket_id
    assert command.agent_run_id == _RUN_ID
    assert command.requeued_at == _FINISHED_AT


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2026, 7, 31, 18, 0),
        datetime(
            2026,
            7,
            31,
            15,
            0,
            tzinfo=timezone(timedelta(hours=-3)),
        ),
    ],
)
def test_requeue_waiting_command_requires_utc_timestamp(
    timestamp: datetime,
) -> None:
    with pytest.raises(ValueError, match="requeued_at"):
        RequeueWaitingAgentRunCommand(
            workspace_id=UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4"),
            ticket_id=UUID("38bb60fe-d2ea-4615-b499-91aa45069019"),
            agent_run_id=_RUN_ID,
            requeued_at=timestamp,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "workspace_id",
        "ticket_id",
        "agent_run_id",
    ],
)
def test_requeue_waiting_command_requires_uuid_ids(
    field_name: str,
) -> None:
    values: dict[str, object] = {
        "workspace_id": UUID("032c8c87-57cc-4d14-bfbd-04968b4e8cd4"),
        "ticket_id": UUID("38bb60fe-d2ea-4615-b499-91aa45069019"),
        "agent_run_id": _RUN_ID,
        "requeued_at": _FINISHED_AT,
    }
    values[field_name] = "not-a-uuid"

    with pytest.raises(TypeError, match=field_name):
        RequeueWaitingAgentRunCommand(**values)  # type: ignore[arg-type]
