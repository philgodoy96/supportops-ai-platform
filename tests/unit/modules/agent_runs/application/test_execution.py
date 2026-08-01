"""Unit tests for AgentRun execution contracts."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from re import escape
from uuid import UUID

import pytest

from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    RetryableAgentRunExecutionError,
    TerminalAgentRunExecutionError,
)
from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    AgentRun,
    AgentRunAttempt,
    AgentRunAttemptOutcome,
    AgentRunStatus,
)
from supportops.modules.tickets.domain.models import Ticket

_NOW = datetime(
    2026,
    7,
    31,
    18,
    0,
    tzinfo=UTC,
)
_WORKSPACE_ID = UUID(
    "032c8c87-57cc-4d14-bfbd-04968b4e8cd4",
)
_TICKET_ID = UUID(
    "38bb60fe-d2ea-4615-b499-91aa45069019",
)
_LEASE_TOKEN = UUID(
    "dd0ae456-3467-41db-93d1-a908f40e8365",
)
_ATTEMPT_ID = UUID(
    "2b39f5b7-b2a4-48d0-b079-fdad286d5315",
)
_EXECUTION_REQUEST_ID = UUID(
    "d1fa068f-2278-47a8-b3c9-39ccf91f0a5e",
)


def create_ticket() -> Ticket:
    """Create a deterministic ticket."""

    return Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Unable to access billing",
        description="The billing page returns an access error.",
        ingestion_request_id=UUID(
            "725eec8a-c504-4071-ac96-c78cc907f26c",
        ),
        correlation_id=UUID(
            "1038c98e-62fd-45df-9839-138f7105cb78",
        ),
        now=_NOW - timedelta(minutes=1),
    )


def create_running_run() -> AgentRun:
    """Create a deterministic claimed AgentRun."""

    initial_run = AgentRun.create_initial(
        agent_run_id=UUID(
            "69184ef1-4d71-452e-8070-0b784c29368e",
        ),
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=UUID(
            "725eec8a-c504-4071-ac96-c78cc907f26c",
        ),
        correlation_id=UUID(
            "1038c98e-62fd-45df-9839-138f7105cb78",
        ),
        workflow_version=DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
        max_attempts=3,
        now=_NOW - timedelta(minutes=1),
    )

    return replace(
        initial_run,
        status=AgentRunStatus.RUNNING,
        attempt_count=1,
        lease_owner="worker-a",
        lease_token=_LEASE_TOKEN,
        lease_expires_at=_NOW + timedelta(seconds=45),
        first_started_at=_NOW,
        updated_at=_NOW,
    )


def create_active_attempt() -> AgentRunAttempt:
    """Create an active attempt matching create_running_run()."""

    return AgentRunAttempt.start(
        attempt_id=_ATTEMPT_ID,
        agent_run_id=create_running_run().id,
        attempt_number=1,
        worker_id="worker-a",
        lease_token=_LEASE_TOKEN,
        execution_request_id=_EXECUTION_REQUEST_ID,
        now=_NOW,
    )


def test_execution_context_accepts_matching_claimed_run() -> None:
    run = create_running_run()
    attempt = create_active_attempt()
    ticket = create_ticket()

    context = AgentRunExecutionContext(
        agent_run=run,
        attempt=attempt,
        ticket=ticket,
    )

    assert context.agent_run == run
    assert context.attempt == attempt
    assert context.ticket == ticket


def test_execution_context_requires_running_run() -> None:
    run = AgentRun.create_initial(
        agent_run_id=UUID(
            "69184ef1-4d71-452e-8070-0b784c29368e",
        ),
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=UUID(
            "725eec8a-c504-4071-ac96-c78cc907f26c",
        ),
        correlation_id=UUID(
            "1038c98e-62fd-45df-9839-138f7105cb78",
        ),
        workflow_version=DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
        max_attempts=3,
        now=_NOW,
    )

    with pytest.raises(
        ValueError,
        match=escape(
            "AgentRun execution requires a running AgentRun.",
        ),
    ):
        AgentRunExecutionContext(
            agent_run=run,
            attempt=create_active_attempt(),
            ticket=create_ticket(),
        )


def test_execution_context_requires_matching_ticket() -> None:
    ticket = replace(
        create_ticket(),
        id=UUID(
            "43499fc4-3638-4097-aaf2-3c5300cf6cd6",
        ),
    )

    with pytest.raises(
        ValueError,
        match=escape(
            "AgentRun and ticket must reference the same ticket.",
        ),
    ):
        AgentRunExecutionContext(
            agent_run=create_running_run(),
            attempt=create_active_attempt(),
            ticket=ticket,
        )


def test_execution_context_requires_matching_workspace() -> None:
    ticket = replace(
        create_ticket(),
        workspace_id=UUID(
            "35e64ab6-d1b2-4386-a51e-48f11fa6d058",
        ),
    )

    with pytest.raises(
        ValueError,
        match=escape(
            "AgentRun and ticket must belong to the same workspace.",
        ),
    ):
        AgentRunExecutionContext(
            agent_run=create_running_run(),
            attempt=create_active_attempt(),
            ticket=ticket,
        )


def test_execution_context_requires_attempt_for_same_run() -> None:
    attempt = AgentRunAttempt.start(
        attempt_id=_ATTEMPT_ID,
        agent_run_id=UUID(
            "43499fc4-3638-4097-aaf2-3c5300cf6cd6",
        ),
        attempt_number=1,
        worker_id="worker-a",
        lease_token=_LEASE_TOKEN,
        execution_request_id=_EXECUTION_REQUEST_ID,
        now=_NOW,
    )

    with pytest.raises(
        ValueError,
        match=escape(
            "AgentRun execution attempt must reference the same AgentRun.",
        ),
    ):
        AgentRunExecutionContext(
            agent_run=create_running_run(),
            attempt=attempt,
            ticket=create_ticket(),
        )


def test_execution_context_requires_matching_attempt_number() -> None:
    attempt = AgentRunAttempt.start(
        attempt_id=_ATTEMPT_ID,
        agent_run_id=create_running_run().id,
        attempt_number=2,
        worker_id="worker-a",
        lease_token=_LEASE_TOKEN,
        execution_request_id=_EXECUTION_REQUEST_ID,
        now=_NOW,
    )

    with pytest.raises(
        ValueError,
        match=escape(
            "AgentRun execution attempt number must match the AgentRun.",
        ),
    ):
        AgentRunExecutionContext(
            agent_run=create_running_run(),
            attempt=attempt,
            ticket=create_ticket(),
        )


def test_execution_context_requires_matching_lease_token() -> None:
    attempt = AgentRunAttempt.start(
        attempt_id=_ATTEMPT_ID,
        agent_run_id=create_running_run().id,
        attempt_number=1,
        worker_id="worker-a",
        lease_token=UUID(
            "43499fc4-3638-4097-aaf2-3c5300cf6cd6",
        ),
        execution_request_id=_EXECUTION_REQUEST_ID,
        now=_NOW,
    )

    with pytest.raises(
        ValueError,
        match=escape(
            "AgentRun execution attempt must share the AgentRun lease token.",
        ),
    ):
        AgentRunExecutionContext(
            agent_run=create_running_run(),
            attempt=attempt,
            ticket=create_ticket(),
        )


def test_execution_context_requires_matching_worker() -> None:
    attempt = AgentRunAttempt.start(
        attempt_id=_ATTEMPT_ID,
        agent_run_id=create_running_run().id,
        attempt_number=1,
        worker_id="worker-b",
        lease_token=_LEASE_TOKEN,
        execution_request_id=_EXECUTION_REQUEST_ID,
        now=_NOW,
    )

    with pytest.raises(
        ValueError,
        match=escape(
            "AgentRun execution attempt must share the AgentRun worker.",
        ),
    ):
        AgentRunExecutionContext(
            agent_run=create_running_run(),
            attempt=attempt,
            ticket=create_ticket(),
        )


def test_execution_context_requires_active_attempt() -> None:
    attempt = replace(
        create_active_attempt(),
        finished_at=_NOW + timedelta(seconds=1),
        outcome=AgentRunAttemptOutcome.SUCCEEDED,
    )

    with pytest.raises(
        ValueError,
        match=escape(
            "AgentRun execution requires an active attempt.",
        ),
    ):
        AgentRunExecutionContext(
            agent_run=create_running_run(),
            attempt=attempt,
            ticket=create_ticket(),
        )


@pytest.mark.parametrize(
    "error_type",
    [
        RetryableAgentRunExecutionError,
        TerminalAgentRunExecutionError,
    ],
)
def test_execution_error_preserves_safe_details(
    error_type: type[RetryableAgentRunExecutionError | TerminalAgentRunExecutionError],
) -> None:
    error = error_type(
        error_code="executor_failure",
        error_summary="The executor could not process the AgentRun.",
    )

    assert error.error_code == "executor_failure"
    assert error.error_summary == "The executor could not process the AgentRun."
    assert str(error) == "The executor could not process the AgentRun."


@pytest.mark.parametrize(
    ("error_code", "error_summary", "expected_message"),
    [
        (
            "",
            "Safe summary.",
            "error_code is required.",
        ),
        (
            " executor_failure",
            "Safe summary.",
            ("error_code must not contain surrounding whitespace."),
        ),
        (
            "a" * 65,
            "Safe summary.",
            "error_code exceeds the maximum length.",
        ),
        (
            "executor_failure",
            "",
            "error_summary is required.",
        ),
        (
            "executor_failure",
            " unsafe summary ",
            ("error_summary must not contain surrounding whitespace."),
        ),
        (
            "executor_failure",
            "a" * 513,
            "error_summary exceeds the maximum length.",
        ),
    ],
)
def test_execution_error_rejects_unsafe_details(
    error_code: str,
    error_summary: str,
    expected_message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=escape(expected_message),
    ):
        RetryableAgentRunExecutionError(
            error_code=error_code,
            error_summary=error_summary,
        )
