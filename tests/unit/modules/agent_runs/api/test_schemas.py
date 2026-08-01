"""Unit tests for AgentRun HTTP response projections."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

from supportops.modules.agent_runs.api.schemas import (
    AgentRunAttemptResponse,
    AgentRunResponse,
)
from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    AgentRun,
    AgentRunAttempt,
    AgentRunAttemptOutcome,
    AgentRunStatus,
)

_NOW = datetime(
    2026,
    7,
    31,
    21,
    0,
    tzinfo=UTC,
)
_WORKSPACE_ID = UUID(
    "032c8c87-57cc-4d14-bfbd-04968b4e8cd4",
)
_TICKET_ID = UUID(
    "38bb60fe-d2ea-4615-b499-91aa45069019",
)
_AGENT_RUN_ID = UUID(
    "69184ef1-4d71-452e-8070-0b784c29368e",
)
_ATTEMPT_ID = UUID(
    "2b39f5b7-b2a4-48d0-b079-fdad286d5315",
)
_INGESTION_REQUEST_ID = UUID(
    "725eec8a-c504-4071-ac96-c78cc907f26c",
)
_CORRELATION_ID = UUID(
    "1038c98e-62fd-45df-9839-138f7105cb78",
)
_LEASE_TOKEN = UUID(
    "dd0ae456-3467-41db-93d1-a908f40e8365",
)
_EXECUTION_REQUEST_ID = UUID(
    "d1fa068f-2278-47a8-b3c9-39ccf91f0a5e",
)


def create_agent_run() -> AgentRun:
    """Create one deterministic AgentRun."""

    return AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=_INGESTION_REQUEST_ID,
        correlation_id=_CORRELATION_ID,
        workflow_version=DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
        max_attempts=3,
        now=_NOW,
    )


def create_attempt() -> AgentRunAttempt:
    """Create one deterministic AgentRun attempt."""

    return AgentRunAttempt.start(
        attempt_id=_ATTEMPT_ID,
        agent_run_id=_AGENT_RUN_ID,
        attempt_number=1,
        worker_id="worker-a",
        lease_token=_LEASE_TOKEN,
        execution_request_id=_EXECUTION_REQUEST_ID,
        now=_NOW,
    )


def test_agent_run_response_projects_safe_fields() -> None:
    agent_run = replace(
        create_agent_run(),
        status=AgentRunStatus.RETRY_SCHEDULED,
        attempt_count=1,
        available_at=_NOW + timedelta(seconds=2),
        first_started_at=_NOW,
        last_error_code="unexpected_executor_failure",
        last_error_summary=("The executor failed unexpectedly and may be retried."),
        updated_at=_NOW + timedelta(seconds=1),
    )

    response = AgentRunResponse.from_domain(agent_run)

    assert response.id == _AGENT_RUN_ID
    assert response.workspace_id == _WORKSPACE_ID
    assert response.ticket_id == _TICKET_ID
    assert response.status is AgentRunStatus.RETRY_SCHEDULED
    assert response.workflow.name == "ticket-processing"
    assert response.workflow.version == "deterministic-baseline-v1"
    assert response.workflow.trigger_key == "initial-ticket-processing"
    assert response.attempt_count == 1
    assert response.max_attempts == 3
    assert response.correlation_id == _CORRELATION_ID
    assert response.last_error is not None
    assert response.last_error.code == "unexpected_executor_failure"


def test_agent_run_response_omits_internal_fields() -> None:
    agent_run = replace(
        create_agent_run(),
        status=AgentRunStatus.RUNNING,
        attempt_count=1,
        first_started_at=_NOW,
        lease_owner="worker-a",
        lease_token=_LEASE_TOKEN,
        lease_expires_at=_NOW + timedelta(seconds=45),
    )

    payload = AgentRunResponse.from_domain(
        agent_run,
    ).model_dump()

    assert "lease_owner" not in payload
    assert "lease_token" not in payload
    assert "lease_expires_at" not in payload
    assert "ingestion_request_id" not in payload


def test_agent_run_response_requires_complete_safe_error_pair() -> None:
    agent_run = create_agent_run()
    # Bypass AgentRun invariants to assert defensive projection behavior.
    object.__setattr__(agent_run, "last_error_code", "retryable_failure")

    response = AgentRunResponse.from_domain(agent_run)

    assert response.last_error is None


def test_attempt_response_projects_safe_fields() -> None:
    attempt = replace(
        create_attempt(),
        finished_at=_NOW + timedelta(seconds=1),
        outcome=AgentRunAttemptOutcome.RETRYABLE_FAILURE,
        error_code="unexpected_executor_failure",
        error_summary=("The executor failed unexpectedly and may be retried."),
    )

    response = AgentRunAttemptResponse.from_domain(attempt)

    assert response.id == _ATTEMPT_ID
    assert response.attempt_number == 1
    assert response.worker_id == "worker-a"
    assert response.outcome is AgentRunAttemptOutcome.RETRYABLE_FAILURE
    assert response.error is not None
    assert response.error.code == "unexpected_executor_failure"


def test_attempt_response_omits_internal_fencing_fields() -> None:
    payload = AgentRunAttemptResponse.from_domain(
        create_attempt(),
    ).model_dump()

    assert "agent_run_id" not in payload
    assert "lease_token" not in payload
    assert "execution_request_id" not in payload


def test_attempt_response_supports_active_attempt() -> None:
    response = AgentRunAttemptResponse.from_domain(
        create_attempt(),
    )

    assert response.finished_at is None
    assert response.outcome is None
    assert response.error is None
