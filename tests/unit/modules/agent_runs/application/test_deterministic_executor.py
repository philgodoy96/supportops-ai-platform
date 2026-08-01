"""Unit tests for the deterministic ticket processing executor."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from supportops.modules.agent_runs.application.deterministic_executor import (
    DeterministicTicketProcessingExecutor,
)
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    TerminalAgentRunExecutionError,
)
from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    AgentRun,
    AgentRunAttempt,
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


def create_context() -> AgentRunExecutionContext:
    """Create a valid deterministic baseline execution context."""

    ticket = Ticket.create(
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
    initial_run = AgentRun.create_initial(
        agent_run_id=UUID(
            "69184ef1-4d71-452e-8070-0b784c29368e",
        ),
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=ticket.ingestion_request_id,
        correlation_id=ticket.correlation_id,
        workflow_version=DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
        max_attempts=3,
        now=_NOW - timedelta(minutes=1),
    )
    running_run = replace(
        initial_run,
        status=AgentRunStatus.RUNNING,
        attempt_count=1,
        lease_owner="worker-a",
        lease_token=_LEASE_TOKEN,
        lease_expires_at=_NOW + timedelta(seconds=45),
        first_started_at=_NOW,
        updated_at=_NOW,
    )
    attempt = AgentRunAttempt.start(
        attempt_id=_ATTEMPT_ID,
        agent_run_id=running_run.id,
        attempt_number=1,
        worker_id="worker-a",
        lease_token=_LEASE_TOKEN,
        execution_request_id=_EXECUTION_REQUEST_ID,
        now=_NOW,
    )

    return AgentRunExecutionContext(
        agent_run=running_run,
        attempt=attempt,
        ticket=ticket,
    )


async def test_deterministic_executor_completes_supported_workflow() -> None:
    executor = DeterministicTicketProcessingExecutor()

    await executor.execute(create_context())


@pytest.mark.parametrize(
    ("field_name", "value", "expected_code"),
    [
        (
            "workflow_name",
            "unsupported-workflow",
            "unsupported_workflow",
        ),
        (
            "workflow_version",
            "unsupported-version",
            "unsupported_workflow_version",
        ),
        (
            "trigger_key",
            "unsupported-trigger",
            "unsupported_trigger",
        ),
    ],
)
async def test_deterministic_executor_rejects_unsupported_contract(
    field_name: str,
    value: str,
    expected_code: str,
) -> None:
    context = create_context()
    if field_name == "workflow_name":
        agent_run = replace(
            context.agent_run,
            workflow_name=value,
        )
    elif field_name == "workflow_version":
        agent_run = replace(
            context.agent_run,
            workflow_version=value,
        )
    elif field_name == "trigger_key":
        agent_run = replace(
            context.agent_run,
            trigger_key=value,
        )
    else:
        raise AssertionError(f"Unexpected field: {field_name}")

    context = AgentRunExecutionContext(
        agent_run=agent_run,
        attempt=context.attempt,
        ticket=context.ticket,
    )
    executor = DeterministicTicketProcessingExecutor()

    with pytest.raises(
        TerminalAgentRunExecutionError,
    ) as captured:
        await executor.execute(context)

    assert captured.value.error_code == expected_code
