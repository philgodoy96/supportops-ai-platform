"""Unit tests for versioned AgentRun executor dispatch."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    TerminalAgentRunExecutionError,
)
from supportops.modules.agent_runs.application.executor_registry import (
    AgentRunExecutorRegistration,
    AgentRunExecutorRegistry,
    DuplicateAgentRunExecutorRegistrationError,
)
from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    INITIAL_TICKET_PROCESSING_WORKFLOW_NAME,
    AgentRun,
    AgentRunAttempt,
    AgentRunStatus,
)
from supportops.modules.tickets.domain.models import Ticket

_NOW = datetime(
    2026,
    8,
    1,
    19,
    30,
    tzinfo=UTC,
)
_WORKSPACE_ID = UUID(
    "d38e99b9-b880-4866-9448-55c27f38221f",
)
_TICKET_ID = UUID(
    "8e82a414-6a8e-447f-9437-38d66b766155",
)
_AGENT_RUN_ID = UUID(
    "ac9b9917-bc70-4da5-8294-816017fb0908",
)
_ATTEMPT_ID = UUID(
    "46e21997-629c-4f21-925e-54193056277e",
)
_LEASE_TOKEN = UUID(
    "ae23224b-38de-4db7-8924-66f7be90df20",
)
_EXECUTION_REQUEST_ID = UUID(
    "c204478c-88ab-4b3a-bfdf-167f6f656c63",
)


class RecordingExecutor:
    """Record every context delegated by the registry."""

    def __init__(
        self,
        *,
        error: Exception | None = None,
    ) -> None:
        self.contexts: list[AgentRunExecutionContext] = []
        self._error = error

    async def execute(
        self,
        context: AgentRunExecutionContext,
    ) -> None:
        self.contexts.append(context)

        if self._error is not None:
            raise self._error


def _context(
    *,
    workflow_name: str = (INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
    workflow_version: str = (DETERMINISTIC_BASELINE_WORKFLOW_VERSION),
) -> AgentRunExecutionContext:
    ticket = Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Unable to access billing",
        description=("The billing page returns an access error."),
        ingestion_request_id=UUID(
            "b018a83d-31fe-4df8-9401-f66829220a0c",
        ),
        correlation_id=UUID(
            "3a2285fd-471d-42ff-a8d8-38fedbaf8de9",
        ),
        now=_NOW - timedelta(minutes=1),
    )
    initial_run = AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=(ticket.ingestion_request_id),
        correlation_id=ticket.correlation_id,
        workflow_version=DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
        max_attempts=3,
        now=_NOW - timedelta(minutes=1),
    )
    running_run = replace(
        initial_run,
        workflow_name=workflow_name,
        workflow_version=workflow_version,
        status=AgentRunStatus.RUNNING,
        attempt_count=1,
        lease_owner="worker-a",
        lease_token=_LEASE_TOKEN,
        lease_expires_at=(_NOW + timedelta(seconds=45)),
        first_started_at=_NOW,
        updated_at=_NOW,
    )
    attempt = AgentRunAttempt.start(
        attempt_id=_ATTEMPT_ID,
        agent_run_id=_AGENT_RUN_ID,
        attempt_number=1,
        worker_id="worker-a",
        lease_token=_LEASE_TOKEN,
        execution_request_id=(_EXECUTION_REQUEST_ID),
        now=_NOW,
    )

    return AgentRunExecutionContext(
        agent_run=running_run,
        attempt=attempt,
        ticket=ticket,
    )


async def test_routes_exact_workflow_and_version() -> None:
    executor = RecordingExecutor()
    registry = AgentRunExecutorRegistry(
        (
            AgentRunExecutorRegistration(
                workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
                workflow_version=(DETERMINISTIC_BASELINE_WORKFLOW_VERSION),
                executor=executor,
            ),
        ),
    )
    context = _context()

    await registry.execute(context)

    assert executor.contexts == [
        context,
    ]
    assert len(registry) == 1


def test_resolve_returns_exact_registered_executor() -> None:
    baseline_executor = RecordingExecutor()
    classification_executor = RecordingExecutor()
    registry = AgentRunExecutorRegistry(
        (
            AgentRunExecutorRegistration(
                workflow_name="ticket-processing",
                workflow_version=("deterministic-baseline-v1"),
                executor=baseline_executor,
            ),
            AgentRunExecutorRegistration(
                workflow_name="ticket-processing",
                workflow_version=("ticket-classification-v1"),
                executor=classification_executor,
            ),
        ),
    )

    resolved = registry.resolve(
        workflow_name="ticket-processing",
        workflow_version="ticket-classification-v1",
    )

    assert resolved is classification_executor
    assert resolved is not baseline_executor
    assert len(registry) == 2


def test_rejects_duplicate_exact_registration() -> None:
    with pytest.raises(
        DuplicateAgentRunExecutorRegistrationError,
        match=("ticket-processing/ticket-classification-v1"),
    ):
        AgentRunExecutorRegistry(
            (
                AgentRunExecutorRegistration(
                    workflow_name="ticket-processing",
                    workflow_version=("ticket-classification-v1"),
                    executor=RecordingExecutor(),
                ),
                AgentRunExecutorRegistration(
                    workflow_name="ticket-processing",
                    workflow_version=("ticket-classification-v1"),
                    executor=RecordingExecutor(),
                ),
            ),
        )


def test_allows_same_workflow_with_different_versions() -> None:
    registry = AgentRunExecutorRegistry(
        (
            AgentRunExecutorRegistration(
                workflow_name="ticket-processing",
                workflow_version="version-one",
                executor=RecordingExecutor(),
            ),
            AgentRunExecutorRegistration(
                workflow_name="ticket-processing",
                workflow_version="version-two",
                executor=RecordingExecutor(),
            ),
        ),
    )

    assert len(registry) == 2


def test_allows_same_version_for_different_workflows() -> None:
    registry = AgentRunExecutorRegistry(
        (
            AgentRunExecutorRegistration(
                workflow_name="first-workflow",
                workflow_version="version-one",
                executor=RecordingExecutor(),
            ),
            AgentRunExecutorRegistration(
                workflow_name="second-workflow",
                workflow_version="version-one",
                executor=RecordingExecutor(),
            ),
        ),
    )

    assert len(registry) == 2


async def test_unknown_workflow_is_terminal() -> None:
    registry = AgentRunExecutorRegistry(
        (
            AgentRunExecutorRegistration(
                workflow_name="ticket-processing",
                workflow_version=("deterministic-baseline-v1"),
                executor=RecordingExecutor(),
            ),
        ),
    )

    with pytest.raises(
        TerminalAgentRunExecutionError,
    ) as captured:
        await registry.execute(
            _context(
                workflow_name="unknown-workflow",
            ),
        )

    assert captured.value.error_code == ("unsupported_workflow")
    assert captured.value.error_summary == (
        "The AgentRun workflow is not supported by the configured executor registry."
    )


async def test_unknown_version_is_terminal() -> None:
    executor = RecordingExecutor()
    registry = AgentRunExecutorRegistry(
        (
            AgentRunExecutorRegistration(
                workflow_name="ticket-processing",
                workflow_version=("deterministic-baseline-v1"),
                executor=executor,
            ),
        ),
    )

    with pytest.raises(
        TerminalAgentRunExecutionError,
    ) as captured:
        await registry.execute(
            _context(
                workflow_version="unknown-version",
            ),
        )

    assert captured.value.error_code == ("unsupported_workflow_version")
    assert captured.value.error_summary == (
        "The AgentRun workflow version is not supported by the configured executor registry."
    )
    assert executor.contexts == []


async def test_preserves_selected_executor_failure() -> None:
    expected_error = TerminalAgentRunExecutionError(
        error_code="unsupported_trigger",
        error_summary=("The AgentRun trigger is not supported by the selected executor."),
    )
    registry = AgentRunExecutorRegistry(
        (
            AgentRunExecutorRegistration(
                workflow_name="ticket-processing",
                workflow_version=("deterministic-baseline-v1"),
                executor=RecordingExecutor(
                    error=expected_error,
                ),
            ),
        ),
    )

    with pytest.raises(
        TerminalAgentRunExecutionError,
    ) as captured:
        await registry.execute(_context())

    assert captured.value is expected_error


@pytest.mark.parametrize(
    ("field_name", "value", "expected_message"),
    [
        (
            "workflow_name",
            "",
            "workflow_name is required",
        ),
        (
            "workflow_name",
            " ticket-processing",
            ("workflow_name must not contain surrounding whitespace"),
        ),
        (
            "workflow_name",
            "x" * 65,
            "workflow_name exceeds the maximum length",
        ),
        (
            "workflow_version",
            "",
            "workflow_version is required",
        ),
        (
            "workflow_version",
            " version-one",
            ("workflow_version must not contain surrounding whitespace"),
        ),
        (
            "workflow_version",
            "x" * 65,
            ("workflow_version exceeds the maximum length"),
        ),
    ],
)
def test_registration_rejects_invalid_identifiers(
    field_name: str,
    value: str,
    expected_message: str,
) -> None:
    values = {
        "workflow_name": "ticket-processing",
        "workflow_version": "version-one",
    }
    values[field_name] = value

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        AgentRunExecutorRegistration(
            workflow_name=values["workflow_name"],
            workflow_version=values["workflow_version"],
            executor=RecordingExecutor(),
        )


def test_registration_requires_executor_contract() -> None:
    with pytest.raises(
        TypeError,
        match=("must implement the AgentRunExecutor contract"),
    ):
        AgentRunExecutorRegistration(
            workflow_name="ticket-processing",
            workflow_version="version-one",
            executor=object(),  # type: ignore[arg-type]
        )
