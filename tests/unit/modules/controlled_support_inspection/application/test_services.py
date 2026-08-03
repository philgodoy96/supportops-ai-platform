"""Unit tests for controlled-support inspection assembly."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import UUID

import pytest

from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_WORKFLOW_VERSION,
)
from supportops.modules.agent_runs.domain.models import (
    AgentRun,
)
from supportops.modules.controlled_support_inspection.application.errors import (
    ControlledSupportInspectionInconsistentError,
    ControlledSupportInspectionNotFoundError,
    UnsupportedAgentRunInspectionError,
)
from supportops.modules.controlled_support_inspection.application.repository import (
    ControlledSupportInspectionData,
    ControlledSupportInspectionIdentity,
)
from supportops.modules.controlled_support_inspection.application.services import (
    GetControlledSupportInspection,
)
from supportops.modules.controlled_support_inspection.domain.models import (
    ControlledSupportInspectionStatus,
)

_NOW = datetime(
    2026,
    8,
    2,
    18,
    0,
    tzinfo=UTC,
)
_WORKSPACE_ID = UUID("10000000-0000-4000-8000-000000000001")
_TICKET_ID = UUID("20000000-0000-4000-8000-000000000002")
_AGENT_RUN_ID = UUID("30000000-0000-4000-8000-000000000003")


class StubTransactionManager:
    """Record one application-owned transaction."""

    def __init__(self) -> None:
        self.transaction_count = 0

    @asynccontextmanager
    async def transaction(
        self,
    ) -> AsyncIterator[None]:
        self.transaction_count += 1
        yield


class StubRepository:
    """Return one configured inspection result."""

    def __init__(
        self,
        result: ControlledSupportInspectionData | None,
    ) -> None:
        self.result = result
        self.identities: list[ControlledSupportInspectionIdentity] = []

    async def get_inspection_data(
        self,
        identity: ControlledSupportInspectionIdentity,
    ) -> ControlledSupportInspectionData | None:
        self.identities.append(identity)

        return self.result


class InconsistentRepository:
    """Raise a persistence-data validation failure."""

    async def get_inspection_data(
        self,
        identity: ControlledSupportInspectionIdentity,
    ) -> ControlledSupportInspectionData | None:
        del identity

        raise ValueError("Synthetic durable-data inconsistency.")


def _agent_run(
    *,
    workflow_version: str = (CONTROLLED_SUPPORT_WORKFLOW_VERSION),
) -> AgentRun:
    return AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=UUID("40000000-0000-4000-8000-000000000004"),
        correlation_id=UUID("50000000-0000-4000-8000-000000000005"),
        workflow_version=workflow_version,
        max_retryable_failures=3,
        now=_NOW,
    )


def _data(
    *,
    workflow_version: str = (CONTROLLED_SUPPORT_WORKFLOW_VERSION),
) -> ControlledSupportInspectionData:
    return ControlledSupportInspectionData(
        agent_run=_agent_run(workflow_version=workflow_version),
        attempts=(),
        classification=None,
        tool_calls=(),
        llm_invocations=(),
        recommendation=None,
        citations=(),
    )


async def test_returns_queued_inspection() -> None:
    repository = StubRepository(_data())
    transaction_manager = StubTransactionManager()
    service = GetControlledSupportInspection(
        repository=repository,
        transaction_manager=transaction_manager,
    )

    inspection = await service.execute(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    assert inspection.agent_run.status is (ControlledSupportInspectionStatus.QUEUED)
    assert inspection.classification is None
    assert inspection.tool_calls == ()
    assert inspection.recommendation is None
    assert inspection.llm_usage.invocation_count == 0
    assert transaction_manager.transaction_count == 1
    assert repository.identities == [
        ControlledSupportInspectionIdentity(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=_AGENT_RUN_ID,
        )
    ]


async def test_missing_inspection_raises_stable_not_found() -> None:
    service = GetControlledSupportInspection(
        repository=StubRepository(None),
        transaction_manager=StubTransactionManager(),
    )

    with pytest.raises(ControlledSupportInspectionNotFoundError):
        await service.execute(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=_AGENT_RUN_ID,
        )


async def test_unsupported_workflow_raises_conflict_error() -> None:
    service = GetControlledSupportInspection(
        repository=StubRepository(_data(workflow_version="ticket-classification-v1")),
        transaction_manager=StubTransactionManager(),
    )

    with pytest.raises(UnsupportedAgentRunInspectionError):
        await service.execute(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=_AGENT_RUN_ID,
        )


async def test_repository_validation_failure_is_normalized() -> None:
    service = GetControlledSupportInspection(
        repository=InconsistentRepository(),
        transaction_manager=StubTransactionManager(),
    )

    with pytest.raises(ControlledSupportInspectionInconsistentError):
        await service.execute(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=_AGENT_RUN_ID,
        )
