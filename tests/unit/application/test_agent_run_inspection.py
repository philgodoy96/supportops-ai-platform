"""Unit tests for cross-module AgentRun inspection."""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import pytest

from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
)
from supportops.application.agent_run_inspection import (
    GetAgentRunInspection,
)
from supportops.modules.agent_runs.application.errors import (
    AgentRunNotFoundError,
)
from supportops.modules.agent_runs.domain.models import (
    TICKET_CLASSIFICATION_WORKFLOW_VERSION,
    AgentRun,
    AgentRunAttempt,
)
from supportops.modules.agent_runs.domain.repositories import (
    AgentRunQueryRepository,
)
from supportops.modules.ticket_classifications.domain.inspection import (
    AgentRunClassificationReference,
)
from supportops.modules.ticket_classifications.domain.repositories import (
    TicketClassificationQueryRepository,
)

_NOW = datetime(
    2026,
    8,
    1,
    21,
    30,
    tzinfo=UTC,
)
_WORKSPACE_ID = UUID(
    "11111111-2222-4333-8444-555555555555",
)
_OTHER_WORKSPACE_ID = UUID(
    "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
)
_TICKET_ID = UUID(
    "22222222-3333-4444-8555-666666666666",
)
_AGENT_RUN_ID = UUID(
    "33333333-4444-4555-8666-777777777777",
)
_CLASSIFICATION_ID = UUID(
    "44444444-5555-4666-8777-888888888888",
)
_INGESTION_REQUEST_ID = UUID(
    "55555555-6666-4777-8888-999999999999",
)
_CORRELATION_ID = UUID(
    "66666666-7777-4888-8999-aaaaaaaaaaaa",
)


class FakeAgentRunQueryRepository:
    """Return one configured workspace-scoped AgentRun."""

    def __init__(
        self,
        agent_run: AgentRun | None,
    ) -> None:
        self._agent_run = agent_run
        self.get_calls: list[tuple[UUID, UUID]] = []

    async def get(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> AgentRun | None:
        self.get_calls.append(
            (
                workspace_id,
                agent_run_id,
            ),
        )

        if self._agent_run is None:
            return None

        if self._agent_run.workspace_id != workspace_id or self._agent_run.id != agent_run_id:
            return None

        return self._agent_run

    async def list_attempts(
        self,
        *,
        agent_run_id: UUID,
    ) -> Sequence[AgentRunAttempt]:
        raise AssertionError(
            "AgentRun inspection must not list attempts.",
        )


class FakeClassificationQueryRepository:
    """Return one configured classification reference."""

    def __init__(
        self,
        reference: AgentRunClassificationReference | None,
    ) -> None:
        self._reference = reference
        self.reference_calls: list[tuple[UUID, UUID]] = []

    async def get_reference_by_agent_run_id(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> AgentRunClassificationReference | None:
        self.reference_calls.append(
            (
                workspace_id,
                agent_run_id,
            ),
        )
        return self._reference


def _agent_run() -> AgentRun:
    return AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=_INGESTION_REQUEST_ID,
        correlation_id=_CORRELATION_ID,
        workflow_version=(TICKET_CLASSIFICATION_WORKFLOW_VERSION),
        max_retryable_failures=3,
        now=_NOW,
    )


def _reference() -> AgentRunClassificationReference:
    return AgentRunClassificationReference(
        id=_CLASSIFICATION_ID,
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        created_at=_NOW,
    )


async def test_returns_agent_run_with_classification_reference() -> None:
    agent_run_repository = FakeAgentRunQueryRepository(
        _agent_run(),
    )
    classification_repository = FakeClassificationQueryRepository(
        _reference(),
    )
    service = GetAgentRunInspection(
        agent_run_repository=cast(
            AgentRunQueryRepository,
            agent_run_repository,
        ),
        classification_repository=cast(
            TicketClassificationQueryRepository,
            classification_repository,
        ),
    )

    result = await service.execute(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    assert result.agent_run == _agent_run()
    assert result.classification == _reference()
    assert agent_run_repository.get_calls == [
        (
            _WORKSPACE_ID,
            _AGENT_RUN_ID,
        ),
    ]
    assert classification_repository.reference_calls == [
        (
            _WORKSPACE_ID,
            _AGENT_RUN_ID,
        ),
    ]


async def test_returns_none_when_run_has_no_classification() -> None:
    service = GetAgentRunInspection(
        agent_run_repository=cast(
            AgentRunQueryRepository,
            FakeAgentRunQueryRepository(
                _agent_run(),
            ),
        ),
        classification_repository=cast(
            TicketClassificationQueryRepository,
            FakeClassificationQueryRepository(
                None,
            ),
        ),
    )

    result = await service.execute(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    assert result.agent_run == _agent_run()
    assert result.classification is None


@pytest.mark.parametrize(
    "workspace_id",
    [
        _WORKSPACE_ID,
        _OTHER_WORKSPACE_ID,
    ],
)
async def test_missing_or_cross_workspace_run_uses_stable_not_found(
    workspace_id: UUID,
) -> None:
    classification_repository = FakeClassificationQueryRepository(
        _reference(),
    )
    service = GetAgentRunInspection(
        agent_run_repository=cast(
            AgentRunQueryRepository,
            FakeAgentRunQueryRepository(None if workspace_id == _WORKSPACE_ID else _agent_run()),
        ),
        classification_repository=cast(
            TicketClassificationQueryRepository,
            classification_repository,
        ),
    )

    with pytest.raises(
        AgentRunNotFoundError,
        match="AgentRun was not found",
    ):
        await service.execute(
            workspace_id=workspace_id,
            agent_run_id=_AGENT_RUN_ID,
        )

    assert classification_repository.reference_calls == []
