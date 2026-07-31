"""Unit tests for AgentRun inspection application services."""

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

import pytest

from supportops.modules.agent_runs.application.errors import (
    AgentRunNotFoundError,
)
from supportops.modules.agent_runs.application.services import (
    GetAgentRun,
    ListAgentRunAttempts,
)
from supportops.modules.agent_runs.domain.models import (
    AgentRun,
    AgentRunAttempt,
)

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
_OTHER_WORKSPACE_ID = UUID(
    "db94eb06-c97d-47f8-9214-79558ba933c9",
)
_TICKET_ID = UUID(
    "38bb60fe-d2ea-4615-b499-91aa45069019",
)
_AGENT_RUN_ID = UUID(
    "69184ef1-4d71-452e-8070-0b784c29368e",
)
_INGESTION_REQUEST_ID = UUID(
    "725eec8a-c504-4071-ac96-c78cc907f26c",
)
_CORRELATION_ID = UUID(
    "1038c98e-62fd-45df-9839-138f7105cb78",
)
_ATTEMPT_ONE_ID = UUID(
    "2b39f5b7-b2a4-48d0-b079-fdad286d5315",
)
_ATTEMPT_TWO_ID = UUID(
    "626e0940-cf3b-4b9f-ad49-98bce214469b",
)
_LEASE_TOKEN_ONE = UUID(
    "dd0ae456-3467-41db-93d1-a908f40e8365",
)
_LEASE_TOKEN_TWO = UUID(
    "b36000c4-62d7-4fe1-ad40-96872a245409",
)
_EXECUTION_REQUEST_ONE = UUID(
    "d1fa068f-2278-47a8-b3c9-39ccf91f0a5e",
)
_EXECUTION_REQUEST_TWO = UUID(
    "99988e91-f292-4ada-81b6-58551c96f02b",
)


class FakeAgentRunQueryRepository:
    """Store deterministic AgentRun query results for application tests."""

    def __init__(
        self,
        *,
        agent_runs: Sequence[AgentRun] = (),
        attempts: Sequence[AgentRunAttempt] = (),
    ) -> None:
        self._agent_runs = {
            (agent_run.workspace_id, agent_run.id): agent_run for agent_run in agent_runs
        }
        self._attempts = tuple(attempts)
        self.get_calls: list[tuple[UUID, UUID]] = []
        self.list_attempt_calls: list[UUID] = []

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
        return self._agent_runs.get(
            (
                workspace_id,
                agent_run_id,
            ),
        )

    async def list_attempts(
        self,
        *,
        agent_run_id: UUID,
    ) -> Sequence[AgentRunAttempt]:
        self.list_attempt_calls.append(agent_run_id)

        return tuple(
            sorted(
                (attempt for attempt in self._attempts if attempt.agent_run_id == agent_run_id),
                key=lambda attempt: attempt.attempt_number,
            ),
        )


def create_agent_run() -> AgentRun:
    """Create one deterministic queued AgentRun."""

    return AgentRun.create_initial(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=_INGESTION_REQUEST_ID,
        correlation_id=_CORRELATION_ID,
        max_attempts=3,
        agent_run_id=_AGENT_RUN_ID,
        now=_NOW,
    )


def create_attempt(
    *,
    attempt_id: UUID,
    attempt_number: int,
    lease_token: UUID,
    execution_request_id: UUID,
) -> AgentRunAttempt:
    """Create one deterministic active attempt."""

    return AgentRunAttempt.start(
        agent_run_id=_AGENT_RUN_ID,
        attempt_number=attempt_number,
        worker_id="worker-a",
        lease_token=lease_token,
        execution_request_id=execution_request_id,
        attempt_id=attempt_id,
        now=_NOW,
    )


async def test_get_agent_run_returns_workspace_scoped_run() -> None:
    agent_run = create_agent_run()
    repository = FakeAgentRunQueryRepository(
        agent_runs=(agent_run,),
    )
    service = GetAgentRun(repository=repository)

    result = await service.execute(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    assert result == agent_run
    assert repository.get_calls == [
        (
            _WORKSPACE_ID,
            _AGENT_RUN_ID,
        ),
    ]


async def test_get_agent_run_raises_when_run_does_not_exist() -> None:
    repository = FakeAgentRunQueryRepository()
    service = GetAgentRun(repository=repository)

    with pytest.raises(
        AgentRunNotFoundError,
        match="AgentRun was not found",
    ):
        await service.execute(
            workspace_id=_WORKSPACE_ID,
            agent_run_id=_AGENT_RUN_ID,
        )


async def test_get_agent_run_hides_cross_workspace_resource() -> None:
    agent_run = create_agent_run()
    repository = FakeAgentRunQueryRepository(
        agent_runs=(agent_run,),
    )
    service = GetAgentRun(repository=repository)

    with pytest.raises(
        AgentRunNotFoundError,
        match="AgentRun was not found",
    ):
        await service.execute(
            workspace_id=_OTHER_WORKSPACE_ID,
            agent_run_id=_AGENT_RUN_ID,
        )

    assert repository.get_calls == [
        (
            _OTHER_WORKSPACE_ID,
            _AGENT_RUN_ID,
        ),
    ]


async def test_list_attempts_returns_empty_history_for_queued_run() -> None:
    agent_run = create_agent_run()
    repository = FakeAgentRunQueryRepository(
        agent_runs=(agent_run,),
    )
    service = ListAgentRunAttempts(
        repository=repository,
    )

    result = await service.execute(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    assert result == ()
    assert repository.get_calls == [
        (
            _WORKSPACE_ID,
            _AGENT_RUN_ID,
        ),
    ]
    assert repository.list_attempt_calls == [
        _AGENT_RUN_ID,
    ]


async def test_list_attempts_returns_deterministic_history() -> None:
    agent_run = create_agent_run()
    first_attempt = create_attempt(
        attempt_id=_ATTEMPT_ONE_ID,
        attempt_number=1,
        lease_token=_LEASE_TOKEN_ONE,
        execution_request_id=_EXECUTION_REQUEST_ONE,
    )
    second_attempt = create_attempt(
        attempt_id=_ATTEMPT_TWO_ID,
        attempt_number=2,
        lease_token=_LEASE_TOKEN_TWO,
        execution_request_id=_EXECUTION_REQUEST_TWO,
    )
    repository = FakeAgentRunQueryRepository(
        agent_runs=(agent_run,),
        attempts=(
            second_attempt,
            first_attempt,
        ),
    )
    service = ListAgentRunAttempts(
        repository=repository,
    )

    result = await service.execute(
        workspace_id=_WORKSPACE_ID,
        agent_run_id=_AGENT_RUN_ID,
    )

    assert result == (
        first_attempt,
        second_attempt,
    )
    assert repository.list_attempt_calls == [
        _AGENT_RUN_ID,
    ]


async def test_list_attempts_rejects_missing_run_before_history_query() -> None:
    repository = FakeAgentRunQueryRepository()
    service = ListAgentRunAttempts(
        repository=repository,
    )

    with pytest.raises(
        AgentRunNotFoundError,
        match="AgentRun was not found",
    ):
        await service.execute(
            workspace_id=_WORKSPACE_ID,
            agent_run_id=_AGENT_RUN_ID,
        )

    assert repository.list_attempt_calls == []


async def test_list_attempts_rejects_cross_workspace_run() -> None:
    agent_run = create_agent_run()
    repository = FakeAgentRunQueryRepository(
        agent_runs=(agent_run,),
    )
    service = ListAgentRunAttempts(
        repository=repository,
    )

    with pytest.raises(
        AgentRunNotFoundError,
        match="AgentRun was not found",
    ):
        await service.execute(
            workspace_id=_OTHER_WORKSPACE_ID,
            agent_run_id=_AGENT_RUN_ID,
        )

    assert repository.list_attempt_calls == []
