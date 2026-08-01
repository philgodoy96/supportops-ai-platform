"""Unit tests for the SQLAlchemy AgentRun repository."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, create_autospec
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    AgentRun,
)
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunRecord,
)
from supportops.modules.agent_runs.infrastructure.repository import (
    SqlAlchemyAgentRunRepository,
)


def create_agent_run() -> AgentRun:
    return AgentRun.create_initial(
        agent_run_id=UUID(
            "69184ef1-4d71-452e-8070-0b784c29368e",
        ),
        workspace_id=UUID(
            "032c8c87-57cc-4d14-bfbd-04968b4e8cd4",
        ),
        ticket_id=UUID(
            "38bb60fe-d2ea-4615-b499-91aa45069019",
        ),
        ingestion_request_id=UUID(
            "725eec8a-c504-4071-ac96-c78cc907f26c",
        ),
        correlation_id=UUID(
            "1038c98e-62fd-45df-9839-138f7105cb78",
        ),
        workflow_version=DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
        max_attempts=3,
        now=datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    )


async def test_add_persists_agent_run_record_and_flushes() -> None:
    session = create_autospec(
        AsyncSession,
        instance=True,
    )
    session.add = MagicMock()
    session.flush = AsyncMock()

    repository = SqlAlchemyAgentRunRepository(session)
    agent_run = create_agent_run()

    await repository.add(agent_run)

    session.add.assert_called_once()

    persisted_record = session.add.call_args.args[0]

    assert isinstance(persisted_record, AgentRunRecord)
    assert persisted_record.to_domain() == agent_run
    session.flush.assert_awaited_once_with()


async def test_add_does_not_commit_the_active_transaction() -> None:
    session = create_autospec(
        AsyncSession,
        instance=True,
    )
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()

    repository = SqlAlchemyAgentRunRepository(session)

    await repository.add(create_agent_run())

    session.flush.assert_awaited_once_with()
    session.commit.assert_not_awaited()


async def test_add_propagates_flush_failure() -> None:
    session = create_autospec(
        AsyncSession,
        instance=True,
    )
    session.add = MagicMock()
    session.flush = AsyncMock(
        side_effect=RuntimeError("database flush failed"),
    )

    repository = SqlAlchemyAgentRunRepository(session)

    try:
        await repository.add(create_agent_run())
    except RuntimeError as error:
        assert str(error) == "database flush failed"
    else:
        raise AssertionError("Expected the flush failure to propagate.")

    session.add.assert_called_once()
    session.flush.assert_awaited_once_with()
