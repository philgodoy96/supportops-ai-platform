"""Unit tests for session-scoped PostgreSQL worker composition."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.agent_tools.infrastructure.repository import (
    SqlAlchemyAgentToolCallExecutionRepository,
)
from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.agent_runs.application.retry_policy import (
    AgentRunRetryPolicy,
)
from supportops.modules.agent_runs.application.worker import (
    RunAgentWorkerCycle,
    WorkerCycleOutcome,
    WorkerCycleResult,
)
from supportops.modules.agent_runs.infrastructure.repository import (
    SqlAlchemyAgentRunRepository,
)
from supportops.modules.agent_runs.infrastructure.worker_runtime import (
    PostgreSqlAgentWorkerCycleRunner,
)
from supportops.modules.approvals.infrastructure.repository import (
    SqlAlchemyApprovalRequestRepository,
)

_NOW = datetime(
    2026,
    7,
    31,
    18,
    0,
    tzinfo=UTC,
)
_RUN_ID = UUID(
    "69184ef1-4d71-452e-8070-0b784c29368e",
)


class RecordingSessionFactory:
    """Produce scoped fake sessions and record lifecycle events."""

    def __init__(self) -> None:
        self.sessions_created = 0
        self.sessions_entered = 0
        self.sessions_exited = 0
        self.session = AsyncMock(spec=AsyncSession)

    def __call__(self) -> AbstractAsyncContextManager[AsyncSession]:
        self.sessions_created += 1

        @asynccontextmanager
        async def session_scope() -> AsyncIterator[AsyncSession]:
            self.sessions_entered += 1

            try:
                yield self.session
            finally:
                self.sessions_exited += 1

        return session_scope()


class RecordingCycle:
    """Return a predefined result and record invocation."""

    def __init__(
        self,
        result: WorkerCycleResult,
    ) -> None:
        self._result = result
        self.executions = 0

    async def execute(self) -> WorkerCycleResult:
        self.executions += 1
        return self._result


class TestablePostgreSqlAgentWorkerCycleRunner(PostgreSqlAgentWorkerCycleRunner):
    """Override composition so session scoping can be tested in isolation."""

    def __init__(
        self,
        *,
        session_factory: RecordingSessionFactory,
        cycle: RecordingCycle,
    ) -> None:
        super().__init__(
            session_factory=session_factory,  # type: ignore[arg-type]
            worker_id="worker-a",
            executor_factory=lambda session, transaction_manager: AsyncMock(),
            retry_policy=AsyncMock(),
            lease_seconds=45.0,
            execution_timeout_seconds=30.0,
            approval_expiration_batch_size=100,
            utc_now=lambda: _NOW,
            uuid_provider=lambda: UUID(
                "dd0ae456-3467-41db-93d1-a908f40e8365",
            ),
        )
        self._recording_cycle = cycle
        self.sessions_received: list[AsyncSession] = []

    def _build_cycle(
        self,
        session: AsyncSession,
    ) -> RunAgentWorkerCycle:
        self.sessions_received.append(session)
        return cast(RunAgentWorkerCycle, self._recording_cycle)


async def test_runner_opens_and_closes_session_for_one_cycle() -> None:
    session_factory = RecordingSessionFactory()
    cycle = RecordingCycle(
        WorkerCycleResult(
            outcome=WorkerCycleOutcome.PROCESSED,
            recovered_expired_run=False,
            agent_run_id=_RUN_ID,
        ),
    )
    runner = TestablePostgreSqlAgentWorkerCycleRunner(
        session_factory=session_factory,
        cycle=cycle,
    )

    result = await runner.execute()

    assert result.outcome is WorkerCycleOutcome.PROCESSED
    assert result.agent_run_id == _RUN_ID

    assert session_factory.sessions_created == 1
    assert session_factory.sessions_entered == 1
    assert session_factory.sessions_exited == 1

    assert runner.sessions_received == [
        session_factory.session,
    ]
    assert cycle.executions == 1


async def test_runner_uses_a_new_session_for_each_cycle() -> None:
    session_factory = RecordingSessionFactory()
    cycle = RecordingCycle(
        WorkerCycleResult(
            outcome=WorkerCycleOutcome.IDLE,
            recovered_expired_run=False,
            agent_run_id=None,
        ),
    )
    runner = TestablePostgreSqlAgentWorkerCycleRunner(
        session_factory=session_factory,
        cycle=cycle,
    )

    first_result = await runner.execute()
    second_result = await runner.execute()

    assert first_result.outcome is WorkerCycleOutcome.IDLE
    assert second_result.outcome is WorkerCycleOutcome.IDLE

    assert session_factory.sessions_created == 2
    assert session_factory.sessions_entered == 2
    assert session_factory.sessions_exited == 2
    assert cycle.executions == 2


async def test_runner_closes_session_when_cycle_raises() -> None:
    session_factory = RecordingSessionFactory()
    cycle = AsyncMock()
    cycle.execute.side_effect = RuntimeError(
        "unexpected worker cycle failure",
    )

    class FailingRunner(PostgreSqlAgentWorkerCycleRunner):
        def _build_cycle(
            self,
            session: AsyncSession,
        ) -> RunAgentWorkerCycle:
            del session
            return cast(RunAgentWorkerCycle, cycle)

    runner = FailingRunner(
        session_factory=session_factory,  # type: ignore[arg-type]
        worker_id="worker-a",
        executor_factory=lambda session, transaction_manager: AsyncMock(),
        retry_policy=AsyncMock(),
        lease_seconds=45.0,
        execution_timeout_seconds=30.0,
        approval_expiration_batch_size=100,
        utc_now=lambda: _NOW,
    )

    with pytest.raises(
        RuntimeError,
        match="unexpected worker cycle failure",
    ):
        await runner.execute()

    assert session_factory.sessions_created == 1
    assert session_factory.sessions_entered == 1
    assert session_factory.sessions_exited == 1


def test_build_cycle_injects_process_observability_client() -> None:
    session = AsyncMock(spec=AsyncSession)
    observability_client = object()

    runner = PostgreSqlAgentWorkerCycleRunner(
        session_factory=AsyncMock(),
        worker_id="worker-a",
        executor_factory=lambda session, transaction_manager: AsyncMock(),
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        lease_seconds=45.0,
        execution_timeout_seconds=30.0,
        approval_expiration_batch_size=25,
        utc_now=lambda: _NOW,
        uuid_provider=lambda: UUID(
            "dd0ae456-3467-41db-93d1-a908f40e8365",
        ),
        observability_client=observability_client,  # type: ignore[arg-type]
    )

    cycle = runner._build_cycle(session)

    assert cycle._processor._observability_client is observability_client
    assert cycle._expire_pending_approvals._observability_client is (observability_client)
    assert cycle._processor._flush_observability_at_attempt_end is False


@pytest.mark.parametrize("flush_at_attempt_end", [False, True])
def test_build_cycle_passes_flush_policy_to_processor(
    flush_at_attempt_end: bool,
) -> None:
    session = AsyncMock(spec=AsyncSession)
    observability_client = object()

    runner = PostgreSqlAgentWorkerCycleRunner(
        session_factory=AsyncMock(),
        worker_id="worker-a",
        executor_factory=lambda session, transaction_manager: AsyncMock(),
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        lease_seconds=45.0,
        execution_timeout_seconds=30.0,
        approval_expiration_batch_size=25,
        utc_now=lambda: _NOW,
        uuid_provider=lambda: UUID(
            "dd0ae456-3467-41db-93d1-a908f40e8365",
        ),
        observability_client=observability_client,  # type: ignore[arg-type]
        flush_observability_at_attempt_end=flush_at_attempt_end,
    )

    cycle = runner._build_cycle(session)

    assert cycle._processor._observability_client is observability_client
    assert cycle._processor._flush_observability_at_attempt_end is (flush_at_attempt_end)
    assert runner._flush_observability_at_attempt_end is flush_at_attempt_end


def test_build_cycle_calls_executor_factory_with_session_and_transaction_manager() -> None:
    session = AsyncMock(spec=AsyncSession)
    created_executor = AsyncMock()
    factory_calls: list[tuple[AsyncSession, object]] = []

    def executor_factory(
        received_session: AsyncSession,
        transaction_manager: object,
    ) -> AsyncMock:
        factory_calls.append((received_session, transaction_manager))
        return created_executor

    runner = PostgreSqlAgentWorkerCycleRunner(
        session_factory=AsyncMock(),
        worker_id="worker-a",
        executor_factory=executor_factory,
        retry_policy=AgentRunRetryPolicy(
            base_delay_seconds=2.0,
            maximum_delay_seconds=60.0,
        ),
        lease_seconds=45.0,
        execution_timeout_seconds=30.0,
        approval_expiration_batch_size=25,
        utc_now=lambda: _NOW,
        uuid_provider=lambda: UUID(
            "dd0ae456-3467-41db-93d1-a908f40e8365",
        ),
    )

    cycle = runner._build_cycle(session)

    assert len(factory_calls) == 1
    assert factory_calls[0][0] is session
    assert isinstance(factory_calls[0][1], SqlAlchemyTransactionManager)
    assert cycle._processor._executor is created_executor
    assert cycle._processor._observability_client is runner._observability_client
    assert cycle._approval_expiration_batch_size == 25
    assert cycle._expire_pending_approvals is not None
    assert isinstance(
        cycle._expire_pending_approvals._approval_request_repository,
        SqlAlchemyApprovalRequestRepository,
    )
    assert isinstance(
        cycle._expire_pending_approvals._agent_run_repository,
        SqlAlchemyAgentRunRepository,
    )
    assert isinstance(
        cycle._expire_pending_approvals._agent_tool_call_repository,
        SqlAlchemyAgentToolCallExecutionRepository,
    )
    assert cycle._expire_pending_approvals._approval_request_repository._session is session
    assert cycle._expire_pending_approvals._agent_run_repository._session is session
    assert cycle._expire_pending_approvals._agent_tool_call_repository._session is session
    assert cycle._expire_pending_approvals._transaction_manager is (cycle._transaction_manager)
    assert isinstance(
        cycle._expire_pending_approvals._transaction_manager,
        SqlAlchemyTransactionManager,
    )
