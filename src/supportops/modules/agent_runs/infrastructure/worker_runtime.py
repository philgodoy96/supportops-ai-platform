"""Session-scoped PostgreSQL composition for one AgentRun worker cycle."""

from collections.abc import Callable
from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from supportops.core.transactions import TransactionManager
from supportops.infrastructure.postgresql.transaction import (
    SqlAlchemyTransactionManager,
)
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutor,
)
from supportops.modules.agent_runs.application.processor import (
    ProcessClaimedAgentRun,
)
from supportops.modules.agent_runs.application.retry_policy import (
    AgentRunRetryPolicy,
)
from supportops.modules.agent_runs.application.worker import (
    RunAgentWorkerCycle,
    WorkerCycleResult,
)
from supportops.modules.agent_runs.infrastructure.repository import (
    SqlAlchemyAgentRunRepository,
)
from supportops.modules.tickets.infrastructure.repository import (
    SqlAlchemyTicketRepository,
)

UtcNowProvider = Callable[[], datetime]
UuidProvider = Callable[[], UUID]
AgentRunExecutorFactory = Callable[
    [AsyncSession, TransactionManager],
    AgentRunExecutor,
]


class PostgreSqlAgentWorkerCycleRunner:
    """Run one fully composed worker cycle inside a scoped session."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        worker_id: str,
        executor_factory: AgentRunExecutorFactory,
        retry_policy: AgentRunRetryPolicy,
        lease_seconds: float,
        execution_timeout_seconds: float,
        utc_now: UtcNowProvider | None = None,
        uuid_provider: UuidProvider | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._worker_id = worker_id
        self._executor_factory = executor_factory
        self._retry_policy = retry_policy
        self._lease_seconds = lease_seconds
        self._execution_timeout_seconds = execution_timeout_seconds
        self._utc_now = utc_now
        self._uuid_provider = uuid_provider

    async def execute(self) -> WorkerCycleResult:
        """Open one session, execute one cycle, and close the session."""

        async with self._session_factory() as session:
            cycle = self._build_cycle(session)
            return await cycle.execute()

    def _build_cycle(
        self,
        session: AsyncSession,
    ) -> RunAgentWorkerCycle:
        agent_run_repository = SqlAlchemyAgentRunRepository(session)
        ticket_repository = SqlAlchemyTicketRepository(session)
        transaction_manager = SqlAlchemyTransactionManager(session)
        executor = self._executor_factory(
            session,
            transaction_manager,
        )

        processor = ProcessClaimedAgentRun(
            ticket_repository=ticket_repository,
            agent_run_repository=agent_run_repository,
            transaction_manager=transaction_manager,
            executor=executor,
            retry_policy=self._retry_policy,
            execution_timeout_seconds=self._execution_timeout_seconds,
            utc_now=self._utc_now,
        )

        return RunAgentWorkerCycle(
            worker_id=self._worker_id,
            agent_run_repository=agent_run_repository,
            transaction_manager=transaction_manager,
            processor=processor,
            retry_policy=self._retry_policy,
            lease_seconds=self._lease_seconds,
            utc_now=self._utc_now,
            uuid_provider=self._uuid_provider,
        )
