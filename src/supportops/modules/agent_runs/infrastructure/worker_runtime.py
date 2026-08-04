"""Session-scoped PostgreSQL composition for one AgentRun worker cycle."""

from collections.abc import Callable
from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from supportops.agent_tools.infrastructure.repository import (
    SqlAlchemyAgentToolCallExecutionRepository,
)
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
from supportops.modules.approvals.application.services import (
    ExpirePendingApprovalRequests,
)
from supportops.modules.approvals.infrastructure.repository import (
    SqlAlchemyApprovalRequestRepository,
)
from supportops.modules.tickets.infrastructure.repository import (
    SqlAlchemyTicketRepository,
)
from supportops.observability.contracts import ObservabilityClient
from supportops.observability.noop import NoOpObservabilityClient

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
        approval_expiration_batch_size: int,
        utc_now: UtcNowProvider | None = None,
        uuid_provider: UuidProvider | None = None,
        observability_client: ObservabilityClient | None = None,
        flush_observability_at_attempt_end: bool = False,
    ) -> None:
        self._session_factory = session_factory
        self._worker_id = worker_id
        self._executor_factory = executor_factory
        self._retry_policy = retry_policy
        self._lease_seconds = lease_seconds
        self._execution_timeout_seconds = execution_timeout_seconds
        self._approval_expiration_batch_size = approval_expiration_batch_size
        self._utc_now = utc_now
        self._uuid_provider = uuid_provider
        self._observability_client = observability_client or NoOpObservabilityClient()
        self._flush_observability_at_attempt_end = flush_observability_at_attempt_end

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
        approval_request_repository = SqlAlchemyApprovalRequestRepository(
            session,
        )
        agent_tool_call_repository = SqlAlchemyAgentToolCallExecutionRepository(
            session,
        )
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
            observability_client=self._observability_client,
            flush_observability_at_attempt_end=(self._flush_observability_at_attempt_end),
        )

        expire_pending_approvals = ExpirePendingApprovalRequests(
            transaction_manager=transaction_manager,
            approval_request_repository=approval_request_repository,
            agent_run_repository=agent_run_repository,
            agent_tool_call_repository=agent_tool_call_repository,
            observability_client=self._observability_client,
        )

        return RunAgentWorkerCycle(
            worker_id=self._worker_id,
            agent_run_repository=agent_run_repository,
            transaction_manager=transaction_manager,
            processor=processor,
            retry_policy=self._retry_policy,
            lease_seconds=self._lease_seconds,
            expire_pending_approvals=expire_pending_approvals,
            approval_expiration_batch_size=(self._approval_expiration_batch_size),
            utc_now=self._utc_now,
            uuid_provider=self._uuid_provider,
        )
