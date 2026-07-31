"""Application orchestration for one claimed AgentRun execution."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from supportops.core.transactions import TransactionManager
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    AgentRunExecutor,
    RetryableAgentRunExecutionError,
    TerminalAgentRunExecutionError,
)
from supportops.modules.agent_runs.application.retry_policy import (
    AgentRunRetryPolicy,
)
from supportops.modules.agent_runs.domain.claiming import AgentRunClaim
from supportops.modules.agent_runs.domain.models import (
    AgentRunAttemptOutcome,
)
from supportops.modules.agent_runs.domain.repositories import (
    AgentRunRepository,
)
from supportops.modules.agent_runs.domain.transitions import (
    AgentRunFailureDisposition,
    AgentRunTransitionResult,
    CompleteAgentRunCommand,
    FailAgentRunCommand,
)
from supportops.modules.tickets.domain.repositories import (
    TicketRepository,
)

UtcNowProvider = Callable[[], datetime]


class ProcessClaimedAgentRun:
    """Execute one claimed AgentRun and persist its fenced outcome."""

    def __init__(
        self,
        *,
        ticket_repository: TicketRepository,
        agent_run_repository: AgentRunRepository,
        transaction_manager: TransactionManager,
        executor: AgentRunExecutor,
        retry_policy: AgentRunRetryPolicy,
        execution_timeout_seconds: float,
        utc_now: UtcNowProvider | None = None,
    ) -> None:
        if execution_timeout_seconds <= 0:
            raise ValueError(
                "execution_timeout_seconds must be greater than zero.",
            )

        self._ticket_repository = ticket_repository
        self._agent_run_repository = agent_run_repository
        self._transaction_manager = transaction_manager
        self._executor = executor
        self._retry_policy = retry_policy
        self._execution_timeout_seconds = execution_timeout_seconds
        self._utc_now = utc_now or _utc_now

    async def execute(
        self,
        claim: AgentRunClaim,
    ) -> AgentRunTransitionResult:
        """Execute one claimed run outside the persistence transactions."""

        run = claim.agent_run

        async with self._transaction_manager.transaction():
            ticket = await self._ticket_repository.get(
                run.workspace_id,
                run.ticket_id,
            )

        if ticket is None:
            return await self._persist_failure(
                claim=claim,
                finished_at=self._utc_now(),
                outcome=AgentRunAttemptOutcome.TERMINAL_FAILURE,
                error_code="ticket_not_found",
                error_summary=("The ticket referenced by the AgentRun was not found."),
                retryable=False,
            )

        context = AgentRunExecutionContext(
            agent_run=run,
            ticket=ticket,
        )

        try:
            async with asyncio.timeout(
                self._execution_timeout_seconds,
            ):
                await self._executor.execute(context)
        except TimeoutError:
            return await self._persist_failure(
                claim=claim,
                finished_at=self._utc_now(),
                outcome=AgentRunAttemptOutcome.TIMED_OUT,
                error_code="executor_timeout",
                error_summary=("The configured executor exceeded its execution timeout."),
                retryable=True,
            )
        except TerminalAgentRunExecutionError as error:
            return await self._persist_failure(
                claim=claim,
                finished_at=self._utc_now(),
                outcome=AgentRunAttemptOutcome.TERMINAL_FAILURE,
                error_code=error.error_code,
                error_summary=error.error_summary,
                retryable=False,
            )
        except RetryableAgentRunExecutionError as error:
            return await self._persist_failure(
                claim=claim,
                finished_at=self._utc_now(),
                outcome=AgentRunAttemptOutcome.RETRYABLE_FAILURE,
                error_code=error.error_code,
                error_summary=error.error_summary,
                retryable=True,
            )
        except Exception:
            return await self._persist_failure(
                claim=claim,
                finished_at=self._utc_now(),
                outcome=AgentRunAttemptOutcome.RETRYABLE_FAILURE,
                error_code="unexpected_executor_failure",
                error_summary=("The executor failed unexpectedly while processing the AgentRun."),
                retryable=True,
            )

        finished_at = self._utc_now()

        async with self._transaction_manager.transaction():
            return await self._agent_run_repository.mark_succeeded(
                CompleteAgentRunCommand(
                    agent_run_id=run.id,
                    lease_token=claim.attempt.lease_token,
                    finished_at=finished_at,
                ),
            )

    async def _persist_failure(
        self,
        *,
        claim: AgentRunClaim,
        finished_at: datetime,
        outcome: AgentRunAttemptOutcome,
        error_code: str,
        error_summary: str,
        retryable: bool,
    ) -> AgentRunTransitionResult:
        run = claim.agent_run

        retry_allowed = retryable and self._retry_policy.can_retry(
            attempt_count=run.attempt_count,
            max_attempts=run.max_attempts,
        )

        if retry_allowed:
            disposition = AgentRunFailureDisposition.RETRY_SCHEDULED
            delay_seconds = self._retry_policy.delay_seconds_for_attempt(
                attempt_number=run.attempt_count,
            )
            retry_available_at = finished_at + timedelta(
                seconds=delay_seconds,
            )
        else:
            disposition = AgentRunFailureDisposition.FAILED
            retry_available_at = None

        async with self._transaction_manager.transaction():
            return await self._agent_run_repository.record_failure(
                FailAgentRunCommand(
                    agent_run_id=run.id,
                    lease_token=claim.attempt.lease_token,
                    finished_at=finished_at,
                    outcome=outcome,
                    disposition=disposition,
                    error_code=error_code,
                    error_summary=error_summary,
                    retry_available_at=retry_available_at,
                ),
            )


def _utc_now() -> datetime:
    return datetime.now(UTC)
