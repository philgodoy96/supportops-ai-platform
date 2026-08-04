"""Application orchestration for one claimed AgentRun execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from supportops.core.transactions import TransactionManager
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    AgentRunExecutor,
    CompletedExecution,
    PausedForApproval,
    RetryableAgentRunExecutionError,
    TerminalAgentRunExecutionError,
)
from supportops.modules.agent_runs.application.retry_policy import (
    AgentRunRetryPolicy,
)
from supportops.modules.agent_runs.domain.claiming import AgentRunClaim
from supportops.modules.agent_runs.domain.models import (
    AgentRun,
    AgentRunAttempt,
    AgentRunAttemptOutcome,
    AgentRunStatus,
)
from supportops.modules.agent_runs.domain.repositories import (
    AgentRunRepository,
)
from supportops.modules.agent_runs.domain.transitions import (
    AgentRunFailureDisposition,
    AgentRunTransitionResult,
    CompleteAgentRunCommand,
    FailAgentRunCommand,
    WaitForApprovalAgentRunCommand,
)
from supportops.modules.tickets.domain.repositories import (
    TicketRepository,
)
from supportops.observability.contracts import (
    ObservabilityClient,
    ObservationScope,
    TraceScope,
)
from supportops.observability.identity import (
    TraceIdentity,
    agent_run_trace_identity,
)
from supportops.observability.models import (
    FieldPaths,
    JsonValue,
    ObservationAttributes,
    ObservationStatus,
    ObservationType,
    ObservationUpdate,
    TraceAttributes,
)
from supportops.observability.noop import NoOpObservabilityClient

UtcNowProvider = Callable[[], datetime]

_AGENT_RUN_TRACE_METADATA_PATHS: FieldPaths = frozenset(
    {
        ("agent_run_id",),
        ("workspace_id",),
        ("ticket_id",),
        ("workflow_name",),
        ("workflow_version",),
        ("trigger_key",),
        ("correlation_id",),
        ("agent_run_status",),
        ("latest_attempt_outcome",),
    }
)

_WORKER_ATTEMPT_METADATA_PATHS: FieldPaths = frozenset(
    {
        ("agent_run_id",),
        ("agent_run_attempt_id",),
        ("attempt_number",),
        ("execution_request_id",),
        ("workspace_id",),
        ("ticket_id",),
        ("workflow_name",),
        ("workflow_version",),
        ("trigger_key",),
        ("correlation_id",),
        ("worker_id",),
        ("attempt_outcome",),
    }
)


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
        observability_client: ObservabilityClient | None = None,
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
        self._observability_client = observability_client or NoOpObservabilityClient()

    async def execute(
        self,
        claim: AgentRunClaim,
    ) -> AgentRunTransitionResult:
        """Execute one claimed run outside the persistence transactions."""

        run = claim.agent_run
        identity = agent_run_trace_identity(
            agent_run_id=run.id,
            ticket_id=run.ticket_id,
        )
        telemetry = _SafeAgentRunAttemptTelemetry(
            client=self._observability_client,
            identity=identity,
            run=run,
            attempt=claim.attempt,
        )
        telemetry.start()

        try:
            result, outcome = await self._execute_claimed(claim)
            telemetry.complete(outcome)
            return result
        except Exception:
            telemetry.complete(
                _AttemptTelemetryOutcome(
                    attempt_status=ObservationStatus.ERROR,
                    attempt_outcome=None,
                    error_code="unhandled_business_error",
                    trace_status=ObservationStatus.ERROR,
                    agent_run_status=None,
                    latest_attempt_outcome=None,
                )
            )
            raise
        finally:
            telemetry.close()

    async def _execute_claimed(
        self,
        claim: AgentRunClaim,
    ) -> tuple[AgentRunTransitionResult, _AttemptTelemetryOutcome]:
        run = claim.agent_run

        async with self._transaction_manager.transaction():
            ticket = await self._ticket_repository.get(
                run.workspace_id,
                run.ticket_id,
            )

        if ticket is None:
            result = await self._persist_failure(
                claim=claim,
                finished_at=self._utc_now(),
                outcome=AgentRunAttemptOutcome.TERMINAL_FAILURE,
                error_code="ticket_not_found",
                error_summary=("The ticket referenced by the AgentRun was not found."),
                retryable=False,
            )
            return result, _map_failure_outcome(
                transition_result=result,
                attempt_outcome=AgentRunAttemptOutcome.TERMINAL_FAILURE,
                error_code="ticket_not_found",
                disposition=AgentRunFailureDisposition.FAILED,
            )

        context = AgentRunExecutionContext(
            agent_run=run,
            attempt=claim.attempt,
            ticket=ticket,
        )

        try:
            async with asyncio.timeout(
                self._execution_timeout_seconds,
            ):
                execution_result = await self._executor.execute(context)
        except TimeoutError:
            result = await self._persist_failure(
                claim=claim,
                finished_at=self._utc_now(),
                outcome=AgentRunAttemptOutcome.TIMED_OUT,
                error_code="executor_timeout",
                error_summary=("The configured executor exceeded its execution timeout."),
                retryable=True,
            )
            return result, _map_failure_outcome(
                transition_result=result,
                attempt_outcome=AgentRunAttemptOutcome.TIMED_OUT,
                error_code="executor_timeout",
                disposition=_failure_disposition(
                    claim=claim,
                    retryable=True,
                    retry_policy=self._retry_policy,
                ),
            )
        except TerminalAgentRunExecutionError as error:
            result = await self._persist_failure(
                claim=claim,
                finished_at=self._utc_now(),
                outcome=AgentRunAttemptOutcome.TERMINAL_FAILURE,
                error_code=error.error_code,
                error_summary=error.error_summary,
                retryable=False,
            )
            return result, _map_failure_outcome(
                transition_result=result,
                attempt_outcome=AgentRunAttemptOutcome.TERMINAL_FAILURE,
                error_code=error.error_code,
                disposition=AgentRunFailureDisposition.FAILED,
            )
        except RetryableAgentRunExecutionError as error:
            disposition = _failure_disposition(
                claim=claim,
                retryable=True,
                retry_policy=self._retry_policy,
            )
            result = await self._persist_failure(
                claim=claim,
                finished_at=self._utc_now(),
                outcome=AgentRunAttemptOutcome.RETRYABLE_FAILURE,
                error_code=error.error_code,
                error_summary=error.error_summary,
                retryable=True,
            )
            return result, _map_failure_outcome(
                transition_result=result,
                attempt_outcome=AgentRunAttemptOutcome.RETRYABLE_FAILURE,
                error_code=error.error_code,
                disposition=disposition,
            )
        except Exception:
            disposition = _failure_disposition(
                claim=claim,
                retryable=True,
                retry_policy=self._retry_policy,
            )
            result = await self._persist_failure(
                claim=claim,
                finished_at=self._utc_now(),
                outcome=AgentRunAttemptOutcome.RETRYABLE_FAILURE,
                error_code="unexpected_executor_failure",
                error_summary=("The executor failed unexpectedly while processing the AgentRun."),
                retryable=True,
            )
            return result, _map_failure_outcome(
                transition_result=result,
                attempt_outcome=AgentRunAttemptOutcome.RETRYABLE_FAILURE,
                error_code="unexpected_executor_failure",
                disposition=disposition,
            )

        finished_at = self._utc_now()

        match execution_result:
            case CompletedExecution():
                async with self._transaction_manager.transaction():
                    result = await self._agent_run_repository.mark_succeeded(
                        CompleteAgentRunCommand(
                            agent_run_id=run.id,
                            lease_token=claim.attempt.lease_token,
                            finished_at=finished_at,
                        ),
                    )
                return result, _map_success_outcome(result)
            case PausedForApproval():
                async with self._transaction_manager.transaction():
                    result = await self._agent_run_repository.mark_waiting_for_approval(
                        WaitForApprovalAgentRunCommand(
                            agent_run_id=run.id,
                            lease_token=claim.attempt.lease_token,
                            finished_at=finished_at,
                        ),
                    )
                return result, _map_pause_outcome(result)
            case _:
                raise RuntimeError(
                    "AgentRun executor returned an unsupported execution result.",
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

        retry_allowed = retryable and self._retry_policy.can_retry_after_failure(
            retryable_failure_count=run.retryable_failure_count,
            max_retryable_failures=run.max_retryable_failures,
        )

        if retry_allowed:
            disposition = AgentRunFailureDisposition.RETRY_SCHEDULED
            delay_seconds = self._retry_policy.delay_seconds_for_failure(
                failure_number=run.retryable_failure_count + 1,
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


@dataclass(frozen=True, slots=True)
class _AttemptTelemetryOutcome:
    attempt_status: ObservationStatus
    attempt_outcome: str | None
    error_code: str | None
    trace_status: ObservationStatus
    agent_run_status: str | None
    latest_attempt_outcome: str | None


class _SafeAgentRunAttemptTelemetry:
    """Isolate AgentRun telemetry failures from business execution."""

    def __init__(
        self,
        *,
        client: ObservabilityClient,
        identity: TraceIdentity,
        run: AgentRun,
        attempt: AgentRunAttempt,
    ) -> None:
        self._client = client
        self._identity = identity
        self._run = run
        self._attempt = attempt
        self._trace_manager: AbstractContextManager[TraceScope] | None = None
        self._trace: TraceScope | None = None
        self._attempt_manager: AbstractContextManager[ObservationScope] | None = None
        self._attempt_scope: ObservationScope | None = None

    def start(self) -> None:
        try:
            self._trace_manager = self._client.start_trace(
                TraceAttributes(
                    trace_seed=self._identity.trace_seed,
                    name=self._identity.trace_name,
                    session_id=self._identity.session_id,
                    metadata=_agent_run_trace_metadata(self._run),
                    metadata_paths=_AGENT_RUN_TRACE_METADATA_PATHS,
                    tags=self._identity.tags,
                )
            )
            self._trace = self._trace_manager.__enter__()
        except Exception:
            self._trace_manager = None
            self._trace = None
            return

        try:
            self._attempt_manager = self._trace.start_observation(
                ObservationAttributes(
                    name="worker-attempt",
                    observation_type=ObservationType.SPAN,
                    metadata=_worker_attempt_metadata(
                        run=self._run,
                        attempt=self._attempt,
                    ),
                    metadata_paths=_WORKER_ATTEMPT_METADATA_PATHS,
                    input_paths=frozenset(),
                    output_paths=frozenset(),
                )
            )
            self._attempt_scope = self._attempt_manager.__enter__()
        except Exception:
            self._attempt_manager = None
            self._attempt_scope = None

    def complete(self, outcome: _AttemptTelemetryOutcome) -> None:
        attempt_metadata: dict[str, JsonValue] = {}
        if outcome.attempt_outcome is not None:
            attempt_metadata["attempt_outcome"] = outcome.attempt_outcome

        self._safe_attempt_update(
            ObservationUpdate(
                status=outcome.attempt_status,
                metadata=attempt_metadata,
                error_code=outcome.error_code,
            )
        )

        trace_metadata: dict[str, JsonValue] = {}
        if outcome.agent_run_status is not None:
            trace_metadata["agent_run_status"] = outcome.agent_run_status
        if outcome.latest_attempt_outcome is not None:
            trace_metadata["latest_attempt_outcome"] = outcome.latest_attempt_outcome

        self._safe_trace_update(
            ObservationUpdate(
                status=outcome.trace_status,
                metadata=trace_metadata,
                error_code=outcome.error_code,
            )
        )

    def close(self) -> None:
        if self._attempt_manager is not None:
            try:
                self._attempt_manager.__exit__(None, None, None)
            except Exception:
                pass
            finally:
                self._attempt_manager = None
                self._attempt_scope = None

        if self._trace_manager is not None:
            try:
                self._trace_manager.__exit__(None, None, None)
            except Exception:
                pass
            finally:
                self._trace_manager = None
                self._trace = None

    def _safe_attempt_update(self, update: ObservationUpdate) -> None:
        if self._attempt_scope is None:
            return

        try:
            self._attempt_scope.update(update)
        except Exception:
            return

    def _safe_trace_update(self, update: ObservationUpdate) -> None:
        if self._trace is None:
            return

        try:
            self._trace.update(update)
        except Exception:
            return


def _agent_run_trace_metadata(run: AgentRun) -> dict[str, JsonValue]:
    return {
        "agent_run_id": str(run.id),
        "workspace_id": str(run.workspace_id),
        "ticket_id": str(run.ticket_id),
        "workflow_name": run.workflow_name,
        "workflow_version": run.workflow_version,
        "trigger_key": run.trigger_key,
        "correlation_id": str(run.correlation_id),
    }


def _worker_attempt_metadata(
    *,
    run: AgentRun,
    attempt: AgentRunAttempt,
) -> dict[str, JsonValue]:
    return {
        "agent_run_id": str(run.id),
        "agent_run_attempt_id": str(attempt.id),
        "attempt_number": attempt.attempt_number,
        "execution_request_id": str(attempt.execution_request_id),
        "workspace_id": str(run.workspace_id),
        "ticket_id": str(run.ticket_id),
        "workflow_name": run.workflow_name,
        "workflow_version": run.workflow_version,
        "trigger_key": run.trigger_key,
        "correlation_id": str(run.correlation_id),
        "worker_id": attempt.worker_id,
    }


def _map_success_outcome(
    transition_result: AgentRunTransitionResult,
) -> _AttemptTelemetryOutcome:
    if transition_result is AgentRunTransitionResult.LEASE_LOST:
        return _lease_lost_outcome()

    return _AttemptTelemetryOutcome(
        attempt_status=ObservationStatus.OK,
        attempt_outcome=AgentRunAttemptOutcome.SUCCEEDED.value,
        error_code=None,
        trace_status=ObservationStatus.OK,
        agent_run_status=AgentRunStatus.SUCCEEDED.value,
        latest_attempt_outcome=None,
    )


def _map_pause_outcome(
    transition_result: AgentRunTransitionResult,
) -> _AttemptTelemetryOutcome:
    if transition_result is AgentRunTransitionResult.LEASE_LOST:
        return _lease_lost_outcome()

    return _AttemptTelemetryOutcome(
        attempt_status=ObservationStatus.OK,
        attempt_outcome=AgentRunAttemptOutcome.AWAITING_APPROVAL.value,
        error_code=None,
        trace_status=ObservationStatus.OK,
        agent_run_status=AgentRunStatus.WAITING_FOR_APPROVAL.value,
        latest_attempt_outcome=None,
    )


def _map_failure_outcome(
    *,
    transition_result: AgentRunTransitionResult,
    attempt_outcome: AgentRunAttemptOutcome,
    error_code: str,
    disposition: AgentRunFailureDisposition,
) -> _AttemptTelemetryOutcome:
    if transition_result is AgentRunTransitionResult.LEASE_LOST:
        return _lease_lost_outcome()

    if disposition is AgentRunFailureDisposition.RETRY_SCHEDULED:
        return _AttemptTelemetryOutcome(
            attempt_status=ObservationStatus.ERROR,
            attempt_outcome=attempt_outcome.value,
            error_code=error_code,
            trace_status=ObservationStatus.ERROR,
            agent_run_status=AgentRunStatus.RETRY_SCHEDULED.value,
            latest_attempt_outcome=attempt_outcome.value,
        )

    return _AttemptTelemetryOutcome(
        attempt_status=ObservationStatus.ERROR,
        attempt_outcome=attempt_outcome.value,
        error_code=error_code,
        trace_status=ObservationStatus.ERROR,
        agent_run_status=AgentRunStatus.FAILED.value,
        latest_attempt_outcome=None,
    )


def _lease_lost_outcome() -> _AttemptTelemetryOutcome:
    return _AttemptTelemetryOutcome(
        attempt_status=ObservationStatus.ERROR,
        attempt_outcome="lease_lost",
        error_code="lease_lost",
        trace_status=ObservationStatus.ERROR,
        agent_run_status=None,
        latest_attempt_outcome="lease_lost",
    )


def _failure_disposition(
    *,
    claim: AgentRunClaim,
    retryable: bool,
    retry_policy: AgentRunRetryPolicy,
) -> AgentRunFailureDisposition:
    run = claim.agent_run
    retry_allowed = retryable and retry_policy.can_retry_after_failure(
        retryable_failure_count=run.retryable_failure_count,
        max_retryable_failures=run.max_retryable_failures,
    )

    if retry_allowed:
        return AgentRunFailureDisposition.RETRY_SCHEDULED

    return AgentRunFailureDisposition.FAILED


def _utc_now() -> datetime:
    return datetime.now(UTC)
