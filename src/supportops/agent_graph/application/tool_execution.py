"""Recoverable fenced execution for validated tool decisions."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from supportops.agent_graph.application.tool_transitions import (
    ToolStateTransitionConflictError,
    record_persisted_tool_call,
)
from supportops.agent_graph.domain.routing import (
    CONTROLLED_SUPPORT_RUNTIME_LIMITS,
    ControlledSupportRuntimeLimits,
    reserve_next_tool_call,
)
from supportops.agent_graph.domain.state import (
    ControlledSupportGraphStateSnapshot,
)
from supportops.agent_tools.application.execution import (
    ExecuteToolCommand,
    ToolExecutionContext,
    ToolExecutionResult,
)
from supportops.agent_tools.application.persistence import (
    AgentToolCallExecutionRepository,
    AgentToolCallPersistenceResult,
    PersistAgentToolCallCommand,
)
from supportops.agent_tools.application.queries import (
    AgentToolCallLookup,
    AgentToolCallQueryRepository,
)
from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.ai.gateway.tool_decisions import (
    LLMExecutableToolCallDecision,
)
from supportops.core.transactions import TransactionManager
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    RetryableAgentRunExecutionError,
)

UtcNowProvider = Callable[[], datetime]


class ReadOnlyToolExecutor(Protocol):
    """Execute one bounded read-only tool command."""

    async def execute(
        self,
        command: ExecuteToolCommand,
    ) -> ToolExecutionResult:
        """Return one terminal audit and validated output."""

        ...


@dataclass(frozen=True, slots=True)
class ControlledToolExecutionOutcome:
    """State and audit produced by execution or crash recovery."""

    state: ControlledSupportGraphStateSnapshot
    audit: AgentToolCall
    recovered: bool


class ControlledToolDecisionExecutor:
    """Execute or recover one exact controlled tool decision."""

    def __init__(
        self,
        *,
        executor: ReadOnlyToolExecutor,
        transaction_manager: TransactionManager,
        execution_repository: (AgentToolCallExecutionRepository),
        query_repository: AgentToolCallQueryRepository,
        limits: ControlledSupportRuntimeLimits = (CONTROLLED_SUPPORT_RUNTIME_LIMITS),
        utc_now: UtcNowProvider | None = None,
    ) -> None:
        self._executor = executor
        self._transaction_manager = transaction_manager
        self._execution_repository = execution_repository
        self._query_repository = query_repository
        self._limits = limits
        self._utc_now = utc_now or _utc_now

    async def recover_next_persisted_outcome(
        self,
        *,
        state: ControlledSupportGraphStateSnapshot,
        context: AgentRunExecutionContext,
    ) -> ControlledToolExecutionOutcome | None:
        """Recover the next audit after a pre-checkpoint crash."""

        _validate_context_ownership(
            state=state,
            context=context,
        )
        sequence = state.tool_call_count + 1
        audit = await self._load_audit(
            state=state,
            context=context,
            sequence=sequence,
        )

        if audit is None:
            return None

        recovered_state = record_persisted_tool_call(
            state,
            audit,
            expected_attempt_id=context.attempt.id,
            limits=self._limits,
        )

        return ControlledToolExecutionOutcome(
            state=recovered_state,
            audit=audit,
            recovered=True,
        )

    async def execute(
        self,
        *,
        state: ControlledSupportGraphStateSnapshot,
        context: AgentRunExecutionContext,
        decision: LLMExecutableToolCallDecision,
    ) -> ControlledToolExecutionOutcome:
        """Execute or reuse one terminal audit for a tool decision."""

        _validate_context_ownership(
            state=state,
            context=context,
        )
        sequence = reserve_next_tool_call(
            state,
            limits=self._limits,
        )
        existing_audit = await self._load_audit(
            state=state,
            context=context,
            sequence=sequence,
        )

        if existing_audit is not None:
            _validate_audit_matches_decision(
                audit=existing_audit,
                decision=decision,
            )

            recovered_state = record_persisted_tool_call(
                state,
                existing_audit,
                expected_attempt_id=context.attempt.id,
                limits=self._limits,
            )

            return ControlledToolExecutionOutcome(
                state=recovered_state,
                audit=existing_audit,
                recovered=True,
            )

        execution_result = await self._executor.execute(
            ExecuteToolCommand(
                context=ToolExecutionContext(
                    workspace_id=(context.agent_run.workspace_id),
                    ticket_id=context.ticket.id,
                    agent_run_id=context.agent_run.id,
                    agent_run_attempt_id=context.attempt.id,
                ),
                sequence=sequence,
                provider_tool_call_id=(decision.provider_tool_call_id),
                tool_name=decision.tool_name,
                tool_version=decision.tool_version,
                arguments=decision.arguments,
                prior_fingerprints=frozenset(state.seen_tool_call_fingerprints),
            )
        )
        persisted_at = self._utc_now()
        persistence_result = await self._persist_audit(
            PersistAgentToolCallCommand(
                workspace_id=(context.agent_run.workspace_id),
                ticket_id=context.ticket.id,
                agent_run_id=context.agent_run.id,
                agent_run_attempt_id=context.attempt.id,
                lease_token=context.attempt.lease_token,
                persisted_at=persisted_at,
                tool_call=execution_result.audit,
            )
        )

        if persistence_result is (AgentToolCallPersistenceResult.LEASE_LOST):
            raise RetryableAgentRunExecutionError(
                error_code="tool_call_lease_lost",
                error_summary=(
                    "The AgentRun lease was lost before the tool-call audit could be persisted."
                ),
            )

        if persistence_result not in {
            AgentToolCallPersistenceResult.APPLIED,
            AgentToolCallPersistenceResult.ALREADY_RECORDED,
        }:
            raise RuntimeError("Tool-call persistence returned an invalid result.")

        updated_state = record_persisted_tool_call(
            state,
            execution_result.audit,
            expected_attempt_id=context.attempt.id,
            limits=self._limits,
        )

        return ControlledToolExecutionOutcome(
            state=updated_state,
            audit=execution_result.audit,
            recovered=False,
        )

    async def _load_audit(
        self,
        *,
        state: ControlledSupportGraphStateSnapshot,
        context: AgentRunExecutionContext,
        sequence: int,
    ) -> AgentToolCall | None:
        query = AgentToolCallLookup(
            workspace_id=state.workspace_id,
            ticket_id=state.ticket_id,
            agent_run_id=state.agent_run_id,
            agent_run_attempt_id=context.attempt.id,
            sequence=sequence,
        )

        async with self._transaction_manager.transaction():
            return await self._query_repository.get_by_attempt_sequence(query)

    async def _persist_audit(
        self,
        command: PersistAgentToolCallCommand,
    ) -> AgentToolCallPersistenceResult:
        async with self._transaction_manager.transaction():
            return await self._execution_repository.persist_fenced(command)


def _validate_context_ownership(
    *,
    state: ControlledSupportGraphStateSnapshot,
    context: AgentRunExecutionContext,
) -> None:
    ownership_values = (
        (
            context.agent_run.workspace_id,
            state.workspace_id,
            "workspace",
        ),
        (
            context.ticket.id,
            state.ticket_id,
            "ticket",
        ),
        (
            context.agent_run.id,
            state.agent_run_id,
            "AgentRun",
        ),
    )

    for actual, expected, resource_name in ownership_values:
        if actual != expected:
            raise ValueError(
                f"Graph state {resource_name} ownership does "
                "not match the AgentRun execution context."
            )


def _validate_audit_matches_decision(
    *,
    audit: AgentToolCall,
    decision: LLMExecutableToolCallDecision,
) -> None:
    if audit.provider_tool_call_id != decision.provider_tool_call_id:
        raise ToolStateTransitionConflictError(
            "The recovered audit provider call does not match the validated tool decision."
        )

    if audit.tool_name != decision.tool_name:
        raise ToolStateTransitionConflictError(
            "The recovered audit tool name does not match the validated tool decision."
        )

    if audit.tool_version != decision.tool_version:
        raise ToolStateTransitionConflictError(
            "The recovered audit tool version does not match the validated tool decision."
        )


def _utc_now() -> datetime:
    return datetime.now(UTC)
