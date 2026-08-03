"""Unit tests for recoverable fenced tool execution."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from supportops.agent_graph.application.tool_execution import (
    ControlledToolDecisionExecutor,
)
from supportops.agent_graph.application.transitions import (
    attach_classification,
    reserve_decision_turn,
)
from supportops.agent_graph.domain.state import (
    ControlledSupportGraphStateSnapshot,
    create_initial_controlled_support_state,
    validate_controlled_support_state,
)
from supportops.agent_tools.application.execution import (
    ExecuteToolCommand,
    ToolExecutionResult,
)
from supportops.agent_tools.application.persistence import (
    AgentToolCallPersistenceResult,
    PersistAgentToolCallCommand,
)
from supportops.agent_tools.application.queries import (
    AgentToolCallLookup,
)
from supportops.agent_tools.domain.audit import (
    AgentToolCall,
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import (
    StrictToolSchema,
    ToolFailurePolicy,
    ToolSafetyLevel,
)
from supportops.agent_tools.tools.search_knowledge import (
    SearchKnowledgeEvidence,
    SearchKnowledgeInput,
    SearchKnowledgeOutput,
)
from supportops.ai.gateway.tool_decisions import (
    LLMExecutableToolCallDecision,
)
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    RetryableAgentRunExecutionError,
)
from supportops.modules.agent_runs.domain.models import (
    INITIAL_TICKET_PROCESSING_TRIGGER_KEY,
    INITIAL_TICKET_PROCESSING_WORKFLOW_NAME,
    AgentRun,
    AgentRunAttempt,
    AgentRunStatus,
)
from supportops.modules.ticket_classifications.domain.models import (
    TicketClassification,
)
from supportops.modules.tickets.domain.models import Ticket

_WORKSPACE_ID = UUID("10000000-0000-4000-8000-000000000001")
_TICKET_ID = UUID("20000000-0000-4000-8000-000000000002")
_AGENT_RUN_ID = UUID("30000000-0000-4000-8000-000000000003")
_ATTEMPT_ID = UUID("40000000-0000-4000-8000-000000000004")
_LEASE_TOKEN = UUID("50000000-0000-4000-8000-000000000005")
_EXECUTION_REQUEST_ID = UUID("60000000-0000-4000-8000-000000000006")
_CLASSIFICATION_ID = UUID("70000000-0000-4000-8000-000000000007")
_CLASSIFICATION_INVOCATION_ID = UUID("80000000-0000-4000-8000-000000000008")
_TOOL_CALL_ID = UUID("90000000-0000-4000-8000-000000000009")
_RETRIEVAL_QUERY_ID = UUID("a0000000-0000-4000-8000-000000000010")
_CHUNK_ID = UUID("b0000000-0000-4000-8000-000000000011")

_NOW = datetime(
    2026,
    8,
    2,
    18,
    0,
    tzinfo=UTC,
)
_FINISHED_AT = _NOW + timedelta(milliseconds=25)
_PERSISTED_AT = _NOW + timedelta(seconds=1)


class RecordingTransactionManager:
    """Record transaction boundaries and expose active state."""

    def __init__(self) -> None:
        self.active = False
        self.enter_count = 0
        self.exit_count = 0

    @asynccontextmanager
    async def transaction(
        self,
    ) -> AsyncIterator[None]:
        if self.active:
            raise AssertionError("Nested transactions are not expected.")

        self.active = True
        self.enter_count += 1

        try:
            yield
        finally:
            self.active = False
            self.exit_count += 1


class RecordingQueryRepository:
    """Return one configured terminal audit."""

    def __init__(
        self,
        *,
        transaction_manager: RecordingTransactionManager,
        audit: AgentToolCall | None = None,
    ) -> None:
        self._transaction_manager = transaction_manager
        self.audit = audit
        self.queries: list[AgentToolCallLookup] = []

    async def get_by_proposal_attempt_sequence(
        self,
        query: AgentToolCallLookup,
    ) -> AgentToolCall | None:
        assert self._transaction_manager.active is True
        self.queries.append(query)

        return self.audit

    async def get_sensitive_by_identity(
        self,
        query: object,
    ) -> AgentToolCall | None:
        del query
        return None


class RecordingExecutionRepository:
    """Record fenced audit persistence."""

    def __init__(
        self,
        *,
        transaction_manager: RecordingTransactionManager,
        result: AgentToolCallPersistenceResult = (AgentToolCallPersistenceResult.APPLIED),
    ) -> None:
        self._transaction_manager = transaction_manager
        self.result = result
        self.commands: list[PersistAgentToolCallCommand] = []

    async def persist_fenced(
        self,
        command: PersistAgentToolCallCommand,
    ) -> AgentToolCallPersistenceResult:
        assert self._transaction_manager.active is True
        self.commands.append(command)

        return self.result


class RecordingToolExecutor:
    """Return one configured result outside transactions."""

    def __init__(
        self,
        *,
        transaction_manager: RecordingTransactionManager,
        result: ToolExecutionResult,
    ) -> None:
        self._transaction_manager = transaction_manager
        self.result = result
        self.commands: list[ExecuteToolCommand] = []

    async def execute(
        self,
        command: ExecuteToolCommand,
    ) -> ToolExecutionResult:
        assert self._transaction_manager.active is False
        self.commands.append(command)

        return self.result


def _context() -> AgentRunExecutionContext:
    ticket = Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Unable to reset account access",
        description=("The customer cannot complete the documented account recovery procedure."),
        external_reference=None,
        ingestion_request_id=UUID("c0000000-0000-4000-8000-000000000012"),
        correlation_id=UUID("d0000000-0000-4000-8000-000000000013"),
        now=_NOW - timedelta(minutes=1),
    )
    initial_run = AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=(ticket.ingestion_request_id),
        correlation_id=ticket.correlation_id,
        workflow_version="controlled-support-v1",
        max_retryable_failures=3,
        now=_NOW - timedelta(minutes=1),
    )
    running_run = replace(
        initial_run,
        workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
        workflow_version="controlled-support-v1",
        trigger_key=(INITIAL_TICKET_PROCESSING_TRIGGER_KEY),
        status=AgentRunStatus.RUNNING,
        attempt_count=1,
        lease_owner="worker-a",
        lease_token=_LEASE_TOKEN,
        lease_expires_at=_NOW + timedelta(minutes=5),
        first_started_at=_NOW,
        updated_at=_NOW,
    )
    attempt = AgentRunAttempt.start(
        attempt_id=_ATTEMPT_ID,
        agent_run_id=_AGENT_RUN_ID,
        attempt_number=1,
        worker_id="worker-a",
        lease_token=_LEASE_TOKEN,
        execution_request_id=_EXECUTION_REQUEST_ID,
        now=_NOW,
    )

    return AgentRunExecutionContext(
        agent_run=running_run,
        attempt=attempt,
        ticket=ticket,
    )


def _classification() -> TicketClassification:
    return TicketClassification.create(
        classification_id=_CLASSIFICATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        accepted_llm_invocation_id=(_CLASSIFICATION_INVOCATION_ID),
        category=TicketCategory.ACCOUNT_ACCESS,
        intent=TicketIntent.REQUEST_ACCESS,
        urgency=TicketUrgency.NORMAL,
        sentiment=TicketSentiment.NEUTRAL,
        requires_human_review=False,
        summary=("The customer needs documented recovery guidance."),
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash="a" * 64,
        provider="mock",
        model="mock-model",
    )


def _state(
    *,
    reserve_decision: bool,
) -> ControlledSupportGraphStateSnapshot:
    state = validate_controlled_support_state(
        create_initial_controlled_support_state(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=_AGENT_RUN_ID,
        )
    )
    state = attach_classification(
        state,
        _classification(),
    )

    if reserve_decision:
        return reserve_decision_turn(state)

    return state


def _arguments() -> SearchKnowledgeInput:
    return SearchKnowledgeInput(
        query="account access reset",
        top_k=5,
        document_ids=None,
    )


def _decision() -> LLMExecutableToolCallDecision:
    return LLMExecutableToolCallDecision(
        provider_tool_call_id="provider-call-1",
        tool_name="search_knowledge",
        tool_version=1,
        arguments=_arguments(),
    )


def _audit(
    *,
    status: AgentToolCallStatus = (AgentToolCallStatus.SUCCEEDED),
    error_code: str | None = None,
) -> AgentToolCall:
    return AgentToolCall.create_terminal(
        tool_call_id=_TOOL_CALL_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        sequence=1,
        provider_tool_call_id="provider-call-1",
        tool_name="search_knowledge",
        tool_version=1,
        safety_level=ToolSafetyLevel.READ_ONLY,
        status=status,
        input_fingerprint="b" * 64,
        safe_input={
            "query_sha256": "c" * 64,
            "query_length": 20,
            "top_k": 5,
            "document_ids": None,
        },
        safe_output=(
            {
                "retrieval_query_id": str(_RETRIEVAL_QUERY_ID),
                "searched_version_count": 1,
                "result_count": 1,
                "evidence": [
                    {
                        "rank": 1,
                        "score": 0.91,
                        "document_id": str(UUID("e0000000-0000-4000-8000-000000000014")),
                        "document_version_id": str(UUID("f0000000-0000-4000-8000-000000000015")),
                        "chunk_id": str(_CHUNK_ID),
                        "chunk_ordinal": 0,
                        "content_sha256": "d" * 64,
                    }
                ],
            }
            if status is AgentToolCallStatus.SUCCEEDED
            else None
        ),
        latency_ms=25,
        error_code=error_code,
        started_at=_NOW,
        finished_at=_FINISHED_AT,
    )


def _execution_result(
    audit: AgentToolCall,
) -> ToolExecutionResult:
    output: StrictToolSchema | None

    if audit.status is AgentToolCallStatus.SUCCEEDED:
        output = SearchKnowledgeOutput(
            retrieval_query_id=_RETRIEVAL_QUERY_ID,
            searched_version_count=1,
            evidence=(
                SearchKnowledgeEvidence(
                    rank=1,
                    score=0.91,
                    content="Authoritative recovery guidance.",
                    content_sha256="d" * 64,
                    token_count=4,
                    document_id=UUID("e0000000-0000-4000-8000-000000000014"),
                    document_title="Account recovery",
                    document_external_reference=None,
                    document_version_id=UUID("f0000000-0000-4000-8000-000000000015"),
                    version_number=1,
                    chunk_id=_CHUNK_ID,
                    chunk_ordinal=0,
                    section_path=("Account recovery",),
                    media_type="text/markdown",
                ),
            ),
        )
    else:
        output = None

    return ToolExecutionResult(
        audit=audit,
        output=output,
        retryable=(audit.status is not AgentToolCallStatus.SUCCEEDED),
        failure_policy=(ToolFailurePolicy.RETRY_AGENT_RUN),
    )


def _service(
    *,
    existing_audit: AgentToolCall | None = None,
    persistence_result: AgentToolCallPersistenceResult = (AgentToolCallPersistenceResult.APPLIED),
    produced_audit: AgentToolCall | None = None,
) -> tuple[
    ControlledToolDecisionExecutor,
    RecordingToolExecutor,
    RecordingExecutionRepository,
    RecordingQueryRepository,
    RecordingTransactionManager,
]:
    transaction_manager = RecordingTransactionManager()
    produced_audit = produced_audit or _audit()
    tool_executor = RecordingToolExecutor(
        transaction_manager=transaction_manager,
        result=_execution_result(produced_audit),
    )
    execution_repository = RecordingExecutionRepository(
        transaction_manager=transaction_manager,
        result=persistence_result,
    )
    query_repository = RecordingQueryRepository(
        transaction_manager=transaction_manager,
        audit=existing_audit,
    )
    service = ControlledToolDecisionExecutor(
        executor=tool_executor,
        transaction_manager=transaction_manager,
        execution_repository=execution_repository,
        query_repository=query_repository,
        utc_now=lambda: _PERSISTED_AT,
    )

    return (
        service,
        tool_executor,
        execution_repository,
        query_repository,
        transaction_manager,
    )


async def test_executes_outside_transaction_then_persists() -> None:
    (
        service,
        tool_executor,
        persistence_repository,
        query_repository,
        transaction_manager,
    ) = _service()

    outcome = await service.execute(
        state=_state(reserve_decision=True),
        context=_context(),
        decision=_decision(),
    )

    assert outcome.recovered is False
    assert outcome.audit == _audit()
    assert outcome.state.tool_call_count == 1
    assert len(tool_executor.commands) == 1
    assert len(persistence_repository.commands) == 1
    assert len(query_repository.queries) == 1
    assert transaction_manager.enter_count == 2
    assert transaction_manager.exit_count == 2

    command = persistence_repository.commands[0]

    assert command.lease_token == _LEASE_TOKEN
    assert command.persisted_at == _PERSISTED_AT
    assert command.tool_call == _audit()


async def test_existing_audit_skips_tool_execution() -> None:
    existing = _audit()
    (
        service,
        tool_executor,
        persistence_repository,
        _,
        transaction_manager,
    ) = _service(existing_audit=existing)

    outcome = await service.execute(
        state=_state(reserve_decision=True),
        context=_context(),
        decision=_decision(),
    )

    assert outcome.recovered is True
    assert outcome.audit == existing
    assert outcome.state.tool_call_count == 1
    assert tool_executor.commands == []
    assert persistence_repository.commands == []
    assert transaction_manager.enter_count == 1


async def test_recovers_post_commit_pre_checkpoint_audit() -> None:
    existing = _audit()
    service, tool_executor, _, _, _ = _service(existing_audit=existing)

    outcome = await service.recover_next_persisted_outcome(
        state=_state(reserve_decision=False),
        context=_context(),
    )

    assert outcome is not None
    assert outcome.recovered is True
    assert outcome.state.decision_turn_count == 1
    assert outcome.state.tool_call_count == 1
    assert tool_executor.commands == []


async def test_recovery_returns_none_when_no_audit_exists() -> None:
    service, tool_executor, _, _, _ = _service()

    outcome = await service.recover_next_persisted_outcome(
        state=_state(reserve_decision=False),
        context=_context(),
    )

    assert outcome is None
    assert tool_executor.commands == []


async def test_lease_loss_does_not_advance_state() -> None:
    service, _, _, _, _ = _service(persistence_result=(AgentToolCallPersistenceResult.LEASE_LOST))
    original_state = _state(reserve_decision=True)

    with pytest.raises(
        RetryableAgentRunExecutionError,
    ) as captured:
        await service.execute(
            state=original_state,
            context=_context(),
            decision=_decision(),
        )

    assert captured.value.error_code == ("tool_call_lease_lost")


async def test_persisted_failure_returns_checkpointable_error() -> None:
    audit = _audit(
        status=AgentToolCallStatus.TIMED_OUT,
        error_code="tool_timeout",
    )
    service, _, _, _, _ = _service(produced_audit=audit)

    outcome = await service.execute(
        state=_state(reserve_decision=True),
        context=_context(),
        decision=_decision(),
    )

    assert outcome.state.tool_call_count == 1
    assert outcome.state.current_error_code == ("tool_timeout")


async def test_recovered_audit_must_match_decision() -> None:
    existing = _audit()
    service, tool_executor, _, _, _ = _service(existing_audit=existing)
    mismatched_decision = LLMExecutableToolCallDecision(
        provider_tool_call_id="different-call",
        tool_name="search_knowledge",
        tool_version=1,
        arguments=_arguments(),
    )

    with pytest.raises(
        RuntimeError,
        match="provider call does not match",
    ):
        await service.execute(
            state=_state(reserve_decision=True),
            context=_context(),
            decision=mismatched_decision,
        )

    assert tool_executor.commands == []


async def test_context_ownership_must_match_state() -> None:
    mismatched_state = _state(reserve_decision=False).model_copy(
        update={
            "ticket_id": UUID("01000000-0000-4000-8000-000000000016"),
        },
    )
    service, _, _, _, _ = _service()

    with pytest.raises(
        ValueError,
        match="ticket ownership",
    ):
        await service.recover_next_persisted_outcome(
            state=mismatched_state,
            context=_context(),
        )
