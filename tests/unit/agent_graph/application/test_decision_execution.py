"""Unit tests for durable controlled support decisions."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from supportops.agent_graph.application.decision_execution import (
    ControlledSupportDecisionExecutor,
)
from supportops.agent_graph.application.transitions import (
    attach_classification,
)
from supportops.agent_graph.domain.completion import (
    CompleteSupportAnalysisInput,
)
from supportops.agent_graph.domain.state import (
    ControlledSupportGraphStateSnapshot,
    create_initial_controlled_support_state,
    validate_controlled_support_state,
)
from supportops.agent_tools.tools.registry import (
    create_controlled_support_tool_registry,
)
from supportops.agent_tools.tools.search_knowledge import (
    SearchKnowledgeInput,
)
from supportops.agent_tools.tools.service_status import (
    DeterministicServiceStatusCatalog,
)
from supportops.ai.gateway.errors import LLMTimeoutError
from supportops.ai.gateway.results import (
    LLMGatewayFailure,
    LLMInvocationStatus,
    LLMInvocationTrace,
)
from supportops.ai.gateway.tool_decisions import (
    LLMExecutableToolCallDecision,
    LLMTerminalControlDecision,
    LLMToolDecisionGatewayResult,
    LLMToolDecisionRequest,
)
from supportops.ai.pricing.catalog import (
    PRICING_CATALOG_VERSION,
)
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketCategory,
    TicketIntent,
    TicketSentiment,
    TicketUrgency,
)
from supportops.knowledge_retrieval.contracts import (
    KnowledgeSearchRequest,
    KnowledgeSearchResult,
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
from supportops.modules.support_recommendations.application.invocation_queries import (
    AttemptLLMInvocationQuery,
)
from supportops.modules.support_recommendations.application.persistence import (
    PersistSupportRecommendationCommand,
    SupportRecommendationPersistenceResult,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendationAction,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
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
_DECISION_INVOCATION_ID = UUID("90000000-0000-4000-8000-000000000009")

_NOW = datetime(
    2026,
    8,
    2,
    18,
    0,
    tzinfo=UTC,
)
_PERSISTED_AT = _NOW + timedelta(seconds=1)


class RecordingTransactionManager:
    """Expose and record application transaction boundaries."""

    def __init__(self) -> None:
        self.active = False
        self.enter_count = 0
        self.exit_count = 0

    @asynccontextmanager
    async def transaction(
        self,
    ) -> AsyncIterator[None]:
        assert self.active is False

        self.active = True
        self.enter_count += 1

        try:
            yield
        finally:
            self.active = False
            self.exit_count += 1


class StubInvocationQueryRepository:
    """Return deterministic existing invocation history."""

    def __init__(
        self,
        *,
        transaction_manager: RecordingTransactionManager,
        invocations: tuple[LLMInvocation, ...],
    ) -> None:
        self._transaction_manager = transaction_manager
        self.invocations = invocations
        self.queries: list[AttemptLLMInvocationQuery] = []

    async def list_by_attempt(
        self,
        query: AttemptLLMInvocationQuery,
    ) -> tuple[LLMInvocation, ...]:
        assert self._transaction_manager.active is True
        self.queries.append(query)

        return self.invocations


class StubExecutionRepository:
    """Record fenced invocation persistence commands."""

    def __init__(
        self,
        *,
        transaction_manager: RecordingTransactionManager,
        result: SupportRecommendationPersistenceResult = (
            SupportRecommendationPersistenceResult.APPLIED
        ),
    ) -> None:
        self._transaction_manager = transaction_manager
        self.result = result
        self.commands: list[PersistSupportRecommendationCommand] = []

    async def persist_fenced(
        self,
        command: PersistSupportRecommendationCommand,
    ) -> SupportRecommendationPersistenceResult:
        assert self._transaction_manager.active is True
        self.commands.append(command)

        return self.result


class StubDecisionGateway:
    """Return or raise one configured decision result."""

    def __init__(
        self,
        result: (LLMToolDecisionGatewayResult | LLMGatewayFailure),
        *,
        transaction_manager: RecordingTransactionManager,
    ) -> None:
        self._result = result
        self._transaction_manager = transaction_manager
        self.requests: list[LLMToolDecisionRequest] = []

    async def decide(
        self,
        request: LLMToolDecisionRequest,
    ) -> LLMToolDecisionGatewayResult:
        assert self._transaction_manager.active is False
        self.requests.append(request)

        if isinstance(
            self._result,
            LLMGatewayFailure,
        ):
            raise self._result

        return self._result


class EmptyKnowledgeSearch:
    """Return deterministic empty authoritative evidence."""

    async def execute(
        self,
        request: KnowledgeSearchRequest,
    ) -> KnowledgeSearchResult:
        return KnowledgeSearchResult(
            request=request,
            searched_version_count=0,
            evidence=(),
        )


def _context() -> AgentRunExecutionContext:
    ticket = Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Unable to reset account access",
        description=("The customer cannot complete the documented account recovery procedure."),
        external_reference=None,
        ingestion_request_id=UUID("a0000000-0000-4000-8000-000000000010"),
        correlation_id=UUID("b0000000-0000-4000-8000-000000000011"),
        now=_NOW - timedelta(minutes=1),
    )
    initial_run = AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=(ticket.ingestion_request_id),
        correlation_id=ticket.correlation_id,
        workflow_version="controlled-support-v1",
        max_attempts=3,
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
        summary=("The customer needs documented account recovery guidance."),
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash="a" * 64,
        provider="mock",
        model="mock-support-model-v1",
        now=_NOW - timedelta(seconds=30),
    )


def _classified_state() -> ControlledSupportGraphStateSnapshot:
    state = validate_controlled_support_state(
        create_initial_controlled_support_state(
            workspace_id=_WORKSPACE_ID,
            ticket_id=_TICKET_ID,
            agent_run_id=_AGENT_RUN_ID,
        )
    )

    return attach_classification(
        state,
        _classification(),
    )


def _existing_classification_invocation() -> LLMInvocation:
    return LLMInvocation.create(
        invocation_id=_CLASSIFICATION_INVOCATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        invocation_sequence=1,
        status=LLMInvocationStatus.SUCCEEDED,
        provider="mock",
        model="mock-support-model-v1",
        provider_request_id="classification-request",
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash="a" * 64,
        schema_version=(TICKET_CLASSIFICATION_SCHEMA_VERSION),
        input_tokens=20,
        cached_input_tokens=0,
        output_tokens=10,
        reasoning_tokens=0,
        total_tokens=30,
        pricing_catalog_version=(PRICING_CATALOG_VERSION),
        pricing_found=False,
        estimated_input_cost_usd=None,
        estimated_cached_input_cost_usd=None,
        estimated_output_cost_usd=None,
        estimated_total_cost_usd=None,
        latency_ms=25,
        error_code=None,
        now=_NOW - timedelta(seconds=30),
    )


def _success_trace() -> LLMInvocationTrace:
    return LLMInvocationTrace(
        invocation_sequence=1,
        status=LLMInvocationStatus.SUCCEEDED,
        provider="mock",
        model="mock-support-model-v1",
        provider_request_id="decision-request",
        usage=None,
        latency_ms=25,
        error_code=None,
    )


def _tool_result() -> LLMToolDecisionGatewayResult:
    return LLMToolDecisionGatewayResult(
        decision=LLMExecutableToolCallDecision(
            provider_tool_call_id="provider-tool-call-1",
            tool_name="search_knowledge",
            tool_version=1,
            arguments=SearchKnowledgeInput(
                query="account access reset",
                top_k=5,
                document_ids=None,
            ),
        ),
        invocations=(_success_trace(),),
        accepted_invocation_sequence=1,
    )


def _terminal_result() -> LLMToolDecisionGatewayResult:
    return LLMToolDecisionGatewayResult(
        decision=LLMTerminalControlDecision(
            provider_tool_call_id="provider-terminal-1",
            control_name="complete_support_analysis",
            control_version=1,
            output=CompleteSupportAnalysisInput(
                recommended_action=(SupportRecommendationAction.RESPOND),
                evidence_sufficient=True,
                requires_human_review=False,
                decision_summary=("Available runbook evidence supports a direct response."),
            ),
        ),
        invocations=(_success_trace(),),
        accepted_invocation_sequence=1,
    )


def _gateway_failure() -> LLMGatewayFailure:
    error = LLMTimeoutError(provider_request_id="decision-timeout")
    trace = LLMInvocationTrace(
        invocation_sequence=1,
        status=LLMInvocationStatus.TIMED_OUT,
        provider="mock",
        model="mock-support-model-v1",
        provider_request_id="decision-timeout",
        usage=None,
        latency_ms=20_000,
        error_code=error.error_code,
    )

    return LLMGatewayFailure(
        error=error,
        invocations=(trace,),
    )


def _service(
    result: (LLMToolDecisionGatewayResult | LLMGatewayFailure),
    *,
    persistence_result: (
        SupportRecommendationPersistenceResult
    ) = SupportRecommendationPersistenceResult.APPLIED,
) -> tuple[
    ControlledSupportDecisionExecutor,
    StubDecisionGateway,
    StubExecutionRepository,
    StubInvocationQueryRepository,
    RecordingTransactionManager,
]:
    transaction_manager = RecordingTransactionManager()
    query_repository = StubInvocationQueryRepository(
        transaction_manager=transaction_manager,
        invocations=(_existing_classification_invocation(),),
    )
    execution_repository = StubExecutionRepository(
        transaction_manager=transaction_manager,
        result=persistence_result,
    )
    gateway = StubDecisionGateway(
        result,
        transaction_manager=transaction_manager,
    )
    registry = create_controlled_support_tool_registry(
        knowledge_search=EmptyKnowledgeSearch(),
        service_status_catalog=(DeterministicServiceStatusCatalog(())),
    )
    service = ControlledSupportDecisionExecutor(
        gateway=gateway,
        tool_registry=registry,
        model="mock-support-model-v1",
        request_timeout_seconds=20,
        transaction_manager=transaction_manager,
        invocation_query_repository=query_repository,
        execution_repository=execution_repository,
        utc_now=lambda: _PERSISTED_AT,
        uuid_factory=lambda: _DECISION_INVOCATION_ID,
    )

    return (
        service,
        gateway,
        execution_repository,
        query_repository,
        transaction_manager,
    )


async def test_executes_tool_decision_outside_transaction() -> None:
    (
        service,
        gateway,
        repository,
        query_repository,
        transaction_manager,
    ) = _service(_tool_result())

    outcome = await service.execute(
        state=_classified_state(),
        context=_context(),
        tool_observations=(),
    )

    assert isinstance(
        outcome.decision,
        LLMExecutableToolCallDecision,
    )
    assert outcome.state.decision_turn_count == 1
    assert outcome.state.tool_call_count == 0
    assert outcome.state.analysis_completion is None
    assert outcome.accepted_invocation_id == (_DECISION_INVOCATION_ID)

    assert len(gateway.requests) == 1
    assert len(query_repository.queries) == 1
    assert len(repository.commands) == 1
    assert transaction_manager.enter_count == 2
    assert transaction_manager.exit_count == 2

    request = gateway.requests[0]

    assert [definition.name for definition in request.tools] == [
        "lookup_service_status",
        "search_knowledge",
    ]
    assert request.metadata["supportops_decision_turn"] == "1"

    command = repository.commands[0]

    assert [invocation.invocation_sequence for invocation in command.invocations] == [
        1,
        2,
    ]
    assert command.invocations[-1].id == (_DECISION_INVOCATION_ID)
    assert command.recommendation is None
    assert command.citations == ()


async def test_terminal_decision_updates_graph_state() -> None:
    service, _, repository, _, _ = _service(_terminal_result())

    outcome = await service.execute(
        state=_classified_state(),
        context=_context(),
        tool_observations=(),
    )

    assert isinstance(
        outcome.decision,
        LLMTerminalControlDecision,
    )
    assert outcome.state.decision_turn_count == 1
    assert outcome.state.analysis_completion is not None
    assert outcome.state.analysis_completion.recommended_action == "respond"
    assert outcome.state.analysis_completion.evidence_sufficient is True
    assert len(repository.commands) == 1


async def test_gateway_failure_is_persisted_before_retry_error() -> None:
    service, _, repository, _, _ = _service(_gateway_failure())

    with pytest.raises(
        RetryableAgentRunExecutionError,
    ) as captured:
        await service.execute(
            state=_classified_state(),
            context=_context(),
            tool_observations=(),
        )

    assert captured.value.error_code == "llm_timeout"
    assert len(repository.commands) == 1

    persisted = repository.commands[0]

    assert len(persisted.invocations) == 2
    assert persisted.invocations[-1].status is (LLMInvocationStatus.TIMED_OUT)
    assert persisted.recommendation is None


async def test_lease_loss_prevents_decision_state_return() -> None:
    service, _, _, _, _ = _service(
        _tool_result(),
        persistence_result=(SupportRecommendationPersistenceResult.LEASE_LOST),
    )

    with pytest.raises(
        RetryableAgentRunExecutionError,
    ) as captured:
        await service.execute(
            state=_classified_state(),
            context=_context(),
            tool_observations=(),
        )

    assert captured.value.error_code == ("support_decision_lease_lost")


async def test_context_ownership_must_match_state() -> None:
    service, gateway, _, _, _ = _service(_tool_result())
    mismatched_state = _classified_state().model_copy(
        update={
            "ticket_id": UUID("c0000000-0000-4000-8000-000000000012"),
        },
    )

    with pytest.raises(
        ValueError,
        match="ticket ownership",
    ):
        await service.execute(
            state=mismatched_state,
            context=_context(),
            tool_observations=(),
        )

    assert gateway.requests == []


async def test_rejects_non_contiguous_existing_history() -> None:
    (
        service,
        gateway,
        _,
        query_repository,
        _,
    ) = _service(_tool_result())
    query_repository.invocations = (
        replace(
            _existing_classification_invocation(),
            invocation_sequence=2,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="not contiguous and ordered",
    ):
        await service.execute(
            state=_classified_state(),
            context=_context(),
            tool_observations=(),
        )

    assert gateway.requests == []
