"""Unit tests for grounded recommendation execution."""

from collections.abc import AsyncIterator
from contextlib import AbstractContextManager, asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from supportops.agent_graph.application.recommendation_execution import (
    ControlledSupportRecommendationExecutor,
)
from supportops.agent_graph.application.tool_observations import (
    ControlledToolObservationBundle,
)
from supportops.agent_graph.application.transitions import (
    attach_analysis_completion,
    attach_classification,
    reserve_decision_turn,
)
from supportops.agent_graph.domain.completion import (
    CompleteSupportAnalysisInput,
)
from supportops.agent_graph.domain.state import (
    ControlledSupportGraphStateSnapshot,
    create_initial_controlled_support_state,
    validate_controlled_support_state,
)
from supportops.ai.gateway.errors import LLMTimeoutError
from supportops.ai.gateway.results import (
    LLMGatewayFailure,
    LLMGatewayResult,
    LLMInvocationStatus,
    LLMInvocationTrace,
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
from supportops.modules.support_recommendations.application.persistence import (
    PersistSupportRecommendationCommand,
    SupportRecommendationPersistenceResult,
)
from supportops.modules.support_recommendations.application.schemas import (
    SupportRecommendationResult,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendation,
    SupportRecommendationAction,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
    TicketClassification,
)
from supportops.modules.tickets.domain.models import Ticket
from supportops.observability.context import (
    ActiveObservationContext,
    observation_context_scope,
)
from supportops.observability.contracts import TraceScope
from supportops.observability.models import (
    EventObservation,
    ObservabilityProvider,
    ObservationAttributes,
    ObservationStatus,
    TraceAttributes,
)

_WORKSPACE_ID = UUID("10000000-0000-4000-8000-000000000001")
_TICKET_ID = UUID("20000000-0000-4000-8000-000000000002")
_AGENT_RUN_ID = UUID("30000000-0000-4000-8000-000000000003")
_ATTEMPT_ID = UUID("40000000-0000-4000-8000-000000000004")
_LEASE_TOKEN = UUID("50000000-0000-4000-8000-000000000005")
_CLASSIFICATION_ID = UUID("60000000-0000-4000-8000-000000000006")
_CLASSIFICATION_INVOCATION_ID = UUID("70000000-0000-4000-8000-000000000007")
_RECOMMENDATION_INVOCATION_ID = UUID("80000000-0000-4000-8000-000000000008")
_RECOMMENDATION_ID = UUID("90000000-0000-4000-8000-000000000009")

_NOW = datetime(
    2026,
    8,
    2,
    18,
    0,
    tzinfo=UTC,
)


class RecordingTransactionManager:
    """Record transaction boundaries."""

    def __init__(self) -> None:
        self.active = False
        self.enter_count = 0

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


class EmptyObservationAssembler:
    """Return no tool observations."""

    async def assemble(
        self,
        *,
        state: object,
        context: object,
    ) -> ControlledToolObservationBundle:
        del state, context

        return ControlledToolObservationBundle(
            observations=(),
            citation_sources=(),
        )


class StubGateway:
    """Return or raise one configured Gateway result."""

    def __init__(
        self,
        result: LLMGatewayResult | LLMGatewayFailure,
        transaction_manager: RecordingTransactionManager,
    ) -> None:
        self._result = result
        self._transaction_manager = transaction_manager
        self.requests: list[object] = []

    async def generate(
        self,
        request: object,
    ) -> LLMGatewayResult:
        assert self._transaction_manager.active is False
        self.requests.append(request)

        if isinstance(
            self._result,
            LLMGatewayFailure,
        ):
            raise self._result

        return self._result


class StubInvocationRepository:
    """Return configured invocation history."""

    def __init__(
        self,
        transaction_manager: RecordingTransactionManager,
        invocations: tuple[LLMInvocation, ...],
    ) -> None:
        self._transaction_manager = transaction_manager
        self.invocations = invocations

    async def list_by_attempt(
        self,
        query: object,
    ) -> tuple[LLMInvocation, ...]:
        del query
        assert self._transaction_manager.active is True

        return self.invocations


class StubRecommendationQueryRepository:
    """Return an existing recommendation when configured."""

    def __init__(
        self,
        transaction_manager: RecordingTransactionManager,
        recommendation: SupportRecommendation | None = None,
    ) -> None:
        self._transaction_manager = transaction_manager
        self.recommendation = recommendation

    async def get_by_agent_run_id(
        self,
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> SupportRecommendation | None:
        assert self._transaction_manager.active is True
        assert workspace_id == _WORKSPACE_ID
        assert agent_run_id == _AGENT_RUN_ID

        return self.recommendation


class StubExecutionRepository:
    """Record fenced recommendation commands."""

    def __init__(
        self,
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


def _context() -> AgentRunExecutionContext:
    ticket = Ticket.create(
        ticket_id=_TICKET_ID,
        workspace_id=_WORKSPACE_ID,
        subject="Unable to reset account access",
        description=("The customer cannot complete the documented account recovery procedure."),
        external_reference=None,
        ingestion_request_id=UUID("a0000000-0000-4000-8000-000000000010"),
        correlation_id=UUID("b0000000-0000-4000-8000-000000000011"),
        now=_NOW,
    )
    initial_run = AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=(ticket.ingestion_request_id),
        correlation_id=ticket.correlation_id,
        workflow_version="controlled-support-v1",
        max_retryable_failures=3,
        now=_NOW,
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
        execution_request_id=UUID("c0000000-0000-4000-8000-000000000012"),
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


def _state() -> ControlledSupportGraphStateSnapshot:
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
    state = reserve_decision_turn(state)

    return attach_analysis_completion(
        state,
        CompleteSupportAnalysisInput(
            recommended_action=(SupportRecommendationAction.RESPOND),
            evidence_sufficient=True,
            requires_human_review=False,
            decision_summary=("The available evidence supports a direct response."),
        ),
    )


def _trace(
    *,
    status: LLMInvocationStatus = (LLMInvocationStatus.SUCCEEDED),
    error_code: object | None = None,
) -> LLMInvocationTrace:
    return LLMInvocationTrace(
        invocation_sequence=1,
        status=status,
        provider="mock",
        model="mock-model",
        provider_request_id="request-1",
        usage=None,
        latency_ms=25,
        error_code=error_code,  # type: ignore[arg-type]
    )


def _success_result() -> LLMGatewayResult:
    return LLMGatewayResult(
        output=SupportRecommendationResult(
            recommended_action=(SupportRecommendationAction.RESPOND),
            response_text=("Follow the documented account recovery procedure."),
            requires_human_review=False,
            decision_summary=("The evidence supports a direct response."),
            schema_version="support-recommendation-v1",
        ),
        invocations=(_trace(),),
        accepted_invocation_sequence=1,
    )


def _failure() -> LLMGatewayFailure:
    error = LLMTimeoutError(provider_request_id="request-timeout")

    return LLMGatewayFailure(
        error=error,
        invocations=(
            _trace(
                status=LLMInvocationStatus.TIMED_OUT,
                error_code=error.error_code,
            ),
        ),
    )


def _service(
    result: LLMGatewayResult | LLMGatewayFailure,
    *,
    persistence_result: (
        SupportRecommendationPersistenceResult
    ) = SupportRecommendationPersistenceResult.APPLIED,
    existing_recommendation: (SupportRecommendation | None) = None,
    observability_client: object | None = None,
) -> tuple[
    ControlledSupportRecommendationExecutor,
    StubGateway,
    StubExecutionRepository,
]:
    transaction_manager = RecordingTransactionManager()
    gateway = StubGateway(
        result,
        transaction_manager,
    )
    execution_repository = StubExecutionRepository(
        transaction_manager,
        persistence_result,
    )
    ids = iter(
        (
            _RECOMMENDATION_INVOCATION_ID,
            _RECOMMENDATION_ID,
        )
    )
    service = ControlledSupportRecommendationExecutor(
        gateway=gateway,  # type: ignore[arg-type]
        model="mock-model",
        request_timeout_seconds=20,
        transaction_manager=transaction_manager,
        observation_assembler=(
            EmptyObservationAssembler()  # type: ignore[arg-type]
        ),
        invocation_query_repository=(
            StubInvocationRepository(
                transaction_manager,
                (),
            )
        ),
        recommendation_query_repository=(
            StubRecommendationQueryRepository(
                transaction_manager,
                existing_recommendation,
            )
        ),
        execution_repository=execution_repository,
        utc_now=lambda: _NOW,
        uuid_factory=lambda: next(ids),
        observability_client=observability_client,  # type: ignore[arg-type]
    )

    return (
        service,
        gateway,
        execution_repository,
    )


class RecordingObservabilityClient:
    def __init__(self, *, fail_event: bool = False) -> None:
        self._fail_event = fail_event
        self.events: list[EventObservation] = []
        self.trace_events: list[tuple[object, EventObservation]] = []
        self.started_observations: list[ObservationAttributes] = []

    @property
    def provider(self) -> ObservabilityProvider:
        return ObservabilityProvider.NOOP

    @property
    def enabled(self) -> bool:
        return True

    def start_trace(
        self,
        attributes: TraceAttributes,
    ) -> AbstractContextManager[TraceScope]:
        del attributes
        raise AssertionError("Recommendation events must not open traces.")

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[object]:
        self.started_observations.append(attributes)
        raise AssertionError("Recommendation events must not open observations.")

    def record_event(self, event: EventObservation) -> None:
        if self._fail_event:
            raise RuntimeError("synthetic event failure")

        self.events.append(event)

    def record_trace_event(self, *, identity: object, event: EventObservation) -> None:
        if self._fail_event:
            raise RuntimeError("synthetic trace event failure")

        self.trace_events.append((identity, event))

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


def _recommendation_events(
    observability: RecordingObservabilityClient,
) -> list[EventObservation]:
    return [
        *observability.events,
        *(event for _, event in observability.trace_events),
    ]


async def test_drafts_persists_and_updates_state() -> None:
    service, gateway, repository = _service(_success_result())

    outcome = await service.execute(
        state=_state(),
        context=_context(),
    )

    assert outcome.recovered is False
    assert outcome.state.recommendation_id == (_RECOMMENDATION_ID)
    assert outcome.state.recommendation_invocation_id == (_RECOMMENDATION_INVOCATION_ID)
    assert len(gateway.requests) == 1
    assert len(repository.commands) == 1

    command = repository.commands[0]

    assert command.recommendation is not None
    assert command.recommendation.id == (_RECOMMENDATION_ID)
    assert len(command.invocations) == 1
    assert command.citations == ()


async def test_recovers_existing_without_provider_call() -> None:
    recommendation = SupportRecommendation.create(
        recommendation_id=_RECOMMENDATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        classification_id=_CLASSIFICATION_ID,
        accepted_llm_invocation_id=(_RECOMMENDATION_INVOCATION_ID),
        recommended_action=(SupportRecommendationAction.RESPOND),
        response_text=("Follow the documented recovery procedure."),
        requires_human_review=False,
        decision_summary=("The evidence supports a direct response."),
        prompt_id="support-recommendation-draft",
        prompt_version=1,
        prompt_content_hash="a" * 64,
        provider="mock",
        model="mock-model",
    )
    service, gateway, repository = _service(
        _success_result(),
        existing_recommendation=recommendation,
    )

    outcome = await service.execute(
        state=_state(),
        context=_context(),
    )

    assert outcome.recovered is True
    assert outcome.recommendation == recommendation
    assert gateway.requests == []
    assert repository.commands == []


async def test_gateway_failure_is_persisted() -> None:
    service, _, repository = _service(_failure())

    with pytest.raises(
        RetryableAgentRunExecutionError,
    ) as captured:
        await service.execute(
            state=_state(),
            context=_context(),
        )

    assert captured.value.error_code == "llm_timeout"
    assert len(repository.commands) == 1
    assert repository.commands[0].recommendation is None
    assert len(repository.commands[0].invocations) == 1


async def test_lease_loss_prevents_state_return() -> None:
    service, _, _ = _service(
        _success_result(),
        persistence_result=(SupportRecommendationPersistenceResult.LEASE_LOST),
    )

    with pytest.raises(
        RetryableAgentRunExecutionError,
    ) as captured:
        await service.execute(
            state=_state(),
            context=_context(),
        )

    assert captured.value.error_code == ("support_recommendation_lease_lost")


async def test_applied_recommendation_emits_generated_and_persisted_events() -> None:
    observability = RecordingObservabilityClient()
    order: list[str] = []
    service, gateway, repository = _service(
        _success_result(),
        observability_client=observability,
    )
    original_persist = repository.persist_fenced

    async def tracking_persist(
        command: PersistSupportRecommendationCommand,
    ) -> SupportRecommendationPersistenceResult:
        order.append("persisted")
        return await original_persist(command)

    repository.persist_fenced = tracking_persist  # type: ignore[method-assign]
    original_record = observability.record_event

    def tracking_record(event: EventObservation) -> None:
        order.append(f"event:{event.name}")
        original_record(event)

    observability.record_event = tracking_record  # type: ignore[method-assign]

    with observation_context_scope(
        ActiveObservationContext(
            name="graph-node.draft_recommendation",
            observation_id="node-1",
        )
    ):
        outcome = await service.execute(
            state=_state(),
            context=_context(),
        )

    events = _recommendation_events(observability)
    assert outcome.recovered is False
    assert [event.name for event in events] == [
        "recommendation.generated",
        "recommendation.persisted",
    ]
    generated, persisted = events
    assert generated.status is ObservationStatus.OK
    assert generated.metadata["recommendation_outcome"] == "generated"
    assert generated.metadata["recommendation_id"] == str(_RECOMMENDATION_ID)
    assert generated.metadata["recommended_action"] == (SupportRecommendationAction.RESPOND.value)
    assert generated.metadata["citation_count"] == 0
    assert generated.metadata["requires_human_review"] is False
    assert generated.metadata["schema_version"] == "support-recommendation-v1"
    assert generated.metadata["prompt_id"] == "support-recommendation-draft"
    assert generated.metadata["prompt_version"] == 1
    assert generated.metadata["status"] == ObservationStatus.OK.value
    assert "grounding_status" not in generated.metadata
    assert persisted.metadata["recommendation_outcome"] == "persisted"
    assert order.index("event:recommendation.generated") < order.index("persisted")
    assert order.index("persisted") < order.index("event:recommendation.persisted")
    assert len(gateway.requests) == 1
    assert observability.started_observations == []
    exported = repr(events)
    assert "Follow the documented account recovery procedure." not in exported
    assert "The evidence supports a direct response." not in exported


async def test_existing_recommendation_recovery_emits_no_lifecycle_events() -> None:
    recommendation = SupportRecommendation.create(
        recommendation_id=_RECOMMENDATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        classification_id=_CLASSIFICATION_ID,
        accepted_llm_invocation_id=(_RECOMMENDATION_INVOCATION_ID),
        recommended_action=(SupportRecommendationAction.RESPOND),
        response_text=("Follow the documented recovery procedure."),
        requires_human_review=False,
        decision_summary=("The evidence supports a direct response."),
        prompt_id="support-recommendation-draft",
        prompt_version=1,
        prompt_content_hash="a" * 64,
        provider="mock",
        model="mock-model",
    )
    observability = RecordingObservabilityClient()
    service, gateway, repository = _service(
        _success_result(),
        existing_recommendation=recommendation,
        observability_client=observability,
    )

    outcome = await service.execute(
        state=_state(),
        context=_context(),
    )

    assert outcome.recovered is True
    assert gateway.requests == []
    assert repository.commands == []
    assert _recommendation_events(observability) == []


async def test_already_recommended_persistence_does_not_emit_persisted_event() -> None:
    existing = SupportRecommendation.create(
        recommendation_id=_RECOMMENDATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        classification_id=_CLASSIFICATION_ID,
        accepted_llm_invocation_id=(_RECOMMENDATION_INVOCATION_ID),
        recommended_action=(SupportRecommendationAction.RESPOND),
        response_text=("Follow the documented recovery procedure."),
        requires_human_review=False,
        decision_summary=("The evidence supports a direct response."),
        prompt_id="support-recommendation-draft",
        prompt_version=1,
        prompt_content_hash="a" * 64,
        provider="mock",
        model="mock-model",
    )
    observability = RecordingObservabilityClient()
    transaction_manager = RecordingTransactionManager()
    gateway = StubGateway(_success_result(), transaction_manager)
    execution_repository = StubExecutionRepository(
        transaction_manager,
        SupportRecommendationPersistenceResult.ALREADY_RECOMMENDED,
    )
    recommendation_query = StubRecommendationQueryRepository(
        transaction_manager,
        None,
    )
    ids = iter((_RECOMMENDATION_INVOCATION_ID, UUID("90000000-0000-4000-8000-000000000099")))
    service = ControlledSupportRecommendationExecutor(
        gateway=gateway,  # type: ignore[arg-type]
        model="mock-model",
        request_timeout_seconds=20,
        transaction_manager=transaction_manager,
        observation_assembler=EmptyObservationAssembler(),  # type: ignore[arg-type]
        invocation_query_repository=StubInvocationRepository(transaction_manager, ()),
        recommendation_query_repository=recommendation_query,
        execution_repository=execution_repository,
        utc_now=lambda: _NOW,
        uuid_factory=lambda: next(ids),
        observability_client=observability,  # type: ignore[arg-type]
    )

    async def load_after_conflict(
        *,
        workspace_id: UUID,
        agent_run_id: UUID,
    ) -> SupportRecommendation | None:
        assert workspace_id == _WORKSPACE_ID
        assert agent_run_id == _AGENT_RUN_ID
        if execution_repository.commands:
            return existing
        return None

    recommendation_query.get_by_agent_run_id = load_after_conflict  # type: ignore[method-assign]

    outcome = await service.execute(
        state=_state(),
        context=_context(),
    )

    events = _recommendation_events(observability)
    assert outcome.recovered is True
    assert outcome.recommendation == existing
    assert [event.name for event in events] == ["recommendation.generated"]
    assert all(event.name != "recommendation.persisted" for event in events)


async def test_lease_lost_emits_generated_and_failed_events() -> None:
    observability = RecordingObservabilityClient()
    service, _, _ = _service(
        _success_result(),
        persistence_result=(SupportRecommendationPersistenceResult.LEASE_LOST),
        observability_client=observability,
    )

    with pytest.raises(RetryableAgentRunExecutionError) as captured:
        await service.execute(
            state=_state(),
            context=_context(),
        )

    assert type(captured.value) is RetryableAgentRunExecutionError
    assert captured.value.error_code == "support_recommendation_lease_lost"
    events = _recommendation_events(observability)
    assert [event.name for event in events] == [
        "recommendation.generated",
        "recommendation.failed",
    ]
    failed = events[1]
    assert failed.status is ObservationStatus.ERROR
    assert failed.metadata["recommendation_outcome"] == "failed"
    assert failed.metadata["error_code"] == "support_recommendation_lease_lost"
    assert failed.error_code == "support_recommendation_lease_lost"
    assert "lease was lost" not in repr(failed).lower()


async def test_gateway_failure_does_not_emit_recommendation_lifecycle_events() -> None:
    observability = RecordingObservabilityClient()
    service, _, repository = _service(
        _failure(),
        observability_client=observability,
    )

    with pytest.raises(RetryableAgentRunExecutionError):
        await service.execute(
            state=_state(),
            context=_context(),
        )

    assert repository.commands[0].recommendation is None
    assert _recommendation_events(observability) == []


async def test_recommendation_observability_failures_fail_open() -> None:
    observability = RecordingObservabilityClient(fail_event=True)
    service, _, repository = _service(
        _success_result(),
        observability_client=observability,
    )

    outcome = await service.execute(
        state=_state(),
        context=_context(),
    )

    assert outcome.recovered is False
    assert outcome.recommendation.id == _RECOMMENDATION_ID
    assert len(repository.commands) == 1
    assert _recommendation_events(observability) == []
