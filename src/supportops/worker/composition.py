"""Process-scoped LLM and session-scoped executor composition."""

from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

from pydantic import JsonValue, SecretStr
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from supportops.agent_graph.application.decision_execution import (
    ControlledSupportDecisionExecutor,
)
from supportops.agent_graph.application.human_approved_nodes import (
    HumanApprovedDecisionExecutionOutcome,
    HumanApprovedSupportWorkflowNodes,
)
from supportops.agent_graph.application.human_approved_recommendation import (
    HumanApprovedRecommendationOutcome,
)
from supportops.agent_graph.application.human_approved_workflow import (
    HumanApprovedSupportWorkflowExecutor,
    compile_human_approved_support_graph,
)
from supportops.agent_graph.application.recommendation_execution import (
    ControlledSupportRecommendationExecutor,
)
from supportops.agent_graph.application.resume_planning import (
    HumanApprovedGraphResumePlanner,
)
from supportops.agent_graph.application.sensitive_proposal import (
    SensitiveProposalService,
)
from supportops.agent_graph.application.sensitive_tool_execution import (
    SensitiveToolExecutionNode,
)
from supportops.agent_graph.application.tool_execution import (
    ControlledToolDecisionExecutor,
)
from supportops.agent_graph.application.tool_observations import (
    ControlledToolObservationAssembler,
)
from supportops.agent_graph.application.workflow import (
    ControlledSupportWorkflowExecutor,
    ControlledSupportWorkflowNodes,
    compile_controlled_support_graph,
)
from supportops.agent_graph.domain.human_approved_state import (
    HUMAN_APPROVED_GRAPH_STATE_MAX_DECISION_TURNS,
    HUMAN_APPROVED_GRAPH_STATE_MAX_TOOL_CALLS,
    HUMAN_APPROVED_SUPPORT_WORKFLOW_VERSION,
    HumanApprovedSupportGraphStateSnapshot,
)
from supportops.agent_graph.domain.state import (
    CONTROLLED_SUPPORT_WORKFLOW_VERSION,
)
from supportops.agent_graph.infrastructure.checkpoints import (
    PostgresCheckpointRuntime,
    create_postgres_checkpoint_runtime,
)
from supportops.agent_tools.application.execution import (
    BoundedReadOnlyToolExecutor,
)
from supportops.agent_tools.application.sensitive_bindings import (
    SensitiveToolRegistry,
)
from supportops.agent_tools.application.sensitive_execution import (
    ExecuteApprovedTicketEscalation,
)
from supportops.agent_tools.infrastructure.grant_repository import (
    SqlAlchemySensitiveExecutionGrantRepository,
)
from supportops.agent_tools.infrastructure.query_repository import (
    SqlAlchemyAgentToolCallQueryRepository,
)
from supportops.agent_tools.infrastructure.repository import (
    SqlAlchemyAgentToolCallExecutionRepository,
)
from supportops.agent_tools.tools.escalate_ticket import (
    create_escalate_ticket_binding,
)
from supportops.agent_tools.tools.registry import (
    create_controlled_support_tool_registry,
)
from supportops.agent_tools.tools.service_status import (
    DeterministicServiceStatusCatalog,
)
from supportops.ai.embeddings.contracts import EmbeddingProvider
from supportops.ai.gateway.contracts import (
    LLMOperation,
    LLMProvider,
    LLMRequest,
)
from supportops.ai.gateway.results import (
    LLMGatewayFailure,
    LLMGatewayResult,
    LLMInvocationTrace,
)
from supportops.ai.gateway.service import LLMGateway
from supportops.ai.gateway.tool_decisions import (
    LLMHumanApprovedToolDecisionRequest,
    LLMToolDecisionGateway,
    LLMToolDecisionGatewayResult,
    LLMToolDecisionProvider,
)
from supportops.ai.pricing.catalog import (
    DEFAULT_PRICING_CATALOG,
    PricingCatalog,
)
from supportops.ai.pricing.estimation import estimate_llm_cost
from supportops.ai.prompts.human_approved_support_decision_v1 import (
    HUMAN_APPROVED_SUPPORT_DECISION_PROMPT_VERSION,
    render_human_approved_support_decision_prompt,
)
from supportops.ai.prompts.human_approved_support_recommendation_v1 import (
    HUMAN_APPROVED_SUPPORT_RECOMMENDATION_PROMPT_VERSION,
    render_human_approved_support_recommendation_prompt,
)
from supportops.ai.providers.mock import (
    MOCK_TICKET_CLASSIFIER_MODEL,
    MockLLMProvider,
)
from supportops.ai.providers.openai import OpenAILLMProvider
from supportops.ai.schemas.human_approved_support_decision import (
    COMPLETE_HUMAN_APPROVED_SUPPORT_ANALYSIS_CONTROL,
)
from supportops.core.settings import Settings
from supportops.core.transactions import TransactionManager
from supportops.infrastructure.qdrant import (
    close_qdrant_client,
    create_qdrant_client,
)
from supportops.knowledge_index.composition import (
    build_knowledge_index_profile,
    create_embedding_provider,
)
from supportops.knowledge_index.vector_store.qdrant import (
    QdrantKnowledgeVectorStore,
)
from supportops.knowledge_retrieval.postgresql import (
    SqlAlchemyActiveKnowledgeVersionResolver,
    SqlAlchemyKnowledgeChunkHydrator,
)
from supportops.knowledge_retrieval.qdrant import (
    QdrantKnowledgeVectorSearcher,
)
from supportops.knowledge_retrieval.service import SearchKnowledge
from supportops.modules.agent_runs.application.deterministic_executor import (
    DeterministicTicketProcessingExecutor,
)
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    RetryableAgentRunExecutionError,
    TerminalAgentRunExecutionError,
)
from supportops.modules.agent_runs.application.executor_registry import (
    AgentRunExecutorRegistration,
    AgentRunExecutorRegistry,
)
from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    INITIAL_TICKET_PROCESSING_WORKFLOW_NAME,
    TICKET_CLASSIFICATION_WORKFLOW_VERSION,
)
from supportops.modules.approvals.infrastructure.repository import (
    SqlAlchemyApprovalRequestRepository,
)
from supportops.modules.knowledge_documents.domain.models import (
    KnowledgeIndexProfile,
)
from supportops.modules.support_recommendations.application.invocation_queries import (
    AttemptLLMInvocationQuery,
    AttemptLLMInvocationQueryRepository,
)
from supportops.modules.support_recommendations.application.persistence import (
    PersistSupportRecommendationCommand,
    SupportRecommendationExecutionRepository,
    SupportRecommendationPersistenceResult,
)
from supportops.modules.support_recommendations.application.queries import (
    SupportRecommendationQueryRepository,
)
from supportops.modules.support_recommendations.application.schemas import (
    SupportRecommendationResult,
)
from supportops.modules.support_recommendations.domain.models import (
    SupportRecommendation,
)
from supportops.modules.support_recommendations.infrastructure.invocation_query_repository import (
    SqlAlchemyAttemptLLMInvocationQueryRepository,
)
from supportops.modules.support_recommendations.infrastructure.query_repository import (
    SqlAlchemySupportRecommendationQueryRepository,
)
from supportops.modules.support_recommendations.infrastructure.repository import (
    SqlAlchemySupportRecommendationExecutionRepository,
)
from supportops.modules.ticket_classifications.application.executor import (
    TicketClassificationExecutor,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
)
from supportops.modules.ticket_classifications.domain.repositories import (
    TicketClassificationRepository,
)
from supportops.modules.ticket_classifications.infrastructure.repository import (
    SqlAlchemyClassificationPersistenceRepository,
    SqlAlchemyTicketClassificationQueryRepository,
)
from supportops.modules.tickets.infrastructure.escalation_repository import (
    SqlAlchemyTicketEscalationRepository,
)
from supportops.observability.composition import create_observability_client
from supportops.observability.contracts import ObservabilityClient
from supportops.observability.noop import NoOpObservabilityClient

MOCK_LLM_PROVIDER_NAME = "mock"
OPENAI_LLM_PROVIDER_NAME = "openai"

type CheckpointRuntimeFactory = Callable[
    ...,
    Awaitable[PostgresCheckpointRuntime],
]
type EmbeddingProviderFactory = Callable[..., EmbeddingProvider]
type QdrantClientFactory = Callable[[Settings], AsyncQdrantClient]
type KnowledgeIndexProfileFactory = Callable[
    [Settings],
    KnowledgeIndexProfile,
]
type ObservabilityClientFactory = Callable[[Settings], ObservabilityClient]
type UtcNowProvider = Callable[[], datetime]
type UuidFactory = Callable[[], UUID]


@dataclass(frozen=True, slots=True)
class WorkerLLMRuntime:
    """Own one process-scoped provider and application LLM Gateway."""

    provider: LLMProvider
    gateway: LLMGateway
    model: str

    async def close(self) -> None:
        """Release provider-owned process resources."""

        await self.provider.close()


@dataclass(slots=True)
class WorkerControlledSupportRuntime:
    """Own process-scoped controlled-support infrastructure resources."""

    checkpoint_runtime: PostgresCheckpointRuntime
    embedding_provider: EmbeddingProvider
    qdrant_client: AsyncQdrantClient
    index_profile: KnowledgeIndexProfile
    vector_store: QdrantKnowledgeVectorStore
    vector_searcher: QdrantKnowledgeVectorSearcher
    observability_client: ObservabilityClient = field(
        default_factory=NoOpObservabilityClient,
    )
    approval_ttl_seconds: float = 86400.0
    agent_graph_tool_timeout_seconds: float = 15.0
    _closed: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    async def close(self) -> None:
        """Release process-owned resources idempotently."""

        if self._closed:
            return

        self._closed = True
        failures: list[Exception] = []

        try:
            await self.checkpoint_runtime.close()
        except Exception as error:
            failures.append(error)

        try:
            await self.embedding_provider.close()
        except Exception as error:
            failures.append(error)

        try:
            await close_qdrant_client(self.qdrant_client)
        except Exception as error:
            failures.append(error)

        with suppress(Exception):
            self.observability_client.shutdown()

        if failures:
            primary_failure = failures[0]
            for secondary_failure in failures[1:]:
                primary_failure.add_note(
                    "An additional controlled-support runtime "
                    "resource failed to close: "
                    f"{type(secondary_failure).__name__}."
                )
            raise primary_failure


class HumanApprovedSupportDecisionExecutor:
    """Execute and durably record one human-approved decision turn."""

    def __init__(
        self,
        *,
        gateway: LLMToolDecisionGateway,
        sensitive_tool_registry: SensitiveToolRegistry,
        model: str,
        request_timeout_seconds: float,
        transaction_manager: TransactionManager,
        invocation_query_repository: AttemptLLMInvocationQueryRepository,
        execution_repository: SupportRecommendationExecutionRepository,
        pricing_catalog: PricingCatalog = DEFAULT_PRICING_CATALOG,
        utc_now: UtcNowProvider | None = None,
        uuid_factory: UuidFactory = uuid4,
    ) -> None:
        _validate_required_text(
            model,
            field_name="model",
        )

        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive.")

        self._gateway = gateway
        self._sensitive_tool_registry = sensitive_tool_registry
        self._model = model
        self._request_timeout_seconds = request_timeout_seconds
        self._transaction_manager = transaction_manager
        self._invocation_query_repository = invocation_query_repository
        self._execution_repository = execution_repository
        self._pricing_catalog = pricing_catalog
        self._utc_now = utc_now or _utc_now
        self._uuid_factory = uuid_factory

    async def execute(
        self,
        *,
        state: HumanApprovedSupportGraphStateSnapshot,
        context: AgentRunExecutionContext,
        tool_observations: tuple[
            Mapping[str, JsonValue],
            ...,
        ],
    ) -> HumanApprovedDecisionExecutionOutcome:
        """Execute one decision turn without proposing or executing tools."""

        _validate_human_approved_context_ownership(
            state=state,
            context=context,
        )
        classification = _human_approved_classification_projection(
            state,
        )
        remaining_tool_calls = max(
            0,
            HUMAN_APPROVED_GRAPH_STATE_MAX_TOOL_CALLS - state.tool_call_count,
        )
        remaining_decision_turns = max(
            0,
            HUMAN_APPROVED_GRAPH_STATE_MAX_DECISION_TURNS - state.decision_turn_count,
        )
        visible_definitions = (
            self._sensitive_tool_registry.definitions if remaining_tool_calls > 0 else ()
        )
        rendered_prompt = render_human_approved_support_decision_prompt(
            version=HUMAN_APPROVED_SUPPORT_DECISION_PROMPT_VERSION,
            subject=context.ticket.subject,
            description=context.ticket.description,
            classification=classification,
            tool_observations=tool_observations,
            available_tool_names=tuple(definition.name for definition in visible_definitions),
            remaining_tool_calls=remaining_tool_calls,
            remaining_decision_turns=remaining_decision_turns,
        )
        existing_invocations = await self._load_existing_invocations(
            context=context,
        )
        request = LLMHumanApprovedToolDecisionRequest(
            operation=LLMOperation.SUPPORT_ACTION_DECISION,
            model=self._model,
            instructions=rendered_prompt.instructions,
            input=rendered_prompt.input,
            sensitive_tools=visible_definitions,
            terminal_control=(COMPLETE_HUMAN_APPROVED_SUPPORT_ANALYSIS_CONTROL),
            timeout_seconds=self._request_timeout_seconds,
            prompt_id=rendered_prompt.definition.prompt_id,
            prompt_version=rendered_prompt.definition.version,
            metadata=_build_human_approved_request_metadata(
                context=context,
                decision_turn=state.decision_turn_count + 1,
                prompt_id=rendered_prompt.definition.prompt_id,
                prompt_version=rendered_prompt.definition.version,
                prompt_content_hash=(rendered_prompt.definition.content_hash),
                schema_version=(rendered_prompt.definition.output_schema_id),
            ),
        )

        try:
            result = await self._gateway.decide_human_approved(request)
        except LLMGatewayFailure as failure:
            await self._persist_gateway_failure(
                context=context,
                existing_invocations=existing_invocations,
                traces=failure.invocations,
                prompt_id=rendered_prompt.definition.prompt_id,
                prompt_version=rendered_prompt.definition.version,
                prompt_content_hash=(rendered_prompt.definition.content_hash),
                schema_version=(rendered_prompt.definition.output_schema_id),
            )
            _raise_human_approved_gateway_failure(failure)

        return await self._handle_gateway_success(
            context=context,
            existing_invocations=existing_invocations,
            result=result,
            prompt_id=rendered_prompt.definition.prompt_id,
            prompt_version=rendered_prompt.definition.version,
            prompt_content_hash=(rendered_prompt.definition.content_hash),
            schema_version=(rendered_prompt.definition.output_schema_id),
        )

    async def _handle_gateway_success(
        self,
        *,
        context: AgentRunExecutionContext,
        existing_invocations: tuple[LLMInvocation, ...],
        result: LLMToolDecisionGatewayResult,
        prompt_id: str,
        prompt_version: int,
        prompt_content_hash: str,
        schema_version: str,
    ) -> HumanApprovedDecisionExecutionOutcome:
        persisted_at = self._utc_now()
        current_invocations = self._materialize_invocations(
            context=context,
            traces=result.invocations,
            sequence_offset=len(existing_invocations),
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            prompt_content_hash=prompt_content_hash,
            schema_version=schema_version,
            persisted_at=persisted_at,
        )
        accepted_invocation = _find_human_approved_invocation(
            invocations=current_invocations,
            sequence_offset=len(existing_invocations),
            local_sequence=result.accepted_invocation_sequence,
        )
        persistence_result = await self._persist_invocations(
            context=context,
            persisted_at=persisted_at,
            invocations=(
                *existing_invocations,
                *current_invocations,
            ),
        )
        _require_human_approved_persistence_result(persistence_result)

        return HumanApprovedDecisionExecutionOutcome(
            decision=result.decision,
            accepted_invocation_id=accepted_invocation.id,
        )

    async def _persist_gateway_failure(
        self,
        *,
        context: AgentRunExecutionContext,
        existing_invocations: tuple[LLMInvocation, ...],
        traces: tuple[LLMInvocationTrace, ...],
        prompt_id: str,
        prompt_version: int,
        prompt_content_hash: str,
        schema_version: str,
    ) -> None:
        persisted_at = self._utc_now()
        current_invocations = self._materialize_invocations(
            context=context,
            traces=traces,
            sequence_offset=len(existing_invocations),
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            prompt_content_hash=prompt_content_hash,
            schema_version=schema_version,
            persisted_at=persisted_at,
        )
        persistence_result = await self._persist_invocations(
            context=context,
            persisted_at=persisted_at,
            invocations=(
                *existing_invocations,
                *current_invocations,
            ),
        )

        if persistence_result is SupportRecommendationPersistenceResult.LEASE_LOST:
            raise RetryableAgentRunExecutionError(
                error_code="human_approved_decision_lease_lost",
                error_summary=(
                    "The AgentRun lease was lost before human-approved "
                    "decision invocations could be persisted."
                ),
            )

        if persistence_result not in {
            SupportRecommendationPersistenceResult.APPLIED,
            SupportRecommendationPersistenceResult.ALREADY_RECORDED,
        }:
            raise RuntimeError(
                "Failed human-approved decision persistence returned an invalid result."
            )

    async def _load_existing_invocations(
        self,
        *,
        context: AgentRunExecutionContext,
    ) -> tuple[LLMInvocation, ...]:
        query = AttemptLLMInvocationQuery(
            workspace_id=context.agent_run.workspace_id,
            ticket_id=context.ticket.id,
            agent_run_id=context.agent_run.id,
            agent_run_attempt_id=context.attempt.id,
        )

        async with self._transaction_manager.transaction():
            invocations = await self._invocation_query_repository.list_by_attempt(query)

        actual_sequences = tuple(invocation.invocation_sequence for invocation in invocations)
        expected_sequences = tuple(range(1, len(invocations) + 1))
        if actual_sequences != expected_sequences:
            raise RuntimeError(
                "Persisted attempt invocation sequences are not contiguous and ordered."
            )

        return invocations

    async def _persist_invocations(
        self,
        *,
        context: AgentRunExecutionContext,
        persisted_at: datetime,
        invocations: tuple[LLMInvocation, ...],
    ) -> SupportRecommendationPersistenceResult:
        command = PersistSupportRecommendationCommand(
            workspace_id=context.agent_run.workspace_id,
            ticket_id=context.ticket.id,
            agent_run_id=context.agent_run.id,
            agent_run_attempt_id=context.attempt.id,
            lease_token=context.attempt.lease_token,
            persisted_at=persisted_at,
            invocations=invocations,
            recommendation=None,
            citations=(),
        )

        async with self._transaction_manager.transaction():
            return await self._execution_repository.persist_fenced(command)

    def _materialize_invocations(
        self,
        *,
        context: AgentRunExecutionContext,
        traces: tuple[LLMInvocationTrace, ...],
        sequence_offset: int,
        prompt_id: str,
        prompt_version: int,
        prompt_content_hash: str,
        schema_version: str,
        persisted_at: datetime,
    ) -> tuple[LLMInvocation, ...]:
        if not traces:
            raise ValueError("Decision execution requires invocation traces.")

        actual_sequences = tuple(trace.invocation_sequence for trace in traces)
        expected_sequences = tuple(range(1, len(traces) + 1))
        if actual_sequences != expected_sequences:
            raise RuntimeError(
                "Gateway trace sequences must be contiguous, ordered, and start at one."
            )

        invocations: list[LLMInvocation] = []

        for trace in traces:
            usage = trace.usage
            cost_estimate = estimate_llm_cost(
                provider=trace.provider,
                model=trace.model,
                usage=usage,
                catalog=self._pricing_catalog,
            )
            invocations.append(
                LLMInvocation.create(
                    invocation_id=self._uuid_factory(),
                    workspace_id=context.agent_run.workspace_id,
                    ticket_id=context.ticket.id,
                    agent_run_id=context.agent_run.id,
                    agent_run_attempt_id=context.attempt.id,
                    invocation_sequence=(sequence_offset + trace.invocation_sequence),
                    status=trace.status,
                    provider=trace.provider,
                    model=trace.model,
                    provider_request_id=trace.provider_request_id,
                    prompt_id=prompt_id,
                    prompt_version=prompt_version,
                    prompt_content_hash=prompt_content_hash,
                    schema_version=schema_version,
                    input_tokens=(usage.input_tokens if usage is not None else None),
                    cached_input_tokens=(usage.cached_input_tokens if usage is not None else None),
                    output_tokens=(usage.output_tokens if usage is not None else None),
                    reasoning_tokens=(usage.reasoning_tokens if usage is not None else None),
                    total_tokens=(usage.total_tokens if usage is not None else None),
                    pricing_catalog_version=(cost_estimate.pricing_catalog_version),
                    pricing_found=cost_estimate.pricing_found,
                    estimated_input_cost_usd=(cost_estimate.estimated_input_cost_usd),
                    estimated_cached_input_cost_usd=(cost_estimate.estimated_cached_input_cost_usd),
                    estimated_output_cost_usd=(cost_estimate.estimated_output_cost_usd),
                    estimated_total_cost_usd=(cost_estimate.estimated_total_cost_usd),
                    latency_ms=trace.latency_ms,
                    error_code=trace.error_code,
                    now=persisted_at,
                )
            )

        return tuple(invocations)


class HumanApprovedSupportRecommendationExecutor:
    """Draft and persist one approval-aware grounded recommendation."""

    def __init__(
        self,
        *,
        gateway: LLMGateway,
        model: str,
        request_timeout_seconds: float,
        transaction_manager: TransactionManager,
        invocation_query_repository: AttemptLLMInvocationQueryRepository,
        recommendation_query_repository: SupportRecommendationQueryRepository,
        execution_repository: SupportRecommendationExecutionRepository,
        pricing_catalog: PricingCatalog = DEFAULT_PRICING_CATALOG,
        utc_now: UtcNowProvider | None = None,
        uuid_factory: UuidFactory = uuid4,
    ) -> None:
        _validate_required_text(
            model,
            field_name="model",
        )
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive.")

        self._gateway = gateway
        self._model = model
        self._request_timeout_seconds = request_timeout_seconds
        self._transaction_manager = transaction_manager
        self._invocation_query_repository = invocation_query_repository
        self._recommendation_query_repository = recommendation_query_repository
        self._execution_repository = execution_repository
        self._pricing_catalog = pricing_catalog
        self._utc_now = utc_now or _utc_now
        self._uuid_factory = uuid_factory

    async def execute(
        self,
        *,
        context: AgentRunExecutionContext,
        state: HumanApprovedSupportGraphStateSnapshot,
        workflow: Mapping[str, JsonValue],
    ) -> HumanApprovedRecommendationOutcome:
        """Return one durable recommendation for the approval-aware graph."""

        _validate_human_approved_context_ownership(
            state=state,
            context=context,
        )
        if state.classification_id is None:
            raise ValueError(
                "Recommendation drafting requires a persisted classification.",
            )
        if state.recommendation_id is not None:
            raise ValueError(
                "Graph state already contains a persisted recommendation.",
            )
        if state.current_error_code is not None:
            raise ValueError(
                "Recommendation drafting cannot continue after a graph error.",
            )

        existing = await self._load_existing_recommendation(context=context)
        if existing is not None:
            return HumanApprovedRecommendationOutcome(
                invocation_id=existing.accepted_llm_invocation_id,
                recommendation=existing,
            )

        rendered_prompt = render_human_approved_support_recommendation_prompt(
            version=HUMAN_APPROVED_SUPPORT_RECOMMENDATION_PROMPT_VERSION,
            workflow=workflow,
        )
        existing_invocations = await self._load_existing_invocations(
            context=context,
        )
        request = LLMRequest(
            operation=LLMOperation.SUPPORT_RECOMMENDATION_DRAFT,
            model=self._model,
            instructions=rendered_prompt.instructions,
            input=rendered_prompt.input,
            output_schema=SupportRecommendationResult,
            timeout_seconds=self._request_timeout_seconds,
            metadata=_build_human_approved_recommendation_metadata(
                context=context,
                prompt_id=rendered_prompt.definition.prompt_id,
                prompt_version=rendered_prompt.definition.version,
                prompt_content_hash=(rendered_prompt.definition.content_hash),
                schema_version=(rendered_prompt.definition.output_schema_id),
            ),
        )

        try:
            gateway_result = await self._gateway.generate(request)
        except LLMGatewayFailure as failure:
            await self._persist_gateway_failure(
                context=context,
                existing_invocations=existing_invocations,
                traces=failure.invocations,
                prompt_id=rendered_prompt.definition.prompt_id,
                prompt_version=rendered_prompt.definition.version,
                prompt_content_hash=(rendered_prompt.definition.content_hash),
                schema_version=(rendered_prompt.definition.output_schema_id),
            )
            _raise_human_approved_gateway_failure(failure)

        return await self._handle_gateway_success(
            state=state,
            context=context,
            existing_invocations=existing_invocations,
            result=gateway_result,
            prompt_id=rendered_prompt.definition.prompt_id,
            prompt_version=rendered_prompt.definition.version,
            prompt_content_hash=(rendered_prompt.definition.content_hash),
            schema_version=(rendered_prompt.definition.output_schema_id),
        )

    async def _handle_gateway_success(
        self,
        *,
        state: HumanApprovedSupportGraphStateSnapshot,
        context: AgentRunExecutionContext,
        existing_invocations: tuple[LLMInvocation, ...],
        result: LLMGatewayResult,
        prompt_id: str,
        prompt_version: int,
        prompt_content_hash: str,
        schema_version: str,
    ) -> HumanApprovedRecommendationOutcome:
        output = _require_human_approved_recommendation_output(result.output)
        persisted_at = self._utc_now()
        current_invocations = self._materialize_invocations(
            context=context,
            traces=result.invocations,
            sequence_offset=len(existing_invocations),
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            prompt_content_hash=prompt_content_hash,
            schema_version=schema_version,
            persisted_at=persisted_at,
        )
        accepted_invocation = _find_human_approved_invocation(
            invocations=current_invocations,
            sequence_offset=len(existing_invocations),
            local_sequence=result.accepted_invocation_sequence,
        )
        if state.classification_id is None:
            raise ValueError(
                "Recommendation drafting requires a persisted classification.",
            )
        recommendation = SupportRecommendation.create(
            recommendation_id=self._uuid_factory(),
            workspace_id=state.workspace_id,
            ticket_id=state.ticket_id,
            agent_run_id=state.agent_run_id,
            classification_id=state.classification_id,
            accepted_llm_invocation_id=accepted_invocation.id,
            recommended_action=output.recommended_action,
            response_text=output.response_text,
            requires_human_review=output.requires_human_review,
            decision_summary=output.decision_summary,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            prompt_content_hash=prompt_content_hash,
            provider=accepted_invocation.provider,
            model=accepted_invocation.model,
            now=persisted_at,
        )
        persistence_result = await self._persist(
            PersistSupportRecommendationCommand(
                workspace_id=state.workspace_id,
                ticket_id=state.ticket_id,
                agent_run_id=state.agent_run_id,
                agent_run_attempt_id=context.attempt.id,
                lease_token=context.attempt.lease_token,
                persisted_at=persisted_at,
                invocations=(
                    *existing_invocations,
                    *current_invocations,
                ),
                recommendation=recommendation,
                citations=(),
            )
        )

        if persistence_result is SupportRecommendationPersistenceResult.LEASE_LOST:
            raise RetryableAgentRunExecutionError(
                error_code="human_approved_recommendation_lease_lost",
                error_summary=(
                    "The AgentRun lease was lost before the human-approved "
                    "recommendation could be persisted."
                ),
            )

        if persistence_result is SupportRecommendationPersistenceResult.APPLIED:
            return HumanApprovedRecommendationOutcome(
                invocation_id=accepted_invocation.id,
                recommendation=recommendation,
            )

        if persistence_result is (SupportRecommendationPersistenceResult.ALREADY_RECOMMENDED):
            persisted = await self._load_existing_recommendation(
                context=context,
            )
            if persisted is None:
                raise RuntimeError(
                    "Recommendation persistence reported an "
                    "existing recommendation that could not be loaded."
                )
            return HumanApprovedRecommendationOutcome(
                invocation_id=persisted.accepted_llm_invocation_id,
                recommendation=persisted,
            )

        raise RuntimeError(
            "Successful human-approved recommendation persistence returned an invalid result."
        )

    async def _persist_gateway_failure(
        self,
        *,
        context: AgentRunExecutionContext,
        existing_invocations: tuple[LLMInvocation, ...],
        traces: tuple[LLMInvocationTrace, ...],
        prompt_id: str,
        prompt_version: int,
        prompt_content_hash: str,
        schema_version: str,
    ) -> None:
        persisted_at = self._utc_now()
        current_invocations = self._materialize_invocations(
            context=context,
            traces=traces,
            sequence_offset=len(existing_invocations),
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            prompt_content_hash=prompt_content_hash,
            schema_version=schema_version,
            persisted_at=persisted_at,
        )
        persistence_result = await self._persist(
            PersistSupportRecommendationCommand(
                workspace_id=context.agent_run.workspace_id,
                ticket_id=context.ticket.id,
                agent_run_id=context.agent_run.id,
                agent_run_attempt_id=context.attempt.id,
                lease_token=context.attempt.lease_token,
                persisted_at=persisted_at,
                invocations=(
                    *existing_invocations,
                    *current_invocations,
                ),
                recommendation=None,
                citations=(),
            )
        )

        if persistence_result is SupportRecommendationPersistenceResult.LEASE_LOST:
            raise RetryableAgentRunExecutionError(
                error_code="human_approved_recommendation_lease_lost",
                error_summary=(
                    "The AgentRun lease was lost before human-approved "
                    "recommendation failure invocations could be persisted."
                ),
            )

        if persistence_result not in {
            SupportRecommendationPersistenceResult.APPLIED,
            SupportRecommendationPersistenceResult.ALREADY_RECORDED,
        }:
            raise RuntimeError(
                "Failed human-approved recommendation persistence returned an invalid result."
            )

    async def _load_existing_recommendation(
        self,
        *,
        context: AgentRunExecutionContext,
    ) -> SupportRecommendation | None:
        async with self._transaction_manager.transaction():
            return await self._recommendation_query_repository.get_by_agent_run_id(
                workspace_id=context.agent_run.workspace_id,
                agent_run_id=context.agent_run.id,
            )

    async def _load_existing_invocations(
        self,
        *,
        context: AgentRunExecutionContext,
    ) -> tuple[LLMInvocation, ...]:
        query = AttemptLLMInvocationQuery(
            workspace_id=context.agent_run.workspace_id,
            ticket_id=context.ticket.id,
            agent_run_id=context.agent_run.id,
            agent_run_attempt_id=context.attempt.id,
        )

        async with self._transaction_manager.transaction():
            invocations = await self._invocation_query_repository.list_by_attempt(
                query,
            )

        actual_sequences = tuple(invocation.invocation_sequence for invocation in invocations)
        expected_sequences = tuple(range(1, len(invocations) + 1))
        if actual_sequences != expected_sequences:
            raise RuntimeError(
                "Persisted attempt invocation sequences are not contiguous and ordered."
            )

        return invocations

    async def _persist(
        self,
        command: PersistSupportRecommendationCommand,
    ) -> SupportRecommendationPersistenceResult:
        async with self._transaction_manager.transaction():
            return await self._execution_repository.persist_fenced(command)

    def _materialize_invocations(
        self,
        *,
        context: AgentRunExecutionContext,
        traces: tuple[LLMInvocationTrace, ...],
        sequence_offset: int,
        prompt_id: str,
        prompt_version: int,
        prompt_content_hash: str,
        schema_version: str,
        persisted_at: datetime,
    ) -> tuple[LLMInvocation, ...]:
        if not traces:
            raise ValueError("Recommendation execution requires invocation traces.")

        actual_sequences = tuple(trace.invocation_sequence for trace in traces)
        expected_sequences = tuple(range(1, len(traces) + 1))
        if actual_sequences != expected_sequences:
            raise RuntimeError(
                "Gateway trace sequences must be contiguous, ordered, and start at one."
            )

        invocations: list[LLMInvocation] = []
        for trace in traces:
            usage = trace.usage
            cost_estimate = estimate_llm_cost(
                provider=trace.provider,
                model=trace.model,
                usage=usage,
                catalog=self._pricing_catalog,
            )
            invocations.append(
                LLMInvocation.create(
                    invocation_id=self._uuid_factory(),
                    workspace_id=context.agent_run.workspace_id,
                    ticket_id=context.ticket.id,
                    agent_run_id=context.agent_run.id,
                    agent_run_attempt_id=context.attempt.id,
                    invocation_sequence=(sequence_offset + trace.invocation_sequence),
                    status=trace.status,
                    provider=trace.provider,
                    model=trace.model,
                    provider_request_id=trace.provider_request_id,
                    prompt_id=prompt_id,
                    prompt_version=prompt_version,
                    prompt_content_hash=prompt_content_hash,
                    schema_version=schema_version,
                    input_tokens=(usage.input_tokens if usage is not None else None),
                    cached_input_tokens=(usage.cached_input_tokens if usage is not None else None),
                    output_tokens=(usage.output_tokens if usage is not None else None),
                    reasoning_tokens=(usage.reasoning_tokens if usage is not None else None),
                    total_tokens=(usage.total_tokens if usage is not None else None),
                    pricing_catalog_version=(cost_estimate.pricing_catalog_version),
                    pricing_found=cost_estimate.pricing_found,
                    estimated_input_cost_usd=(cost_estimate.estimated_input_cost_usd),
                    estimated_cached_input_cost_usd=(cost_estimate.estimated_cached_input_cost_usd),
                    estimated_output_cost_usd=(cost_estimate.estimated_output_cost_usd),
                    estimated_total_cost_usd=(cost_estimate.estimated_total_cost_usd),
                    latency_ms=trace.latency_ms,
                    error_code=trace.error_code,
                    now=persisted_at,
                )
            )

        return tuple(invocations)


def create_worker_llm_runtime(
    *,
    provider_name: str,
    openai_api_key: str | None,
    openai_model: str,
    openai_base_url: str | None,
    request_timeout_seconds: float,
    transport_max_retries: int,
    max_repair_attempts: int,
    observability_client: ObservabilityClient | None = None,
) -> WorkerLLMRuntime:
    """Create one explicitly configured process-scoped LLM runtime."""

    _validate_required_text(
        provider_name,
        field_name="provider_name",
    )

    provider: LLMProvider
    model: str

    if provider_name == MOCK_LLM_PROVIDER_NAME:
        provider = MockLLMProvider(
            model=MOCK_TICKET_CLASSIFIER_MODEL,
        )
        model = MOCK_TICKET_CLASSIFIER_MODEL
    elif provider_name == OPENAI_LLM_PROVIDER_NAME:
        if openai_api_key is None:
            raise ValueError(
                "openai_api_key is required when the OpenAI provider is selected.",
            )

        provider = OpenAILLMProvider.create(
            api_key=openai_api_key,
            model=openai_model,
            timeout_seconds=request_timeout_seconds,
            transport_max_retries=transport_max_retries,
            base_url=openai_base_url,
        )
        model = openai_model
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider_name}.",
        )

    gateway = LLMGateway(
        provider=provider,
        max_repair_attempts=max_repair_attempts,
        observability_client=observability_client,
    )

    return WorkerLLMRuntime(
        provider=provider,
        gateway=gateway,
        model=model,
    )


async def create_worker_controlled_support_runtime(
    *,
    settings: Settings,
    checkpoint_runtime_factory: CheckpointRuntimeFactory = (create_postgres_checkpoint_runtime),
    embedding_provider_factory: EmbeddingProviderFactory = (create_embedding_provider),
    qdrant_client_factory: QdrantClientFactory = (create_qdrant_client),
    index_profile_factory: KnowledgeIndexProfileFactory = (build_knowledge_index_profile),
    observability_client: ObservabilityClient | None = None,
    observability_client_factory: ObservabilityClientFactory = (create_observability_client),
) -> WorkerControlledSupportRuntime:
    """Create one process-scoped controlled-support runtime."""

    checkpoint_database_url = SecretStr(
        _to_psycopg_connection_url(
            str(settings.postgresql_url),
        ),
    )
    checkpoint_runtime: PostgresCheckpointRuntime | None = None
    embedding_provider: EmbeddingProvider | None = None
    qdrant_client: AsyncQdrantClient | None = None
    owned_observability_client = observability_client
    created_observability_client = False

    try:
        if owned_observability_client is None:
            owned_observability_client = observability_client_factory(
                settings,
            )
            created_observability_client = True

        checkpoint_runtime = await checkpoint_runtime_factory(
            database_url=checkpoint_database_url,
        )
        await checkpoint_runtime.setup()

        index_profile = index_profile_factory(settings)
        embedding_provider = embedding_provider_factory(
            settings,
            observability_client=owned_observability_client,
        )
        qdrant_client = qdrant_client_factory(settings)
        vector_store = QdrantKnowledgeVectorStore(
            client=qdrant_client,
        )
        vector_searcher = QdrantKnowledgeVectorSearcher(
            client=qdrant_client,
            collection_guard=vector_store,
        )
    except Exception:
        await _close_partial_controlled_support_resources(
            checkpoint_runtime=checkpoint_runtime,
            embedding_provider=embedding_provider,
            qdrant_client=qdrant_client,
            observability_client=(
                owned_observability_client if created_observability_client else None
            ),
        )
        raise

    if owned_observability_client is None:
        raise RuntimeError(
            "Controlled-support runtime requires an observability client.",
        )

    return WorkerControlledSupportRuntime(
        checkpoint_runtime=checkpoint_runtime,
        embedding_provider=embedding_provider,
        qdrant_client=qdrant_client,
        index_profile=index_profile,
        vector_store=vector_store,
        vector_searcher=vector_searcher,
        observability_client=owned_observability_client,
        approval_ttl_seconds=settings.approval_ttl_seconds,
        agent_graph_tool_timeout_seconds=(settings.agent_graph_tool_timeout_seconds),
    )


def create_session_scoped_executor_registry(
    *,
    session: AsyncSession,
    transaction_manager: TransactionManager,
    gateway: LLMGateway,
    provider: LLMProvider,
    model: str,
    request_timeout_seconds: float,
    controlled_runtime: WorkerControlledSupportRuntime,
    embedding_timeout_seconds: float,
) -> AgentRunExecutorRegistry:
    """Create all workflow executors owned by one database session."""

    classification_repository = SqlAlchemyClassificationPersistenceRepository(
        session,
    )
    classification_executor = TicketClassificationExecutor(
        gateway=gateway,
        model=model,
        request_timeout_seconds=request_timeout_seconds,
        transaction_manager=transaction_manager,
        classification_repository=(classification_repository),
        execution_repository=classification_repository,
    )
    classification_query_repository = SqlAlchemyTicketClassificationQueryRepository(
        session,
    )

    active_version_resolver = SqlAlchemyActiveKnowledgeVersionResolver(
        session,
    )
    chunk_hydrator = SqlAlchemyKnowledgeChunkHydrator(session)
    knowledge_search = SearchKnowledge(
        active_version_resolver=active_version_resolver,
        chunk_hydrator=chunk_hydrator,
        embedding_provider=(controlled_runtime.embedding_provider),
        vector_searcher=controlled_runtime.vector_searcher,
        index_profile=controlled_runtime.index_profile,
        embedding_timeout_seconds=embedding_timeout_seconds,
        observability_client=(controlled_runtime.observability_client),
    )
    tool_registry = create_controlled_support_tool_registry(
        knowledge_search=knowledge_search,
        service_status_catalog=(DeterministicServiceStatusCatalog(())),
    )
    bounded_tool_executor = BoundedReadOnlyToolExecutor(
        registry=tool_registry,
    )
    tool_call_execution_repository = SqlAlchemyAgentToolCallExecutionRepository(
        session,
    )
    tool_call_query_repository = SqlAlchemyAgentToolCallQueryRepository(
        session,
    )
    invocation_query_repository = SqlAlchemyAttemptLLMInvocationQueryRepository(
        session,
    )
    recommendation_execution_repository = SqlAlchemySupportRecommendationExecutionRepository(
        session,
    )
    recommendation_query_repository = SqlAlchemySupportRecommendationQueryRepository(
        session,
    )
    observation_assembler = ControlledToolObservationAssembler(
        transaction_manager=transaction_manager,
        tool_call_repository=tool_call_query_repository,
        chunk_hydrator=chunk_hydrator,
    )
    decision_gateway = LLMToolDecisionGateway(
        provider=cast(LLMToolDecisionProvider, provider),
        observability_client=(controlled_runtime.observability_client),
    )
    decision_executor = ControlledSupportDecisionExecutor(
        gateway=decision_gateway,
        tool_registry=tool_registry,
        model=model,
        request_timeout_seconds=request_timeout_seconds,
        transaction_manager=transaction_manager,
        invocation_query_repository=(invocation_query_repository),
        execution_repository=(recommendation_execution_repository),
    )
    tool_executor = ControlledToolDecisionExecutor(
        executor=bounded_tool_executor,
        transaction_manager=transaction_manager,
        execution_repository=(tool_call_execution_repository),
        query_repository=tool_call_query_repository,
    )
    recommendation_executor = ControlledSupportRecommendationExecutor(
        gateway=gateway,
        model=model,
        request_timeout_seconds=request_timeout_seconds,
        transaction_manager=transaction_manager,
        observation_assembler=observation_assembler,
        invocation_query_repository=(invocation_query_repository),
        recommendation_query_repository=(recommendation_query_repository),
        execution_repository=(recommendation_execution_repository),
    )
    nodes = ControlledSupportWorkflowNodes(
        transaction_manager=transaction_manager,
        classification_repository=cast(
            TicketClassificationRepository,
            classification_query_repository,
        ),
        classification_executor=classification_executor,
        observation_assembler=observation_assembler,
        decision_executor=decision_executor,
        tool_executor=tool_executor,
        recommendation_executor=recommendation_executor,
    )
    graph = compile_controlled_support_graph(
        nodes=nodes,
        checkpointer=(controlled_runtime.checkpoint_runtime.checkpointer),
    )
    controlled_executor = ControlledSupportWorkflowExecutor(
        graph=graph,
    )

    sensitive_tool_registry = SensitiveToolRegistry(
        (
            create_escalate_ticket_binding(
                timeout_seconds=(controlled_runtime.agent_graph_tool_timeout_seconds),
            ),
        ),
    )
    human_approved_decision_executor = HumanApprovedSupportDecisionExecutor(
        gateway=decision_gateway,
        sensitive_tool_registry=sensitive_tool_registry,
        model=model,
        request_timeout_seconds=request_timeout_seconds,
        transaction_manager=transaction_manager,
        invocation_query_repository=(invocation_query_repository),
        execution_repository=(recommendation_execution_repository),
    )
    approval_request_repository = SqlAlchemyApprovalRequestRepository(
        session,
    )
    sensitive_proposal_service = SensitiveProposalService(
        transaction_manager=transaction_manager,
        sensitive_tool_registry=sensitive_tool_registry,
        tool_call_execution_repository=(tool_call_execution_repository),
        tool_call_query_repository=tool_call_query_repository,
        approval_request_repository=(approval_request_repository),
        approval_ttl_seconds=(controlled_runtime.approval_ttl_seconds),
    )
    grant_repository = SqlAlchemySensitiveExecutionGrantRepository(
        session,
    )
    escalation_repository = SqlAlchemyTicketEscalationRepository(
        session,
    )
    approved_escalation_executor = ExecuteApprovedTicketEscalation(
        transaction_manager=transaction_manager,
        approval_request_repository=(approval_request_repository),
        tool_call_repository=tool_call_execution_repository,
        grant_repository=grant_repository,
        escalation_repository=escalation_repository,
    )
    sensitive_tool_execution = SensitiveToolExecutionNode(
        executor=approved_escalation_executor,
    )
    human_approved_recommendation_executor = HumanApprovedSupportRecommendationExecutor(
        gateway=gateway,
        model=model,
        request_timeout_seconds=request_timeout_seconds,
        transaction_manager=transaction_manager,
        invocation_query_repository=(invocation_query_repository),
        recommendation_query_repository=(recommendation_query_repository),
        execution_repository=(recommendation_execution_repository),
    )
    human_approved_nodes = HumanApprovedSupportWorkflowNodes(
        transaction_manager=transaction_manager,
        classification_repository=cast(
            TicketClassificationRepository,
            classification_query_repository,
        ),
        classification_executor=classification_executor,
        decision_executor=human_approved_decision_executor,
        sensitive_tool_registry=sensitive_tool_registry,
        sensitive_proposal_service=sensitive_proposal_service,
        sensitive_tool_execution=sensitive_tool_execution,
        approval_request_repository=(approval_request_repository),
        recommendation_executor=(human_approved_recommendation_executor),
    )
    human_approved_graph = compile_human_approved_support_graph(
        nodes=human_approved_nodes,
        checkpointer=(controlled_runtime.checkpoint_runtime.checkpointer),
    )
    human_approved_resume_planner = HumanApprovedGraphResumePlanner(
        approval_request_repository=(approval_request_repository),
        tool_call_query_repository=(tool_call_query_repository),
    )
    human_approved_executor = HumanApprovedSupportWorkflowExecutor(
        graph=human_approved_graph,
        resume_planner=human_approved_resume_planner,
    )

    return AgentRunExecutorRegistry(
        (
            AgentRunExecutorRegistration(
                workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
                workflow_version=(DETERMINISTIC_BASELINE_WORKFLOW_VERSION),
                executor=(DeterministicTicketProcessingExecutor()),
            ),
            AgentRunExecutorRegistration(
                workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
                workflow_version=(TICKET_CLASSIFICATION_WORKFLOW_VERSION),
                executor=classification_executor,
            ),
            AgentRunExecutorRegistration(
                workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
                workflow_version=(CONTROLLED_SUPPORT_WORKFLOW_VERSION),
                executor=controlled_executor,
            ),
            AgentRunExecutorRegistration(
                workflow_name=(INITIAL_TICKET_PROCESSING_WORKFLOW_NAME),
                workflow_version=(HUMAN_APPROVED_SUPPORT_WORKFLOW_VERSION),
                executor=human_approved_executor,
            ),
        ),
    )


def _to_psycopg_connection_url(database_url: str) -> str:
    for sqlalchemy_prefix in (
        "postgresql+asyncpg://",
        "postgresql+psycopg://",
    ):
        if database_url.startswith(sqlalchemy_prefix):
            return "postgresql://" + database_url[len(sqlalchemy_prefix) :]

    if database_url.startswith("postgresql://"):
        return database_url

    raise ValueError(
        "postgresql_url must use a PostgreSQL connection URL scheme.",
    )


async def _close_partial_controlled_support_resources(
    *,
    checkpoint_runtime: PostgresCheckpointRuntime | None,
    embedding_provider: EmbeddingProvider | None,
    qdrant_client: AsyncQdrantClient | None,
    observability_client: ObservabilityClient | None,
) -> None:
    failures: list[Exception] = []

    if checkpoint_runtime is not None:
        try:
            await checkpoint_runtime.close()
        except Exception as error:
            failures.append(error)

    if embedding_provider is not None:
        try:
            await embedding_provider.close()
        except Exception as error:
            failures.append(error)

    if qdrant_client is not None:
        try:
            await close_qdrant_client(qdrant_client)
        except Exception as error:
            failures.append(error)

    if observability_client is not None:
        with suppress(Exception):
            observability_client.shutdown()

    if failures:
        primary_failure = failures[0]
        for secondary_failure in failures[1:]:
            primary_failure.add_note(
                "An additional partially created resource "
                "failed to close: "
                f"{type(secondary_failure).__name__}."
            )
        raise primary_failure


def _human_approved_classification_projection(
    state: HumanApprovedSupportGraphStateSnapshot,
) -> dict[str, JsonValue]:
    required_values = (
        state.classification_id,
        state.classification_category,
        state.classification_intent,
        state.classification_urgency,
        state.classification_sentiment,
        state.classification_requires_human_review,
        state.classification_summary,
    )

    if any(value is None for value in required_values):
        raise ValueError(
            "Decision execution requires a complete persisted classification projection."
        )

    assert state.classification_id is not None
    assert state.classification_category is not None
    assert state.classification_intent is not None
    assert state.classification_urgency is not None
    assert state.classification_sentiment is not None
    assert state.classification_requires_human_review is not None
    assert state.classification_summary is not None

    return {
        "classification_id": str(state.classification_id),
        "category": state.classification_category.value,
        "intent": state.classification_intent.value,
        "urgency": state.classification_urgency.value,
        "sentiment": state.classification_sentiment.value,
        "requires_human_review": (state.classification_requires_human_review),
        "summary": state.classification_summary,
    }


def _find_human_approved_invocation(
    *,
    invocations: tuple[LLMInvocation, ...],
    sequence_offset: int,
    local_sequence: int,
) -> LLMInvocation:
    global_sequence = sequence_offset + local_sequence
    invocation = next(
        (
            candidate
            for candidate in invocations
            if candidate.invocation_sequence == global_sequence
        ),
        None,
    )

    if invocation is None:
        raise RuntimeError("The accepted decision invocation was not materialized.")

    return invocation


def _require_human_approved_persistence_result(
    result: SupportRecommendationPersistenceResult,
) -> None:
    if result is SupportRecommendationPersistenceResult.LEASE_LOST:
        raise RetryableAgentRunExecutionError(
            error_code="human_approved_decision_lease_lost",
            error_summary=(
                "The AgentRun lease was lost before human-approved "
                "decision invocations could be persisted."
            ),
        )

    if result is SupportRecommendationPersistenceResult.APPLIED:
        return

    if result is SupportRecommendationPersistenceResult.ALREADY_RECOMMENDED:
        raise RuntimeError("Decision execution found an already persisted recommendation.")

    raise RuntimeError("Successful decision persistence returned an invalid result.")


def _validate_human_approved_context_ownership(
    *,
    state: HumanApprovedSupportGraphStateSnapshot,
    context: AgentRunExecutionContext,
) -> None:
    if (
        state.workspace_id != context.agent_run.workspace_id
        or state.ticket_id != context.ticket.id
        or state.agent_run_id != context.agent_run.id
    ):
        raise ValueError("Graph state ownership does not match the AgentRun execution context.")


def _build_human_approved_request_metadata(
    *,
    context: AgentRunExecutionContext,
    decision_turn: int,
    prompt_id: str,
    prompt_version: int,
    prompt_content_hash: str,
    schema_version: str,
) -> dict[str, str]:
    return {
        "supportops_workspace_id": str(context.agent_run.workspace_id),
        "supportops_ticket_id": str(context.ticket.id),
        "supportops_agent_run_id": str(context.agent_run.id),
        "supportops_agent_run_attempt_id": str(context.attempt.id),
        "supportops_correlation_id": str(context.agent_run.correlation_id),
        "supportops_workflow_name": (context.agent_run.workflow_name),
        "supportops_workflow_version": (context.agent_run.workflow_version),
        "supportops_decision_turn": str(decision_turn),
        "supportops_prompt_id": prompt_id,
        "supportops_prompt_version": str(prompt_version),
        "supportops_prompt_content_hash": (prompt_content_hash),
        "supportops_schema_version": schema_version,
    }


def _build_human_approved_recommendation_metadata(
    *,
    context: AgentRunExecutionContext,
    prompt_id: str,
    prompt_version: int,
    prompt_content_hash: str,
    schema_version: str,
) -> dict[str, str]:
    return {
        "supportops_workspace_id": str(context.agent_run.workspace_id),
        "supportops_ticket_id": str(context.ticket.id),
        "supportops_agent_run_id": str(context.agent_run.id),
        "supportops_agent_run_attempt_id": str(context.attempt.id),
        "supportops_correlation_id": str(context.agent_run.correlation_id),
        "supportops_workflow_name": (context.agent_run.workflow_name),
        "supportops_workflow_version": (context.agent_run.workflow_version),
        "supportops_prompt_id": prompt_id,
        "supportops_prompt_version": str(prompt_version),
        "supportops_prompt_content_hash": (prompt_content_hash),
        "supportops_schema_version": schema_version,
    }


def _require_human_approved_recommendation_output(
    output: object,
) -> SupportRecommendationResult:
    if not isinstance(output, SupportRecommendationResult):
        raise RuntimeError(
            "The human-approved recommendation Gateway returned an unexpected output schema."
        )
    return output


def _raise_human_approved_gateway_failure(
    failure: LLMGatewayFailure,
) -> None:
    if failure.retryable:
        raise RetryableAgentRunExecutionError(
            error_code=failure.error_code.value,
            error_summary=failure.error.safe_summary,
        ) from failure

    if failure.terminal:
        raise TerminalAgentRunExecutionError(
            error_code=failure.error_code.value,
            error_summary=failure.error.safe_summary,
        ) from failure

    raise RuntimeError(
        "The LLM Gateway failure defines neither retryable nor terminal behavior."
    ) from failure


def _validate_required_text(
    value: str,
    *,
    field_name: str,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")

    if value != value.strip():
        raise ValueError(
            f"{field_name} must not contain surrounding whitespace.",
        )


def _utc_now() -> datetime:
    return datetime.now(UTC)
