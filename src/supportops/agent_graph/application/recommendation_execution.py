"""Grounded recommendation drafting and fenced persistence."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, JsonValue

from supportops.agent_graph.application.tool_observations import (
    ControlledToolObservationAssembler,
    ControlledToolObservationBundle,
)
from supportops.agent_graph.application.transitions import (
    attach_recommendation,
    attach_recommendation_invocation,
)
from supportops.agent_graph.domain.state import (
    ControlledSupportGraphStateSnapshot,
)
from supportops.agent_tools.application.execution import (
    ToolExecutionContext,
)
from supportops.ai.gateway.contracts import (
    LLMOperation,
    LLMRequest,
)
from supportops.ai.gateway.results import (
    LLMGatewayFailure,
    LLMGatewayResult,
    LLMInvocationTrace,
)
from supportops.ai.gateway.service import LLMGateway
from supportops.ai.pricing.catalog import (
    DEFAULT_PRICING_CATALOG,
    PricingCatalog,
)
from supportops.ai.pricing.estimation import (
    estimate_llm_cost,
)
from supportops.ai.prompts.support_recommendation_v1 import (
    SUPPORT_RECOMMENDATION_PROMPT_VERSION,
    render_support_recommendation_prompt,
)
from supportops.core.transactions import TransactionManager
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    RetryableAgentRunExecutionError,
    TerminalAgentRunExecutionError,
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
    SupportRecommendationCitation,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
)
from supportops.observability.context import (
    current_observation_context,
    current_trace_context,
)
from supportops.observability.contracts import ObservabilityClient
from supportops.observability.identity import agent_run_trace_identity
from supportops.observability.models import (
    EventObservation,
    ObservationStatus,
)
from supportops.observability.noop import NoOpObservabilityClient

type UtcNowProvider = Callable[[], datetime]
type UuidFactory = Callable[[], UUID]

_RECOMMENDATION_GENERATED_EVENT: Final = "recommendation.generated"
_RECOMMENDATION_PERSISTED_EVENT: Final = "recommendation.persisted"
_RECOMMENDATION_FAILED_EVENT: Final = "recommendation.failed"
_LEASE_LOST_ERROR_CODE: Final = "support_recommendation_lease_lost"

_RECOMMENDATION_EVENT_METADATA_KEYS: Final = frozenset(
    {
        "recommendation_id",
        "agent_run_id",
        "agent_run_attempt_id",
        "workspace_id",
        "ticket_id",
        "recommended_action",
        "citation_count",
        "requires_human_review",
        "schema_version",
        "prompt_id",
        "prompt_version",
        "status",
        "recommendation_outcome",
        "error_code",
    }
)
_RECOMMENDATION_EVENT_METADATA_PATHS: Final = frozenset(
    (key,) for key in _RECOMMENDATION_EVENT_METADATA_KEYS
)


@dataclass(frozen=True, slots=True)
class RecommendationExecutionOutcome:
    """Persisted recommendation and checkpoint-compatible state."""

    state: ControlledSupportGraphStateSnapshot
    recommendation: SupportRecommendation
    recovered: bool


class ControlledSupportRecommendationExecutor:
    """Draft and atomically persist one grounded recommendation."""

    def __init__(
        self,
        *,
        gateway: LLMGateway,
        model: str,
        request_timeout_seconds: float,
        transaction_manager: TransactionManager,
        observation_assembler: ControlledToolObservationAssembler,
        invocation_query_repository: (AttemptLLMInvocationQueryRepository),
        recommendation_query_repository: (SupportRecommendationQueryRepository),
        execution_repository: (SupportRecommendationExecutionRepository),
        pricing_catalog: PricingCatalog = (DEFAULT_PRICING_CATALOG),
        utc_now: UtcNowProvider | None = None,
        uuid_factory: UuidFactory = uuid4,
        observability_client: ObservabilityClient | None = None,
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
        self._observation_assembler = observation_assembler
        self._invocation_query_repository = invocation_query_repository
        self._recommendation_query_repository = recommendation_query_repository
        self._execution_repository = execution_repository
        self._pricing_catalog = pricing_catalog
        self._utc_now = utc_now or _utc_now
        self._uuid_factory = uuid_factory
        self._observability_client = observability_client or NoOpObservabilityClient()

    async def execute(
        self,
        *,
        state: ControlledSupportGraphStateSnapshot,
        context: AgentRunExecutionContext,
    ) -> RecommendationExecutionOutcome:
        """Draft or recover one persisted recommendation."""

        _validate_context_ownership(
            state=state,
            context=context,
        )
        _validate_state_readiness(state)

        existing = await self._load_existing_recommendation(
            context=context,
        )

        if existing is not None:
            return _recover_existing_recommendation(
                state=state,
                recommendation=existing,
            )

        observations = await self._observation_assembler.assemble(
            state=state,
            context=ToolExecutionContext(
                workspace_id=state.workspace_id,
                ticket_id=state.ticket_id,
                agent_run_id=state.agent_run_id,
                agent_run_attempt_id=(context.attempt.id),
            ),
        )
        rendered_prompt = render_support_recommendation_prompt(
            version=(SUPPORT_RECOMMENDATION_PROMPT_VERSION),
            subject=context.ticket.subject,
            description=context.ticket.description,
            classification=(_classification_projection(state)),
            terminal_analysis=(_terminal_analysis_projection(state)),
            tool_observations=(observations.to_prompt_observations()),
        )
        existing_invocations = await self._load_existing_invocations(
            context=context,
        )
        request = LLMRequest(
            operation=(LLMOperation.SUPPORT_RECOMMENDATION_DRAFT),
            model=self._model,
            instructions=rendered_prompt.instructions,
            input=rendered_prompt.input,
            output_schema=SupportRecommendationResult,
            timeout_seconds=(self._request_timeout_seconds),
            metadata=_build_request_metadata(
                context=context,
                prompt_id=(rendered_prompt.definition.prompt_id),
                prompt_version=(rendered_prompt.definition.version),
                prompt_content_hash=(rendered_prompt.definition.content_hash),
                schema_version=(rendered_prompt.definition.output_schema_id),
            ),
        )

        try:
            gateway_result = await self._gateway.generate(request)
        except LLMGatewayFailure as failure:
            await self._handle_gateway_failure(
                context=context,
                existing_invocations=(existing_invocations),
                failure=failure,
                prompt_id=(rendered_prompt.definition.prompt_id),
                prompt_version=(rendered_prompt.definition.version),
                prompt_content_hash=(rendered_prompt.definition.content_hash),
                schema_version=(rendered_prompt.definition.output_schema_id),
            )
            raise AssertionError("unreachable") from None

        return await self._handle_gateway_success(
            state=state,
            context=context,
            observations=observations,
            existing_invocations=existing_invocations,
            result=gateway_result,
            prompt_id=rendered_prompt.definition.prompt_id,
            prompt_version=(rendered_prompt.definition.version),
            prompt_content_hash=(rendered_prompt.definition.content_hash),
            schema_version=(rendered_prompt.definition.output_schema_id),
        )

    async def _handle_gateway_success(
        self,
        *,
        state: ControlledSupportGraphStateSnapshot,
        context: AgentRunExecutionContext,
        observations: ControlledToolObservationBundle,
        existing_invocations: tuple[LLMInvocation, ...],
        result: LLMGatewayResult,
        prompt_id: str,
        prompt_version: int,
        prompt_content_hash: str,
        schema_version: str,
    ) -> RecommendationExecutionOutcome:
        output = _require_recommendation_output(result.output)
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
        accepted_invocation = _find_current_invocation(
            invocations=current_invocations,
            sequence_offset=len(existing_invocations),
            local_sequence=(result.accepted_invocation_sequence),
        )
        recommendation = SupportRecommendation.create(
            recommendation_id=self._uuid_factory(),
            workspace_id=state.workspace_id,
            ticket_id=state.ticket_id,
            agent_run_id=state.agent_run_id,
            classification_id=_require_classification_id(state),
            accepted_llm_invocation_id=(accepted_invocation.id),
            recommended_action=(output.recommended_action),
            response_text=output.response_text,
            requires_human_review=(output.requires_human_review),
            decision_summary=output.decision_summary,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            prompt_content_hash=prompt_content_hash,
            provider=accepted_invocation.provider,
            model=accepted_invocation.model,
            now=persisted_at,
        )
        citations = tuple(
            SupportRecommendationCitation.create(
                citation_id=self._uuid_factory(),
                workspace_id=state.workspace_id,
                support_recommendation_id=recommendation.id,
                ordinal=ordinal,
                retrieval_query_id=(source.retrieval_query_id),
                retrieval_rank=source.retrieval_rank,
                retrieval_score=source.retrieval_score,
                document_id=source.document_id,
                document_version_id=(source.document_version_id),
                chunk_id=source.chunk_id,
                now=persisted_at,
            )
            for ordinal, source in enumerate(
                observations.citation_sources,
                start=1,
            )
        )
        safe_record_recommendation_event(
            client=self._observability_client,
            context=context,
            event=recommendation_generated_event(
                context=context,
                recommendation=recommendation,
                citation_count=len(citations),
            ),
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
                citations=citations,
            )
        )

        if persistence_result is (SupportRecommendationPersistenceResult.LEASE_LOST):
            safe_record_recommendation_event(
                client=self._observability_client,
                context=context,
                event=recommendation_failed_event(
                    context=context,
                    recommendation=recommendation,
                    citation_count=len(citations),
                    error_code=_LEASE_LOST_ERROR_CODE,
                ),
            )
            _raise_lease_lost()

        if persistence_result is (SupportRecommendationPersistenceResult.APPLIED):
            safe_record_recommendation_event(
                client=self._observability_client,
                context=context,
                event=recommendation_persisted_event(
                    context=context,
                    recommendation=recommendation,
                    citation_count=len(citations),
                ),
            )
            return _apply_persisted_recommendation(
                state=state,
                recommendation=recommendation,
                recovered=False,
            )

        if persistence_result is (SupportRecommendationPersistenceResult.ALREADY_RECOMMENDED):
            persisted = await self._load_existing_recommendation(
                context=context,
            )

            if persisted is None:
                raise RuntimeError(
                    "Recommendation persistence reported an "
                    "existing recommendation that could not "
                    "be loaded."
                )

            return _recover_existing_recommendation(
                state=state,
                recommendation=persisted,
            )

        raise RuntimeError("Successful recommendation persistence returned an invalid result.")

    async def _handle_gateway_failure(
        self,
        *,
        context: AgentRunExecutionContext,
        existing_invocations: tuple[LLMInvocation, ...],
        failure: LLMGatewayFailure,
        prompt_id: str,
        prompt_version: int,
        prompt_content_hash: str,
        schema_version: str,
    ) -> None:
        persisted_at = self._utc_now()
        current_invocations = self._materialize_invocations(
            context=context,
            traces=failure.invocations,
            sequence_offset=len(existing_invocations),
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            prompt_content_hash=prompt_content_hash,
            schema_version=schema_version,
            persisted_at=persisted_at,
        )
        persistence_result = await self._persist(
            PersistSupportRecommendationCommand(
                workspace_id=(context.agent_run.workspace_id),
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

        if persistence_result is (SupportRecommendationPersistenceResult.LEASE_LOST):
            _raise_lease_lost()

        if persistence_result not in {
            SupportRecommendationPersistenceResult.APPLIED,
            SupportRecommendationPersistenceResult.ALREADY_RECORDED,
        }:
            raise RuntimeError(
                "Failed recommendation invocation persistence returned an invalid result."
            )

        _raise_gateway_failure(failure)

    async def _load_existing_recommendation(
        self,
        *,
        context: AgentRunExecutionContext,
    ) -> SupportRecommendation | None:
        async with self._transaction_manager.transaction():
            return await self._recommendation_query_repository.get_by_agent_run_id(
                workspace_id=(context.agent_run.workspace_id),
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
            invocations = await self._invocation_query_repository.list_by_attempt(query)

        _validate_invocation_sequences(invocations)

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
        _validate_trace_sequences(traces)
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
                    workspace_id=(context.agent_run.workspace_id),
                    ticket_id=context.ticket.id,
                    agent_run_id=context.agent_run.id,
                    agent_run_attempt_id=context.attempt.id,
                    invocation_sequence=(sequence_offset + trace.invocation_sequence),
                    status=trace.status,
                    provider=trace.provider,
                    model=trace.model,
                    provider_request_id=(trace.provider_request_id),
                    prompt_id=prompt_id,
                    prompt_version=prompt_version,
                    prompt_content_hash=(prompt_content_hash),
                    schema_version=schema_version,
                    input_tokens=(usage.input_tokens if usage is not None else None),
                    cached_input_tokens=(usage.cached_input_tokens if usage is not None else None),
                    output_tokens=(usage.output_tokens if usage is not None else None),
                    reasoning_tokens=(usage.reasoning_tokens if usage is not None else None),
                    total_tokens=(usage.total_tokens if usage is not None else None),
                    pricing_catalog_version=(cost_estimate.pricing_catalog_version),
                    pricing_found=(cost_estimate.pricing_found),
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


def _apply_persisted_recommendation(
    *,
    state: ControlledSupportGraphStateSnapshot,
    recommendation: SupportRecommendation,
    recovered: bool,
) -> RecommendationExecutionOutcome:
    state_with_invocation = attach_recommendation_invocation(
        state,
        recommendation.accepted_llm_invocation_id,
    )
    completed_state = attach_recommendation(
        state_with_invocation,
        recommendation,
    )

    return RecommendationExecutionOutcome(
        state=completed_state,
        recommendation=recommendation,
        recovered=recovered,
    )


def _recover_existing_recommendation(
    *,
    state: ControlledSupportGraphStateSnapshot,
    recommendation: SupportRecommendation,
) -> RecommendationExecutionOutcome:
    return _apply_persisted_recommendation(
        state=state,
        recommendation=recommendation,
        recovered=True,
    )


def _classification_projection(
    state: ControlledSupportGraphStateSnapshot,
) -> dict[str, JsonValue]:
    classification_id = _require_classification_id(state)

    if (
        state.classification_category is None
        or state.classification_intent is None
        or state.classification_urgency is None
        or state.classification_sentiment is None
        or state.classification_requires_human_review is None
        or state.classification_summary is None
    ):
        raise ValueError("Recommendation drafting requires a complete classification projection.")

    return {
        "classification_id": str(classification_id),
        "category": state.classification_category.value,
        "intent": state.classification_intent.value,
        "urgency": state.classification_urgency.value,
        "sentiment": state.classification_sentiment.value,
        "requires_human_review": (state.classification_requires_human_review),
        "summary": state.classification_summary,
    }


def _terminal_analysis_projection(
    state: ControlledSupportGraphStateSnapshot,
) -> dict[str, JsonValue]:
    completion = state.analysis_completion

    if completion is None:
        raise ValueError("Recommendation drafting requires terminal analysis completion.")

    return cast(
        dict[str, JsonValue],
        completion.model_dump(mode="json"),
    )


def _require_classification_id(
    state: ControlledSupportGraphStateSnapshot,
) -> UUID:
    if state.classification_id is None:
        raise ValueError("Recommendation drafting requires a persisted classification.")

    return state.classification_id


def _validate_state_readiness(
    state: ControlledSupportGraphStateSnapshot,
) -> None:
    _require_classification_id(state)

    if state.analysis_completion is None:
        raise ValueError("Recommendation drafting requires terminal analysis completion.")

    if state.recommendation_id is not None:
        raise ValueError("Graph state already contains a persisted recommendation.")

    if state.current_error_code is not None:
        raise ValueError("Recommendation drafting cannot continue after a graph error.")


def _require_recommendation_output(
    output: BaseModel,
) -> SupportRecommendationResult:
    if not isinstance(
        output,
        SupportRecommendationResult,
    ):
        raise RuntimeError("The recommendation Gateway returned an unexpected output schema.")

    return output


def _find_current_invocation(
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
        raise RuntimeError("The accepted recommendation invocation was not materialized.")

    return invocation


def _validate_invocation_sequences(
    invocations: tuple[LLMInvocation, ...],
) -> None:
    actual = tuple(invocation.invocation_sequence for invocation in invocations)
    expected = tuple(
        range(
            1,
            len(invocations) + 1,
        )
    )

    if actual != expected:
        raise RuntimeError("Persisted attempt invocation sequences are not contiguous and ordered.")


def _validate_trace_sequences(
    traces: tuple[LLMInvocationTrace, ...],
) -> None:
    if not traces:
        raise ValueError("Recommendation generation requires invocation traces.")

    actual = tuple(trace.invocation_sequence for trace in traces)
    expected = tuple(
        range(
            1,
            len(traces) + 1,
        )
    )

    if actual != expected:
        raise RuntimeError(
            "Gateway invocation traces must be contiguous, ordered, and start at one."
        )


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


def _build_request_metadata(
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


def _raise_gateway_failure(
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


def _raise_lease_lost() -> None:
    raise RetryableAgentRunExecutionError(
        error_code=_LEASE_LOST_ERROR_CODE,
        error_summary=(
            "The AgentRun lease was lost before the support recommendation could be persisted."
        ),
    )


def safe_record_recommendation_event(
    *,
    client: ObservabilityClient,
    context: AgentRunExecutionContext,
    event: EventObservation | None,
) -> None:
    if event is None:
        return

    try:
        if current_observation_context() is not None or current_trace_context() is not None:
            client.record_event(event)
            return

        client.record_trace_event(
            identity=agent_run_trace_identity(
                agent_run_id=context.agent_run.id,
                ticket_id=context.ticket.id,
            ),
            event=event,
        )
    except Exception:
        return


def recommendation_generated_event(
    *,
    context: AgentRunExecutionContext,
    recommendation: SupportRecommendation,
    citation_count: int,
) -> EventObservation | None:
    return _recommendation_lifecycle_event(
        name=_RECOMMENDATION_GENERATED_EVENT,
        status=ObservationStatus.OK,
        outcome="generated",
        context=context,
        recommendation=recommendation,
        citation_count=citation_count,
    )


def recommendation_persisted_event(
    *,
    context: AgentRunExecutionContext,
    recommendation: SupportRecommendation,
    citation_count: int,
) -> EventObservation | None:
    return _recommendation_lifecycle_event(
        name=_RECOMMENDATION_PERSISTED_EVENT,
        status=ObservationStatus.OK,
        outcome="persisted",
        context=context,
        recommendation=recommendation,
        citation_count=citation_count,
    )


def recommendation_failed_event(
    *,
    context: AgentRunExecutionContext,
    recommendation: SupportRecommendation,
    citation_count: int,
    error_code: str,
) -> EventObservation | None:
    return _recommendation_lifecycle_event(
        name=_RECOMMENDATION_FAILED_EVENT,
        status=ObservationStatus.ERROR,
        outcome="failed",
        context=context,
        recommendation=recommendation,
        citation_count=citation_count,
        error_code=error_code,
    )


def _recommendation_lifecycle_event(
    *,
    name: str,
    status: ObservationStatus,
    outcome: str,
    context: AgentRunExecutionContext,
    recommendation: SupportRecommendation,
    citation_count: int,
    error_code: str | None = None,
) -> EventObservation | None:
    try:
        metadata = {
            "recommendation_id": str(recommendation.id),
            "agent_run_id": str(recommendation.agent_run_id),
            "agent_run_attempt_id": str(context.attempt.id),
            "workspace_id": str(recommendation.workspace_id),
            "ticket_id": str(recommendation.ticket_id),
            "recommended_action": recommendation.recommended_action.value,
            "citation_count": citation_count,
            "requires_human_review": recommendation.requires_human_review,
            "schema_version": recommendation.schema_version,
            "prompt_id": recommendation.prompt_id,
            "prompt_version": recommendation.prompt_version,
            "status": status.value,
            "recommendation_outcome": outcome,
        }
        if error_code is not None:
            metadata["error_code"] = error_code

        return EventObservation(
            name=name,
            status=status,
            metadata=metadata,
            metadata_paths=_RECOMMENDATION_EVENT_METADATA_PATHS,
            error_code=error_code,
        )
    except Exception:
        return None


def _validate_required_text(
    value: str,
    *,
    field_name: str,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")

    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")


def _utc_now() -> datetime:
    return datetime.now(UTC)
