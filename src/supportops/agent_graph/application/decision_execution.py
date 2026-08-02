"""Durable execution of one controlled support decision turn."""

from collections.abc import (
    Callable,
    Mapping,
)
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import JsonValue

from supportops.agent_graph.application.transitions import (
    attach_analysis_completion,
    reserve_decision_turn,
)
from supportops.agent_graph.domain.completion import (
    COMPLETE_SUPPORT_ANALYSIS_CONTROL,
    CompleteSupportAnalysisInput,
)
from supportops.agent_graph.domain.routing import (
    CONTROLLED_SUPPORT_RUNTIME_LIMITS,
    ControlledSupportRuntimeLimits,
    calculate_remaining_workflow_budget,
)
from supportops.agent_graph.domain.state import (
    ControlledSupportGraphStateSnapshot,
)
from supportops.agent_tools.application.bindings import (
    ExecutableToolRegistry,
)
from supportops.ai.gateway.contracts import LLMOperation
from supportops.ai.gateway.results import (
    LLMGatewayFailure,
    LLMInvocationTrace,
)
from supportops.ai.gateway.tool_decisions import (
    LLMExecutableToolCallDecision,
    LLMTerminalControlDecision,
    LLMToolDecisionGatewayResult,
    LLMToolDecisionRequest,
)
from supportops.ai.pricing.catalog import (
    DEFAULT_PRICING_CATALOG,
    PricingCatalog,
)
from supportops.ai.pricing.estimation import (
    estimate_llm_cost,
)
from supportops.ai.prompts.support_tool_decision_v1 import (
    SUPPORT_TOOL_DECISION_PROMPT_VERSION,
    render_support_tool_decision_prompt,
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
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
)

type UtcNowProvider = Callable[[], datetime]
type UuidFactory = Callable[[], UUID]


class SupportToolDecisionGateway(Protocol):
    """Provider-independent Gateway for one controlled decision."""

    async def decide(
        self,
        request: LLMToolDecisionRequest,
    ) -> LLMToolDecisionGatewayResult:
        """Return one validated executable or terminal decision."""

        ...


@dataclass(frozen=True, slots=True)
class ControlledDecisionExecutionOutcome:
    """Durable decision result prepared for graph routing."""

    state: ControlledSupportGraphStateSnapshot
    decision: LLMExecutableToolCallDecision | LLMTerminalControlDecision
    accepted_invocation_id: UUID
    current_invocations: tuple[LLMInvocation, ...]

    def __post_init__(self) -> None:
        if not self.current_invocations:
            raise ValueError("A decision outcome requires current invocations.")

        invocation_ids = {invocation.id for invocation in self.current_invocations}

        if self.accepted_invocation_id not in invocation_ids:
            raise ValueError("accepted_invocation_id must reference a current decision invocation.")


class ControlledSupportDecisionExecutor:
    """Execute and durably record one controlled LLM decision."""

    def __init__(
        self,
        *,
        gateway: SupportToolDecisionGateway,
        tool_registry: ExecutableToolRegistry,
        model: str,
        request_timeout_seconds: float,
        transaction_manager: TransactionManager,
        invocation_query_repository: (AttemptLLMInvocationQueryRepository),
        execution_repository: (SupportRecommendationExecutionRepository),
        limits: ControlledSupportRuntimeLimits = (CONTROLLED_SUPPORT_RUNTIME_LIMITS),
        pricing_catalog: PricingCatalog = (DEFAULT_PRICING_CATALOG),
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
        self._tool_registry = tool_registry
        self._model = model
        self._request_timeout_seconds = request_timeout_seconds
        self._transaction_manager = transaction_manager
        self._invocation_query_repository = invocation_query_repository
        self._execution_repository = execution_repository
        self._limits = limits
        self._pricing_catalog = pricing_catalog
        self._utc_now = utc_now or _utc_now
        self._uuid_factory = uuid_factory

    async def execute(
        self,
        *,
        state: ControlledSupportGraphStateSnapshot,
        context: AgentRunExecutionContext,
        tool_observations: tuple[
            Mapping[str, JsonValue],
            ...,
        ],
    ) -> ControlledDecisionExecutionOutcome:
        """Execute one decision turn without executing its tool."""

        _validate_context_ownership(
            state=state,
            context=context,
        )
        classification = _classification_projection(state)
        budget = calculate_remaining_workflow_budget(
            state,
            limits=self._limits,
        )
        reserved_state = reserve_decision_turn(
            state,
            limits=self._limits,
        )
        visible_definitions = self._tool_registry.definitions if budget.tool_calls > 0 else ()
        rendered_prompt = render_support_tool_decision_prompt(
            version=SUPPORT_TOOL_DECISION_PROMPT_VERSION,
            subject=context.ticket.subject,
            description=context.ticket.description,
            classification=classification,
            tool_observations=tool_observations,
            available_tool_names=tuple(definition.name for definition in visible_definitions),
            remaining_tool_calls=budget.tool_calls,
            remaining_decision_turns=(budget.decision_turns),
        )
        existing_invocations = await self._load_existing_invocations(
            context=context,
        )
        request = LLMToolDecisionRequest(
            operation=LLMOperation.SUPPORT_ACTION_DECISION,
            model=self._model,
            instructions=rendered_prompt.instructions,
            input=rendered_prompt.input,
            tools=visible_definitions,
            terminal_control=(COMPLETE_SUPPORT_ANALYSIS_CONTROL),
            timeout_seconds=(self._request_timeout_seconds),
            metadata=_build_request_metadata(
                context=context,
                decision_turn=(reserved_state.decision_turn_count),
                prompt_id=(rendered_prompt.definition.prompt_id),
                prompt_version=(rendered_prompt.definition.version),
                prompt_content_hash=(rendered_prompt.definition.content_hash),
                schema_version=(rendered_prompt.definition.output_schema_id),
            ),
        )

        try:
            result = await self._gateway.decide(request)
        except LLMGatewayFailure as failure:
            await self._persist_gateway_failure(
                context=context,
                existing_invocations=existing_invocations,
                traces=failure.invocations,
                prompt_id=(rendered_prompt.definition.prompt_id),
                prompt_version=(rendered_prompt.definition.version),
                prompt_content_hash=(rendered_prompt.definition.content_hash),
                schema_version=(rendered_prompt.definition.output_schema_id),
            )
            _raise_gateway_failure(failure)

        return await self._handle_gateway_success(
            state=reserved_state,
            context=context,
            existing_invocations=existing_invocations,
            result=result,
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
        existing_invocations: tuple[LLMInvocation, ...],
        result: LLMToolDecisionGatewayResult,
        prompt_id: str,
        prompt_version: int,
        prompt_content_hash: str,
        schema_version: str,
    ) -> ControlledDecisionExecutionOutcome:
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
        persistence_result = await self._persist_invocations(
            context=context,
            persisted_at=persisted_at,
            invocations=(
                *existing_invocations,
                *current_invocations,
            ),
        )
        _require_successful_persistence_result(persistence_result)

        updated_state = state
        decision = result.decision

        if isinstance(
            decision,
            LLMTerminalControlDecision,
        ):
            completion = _require_completion_output(decision)
            updated_state = attach_analysis_completion(
                state,
                completion,
            )

        return ControlledDecisionExecutionOutcome(
            state=updated_state,
            decision=decision,
            accepted_invocation_id=(accepted_invocation.id),
            current_invocations=current_invocations,
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

        if persistence_result is (SupportRecommendationPersistenceResult.LEASE_LOST):
            _raise_lease_lost()

        if persistence_result not in {
            SupportRecommendationPersistenceResult.APPLIED,
            SupportRecommendationPersistenceResult.ALREADY_RECORDED,
        }:
            raise RuntimeError("Failed decision persistence returned an invalid result.")

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

        _validate_existing_invocation_sequences(invocations)

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
        _validate_local_trace_sequences(traces)

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


def _classification_projection(
    state: ControlledSupportGraphStateSnapshot,
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
        raise RuntimeError("The accepted decision invocation was not materialized.")

    return invocation


def _require_completion_output(
    decision: LLMTerminalControlDecision,
) -> CompleteSupportAnalysisInput:
    if not isinstance(
        decision.output,
        CompleteSupportAnalysisInput,
    ):
        raise RuntimeError("The terminal decision returned an unexpected output schema.")

    return decision.output


def _require_successful_persistence_result(
    result: SupportRecommendationPersistenceResult,
) -> None:
    if result is SupportRecommendationPersistenceResult.LEASE_LOST:
        _raise_lease_lost()

    if result is SupportRecommendationPersistenceResult.APPLIED:
        return

    if result is (SupportRecommendationPersistenceResult.ALREADY_RECOMMENDED):
        raise RuntimeError("Decision execution found an already persisted recommendation.")

    raise RuntimeError("Successful decision persistence returned an invalid result.")


def _validate_existing_invocation_sequences(
    invocations: tuple[LLMInvocation, ...],
) -> None:
    actual_sequences = tuple(invocation.invocation_sequence for invocation in invocations)
    expected_sequences = tuple(
        range(
            1,
            len(invocations) + 1,
        )
    )

    if actual_sequences != expected_sequences:
        raise RuntimeError("Persisted attempt invocation sequences are not contiguous and ordered.")


def _validate_local_trace_sequences(
    traces: tuple[LLMInvocationTrace, ...],
) -> None:
    if not traces:
        raise ValueError("Decision execution requires invocation traces.")

    actual_sequences = tuple(trace.invocation_sequence for trace in traces)
    expected_sequences = tuple(
        range(
            1,
            len(traces) + 1,
        )
    )

    if actual_sequences != expected_sequences:
        raise RuntimeError("Gateway trace sequences must be contiguous, ordered, and start at one.")


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
        error_code="support_decision_lease_lost",
        error_summary=(
            "The AgentRun lease was lost before support decision invocations could be persisted."
        ),
    )


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
