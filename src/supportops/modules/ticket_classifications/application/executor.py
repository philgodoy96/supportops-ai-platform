"""Durable AgentRun executor for structured ticket classification."""

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel

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
from supportops.ai.prompts.ticket_classification_v1 import (
    TICKET_CLASSIFICATION_PROMPT_VERSION,
    render_ticket_classification_prompt,
)
from supportops.ai.schemas.ticket_classification import (
    TicketClassificationResult,
)
from supportops.core.transactions import TransactionManager
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    RetryableAgentRunExecutionError,
    TerminalAgentRunExecutionError,
)
from supportops.modules.agent_runs.domain.models import (
    INITIAL_TICKET_PROCESSING_TRIGGER_KEY,
    INITIAL_TICKET_PROCESSING_WORKFLOW_NAME,
)
from supportops.modules.ticket_classifications.application.persistence import (
    ClassificationExecutionRepository,
    ClassificationPersistenceResult,
    PersistClassificationExecutionCommand,
)
from supportops.modules.ticket_classifications.domain.models import (
    LLMInvocation,
    TicketClassification,
)
from supportops.modules.ticket_classifications.domain.repositories import (
    TicketClassificationRepository,
)

TICKET_CLASSIFICATION_WORKFLOW_VERSION = "ticket-classification-v1"

UtcNowProvider = Callable[[], datetime]
UuidFactory = Callable[[], UUID]


class TicketClassificationExecutor:
    """Execute and durably persist structured ticket classification."""

    def __init__(
        self,
        *,
        gateway: LLMGateway,
        model: str,
        request_timeout_seconds: float,
        transaction_manager: TransactionManager,
        classification_repository: (TicketClassificationRepository),
        execution_repository: ClassificationExecutionRepository,
        pricing_catalog: PricingCatalog = DEFAULT_PRICING_CATALOG,
        utc_now: UtcNowProvider | None = None,
        uuid_factory: UuidFactory = uuid4,
    ) -> None:
        _validate_required_text(
            model,
            field_name="model",
        )

        if request_timeout_seconds <= 0:
            raise ValueError(
                "request_timeout_seconds must be positive.",
            )

        self._gateway = gateway
        self._model = model
        self._request_timeout_seconds = request_timeout_seconds
        self._transaction_manager = transaction_manager
        self._classification_repository = classification_repository
        self._execution_repository = execution_repository
        self._pricing_catalog = pricing_catalog
        self._utc_now = utc_now or _utc_now
        self._uuid_factory = uuid_factory

    async def execute(
        self,
        context: AgentRunExecutionContext,
    ) -> None:
        """Classify one supported ticket-processing AgentRun."""

        _validate_supported_workflow(context)

        if await self._classification_already_exists(context):
            return

        rendered_prompt = render_ticket_classification_prompt(
            version=TICKET_CLASSIFICATION_PROMPT_VERSION,
            subject=context.ticket.subject,
            description=context.ticket.description,
        )
        request = LLMRequest(
            operation=LLMOperation.TICKET_CLASSIFICATION,
            model=self._model,
            instructions=rendered_prompt.instructions,
            input=rendered_prompt.input,
            output_schema=TicketClassificationResult,
            timeout_seconds=self._request_timeout_seconds,
            metadata=_build_request_metadata(
                context=context,
                prompt_id=rendered_prompt.definition.prompt_id,
                prompt_version=(rendered_prompt.definition.version),
                prompt_content_hash=(rendered_prompt.definition.content_hash),
                schema_version=(rendered_prompt.definition.output_schema_id),
            ),
        )

        try:
            gateway_result = await self._gateway.generate(
                request,
            )
        except LLMGatewayFailure as gateway_failure:
            await self._handle_gateway_failure(
                context=context,
                failure=gateway_failure,
                prompt_id=rendered_prompt.definition.prompt_id,
                prompt_version=(rendered_prompt.definition.version),
                prompt_content_hash=(rendered_prompt.definition.content_hash),
                schema_version=(rendered_prompt.definition.output_schema_id),
            )
            return

        await self._handle_gateway_success(
            context=context,
            result=gateway_result,
            prompt_id=rendered_prompt.definition.prompt_id,
            prompt_version=(rendered_prompt.definition.version),
            prompt_content_hash=(rendered_prompt.definition.content_hash),
            schema_version=(rendered_prompt.definition.output_schema_id),
        )

    async def _classification_already_exists(
        self,
        context: AgentRunExecutionContext,
    ) -> bool:
        async with self._transaction_manager.transaction():
            existing = await self._classification_repository.get_by_agent_run_id(
                workspace_id=(context.agent_run.workspace_id),
                agent_run_id=context.agent_run.id,
            )

        return existing is not None

    async def _handle_gateway_success(
        self,
        *,
        context: AgentRunExecutionContext,
        result: LLMGatewayResult,
        prompt_id: str,
        prompt_version: int,
        prompt_content_hash: str,
        schema_version: str,
    ) -> None:
        output = _require_ticket_classification_output(
            result.output,
        )
        persisted_at = self._utc_now()
        invocations = self._materialize_invocations(
            context=context,
            traces=result.invocations,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            prompt_content_hash=prompt_content_hash,
            schema_version=schema_version,
            persisted_at=persisted_at,
        )
        accepted_invocation = _find_invocation_by_sequence(
            invocations,
            sequence=result.accepted_invocation_sequence,
        )
        classification = TicketClassification.create(
            classification_id=self._uuid_factory(),
            workspace_id=context.agent_run.workspace_id,
            ticket_id=context.ticket.id,
            agent_run_id=context.agent_run.id,
            accepted_llm_invocation_id=(accepted_invocation.id),
            category=output.category,
            intent=output.intent,
            urgency=output.urgency,
            sentiment=output.sentiment,
            requires_human_review=(output.requires_human_review),
            summary=output.summary,
            schema_version=output.schema_version,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            prompt_content_hash=prompt_content_hash,
            provider=accepted_invocation.provider,
            model=accepted_invocation.model,
            now=persisted_at,
        )
        persistence_result = await self._persist_execution(
            PersistClassificationExecutionCommand(
                workspace_id=context.agent_run.workspace_id,
                ticket_id=context.ticket.id,
                agent_run_id=context.agent_run.id,
                agent_run_attempt_id=context.attempt.id,
                lease_token=context.attempt.lease_token,
                persisted_at=persisted_at,
                invocations=invocations,
                classification=classification,
            ),
        )

        if persistence_result in {
            ClassificationPersistenceResult.APPLIED,
            ClassificationPersistenceResult.ALREADY_CLASSIFIED,
        }:
            return

        if persistence_result is ClassificationPersistenceResult.LEASE_LOST:
            _raise_lease_lost()

        raise RuntimeError(
            "Successful classification persistence returned an invalid persistence result.",
        )

    async def _handle_gateway_failure(
        self,
        *,
        context: AgentRunExecutionContext,
        failure: LLMGatewayFailure,
        prompt_id: str,
        prompt_version: int,
        prompt_content_hash: str,
        schema_version: str,
    ) -> None:
        persisted_at = self._utc_now()
        invocations = self._materialize_invocations(
            context=context,
            traces=failure.invocations,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            prompt_content_hash=prompt_content_hash,
            schema_version=schema_version,
            persisted_at=persisted_at,
        )
        persistence_result = await self._persist_execution(
            PersistClassificationExecutionCommand(
                workspace_id=context.agent_run.workspace_id,
                ticket_id=context.ticket.id,
                agent_run_id=context.agent_run.id,
                agent_run_attempt_id=context.attempt.id,
                lease_token=context.attempt.lease_token,
                persisted_at=persisted_at,
                invocations=invocations,
                classification=None,
            ),
        )

        if persistence_result is ClassificationPersistenceResult.ALREADY_CLASSIFIED:
            return

        if persistence_result is ClassificationPersistenceResult.LEASE_LOST:
            _raise_lease_lost()

        if persistence_result not in {
            ClassificationPersistenceResult.APPLIED,
            ClassificationPersistenceResult.ALREADY_RECORDED,
        }:
            raise RuntimeError(
                "Failed classification persistence returned an invalid persistence result.",
            )

        _raise_gateway_failure(failure)

    def _materialize_invocations(
        self,
        *,
        context: AgentRunExecutionContext,
        traces: tuple[LLMInvocationTrace, ...],
        prompt_id: str,
        prompt_version: int,
        prompt_content_hash: str,
        schema_version: str,
        persisted_at: datetime,
    ) -> tuple[LLMInvocation, ...]:
        invocations = []

        for trace in traces:
            cost_estimate = estimate_llm_cost(
                provider=trace.provider,
                model=trace.model,
                usage=trace.usage,
                catalog=self._pricing_catalog,
            )
            usage = trace.usage

            invocations.append(
                LLMInvocation.create(
                    invocation_id=self._uuid_factory(),
                    workspace_id=(context.agent_run.workspace_id),
                    ticket_id=context.ticket.id,
                    agent_run_id=context.agent_run.id,
                    agent_run_attempt_id=context.attempt.id,
                    invocation_sequence=(trace.invocation_sequence),
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
                ),
            )

        return tuple(invocations)

    async def _persist_execution(
        self,
        command: PersistClassificationExecutionCommand,
    ) -> ClassificationPersistenceResult:
        async with self._transaction_manager.transaction():
            return await self._execution_repository.persist_fenced(
                command,
            )


def _validate_supported_workflow(
    context: AgentRunExecutionContext,
) -> None:
    run = context.agent_run

    if run.workflow_name != INITIAL_TICKET_PROCESSING_WORKFLOW_NAME:
        raise TerminalAgentRunExecutionError(
            error_code="unsupported_workflow",
            error_summary=(
                "The AgentRun workflow is not supported by the ticket classification executor."
            ),
        )

    if run.workflow_version != TICKET_CLASSIFICATION_WORKFLOW_VERSION:
        raise TerminalAgentRunExecutionError(
            error_code="unsupported_workflow_version",
            error_summary=(
                "The AgentRun workflow version is not supported "
                "by the ticket classification executor."
            ),
        )

    if run.trigger_key != INITIAL_TICKET_PROCESSING_TRIGGER_KEY:
        raise TerminalAgentRunExecutionError(
            error_code="unsupported_trigger",
            error_summary=(
                "The AgentRun trigger is not supported by the ticket classification executor."
            ),
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
        "supportops_workspace_id": str(
            context.agent_run.workspace_id,
        ),
        "supportops_ticket_id": str(context.ticket.id),
        "supportops_agent_run_id": str(
            context.agent_run.id,
        ),
        "supportops_agent_run_attempt_id": str(
            context.attempt.id,
        ),
        "supportops_correlation_id": str(
            context.agent_run.correlation_id,
        ),
        "supportops_workflow_name": (context.agent_run.workflow_name),
        "supportops_workflow_version": (context.agent_run.workflow_version),
        "supportops_prompt_id": prompt_id,
        "supportops_prompt_version": str(prompt_version),
        "supportops_prompt_content_hash": (prompt_content_hash),
        "supportops_schema_version": schema_version,
    }


def _require_ticket_classification_output(
    output: BaseModel,
) -> TicketClassificationResult:
    if not isinstance(
        output,
        TicketClassificationResult,
    ):
        raise RuntimeError(
            "The ticket classification Gateway returned an unexpected output schema.",
        )

    return output


def _find_invocation_by_sequence(
    invocations: tuple[LLMInvocation, ...],
    *,
    sequence: int,
) -> LLMInvocation:
    invocation = next(
        (candidate for candidate in invocations if candidate.invocation_sequence == sequence),
        None,
    )

    if invocation is None:
        raise RuntimeError(
            "The accepted invocation sequence was not materialized.",
        )

    return invocation


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
        "The LLM Gateway failure defines neither retryable nor terminal behavior.",
    ) from failure


def _raise_lease_lost() -> None:
    raise RetryableAgentRunExecutionError(
        error_code="classification_lease_lost",
        error_summary=(
            "The AgentRun lease was lost before classification results could be persisted."
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
        raise ValueError(
            f"{field_name} must not contain surrounding whitespace.",
        )


def _utc_now() -> datetime:
    return datetime.now(UTC)
