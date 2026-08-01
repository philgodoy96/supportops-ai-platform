"""Provider-independent prediction for classification evaluation cases."""

from supportops.ai.gateway.contracts import (
    LLMOperation,
    LLMRequest,
    LLMTokenUsage,
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
from supportops.ai.pricing.estimation import estimate_llm_cost
from supportops.ai.prompts.ticket_classification_v1 import (
    TICKET_CLASSIFICATION_PROMPT_VERSION,
    render_ticket_classification_prompt,
)
from supportops.ai.schemas.ticket_classification import (
    TicketClassificationResult,
)
from supportops.evaluation.ticket_classification.models import (
    TicketClassificationEvaluationCase,
)
from supportops.evaluation.ticket_classification.predictions import (
    TicketClassificationEvaluationPrediction,
    TicketClassificationFailedPrediction,
    TicketClassificationPredictionCost,
    TicketClassificationPredictionInvocation,
    TicketClassificationPredictionProvenance,
    TicketClassificationPredictionUsage,
    TicketClassificationSuccessfulPrediction,
)


class TicketClassificationEvaluationPredictor:
    """Execute one evaluation case through the application LLM Gateway."""

    def __init__(
        self,
        *,
        gateway: LLMGateway,
        provider_name: str,
        model: str,
        request_timeout_seconds: float,
        pricing_catalog: PricingCatalog = DEFAULT_PRICING_CATALOG,
    ) -> None:
        _validate_required_text(
            provider_name,
            field_name="provider_name",
        )
        _validate_required_text(
            model,
            field_name="model",
        )

        if request_timeout_seconds <= 0:
            raise ValueError(
                "request_timeout_seconds must be positive.",
            )

        self._gateway = gateway
        self._provider_name = provider_name
        self._model = model
        self._request_timeout_seconds = request_timeout_seconds
        self._pricing_catalog = pricing_catalog

    async def predict(
        self,
        *,
        case: TicketClassificationEvaluationCase,
        dataset_id: str,
        dataset_version: int,
    ) -> TicketClassificationEvaluationPrediction:
        """Produce one successful or failed normalized prediction."""

        _validate_required_text(
            dataset_id,
            field_name="dataset_id",
        )

        if dataset_version <= 0:
            raise ValueError(
                "dataset_version must be positive.",
            )

        rendered_prompt = render_ticket_classification_prompt(
            version=TICKET_CLASSIFICATION_PROMPT_VERSION,
            subject=case.ticket.subject,
            description=case.ticket.description,
        )
        definition = rendered_prompt.definition

        provenance = TicketClassificationPredictionProvenance(
            prompt_id=definition.prompt_id,
            prompt_version=definition.version,
            prompt_content_hash=definition.content_hash,
            provider=self._provider_name,
            model=self._model,
        )
        request = LLMRequest(
            operation=LLMOperation.TICKET_CLASSIFICATION,
            model=self._model,
            instructions=rendered_prompt.instructions,
            input=rendered_prompt.input,
            output_schema=TicketClassificationResult,
            timeout_seconds=self._request_timeout_seconds,
            metadata={
                "supportops_evaluation_dataset_id": dataset_id,
                "supportops_evaluation_dataset_version": str(
                    dataset_version,
                ),
                "supportops_evaluation_case_id": case.case_id,
                "supportops_prompt_id": definition.prompt_id,
                "supportops_prompt_version": str(
                    definition.version,
                ),
                "supportops_prompt_content_hash": (definition.content_hash),
                "supportops_schema_version": (definition.output_schema_id),
            },
        )

        try:
            gateway_result = await self._gateway.generate(
                request,
            )
        except LLMGatewayFailure as failure:
            return TicketClassificationFailedPrediction(
                case_id=case.case_id,
                status="failed",
                error_code=failure.error_code,
                provenance=provenance,
                invocations=self._materialize_invocations(
                    failure.invocations,
                ),
            )

        output = _require_ticket_classification_output(
            gateway_result,
        )

        return TicketClassificationSuccessfulPrediction(
            case_id=case.case_id,
            status="succeeded",
            provenance=provenance,
            output=output,
            invocations=self._materialize_invocations(
                gateway_result.invocations,
            ),
        )

    def _materialize_invocations(
        self,
        traces: tuple[LLMInvocationTrace, ...],
    ) -> tuple[
        TicketClassificationPredictionInvocation,
        ...,
    ]:
        return tuple(self._materialize_invocation(trace) for trace in traces)

    def _materialize_invocation(
        self,
        trace: LLMInvocationTrace,
    ) -> TicketClassificationPredictionInvocation:
        cost = estimate_llm_cost(
            provider=trace.provider,
            model=trace.model,
            usage=trace.usage,
            catalog=self._pricing_catalog,
        )

        return TicketClassificationPredictionInvocation(
            invocation_sequence=trace.invocation_sequence,
            status=trace.status,
            provider=trace.provider,
            model=trace.model,
            usage=_prediction_usage(trace.usage),
            cost=TicketClassificationPredictionCost(
                pricing_catalog_version=(cost.pricing_catalog_version),
                pricing_found=cost.pricing_found,
                estimated_input_cost_usd=(cost.estimated_input_cost_usd),
                estimated_cached_input_cost_usd=(cost.estimated_cached_input_cost_usd),
                estimated_output_cost_usd=(cost.estimated_output_cost_usd),
                estimated_total_cost_usd=(cost.estimated_total_cost_usd),
            ),
            latency_ms=trace.latency_ms,
            error_code=trace.error_code,
        )


def _prediction_usage(
    usage: LLMTokenUsage | None,
) -> TicketClassificationPredictionUsage | None:
    if usage is None:
        return None

    return TicketClassificationPredictionUsage(
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        total_tokens=usage.total_tokens,
    )


def _require_ticket_classification_output(
    result: LLMGatewayResult,
) -> TicketClassificationResult:
    if not isinstance(
        result.output,
        TicketClassificationResult,
    ):
        raise RuntimeError(
            "The evaluation Gateway returned an unexpected output schema.",
        )

    return result.output


def _validate_required_text(
    value: str,
    *,
    field_name: str,
) -> None:
    if not value:
        raise ValueError(
            f"{field_name} is required.",
        )

    if value != value.strip():
        raise ValueError(
            f"{field_name} must not contain surrounding whitespace.",
        )
