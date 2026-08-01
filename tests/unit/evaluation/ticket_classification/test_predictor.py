"""Unit tests for provider-independent evaluation prediction."""

from datetime import UTC, datetime
from decimal import Decimal

from supportops.ai.gateway.contracts import LLMTokenUsage
from supportops.ai.gateway.service import LLMGateway
from supportops.ai.providers.mock import (
    MockLLMOutcome,
    MockLLMProvider,
)
from supportops.evaluation.ticket_classification.models import (
    TicketClassificationEvaluationCase,
)
from supportops.evaluation.ticket_classification.predictions import (
    TicketClassificationFailedPrediction,
    TicketClassificationSuccessfulPrediction,
)
from supportops.evaluation.ticket_classification.predictor import (
    TicketClassificationEvaluationPredictor,
)

_NOW = datetime(
    2026,
    8,
    1,
    22,
    0,
    tzinfo=UTC,
)
_MODEL = "mock-ticket-classifier-v1"
_ZERO_COST = Decimal("0E-12")


def _case() -> TicketClassificationEvaluationCase:
    return TicketClassificationEvaluationCase.model_validate(
        {
            "case_id": "billing-duplicate-charge-001",
            "tags": [
                "billing",
                "individual-impact",
            ],
            "ticket": {
                "subject": "Duplicated invoice charge",
                "description": ("The latest invoice contains the same charge twice."),
            },
            "expected": {
                "category": "billing",
                "intent": "ask_question",
                "urgency": "normal",
                "sentiment": "neutral",
                "requires_human_review": False,
                "schema_version": ("ticket-classification-v1"),
            },
        },
    )


def _success_outcome() -> MockLLMOutcome:
    return MockLLMOutcome.success(
        {
            "category": "billing",
            "intent": "ask_question",
            "urgency": "normal",
            "sentiment": "neutral",
            "requires_human_review": False,
            "summary": ("The customer is asking about a duplicated charge."),
            "schema_version": ("ticket-classification-v1"),
        },
        usage=LLMTokenUsage(
            input_tokens=100,
            cached_input_tokens=0,
            output_tokens=20,
            reasoning_tokens=None,
            total_tokens=120,
        ),
    )


def _predictor(
    *,
    provider: MockLLMProvider,
    max_repair_attempts: int = 1,
) -> TicketClassificationEvaluationPredictor:
    return TicketClassificationEvaluationPredictor(
        gateway=LLMGateway(
            provider=provider,
            max_repair_attempts=max_repair_attempts,
        ),
        provider_name=provider.provider_name,
        model=provider.model,
        request_timeout_seconds=12,
    )


async def test_successful_prediction_preserves_provenance_usage_and_cost() -> None:
    provider = MockLLMProvider.with_strict_outcomes(
        (_success_outcome(),),
    )

    try:
        prediction = await _predictor(
            provider=provider,
        ).predict(
            case=_case(),
            dataset_id="ticket-classification-eval",
            dataset_version=1,
        )
    finally:
        await provider.close()

    assert isinstance(
        prediction,
        TicketClassificationSuccessfulPrediction,
    )
    assert prediction.case_id == ("billing-duplicate-charge-001")
    assert prediction.output.category.value == "billing"
    assert prediction.output.intent.value == "ask_question"
    assert prediction.provenance.prompt_id == ("ticket-classification")
    assert prediction.provenance.prompt_version == 1
    assert (
        len(
            prediction.provenance.prompt_content_hash,
        )
        == 64
    )
    assert prediction.provenance.provider == "mock"
    assert prediction.provenance.model == _MODEL

    assert len(prediction.invocations) == 1
    invocation = prediction.invocations[0]

    assert invocation.invocation_sequence == 1
    assert invocation.status.value == "succeeded"
    assert invocation.usage is not None
    assert invocation.usage.total_tokens == 120
    assert invocation.cost.pricing_found is True
    assert invocation.cost.estimated_input_cost_usd == _ZERO_COST
    assert invocation.cost.estimated_cached_input_cost_usd == _ZERO_COST
    assert invocation.cost.estimated_output_cost_usd == _ZERO_COST
    assert invocation.cost.estimated_total_cost_usd == _ZERO_COST

    payload = prediction.model_dump(
        mode="json",
    )

    assert "provider_request_id" not in str(payload)
    assert "raw_prompt" not in str(payload)
    assert "raw_response" not in str(payload)


async def test_gateway_failure_becomes_failed_prediction() -> None:
    provider = MockLLMProvider.with_strict_outcomes(
        (MockLLMOutcome.timeout(),),
    )

    try:
        prediction = await _predictor(
            provider=provider,
            max_repair_attempts=0,
        ).predict(
            case=_case(),
            dataset_id="ticket-classification-eval",
            dataset_version=1,
        )
    finally:
        await provider.close()

    assert isinstance(
        prediction,
        TicketClassificationFailedPrediction,
    )
    assert prediction.error_code.value == "llm_timeout"
    assert len(prediction.invocations) == 1

    invocation = prediction.invocations[0]

    assert invocation.status.value == "timed_out"
    assert invocation.error_code is not None
    assert invocation.error_code.value == "llm_timeout"
    assert invocation.usage is None
    assert invocation.cost.pricing_found is True
    assert invocation.cost.estimated_total_cost_usd is None


async def test_repair_history_is_preserved() -> None:
    provider = MockLLMProvider.with_strict_outcomes(
        (
            MockLLMOutcome.incomplete_response(),
            _success_outcome(),
        ),
    )

    try:
        prediction = await _predictor(
            provider=provider,
            max_repair_attempts=1,
        ).predict(
            case=_case(),
            dataset_id="ticket-classification-eval",
            dataset_version=1,
        )
    finally:
        await provider.close()

    assert isinstance(
        prediction,
        TicketClassificationSuccessfulPrediction,
    )
    assert provider.invocation_count == 2
    assert [invocation.invocation_sequence for invocation in prediction.invocations] == [
        1,
        2,
    ]
    assert [invocation.status.value for invocation in prediction.invocations] == [
        "incomplete",
        "succeeded",
    ]
    assert prediction.invocations[0].error_code is not None
    assert prediction.invocations[0].error_code.value == "llm_incomplete_response"
    assert prediction.invocations[1].error_code is None


async def test_unknown_model_pricing_remains_null() -> None:
    provider = MockLLMProvider.with_strict_outcomes(
        (_success_outcome(),),
        model="unpriced-evaluation-model",
    )

    try:
        prediction = await _predictor(
            provider=provider,
        ).predict(
            case=_case(),
            dataset_id="ticket-classification-eval",
            dataset_version=1,
        )
    finally:
        await provider.close()

    invocation = prediction.invocations[0]

    assert invocation.cost.pricing_found is False
    assert invocation.cost.estimated_input_cost_usd is None
    assert invocation.cost.estimated_cached_input_cost_usd is None
    assert invocation.cost.estimated_output_cost_usd is None
    assert invocation.cost.estimated_total_cost_usd is None


def test_predictor_requires_positive_timeout() -> None:
    provider = MockLLMProvider()

    try:
        try:
            TicketClassificationEvaluationPredictor(
                gateway=LLMGateway(
                    provider=provider,
                    max_repair_attempts=1,
                ),
                provider_name="mock",
                model=_MODEL,
                request_timeout_seconds=0,
            )
        except ValueError as error:
            assert str(error) == ("request_timeout_seconds must be positive.")
        else:
            raise AssertionError(
                "Expected timeout validation failure.",
            )
    finally:
        # This test is synchronous; provider close is verified
        # by asynchronous runtime tests in the CLI step.
        assert _NOW.tzinfo is UTC
