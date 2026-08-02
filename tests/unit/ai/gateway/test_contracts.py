"""Unit tests for provider-independent LLM contracts."""

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest
from pydantic import BaseModel, ConfigDict

from supportops.ai.gateway.contracts import (
    LLMOperation,
    LLMProvider,
    LLMProviderResponse,
    LLMRequest,
    LLMTokenUsage,
)


class ExampleStructuredOutput(BaseModel):
    """Minimal structured output used by contract tests."""

    model_config = ConfigDict(extra="forbid")

    category: str


class StubLLMProvider:
    """Protocol-compatible asynchronous provider test double."""

    @property
    def provider_name(self) -> str:
        return "stub"

    async def generate(self, request: LLMRequest) -> LLMProviderResponse:
        return LLMProviderResponse(
            parsed_output={"category": "billing"},
            provider=self.provider_name,
            model=request.model,
            provider_request_id="stub-request-1",
            usage=LLMTokenUsage(
                input_tokens=10,
                output_tokens=4,
                total_tokens=14,
            ),
            finish_reason="completed",
        )

    async def close(self) -> None:
        return None


def _request() -> LLMRequest:
    return LLMRequest(
        operation=LLMOperation.TICKET_CLASSIFICATION,
        model="test-model",
        instructions="Classify the supplied support ticket.",
        input='{"subject":"Invoice question"}',
        output_schema=ExampleStructuredOutput,
        timeout_seconds=12,
        metadata={"correlation_id": "corr-1"},
    )


def test_support_recommendation_draft_operation_has_stable_value() -> None:
    assert LLMOperation.SUPPORT_RECOMMENDATION_DRAFT.value == "support_recommendation_draft"


def test_request_copies_and_freezes_metadata() -> None:
    metadata = {"correlation_id": "corr-1"}
    request = replace(_request(), metadata=metadata)

    metadata["correlation_id"] = "changed"

    assert request.metadata == {"correlation_id": "corr-1"}
    with pytest.raises(TypeError):
        cast(dict[str, str], request.metadata)["request_id"] = "request-1"


@pytest.mark.parametrize(
    "invalid_request",
    [
        lambda: replace(_request(), model=""),
        lambda: replace(
            _request(),
            instructions=" surrounding whitespace ",
        ),
        lambda: replace(_request(), input=""),
    ],
)
def test_request_rejects_invalid_required_text(
    invalid_request: Callable[[], LLMRequest],
) -> None:
    with pytest.raises(ValueError):
        invalid_request()


def test_request_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        replace(_request(), timeout_seconds=0)


def test_request_requires_a_pydantic_output_schema() -> None:
    class NotAPydanticModel:
        pass

    invalid_schema = cast(type[BaseModel], NotAPydanticModel)

    with pytest.raises(TypeError, match="Pydantic BaseModel"):
        replace(_request(), output_schema=invalid_schema)


def test_token_usage_preserves_unknown_values() -> None:
    usage = LLMTokenUsage(input_tokens=10)

    assert usage.input_tokens == 10
    assert usage.cached_input_tokens is None
    assert usage.output_tokens is None
    assert usage.reasoning_tokens is None
    assert usage.total_tokens is None


@pytest.mark.parametrize(
    "invalid_usage",
    [
        lambda: LLMTokenUsage(input_tokens=-1),
        lambda: LLMTokenUsage(cached_input_tokens=-1),
        lambda: LLMTokenUsage(output_tokens=-1),
        lambda: LLMTokenUsage(reasoning_tokens=-1),
        lambda: LLMTokenUsage(total_tokens=-1),
    ],
)
def test_token_usage_rejects_negative_values(
    invalid_usage: Callable[[], LLMTokenUsage],
) -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        invalid_usage()


def test_token_usage_rejects_inconsistent_totals() -> None:
    with pytest.raises(
        ValueError,
        match="total_tokens must equal input_tokens plus output_tokens",
    ):
        LLMTokenUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=16,
        )


def test_provider_response_copies_and_freezes_parsed_output() -> None:
    parsed_output: dict[str, object] = {"category": "billing"}
    response = LLMProviderResponse(
        parsed_output=parsed_output,
        provider="mock",
        model="mock-ticket-classifier-v1",
    )

    parsed_output["category"] = "security"

    assert response.parsed_output == {"category": "billing"}
    with pytest.raises(TypeError):
        cast(dict[str, object], response.parsed_output)["category"] = "security"


async def test_provider_protocol_supports_async_generation_and_close() -> None:
    provider: LLMProvider = StubLLMProvider()

    response = await provider.generate(_request())
    await provider.close()

    assert provider.provider_name == "stub"
    assert response.provider == "stub"
    assert response.model == "test-model"
    assert response.parsed_output == {"category": "billing"}
