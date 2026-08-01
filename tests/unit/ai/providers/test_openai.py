"""Unit tests for the OpenAI Responses API provider adapter."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    OpenAIError,
    RateLimitError,
)
from pydantic import ValidationError

from supportops.ai.gateway.contracts import (
    LLMOperation,
    LLMRequest,
)
from supportops.ai.gateway.errors import (
    LLMAuthenticationError,
    LLMIncompleteResponseError,
    LLMInvalidRequestError,
    LLMOutputValidationError,
    LLMProviderUnavailableError,
    LLMQuotaError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
    LLMUnexpectedProviderError,
)
from supportops.ai.providers.openai import (
    OPENAI_LLM_PROVIDER_NAME,
    OpenAILLMProvider,
)
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
    TicketClassificationResult,
)

_MODEL = "gpt-5-nano"


def _http_request() -> httpx.Request:
    return httpx.Request(
        "POST",
        "https://api.openai.com/v1/responses",
    )


def _http_response(
    status_code: int,
) -> httpx.Response:
    return httpx.Response(
        status_code,
        headers={
            "x-request-id": "req_error_1",
        },
        request=_http_request(),
    )


@dataclass(frozen=True, slots=True)
class FakeInputTokenDetails:
    cached_tokens: int


@dataclass(frozen=True, slots=True)
class FakeOutputTokenDetails:
    reasoning_tokens: int


@dataclass(frozen=True, slots=True)
class FakeUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: FakeInputTokenDetails | None = None
    output_tokens_details: FakeOutputTokenDetails | None = None


@dataclass(frozen=True, slots=True)
class FakeContent:
    type: str


@dataclass(frozen=True, slots=True)
class FakeMessage:
    type: str = "message"
    content: tuple[FakeContent, ...] = ()


@dataclass(frozen=True, slots=True)
class FakeParsedResponse:
    output_parsed: object | None
    status: str | None = "completed"
    usage: FakeUsage | None = None
    output: tuple[FakeMessage, ...] = ()
    _request_id: str | None = "req_openai_1"


class FakeResponsesAPI:
    def __init__(
        self,
        outcome: FakeParsedResponse | Exception,
    ) -> None:
        self._outcome = outcome
        self.calls: list[dict[str, object]] = []

    async def parse(
        self,
        **kwargs: object,
    ) -> FakeParsedResponse:
        self.calls.append(dict(kwargs))

        if isinstance(self._outcome, Exception):
            raise self._outcome

        return self._outcome


class FakeAsyncOpenAI:
    def __init__(
        self,
        outcome: FakeParsedResponse | Exception,
    ) -> None:
        self.responses = FakeResponsesAPI(outcome)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _classification() -> TicketClassificationResult:
    return TicketClassificationResult.model_validate(
        {
            "category": "billing",
            "intent": "ask_question",
            "urgency": "normal",
            "sentiment": "neutral",
            "requires_human_review": False,
            "summary": "The customer is asking about an invoice charge.",
            "schema_version": TICKET_CLASSIFICATION_SCHEMA_VERSION,
        },
    )


def _request(
    *,
    model: str = _MODEL,
    metadata: Mapping[str, str] | None = None,
) -> LLMRequest:
    return LLMRequest(
        operation=LLMOperation.TICKET_CLASSIFICATION,
        model=model,
        instructions="Classify the supplied support ticket.",
        input='{"subject":"Invoice question"}',
        output_schema=TicketClassificationResult,
        timeout_seconds=12,
        metadata={} if metadata is None else metadata,
    )


def _provider(
    outcome: FakeParsedResponse | Exception,
) -> tuple[OpenAILLMProvider, FakeAsyncOpenAI]:
    fake_client = FakeAsyncOpenAI(outcome)
    provider = OpenAILLMProvider(
        client=cast(AsyncOpenAI, fake_client),
        model=_MODEL,
    )

    return provider, fake_client


async def test_constructs_responses_api_structured_output_request() -> None:
    provider, fake_client = _provider(
        FakeParsedResponse(
            output_parsed=_classification(),
        ),
    )

    await provider.generate(
        _request(
            metadata={
                "correlation_id": "corr-1",
                "prompt_id": "ticket-classification",
            },
        ),
    )

    assert fake_client.responses.calls == [
        {
            "model": _MODEL,
            "instructions": ("Classify the supplied support ticket."),
            "input": '{"subject":"Invoice question"}',
            "text_format": TicketClassificationResult,
            "metadata": {
                "correlation_id": "corr-1",
                "prompt_id": "ticket-classification",
            },
            "store": False,
            "timeout": 12,
        },
    ]


async def test_maps_successful_response_and_usage() -> None:
    provider, _ = _provider(
        FakeParsedResponse(
            output_parsed=_classification(),
            usage=FakeUsage(
                input_tokens=100,
                input_tokens_details=FakeInputTokenDetails(
                    cached_tokens=40,
                ),
                output_tokens=25,
                output_tokens_details=FakeOutputTokenDetails(
                    reasoning_tokens=5,
                ),
                total_tokens=125,
            ),
        ),
    )

    response = await provider.generate(_request())

    assert response.provider == OPENAI_LLM_PROVIDER_NAME
    assert response.model == _MODEL
    assert response.provider_request_id == "req_openai_1"
    assert response.finish_reason == "completed"
    assert response.parsed_output["category"] == "billing"
    assert response.usage is not None
    assert response.usage.input_tokens == 100
    assert response.usage.cached_input_tokens == 40
    assert response.usage.output_tokens == 25
    assert response.usage.reasoning_tokens == 5
    assert response.usage.total_tokens == 125


async def test_preserves_unknown_usage_as_none() -> None:
    provider, _ = _provider(
        FakeParsedResponse(
            output_parsed=_classification(),
            usage=None,
        ),
    )

    response = await provider.generate(_request())

    assert response.usage is None


async def test_detects_explicit_refusal_before_accepting_output() -> None:
    provider, _ = _provider(
        FakeParsedResponse(
            output_parsed=None,
            output=(
                FakeMessage(
                    content=(FakeContent(type="refusal"),),
                ),
            ),
        ),
    )

    with pytest.raises(LLMRefusalError) as captured:
        await provider.generate(_request())

    assert captured.value.provider_request_id == "req_openai_1"


async def test_detects_incomplete_response_status() -> None:
    provider, _ = _provider(
        FakeParsedResponse(
            output_parsed=None,
            status="incomplete",
        ),
    )

    with pytest.raises(
        LLMIncompleteResponseError,
    ) as captured:
        await provider.generate(_request())

    assert captured.value.provider_request_id == "req_openai_1"


async def test_completed_response_requires_parsed_output() -> None:
    provider, _ = _provider(
        FakeParsedResponse(
            output_parsed=None,
        ),
    )

    with pytest.raises(LLMIncompleteResponseError):
        await provider.generate(_request())


async def test_rejects_non_pydantic_parsed_output() -> None:
    provider, _ = _provider(
        FakeParsedResponse(
            output_parsed={
                "category": "billing",
            },
        ),
    )

    with pytest.raises(LLMUnexpectedProviderError):
        await provider.generate(_request())


async def test_rejects_request_for_another_model_without_calling_sdk() -> None:
    provider, fake_client = _provider(
        FakeParsedResponse(
            output_parsed=_classification(),
        ),
    )

    with pytest.raises(LLMInvalidRequestError):
        await provider.generate(
            _request(model="another-model"),
        )

    assert fake_client.responses.calls == []


async def test_close_is_idempotent_and_closes_sdk_client() -> None:
    provider, fake_client = _provider(
        FakeParsedResponse(
            output_parsed=_classification(),
        ),
    )

    await provider.close()
    await provider.close()

    assert fake_client.closed is True

    with pytest.raises(RuntimeError, match="provider is closed"):
        await provider.generate(_request())


@pytest.mark.parametrize(
    ("sdk_error", "expected_error"),
    [
        (
            APITimeoutError(
                request=_http_request(),
            ),
            LLMTimeoutError,
        ),
        (
            APIConnectionError(
                request=_http_request(),
            ),
            LLMProviderUnavailableError,
        ),
        (
            AuthenticationError(
                "Invalid credentials.",
                response=_http_response(401),
                body={
                    "code": "invalid_api_key",
                    "type": "invalid_request_error",
                },
            ),
            LLMAuthenticationError,
        ),
        (
            BadRequestError(
                "Invalid model.",
                response=_http_response(400),
                body={
                    "code": "model_not_found",
                    "type": "invalid_request_error",
                },
            ),
            LLMInvalidRequestError,
        ),
        (
            ConflictError(
                "Temporary conflict.",
                response=_http_response(409),
                body=None,
            ),
            LLMProviderUnavailableError,
        ),
        (
            InternalServerError(
                "Provider failure.",
                response=_http_response(503),
                body=None,
            ),
            LLMProviderUnavailableError,
        ),
        (
            OpenAIError("Unexpected SDK failure."),
            LLMUnexpectedProviderError,
        ),
    ],
)
async def test_normalizes_sdk_failures(
    sdk_error: Exception,
    expected_error: type[Exception],
) -> None:
    provider, _ = _provider(sdk_error)

    with pytest.raises(expected_error):
        await provider.generate(_request())


async def test_maps_temporary_rate_limit_to_retryable_error() -> None:
    provider, _ = _provider(
        RateLimitError(
            "Rate limit reached.",
            response=_http_response(429),
            body={
                "code": "rate_limit_exceeded",
                "type": "requests",
            },
        ),
    )

    with pytest.raises(LLMRateLimitError) as captured:
        await provider.generate(_request())

    assert captured.value.provider_request_id == "req_error_1"


@pytest.mark.parametrize(
    "quota_body",
    [
        {
            "code": "credit_balance_exhausted",
            "type": "insufficient_quota",
        },
        {
            "code": "organization_spend_limit_exceeded",
            "type": "insufficient_quota",
        },
        {
            "code": "project_spend_limit_exceeded",
            "type": "insufficient_quota",
        },
        {
            "code": "organization_usage_limit_exceeded",
            "type": "insufficient_quota",
        },
    ],
)
async def test_maps_quota_rate_limit_to_terminal_error(
    quota_body: dict[str, str],
) -> None:
    provider, _ = _provider(
        RateLimitError(
            "Quota exhausted.",
            response=_http_response(429),
            body=quota_body,
        ),
    )

    with pytest.raises(LLMQuotaError) as captured:
        await provider.generate(_request())

    assert captured.value.provider_request_id == "req_error_1"


async def test_maps_unclassified_server_status_to_unavailable() -> None:
    provider, _ = _provider(
        APIStatusError(
            "Server failure.",
            response=_http_response(502),
            body=None,
        ),
    )

    with pytest.raises(LLMProviderUnavailableError):
        await provider.generate(_request())


async def test_maps_unclassified_client_status_to_invalid_request() -> None:
    provider, _ = _provider(
        APIStatusError(
            "Client failure.",
            response=_http_response(418),
            body=None,
        ),
    )

    with pytest.raises(LLMInvalidRequestError):
        await provider.generate(_request())


async def test_maps_sdk_response_validation_failure() -> None:
    response = _http_response(200)
    provider, _ = _provider(
        APIResponseValidationError(
            response=response,
            body={
                "unexpected": "payload",
            },
        ),
    )

    with pytest.raises(LLMUnexpectedProviderError):
        await provider.generate(_request())


async def test_maps_structured_output_validation_failure() -> None:
    validation_error = _classification_validation_error()
    provider, _ = _provider(validation_error)

    with pytest.raises(LLMOutputValidationError):
        await provider.generate(_request())


@pytest.mark.parametrize(
    (
        "api_key",
        "timeout_seconds",
        "transport_max_retries",
        "expected_message",
    ),
    [
        (
            "",
            12,
            1,
            "api_key is required",
        ),
        (
            "test-key",
            0,
            1,
            "timeout_seconds must be positive",
        ),
        (
            "test-key",
            12,
            -1,
            "transport_max_retries must be non-negative",
        ),
    ],
)
def test_factory_rejects_invalid_configuration(
    api_key: str,
    timeout_seconds: float,
    transport_max_retries: int,
    expected_message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        OpenAILLMProvider.create(
            api_key=api_key,
            model=_MODEL,
            timeout_seconds=timeout_seconds,
            transport_max_retries=transport_max_retries,
        )


def _classification_validation_error() -> ValidationError:
    try:
        TicketClassificationResult.model_validate(
            {
                "category": "invented-category",
            },
        )
    except ValidationError as error:
        return error

    raise AssertionError(
        "Expected invalid classification payload to fail validation.",
    )
