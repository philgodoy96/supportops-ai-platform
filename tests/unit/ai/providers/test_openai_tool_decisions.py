"""Unit tests for OpenAI Responses API native tool decisions."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAIError,
    RateLimitError,
)
from pydantic import JsonValue

from supportops.agent_tools.domain.contracts import (
    ProviderToolDefinition,
)
from supportops.ai.gateway.contracts import (
    LLMOperation,
)
from supportops.ai.gateway.errors import (
    LLMAuthenticationError,
    LLMIncompleteResponseError,
    LLMInvalidRequestError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
    LLMUnexpectedProviderError,
)
from supportops.ai.gateway.tool_decisions import (
    COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME,
    LLMProviderToolDecisionRequest,
)
from supportops.ai.providers.openai import (
    OPENAI_LLM_PROVIDER_NAME,
    OpenAILLMProvider,
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
    """Synthetic OpenAI input-token details."""

    cached_tokens: int


@dataclass(frozen=True, slots=True)
class FakeOutputTokenDetails:
    """Synthetic OpenAI output-token details."""

    reasoning_tokens: int


@dataclass(frozen=True, slots=True)
class FakeUsage:
    """Synthetic OpenAI response usage."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: FakeInputTokenDetails | None = None
    output_tokens_details: FakeOutputTokenDetails | None = None


@dataclass(frozen=True, slots=True)
class FakeContent:
    """Synthetic message content item."""

    type: str


@dataclass(frozen=True, slots=True)
class FakeMessage:
    """Synthetic Responses API message output."""

    type: str = "message"
    content: tuple[FakeContent, ...] = ()


@dataclass(frozen=True, slots=True)
class FakeFunctionCall:
    """Synthetic Responses API function-call output."""

    call_id: str
    name: str
    arguments: str
    type: str = "function_call"


@dataclass(frozen=True, slots=True)
class FakeOtherOutput:
    """Synthetic non-function Responses API output."""

    type: str


@dataclass(frozen=True, slots=True)
class FakeToolDecisionResponse:
    """Synthetic Responses API tool-decision response."""

    output: tuple[object, ...]
    status: str | None = "completed"
    usage: FakeUsage | None = None
    _request_id: str | None = "req_openai_tool_1"


class FakeResponsesAPI:
    """Record Responses API create calls without network access."""

    def __init__(
        self,
        outcome: FakeToolDecisionResponse | Exception,
    ) -> None:
        self._outcome = outcome
        self.calls: list[dict[str, object]] = []

    async def create(
        self,
        **kwargs: object,
    ) -> FakeToolDecisionResponse:
        """Return one configured SDK result."""

        self.calls.append(dict(kwargs))

        if isinstance(self._outcome, Exception):
            raise self._outcome

        return self._outcome


class FakeAsyncOpenAI:
    """Minimal process-scoped AsyncOpenAI test double."""

    def __init__(
        self,
        outcome: FakeToolDecisionResponse | Exception,
    ) -> None:
        self.responses = FakeResponsesAPI(outcome)
        self.closed = False
        self.close_calls = 0

    async def close(self) -> None:
        """Record SDK client cleanup."""

        self.close_calls += 1
        self.closed = True


def _json_schema(
    value: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    return dict(value)


def _search_knowledge_definition() -> ProviderToolDefinition:
    return ProviderToolDefinition(
        name="search_knowledge",
        version=1,
        description=("Search active workspace-scoped runbook evidence."),
        input_schema=_json_schema(
            {
                "type": "object",
                "properties": {
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "document_ids": {
                        "anyOf": [
                            {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "format": "uuid",
                                },
                            },
                            {
                                "type": "null",
                            },
                        ],
                    },
                },
                "required": [
                    "top_k",
                    "document_ids",
                ],
                "additionalProperties": False,
            }
        ),
        strict=True,
    )


def _terminal_control_definition() -> ProviderToolDefinition:
    return ProviderToolDefinition(
        name=COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME,
        version=1,
        description=("Complete controlled support analysis without executing an external action."),
        input_schema=_json_schema(
            {
                "type": "object",
                "properties": {
                    "recommended_action": {
                        "type": "string",
                        "enum": [
                            "respond",
                            "request_more_information",
                            "recommend_escalation",
                        ],
                    },
                    "evidence_sufficient": {
                        "type": "boolean",
                    },
                    "requires_human_review": {
                        "type": "boolean",
                    },
                    "decision_summary": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500,
                    },
                },
                "required": [
                    "recommended_action",
                    "evidence_sufficient",
                    "requires_human_review",
                    "decision_summary",
                ],
                "additionalProperties": False,
            }
        ),
        strict=True,
    )


def _request(
    *,
    model: str = _MODEL,
    metadata: Mapping[str, str] | None = None,
) -> LLMProviderToolDecisionRequest:
    return LLMProviderToolDecisionRequest(
        operation=LLMOperation.SUPPORT_ACTION_DECISION,
        model=model,
        instructions=("Select exactly one approved support function."),
        input=(
            '{"ticket":{"subject":"Cannot reset access"},"classification":{"category":"how_to"}}'
        ),
        functions=(
            _search_knowledge_definition(),
            _terminal_control_definition(),
        ),
        timeout_seconds=20,
        metadata={} if metadata is None else metadata,
        tool_choice="required",
        parallel_tool_calls=False,
    )


def _provider(
    outcome: FakeToolDecisionResponse | Exception,
) -> tuple[OpenAILLMProvider, FakeAsyncOpenAI]:
    fake_client = FakeAsyncOpenAI(outcome)
    provider = OpenAILLMProvider(
        client=cast(AsyncOpenAI, fake_client),
        model=_MODEL,
    )

    return provider, fake_client


def _function_response(
    *,
    name: str = "search_knowledge",
    arguments: str = ('{"top_k":5,"document_ids":null}'),
    call_id: str = "call_openai_1",
    usage: FakeUsage | None = None,
) -> FakeToolDecisionResponse:
    return FakeToolDecisionResponse(
        output=(
            FakeFunctionCall(
                call_id=call_id,
                name=name,
                arguments=arguments,
            ),
        ),
        usage=usage,
    )


async def test_constructs_strict_responses_api_tool_request() -> None:
    provider, fake_client = _provider(_function_response())

    await provider.decide(
        _request(
            metadata={
                "correlation_id": "corr-1",
                "prompt_id": "support-action-decision",
            }
        )
    )

    assert fake_client.responses.calls == [
        {
            "model": _MODEL,
            "instructions": ("Select exactly one approved support function."),
            "input": (
                '{"ticket":{"subject":"Cannot reset access"},'
                '"classification":{"category":"how_to"}}'
            ),
            "tools": [
                {
                    "type": "function",
                    "name": "search_knowledge",
                    "description": ("Search active workspace-scoped runbook evidence."),
                    "parameters": (_search_knowledge_definition().input_schema),
                    "strict": True,
                },
                {
                    "type": "function",
                    "name": (COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME),
                    "description": (
                        "Complete controlled support analysis without executing an external action."
                    ),
                    "parameters": (_terminal_control_definition().input_schema),
                    "strict": True,
                },
            ],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "metadata": {
                "correlation_id": "corr-1",
                "prompt_id": "support-action-decision",
            },
            "store": False,
            "timeout": 20,
        }
    ]


async def test_maps_single_function_call_and_usage() -> None:
    provider, _ = _provider(
        _function_response(
            usage=FakeUsage(
                input_tokens=80,
                input_tokens_details=FakeInputTokenDetails(cached_tokens=20),
                output_tokens=15,
                output_tokens_details=FakeOutputTokenDetails(reasoning_tokens=4),
                total_tokens=95,
            )
        )
    )

    response = await provider.decide(_request())

    assert response.provider_tool_call_id == "call_openai_1"
    assert response.function_name == "search_knowledge"
    assert response.arguments_json == ('{"top_k":5,"document_ids":null}')
    assert response.provider == OPENAI_LLM_PROVIDER_NAME
    assert response.model == _MODEL
    assert response.provider_request_id == ("req_openai_tool_1")
    assert response.finish_reason == "completed"
    assert response.usage is not None
    assert response.usage.input_tokens == 80
    assert response.usage.cached_input_tokens == 20
    assert response.usage.output_tokens == 15
    assert response.usage.reasoning_tokens == 4
    assert response.usage.total_tokens == 95


async def test_maps_terminal_control_as_normal_function_call() -> None:
    arguments = (
        "{"
        '"recommended_action":"respond",'
        '"evidence_sufficient":true,'
        '"requires_human_review":false,'
        '"decision_summary":"Runbook evidence is available."'
        "}"
    )
    provider, _ = _provider(
        _function_response(
            name=COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME,
            arguments=arguments,
        )
    )

    response = await provider.decide(_request())

    assert response.function_name == (COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME)
    assert response.arguments_json == arguments


@pytest.mark.parametrize(
    ("function_name", "arguments"),
    [
        (
            "unknown_tool",
            '{"workspace_id":"untrusted-workspace"}',
        ),
        (
            "search_knowledge",
            "{malformed-json",
        ),
    ],
)
async def test_preserves_untrusted_function_output_for_gateway(
    function_name: str,
    arguments: str,
) -> None:
    provider, _ = _provider(
        _function_response(
            name=function_name,
            arguments=arguments,
        )
    )

    response = await provider.decide(_request())

    assert response.function_name == function_name
    assert response.arguments_json == arguments


async def test_ignores_non_function_outputs_when_one_call_exists() -> None:
    provider, _ = _provider(
        FakeToolDecisionResponse(
            output=(
                FakeOtherOutput(type="reasoning"),
                FakeFunctionCall(
                    call_id="call_openai_1",
                    name="search_knowledge",
                    arguments=('{"top_k":5,"document_ids":null}'),
                ),
            )
        )
    )

    response = await provider.decide(_request())

    assert response.function_name == "search_knowledge"


async def test_rejects_response_without_function_call() -> None:
    provider, _ = _provider(FakeToolDecisionResponse(output=(FakeOtherOutput(type="reasoning"),)))

    with pytest.raises(LLMIncompleteResponseError) as exc_info:
        await provider.decide(_request())

    assert exc_info.value.provider_request_id == ("req_openai_tool_1")


async def test_rejects_multiple_function_calls() -> None:
    provider, _ = _provider(
        FakeToolDecisionResponse(
            output=(
                FakeFunctionCall(
                    call_id="call_openai_1",
                    name="search_knowledge",
                    arguments=('{"top_k":5,"document_ids":null}'),
                ),
                FakeFunctionCall(
                    call_id="call_openai_2",
                    name=(COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME),
                    arguments=(
                        "{"
                        '"recommended_action":"respond",'
                        '"evidence_sufficient":true,'
                        '"requires_human_review":false,'
                        '"decision_summary":"Complete."'
                        "}"
                    ),
                ),
            )
        )
    )

    with pytest.raises(LLMUnexpectedProviderError) as exc_info:
        await provider.decide(_request())

    assert exc_info.value.provider_request_id == ("req_openai_tool_1")


@pytest.mark.parametrize(
    ("call_id", "name", "arguments"),
    [
        (
            "",
            "search_knowledge",
            '{"top_k":5,"document_ids":null}',
        ),
        (
            "call_openai_1",
            "",
            '{"top_k":5,"document_ids":null}',
        ),
        (
            "call_openai_1",
            " search_knowledge ",
            '{"top_k":5,"document_ids":null}',
        ),
        (
            "call_openai_1",
            "search_knowledge",
            "",
        ),
    ],
)
async def test_rejects_structurally_invalid_function_call(
    call_id: str,
    name: str,
    arguments: str,
) -> None:
    provider, _ = _provider(
        _function_response(
            call_id=call_id,
            name=name,
            arguments=arguments,
        )
    )

    with pytest.raises(LLMUnexpectedProviderError) as exc_info:
        await provider.decide(_request())

    assert exc_info.value.provider_request_id == ("req_openai_tool_1")


async def test_detects_explicit_refusal() -> None:
    provider, _ = _provider(
        FakeToolDecisionResponse(output=(FakeMessage(content=(FakeContent(type="refusal"),)),))
    )

    with pytest.raises(LLMRefusalError) as exc_info:
        await provider.decide(_request())

    assert exc_info.value.provider_request_id == ("req_openai_tool_1")


async def test_detects_incomplete_response_status() -> None:
    provider, _ = _provider(
        FakeToolDecisionResponse(
            output=(),
            status="incomplete",
        )
    )

    with pytest.raises(LLMIncompleteResponseError) as exc_info:
        await provider.decide(_request())

    assert exc_info.value.provider_request_id == ("req_openai_tool_1")


async def test_rejects_unexpected_response_status() -> None:
    provider, _ = _provider(
        FakeToolDecisionResponse(
            output=(),
            status="failed",
        )
    )

    with pytest.raises(LLMUnexpectedProviderError) as exc_info:
        await provider.decide(_request())

    assert exc_info.value.provider_request_id == ("req_openai_tool_1")


async def test_rejects_another_model_without_sdk_call() -> None:
    provider, fake_client = _provider(_function_response())

    with pytest.raises(LLMInvalidRequestError):
        await provider.decide(_request(model="another-model"))

    assert fake_client.responses.calls == []


async def test_close_prevents_tool_decisions() -> None:
    provider, fake_client = _provider(_function_response())

    await provider.close()
    await provider.close()

    assert fake_client.closed is True
    assert fake_client.close_calls == 1

    with pytest.raises(
        RuntimeError,
        match="provider is closed",
    ):
        await provider.decide(_request())

    assert fake_client.responses.calls == []


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
                "Invalid tool schema.",
                response=_http_response(400),
                body={
                    "code": "invalid_request",
                    "type": "invalid_request_error",
                },
            ),
            LLMInvalidRequestError,
        ),
        (
            InternalServerError(
                "Provider unavailable.",
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
        await provider.decide(_request())


async def test_maps_rate_limit_to_retryable_error() -> None:
    provider, _ = _provider(
        RateLimitError(
            "Rate limit reached.",
            response=_http_response(429),
            body={
                "code": "rate_limit_exceeded",
                "type": "requests",
            },
        )
    )

    with pytest.raises(LLMRateLimitError) as exc_info:
        await provider.decide(_request())

    assert exc_info.value.provider_request_id == ("req_error_1")
