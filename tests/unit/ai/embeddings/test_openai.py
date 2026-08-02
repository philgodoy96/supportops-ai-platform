"""Unit tests for the OpenAI embedding provider adapter."""

from collections.abc import Sequence
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

from supportops.ai.embeddings.contracts import (
    EmbeddingOperation,
    EmbeddingRequest,
)
from supportops.ai.embeddings.errors import (
    EmbeddingAuthenticationError,
    EmbeddingInvalidRequestError,
    EmbeddingInvalidResponseError,
    EmbeddingProviderUnavailableError,
    EmbeddingQuotaError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
    EmbeddingUnexpectedProviderError,
)
from supportops.ai.embeddings.openai import (
    OPENAI_EMBEDDING_PROVIDER_NAME,
    OpenAIEmbeddingProvider,
)

_MODEL = "text-embedding-3-small"
_DIMENSIONS = 3


def http_request() -> httpx.Request:
    """Create one embeddings API request fixture."""

    return httpx.Request(
        "POST",
        "https://api.openai.com/v1/embeddings",
    )


def http_response(
    status_code: int,
) -> httpx.Response:
    """Create one SDK-compatible HTTP response."""

    return httpx.Response(
        status_code,
        headers={
            "x-request-id": "req_embedding_error_1",
        },
        request=http_request(),
    )


@dataclass(frozen=True, slots=True)
class FakeEmbeddingItem:
    """One SDK-like embedding item."""

    index: int
    embedding: Sequence[float]


@dataclass(frozen=True, slots=True)
class FakeUsage:
    """SDK-like embedding usage."""

    prompt_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class FakeEmbeddingResponse:
    """SDK-like embeddings response."""

    data: tuple[FakeEmbeddingItem, ...]
    model: str = _MODEL
    usage: FakeUsage | None = None
    _request_id: str | None = "req_embedding_1"


class FakeEmbeddingsAPI:
    """Record embeddings calls and return one configured outcome."""

    def __init__(
        self,
        outcome: FakeEmbeddingResponse | Exception,
    ) -> None:
        self._outcome = outcome
        self.calls: list[dict[str, object]] = []

    async def create(
        self,
        **kwargs: object,
    ) -> FakeEmbeddingResponse:
        """Return or raise the configured outcome."""

        self.calls.append(dict(kwargs))

        if isinstance(self._outcome, Exception):
            raise self._outcome

        return self._outcome


class FakeAsyncOpenAI:
    """Minimal SDK-compatible async client."""

    def __init__(
        self,
        outcome: FakeEmbeddingResponse | Exception,
    ) -> None:
        self.embeddings = FakeEmbeddingsAPI(outcome)
        self.closed = False

    async def close(self) -> None:
        """Record client closure."""

        self.closed = True


def create_request(
    *,
    model: str = _MODEL,
    dimensions: int = _DIMENSIONS,
) -> EmbeddingRequest:
    """Create one ordered embedding request."""

    return EmbeddingRequest(
        operation=EmbeddingOperation.KNOWLEDGE_INDEXING,
        model=model,
        inputs=(
            "First authoritative chunk.",
            "Second authoritative chunk.",
        ),
        dimensions=dimensions,
        timeout_seconds=12,
        metadata={
            "workspace_id": "workspace-1",
        },
    )


def create_provider(
    outcome: FakeEmbeddingResponse | Exception,
) -> tuple[
    OpenAIEmbeddingProvider,
    FakeAsyncOpenAI,
]:
    """Create an adapter with one fake SDK client."""

    fake_client = FakeAsyncOpenAI(outcome)
    provider = OpenAIEmbeddingProvider(
        client=cast(AsyncOpenAI, fake_client),
        model=_MODEL,
        dimensions=_DIMENSIONS,
    )

    return provider, fake_client


async def test_constructs_ordered_embeddings_api_request() -> None:
    provider, fake_client = create_provider(
        FakeEmbeddingResponse(
            data=(
                FakeEmbeddingItem(
                    index=0,
                    embedding=(0.1, 0.2, 0.3),
                ),
                FakeEmbeddingItem(
                    index=1,
                    embedding=(0.4, 0.5, 0.6),
                ),
            )
        )
    )

    await provider.embed(create_request())

    assert fake_client.embeddings.calls == [
        {
            "input": [
                "First authoritative chunk.",
                "Second authoritative chunk.",
            ],
            "model": _MODEL,
            "dimensions": _DIMENSIONS,
            "encoding_format": "float",
            "timeout": 12,
        }
    ]


async def test_maps_vectors_usage_and_provenance() -> None:
    provider, _ = create_provider(
        FakeEmbeddingResponse(
            data=(
                FakeEmbeddingItem(
                    index=0,
                    embedding=(0.1, 0.2, 0.3),
                ),
                FakeEmbeddingItem(
                    index=1,
                    embedding=(0.4, 0.5, 0.6),
                ),
            ),
            usage=FakeUsage(
                prompt_tokens=18,
                total_tokens=18,
            ),
        )
    )

    response = await provider.embed(create_request())

    assert response.embeddings == (
        (0.1, 0.2, 0.3),
        (0.4, 0.5, 0.6),
    )
    assert response.provider == (OPENAI_EMBEDDING_PROVIDER_NAME)
    assert response.model == _MODEL
    assert response.dimensions == _DIMENSIONS
    assert response.provider_request_id == ("req_embedding_1")
    assert response.usage is not None
    assert response.usage.input_tokens == 18
    assert response.usage.total_tokens == 18


async def test_restores_input_order_from_provider_indexes() -> None:
    provider, _ = create_provider(
        FakeEmbeddingResponse(
            data=(
                FakeEmbeddingItem(
                    index=1,
                    embedding=(0.4, 0.5, 0.6),
                ),
                FakeEmbeddingItem(
                    index=0,
                    embedding=(0.1, 0.2, 0.3),
                ),
            )
        )
    )

    response = await provider.embed(create_request())

    assert response.embeddings == (
        (0.1, 0.2, 0.3),
        (0.4, 0.5, 0.6),
    )


async def test_preserves_unknown_usage_as_none() -> None:
    provider, _ = create_provider(
        FakeEmbeddingResponse(
            data=(
                FakeEmbeddingItem(
                    index=0,
                    embedding=(0.1, 0.2, 0.3),
                ),
                FakeEmbeddingItem(
                    index=1,
                    embedding=(0.4, 0.5, 0.6),
                ),
            ),
            usage=None,
        )
    )

    response = await provider.embed(create_request())

    assert response.usage is None


@pytest.mark.parametrize(
    "response",
    [
        FakeEmbeddingResponse(
            data=(
                FakeEmbeddingItem(
                    index=0,
                    embedding=(0.1, 0.2, 0.3),
                ),
            )
        ),
        FakeEmbeddingResponse(
            data=(
                FakeEmbeddingItem(
                    index=0,
                    embedding=(0.1, 0.2, 0.3),
                ),
                FakeEmbeddingItem(
                    index=0,
                    embedding=(0.4, 0.5, 0.6),
                ),
            )
        ),
        FakeEmbeddingResponse(
            data=(
                FakeEmbeddingItem(
                    index=0,
                    embedding=(0.1, 0.2),
                ),
                FakeEmbeddingItem(
                    index=1,
                    embedding=(0.4, 0.5, 0.6),
                ),
            )
        ),
        FakeEmbeddingResponse(
            data=(
                FakeEmbeddingItem(
                    index=0,
                    embedding=(0.1, 0.2, 0.3),
                ),
                FakeEmbeddingItem(
                    index=1,
                    embedding=(0.4, 0.5, 0.6),
                ),
            ),
            model="unexpected-model",
        ),
    ],
)
async def test_rejects_malformed_provider_response(
    response: FakeEmbeddingResponse,
) -> None:
    provider, _ = create_provider(response)

    with pytest.raises(EmbeddingInvalidResponseError) as captured:
        await provider.embed(create_request())

    assert captured.value.provider_request_id == ("req_embedding_1")


@pytest.mark.parametrize(
    ("model", "dimensions"),
    [
        ("other-model", _DIMENSIONS),
        (_MODEL, 64),
    ],
)
async def test_rejects_incompatible_request_without_sdk_call(
    model: str,
    dimensions: int,
) -> None:
    provider, fake_client = create_provider(FakeEmbeddingResponse(data=()))

    with pytest.raises(EmbeddingInvalidRequestError):
        await provider.embed(
            create_request(
                model=model,
                dimensions=dimensions,
            )
        )

    assert fake_client.embeddings.calls == []


async def test_close_is_idempotent_and_closes_sdk_client() -> None:
    provider, fake_client = create_provider(FakeEmbeddingResponse(data=()))

    await provider.close()
    await provider.close()

    assert fake_client.closed

    with pytest.raises(
        RuntimeError,
        match="provider is closed",
    ):
        await provider.embed(create_request())


@pytest.mark.parametrize(
    ("sdk_error", "expected_error"),
    [
        (
            APITimeoutError(
                request=http_request(),
            ),
            EmbeddingTimeoutError,
        ),
        (
            APIConnectionError(
                request=http_request(),
            ),
            EmbeddingProviderUnavailableError,
        ),
        (
            AuthenticationError(
                "Invalid credentials.",
                response=http_response(401),
                body={
                    "code": "invalid_api_key",
                    "type": "invalid_request_error",
                },
            ),
            EmbeddingAuthenticationError,
        ),
        (
            BadRequestError(
                "Invalid embedding model.",
                response=http_response(400),
                body={
                    "code": "model_not_found",
                    "type": "invalid_request_error",
                },
            ),
            EmbeddingInvalidRequestError,
        ),
        (
            ConflictError(
                "Temporary conflict.",
                response=http_response(409),
                body=None,
            ),
            EmbeddingProviderUnavailableError,
        ),
        (
            InternalServerError(
                "Provider failure.",
                response=http_response(503),
                body=None,
            ),
            EmbeddingProviderUnavailableError,
        ),
        (
            OpenAIError("Unexpected SDK failure."),
            EmbeddingUnexpectedProviderError,
        ),
    ],
)
async def test_normalizes_sdk_failures(
    sdk_error: Exception,
    expected_error: type[Exception],
) -> None:
    provider, _ = create_provider(sdk_error)

    with pytest.raises(expected_error):
        await provider.embed(create_request())


async def test_maps_temporary_rate_limit() -> None:
    provider, _ = create_provider(
        RateLimitError(
            "Rate limit reached.",
            response=http_response(429),
            body={
                "code": "rate_limit_exceeded",
                "type": "requests",
            },
        )
    )

    with pytest.raises(EmbeddingRateLimitError) as captured:
        await provider.embed(create_request())

    assert captured.value.provider_request_id == ("req_embedding_error_1")


async def test_maps_quota_exhaustion() -> None:
    provider, _ = create_provider(
        RateLimitError(
            "Quota exhausted.",
            response=http_response(429),
            body={
                "code": "project_spend_limit_exceeded",
                "type": "insufficient_quota",
            },
        )
    )

    with pytest.raises(EmbeddingQuotaError):
        await provider.embed(create_request())


async def test_maps_unclassified_server_status_to_unavailable() -> None:
    provider, _ = create_provider(
        APIStatusError(
            "Server failure.",
            response=http_response(502),
            body=None,
        )
    )

    with pytest.raises(EmbeddingProviderUnavailableError):
        await provider.embed(create_request())


async def test_maps_sdk_response_validation_failure() -> None:
    provider, _ = create_provider(
        APIResponseValidationError(
            response=http_response(200),
            body={
                "unexpected": "payload",
            },
        )
    )

    with pytest.raises(EmbeddingInvalidResponseError):
        await provider.embed(create_request())


@pytest.mark.parametrize(
    (
        "api_key",
        "model",
        "dimensions",
        "timeout_seconds",
        "transport_max_retries",
        "expected_message",
    ),
    [
        (
            "",
            _MODEL,
            1536,
            12,
            1,
            "api_key is required",
        ),
        (
            "test-key",
            "",
            1536,
            12,
            1,
            "model is required",
        ),
        (
            "test-key",
            _MODEL,
            0,
            12,
            1,
            "dimensions must be positive",
        ),
        (
            "test-key",
            _MODEL,
            1536,
            0,
            1,
            "timeout_seconds must be positive",
        ),
        (
            "test-key",
            _MODEL,
            1536,
            12,
            -1,
            "transport_max_retries must be non-negative",
        ),
    ],
)
def test_factory_rejects_invalid_configuration(
    api_key: str,
    model: str,
    dimensions: int,
    timeout_seconds: float,
    transport_max_retries: int,
    expected_message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        OpenAIEmbeddingProvider.create(
            api_key=api_key,
            model=model,
            dimensions=dimensions,
            timeout_seconds=timeout_seconds,
            transport_max_retries=transport_max_retries,
        )
