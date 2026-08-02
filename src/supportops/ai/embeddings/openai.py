"""OpenAI adapter for ordered text embedding batches."""

from collections.abc import Sequence
from typing import Protocol, Self, cast

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
    NotFoundError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from supportops.ai.embeddings.contracts import (
    EmbeddingProviderResponse,
    EmbeddingRequest,
    EmbeddingTokenUsage,
    EmbeddingVector,
)
from supportops.ai.embeddings.errors import (
    EmbeddingAuthenticationError,
    EmbeddingError,
    EmbeddingInvalidRequestError,
    EmbeddingInvalidResponseError,
    EmbeddingProviderUnavailableError,
    EmbeddingQuotaError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
    EmbeddingUnexpectedProviderError,
)

OPENAI_EMBEDDING_PROVIDER_NAME = "openai"

_QUOTA_ERROR_CODES = frozenset(
    {
        "credit_balance_exhausted",
        "organization_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
        "project_spend_limit_exceeded",
    }
)


class _OpenAIEmbeddingItem(Protocol):
    index: int
    embedding: Sequence[float]


class _OpenAIEmbeddingUsage(Protocol):
    prompt_tokens: int
    total_tokens: int


class _OpenAIEmbeddingResponse(Protocol):
    _request_id: str | None
    data: Sequence[_OpenAIEmbeddingItem]
    model: str
    usage: _OpenAIEmbeddingUsage | None


class OpenAIEmbeddingProvider:
    """Isolate the official OpenAI SDK from indexing modules."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        dimensions: int,
    ) -> None:
        _validate_required_text(
            model,
            field_name="model",
        )
        if dimensions <= 0:
            raise ValueError("dimensions must be positive.")

        self._client = client
        self._model = model
        self._dimensions = dimensions
        self._closed = False

    @classmethod
    def create(
        cls,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        timeout_seconds: float,
        transport_max_retries: int,
        base_url: str | None = None,
    ) -> Self:
        """Create one process-scoped OpenAI SDK client."""

        _validate_required_text(
            api_key,
            field_name="api_key",
        )
        _validate_required_text(
            model,
            field_name="model",
        )
        if dimensions <= 0:
            raise ValueError("dimensions must be positive.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if transport_max_retries < 0:
            raise ValueError("transport_max_retries must be non-negative.")
        if base_url is not None:
            _validate_required_text(
                base_url,
                field_name="base_url",
            )

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=transport_max_retries,
            timeout=timeout_seconds,
        )

        return cls(
            client=client,
            model=model,
            dimensions=dimensions,
        )

    @property
    def provider_name(self) -> str:
        """Return the stable provider identity."""

        return OPENAI_EMBEDDING_PROVIDER_NAME

    @property
    def model(self) -> str:
        """Return the configured embedding model."""

        return self._model

    @property
    def dimensions(self) -> int:
        """Return the configured embedding dimensions."""

        return self._dimensions

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingProviderResponse:
        """Embed one ordered text batch through the OpenAI API."""

        if self._closed:
            raise RuntimeError("OpenAI embedding provider is closed.")
        if request.model != self._model or request.dimensions != self._dimensions:
            raise EmbeddingInvalidRequestError()

        try:
            response = await self._client.embeddings.create(
                input=list(request.inputs),
                model=request.model,
                dimensions=request.dimensions,
                encoding_format="float",
                timeout=request.timeout_seconds,
            )
        except APITimeoutError as error:
            raise EmbeddingTimeoutError() from error
        except AuthenticationError as error:
            raise EmbeddingAuthenticationError(
                provider_request_id=error.request_id,
            ) from error
        except RateLimitError as error:
            if _is_quota_error(error):
                raise EmbeddingQuotaError(
                    provider_request_id=error.request_id,
                ) from error
            raise EmbeddingRateLimitError(
                provider_request_id=error.request_id,
            ) from error
        except (
            BadRequestError,
            NotFoundError,
            PermissionDeniedError,
            UnprocessableEntityError,
        ) as error:
            raise EmbeddingInvalidRequestError(
                provider_request_id=error.request_id,
            ) from error
        except (
            ConflictError,
            InternalServerError,
        ) as error:
            raise EmbeddingProviderUnavailableError(
                provider_request_id=error.request_id,
            ) from error
        except APIConnectionError as error:
            raise EmbeddingProviderUnavailableError() from error
        except APIResponseValidationError as error:
            raise EmbeddingInvalidResponseError() from error
        except APIStatusError as error:
            raise _normalize_unclassified_status_error(error) from error
        except OpenAIError as error:
            raise EmbeddingUnexpectedProviderError() from error

        parsed_response = cast(
            _OpenAIEmbeddingResponse,
            response,
        )
        provider_request_id = parsed_response._request_id

        if parsed_response.model != request.model:
            raise EmbeddingInvalidResponseError(
                provider_request_id=provider_request_id,
            )

        embeddings = _ordered_embeddings(
            parsed_response.data,
            expected_count=len(request.inputs),
            provider_request_id=provider_request_id,
        )

        try:
            return EmbeddingProviderResponse(
                embeddings=embeddings,
                provider=self.provider_name,
                model=parsed_response.model,
                dimensions=request.dimensions,
                usage=_map_usage(parsed_response.usage),
                provider_request_id=provider_request_id,
            )
        except (
            TypeError,
            ValueError,
        ) as error:
            raise EmbeddingInvalidResponseError(
                provider_request_id=provider_request_id,
            ) from error

    async def close(self) -> None:
        """Close the process-scoped SDK client idempotently."""

        if self._closed:
            return

        self._closed = True
        await self._client.close()


def _ordered_embeddings(
    items: Sequence[_OpenAIEmbeddingItem],
    *,
    expected_count: int,
    provider_request_id: str | None,
) -> tuple[EmbeddingVector, ...]:
    items_by_index: dict[int, _OpenAIEmbeddingItem] = {}

    for item in items:
        if (
            isinstance(item.index, bool)
            or not isinstance(item.index, int)
            or item.index < 0
            or item.index >= expected_count
            or item.index in items_by_index
        ):
            raise EmbeddingInvalidResponseError(
                provider_request_id=provider_request_id,
            )

        items_by_index[item.index] = item

    if set(items_by_index) != set(range(expected_count)):
        raise EmbeddingInvalidResponseError(
            provider_request_id=provider_request_id,
        )

    return tuple(tuple(items_by_index[index].embedding) for index in range(expected_count))


def _map_usage(
    usage: _OpenAIEmbeddingUsage | None,
) -> EmbeddingTokenUsage | None:
    if usage is None:
        return None

    return EmbeddingTokenUsage(
        input_tokens=usage.prompt_tokens,
        total_tokens=usage.total_tokens,
    )


def _is_quota_error(
    error: RateLimitError,
) -> bool:
    return error.code in _QUOTA_ERROR_CODES or error.type == "insufficient_quota"


def _normalize_unclassified_status_error(
    error: APIStatusError,
) -> EmbeddingError:
    if error.status_code == 408:
        return EmbeddingTimeoutError(
            provider_request_id=error.request_id,
        )
    if error.status_code == 409:
        return EmbeddingProviderUnavailableError(
            provider_request_id=error.request_id,
        )
    if error.status_code >= 500:
        return EmbeddingProviderUnavailableError(
            provider_request_id=error.request_id,
        )

    return EmbeddingInvalidRequestError(
        provider_request_id=error.request_id,
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
