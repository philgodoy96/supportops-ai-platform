"""OpenAI Responses API adapter for structured LLM generation."""

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
from pydantic import BaseModel, ValidationError

from supportops.ai.gateway.contracts import (
    LLMProviderResponse,
    LLMRequest,
    LLMTokenUsage,
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

OPENAI_LLM_PROVIDER_NAME = "openai"

_QUOTA_ERROR_CODES = frozenset(
    {
        "credit_balance_exhausted",
        "organization_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
        "project_spend_limit_exceeded",
    },
)


class _OpenAIInputTokenDetails(Protocol):
    cached_tokens: int


class _OpenAIOutputTokenDetails(Protocol):
    reasoning_tokens: int


class _OpenAIUsage(Protocol):
    input_tokens: int
    input_tokens_details: _OpenAIInputTokenDetails | None
    output_tokens: int
    output_tokens_details: _OpenAIOutputTokenDetails | None
    total_tokens: int


class _OpenAIResponseContent(Protocol):
    type: str


class _OpenAIResponseMessage(Protocol):
    type: str
    content: Sequence[_OpenAIResponseContent]


class _ParsedOpenAIResponse(Protocol):
    _request_id: str | None
    output: Sequence[_OpenAIResponseMessage]
    output_parsed: object | None
    status: str | None
    usage: _OpenAIUsage | None


class OpenAILLMProvider:
    """OpenAI adapter that isolates the official SDK from business modules."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
    ) -> None:
        _validate_required_text(model, field_name="model")

        self._client = client
        self._model = model
        self._closed = False

    @classmethod
    def create(
        cls,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        transport_max_retries: int,
        base_url: str | None = None,
    ) -> Self:
        """Create one process-scoped OpenAI provider and SDK client."""

        _validate_required_text(api_key, field_name="api_key")
        _validate_required_text(model, field_name="model")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        if transport_max_retries < 0:
            raise ValueError(
                "transport_max_retries must be non-negative.",
            )

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
        )

    @property
    def provider_name(self) -> str:
        """Return the stable provider identifier used for provenance."""

        return OPENAI_LLM_PROVIDER_NAME

    @property
    def model(self) -> str:
        """Return the deployment-configured model identifier."""

        return self._model

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMProviderResponse:
        """Generate one structured result through the Responses API."""

        if self._closed:
            raise RuntimeError("OpenAI LLM provider is closed.")

        if request.model != self._model:
            raise LLMInvalidRequestError()

        try:
            response = await self._client.responses.parse(
                model=request.model,
                instructions=request.instructions,
                input=request.input,
                text_format=request.output_schema,
                metadata=dict(request.metadata),
                store=False,
                timeout=request.timeout_seconds,
            )
        except APITimeoutError as error:
            raise LLMTimeoutError() from error
        except AuthenticationError as error:
            raise LLMAuthenticationError(
                provider_request_id=error.request_id,
            ) from error
        except RateLimitError as error:
            if _is_quota_error(error):
                raise LLMQuotaError(
                    provider_request_id=error.request_id,
                ) from error

            raise LLMRateLimitError(
                provider_request_id=error.request_id,
            ) from error
        except (
            BadRequestError,
            NotFoundError,
            PermissionDeniedError,
            UnprocessableEntityError,
        ) as error:
            raise LLMInvalidRequestError(
                provider_request_id=error.request_id,
            ) from error
        except ConflictError as error:
            raise LLMProviderUnavailableError(
                provider_request_id=error.request_id,
            ) from error
        except InternalServerError as error:
            raise LLMProviderUnavailableError(
                provider_request_id=error.request_id,
            ) from error
        except APIConnectionError as error:
            raise LLMProviderUnavailableError() from error
        except APIResponseValidationError as error:
            raise LLMUnexpectedProviderError() from error
        except APIStatusError as error:
            raise _normalize_unclassified_status_error(error) from error
        except ValidationError as error:
            raise LLMOutputValidationError() from error
        except OpenAIError as error:
            raise LLMUnexpectedProviderError() from error

        parsed_response = cast(
            _ParsedOpenAIResponse,
            response,
        )

        provider_request_id = parsed_response._request_id

        if _contains_refusal(parsed_response):
            raise LLMRefusalError(
                provider_request_id=provider_request_id,
            )

        if parsed_response.status == "incomplete":
            raise LLMIncompleteResponseError(
                provider_request_id=provider_request_id,
            )

        if parsed_response.status != "completed":
            raise LLMUnexpectedProviderError(
                provider_request_id=provider_request_id,
            )

        parsed_output = parsed_response.output_parsed

        if parsed_output is None:
            raise LLMIncompleteResponseError(
                provider_request_id=provider_request_id,
            )

        if not isinstance(parsed_output, BaseModel):
            raise LLMUnexpectedProviderError(
                provider_request_id=provider_request_id,
            )

        return LLMProviderResponse(
            parsed_output=parsed_output.model_dump(
                mode="python",
            ),
            provider=self.provider_name,
            model=request.model,
            provider_request_id=provider_request_id,
            usage=_map_usage(parsed_response.usage),
            finish_reason=parsed_response.status,
        )

    async def close(self) -> None:
        """Close the process-scoped OpenAI SDK client idempotently."""

        if self._closed:
            return

        self._closed = True
        await self._client.close()


def _contains_refusal(
    response: _ParsedOpenAIResponse,
) -> bool:
    for output in response.output:
        if output.type != "message":
            continue

        for content in output.content:
            if content.type == "refusal":
                return True

    return False


def _map_usage(
    usage: _OpenAIUsage | None,
) -> LLMTokenUsage | None:
    if usage is None:
        return None

    cached_input_tokens = None
    if usage.input_tokens_details is not None:
        cached_input_tokens = usage.input_tokens_details.cached_tokens

    reasoning_tokens = None
    if usage.output_tokens_details is not None:
        reasoning_tokens = usage.output_tokens_details.reasoning_tokens

    return LLMTokenUsage(
        input_tokens=usage.input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=usage.total_tokens,
    )


def _is_quota_error(
    error: RateLimitError,
) -> bool:
    return error.code in _QUOTA_ERROR_CODES or error.type == "insufficient_quota"


def _normalize_unclassified_status_error(
    error: APIStatusError,
) -> Exception:
    if error.status_code == 408:
        return LLMTimeoutError(
            provider_request_id=error.request_id,
        )

    if error.status_code == 409:
        return LLMProviderUnavailableError(
            provider_request_id=error.request_id,
        )

    if error.status_code >= 500:
        return LLMProviderUnavailableError(
            provider_request_id=error.request_id,
        )

    return LLMInvalidRequestError(
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
        raise ValueError(
            f"{field_name} must not contain surrounding whitespace.",
        )
