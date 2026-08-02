"""Unit tests for normalized embedding provider errors."""

import pytest

from supportops.ai.embeddings.errors import (
    EmbeddingAuthenticationError,
    EmbeddingError,
    EmbeddingErrorCode,
    EmbeddingInvalidRequestError,
    EmbeddingInvalidResponseError,
    EmbeddingProviderUnavailableError,
    EmbeddingQuotaError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
    EmbeddingUnexpectedProviderError,
)


@pytest.mark.parametrize(
    (
        "error",
        "error_code",
        "retryable",
        "terminal",
    ),
    [
        (
            EmbeddingTimeoutError(),
            EmbeddingErrorCode.TIMEOUT,
            True,
            False,
        ),
        (
            EmbeddingRateLimitError(),
            EmbeddingErrorCode.RATE_LIMITED,
            True,
            False,
        ),
        (
            EmbeddingAuthenticationError(),
            EmbeddingErrorCode.AUTHENTICATION_FAILED,
            False,
            True,
        ),
        (
            EmbeddingQuotaError(),
            EmbeddingErrorCode.QUOTA_EXHAUSTED,
            False,
            True,
        ),
        (
            EmbeddingInvalidRequestError(),
            EmbeddingErrorCode.INVALID_REQUEST,
            False,
            True,
        ),
        (
            EmbeddingProviderUnavailableError(),
            EmbeddingErrorCode.PROVIDER_UNAVAILABLE,
            True,
            False,
        ),
        (
            EmbeddingInvalidResponseError(),
            EmbeddingErrorCode.INVALID_RESPONSE,
            True,
            False,
        ),
        (
            EmbeddingUnexpectedProviderError(),
            EmbeddingErrorCode.UNEXPECTED_PROVIDER_FAILURE,
            True,
            False,
        ),
    ],
)
def test_error_taxonomy_exposes_stable_operational_semantics(
    error: EmbeddingError,
    error_code: EmbeddingErrorCode,
    retryable: bool,
    terminal: bool,
) -> None:
    assert error.error_code is error_code
    assert error.retryable is retryable
    assert error.terminal is terminal
    assert str(error) == error.safe_summary


def test_error_preserves_provider_request_id() -> None:
    error = EmbeddingRateLimitError(provider_request_id="req_embedding_error_1")

    assert error.provider_request_id == ("req_embedding_error_1")


@pytest.mark.parametrize(
    "provider_request_id",
    [
        "",
        " request-id",
        "request-id ",
    ],
)
def test_error_rejects_invalid_provider_request_id(
    provider_request_id: str,
) -> None:
    with pytest.raises(ValueError):
        EmbeddingTimeoutError(provider_request_id=provider_request_id)
