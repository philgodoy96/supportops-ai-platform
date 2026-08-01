"""Unit tests for the application-owned LLM error taxonomy."""

from collections.abc import Callable

import pytest

from supportops.ai.gateway.errors import (
    LLMAuthenticationError,
    LLMError,
    LLMErrorCode,
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

LLM_ERROR_TYPES: tuple[type[LLMError], ...] = (
    LLMTimeoutError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMQuotaError,
    LLMInvalidRequestError,
    LLMProviderUnavailableError,
    LLMRefusalError,
    LLMIncompleteResponseError,
    LLMOutputValidationError,
    LLMUnexpectedProviderError,
)


@pytest.mark.parametrize(
    (
        "error_factory",
        "expected_code",
        "expected_retryable",
        "expected_terminal",
        "expected_repairable",
    ),
    [
        (
            LLMTimeoutError,
            LLMErrorCode.TIMEOUT,
            True,
            False,
            False,
        ),
        (
            LLMRateLimitError,
            LLMErrorCode.RATE_LIMITED,
            True,
            False,
            False,
        ),
        (
            LLMAuthenticationError,
            LLMErrorCode.AUTHENTICATION_FAILED,
            False,
            True,
            False,
        ),
        (
            LLMQuotaError,
            LLMErrorCode.QUOTA_EXHAUSTED,
            False,
            True,
            False,
        ),
        (
            LLMInvalidRequestError,
            LLMErrorCode.INVALID_REQUEST,
            False,
            True,
            False,
        ),
        (
            LLMProviderUnavailableError,
            LLMErrorCode.PROVIDER_UNAVAILABLE,
            True,
            False,
            False,
        ),
        (
            LLMRefusalError,
            LLMErrorCode.REFUSAL,
            False,
            True,
            False,
        ),
        (
            LLMIncompleteResponseError,
            LLMErrorCode.INCOMPLETE_RESPONSE,
            True,
            False,
            True,
        ),
        (
            LLMOutputValidationError,
            LLMErrorCode.OUTPUT_VALIDATION_FAILED,
            False,
            True,
            True,
        ),
        (
            LLMUnexpectedProviderError,
            LLMErrorCode.UNEXPECTED_PROVIDER_FAILURE,
            True,
            False,
            False,
        ),
    ],
)
def test_error_classification_is_explicit(
    error_factory: Callable[..., LLMError],
    expected_code: LLMErrorCode,
    expected_retryable: bool,
    expected_terminal: bool,
    expected_repairable: bool,
) -> None:
    error = error_factory()

    assert error.error_code is expected_code
    assert error.retryable is expected_retryable
    assert error.terminal is expected_terminal
    assert error.repairable is expected_repairable


@pytest.mark.parametrize("error_type", LLM_ERROR_TYPES)
def test_error_is_exactly_retryable_or_terminal(
    error_type: type[LLMError],
) -> None:
    error = error_type()

    assert error.retryable is not error.terminal


@pytest.mark.parametrize("error_type", LLM_ERROR_TYPES)
def test_error_string_contains_only_the_safe_summary(
    error_type: type[LLMError],
) -> None:
    error = error_type(provider_request_id="provider-request-1")

    assert str(error) == error.safe_summary
    assert "provider-request-1" not in str(error)


def test_error_preserves_provider_request_identifier() -> None:
    error = LLMProviderUnavailableError(
        provider_request_id="provider-request-1",
    )

    assert error.provider_request_id == "provider-request-1"


@pytest.mark.parametrize(
    "invalid_provider_request_id",
    [
        "",
        " provider-request-1",
        "provider-request-1 ",
    ],
)
def test_error_rejects_invalid_provider_request_identifier(
    invalid_provider_request_id: str,
) -> None:
    with pytest.raises(ValueError):
        LLMProviderUnavailableError(
            provider_request_id=invalid_provider_request_id,
        )


def test_refusal_is_not_repairable() -> None:
    error = LLMRefusalError()

    assert error.terminal is True
    assert error.retryable is False
    assert error.repairable is False


def test_output_validation_failure_is_repairable_but_terminal_after_exhaustion() -> None:
    error = LLMOutputValidationError()

    assert error.repairable is True
    assert error.terminal is True
    assert error.retryable is False


def test_incomplete_response_can_be_repaired_or_retried_operationally() -> None:
    error = LLMIncompleteResponseError()

    assert error.repairable is True
    assert error.retryable is True
    assert error.terminal is False
