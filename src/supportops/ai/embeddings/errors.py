"""Application-owned error taxonomy for embedding provider failures."""

from enum import StrEnum
from typing import ClassVar


class EmbeddingErrorCode(StrEnum):
    """Stable operational codes for normalized embedding failures."""

    TIMEOUT = "embedding_timeout"
    RATE_LIMITED = "embedding_rate_limited"
    AUTHENTICATION_FAILED = "embedding_authentication_failed"
    QUOTA_EXHAUSTED = "embedding_quota_exhausted"
    INVALID_REQUEST = "embedding_invalid_request"
    PROVIDER_UNAVAILABLE = "embedding_provider_unavailable"
    INVALID_RESPONSE = "embedding_invalid_response"
    UNEXPECTED_PROVIDER_FAILURE = "embedding_unexpected_provider_failure"


class EmbeddingError(Exception):
    """Base class for safe provider-independent embedding failures."""

    error_code: ClassVar[EmbeddingErrorCode]
    safe_summary: ClassVar[str]
    retryable: ClassVar[bool]
    terminal: ClassVar[bool]

    def __init__(
        self,
        *,
        provider_request_id: str | None = None,
    ) -> None:
        _validate_optional_identifier(
            provider_request_id,
            field_name="provider_request_id",
        )
        self.provider_request_id = provider_request_id
        super().__init__(self.safe_summary)


class EmbeddingTimeoutError(EmbeddingError):
    """Provider request exceeded its configured timeout."""

    error_code = EmbeddingErrorCode.TIMEOUT
    safe_summary = "The embedding provider request exceeded its configured timeout."
    retryable = True
    terminal = False


class EmbeddingRateLimitError(EmbeddingError):
    """Provider temporarily rejected the request because of rate limiting."""

    error_code = EmbeddingErrorCode.RATE_LIMITED
    safe_summary = "The embedding provider temporarily rate-limited the request."
    retryable = True
    terminal = False


class EmbeddingAuthenticationError(EmbeddingError):
    """Provider credentials were rejected."""

    error_code = EmbeddingErrorCode.AUTHENTICATION_FAILED
    safe_summary = "The embedding provider rejected the configured credentials."
    retryable = False
    terminal = True


class EmbeddingQuotaError(EmbeddingError):
    """Provider account quota or billing capacity was exhausted."""

    error_code = EmbeddingErrorCode.QUOTA_EXHAUSTED
    safe_summary = "The embedding provider account has insufficient quota."
    retryable = False
    terminal = True


class EmbeddingInvalidRequestError(EmbeddingError):
    """Provider rejected the request or configured embedding profile."""

    error_code = EmbeddingErrorCode.INVALID_REQUEST
    safe_summary = "The embedding provider rejected the request as invalid."
    retryable = False
    terminal = True


class EmbeddingProviderUnavailableError(EmbeddingError):
    """Provider service was temporarily unavailable."""

    error_code = EmbeddingErrorCode.PROVIDER_UNAVAILABLE
    safe_summary = "The embedding provider is temporarily unavailable."
    retryable = True
    terminal = False


class EmbeddingInvalidResponseError(EmbeddingError):
    """Provider response violated the application-owned contract."""

    error_code = EmbeddingErrorCode.INVALID_RESPONSE
    safe_summary = "The embedding provider returned an invalid response."
    retryable = True
    terminal = False


class EmbeddingUnexpectedProviderError(EmbeddingError):
    """Provider failed in a way that could not be classified safely."""

    error_code = EmbeddingErrorCode.UNEXPECTED_PROVIDER_FAILURE
    safe_summary = "The embedding provider failed unexpectedly."
    retryable = True
    terminal = False


def _validate_optional_identifier(
    value: str | None,
    *,
    field_name: str,
) -> None:
    if value is None:
        return

    if not value:
        raise ValueError(f"{field_name} must not be empty.")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")
