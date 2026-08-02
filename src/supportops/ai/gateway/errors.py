"""Application-owned error taxonomy for LLM provider failures."""

from enum import StrEnum
from typing import ClassVar


class LLMErrorCode(StrEnum):
    """Stable operational codes for normalized LLM failures."""

    TIMEOUT = "llm_timeout"
    RATE_LIMITED = "llm_rate_limited"
    AUTHENTICATION_FAILED = "llm_authentication_failed"
    QUOTA_EXHAUSTED = "llm_quota_exhausted"
    INVALID_REQUEST = "llm_invalid_request"
    PROVIDER_UNAVAILABLE = "llm_provider_unavailable"
    REFUSAL = "llm_refusal"
    INCOMPLETE_RESPONSE = "llm_incomplete_response"
    OUTPUT_VALIDATION_FAILED = "llm_output_validation_failed"
    TOOL_DECISION_VALIDATION_FAILED = "llm_tool_decision_validation_failed"
    UNEXPECTED_PROVIDER_FAILURE = "llm_unexpected_provider_failure"


class LLMError(Exception):
    """Base class for safe provider-independent LLM failures."""

    error_code: ClassVar[LLMErrorCode]
    safe_summary: ClassVar[str]
    retryable: ClassVar[bool]
    terminal: ClassVar[bool]
    repairable: ClassVar[bool]

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


class LLMTimeoutError(LLMError):
    """Provider request exceeded its configured timeout."""

    error_code = LLMErrorCode.TIMEOUT
    safe_summary = "The LLM provider request exceeded its configured timeout."
    retryable = True
    terminal = False
    repairable = False


class LLMRateLimitError(LLMError):
    """Provider temporarily rejected the request because of rate limiting."""

    error_code = LLMErrorCode.RATE_LIMITED
    safe_summary = "The LLM provider temporarily rate-limited the request."
    retryable = True
    terminal = False
    repairable = False


class LLMAuthenticationError(LLMError):
    """Provider credentials were rejected."""

    error_code = LLMErrorCode.AUTHENTICATION_FAILED
    safe_summary = "The LLM provider rejected the configured credentials."
    retryable = False
    terminal = True
    repairable = False


class LLMQuotaError(LLMError):
    """Provider account quota or billing capacity was exhausted."""

    error_code = LLMErrorCode.QUOTA_EXHAUSTED
    safe_summary = "The LLM provider account has insufficient quota."
    retryable = False
    terminal = True
    repairable = False


class LLMInvalidRequestError(LLMError):
    """Provider rejected the request contract or configured model."""

    error_code = LLMErrorCode.INVALID_REQUEST
    safe_summary = "The LLM provider rejected the request as invalid."
    retryable = False
    terminal = True
    repairable = False


class LLMProviderUnavailableError(LLMError):
    """Provider service was temporarily unavailable."""

    error_code = LLMErrorCode.PROVIDER_UNAVAILABLE
    safe_summary = "The LLM provider is temporarily unavailable."
    retryable = True
    terminal = False
    repairable = False


class LLMRefusalError(LLMError):
    """Provider explicitly refused to produce the requested output."""

    error_code = LLMErrorCode.REFUSAL
    safe_summary = "The LLM provider refused to produce the requested output."
    retryable = False
    terminal = True
    repairable = False


class LLMIncompleteResponseError(LLMError):
    """Provider returned an incomplete structured response."""

    error_code = LLMErrorCode.INCOMPLETE_RESPONSE
    safe_summary = "The LLM provider returned an incomplete response."
    retryable = True
    terminal = False
    repairable = True


class LLMOutputValidationError(LLMError):
    """Structured output failed application-owned validation."""

    error_code = LLMErrorCode.OUTPUT_VALIDATION_FAILED
    safe_summary = "The LLM output failed application validation."
    retryable = False
    terminal = True
    repairable = True


class LLMToolDecisionValidationError(LLMError):
    """Provider tool decision failed application-owned validation."""

    error_code = LLMErrorCode.TOOL_DECISION_VALIDATION_FAILED
    safe_summary = "The LLM tool decision failed application validation."
    retryable = False
    terminal = True
    repairable = False


class LLMUnexpectedProviderError(LLMError):
    """Provider failed in a way that could not be classified more precisely."""

    error_code = LLMErrorCode.UNEXPECTED_PROVIDER_FAILURE
    safe_summary = "The LLM provider failed unexpectedly."
    retryable = True
    terminal = False
    repairable = False


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
        raise ValueError(
            f"{field_name} must not contain surrounding whitespace.",
        )
