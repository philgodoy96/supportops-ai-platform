"""Application-owned results and traces produced by the LLM Gateway."""

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from supportops.ai.gateway.contracts import LLMTokenUsage
from supportops.ai.gateway.errors import (
    LLMError,
    LLMErrorCode,
)


class LLMInvocationStatus(StrEnum):
    """Persistent status vocabulary for one logical provider invocation."""

    SUCCEEDED = "succeeded"
    REFUSED = "refused"
    INCOMPLETE = "incomplete"
    VALIDATION_FAILED = "validation_failed"
    PROVIDER_FAILED = "provider_failed"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True, slots=True)
class LLMInvocationTrace:
    """Safe metadata describing one logical provider invocation."""

    invocation_sequence: int
    status: LLMInvocationStatus
    provider: str
    model: str
    provider_request_id: str | None
    usage: LLMTokenUsage | None
    latency_ms: int
    error_code: LLMErrorCode | None

    def __post_init__(self) -> None:
        if self.invocation_sequence <= 0:
            raise ValueError(
                "invocation_sequence must be positive.",
            )

        _validate_required_text(
            self.provider,
            field_name="provider",
        )
        _validate_required_text(
            self.model,
            field_name="model",
        )
        _validate_optional_text(
            self.provider_request_id,
            field_name="provider_request_id",
        )

        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative.")

        if self.status is LLMInvocationStatus.SUCCEEDED:
            if self.error_code is not None:
                raise ValueError(
                    "Successful invocations cannot define an error_code.",
                )
        elif self.error_code is None:
            raise ValueError(
                "Failed invocations require an error_code.",
            )


@dataclass(frozen=True, slots=True)
class LLMGatewayResult:
    """Accepted validated output and all logical invocation traces."""

    output: BaseModel
    invocations: tuple[LLMInvocationTrace, ...]
    accepted_invocation_sequence: int

    def __post_init__(self) -> None:
        if not self.invocations:
            raise ValueError(
                "A successful gateway result requires invocations.",
            )

        accepted_invocation = next(
            (
                invocation
                for invocation in self.invocations
                if invocation.invocation_sequence == self.accepted_invocation_sequence
            ),
            None,
        )

        if accepted_invocation is None:
            raise ValueError(
                "accepted_invocation_sequence must reference an invocation.",
            )

        if accepted_invocation.status is not LLMInvocationStatus.SUCCEEDED:
            raise ValueError(
                "The accepted invocation must have succeeded.",
            )


class LLMGatewayFailure(Exception):
    """Gateway failure preserving normalized error and invocation traces."""

    def __init__(
        self,
        *,
        error: LLMError,
        invocations: tuple[LLMInvocationTrace, ...],
    ) -> None:
        if not invocations:
            raise ValueError(
                "A gateway failure requires at least one invocation.",
            )

        self.error = error
        self.invocations = invocations

        super().__init__(error.safe_summary)

    @property
    def error_code(self) -> LLMErrorCode:
        """Return the stable operational failure code."""

        return self.error.error_code

    @property
    def retryable(self) -> bool:
        """Return whether the outer AgentRun may retry the operation."""

        return self.error.retryable

    @property
    def terminal(self) -> bool:
        """Return whether the failure is terminal after gateway handling."""

        return self.error.terminal

    @property
    def repairable(self) -> bool:
        """Return whether the underlying failure was repair-eligible."""

        return self.error.repairable


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


def _validate_optional_text(
    value: str | None,
    *,
    field_name: str,
) -> None:
    if value is not None:
        _validate_required_text(
            value,
            field_name=field_name,
        )
