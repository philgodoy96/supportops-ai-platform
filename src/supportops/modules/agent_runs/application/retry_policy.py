"""Bounded retry policy for durable AgentRun execution."""

from dataclasses import dataclass

from supportops.modules.agent_runs.domain.recovery import (
    bounded_exponential_retry_delay_seconds,
)


@dataclass(frozen=True, slots=True)
class AgentRunRetryPolicy:
    """Calculate bounded retry eligibility and scheduling delays."""

    base_delay_seconds: float
    maximum_delay_seconds: float

    def __post_init__(self) -> None:
        if self.base_delay_seconds <= 0:
            raise ValueError(
                "base_delay_seconds must be greater than zero.",
            )

        if self.maximum_delay_seconds <= 0:
            raise ValueError(
                "maximum_delay_seconds must be greater than zero.",
            )

        if self.maximum_delay_seconds < self.base_delay_seconds:
            raise ValueError(
                "maximum_delay_seconds must not be smaller than base_delay_seconds.",
            )

    def can_retry_after_failure(
        self,
        *,
        retryable_failure_count: int,
        max_retryable_failures: int,
    ) -> bool:
        """Return whether another claim remains after counting this failure."""

        _validate_failure_budget(
            retryable_failure_count=retryable_failure_count,
            max_retryable_failures=max_retryable_failures,
        )
        return (retryable_failure_count + 1) < max_retryable_failures

    def delay_seconds_for_failure(
        self,
        *,
        failure_number: int,
    ) -> float:
        """Return the bounded delay after a retryable failure ordinal."""

        return bounded_exponential_retry_delay_seconds(
            failure_number=failure_number,
            base_delay_seconds=self.base_delay_seconds,
            maximum_delay_seconds=self.maximum_delay_seconds,
        )


def _validate_failure_budget(
    *,
    retryable_failure_count: int,
    max_retryable_failures: int,
) -> None:
    if retryable_failure_count < 0:
        raise ValueError("retryable_failure_count must not be negative.")

    if max_retryable_failures < 1:
        raise ValueError("max_retryable_failures must be at least one.")

    if retryable_failure_count > max_retryable_failures:
        raise ValueError(
            "retryable_failure_count must not exceed max_retryable_failures.",
        )
