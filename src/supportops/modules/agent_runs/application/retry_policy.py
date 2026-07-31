"""Bounded retry policy for durable AgentRun execution."""

from dataclasses import dataclass


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

    def can_retry(
        self,
        *,
        attempt_count: int,
        max_attempts: int,
    ) -> bool:
        """Return whether another execution attempt may be started."""

        _validate_attempt_budget(
            attempt_count=attempt_count,
            max_attempts=max_attempts,
        )
        return attempt_count < max_attempts

    def delay_seconds_for_attempt(
        self,
        *,
        attempt_number: int,
    ) -> float:
        """Return the bounded delay after a failed attempt."""

        if attempt_number < 1:
            raise ValueError("attempt_number must be at least one.")

        exponential_delay = self.base_delay_seconds * (2.0 ** (attempt_number - 1))
        return min(
            exponential_delay,
            self.maximum_delay_seconds,
        )


def _validate_attempt_budget(
    *,
    attempt_count: int,
    max_attempts: int,
) -> None:
    if attempt_count < 0:
        raise ValueError("attempt_count must not be negative.")

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one.")

    if attempt_count > max_attempts:
        raise ValueError(
            "attempt_count must not exceed max_attempts.",
        )
