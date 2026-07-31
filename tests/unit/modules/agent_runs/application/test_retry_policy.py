"""Unit tests for the AgentRun retry policy."""

from re import escape

import pytest

from supportops.modules.agent_runs.application.retry_policy import (
    AgentRunRetryPolicy,
)


def create_policy() -> AgentRunRetryPolicy:
    return AgentRunRetryPolicy(
        base_delay_seconds=2.0,
        maximum_delay_seconds=60.0,
    )


@pytest.mark.parametrize(
    ("attempt_number", "expected_delay"),
    [
        (1, 2.0),
        (2, 4.0),
        (3, 8.0),
        (4, 16.0),
        (5, 32.0),
    ],
)
def test_retry_delay_grows_exponentially(
    attempt_number: int,
    expected_delay: float,
) -> None:
    policy = create_policy()

    delay = policy.delay_seconds_for_attempt(
        attempt_number=attempt_number,
    )

    assert delay == expected_delay


@pytest.mark.parametrize(
    "attempt_number",
    [
        6,
        7,
        10,
        100,
    ],
)
def test_retry_delay_is_bounded_by_maximum(
    attempt_number: int,
) -> None:
    policy = create_policy()

    delay = policy.delay_seconds_for_attempt(
        attempt_number=attempt_number,
    )

    assert delay == 60.0


@pytest.mark.parametrize(
    "attempt_number",
    [
        0,
        -1,
    ],
)
def test_retry_delay_rejects_invalid_attempt_number(
    attempt_number: int,
) -> None:
    policy = create_policy()

    with pytest.raises(
        ValueError,
        match=escape("attempt_number must be at least one."),
    ):
        policy.delay_seconds_for_attempt(
            attempt_number=attempt_number,
        )


@pytest.mark.parametrize(
    ("attempt_count", "max_attempts", "expected"),
    [
        (0, 3, True),
        (1, 3, True),
        (2, 3, True),
        (3, 3, False),
        (1, 1, False),
    ],
)
def test_retry_eligibility_respects_attempt_budget(
    attempt_count: int,
    max_attempts: int,
    expected: bool,
) -> None:
    policy = create_policy()

    assert (
        policy.can_retry(
            attempt_count=attempt_count,
            max_attempts=max_attempts,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("attempt_count", "max_attempts", "expected_message"),
    [
        (-1, 3, "attempt_count must not be negative."),
        (0, 0, "max_attempts must be at least one."),
        (4, 3, "attempt_count must not exceed max_attempts."),
    ],
)
def test_retry_eligibility_rejects_invalid_attempt_budget(
    attempt_count: int,
    max_attempts: int,
    expected_message: str,
) -> None:
    policy = create_policy()

    with pytest.raises(
        ValueError,
        match=escape(expected_message),
    ):
        policy.can_retry(
            attempt_count=attempt_count,
            max_attempts=max_attempts,
        )


@pytest.mark.parametrize(
    ("base_delay", "maximum_delay", "expected_message"),
    [
        (
            0.0,
            60.0,
            "base_delay_seconds must be greater than zero.",
        ),
        (
            -1.0,
            60.0,
            "base_delay_seconds must be greater than zero.",
        ),
        (
            2.0,
            0.0,
            "maximum_delay_seconds must be greater than zero.",
        ),
        (
            2.0,
            -1.0,
            "maximum_delay_seconds must be greater than zero.",
        ),
        (
            10.0,
            5.0,
            ("maximum_delay_seconds must not be smaller than base_delay_seconds."),
        ),
    ],
)
def test_retry_policy_rejects_invalid_configuration(
    base_delay: float,
    maximum_delay: float,
    expected_message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=escape(expected_message),
    ):
        AgentRunRetryPolicy(
            base_delay_seconds=base_delay,
            maximum_delay_seconds=maximum_delay,
        )


def test_retry_policy_accepts_equal_base_and_maximum_delay() -> None:
    policy = AgentRunRetryPolicy(
        base_delay_seconds=5.0,
        maximum_delay_seconds=5.0,
    )

    assert (
        policy.delay_seconds_for_attempt(
            attempt_number=1,
        )
        == 5.0
    )
    assert (
        policy.delay_seconds_for_attempt(
            attempt_number=10,
        )
        == 5.0
    )
