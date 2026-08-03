"""Unit tests for the AgentRunAttempt domain entity."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone
from re import escape
from uuid import UUID

import pytest

from supportops.modules.agent_runs.domain.models import (
    AgentRunAttempt,
    AgentRunAttemptOutcome,
)


def create_attempt(
    *,
    now: datetime | None = None,
    attempt_number: int = 1,
    worker_id: str = "worker-a",
) -> AgentRunAttempt:
    return AgentRunAttempt.start(
        agent_run_id=UUID(
            "db3caf43-5c48-41d0-9a11-6cc39ea96682",
        ),
        attempt_number=attempt_number,
        worker_id=worker_id,
        lease_token=UUID(
            "dd0ae456-3467-41db-93d1-a908f40e8365",
        ),
        execution_request_id=UUID(
            "d1fa068f-2278-47a8-b3c9-39ccf91f0a5e",
        ),
        now=now,
    )


def test_start_attempt_creates_active_attempt() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)

    attempt = create_attempt(now=now)

    assert attempt.id.version == 4
    assert attempt.attempt_number == 1
    assert attempt.worker_id == "worker-a"
    assert attempt.started_at == now
    assert attempt.finished_at is None
    assert attempt.outcome is None
    assert attempt.error_code is None
    assert attempt.error_summary is None
    assert not attempt.is_finished


def test_attempt_ownership_is_immutable() -> None:
    attempt = create_attempt()

    with pytest.raises(FrozenInstanceError):
        attempt.worker_id = "worker-b"  # type: ignore[misc]


@pytest.mark.parametrize(
    "attempt_number",
    [
        0,
        -1,
    ],
)
def test_attempt_number_must_be_positive(
    attempt_number: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=escape("attempt_number must be at least one."),
    ):
        create_attempt(attempt_number=attempt_number)


@pytest.mark.parametrize(
    "worker_id",
    [
        "",
        " worker-a",
        "worker-a ",
        "a" * 129,
    ],
)
def test_attempt_rejects_invalid_worker_id(
    worker_id: str,
) -> None:
    with pytest.raises(ValueError):
        create_attempt(worker_id=worker_id)


def test_attempt_rejects_non_utc_started_at() -> None:
    non_utc = datetime(
        2026,
        7,
        31,
        9,
        0,
        tzinfo=timezone(timedelta(hours=-3)),
    )

    with pytest.raises(
        ValueError,
        match=escape("started_at must be a UTC-aware timestamp."),
    ):
        create_attempt(now=non_utc)


def test_attempt_outcome_and_finished_at_are_required_together() -> None:
    attempt = create_attempt()

    with pytest.raises(
        ValueError,
        match=escape(
            "Attempt outcome and finished_at must be populated together.",
        ),
    ):
        replace(
            attempt,
            outcome=AgentRunAttemptOutcome.SUCCEEDED,
        )


def test_finished_at_cannot_precede_started_at() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    attempt = create_attempt(now=now)

    with pytest.raises(
        ValueError,
        match=escape(
            "finished_at must not be earlier than started_at.",
        ),
    ):
        replace(
            attempt,
            finished_at=now - timedelta(seconds=1),
            outcome=AgentRunAttemptOutcome.SUCCEEDED,
        )


def test_successful_attempt_rejects_error_details() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    attempt = create_attempt(now=now)

    with pytest.raises(
        ValueError,
        match=escape(
            "Succeeded attempts must not contain error details.",
        ),
    ):
        replace(
            attempt,
            finished_at=now + timedelta(seconds=1),
            outcome=AgentRunAttemptOutcome.SUCCEEDED,
            error_code="unexpected_executor_failure",
            error_summary="The executor failed unexpectedly.",
        )


@pytest.mark.parametrize(
    "outcome",
    [
        AgentRunAttemptOutcome.RETRYABLE_FAILURE,
        AgentRunAttemptOutcome.TERMINAL_FAILURE,
        AgentRunAttemptOutcome.TIMED_OUT,
        AgentRunAttemptOutcome.LEASE_EXPIRED,
    ],
)
def test_failed_attempt_requires_safe_error_details(
    outcome: AgentRunAttemptOutcome,
) -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    attempt = create_attempt(now=now)

    with pytest.raises(
        ValueError,
        match=escape(
            "Failed attempts require error details.",
        ),
    ):
        replace(
            attempt,
            finished_at=now + timedelta(seconds=1),
            outcome=outcome,
        )


def test_active_attempt_rejects_error_details() -> None:
    attempt = create_attempt()

    with pytest.raises(
        ValueError,
        match=escape(
            "Active attempts must not contain error details.",
        ),
    ):
        replace(
            attempt,
            error_code="retryable_executor_failure",
            error_summary=("The configured executor reported a retryable failure."),
        )


def test_completed_attempt_requires_error_code_and_summary_together() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    attempt = create_attempt(now=now)

    with pytest.raises(
        ValueError,
        match=escape(
            "Error code and summary must be populated or cleared together.",
        ),
    ):
        replace(
            attempt,
            finished_at=now + timedelta(seconds=1),
            outcome=AgentRunAttemptOutcome.TIMED_OUT,
            error_code="executor_timeout",
        )


def test_completed_attempt_is_reported_by_domain_property() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    attempt = create_attempt(now=now)

    completed_attempt = replace(
        attempt,
        finished_at=now + timedelta(seconds=1),
        outcome=AgentRunAttemptOutcome.SUCCEEDED,
    )

    assert completed_attempt.is_finished


def test_awaiting_approval_attempt_is_valid_when_closed_and_error_free() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    attempt = replace(
        create_attempt(now=now),
        finished_at=now + timedelta(seconds=1),
        outcome=AgentRunAttemptOutcome.AWAITING_APPROVAL,
    )

    assert attempt.is_finished
    assert attempt.outcome is AgentRunAttemptOutcome.AWAITING_APPROVAL
    assert attempt.error_code is None
    assert attempt.error_summary is None


def test_awaiting_approval_requires_finished_at() -> None:
    with pytest.raises(
        ValueError,
        match=escape(
            "Attempt outcome and finished_at must be populated together.",
        ),
    ):
        replace(
            create_attempt(),
            outcome=AgentRunAttemptOutcome.AWAITING_APPROVAL,
        )


def test_awaiting_approval_rejects_error_code() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    with pytest.raises(
        ValueError,
        match=escape(
            "Error code and summary must be populated or cleared together.",
        ),
    ):
        replace(
            create_attempt(now=now),
            finished_at=now + timedelta(seconds=1),
            outcome=AgentRunAttemptOutcome.AWAITING_APPROVAL,
            error_code="unexpected_executor_failure",
        )


def test_awaiting_approval_rejects_error_details() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    with pytest.raises(
        ValueError,
        match=escape(
            "Awaiting-approval attempts must not contain error details.",
        ),
    ):
        replace(
            create_attempt(now=now),
            finished_at=now + timedelta(seconds=1),
            outcome=AgentRunAttemptOutcome.AWAITING_APPROVAL,
            error_code="unexpected_executor_failure",
            error_summary="The executor failed unexpectedly.",
        )
