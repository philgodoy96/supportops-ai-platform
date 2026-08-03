"""Unit tests for the durable AgentRun domain entity."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone
from re import escape
from uuid import UUID

import pytest

from supportops.modules.agent_runs.domain.models import (
    AGENT_RUN_WORKFLOW_VERSION_MAX_LENGTH,
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    INITIAL_TICKET_PROCESSING_TRIGGER_KEY,
    INITIAL_TICKET_PROCESSING_WORKFLOW_NAME,
    TICKET_CLASSIFICATION_WORKFLOW_VERSION,
    AgentRun,
    AgentRunStatus,
)


def create_agent_run(
    *,
    now: datetime | None = None,
    workflow_version: str = TICKET_CLASSIFICATION_WORKFLOW_VERSION,
    max_retryable_failures: int = 3,
) -> AgentRun:
    return AgentRun.create_initial(
        workspace_id=UUID(
            "032c8c87-57cc-4d14-bfbd-04968b4e8cd4",
        ),
        ticket_id=UUID(
            "38bb60fe-d2ea-4615-b499-91aa45069019",
        ),
        ingestion_request_id=UUID(
            "725eec8a-c504-4071-ac96-c78cc907f26c",
        ),
        correlation_id=UUID(
            "1038c98e-62fd-45df-9839-138f7105cb78",
        ),
        workflow_version=workflow_version,
        max_retryable_failures=max_retryable_failures,
        now=now,
    )


def test_create_initial_agent_run_assigns_supplied_workflow_contract() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)

    run = create_agent_run(now=now)

    assert run.id.version == 4
    assert run.workflow_name == INITIAL_TICKET_PROCESSING_WORKFLOW_NAME
    assert run.workflow_version == TICKET_CLASSIFICATION_WORKFLOW_VERSION
    assert run.trigger_key == INITIAL_TICKET_PROCESSING_TRIGGER_KEY
    assert run.status is AgentRunStatus.QUEUED
    assert run.available_at == now
    assert run.attempt_count == 0
    assert run.retryable_failure_count == 0
    assert run.max_retryable_failures == 3
    assert run.lease_owner is None
    assert run.lease_token is None
    assert run.lease_expires_at is None
    assert run.first_started_at is None
    assert run.completed_at is None
    assert run.last_error_code is None
    assert run.last_error_summary is None
    assert run.created_at == now
    assert run.updated_at == now
    assert not run.is_terminal
    assert run.retryable_failures_remaining == 3


def test_retryable_failures_remaining_tracks_failure_budget() -> None:
    run = replace(
        create_agent_run(),
        retryable_failure_count=1,
    )

    assert run.retryable_failures_remaining == 2


def test_attempt_count_may_exceed_max_retryable_failures() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    run = replace(
        create_agent_run(now=now),
        attempt_count=5,
        retryable_failure_count=2,
        first_started_at=now + timedelta(minutes=1),
        updated_at=now + timedelta(minutes=1),
    )

    assert run.attempt_count == 5
    assert run.max_retryable_failures == 3
    assert run.retryable_failure_count == 2


def test_create_initial_accepts_deterministic_baseline_workflow_version() -> None:
    run = create_agent_run(
        workflow_version=DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    )

    assert run.workflow_version == DETERMINISTIC_BASELINE_WORKFLOW_VERSION


@pytest.mark.parametrize(
    ("workflow_version", "expected_message"),
    [
        ("", "workflow_version is required."),
        (
            " ticket-classification-v1",
            "workflow_version must not contain surrounding whitespace.",
        ),
        (
            "ticket-classification-v1 ",
            "workflow_version must not contain surrounding whitespace.",
        ),
        (
            "w" * (AGENT_RUN_WORKFLOW_VERSION_MAX_LENGTH + 1),
            "workflow_version exceeds the maximum length.",
        ),
    ],
)
def test_create_initial_rejects_invalid_workflow_version(
    workflow_version: str,
    expected_message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=escape(expected_message),
    ):
        create_agent_run(workflow_version=workflow_version)


def test_agent_run_ownership_is_immutable() -> None:
    run = create_agent_run()

    with pytest.raises(FrozenInstanceError):
        run.workspace_id = UUID(  # type: ignore[misc]
            "4aefba3b-b57e-47d1-889e-bb28762fa1ed",
        )


@pytest.mark.parametrize(
    (
        "attempt_count",
        "retryable_failure_count",
        "max_retryable_failures",
        "expected_message",
    ),
    [
        (
            -1,
            0,
            3,
            "attempt_count must not be negative.",
        ),
        (
            0,
            -1,
            3,
            "retryable_failure_count must not be negative.",
        ),
        (
            0,
            0,
            0,
            "max_retryable_failures must be at least one.",
        ),
        (
            0,
            4,
            3,
            ("retryable_failure_count must not exceed max_retryable_failures."),
        ),
    ],
)
def test_agent_run_rejects_invalid_retry_budget(
    attempt_count: int,
    retryable_failure_count: int,
    max_retryable_failures: int,
    expected_message: str,
) -> None:
    run = create_agent_run()

    with pytest.raises(
        ValueError,
        match=escape(expected_message),
    ):
        replace(
            run,
            attempt_count=attempt_count,
            retryable_failure_count=retryable_failure_count,
            max_retryable_failures=max_retryable_failures,
            first_started_at=(
                datetime(2026, 7, 31, 12, 1, tzinfo=UTC) if attempt_count > 0 else None
            ),
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "available_at",
        "created_at",
        "updated_at",
    ],
)
def test_agent_run_rejects_non_utc_required_timestamp(
    field_name: str,
) -> None:
    run = create_agent_run()
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
        match=escape(f"{field_name} must be a UTC-aware timestamp."),
    ):
        if field_name == "available_at":
            replace(run, available_at=non_utc)
        elif field_name == "created_at":
            replace(run, created_at=non_utc)
        else:
            replace(run, updated_at=non_utc)


def test_agent_run_rejects_partial_lease_ownership() -> None:
    run = create_agent_run()

    with pytest.raises(
        ValueError,
        match=escape(
            "Lease ownership fields must be populated or cleared together.",
        ),
    ):
        replace(
            run,
            lease_owner="worker-a",
        )


def test_running_agent_run_requires_lease_ownership() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    run = create_agent_run(now=now)

    with pytest.raises(
        ValueError,
        match=escape(
            "Running AgentRuns require active lease ownership.",
        ),
    ):
        replace(
            run,
            status=AgentRunStatus.RUNNING,
            attempt_count=1,
            first_started_at=now,
        )


def test_non_running_agent_run_rejects_lease_ownership() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    run = create_agent_run(now=now)

    with pytest.raises(
        ValueError,
        match=escape(
            "Lease ownership is allowed only while an AgentRun is running.",
        ),
    ):
        replace(
            run,
            lease_owner="worker-a",
            lease_token=UUID(
                "dd0ae456-3467-41db-93d1-a908f40e8365",
            ),
            lease_expires_at=now + timedelta(seconds=45),
        )


def test_started_agent_run_requires_first_started_at() -> None:
    run = create_agent_run()

    with pytest.raises(
        ValueError,
        match=escape(
            "first_started_at is required after an attempt has started.",
        ),
    ):
        replace(
            run,
            attempt_count=1,
        )


def test_unstarted_agent_run_rejects_first_started_at() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    run = create_agent_run(now=now)

    with pytest.raises(
        ValueError,
        match=escape(
            "first_started_at must be null before the first attempt.",
        ),
    ):
        replace(
            run,
            first_started_at=now,
        )


@pytest.mark.parametrize(
    "status",
    [
        AgentRunStatus.SUCCEEDED,
        AgentRunStatus.FAILED,
    ],
)
def test_terminal_agent_run_requires_completed_at(
    status: AgentRunStatus,
) -> None:
    run = create_agent_run()

    with pytest.raises(
        ValueError,
        match=escape("Terminal AgentRuns require completed_at."),
    ):
        replace(
            run,
            status=status,
            last_error_code=(
                "terminal_executor_failure" if status is AgentRunStatus.FAILED else None
            ),
            last_error_summary=(
                "The configured executor reported a terminal failure."
                if status is AgentRunStatus.FAILED
                else None
            ),
        )


@pytest.mark.parametrize(
    "status",
    [
        AgentRunStatus.QUEUED,
        AgentRunStatus.RUNNING,
        AgentRunStatus.RETRY_SCHEDULED,
        AgentRunStatus.WAITING_FOR_APPROVAL,
    ],
)
def test_non_terminal_agent_run_rejects_completed_at(
    status: AgentRunStatus,
) -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    run = create_agent_run(now=now)

    with pytest.raises(
        ValueError,
        match=escape(
            "Non-terminal AgentRuns must not define completed_at.",
        ),
    ):
        if status is AgentRunStatus.RUNNING:
            replace(
                run,
                status=status,
                completed_at=now,
                attempt_count=1,
                first_started_at=now,
                lease_owner="worker-a",
                lease_token=UUID(
                    "dd0ae456-3467-41db-93d1-a908f40e8365",
                ),
                lease_expires_at=now + timedelta(seconds=45),
            )
        elif status is AgentRunStatus.RETRY_SCHEDULED:
            replace(
                run,
                status=status,
                completed_at=now,
                attempt_count=1,
                first_started_at=now,
                last_error_code="retryable_executor_failure",
                last_error_summary=("The configured executor reported a retryable failure."),
            )
        elif status is AgentRunStatus.WAITING_FOR_APPROVAL:
            replace(
                run,
                status=status,
                available_at=None,
                completed_at=now,
                attempt_count=1,
                first_started_at=now,
            )
        else:
            replace(
                run,
                status=status,
                completed_at=now,
            )


@pytest.mark.parametrize(
    "status",
    [
        AgentRunStatus.RETRY_SCHEDULED,
        AgentRunStatus.FAILED,
    ],
)
def test_failure_state_requires_safe_error_details(
    status: AgentRunStatus,
) -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    run = create_agent_run(now=now)

    with pytest.raises(
        ValueError,
        match=escape(
            "Retry-scheduled and failed AgentRuns require error details.",
        ),
    ):
        replace(
            run,
            status=status,
            attempt_count=1,
            first_started_at=now,
            completed_at=(now if status is AgentRunStatus.FAILED else None),
        )


def test_succeeded_agent_run_rejects_error_details() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    run = create_agent_run(now=now)

    with pytest.raises(
        ValueError,
        match=escape(
            "Queued and succeeded AgentRuns must not contain error details.",
        ),
    ):
        replace(
            run,
            status=AgentRunStatus.SUCCEEDED,
            attempt_count=1,
            first_started_at=now,
            completed_at=now,
            last_error_code="unexpected_executor_failure",
            last_error_summary="The executor failed unexpectedly.",
        )


def test_agent_run_requires_error_code_and_summary_together() -> None:
    run = create_agent_run()

    with pytest.raises(
        ValueError,
        match=escape(
            "Error code and summary must be populated or cleared together.",
        ),
    ):
        replace(
            run,
            last_error_code="retryable_executor_failure",
        )


def test_terminal_status_is_reported_by_domain_property() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    run = create_agent_run(now=now)

    succeeded_run = replace(
        run,
        status=AgentRunStatus.SUCCEEDED,
        attempt_count=1,
        first_started_at=now,
        completed_at=now,
    )

    assert succeeded_run.is_terminal
    assert succeeded_run.retryable_failures_remaining == 3


def _waiting_run(
    *,
    now: datetime | None = None,
    retryable_failure_count: int = 0,
) -> AgentRun:
    created_at = now or datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    run = create_agent_run(now=created_at)
    return replace(
        run,
        status=AgentRunStatus.WAITING_FOR_APPROVAL,
        available_at=None,
        attempt_count=1,
        retryable_failure_count=retryable_failure_count,
        first_started_at=created_at,
        updated_at=created_at + timedelta(seconds=10),
    )


def test_waiting_for_approval_with_cleared_fields_is_valid() -> None:
    run = _waiting_run()

    assert run.status is AgentRunStatus.WAITING_FOR_APPROVAL
    assert run.available_at is None
    assert run.lease_owner is None
    assert run.lease_token is None
    assert run.lease_expires_at is None
    assert run.completed_at is None
    assert run.last_error_code is None
    assert run.last_error_summary is None
    assert not run.is_terminal


def test_waiting_for_approval_rejects_lease_owner() -> None:
    with pytest.raises(
        ValueError,
        match=escape(
            "Lease ownership fields must be populated or cleared together.",
        ),
    ):
        replace(
            _waiting_run(),
            lease_owner="worker-a",
        )


def test_waiting_for_approval_rejects_lease_token() -> None:
    with pytest.raises(
        ValueError,
        match=escape(
            "Lease ownership fields must be populated or cleared together.",
        ),
    ):
        replace(
            _waiting_run(),
            lease_token=UUID("dd0ae456-3467-41db-93d1-a908f40e8365"),
        )


def test_waiting_for_approval_rejects_lease_expiry() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    with pytest.raises(
        ValueError,
        match=escape(
            "Lease ownership fields must be populated or cleared together.",
        ),
    ):
        replace(
            _waiting_run(now=now),
            lease_expires_at=now + timedelta(seconds=45),
        )


def test_waiting_for_approval_rejects_available_at() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    with pytest.raises(
        ValueError,
        match=escape("Waiting AgentRuns must not define available_at."),
    ):
        replace(
            _waiting_run(now=now),
            available_at=now,
        )


def test_waiting_for_approval_rejects_completed_at() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    with pytest.raises(
        ValueError,
        match=escape(
            "Non-terminal AgentRuns must not define completed_at.",
        ),
    ):
        replace(
            _waiting_run(now=now),
            completed_at=now,
        )


def test_waiting_for_approval_rejects_last_error_code() -> None:
    with pytest.raises(
        ValueError,
        match=escape(
            "Error code and summary must be populated or cleared together.",
        ),
    ):
        replace(
            _waiting_run(),
            last_error_code="unexpected_executor_failure",
        )


def test_waiting_for_approval_rejects_error_details() -> None:
    with pytest.raises(
        ValueError,
        match=escape("Waiting AgentRuns must not contain error details."),
    ):
        replace(
            _waiting_run(),
            last_error_code="unexpected_executor_failure",
            last_error_summary="The executor failed unexpectedly.",
        )


def test_non_waiting_status_rejects_available_at_none() -> None:
    with pytest.raises(
        ValueError,
        match=escape("Non-waiting AgentRuns require available_at."),
    ):
        replace(
            create_agent_run(),
            available_at=None,
        )


def test_waiting_for_approval_is_not_terminal() -> None:
    assert not _waiting_run().is_terminal


def test_waiting_does_not_consume_retryable_failure_budget() -> None:
    run = _waiting_run(retryable_failure_count=1)

    assert run.retryable_failure_count == 1
    assert run.retryable_failures_remaining == 2
    assert run.attempt_count == 1
