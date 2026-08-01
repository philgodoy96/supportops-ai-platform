"""Unit tests for AgentRun SQLAlchemy persistence mappings."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

from sqlalchemy import CheckConstraint, Table, UniqueConstraint

from supportops.modules.agent_runs.domain.models import (
    AgentRun,
    AgentRunAttempt,
    AgentRunAttemptOutcome,
    AgentRunStatus,
)
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunAttemptRecord,
    AgentRunRecord,
)


def create_initial_run() -> AgentRun:
    return AgentRun.create_initial(
        agent_run_id=UUID(
            "69184ef1-4d71-452e-8070-0b784c29368e",
        ),
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
        max_attempts=3,
        now=datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    )


def create_active_attempt() -> AgentRunAttempt:
    return AgentRunAttempt.start(
        attempt_id=UUID(
            "2b39f5b7-b2a4-48d0-b079-fdad286d5315",
        ),
        agent_run_id=UUID(
            "69184ef1-4d71-452e-8070-0b784c29368e",
        ),
        attempt_number=1,
        worker_id="worker-a",
        lease_token=UUID(
            "dd0ae456-3467-41db-93d1-a908f40e8365",
        ),
        execution_request_id=UUID(
            "d1fa068f-2278-47a8-b3c9-39ccf91f0a5e",
        ),
        now=datetime(2026, 7, 31, 12, 1, tzinfo=UTC),
    )


def test_agent_run_record_round_trip_preserves_initial_run() -> None:
    run = create_initial_run()

    record = AgentRunRecord.from_domain(run)
    restored = record.to_domain()

    assert restored == run
    assert record.status == AgentRunStatus.QUEUED.value
    assert record.lease_token is None
    assert record.completed_at is None


def test_agent_run_record_round_trip_preserves_running_run() -> None:
    run = create_initial_run()
    started_at = datetime(2026, 7, 31, 12, 1, tzinfo=UTC)
    lease_token = UUID(
        "dd0ae456-3467-41db-93d1-a908f40e8365",
    )

    running_run = replace(
        run,
        status=AgentRunStatus.RUNNING,
        attempt_count=1,
        lease_owner="worker-a",
        lease_token=lease_token,
        lease_expires_at=started_at + timedelta(seconds=45),
        first_started_at=started_at,
        updated_at=started_at,
    )

    record = AgentRunRecord.from_domain(running_run)
    restored = record.to_domain()

    assert restored == running_run
    assert record.status == AgentRunStatus.RUNNING.value
    assert record.lease_token == lease_token


def test_agent_run_record_round_trip_preserves_failed_run() -> None:
    run = create_initial_run()
    completed_at = datetime(2026, 7, 31, 12, 2, tzinfo=UTC)

    failed_run = replace(
        run,
        status=AgentRunStatus.FAILED,
        attempt_count=1,
        first_started_at=datetime(
            2026,
            7,
            31,
            12,
            1,
            tzinfo=UTC,
        ),
        completed_at=completed_at,
        last_error_code="terminal_executor_failure",
        last_error_summary=("The configured executor reported a terminal failure."),
        updated_at=completed_at,
    )

    record = AgentRunRecord.from_domain(failed_run)
    restored = record.to_domain()

    assert restored == failed_run
    assert record.status == AgentRunStatus.FAILED.value
    assert record.last_error_code == "terminal_executor_failure"


def test_attempt_record_round_trip_preserves_active_attempt() -> None:
    attempt = create_active_attempt()

    record = AgentRunAttemptRecord.from_domain(attempt)
    restored = record.to_domain()

    assert restored == attempt
    assert record.outcome is None
    assert record.finished_at is None


def test_attempt_record_round_trip_preserves_successful_attempt() -> None:
    attempt = create_active_attempt()
    finished_at = attempt.started_at + timedelta(seconds=1)

    succeeded_attempt = replace(
        attempt,
        finished_at=finished_at,
        outcome=AgentRunAttemptOutcome.SUCCEEDED,
    )

    record = AgentRunAttemptRecord.from_domain(succeeded_attempt)
    restored = record.to_domain()

    assert restored == succeeded_attempt
    assert record.outcome == AgentRunAttemptOutcome.SUCCEEDED.value
    assert record.error_code is None


def test_attempt_record_round_trip_preserves_failed_attempt() -> None:
    attempt = create_active_attempt()
    finished_at = attempt.started_at + timedelta(seconds=1)

    failed_attempt = replace(
        attempt,
        finished_at=finished_at,
        outcome=AgentRunAttemptOutcome.TIMED_OUT,
        error_code="executor_timeout",
        error_summary=("The configured executor exceeded its execution timeout."),
    )

    record = AgentRunAttemptRecord.from_domain(failed_attempt)
    restored = record.to_domain()

    assert restored == failed_attempt
    assert record.outcome == AgentRunAttemptOutcome.TIMED_OUT.value
    assert record.error_code == "executor_timeout"


def test_agent_run_metadata_declares_expected_constraints() -> None:
    table = cast(Table, AgentRunRecord.__table__)
    constraint_names = {
        constraint.name
        for constraint in table.constraints
        if isinstance(
            constraint,
            (CheckConstraint, UniqueConstraint),
        )
    }

    assert {
        "uq_agent_runs_workspace_ticket_id",
        "uq_agent_runs_ticket_trigger",
        "ck_agent_runs_agent_run_status",
        "ck_agent_runs_agent_run_attempt_count_non_negative",
        "ck_agent_runs_agent_run_max_attempts_positive",
        "ck_agent_runs_agent_run_attempt_limit",
        "ck_agent_runs_agent_run_started_attempt_state",
        "ck_agent_runs_agent_run_lease_fields_complete",
        "ck_agent_runs_agent_run_lease_state",
        "ck_agent_runs_agent_run_lease_expiration_order",
        "ck_agent_runs_agent_run_completion_state",
        "ck_agent_runs_agent_run_error_fields_complete",
        "ck_agent_runs_agent_run_error_code_format",
        "ck_agent_runs_agent_run_error_summary_format",
        "ck_agent_runs_agent_run_success_error_state",
        "ck_agent_runs_agent_run_failure_error_state",
        "ck_agent_runs_agent_run_timestamp_order",
        "ck_agent_runs_agent_run_lifecycle_timestamp_order",
        "ck_agent_runs_agent_run_lease_owner_format",
    }.issubset(constraint_names)


def test_agent_run_metadata_declares_query_driven_indexes() -> None:
    table = cast(Table, AgentRunRecord.__table__)
    index_names = {index.name for index in table.indexes}

    assert index_names == {
        "ix_agent_runs_available_claim",
        "ix_agent_runs_expired_lease",
        "ix_agent_runs_workspace_ticket_created_id",
    }


def test_attempt_metadata_declares_expected_constraints() -> None:
    table = cast(Table, AgentRunAttemptRecord.__table__)
    constraint_names = {
        constraint.name
        for constraint in table.constraints
        if isinstance(
            constraint,
            (CheckConstraint, UniqueConstraint),
        )
    }

    assert {
        "uq_agent_run_attempts_run_id",
        "uq_agent_run_attempts_run_number",
        "ck_agent_run_attempts_agent_run_attempt_number_positive",
        "ck_agent_run_attempts_agent_run_attempt_worker_id_format",
        "ck_agent_run_attempts_agent_run_attempt_outcome",
        "ck_agent_run_attempts_agent_run_attempt_completion_state",
        "ck_agent_run_attempts_agent_run_attempt_timestamp_order",
        "ck_agent_run_attempts_agent_run_attempt_error_fields_complete",
        "ck_agent_run_attempts_agent_run_attempt_error_code_format",
        "ck_agent_run_attempts_agent_run_attempt_error_summary_format",
        "ck_agent_run_attempts_agent_run_attempt_success_error_state",
        "ck_agent_run_attempts_agent_run_attempt_failure_error_state",
        "ck_agent_run_attempts_agent_run_attempt_active_error_state",
    }.issubset(constraint_names)
