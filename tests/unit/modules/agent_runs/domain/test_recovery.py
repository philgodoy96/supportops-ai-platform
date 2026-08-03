"""Unit tests for expired AgentRun lease recovery contracts."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from re import escape
from uuid import UUID

import pytest

from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    AgentRun,
    AgentRunStatus,
)
from supportops.modules.agent_runs.domain.recovery import (
    ExpiredAgentRunDisposition,
    RecoverExpiredAgentRunCommand,
    RecoverExpiredAgentRunResult,
)

_RECOVERED_AT = datetime(
    2026,
    7,
    31,
    19,
    0,
    tzinfo=UTC,
)
_RUN_ID = UUID(
    "69184ef1-4d71-452e-8070-0b784c29368e",
)
_EXPIRED_LEASE_TOKEN = UUID(
    "dd0ae456-3467-41db-93d1-a908f40e8365",
)


def create_initial_run() -> AgentRun:
    return AgentRun.create_initial(
        agent_run_id=_RUN_ID,
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
        workflow_version=DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
        max_retryable_failures=3,
        now=_RECOVERED_AT - timedelta(minutes=5),
    )


def create_recovered_run(
    *,
    disposition: ExpiredAgentRunDisposition,
) -> AgentRun:
    initial_run = create_initial_run()

    return replace(
        initial_run,
        status=AgentRunStatus(disposition.value),
        available_at=(
            _RECOVERED_AT + timedelta(seconds=2)
            if disposition is ExpiredAgentRunDisposition.RETRY_SCHEDULED
            else initial_run.available_at
        ),
        attempt_count=1,
        retryable_failure_count=1,
        lease_owner=None,
        lease_token=None,
        lease_expires_at=None,
        first_started_at=_RECOVERED_AT - timedelta(minutes=1),
        completed_at=(_RECOVERED_AT if disposition is ExpiredAgentRunDisposition.FAILED else None),
        last_error_code="worker_lease_expired",
        last_error_summary=("The worker lease expired before execution completed."),
        updated_at=_RECOVERED_AT,
    )


def test_recovery_command_preserves_explicit_values() -> None:
    command = RecoverExpiredAgentRunCommand(
        recovered_at=_RECOVERED_AT,
        retry_base_delay_seconds=2.0,
        retry_maximum_delay_seconds=60.0,
        error_code="worker_lease_expired",
        error_summary=("The worker lease expired before execution completed."),
    )

    assert command.recovered_at == _RECOVERED_AT
    assert command.retry_base_delay_seconds == 2.0
    assert command.retry_maximum_delay_seconds == 60.0
    assert command.error_code == "worker_lease_expired"


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2026, 7, 31, 19, 0),
        datetime(
            2026,
            7,
            31,
            16,
            0,
            tzinfo=timezone(timedelta(hours=-3)),
        ),
    ],
)
def test_recovery_command_requires_utc_timestamp(
    timestamp: datetime,
) -> None:
    with pytest.raises(
        ValueError,
        match=escape(
            "recovered_at must be a UTC-aware timestamp.",
        ),
    ):
        RecoverExpiredAgentRunCommand(
            recovered_at=timestamp,
            retry_base_delay_seconds=2.0,
            retry_maximum_delay_seconds=60.0,
            error_code="worker_lease_expired",
            error_summary=("The worker lease expired before execution completed."),
        )


@pytest.mark.parametrize(
    ("field_name", "value", "expected_message"),
    [
        (
            "retry_base_delay_seconds",
            0.0,
            "retry_base_delay_seconds must be greater than zero.",
        ),
        (
            "retry_base_delay_seconds",
            -1.0,
            "retry_base_delay_seconds must be greater than zero.",
        ),
        (
            "retry_maximum_delay_seconds",
            0.0,
            "retry_maximum_delay_seconds must be greater than zero.",
        ),
        (
            "retry_maximum_delay_seconds",
            1.0,
            ("retry_maximum_delay_seconds must not be smaller than retry_base_delay_seconds."),
        ),
    ],
)
def test_recovery_command_requires_valid_retry_delay_bounds(
    field_name: str,
    value: float,
    expected_message: str,
) -> None:
    values = {
        "retry_base_delay_seconds": 2.0,
        "retry_maximum_delay_seconds": 60.0,
    }
    values[field_name] = value

    with pytest.raises(
        ValueError,
        match=escape(expected_message),
    ):
        RecoverExpiredAgentRunCommand(
            recovered_at=_RECOVERED_AT,
            error_code="worker_lease_expired",
            error_summary=("The worker lease expired before execution completed."),
            **values,
        )


@pytest.mark.parametrize(
    ("field_name", "value", "expected_message"),
    [
        (
            "error_code",
            "",
            "error_code is required.",
        ),
        (
            "error_code",
            " worker_lease_expired",
            ("error_code must not contain surrounding whitespace."),
        ),
        (
            "error_summary",
            "",
            "error_summary is required.",
        ),
        (
            "error_summary",
            " unsafe summary ",
            ("error_summary must not contain surrounding whitespace."),
        ),
    ],
)
def test_recovery_command_rejects_invalid_error_text(
    field_name: str,
    value: str,
    expected_message: str,
) -> None:
    values = {
        "error_code": "worker_lease_expired",
        "error_summary": ("The worker lease expired before execution completed."),
    }
    values[field_name] = value

    with pytest.raises(
        ValueError,
        match=escape(expected_message),
    ):
        RecoverExpiredAgentRunCommand(
            recovered_at=_RECOVERED_AT,
            retry_base_delay_seconds=2.0,
            retry_maximum_delay_seconds=60.0,
            **values,
        )


@pytest.mark.parametrize(
    "disposition",
    [
        ExpiredAgentRunDisposition.RETRY_SCHEDULED,
        ExpiredAgentRunDisposition.FAILED,
    ],
)
def test_recovery_result_accepts_matching_state(
    disposition: ExpiredAgentRunDisposition,
) -> None:
    run = create_recovered_run(
        disposition=disposition,
    )

    result = RecoverExpiredAgentRunResult(
        agent_run=run,
        expired_lease_token=_EXPIRED_LEASE_TOKEN,
        disposition=disposition,
    )

    assert result.agent_run == run
    assert result.expired_lease_token == _EXPIRED_LEASE_TOKEN
    assert result.disposition is disposition


def test_recovery_result_rejects_status_mismatch() -> None:
    run = create_recovered_run(
        disposition=ExpiredAgentRunDisposition.RETRY_SCHEDULED,
    )

    with pytest.raises(
        ValueError,
        match=escape(
            "Recovered AgentRun status must match the disposition.",
        ),
    ):
        RecoverExpiredAgentRunResult(
            agent_run=run,
            expired_lease_token=_EXPIRED_LEASE_TOKEN,
            disposition=ExpiredAgentRunDisposition.FAILED,
        )


@pytest.mark.parametrize(
    ("field_name", "value", "expected_message"),
    [
        (
            "lease_owner",
            "worker-a",
            "Recovered AgentRun must not retain a lease owner.",
        ),
        (
            "lease_token",
            _EXPIRED_LEASE_TOKEN,
            "Recovered AgentRun must not retain a lease token.",
        ),
        (
            "lease_expires_at",
            _RECOVERED_AT,
            ("Recovered AgentRun must not retain lease expiration."),
        ),
    ],
)
def test_recovery_result_rejects_retained_lease_fields(
    field_name: str,
    value: object,
    expected_message: str,
) -> None:
    run = create_recovered_run(
        disposition=ExpiredAgentRunDisposition.RETRY_SCHEDULED,
    )
    # Bypass AgentRun invariants to assert recovery-result postconditions.
    object.__setattr__(run, field_name, value)

    with pytest.raises(
        ValueError,
        match=escape(expected_message),
    ):
        RecoverExpiredAgentRunResult(
            agent_run=run,
            expired_lease_token=_EXPIRED_LEASE_TOKEN,
            disposition=ExpiredAgentRunDisposition.RETRY_SCHEDULED,
        )
