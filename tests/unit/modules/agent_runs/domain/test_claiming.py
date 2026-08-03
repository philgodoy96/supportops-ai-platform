"""Unit tests for AgentRun claim domain contracts."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from re import escape
from uuid import UUID

import pytest

from supportops.modules.agent_runs.domain.claiming import (
    AgentRunClaim,
    ClaimAgentRunCommand,
)
from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    AgentRun,
    AgentRunAttempt,
    AgentRunStatus,
)

_RUN_ID = UUID(
    "69184ef1-4d71-452e-8070-0b784c29368e",
)
_LEASE_TOKEN = UUID(
    "dd0ae456-3467-41db-93d1-a908f40e8365",
)
_EXECUTION_REQUEST_ID = UUID(
    "d1fa068f-2278-47a8-b3c9-39ccf91f0a5e",
)
_CLAIMED_AT = datetime(
    2026,
    7,
    31,
    17,
    0,
    tzinfo=UTC,
)


def create_claim_command(
    *,
    worker_id: str = "worker-a",
    claimed_at: datetime = _CLAIMED_AT,
    lease_expires_at: datetime | None = None,
) -> ClaimAgentRunCommand:
    return ClaimAgentRunCommand(
        worker_id=worker_id,
        lease_token=_LEASE_TOKEN,
        execution_request_id=_EXECUTION_REQUEST_ID,
        claimed_at=claimed_at,
        lease_expires_at=(lease_expires_at or claimed_at + timedelta(seconds=45)),
    )


def create_running_run() -> AgentRun:
    initial_run = AgentRun.create_initial(
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
        now=_CLAIMED_AT - timedelta(minutes=1),
    )

    return replace(
        initial_run,
        status=AgentRunStatus.RUNNING,
        attempt_count=1,
        lease_owner="worker-a",
        lease_token=_LEASE_TOKEN,
        lease_expires_at=_CLAIMED_AT + timedelta(seconds=45),
        first_started_at=_CLAIMED_AT,
        updated_at=_CLAIMED_AT,
    )


def create_active_attempt() -> AgentRunAttempt:
    return AgentRunAttempt.start(
        agent_run_id=_RUN_ID,
        attempt_number=1,
        worker_id="worker-a",
        lease_token=_LEASE_TOKEN,
        execution_request_id=_EXECUTION_REQUEST_ID,
        now=_CLAIMED_AT,
    )


def test_claim_command_preserves_explicit_ownership_values() -> None:
    command = create_claim_command()

    assert command.worker_id == "worker-a"
    assert command.lease_token == _LEASE_TOKEN
    assert command.execution_request_id == _EXECUTION_REQUEST_ID
    assert command.claimed_at == _CLAIMED_AT
    assert command.lease_expires_at == _CLAIMED_AT + timedelta(seconds=45)


@pytest.mark.parametrize(
    "worker_id",
    [
        "",
        " worker-a",
        "worker-a ",
        "a" * 129,
    ],
)
def test_claim_command_rejects_invalid_worker_id(
    worker_id: str,
) -> None:
    with pytest.raises(ValueError):
        create_claim_command(worker_id=worker_id)


@pytest.mark.parametrize(
    "field_name",
    [
        "claimed_at",
        "lease_expires_at",
    ],
)
def test_claim_command_rejects_naive_timestamp(
    field_name: str,
) -> None:
    naive_timestamp = datetime(2026, 7, 31, 17, 0)
    claimed_at = naive_timestamp if field_name == "claimed_at" else _CLAIMED_AT
    lease_expires_at = (
        naive_timestamp if field_name == "lease_expires_at" else _CLAIMED_AT + timedelta(seconds=45)
    )

    with pytest.raises(
        ValueError,
        match=escape(
            f"{field_name} must be a UTC-aware timestamp.",
        ),
    ):
        create_claim_command(
            claimed_at=claimed_at,
            lease_expires_at=lease_expires_at,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "claimed_at",
        "lease_expires_at",
    ],
)
def test_claim_command_rejects_non_utc_timestamp(
    field_name: str,
) -> None:
    non_utc_timestamp = datetime(
        2026,
        7,
        31,
        14,
        0,
        tzinfo=timezone(timedelta(hours=-3)),
    )
    claimed_at = non_utc_timestamp if field_name == "claimed_at" else _CLAIMED_AT
    lease_expires_at = (
        non_utc_timestamp
        if field_name == "lease_expires_at"
        else _CLAIMED_AT + timedelta(seconds=45)
    )

    with pytest.raises(
        ValueError,
        match=escape(
            f"{field_name} must be a UTC-aware timestamp.",
        ),
    ):
        create_claim_command(
            claimed_at=claimed_at,
            lease_expires_at=lease_expires_at,
        )


@pytest.mark.parametrize(
    "lease_expires_at",
    [
        _CLAIMED_AT,
        _CLAIMED_AT - timedelta(seconds=1),
    ],
)
def test_claim_command_requires_future_lease_expiration(
    lease_expires_at: datetime,
) -> None:
    with pytest.raises(
        ValueError,
        match=escape(
            "lease_expires_at must be later than claimed_at.",
        ),
    ):
        create_claim_command(
            lease_expires_at=lease_expires_at,
        )


def test_agent_run_claim_accepts_matching_run_and_attempt() -> None:
    run = create_running_run()
    attempt = create_active_attempt()

    claim = AgentRunClaim(
        agent_run=run,
        attempt=attempt,
    )

    assert claim.agent_run == run
    assert claim.attempt == attempt


def test_agent_run_claim_rejects_different_run_ids() -> None:
    run = create_running_run()
    attempt = replace(
        create_active_attempt(),
        agent_run_id=UUID(
            "43499fc4-3638-4097-aaf2-3c5300cf6cd6",
        ),
    )

    with pytest.raises(
        ValueError,
        match=escape(
            "Claimed AgentRun and attempt must reference the same run.",
        ),
    ):
        AgentRunClaim(
            agent_run=run,
            attempt=attempt,
        )


def test_agent_run_claim_rejects_attempt_number_mismatch() -> None:
    run = create_running_run()
    attempt = replace(
        create_active_attempt(),
        attempt_number=2,
    )

    with pytest.raises(
        ValueError,
        match=escape(
            "Claimed AgentRun attempt count must match the attempt number.",
        ),
    ):
        AgentRunClaim(
            agent_run=run,
            attempt=attempt,
        )


def test_agent_run_claim_rejects_lease_token_mismatch() -> None:
    run = create_running_run()
    attempt = replace(
        create_active_attempt(),
        lease_token=UUID(
            "35e64ab6-d1b2-4386-a51e-48f11fa6d058",
        ),
    )

    with pytest.raises(
        ValueError,
        match=escape(
            "Claimed AgentRun and attempt must share the lease token.",
        ),
    ):
        AgentRunClaim(
            agent_run=run,
            attempt=attempt,
        )


def test_agent_run_claim_rejects_worker_id_mismatch() -> None:
    run = create_running_run()
    attempt = replace(
        create_active_attempt(),
        worker_id="worker-b",
    )

    with pytest.raises(
        ValueError,
        match=escape(
            "Claimed AgentRun and attempt must share the worker ID.",
        ),
    ):
        AgentRunClaim(
            agent_run=run,
            attempt=attempt,
        )
