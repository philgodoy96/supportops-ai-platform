"""Integration tests for Alembic migration configuration."""

import subprocess
import sys
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import pytest
from sqlalchemy import Table, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from tests.integration.conftest import clear_integration_business_data

from supportops.agent_tools.infrastructure.grant_models import (
    SensitiveExecutionGrantRecord,
)
from supportops.agent_tools.infrastructure.models import (
    AgentToolCallRecord,
)
from supportops.core.settings import Settings
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunRecord,
)
from supportops.modules.approvals.infrastructure.models import (
    ApprovalRequestRecord,
)

pytestmark = pytest.mark.integration

EXPECTED_HEAD = "e4a7c9d2f1b6"
CONTROLLED_WORKFLOW_REVISION = "e8b7c6d5a4f3"
PRE_CONTROLLED_WORKFLOW_REVISION = "d4e8f2a6c901"
RETRYABLE_FAILURE_BUDGET_REVISION = "f3a9c1d7e5b2"
WAITING_FOR_APPROVAL_REVISION = "b7c4d2e9a1f6"
TOOL_CALL_LIFECYCLE_REVISION = "c9e2f4a7b6d1"
APPROVAL_REQUEST_REVISION = "d6f1a8c3e5b7"


def run_alembic_command(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run Alembic through the active Python environment."""

    return subprocess.run(
        [sys.executable, "-m", "alembic", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


async def relation_exists(
    engine: AsyncEngine,
    relation_name: str,
) -> bool:
    """Return whether the named relation is registered in PostgreSQL."""

    async with engine.connect() as connection:
        result = await connection.execute(
            text("SELECT to_regclass(:relation_name)"),
            {"relation_name": relation_name},
        )
        return result.scalar_one() is not None


async def constraint_exists(
    engine: AsyncEngine,
    *,
    table_name: str,
    constraint_name: str,
) -> bool:
    """Return whether a named PostgreSQL table constraint exists."""

    async with engine.connect() as connection:
        result = await connection.execute(
            text(
                """
                SELECT 1
                FROM pg_constraint AS constraint_def
                JOIN pg_class AS table_def
                    ON table_def.oid = constraint_def.conrelid
                JOIN pg_namespace AS namespace_def
                    ON namespace_def.oid = table_def.relnamespace
                WHERE namespace_def.nspname = 'public'
                  AND table_def.relname = :table_name
                  AND constraint_def.conname = :constraint_name
                """
            ),
            {
                "table_name": table_name,
                "constraint_name": constraint_name,
            },
        )
        return result.scalar_one_or_none() is not None


async def column_exists(
    engine: AsyncEngine,
    *,
    table_name: str,
    column_name: str,
) -> bool:
    """Return whether a named column exists on a public table."""

    async with engine.connect() as connection:
        result = await connection.execute(
            text(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = :table_name
                  AND column_name = :column_name
                """
            ),
            {
                "table_name": table_name,
                "column_name": column_name,
            },
        )
        return result.scalar_one_or_none() is not None


def test_alembic_reports_expected_head() -> None:
    result = run_alembic_command("heads")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{EXPECTED_HEAD} (head)"


def test_alembic_current_connects_to_postgresql() -> None:
    result = run_alembic_command("current")

    assert result.returncode == 0, result.stderr


async def test_alembic_upgrade_creates_business_tables(
    exclusive_integration_database: None,
) -> None:
    settings = Settings()
    engine = create_async_engine(str(settings.postgresql_url))

    upgrade = run_alembic_command("upgrade", "head")
    assert upgrade.returncode == 0, upgrade.stderr

    try:
        assert await relation_exists(engine, "public.alembic_version")
        assert await relation_exists(engine, "public.workspaces")
        assert await relation_exists(engine, "public.tickets")
        assert await relation_exists(engine, "public.knowledge_documents")
        assert await relation_exists(
            engine,
            "public.knowledge_document_versions",
        )
        assert await relation_exists(
            engine,
            "public.knowledge_document_chunks",
        )
        assert await relation_exists(engine, "public.agent_tool_calls")
        assert await relation_exists(engine, "public.approval_requests")
        assert await relation_exists(engine, "public.sensitive_execution_grants")
        assert await relation_exists(engine, "public.support_recommendations")
        assert await relation_exists(
            engine,
            "public.support_recommendation_citations",
        )
        assert await constraint_exists(
            engine,
            table_name="ticket_classifications",
            constraint_name="uq_ticket_classifications_run_id",
        )
        assert await constraint_exists(
            engine,
            table_name="knowledge_document_chunks",
            constraint_name=("uq_knowledge_document_chunks_workspace_document_version_id"),
        )
        assert await column_exists(
            engine,
            table_name="agent_runs",
            column_name="max_retryable_failures",
        )
        assert await column_exists(
            engine,
            table_name="agent_runs",
            column_name="retryable_failure_count",
        )
        assert not await column_exists(
            engine,
            table_name="agent_runs",
            column_name="max_attempts",
        )
    finally:
        await engine.dispose()


async def test_alembic_downgrade_removes_business_tables_and_can_reupgrade(
    exclusive_integration_database: None,
) -> None:
    settings = Settings()
    engine = create_async_engine(str(settings.postgresql_url))

    upgrade = run_alembic_command("upgrade", "head")
    assert upgrade.returncode == 0, upgrade.stderr

    try:
        # Approval-aware tool-call rows block downgrade through c9e2f4a7b6d1;
        # clear first so a partial downgrade cannot drop approval_requests and
        # then fail, poisoning later clean_business_tables fixtures.
        async with engine.begin() as connection:
            await clear_integration_business_data(connection)

        downgrade = run_alembic_command("downgrade", "base")
        assert downgrade.returncode == 0, downgrade.stderr

        assert not await relation_exists(engine, "public.tickets")
        assert not await relation_exists(engine, "public.workspaces")
        assert not await relation_exists(
            engine,
            "public.knowledge_documents",
        )
        assert not await relation_exists(
            engine,
            "public.knowledge_document_versions",
        )
        assert not await relation_exists(
            engine,
            "public.knowledge_document_chunks",
        )
        assert not await relation_exists(engine, "public.agent_tool_calls")
        assert not await relation_exists(engine, "public.approval_requests")
        assert not await relation_exists(
            engine,
            "public.sensitive_execution_grants",
        )
        assert not await relation_exists(
            engine,
            "public.support_recommendations",
        )
        assert not await relation_exists(
            engine,
            "public.support_recommendation_citations",
        )

        reupgrade = run_alembic_command("upgrade", "head")
        assert reupgrade.returncode == 0, reupgrade.stderr

        assert await relation_exists(engine, "public.alembic_version")
        assert await relation_exists(engine, "public.workspaces")
        assert await relation_exists(engine, "public.tickets")
        assert await relation_exists(engine, "public.knowledge_documents")
        assert await relation_exists(
            engine,
            "public.knowledge_document_versions",
        )
        assert await relation_exists(
            engine,
            "public.knowledge_document_chunks",
        )
        assert await relation_exists(engine, "public.agent_tool_calls")
        assert await relation_exists(engine, "public.approval_requests")
        assert await relation_exists(engine, "public.sensitive_execution_grants")
        assert await relation_exists(engine, "public.support_recommendations")
        assert await relation_exists(
            engine,
            "public.support_recommendation_citations",
        )
    finally:
        await engine.dispose()
        run_alembic_command("upgrade", "head")


async def test_alembic_downgrade_controlled_workflow_revision_removes_only_those_tables(
    exclusive_integration_database: None,
) -> None:
    settings = Settings()
    engine = create_async_engine(str(settings.postgresql_url))

    upgrade = run_alembic_command("upgrade", "head")
    assert upgrade.returncode == 0, upgrade.stderr

    try:
        async with engine.begin() as connection:
            await clear_integration_business_data(connection)

        downgrade = run_alembic_command(
            "downgrade",
            PRE_CONTROLLED_WORKFLOW_REVISION,
        )
        assert downgrade.returncode == 0, downgrade.stderr

        assert not await relation_exists(engine, "public.agent_tool_calls")
        assert not await relation_exists(
            engine,
            "public.support_recommendations",
        )
        assert not await relation_exists(
            engine,
            "public.support_recommendation_citations",
        )
        assert await relation_exists(engine, "public.workspaces")
        assert await relation_exists(engine, "public.tickets")
        assert await relation_exists(engine, "public.agent_runs")
        assert await relation_exists(engine, "public.agent_run_attempts")
        assert await relation_exists(engine, "public.llm_invocations")
        assert await relation_exists(
            engine,
            "public.ticket_classifications",
        )
        assert await relation_exists(engine, "public.knowledge_documents")
        assert await relation_exists(
            engine,
            "public.knowledge_document_versions",
        )
        assert await relation_exists(
            engine,
            "public.knowledge_document_chunks",
        )
        assert not await constraint_exists(
            engine,
            table_name="ticket_classifications",
            constraint_name="uq_ticket_classifications_run_id",
        )
        assert not await constraint_exists(
            engine,
            table_name="knowledge_document_chunks",
            constraint_name=("uq_knowledge_document_chunks_workspace_document_version_id"),
        )

        reupgrade = run_alembic_command("upgrade", "head")
        assert reupgrade.returncode == 0, reupgrade.stderr

        assert await relation_exists(engine, "public.agent_tool_calls")
        assert await relation_exists(engine, "public.approval_requests")
        assert await relation_exists(engine, "public.sensitive_execution_grants")
        assert await relation_exists(engine, "public.support_recommendations")
        assert await relation_exists(
            engine,
            "public.support_recommendation_citations",
        )
        assert await constraint_exists(
            engine,
            table_name="ticket_classifications",
            constraint_name="uq_ticket_classifications_run_id",
        )
        assert await constraint_exists(
            engine,
            table_name="knowledge_document_chunks",
            constraint_name=("uq_knowledge_document_chunks_workspace_document_version_id"),
        )
    finally:
        await engine.dispose()
        run_alembic_command("upgrade", "head")


async def test_alembic_retryable_failure_budget_migration_upgrades_and_downgrades(
    exclusive_integration_database: None,
) -> None:
    settings = Settings()
    engine = create_async_engine(str(settings.postgresql_url))

    upgrade_head = run_alembic_command("upgrade", "head")
    assert upgrade_head.returncode == 0, upgrade_head.stderr
    async with engine.begin() as connection:
        await clear_integration_business_data(connection)
    downgrade_to_baseline = run_alembic_command(
        "downgrade",
        CONTROLLED_WORKFLOW_REVISION,
    )
    assert downgrade_to_baseline.returncode == 0, downgrade_to_baseline.stderr

    workspace_id = uuid4()
    ticket_id = uuid4()
    run_id = uuid4()
    succeeded_attempt_id = uuid4()
    retryable_attempt_id = uuid4()
    timed_out_attempt_id = uuid4()
    lease_expired_attempt_id = uuid4()
    terminal_attempt_id = uuid4()
    exceeded_run_id = uuid4()
    exceeded_ticket_id = uuid4()

    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    INSERT INTO workspaces (
                        id, name, slug, created_at, updated_at
                    ) VALUES (
                        :workspace_id,
                        'Retry Budget Workspace',
                        'retry-budget-workspace',
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        TIMESTAMPTZ '2026-08-02 00:00:00+00'
                    )
                    """
                ),
                {"workspace_id": workspace_id},
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO tickets (
                        id, workspace_id, subject, description, status,
                        external_reference, ingestion_request_id,
                        correlation_id, created_at, updated_at
                    ) VALUES (
                        :ticket_id,
                        :workspace_id,
                        'Budget subject',
                        'Budget description',
                        'open',
                        NULL,
                        :ingestion_request_id,
                        :correlation_id,
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        TIMESTAMPTZ '2026-08-02 00:00:00+00'
                    )
                    """
                ),
                {
                    "ticket_id": ticket_id,
                    "workspace_id": workspace_id,
                    "ingestion_request_id": uuid4(),
                    "correlation_id": uuid4(),
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO tickets (
                        id, workspace_id, subject, description, status,
                        external_reference, ingestion_request_id,
                        correlation_id, created_at, updated_at
                    ) VALUES (
                        :ticket_id,
                        :workspace_id,
                        'Exceeded subject',
                        'Exceeded description',
                        'open',
                        NULL,
                        :ingestion_request_id,
                        :correlation_id,
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        TIMESTAMPTZ '2026-08-02 00:00:00+00'
                    )
                    """
                ),
                {
                    "ticket_id": exceeded_ticket_id,
                    "workspace_id": workspace_id,
                    "ingestion_request_id": uuid4(),
                    "correlation_id": uuid4(),
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_runs (
                        id, workspace_id, ticket_id, workflow_name,
                        workflow_version, trigger_key, status, available_at,
                        attempt_count, max_attempts, lease_owner, lease_token,
                        lease_expires_at, first_started_at, completed_at,
                        last_error_code, last_error_summary,
                        ingestion_request_id, correlation_id,
                        created_at, updated_at
                    ) VALUES (
                        :run_id,
                        :workspace_id,
                        :ticket_id,
                        'ticket-processing',
                        'controlled-support-v1',
                        'initial-ticket-processing',
                        'retry_scheduled',
                        TIMESTAMPTZ '2026-08-02 00:10:00+00',
                        5,
                        5,
                        NULL,
                        NULL,
                        NULL,
                        TIMESTAMPTZ '2026-08-02 00:01:00+00',
                        NULL,
                        'retryable_executor_failure',
                        'Historical retryable failure summary',
                        :ingestion_request_id,
                        :correlation_id,
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        TIMESTAMPTZ '2026-08-02 00:10:00+00'
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "ingestion_request_id": uuid4(),
                    "correlation_id": uuid4(),
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_runs (
                        id, workspace_id, ticket_id, workflow_name,
                        workflow_version, trigger_key, status, available_at,
                        attempt_count, max_attempts, lease_owner, lease_token,
                        lease_expires_at, first_started_at, completed_at,
                        last_error_code, last_error_summary,
                        ingestion_request_id, correlation_id,
                        created_at, updated_at
                    ) VALUES (
                        :run_id,
                        :workspace_id,
                        :ticket_id,
                        'ticket-processing',
                        'controlled-support-v1',
                        'exceeded-attempt-count',
                        'failed',
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        2,
                        2,
                        NULL,
                        NULL,
                        NULL,
                        TIMESTAMPTZ '2026-08-02 00:01:00+00',
                        TIMESTAMPTZ '2026-08-02 00:20:00+00',
                        'worker_lease_expired',
                        'Historical lease expiry summary',
                        :ingestion_request_id,
                        :correlation_id,
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        TIMESTAMPTZ '2026-08-02 00:20:00+00'
                    )
                    """
                ),
                {
                    "run_id": exceeded_run_id,
                    "workspace_id": workspace_id,
                    "ticket_id": exceeded_ticket_id,
                    "ingestion_request_id": uuid4(),
                    "correlation_id": uuid4(),
                },
            )

            attempts = (
                (
                    succeeded_attempt_id,
                    1,
                    "succeeded",
                    None,
                    None,
                ),
                (
                    retryable_attempt_id,
                    2,
                    "retryable_failure",
                    "retryable_executor_failure",
                    "Historical retryable failure summary",
                ),
                (
                    timed_out_attempt_id,
                    3,
                    "timed_out",
                    "executor_timeout",
                    "Historical timeout summary",
                ),
                (
                    lease_expired_attempt_id,
                    4,
                    "lease_expired",
                    "worker_lease_expired",
                    "Historical lease expiry summary",
                ),
                (
                    terminal_attempt_id,
                    5,
                    "terminal_failure",
                    "terminal_executor_failure",
                    "Historical terminal failure summary",
                ),
            )
            for (
                attempt_id,
                attempt_number,
                outcome,
                error_code,
                error_summary,
            ) in attempts:
                await connection.execute(
                    text(
                        """
                        INSERT INTO agent_run_attempts (
                            id, agent_run_id, attempt_number, worker_id,
                            lease_token, execution_request_id, started_at,
                            finished_at, outcome, error_code, error_summary
                        ) VALUES (
                            :attempt_id,
                            :run_id,
                            :attempt_number,
                            'worker-a',
                            :lease_token,
                            :execution_request_id,
                            TIMESTAMPTZ '2026-08-02 00:01:00+00',
                            TIMESTAMPTZ '2026-08-02 00:02:00+00',
                            :outcome,
                            :error_code,
                            :error_summary
                        )
                        """
                    ),
                    {
                        "attempt_id": attempt_id,
                        "run_id": run_id,
                        "attempt_number": attempt_number,
                        "lease_token": uuid4(),
                        "execution_request_id": uuid4(),
                        "outcome": outcome,
                        "error_code": error_code,
                        "error_summary": error_summary,
                    },
                )

        upgrade_head = run_alembic_command("upgrade", EXPECTED_HEAD)
        assert upgrade_head.returncode == 0, upgrade_head.stderr

        assert await column_exists(
            engine,
            table_name="agent_runs",
            column_name="max_retryable_failures",
        )
        assert await column_exists(
            engine,
            table_name="agent_runs",
            column_name="retryable_failure_count",
        )
        assert not await column_exists(
            engine,
            table_name="agent_runs",
            column_name="max_attempts",
        )
        assert await constraint_exists(
            engine,
            table_name="agent_runs",
            constraint_name=("ck_agent_runs_agent_run_max_retryable_failures_positive"),
        )
        assert await constraint_exists(
            engine,
            table_name="agent_runs",
            constraint_name=("ck_agent_runs_agent_run_retryable_failure_count_non_negative"),
        )
        assert await constraint_exists(
            engine,
            table_name="agent_runs",
            constraint_name=("ck_agent_runs_agent_run_retryable_failure_limit"),
        )
        assert not await constraint_exists(
            engine,
            table_name="agent_runs",
            constraint_name="ck_agent_runs_agent_run_attempt_limit",
        )

        async with engine.connect() as connection:
            counted = await connection.execute(
                text(
                    """
                    SELECT retryable_failure_count, max_retryable_failures
                    FROM agent_runs
                    WHERE id = :run_id
                    """
                ),
                {"run_id": run_id},
            )
            failure_count, max_failures = counted.one()
            assert failure_count == 3
            assert max_failures == 5

        async with engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    UPDATE agent_runs
                    SET attempt_count = 4,
                        max_retryable_failures = 2,
                        retryable_failure_count = 2
                    WHERE id = :run_id
                    """
                ),
                {"run_id": exceeded_run_id},
            )

        async with engine.connect() as connection:
            exceeded = await connection.execute(
                text(
                    """
                    SELECT attempt_count, max_retryable_failures
                    FROM agent_runs
                    WHERE id = :run_id
                    """
                ),
                {"run_id": exceeded_run_id},
            )
            attempt_count, exceeded_max = exceeded.one()
            assert attempt_count == 4
            assert exceeded_max == 2

        table = cast(Table, AgentRunRecord.__table__)
        constraint_names = {constraint.name for constraint in table.constraints}
        assert {
            "ck_agent_runs_agent_run_max_retryable_failures_positive",
            "ck_agent_runs_agent_run_retryable_failure_count_non_negative",
            "ck_agent_runs_agent_run_retryable_failure_limit",
        }.issubset(constraint_names)
        assert "ck_agent_runs_agent_run_attempt_limit" not in constraint_names
        column_names = {column.name for column in table.c}
        assert "max_retryable_failures" in column_names
        assert "retryable_failure_count" in column_names
        assert "max_attempts" not in column_names

        downgrade = run_alembic_command(
            "downgrade",
            CONTROLLED_WORKFLOW_REVISION,
        )
        assert downgrade.returncode == 0, downgrade.stderr

        assert await column_exists(
            engine,
            table_name="agent_runs",
            column_name="max_attempts",
        )
        assert not await column_exists(
            engine,
            table_name="agent_runs",
            column_name="retryable_failure_count",
        )
        assert not await column_exists(
            engine,
            table_name="agent_runs",
            column_name="max_retryable_failures",
        )

        async with engine.connect() as connection:
            restored = await connection.execute(
                text(
                    """
                    SELECT attempt_count, max_attempts
                    FROM agent_runs
                    WHERE id = :run_id
                    """
                ),
                {"run_id": exceeded_run_id},
            )
            attempt_count, max_attempts = restored.one()
            assert attempt_count == 4
            assert max_attempts == 4

        reupgrade = run_alembic_command("upgrade", "head")
        assert reupgrade.returncode == 0, reupgrade.stderr

        assert await column_exists(
            engine,
            table_name="agent_runs",
            column_name="max_retryable_failures",
        )
        assert await column_exists(
            engine,
            table_name="agent_runs",
            column_name="retryable_failure_count",
        )
    finally:
        await engine.dispose()
        run_alembic_command("upgrade", "head")


async def test_alembic_waiting_for_approval_migration_upgrades_and_downgrades(
    exclusive_integration_database: None,
) -> None:
    settings = Settings()
    engine = create_async_engine(str(settings.postgresql_url))

    upgrade_head = run_alembic_command("upgrade", "head")
    assert upgrade_head.returncode == 0, upgrade_head.stderr

    async with engine.begin() as connection:
        await clear_integration_business_data(connection)

    downgrade_to_budget = run_alembic_command(
        "downgrade",
        RETRYABLE_FAILURE_BUDGET_REVISION,
    )
    assert downgrade_to_budget.returncode == 0, downgrade_to_budget.stderr

    workspace_id = uuid4()
    ticket_id = uuid4()
    run_id = uuid4()
    attempt_id = uuid4()

    try:
        upgrade_waiting = run_alembic_command("upgrade", EXPECTED_HEAD)
        assert upgrade_waiting.returncode == 0, upgrade_waiting.stderr

        assert await constraint_exists(
            engine,
            table_name="agent_runs",
            constraint_name="ck_agent_runs_agent_run_available_at_state",
        )

        async with engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    INSERT INTO workspaces (
                        id, name, slug, created_at, updated_at
                    ) VALUES (
                        :workspace_id,
                        'Waiting Workspace',
                        'waiting-workspace',
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        TIMESTAMPTZ '2026-08-02 00:00:00+00'
                    )
                    """
                ),
                {"workspace_id": workspace_id},
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO tickets (
                        id, workspace_id, subject, description, status,
                        external_reference, ingestion_request_id,
                        correlation_id, created_at, updated_at
                    ) VALUES (
                        :ticket_id,
                        :workspace_id,
                        'Waiting subject',
                        'Waiting description',
                        'open',
                        NULL,
                        :ingestion_request_id,
                        :correlation_id,
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        TIMESTAMPTZ '2026-08-02 00:00:00+00'
                    )
                    """
                ),
                {
                    "ticket_id": ticket_id,
                    "workspace_id": workspace_id,
                    "ingestion_request_id": uuid4(),
                    "correlation_id": uuid4(),
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_runs (
                        id, workspace_id, ticket_id, workflow_name,
                        workflow_version, trigger_key, status, available_at,
                        attempt_count, retryable_failure_count,
                        max_retryable_failures, lease_owner, lease_token,
                        lease_expires_at, first_started_at, completed_at,
                        last_error_code, last_error_summary,
                        ingestion_request_id, correlation_id,
                        created_at, updated_at
                    ) VALUES (
                        :run_id,
                        :workspace_id,
                        :ticket_id,
                        'ticket-processing',
                        'controlled-support-v1',
                        'initial-ticket-processing',
                        'waiting_for_approval',
                        NULL,
                        1,
                        0,
                        3,
                        NULL,
                        NULL,
                        NULL,
                        TIMESTAMPTZ '2026-08-02 00:01:00+00',
                        NULL,
                        NULL,
                        NULL,
                        :ingestion_request_id,
                        :correlation_id,
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        TIMESTAMPTZ '2026-08-02 00:02:00+00'
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "ingestion_request_id": uuid4(),
                    "correlation_id": uuid4(),
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_run_attempts (
                        id, agent_run_id, attempt_number, worker_id,
                        lease_token, execution_request_id, started_at,
                        finished_at, outcome, error_code, error_summary
                    ) VALUES (
                        :attempt_id,
                        :run_id,
                        1,
                        'worker-a',
                        :lease_token,
                        :execution_request_id,
                        TIMESTAMPTZ '2026-08-02 00:01:00+00',
                        TIMESTAMPTZ '2026-08-02 00:02:00+00',
                        'awaiting_approval',
                        NULL,
                        NULL
                    )
                    """
                ),
                {
                    "attempt_id": attempt_id,
                    "run_id": run_id,
                    "lease_token": uuid4(),
                    "execution_request_id": uuid4(),
                },
            )

        async with engine.begin() as connection:
            with pytest.raises(IntegrityError):
                await connection.execute(
                    text(
                        """
                        UPDATE agent_runs
                        SET available_at = TIMESTAMPTZ '2026-08-02 00:00:00+00'
                        WHERE id = :run_id
                        """
                    ),
                    {"run_id": run_id},
                )

        async with engine.begin() as connection:
            with pytest.raises(IntegrityError):
                await connection.execute(
                    text(
                        """
                        INSERT INTO agent_runs (
                            id, workspace_id, ticket_id, workflow_name,
                            workflow_version, trigger_key, status, available_at,
                            attempt_count, retryable_failure_count,
                            max_retryable_failures, lease_owner, lease_token,
                            lease_expires_at, first_started_at, completed_at,
                            last_error_code, last_error_summary,
                            ingestion_request_id, correlation_id,
                            created_at, updated_at
                        ) VALUES (
                            :run_id,
                            :workspace_id,
                            :ticket_id,
                            'ticket-processing',
                            'controlled-support-v1',
                            'queued-null-available',
                            'queued',
                            NULL,
                            0,
                            0,
                            3,
                            NULL,
                            NULL,
                            NULL,
                            NULL,
                            NULL,
                            NULL,
                            NULL,
                            :ingestion_request_id,
                            :correlation_id,
                            TIMESTAMPTZ '2026-08-02 00:00:00+00',
                            TIMESTAMPTZ '2026-08-02 00:00:00+00'
                        )
                        """
                    ),
                    {
                        "run_id": uuid4(),
                        "workspace_id": workspace_id,
                        "ticket_id": ticket_id,
                        "ingestion_request_id": uuid4(),
                        "correlation_id": uuid4(),
                    },
                )

        downgrade_blocked = run_alembic_command(
            "downgrade",
            RETRYABLE_FAILURE_BUDGET_REVISION,
        )
        assert downgrade_blocked.returncode != 0
        assert (
            "Cannot downgrade while AgentRuns are waiting for approval" in downgrade_blocked.stderr
            or "Cannot downgrade while AgentRuns are waiting for approval"
            in downgrade_blocked.stdout
        )

        async with engine.begin() as connection:
            await connection.execute(
                text("DELETE FROM agent_run_attempts WHERE id = :attempt_id"),
                {"attempt_id": attempt_id},
            )
            await connection.execute(
                text(
                    """
                    UPDATE agent_runs
                    SET status = 'queued',
                        available_at = TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        attempt_count = 0,
                        first_started_at = NULL,
                        updated_at = TIMESTAMPTZ '2026-08-02 00:00:00+00'
                    WHERE id = :run_id
                    """
                ),
                {"run_id": run_id},
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_run_attempts (
                        id, agent_run_id, attempt_number, worker_id,
                        lease_token, execution_request_id, started_at,
                        finished_at, outcome, error_code, error_summary
                    ) VALUES (
                        :attempt_id,
                        :run_id,
                        1,
                        'worker-a',
                        :lease_token,
                        :execution_request_id,
                        TIMESTAMPTZ '2026-08-02 00:01:00+00',
                        TIMESTAMPTZ '2026-08-02 00:02:00+00',
                        'awaiting_approval',
                        NULL,
                        NULL
                    )
                    """
                ),
                {
                    "attempt_id": attempt_id,
                    "run_id": run_id,
                    "lease_token": uuid4(),
                    "execution_request_id": uuid4(),
                },
            )

        downgrade_blocked_attempt = run_alembic_command(
            "downgrade",
            RETRYABLE_FAILURE_BUDGET_REVISION,
        )
        assert downgrade_blocked_attempt.returncode != 0
        assert (
            "Cannot downgrade while awaiting-approval attempts exist"
            in downgrade_blocked_attempt.stderr
            or "Cannot downgrade while awaiting-approval attempts exist"
            in downgrade_blocked_attempt.stdout
        )

        async with engine.begin() as connection:
            await connection.execute(
                text("DELETE FROM agent_run_attempts WHERE id = :attempt_id"),
                {"attempt_id": attempt_id},
            )

        downgrade_clean = run_alembic_command(
            "downgrade",
            RETRYABLE_FAILURE_BUDGET_REVISION,
        )
        assert downgrade_clean.returncode == 0, downgrade_clean.stderr

        assert not await constraint_exists(
            engine,
            table_name="agent_runs",
            constraint_name="ck_agent_runs_agent_run_available_at_state",
        )

        reupgrade = run_alembic_command("upgrade", EXPECTED_HEAD)
        assert reupgrade.returncode == 0, reupgrade.stderr

        assert await constraint_exists(
            engine,
            table_name="agent_runs",
            constraint_name="ck_agent_runs_agent_run_available_at_state",
        )

        table = cast(Table, AgentRunRecord.__table__)
        constraint_names = {constraint.name for constraint in table.constraints}
        assert "ck_agent_runs_agent_run_available_at_state" in constraint_names
        assert "ck_agent_runs_agent_run_status" in constraint_names
        assert table.c.available_at.nullable is True
    finally:
        await engine.dispose()
        run_alembic_command("upgrade", "head")


async def index_exists(
    engine: AsyncEngine,
    *,
    table_name: str,
    index_name: str,
) -> bool:
    """Return whether a named PostgreSQL index exists."""

    async with engine.connect() as connection:
        result = await connection.execute(
            text(
                """
                SELECT 1
                FROM pg_indexes
                WHERE schemaname = 'public'
                  AND tablename = :table_name
                  AND indexname = :index_name
                """
            ),
            {
                "table_name": table_name,
                "index_name": index_name,
            },
        )
        return result.scalar_one_or_none() is not None


async def test_alembic_tool_call_lifecycle_migration_upgrades_and_downgrades(
    exclusive_integration_database: None,
) -> None:
    settings = Settings()
    engine = create_async_engine(str(settings.postgresql_url))

    upgrade_head = run_alembic_command("upgrade", "head")
    assert upgrade_head.returncode == 0, upgrade_head.stderr

    async with engine.begin() as connection:
        await clear_integration_business_data(connection)

    downgrade_to_waiting = run_alembic_command(
        "downgrade",
        WAITING_FOR_APPROVAL_REVISION,
    )
    assert downgrade_to_waiting.returncode == 0, downgrade_to_waiting.stderr

    workspace_id = uuid4()
    ticket_id = uuid4()
    run_id = uuid4()
    attempt_one_id = uuid4()
    attempt_two_id = uuid4()
    historical_tool_call_id = uuid4()
    fingerprint = "a" * 64

    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    INSERT INTO workspaces (
                        id, name, slug, created_at, updated_at
                    ) VALUES (
                        :workspace_id,
                        'Tool Call Workspace',
                        'tool-call-workspace',
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        TIMESTAMPTZ '2026-08-02 00:00:00+00'
                    )
                    """
                ),
                {"workspace_id": workspace_id},
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO tickets (
                        id, workspace_id, subject, description, status,
                        external_reference, ingestion_request_id,
                        correlation_id, created_at, updated_at
                    ) VALUES (
                        :ticket_id,
                        :workspace_id,
                        'Tool call subject',
                        'Tool call description',
                        'open',
                        NULL,
                        :ingestion_request_id,
                        :correlation_id,
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        TIMESTAMPTZ '2026-08-02 00:00:00+00'
                    )
                    """
                ),
                {
                    "ticket_id": ticket_id,
                    "workspace_id": workspace_id,
                    "ingestion_request_id": uuid4(),
                    "correlation_id": uuid4(),
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_runs (
                        id, workspace_id, ticket_id, workflow_name,
                        workflow_version, trigger_key, status, available_at,
                        attempt_count, retryable_failure_count,
                        max_retryable_failures, lease_owner, lease_token,
                        lease_expires_at, first_started_at, completed_at,
                        last_error_code, last_error_summary,
                        ingestion_request_id, correlation_id,
                        created_at, updated_at
                    ) VALUES (
                        :run_id,
                        :workspace_id,
                        :ticket_id,
                        'ticket-processing',
                        'controlled-support-v1',
                        'initial-ticket-processing',
                        'running',
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        1,
                        0,
                        3,
                        'worker-a',
                        :lease_token,
                        TIMESTAMPTZ '2026-08-02 00:10:00+00',
                        TIMESTAMPTZ '2026-08-02 00:01:00+00',
                        NULL,
                        NULL,
                        NULL,
                        :ingestion_request_id,
                        :correlation_id,
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        TIMESTAMPTZ '2026-08-02 00:01:00+00'
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "lease_token": uuid4(),
                    "ingestion_request_id": uuid4(),
                    "correlation_id": uuid4(),
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_run_attempts (
                        id, agent_run_id, attempt_number, worker_id,
                        lease_token, execution_request_id, started_at,
                        finished_at, outcome, error_code, error_summary
                    ) VALUES (
                        :attempt_id,
                        :run_id,
                        1,
                        'worker-a',
                        :lease_token,
                        :execution_request_id,
                        TIMESTAMPTZ '2026-08-02 00:01:00+00',
                        NULL,
                        NULL,
                        NULL,
                        NULL
                    )
                    """
                ),
                {
                    "attempt_id": attempt_one_id,
                    "run_id": run_id,
                    "lease_token": uuid4(),
                    "execution_request_id": uuid4(),
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_tool_calls (
                        id, workspace_id, ticket_id, agent_run_id,
                        agent_run_attempt_id, sequence, provider_tool_call_id,
                        tool_name, tool_version, safety_level, status,
                        input_fingerprint, safe_input, safe_output,
                        latency_ms, error_code, started_at, finished_at
                    ) VALUES (
                        :tool_call_id,
                        :workspace_id,
                        :ticket_id,
                        :run_id,
                        :attempt_id,
                        1,
                        'provider-tool-call-1',
                        'search_knowledge',
                        1,
                        'read_only',
                        'succeeded',
                        :fingerprint,
                        CAST(:safe_input AS jsonb),
                        CAST(:safe_output AS jsonb),
                        25,
                        NULL,
                        TIMESTAMPTZ '2026-08-02 00:01:30+00',
                        TIMESTAMPTZ '2026-08-02 00:01:30.025+00'
                    )
                    """
                ),
                {
                    "tool_call_id": historical_tool_call_id,
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "run_id": run_id,
                    "attempt_id": attempt_one_id,
                    "fingerprint": fingerprint,
                    "safe_input": '{"top_k":5}',
                    "safe_output": '{"result_count":1}',
                },
            )

        assert await column_exists(
            engine,
            table_name="agent_tool_calls",
            column_name="agent_run_attempt_id",
        )
        assert await column_exists(
            engine,
            table_name="agent_tool_calls",
            column_name="started_at",
        )

        upgrade = run_alembic_command("upgrade", EXPECTED_HEAD)
        assert upgrade.returncode == 0, upgrade.stderr

        assert await column_exists(
            engine,
            table_name="agent_tool_calls",
            column_name="proposed_by_agent_run_attempt_id",
        )
        assert await column_exists(
            engine,
            table_name="agent_tool_calls",
            column_name="executed_by_agent_run_attempt_id",
        )
        assert await column_exists(
            engine,
            table_name="agent_tool_calls",
            column_name="proposed_at",
        )
        assert await column_exists(
            engine,
            table_name="agent_tool_calls",
            column_name="execution_started_at",
        )
        assert not await column_exists(
            engine,
            table_name="agent_tool_calls",
            column_name="agent_run_attempt_id",
        )
        assert not await column_exists(
            engine,
            table_name="agent_tool_calls",
            column_name="started_at",
        )
        assert await index_exists(
            engine,
            table_name="agent_tool_calls",
            index_name=("uq_agent_tool_calls_sensitive_proposal_identity"),
        )
        assert await constraint_exists(
            engine,
            table_name="agent_tool_calls",
            constraint_name=("ck_agent_tool_calls_agent_tool_call_lifecycle_state"),
        )

        async with engine.connect() as connection:
            backfilled = await connection.execute(
                text(
                    """
                    SELECT
                        proposed_by_agent_run_attempt_id,
                        executed_by_agent_run_attempt_id,
                        proposed_at,
                        execution_started_at,
                        finished_at,
                        latency_ms
                    FROM agent_tool_calls
                    WHERE id = :tool_call_id
                    """
                ),
                {"tool_call_id": historical_tool_call_id},
            )
            (
                proposed_by,
                executed_by,
                proposed_at,
                execution_started_at,
                finished_at,
                latency_ms,
            ) = backfilled.one()
            assert proposed_by == attempt_one_id
            assert executed_by == attempt_one_id
            assert proposed_at == execution_started_at
            assert finished_at is not None
            assert latency_ms == 25

        async with engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_run_attempts (
                        id, agent_run_id, attempt_number, worker_id,
                        lease_token, execution_request_id, started_at,
                        finished_at, outcome, error_code, error_summary
                    ) VALUES (
                        :attempt_id,
                        :run_id,
                        2,
                        'worker-a',
                        :lease_token,
                        :execution_request_id,
                        TIMESTAMPTZ '2026-08-02 00:02:00+00',
                        NULL,
                        NULL,
                        NULL,
                        NULL
                    )
                    """
                ),
                {
                    "attempt_id": attempt_two_id,
                    "run_id": run_id,
                    "lease_token": uuid4(),
                    "execution_request_id": uuid4(),
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_tool_calls (
                        id, workspace_id, ticket_id, agent_run_id,
                        proposed_by_agent_run_attempt_id,
                        executed_by_agent_run_attempt_id, sequence,
                        provider_tool_call_id, tool_name, tool_version,
                        safety_level, status, input_fingerprint, safe_input,
                        safe_output, latency_ms, error_code, proposed_at,
                        execution_started_at, finished_at
                    ) VALUES (
                        :tool_call_id,
                        :workspace_id,
                        :ticket_id,
                        :run_id,
                        :attempt_id,
                        NULL,
                        2,
                        'provider-sensitive-1',
                        'escalate_ticket',
                        1,
                        'sensitive_write',
                        'pending_approval',
                        :fingerprint,
                        CAST(:safe_input AS jsonb),
                        NULL,
                        NULL,
                        NULL,
                        TIMESTAMPTZ '2026-08-02 00:02:30+00',
                        NULL,
                        NULL
                    )
                    """
                ),
                {
                    "tool_call_id": uuid4(),
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "run_id": run_id,
                    "attempt_id": attempt_one_id,
                    "fingerprint": "b" * 64,
                    "safe_input": '{"reason_code":"policy"}',
                },
            )

        async with engine.begin() as connection:
            with pytest.raises(IntegrityError):
                await connection.execute(
                    text(
                        """
                        INSERT INTO agent_tool_calls (
                            id, workspace_id, ticket_id, agent_run_id,
                            proposed_by_agent_run_attempt_id,
                            executed_by_agent_run_attempt_id, sequence,
                            provider_tool_call_id, tool_name, tool_version,
                            safety_level, status, input_fingerprint, safe_input,
                            safe_output, latency_ms, error_code, proposed_at,
                            execution_started_at, finished_at
                        ) VALUES (
                            :tool_call_id,
                            :workspace_id,
                            :ticket_id,
                            :run_id,
                            :attempt_id,
                            NULL,
                            3,
                            'provider-sensitive-2',
                            'escalate_ticket',
                            1,
                            'sensitive_write',
                            'pending_approval',
                            :fingerprint,
                            CAST(:safe_input AS jsonb),
                            NULL,
                            NULL,
                            NULL,
                            TIMESTAMPTZ '2026-08-02 00:03:00+00',
                            NULL,
                            NULL
                        )
                        """
                    ),
                    {
                        "tool_call_id": uuid4(),
                        "workspace_id": workspace_id,
                        "ticket_id": ticket_id,
                        "run_id": run_id,
                        "attempt_id": attempt_two_id,
                        "fingerprint": "b" * 64,
                        "safe_input": '{"reason_code":"policy"}',
                    },
                )

        async with engine.begin() as connection:
            with pytest.raises(IntegrityError):
                await connection.execute(
                    text(
                        """
                        INSERT INTO agent_tool_calls (
                            id, workspace_id, ticket_id, agent_run_id,
                            proposed_by_agent_run_attempt_id,
                            executed_by_agent_run_attempt_id, sequence,
                            provider_tool_call_id, tool_name, tool_version,
                            safety_level, status, input_fingerprint, safe_input,
                            safe_output, latency_ms, error_code, proposed_at,
                            execution_started_at, finished_at
                        ) VALUES (
                            :tool_call_id,
                            :workspace_id,
                            :ticket_id,
                            :run_id,
                            :attempt_id,
                            NULL,
                            4,
                            'provider-pending-readonly',
                            'search_knowledge',
                            1,
                            'read_only',
                            'pending_approval',
                            :fingerprint,
                            CAST(:safe_input AS jsonb),
                            NULL,
                            NULL,
                            NULL,
                            TIMESTAMPTZ '2026-08-02 00:03:00+00',
                            NULL,
                            NULL
                        )
                        """
                    ),
                    {
                        "tool_call_id": uuid4(),
                        "workspace_id": workspace_id,
                        "ticket_id": ticket_id,
                        "run_id": run_id,
                        "attempt_id": attempt_one_id,
                        "fingerprint": "c" * 64,
                        "safe_input": '{"top_k":1}',
                    },
                )

        async with engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_tool_calls (
                        id, workspace_id, ticket_id, agent_run_id,
                        proposed_by_agent_run_attempt_id,
                        executed_by_agent_run_attempt_id, sequence,
                        provider_tool_call_id, tool_name, tool_version,
                        safety_level, status, input_fingerprint, safe_input,
                        safe_output, latency_ms, error_code, proposed_at,
                        execution_started_at, finished_at
                    ) VALUES (
                        :tool_call_id,
                        :workspace_id,
                        :ticket_id,
                        :run_id,
                        :attempt_id,
                        :attempt_id,
                        5,
                        'provider-rejected-policy',
                        'search_knowledge',
                        1,
                        'read_only',
                        'rejected',
                        :fingerprint,
                        CAST(:safe_input AS jsonb),
                        NULL,
                        3,
                        'tool_repeated_call',
                        TIMESTAMPTZ '2026-08-02 00:03:10+00',
                        TIMESTAMPTZ '2026-08-02 00:03:10+00',
                        TIMESTAMPTZ '2026-08-02 00:03:10.003+00'
                    )
                    """
                ),
                {
                    "tool_call_id": uuid4(),
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "run_id": run_id,
                    "attempt_id": attempt_one_id,
                    "fingerprint": "d" * 64,
                    "safe_input": '{"top_k":1}',
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_tool_calls (
                        id, workspace_id, ticket_id, agent_run_id,
                        proposed_by_agent_run_attempt_id,
                        executed_by_agent_run_attempt_id, sequence,
                        provider_tool_call_id, tool_name, tool_version,
                        safety_level, status, input_fingerprint, safe_input,
                        safe_output, latency_ms, error_code, proposed_at,
                        execution_started_at, finished_at
                    ) VALUES (
                        :tool_call_id,
                        :workspace_id,
                        :ticket_id,
                        :run_id,
                        :attempt_id,
                        NULL,
                        6,
                        'provider-human-rejected',
                        'escalate_ticket',
                        1,
                        'sensitive_write',
                        'rejected',
                        :fingerprint,
                        CAST(:safe_input AS jsonb),
                        NULL,
                        NULL,
                        NULL,
                        TIMESTAMPTZ '2026-08-02 00:03:20+00',
                        NULL,
                        TIMESTAMPTZ '2026-08-02 00:04:00+00'
                    )
                    """
                ),
                {
                    "tool_call_id": uuid4(),
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "run_id": run_id,
                    "attempt_id": attempt_one_id,
                    "fingerprint": "e" * 64,
                    "safe_input": '{"reason_code":"policy"}',
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_tool_calls (
                        id, workspace_id, ticket_id, agent_run_id,
                        proposed_by_agent_run_attempt_id,
                        executed_by_agent_run_attempt_id, sequence,
                        provider_tool_call_id, tool_name, tool_version,
                        safety_level, status, input_fingerprint, safe_input,
                        safe_output, latency_ms, error_code, proposed_at,
                        execution_started_at, finished_at
                    ) VALUES (
                        :tool_call_id,
                        :workspace_id,
                        :ticket_id,
                        :run_id,
                        :attempt_id,
                        NULL,
                        7,
                        'provider-expired',
                        'escalate_ticket',
                        1,
                        'sensitive_write',
                        'expired',
                        :fingerprint,
                        CAST(:safe_input AS jsonb),
                        NULL,
                        NULL,
                        NULL,
                        TIMESTAMPTZ '2026-08-02 00:03:30+00',
                        NULL,
                        TIMESTAMPTZ '2026-08-02 00:05:00+00'
                    )
                    """
                ),
                {
                    "tool_call_id": uuid4(),
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "run_id": run_id,
                    "attempt_id": attempt_one_id,
                    "fingerprint": "f" * 64,
                    "safe_input": '{"reason_code":"policy"}',
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_tool_calls (
                        id, workspace_id, ticket_id, agent_run_id,
                        proposed_by_agent_run_attempt_id,
                        executed_by_agent_run_attempt_id, sequence,
                        provider_tool_call_id, tool_name, tool_version,
                        safety_level, status, input_fingerprint, safe_input,
                        safe_output, latency_ms, error_code, proposed_at,
                        execution_started_at, finished_at
                    ) VALUES (
                        :tool_call_id,
                        :workspace_id,
                        :ticket_id,
                        :run_id,
                        :attempt_id,
                        :attempt_id,
                        8,
                        'provider-readonly-repeat',
                        'search_knowledge',
                        1,
                        'read_only',
                        'succeeded',
                        :fingerprint,
                        CAST(:safe_input AS jsonb),
                        CAST(:safe_output AS jsonb),
                        10,
                        NULL,
                        TIMESTAMPTZ '2026-08-02 00:03:40+00',
                        TIMESTAMPTZ '2026-08-02 00:03:40+00',
                        TIMESTAMPTZ '2026-08-02 00:03:40.010+00'
                    )
                    """
                ),
                {
                    "tool_call_id": uuid4(),
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "run_id": run_id,
                    "attempt_id": attempt_two_id,
                    "fingerprint": fingerprint,
                    "safe_input": '{"top_k":5}',
                    "safe_output": '{"result_count":1}',
                },
            )

        downgrade_blocked = run_alembic_command(
            "downgrade",
            WAITING_FOR_APPROVAL_REVISION,
        )
        assert downgrade_blocked.returncode != 0
        assert (
            "Cannot downgrade approval-aware tool-call records" in downgrade_blocked.stderr
            or "Cannot downgrade approval-aware tool-call records" in downgrade_blocked.stdout
        )

        async with engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    DELETE FROM agent_tool_calls
                    WHERE status IN (
                        'pending_approval',
                        'expired'
                    )
                    OR (
                        status = 'rejected'
                        AND executed_by_agent_run_attempt_id IS NULL
                    )
                    OR executed_by_agent_run_attempt_id IS NULL
                    OR executed_by_agent_run_attempt_id
                        <> proposed_by_agent_run_attempt_id
                    OR execution_started_at IS NULL
                    OR finished_at IS NULL
                    OR latency_ms IS NULL
                    """
                )
            )

        downgrade_clean = run_alembic_command(
            "downgrade",
            WAITING_FOR_APPROVAL_REVISION,
        )
        assert downgrade_clean.returncode == 0, downgrade_clean.stderr

        assert await column_exists(
            engine,
            table_name="agent_tool_calls",
            column_name="agent_run_attempt_id",
        )
        assert await column_exists(
            engine,
            table_name="agent_tool_calls",
            column_name="started_at",
        )
        assert not await column_exists(
            engine,
            table_name="agent_tool_calls",
            column_name="proposed_at",
        )

        reupgrade = run_alembic_command("upgrade", EXPECTED_HEAD)
        assert reupgrade.returncode == 0, reupgrade.stderr

        table = cast(Table, AgentToolCallRecord.__table__)
        column_names = {column.name for column in table.c}
        assert "proposed_by_agent_run_attempt_id" in column_names
        assert "executed_by_agent_run_attempt_id" in column_names
        assert "proposed_at" in column_names
        assert "execution_started_at" in column_names
        assert table.c.latency_ms.nullable is True
        assert table.c.finished_at.nullable is True
        constraint_names = {constraint.name for constraint in table.constraints}
        assert "ck_agent_tool_calls_agent_tool_call_lifecycle_state" in (constraint_names)
        assert "uq_agent_tool_calls_proposal_attempt_sequence" in (constraint_names)
        index_names = {index.name for index in table.indexes}
        assert "uq_agent_tool_calls_sensitive_proposal_identity" in index_names
    finally:
        await engine.dispose()
        run_alembic_command("upgrade", "head")


async def test_alembic_approval_request_migration_upgrades_and_downgrades(
    exclusive_integration_database: None,
) -> None:
    settings = Settings()
    engine = create_async_engine(str(settings.postgresql_url))

    upgrade_head = run_alembic_command("upgrade", "head")
    assert upgrade_head.returncode == 0, upgrade_head.stderr

    async with engine.begin() as connection:
        await clear_integration_business_data(connection)

    downgrade_prior = run_alembic_command(
        "downgrade",
        TOOL_CALL_LIFECYCLE_REVISION,
    )
    assert downgrade_prior.returncode == 0, downgrade_prior.stderr

    assert not await relation_exists(engine, "public.approval_requests")

    workspace_id = uuid4()
    other_workspace_id = uuid4()
    ticket_id = uuid4()
    run_id = uuid4()
    attempt_id = uuid4()
    tool_call_id = uuid4()
    second_tool_call_id = uuid4()
    invocation_id = uuid4()
    other_run_invocation_id = uuid4()
    other_run_id = uuid4()
    other_attempt_id = uuid4()
    approval_id = uuid4()
    fingerprint = "a" * 64

    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    INSERT INTO workspaces (
                        id, name, slug, created_at, updated_at
                    ) VALUES (
                        :workspace_id,
                        'Approval Workspace',
                        'approval-workspace',
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        TIMESTAMPTZ '2026-08-02 00:00:00+00'
                    )
                    """
                ),
                {"workspace_id": workspace_id},
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO workspaces (
                        id, name, slug, created_at, updated_at
                    ) VALUES (
                        :workspace_id,
                        'Other Approval Workspace',
                        'other-approval-workspace',
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        TIMESTAMPTZ '2026-08-02 00:00:00+00'
                    )
                    """
                ),
                {"workspace_id": other_workspace_id},
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO tickets (
                        id, workspace_id, subject, description, status,
                        external_reference, ingestion_request_id,
                        correlation_id, created_at, updated_at
                    ) VALUES (
                        :ticket_id,
                        :workspace_id,
                        'Approval subject',
                        'Approval description',
                        'open',
                        NULL,
                        :ingestion_request_id,
                        :correlation_id,
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        TIMESTAMPTZ '2026-08-02 00:00:00+00'
                    )
                    """
                ),
                {
                    "ticket_id": ticket_id,
                    "workspace_id": workspace_id,
                    "ingestion_request_id": uuid4(),
                    "correlation_id": uuid4(),
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_runs (
                        id, workspace_id, ticket_id, workflow_name,
                        workflow_version, trigger_key, status, available_at,
                        attempt_count, retryable_failure_count,
                        max_retryable_failures, lease_owner, lease_token,
                        lease_expires_at, first_started_at, completed_at,
                        last_error_code, last_error_summary,
                        ingestion_request_id, correlation_id,
                        created_at, updated_at
                    ) VALUES (
                        :run_id,
                        :workspace_id,
                        :ticket_id,
                        'ticket-processing',
                        'controlled-support-v1',
                        'initial-ticket-processing',
                        'running',
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        1,
                        0,
                        3,
                        'worker-a',
                        :lease_token,
                        TIMESTAMPTZ '2026-08-02 00:10:00+00',
                        TIMESTAMPTZ '2026-08-02 00:01:00+00',
                        NULL,
                        NULL,
                        NULL,
                        :ingestion_request_id,
                        :correlation_id,
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        TIMESTAMPTZ '2026-08-02 00:01:00+00'
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "lease_token": uuid4(),
                    "ingestion_request_id": uuid4(),
                    "correlation_id": uuid4(),
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_runs (
                        id, workspace_id, ticket_id, workflow_name,
                        workflow_version, trigger_key, status, available_at,
                        attempt_count, retryable_failure_count,
                        max_retryable_failures, lease_owner, lease_token,
                        lease_expires_at, first_started_at, completed_at,
                        last_error_code, last_error_summary,
                        ingestion_request_id, correlation_id,
                        created_at, updated_at
                    ) VALUES (
                        :run_id,
                        :workspace_id,
                        :ticket_id,
                        'ticket-processing',
                        'controlled-support-v1',
                        'retry-ticket-processing',
                        'running',
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        1,
                        0,
                        3,
                        'worker-b',
                        :lease_token,
                        TIMESTAMPTZ '2026-08-02 00:10:00+00',
                        TIMESTAMPTZ '2026-08-02 00:02:00+00',
                        NULL,
                        NULL,
                        NULL,
                        :ingestion_request_id,
                        :correlation_id,
                        TIMESTAMPTZ '2026-08-02 00:00:00+00',
                        TIMESTAMPTZ '2026-08-02 00:02:00+00'
                    )
                    """
                ),
                {
                    "run_id": other_run_id,
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "lease_token": uuid4(),
                    "ingestion_request_id": uuid4(),
                    "correlation_id": uuid4(),
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_run_attempts (
                        id, agent_run_id, attempt_number, worker_id,
                        lease_token, execution_request_id, started_at,
                        finished_at, outcome, error_code, error_summary
                    ) VALUES (
                        :attempt_id,
                        :run_id,
                        1,
                        'worker-a',
                        :lease_token,
                        :execution_request_id,
                        TIMESTAMPTZ '2026-08-02 00:01:00+00',
                        NULL,
                        NULL,
                        NULL,
                        NULL
                    )
                    """
                ),
                {
                    "attempt_id": attempt_id,
                    "run_id": run_id,
                    "lease_token": uuid4(),
                    "execution_request_id": uuid4(),
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_run_attempts (
                        id, agent_run_id, attempt_number, worker_id,
                        lease_token, execution_request_id, started_at,
                        finished_at, outcome, error_code, error_summary
                    ) VALUES (
                        :attempt_id,
                        :run_id,
                        1,
                        'worker-b',
                        :lease_token,
                        :execution_request_id,
                        TIMESTAMPTZ '2026-08-02 00:02:00+00',
                        NULL,
                        NULL,
                        NULL,
                        NULL
                    )
                    """
                ),
                {
                    "attempt_id": other_attempt_id,
                    "run_id": other_run_id,
                    "lease_token": uuid4(),
                    "execution_request_id": uuid4(),
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO llm_invocations (
                        id, workspace_id, ticket_id, agent_run_id,
                        agent_run_attempt_id, invocation_sequence, status,
                        provider, model, provider_request_id, prompt_id,
                        prompt_version, prompt_content_hash, schema_version,
                        input_tokens, cached_input_tokens, output_tokens,
                        reasoning_tokens, total_tokens,
                        pricing_catalog_version, pricing_found,
                        estimated_input_cost_usd,
                        estimated_cached_input_cost_usd,
                        estimated_output_cost_usd,
                        estimated_total_cost_usd, latency_ms, error_code,
                        created_at
                    ) VALUES (
                        :invocation_id,
                        :workspace_id,
                        :ticket_id,
                        :run_id,
                        :attempt_id,
                        1,
                        'timed_out',
                        'mock',
                        'mock-model',
                        'mock-request-1',
                        'controlled-support',
                        1,
                        :prompt_hash,
                        'ticket-classification-v1',
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        'supportops-pricing-2026-08-01',
                        TRUE,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        12000,
                        'llm_timeout',
                        TIMESTAMPTZ '2026-08-02 00:01:30+00'
                    )
                    """
                ),
                {
                    "invocation_id": invocation_id,
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "prompt_hash": "b" * 64,
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO llm_invocations (
                        id, workspace_id, ticket_id, agent_run_id,
                        agent_run_attempt_id, invocation_sequence, status,
                        provider, model, provider_request_id, prompt_id,
                        prompt_version, prompt_content_hash, schema_version,
                        input_tokens, cached_input_tokens, output_tokens,
                        reasoning_tokens, total_tokens,
                        pricing_catalog_version, pricing_found,
                        estimated_input_cost_usd,
                        estimated_cached_input_cost_usd,
                        estimated_output_cost_usd,
                        estimated_total_cost_usd, latency_ms, error_code,
                        created_at
                    ) VALUES (
                        :invocation_id,
                        :workspace_id,
                        :ticket_id,
                        :run_id,
                        :attempt_id,
                        1,
                        'timed_out',
                        'mock',
                        'mock-model',
                        'mock-request-2',
                        'controlled-support',
                        1,
                        :prompt_hash,
                        'ticket-classification-v1',
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        'supportops-pricing-2026-08-01',
                        TRUE,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        12000,
                        'llm_timeout',
                        TIMESTAMPTZ '2026-08-02 00:02:30+00'
                    )
                    """
                ),
                {
                    "invocation_id": other_run_invocation_id,
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "run_id": other_run_id,
                    "attempt_id": other_attempt_id,
                    "prompt_hash": "c" * 64,
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_tool_calls (
                        id, workspace_id, ticket_id, agent_run_id,
                        proposed_by_agent_run_attempt_id,
                        executed_by_agent_run_attempt_id, sequence,
                        provider_tool_call_id, tool_name, tool_version,
                        safety_level, status, input_fingerprint, safe_input,
                        safe_output, latency_ms, error_code, proposed_at,
                        execution_started_at, finished_at
                    ) VALUES (
                        :tool_call_id,
                        :workspace_id,
                        :ticket_id,
                        :run_id,
                        :attempt_id,
                        NULL,
                        1,
                        'provider-sensitive-1',
                        'escalate_ticket',
                        1,
                        'sensitive_write',
                        'pending_approval',
                        :fingerprint,
                        CAST(:safe_input AS jsonb),
                        NULL,
                        NULL,
                        NULL,
                        TIMESTAMPTZ '2026-08-02 00:01:40+00',
                        NULL,
                        NULL
                    )
                    """
                ),
                {
                    "tool_call_id": tool_call_id,
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "fingerprint": fingerprint,
                    "safe_input": '{"reason_code":"policy_required"}',
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_tool_calls (
                        id, workspace_id, ticket_id, agent_run_id,
                        proposed_by_agent_run_attempt_id,
                        executed_by_agent_run_attempt_id, sequence,
                        provider_tool_call_id, tool_name, tool_version,
                        safety_level, status, input_fingerprint, safe_input,
                        safe_output, latency_ms, error_code, proposed_at,
                        execution_started_at, finished_at
                    ) VALUES (
                        :tool_call_id,
                        :workspace_id,
                        :ticket_id,
                        :run_id,
                        :attempt_id,
                        NULL,
                        2,
                        'provider-sensitive-2',
                        'escalate_ticket',
                        1,
                        'sensitive_write',
                        'pending_approval',
                        :fingerprint,
                        CAST(:safe_input AS jsonb),
                        NULL,
                        NULL,
                        NULL,
                        TIMESTAMPTZ '2026-08-02 00:01:50+00',
                        NULL,
                        NULL
                    )
                    """
                ),
                {
                    "tool_call_id": second_tool_call_id,
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "fingerprint": "d" * 64,
                    "safe_input": '{"reason_code":"alternate"}',
                },
            )

        upgrade = run_alembic_command("upgrade", EXPECTED_HEAD)
        assert upgrade.returncode == 0, upgrade.stderr

        assert await relation_exists(engine, "public.approval_requests")
        for column_name in (
            "id",
            "workspace_id",
            "ticket_id",
            "agent_run_id",
            "agent_tool_call_id",
            "requested_by_llm_invocation_id",
            "status",
            "tool_name",
            "tool_version",
            "safety_level",
            "input_fingerprint",
            "proposed_input",
            "request_reason",
            "expires_at",
            "decision_actor_reference",
            "decision_comment",
            "decision_request_id",
            "decision_correlation_id",
            "decided_at",
            "created_at",
            "updated_at",
        ):
            assert await column_exists(
                engine,
                table_name="approval_requests",
                column_name=column_name,
            )

        assert await index_exists(
            engine,
            table_name="approval_requests",
            index_name="ix_approval_requests_pending_expiration",
        )

        async with engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    INSERT INTO approval_requests (
                        id, workspace_id, ticket_id, agent_run_id,
                        agent_tool_call_id, requested_by_llm_invocation_id,
                        status, tool_name, tool_version, safety_level,
                        input_fingerprint, proposed_input, request_reason,
                        expires_at, decision_actor_reference, decision_comment,
                        decision_request_id, decision_correlation_id,
                        decided_at, created_at, updated_at
                    ) VALUES (
                        :approval_id,
                        :workspace_id,
                        :ticket_id,
                        :run_id,
                        :tool_call_id,
                        :invocation_id,
                        'pending',
                        'escalate_ticket',
                        1,
                        'sensitive_write',
                        :fingerprint,
                        CAST(:proposed_input AS jsonb),
                        'Requires human review.',
                        TIMESTAMPTZ '2026-08-03 00:00:00+00',
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
                        TIMESTAMPTZ '2026-08-02 00:02:00+00',
                        TIMESTAMPTZ '2026-08-02 00:02:00+00'
                    )
                    """
                ),
                {
                    "approval_id": approval_id,
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "run_id": run_id,
                    "tool_call_id": tool_call_id,
                    "invocation_id": invocation_id,
                    "fingerprint": fingerprint,
                    "proposed_input": '{"reason_code":"policy_required"}',
                },
            )

        async with engine.begin() as connection:
            with pytest.raises(IntegrityError):
                await connection.execute(
                    text(
                        """
                        INSERT INTO approval_requests (
                            id, workspace_id, ticket_id, agent_run_id,
                            agent_tool_call_id,
                            requested_by_llm_invocation_id, status,
                            tool_name, tool_version, safety_level,
                            input_fingerprint, proposed_input,
                            request_reason, expires_at,
                            decision_actor_reference, decision_comment,
                            decision_request_id, decision_correlation_id,
                            decided_at, created_at, updated_at
                        ) VALUES (
                            :approval_id,
                            :workspace_id,
                            :ticket_id,
                            :run_id,
                            :tool_call_id,
                            :invocation_id,
                            'pending',
                            'escalate_ticket',
                            1,
                            'sensitive_write',
                            :fingerprint,
                            CAST(:proposed_input AS jsonb),
                            'Duplicate tool call approval.',
                            TIMESTAMPTZ '2026-08-03 00:00:00+00',
                            NULL,
                            NULL,
                            NULL,
                            NULL,
                            NULL,
                            TIMESTAMPTZ '2026-08-02 00:02:00+00',
                            TIMESTAMPTZ '2026-08-02 00:02:00+00'
                        )
                        """
                    ),
                    {
                        "approval_id": uuid4(),
                        "workspace_id": workspace_id,
                        "ticket_id": ticket_id,
                        "run_id": run_id,
                        "tool_call_id": tool_call_id,
                        "invocation_id": invocation_id,
                        "fingerprint": fingerprint,
                        "proposed_input": '{"reason_code":"policy_required"}',
                    },
                )

        async with engine.begin() as connection:
            with pytest.raises(IntegrityError):
                await connection.execute(
                    text(
                        """
                        INSERT INTO approval_requests (
                            id, workspace_id, ticket_id, agent_run_id,
                            agent_tool_call_id,
                            requested_by_llm_invocation_id, status,
                            tool_name, tool_version, safety_level,
                            input_fingerprint, proposed_input,
                            request_reason, expires_at,
                            decision_actor_reference, decision_comment,
                            decision_request_id, decision_correlation_id,
                            decided_at, created_at, updated_at
                        ) VALUES (
                            :approval_id,
                            :other_workspace_id,
                            :ticket_id,
                            :run_id,
                            :tool_call_id,
                            :invocation_id,
                            'pending',
                            'escalate_ticket',
                            1,
                            'sensitive_write',
                            :fingerprint,
                            CAST(:proposed_input AS jsonb),
                            'Cross-workspace id reuse.',
                            TIMESTAMPTZ '2026-08-03 00:00:00+00',
                            NULL,
                            NULL,
                            NULL,
                            NULL,
                            NULL,
                            TIMESTAMPTZ '2026-08-02 00:02:00+00',
                            TIMESTAMPTZ '2026-08-02 00:02:00+00'
                        )
                        """
                    ),
                    {
                        "approval_id": approval_id,
                        "other_workspace_id": other_workspace_id,
                        "ticket_id": ticket_id,
                        "run_id": run_id,
                        "tool_call_id": second_tool_call_id,
                        "invocation_id": invocation_id,
                        "fingerprint": fingerprint,
                        "proposed_input": '{"reason_code":"policy_required"}',
                    },
                )

        assert await constraint_exists(
            engine,
            table_name="approval_requests",
            constraint_name="uq_approval_requests_workspace_id",
        )
        assert await constraint_exists(
            engine,
            table_name="approval_requests",
            constraint_name="uq_approval_requests_agent_tool_call",
        )

        async def assert_insert_rejected(values: dict[str, object]) -> None:
            async with engine.begin() as connection:
                with pytest.raises(IntegrityError):
                    await connection.execute(
                        text(
                            """
                            INSERT INTO approval_requests (
                                id, workspace_id, ticket_id, agent_run_id,
                                agent_tool_call_id,
                                requested_by_llm_invocation_id, status,
                                tool_name, tool_version, safety_level,
                                input_fingerprint, proposed_input,
                                request_reason, expires_at,
                                decision_actor_reference, decision_comment,
                                decision_request_id,
                                decision_correlation_id, decided_at,
                                created_at, updated_at
                            ) VALUES (
                                :id,
                                :workspace_id,
                                :ticket_id,
                                :agent_run_id,
                                :agent_tool_call_id,
                                :requested_by_llm_invocation_id,
                                :status,
                                :tool_name,
                                :tool_version,
                                :safety_level,
                                :input_fingerprint,
                                CAST(:proposed_input AS jsonb),
                                :request_reason,
                                :expires_at,
                                :decision_actor_reference,
                                :decision_comment,
                                :decision_request_id,
                                :decision_correlation_id,
                                :decided_at,
                                :created_at,
                                :updated_at
                            )
                            """
                        ),
                        values,
                    )

        created_at = datetime(2026, 8, 2, 0, 2, tzinfo=UTC)
        expires_at = datetime(2026, 8, 3, 0, 0, tzinfo=UTC)
        decided_at = datetime(2026, 8, 2, 0, 3, tzinfo=UTC)

        base_values: dict[str, object] = {
            "id": uuid4(),
            "workspace_id": workspace_id,
            "ticket_id": ticket_id,
            "agent_run_id": run_id,
            "agent_tool_call_id": second_tool_call_id,
            "requested_by_llm_invocation_id": invocation_id,
            "status": "pending",
            "tool_name": "escalate_ticket",
            "tool_version": 1,
            "safety_level": "sensitive_write",
            "input_fingerprint": "d" * 64,
            "proposed_input": '{"reason_code":"alternate"}',
            "request_reason": "Requires human review.",
            "expires_at": expires_at,
            "decision_actor_reference": None,
            "decision_comment": None,
            "decision_request_id": None,
            "decision_correlation_id": None,
            "decided_at": None,
            "created_at": created_at,
            "updated_at": created_at,
        }

        await assert_insert_rejected({**base_values, "status": "queued"})
        await assert_insert_rejected({**base_values, "safety_level": "read_only"})
        await assert_insert_rejected({**base_values, "proposed_input": '["array"]'})
        await assert_insert_rejected(
            {
                **base_values,
                "proposed_input": '{"value":"' + ("x" * 9000) + '"}',
            }
        )
        await assert_insert_rejected(
            {
                **base_values,
                "expires_at": created_at,
            }
        )
        await assert_insert_rejected(
            {
                **base_values,
                "decision_actor_reference": "operator:alice",
            }
        )
        await assert_insert_rejected(
            {
                **base_values,
                "status": "approved",
                "decision_actor_reference": "operator:alice",
                "decision_request_id": None,
                "decision_correlation_id": uuid4(),
                "decided_at": decided_at,
                "updated_at": decided_at,
            }
        )
        await assert_insert_rejected(
            {
                **base_values,
                "status": "rejected",
                "decision_actor_reference": "operator:bob",
                "decision_comment": None,
                "decision_request_id": uuid4(),
                "decision_correlation_id": uuid4(),
                "decided_at": decided_at,
                "updated_at": decided_at,
            }
        )
        await assert_insert_rejected(
            {
                **base_values,
                "status": "expired",
                "decision_actor_reference": "operator:alice",
                "decided_at": decided_at,
                "updated_at": decided_at,
            }
        )
        await assert_insert_rejected(
            {
                **base_values,
                "status": "expired",
                "decision_actor_reference": "system:approval-expiration",
                "decision_comment": "expired",
                "decided_at": decided_at,
                "updated_at": decided_at,
            }
        )
        await assert_insert_rejected(
            {
                **base_values,
                "requested_by_llm_invocation_id": other_run_invocation_id,
            }
        )
        await assert_insert_rejected(
            {
                **base_values,
                "agent_tool_call_id": uuid4(),
            }
        )
        await assert_insert_rejected(
            {
                **base_values,
                "ticket_id": uuid4(),
            }
        )

        async with engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    INSERT INTO approval_requests (
                        id, workspace_id, ticket_id, agent_run_id,
                        agent_tool_call_id, requested_by_llm_invocation_id,
                        status, tool_name, tool_version, safety_level,
                        input_fingerprint, proposed_input, request_reason,
                        expires_at, decision_actor_reference, decision_comment,
                        decision_request_id, decision_correlation_id,
                        decided_at, created_at, updated_at
                    ) VALUES (
                        :approval_id,
                        :workspace_id,
                        :ticket_id,
                        :run_id,
                        :tool_call_id,
                        :invocation_id,
                        'approved',
                        'escalate_ticket',
                        1,
                        'sensitive_write',
                        :fingerprint,
                        CAST(:proposed_input AS jsonb),
                        'Requires human review.',
                        TIMESTAMPTZ '2026-08-03 00:00:00+00',
                        'operator:alice',
                        NULL,
                        :decision_request_id,
                        :decision_correlation_id,
                        TIMESTAMPTZ '2026-08-02 00:03:00+00',
                        TIMESTAMPTZ '2026-08-02 00:02:00+00',
                        TIMESTAMPTZ '2026-08-02 00:03:00+00'
                    )
                    """
                ),
                {
                    "approval_id": uuid4(),
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "run_id": run_id,
                    "tool_call_id": second_tool_call_id,
                    "invocation_id": invocation_id,
                    "fingerprint": "d" * 64,
                    "proposed_input": '{"reason_code":"alternate"}',
                    "decision_request_id": uuid4(),
                    "decision_correlation_id": uuid4(),
                },
            )

        downgrade = run_alembic_command(
            "downgrade",
            TOOL_CALL_LIFECYCLE_REVISION,
        )
        assert downgrade.returncode == 0, downgrade.stderr
        assert not await relation_exists(engine, "public.approval_requests")

        reupgrade = run_alembic_command("upgrade", EXPECTED_HEAD)
        assert reupgrade.returncode == 0, reupgrade.stderr
        assert await relation_exists(engine, "public.approval_requests")

        table = cast(Table, ApprovalRequestRecord.__table__)
        column_names = {column.name for column in table.c}
        assert "agent_tool_call_id" in column_names
        assert "requested_by_llm_invocation_id" in column_names
        assert "proposed_input" in column_names
        constraint_names = {constraint.name for constraint in table.constraints}
        assert "ck_approval_requests_approval_request_decision_state" in (constraint_names)
        assert "uq_approval_requests_agent_tool_call" in constraint_names
        index_names = {index.name for index in table.indexes}
        assert "ix_approval_requests_pending_expiration" in index_names
    finally:
        await engine.dispose()
        run_alembic_command("upgrade", "head")


async def test_alembic_sensitive_execution_grant_migration_upgrades_and_downgrades(
    exclusive_integration_database: None,
) -> None:
    settings = Settings()
    engine = create_async_engine(str(settings.postgresql_url))

    upgrade_head = run_alembic_command("upgrade", "head")
    assert upgrade_head.returncode == 0, upgrade_head.stderr

    async with engine.begin() as connection:
        await clear_integration_business_data(connection)

    downgrade_prior = run_alembic_command(
        "downgrade",
        APPROVAL_REQUEST_REVISION,
    )
    assert downgrade_prior.returncode == 0, downgrade_prior.stderr
    assert not await relation_exists(
        engine,
        "public.sensitive_execution_grants",
    )

    workspace_id = uuid4()
    ticket_id = uuid4()
    run_id = uuid4()
    attempt_id = uuid4()
    tool_call_id = uuid4()
    second_tool_call_id = uuid4()
    invocation_id = uuid4()
    approval_id = uuid4()
    second_approval_id = uuid4()
    fingerprint = "a" * 64
    lease_token = uuid4()

    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    INSERT INTO workspaces (
                        id, name, slug, created_at, updated_at
                    ) VALUES (
                        :workspace_id,
                        'Grant Workspace',
                        'grant-workspace',
                        TIMESTAMPTZ '2026-08-03 00:00:00+00',
                        TIMESTAMPTZ '2026-08-03 00:00:00+00'
                    )
                    """
                ),
                {"workspace_id": workspace_id},
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO tickets (
                        id, workspace_id, subject, description, status,
                        external_reference, ingestion_request_id,
                        correlation_id, created_at, updated_at
                    ) VALUES (
                        :ticket_id,
                        :workspace_id,
                        'Grant subject',
                        'Grant description',
                        'open',
                        NULL,
                        :ingestion_request_id,
                        :correlation_id,
                        TIMESTAMPTZ '2026-08-03 00:00:00+00',
                        TIMESTAMPTZ '2026-08-03 00:00:00+00'
                    )
                    """
                ),
                {
                    "ticket_id": ticket_id,
                    "workspace_id": workspace_id,
                    "ingestion_request_id": uuid4(),
                    "correlation_id": uuid4(),
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_runs (
                        id, workspace_id, ticket_id, workflow_name,
                        workflow_version, trigger_key, status, available_at,
                        attempt_count, retryable_failure_count,
                        max_retryable_failures, lease_owner, lease_token,
                        lease_expires_at, first_started_at, completed_at,
                        last_error_code, last_error_summary,
                        ingestion_request_id, correlation_id,
                        created_at, updated_at
                    ) VALUES (
                        :run_id,
                        :workspace_id,
                        :ticket_id,
                        'ticket-processing',
                        'controlled-support-v1',
                        'initial-ticket-processing',
                        'running',
                        TIMESTAMPTZ '2026-08-03 00:00:00+00',
                        1,
                        0,
                        3,
                        'worker-a',
                        :lease_token,
                        TIMESTAMPTZ '2026-08-03 00:10:00+00',
                        TIMESTAMPTZ '2026-08-03 00:01:00+00',
                        NULL,
                        NULL,
                        NULL,
                        :ingestion_request_id,
                        :correlation_id,
                        TIMESTAMPTZ '2026-08-03 00:00:00+00',
                        TIMESTAMPTZ '2026-08-03 00:01:00+00'
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "lease_token": lease_token,
                    "ingestion_request_id": uuid4(),
                    "correlation_id": uuid4(),
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO agent_run_attempts (
                        id, agent_run_id, attempt_number, worker_id,
                        lease_token, execution_request_id, started_at,
                        finished_at, outcome, error_code, error_summary
                    ) VALUES (
                        :attempt_id,
                        :run_id,
                        1,
                        'worker-a',
                        :lease_token,
                        :execution_request_id,
                        TIMESTAMPTZ '2026-08-03 00:01:00+00',
                        NULL,
                        NULL,
                        NULL,
                        NULL
                    )
                    """
                ),
                {
                    "attempt_id": attempt_id,
                    "run_id": run_id,
                    "lease_token": lease_token,
                    "execution_request_id": uuid4(),
                },
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO llm_invocations (
                        id, workspace_id, ticket_id, agent_run_id,
                        agent_run_attempt_id, invocation_sequence, status,
                        provider, model, provider_request_id, prompt_id,
                        prompt_version, prompt_content_hash, schema_version,
                        input_tokens, cached_input_tokens, output_tokens,
                        reasoning_tokens, total_tokens,
                        pricing_catalog_version, pricing_found,
                        estimated_input_cost_usd,
                        estimated_cached_input_cost_usd,
                        estimated_output_cost_usd,
                        estimated_total_cost_usd, latency_ms, error_code,
                        created_at
                    ) VALUES (
                        :invocation_id,
                        :workspace_id,
                        :ticket_id,
                        :run_id,
                        :attempt_id,
                        1,
                        'timed_out',
                        'mock',
                        'mock-model',
                        'mock-request-1',
                        'controlled-support',
                        1,
                        :prompt_hash,
                        'ticket-classification-v1',
                        NULL, NULL, NULL, NULL, NULL,
                        'supportops-pricing-2026-08-01',
                        TRUE,
                        NULL, NULL, NULL, NULL,
                        12000,
                        'llm_timeout',
                        TIMESTAMPTZ '2026-08-03 00:02:00+00'
                    )
                    """
                ),
                {
                    "invocation_id": invocation_id,
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "prompt_hash": "d" * 64,
                },
            )
            for call_id, seq, provider_id, fp, safe_input in (
                (
                    tool_call_id,
                    1,
                    "provider-1",
                    fingerprint,
                    '{"target_queue":"engineering_support"}',
                ),
                (
                    second_tool_call_id,
                    2,
                    "provider-2",
                    "b" * 64,
                    '{"target_queue":"support_operations"}',
                ),
            ):
                await connection.execute(
                    text(
                        """
                        INSERT INTO agent_tool_calls (
                            id, workspace_id, ticket_id, agent_run_id,
                            proposed_by_agent_run_attempt_id,
                            executed_by_agent_run_attempt_id, sequence,
                            provider_tool_call_id, tool_name, tool_version,
                            safety_level, status, input_fingerprint,
                            safe_input, safe_output, latency_ms, error_code,
                            proposed_at, execution_started_at, finished_at
                        ) VALUES (
                            :tool_call_id,
                            :workspace_id,
                            :ticket_id,
                            :run_id,
                            :attempt_id,
                            NULL,
                            :sequence,
                            :provider_id,
                            'escalate_ticket',
                            1,
                            'sensitive_write',
                            'pending_approval',
                            :fingerprint,
                            CAST(:safe_input AS jsonb),
                            NULL,
                            NULL,
                            NULL,
                            TIMESTAMPTZ '2026-08-03 00:02:00+00',
                            NULL,
                            NULL
                        )
                        """
                    ),
                    {
                        "tool_call_id": call_id,
                        "workspace_id": workspace_id,
                        "ticket_id": ticket_id,
                        "run_id": run_id,
                        "attempt_id": attempt_id,
                        "sequence": seq,
                        "provider_id": provider_id,
                        "fingerprint": fp,
                        "safe_input": safe_input,
                    },
                )
            for approval, call_id, fp, proposed in (
                (
                    approval_id,
                    tool_call_id,
                    fingerprint,
                    '{"target_queue":"engineering_support"}',
                ),
                (
                    second_approval_id,
                    second_tool_call_id,
                    "b" * 64,
                    '{"target_queue":"support_operations"}',
                ),
            ):
                await connection.execute(
                    text(
                        """
                        INSERT INTO approval_requests (
                            id, workspace_id, ticket_id, agent_run_id,
                            agent_tool_call_id, requested_by_llm_invocation_id,
                            status, tool_name, tool_version, safety_level,
                            input_fingerprint, proposed_input, request_reason,
                            expires_at, decision_actor_reference,
                            decision_comment, decision_request_id,
                            decision_correlation_id, decided_at,
                            created_at, updated_at
                        ) VALUES (
                            :approval_id,
                            :workspace_id,
                            :ticket_id,
                            :run_id,
                            :tool_call_id,
                            :invocation_id,
                            'approved',
                            'escalate_ticket',
                            1,
                            'sensitive_write',
                            :fingerprint,
                            CAST(:proposed_input AS jsonb),
                            'Requires human review.',
                            TIMESTAMPTZ '2026-08-04 00:00:00+00',
                            'operator:alice',
                            NULL,
                            :decision_request_id,
                            :decision_correlation_id,
                            TIMESTAMPTZ '2026-08-03 00:05:00+00',
                            TIMESTAMPTZ '2026-08-03 00:03:00+00',
                            TIMESTAMPTZ '2026-08-03 00:05:00+00'
                        )
                        """
                    ),
                    {
                        "approval_id": approval,
                        "workspace_id": workspace_id,
                        "ticket_id": ticket_id,
                        "run_id": run_id,
                        "tool_call_id": call_id,
                        "invocation_id": invocation_id,
                        "fingerprint": fp,
                        "proposed_input": proposed,
                        "decision_request_id": uuid4(),
                        "decision_correlation_id": uuid4(),
                    },
                )

        upgrade = run_alembic_command("upgrade", EXPECTED_HEAD)
        assert upgrade.returncode == 0, upgrade.stderr
        assert await relation_exists(
            engine,
            "public.sensitive_execution_grants",
        )

        for column_name in (
            "id",
            "workspace_id",
            "ticket_id",
            "agent_run_id",
            "executed_by_agent_run_attempt_id",
            "approval_request_id",
            "agent_tool_call_id",
            "tool_name",
            "tool_version",
            "safety_level",
            "input_fingerprint",
            "granted_input",
            "decision_actor_reference",
            "decision_request_id",
            "decision_correlation_id",
            "approved_at",
            "created_at",
        ):
            assert await column_exists(
                engine,
                table_name="sensitive_execution_grants",
                column_name=column_name,
            )

        for constraint_name in (
            "ck_sensitive_execution_grants_safety_level",
            "ck_sensitive_execution_grants_input_fingerprint",
            "ck_sensitive_execution_grants_granted_input_object",
            "ck_sensitive_execution_grants_granted_input_size",
            "ck_sensitive_execution_grants_creation_order",
            "uq_sensitive_execution_grants_approval_request",
            "uq_sensitive_execution_grants_agent_tool_call",
            "uq_sensitive_execution_grants_workspace_id",
            "fk_sensitive_execution_grants_workspace_ticket_agent_run",
            "fk_sensitive_execution_grants_execution_attempt",
            "fk_sensitive_execution_grants_approval_request",
            "fk_sensitive_execution_grants_agent_tool_call",
        ):
            assert await constraint_exists(
                engine,
                table_name="sensitive_execution_grants",
                constraint_name=constraint_name,
            )

        grant_id = uuid4()
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    INSERT INTO sensitive_execution_grants (
                        id, workspace_id, ticket_id, agent_run_id,
                        executed_by_agent_run_attempt_id,
                        approval_request_id, agent_tool_call_id,
                        tool_name, tool_version, safety_level,
                        input_fingerprint, granted_input,
                        decision_actor_reference, decision_request_id,
                        decision_correlation_id, approved_at, created_at
                    ) VALUES (
                        :grant_id,
                        :workspace_id,
                        :ticket_id,
                        :run_id,
                        :attempt_id,
                        :approval_id,
                        :tool_call_id,
                        'escalate_ticket',
                        1,
                        'sensitive_write',
                        :fingerprint,
                        CAST(:granted_input AS jsonb),
                        'operator:alice',
                        :decision_request_id,
                        :decision_correlation_id,
                        TIMESTAMPTZ '2026-08-03 00:05:00+00',
                        TIMESTAMPTZ '2026-08-03 00:06:00+00'
                    )
                    """
                ),
                {
                    "grant_id": grant_id,
                    "workspace_id": workspace_id,
                    "ticket_id": ticket_id,
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "approval_id": approval_id,
                    "tool_call_id": tool_call_id,
                    "fingerprint": fingerprint,
                    "granted_input": '{"target_queue":"engineering_support"}',
                    "decision_request_id": uuid4(),
                    "decision_correlation_id": uuid4(),
                },
            )

        async with engine.begin() as connection:
            with pytest.raises(IntegrityError):
                await connection.execute(
                    text(
                        """
                        INSERT INTO sensitive_execution_grants (
                            id, workspace_id, ticket_id, agent_run_id,
                            executed_by_agent_run_attempt_id,
                            approval_request_id, agent_tool_call_id,
                            tool_name, tool_version, safety_level,
                            input_fingerprint, granted_input,
                            decision_actor_reference, decision_request_id,
                            decision_correlation_id, approved_at, created_at
                        ) VALUES (
                            :grant_id,
                            :workspace_id,
                            :ticket_id,
                            :run_id,
                            :attempt_id,
                            :approval_id,
                            :tool_call_id,
                            'escalate_ticket',
                            1,
                            'read_only',
                            :fingerprint,
                            CAST(:granted_input AS jsonb),
                            'operator:alice',
                            :decision_request_id,
                            :decision_correlation_id,
                            TIMESTAMPTZ '2026-08-03 00:05:00+00',
                            TIMESTAMPTZ '2026-08-03 00:06:00+00'
                        )
                        """
                    ),
                    {
                        "grant_id": uuid4(),
                        "workspace_id": workspace_id,
                        "ticket_id": ticket_id,
                        "run_id": run_id,
                        "attempt_id": attempt_id,
                        "approval_id": second_approval_id,
                        "tool_call_id": second_tool_call_id,
                        "fingerprint": "b" * 64,
                        "granted_input": '{"target_queue":"support_operations"}',
                        "decision_request_id": uuid4(),
                        "decision_correlation_id": uuid4(),
                    },
                )

        async with engine.begin() as connection:
            with pytest.raises(IntegrityError):
                await connection.execute(
                    text(
                        """
                        INSERT INTO sensitive_execution_grants (
                            id, workspace_id, ticket_id, agent_run_id,
                            executed_by_agent_run_attempt_id,
                            approval_request_id, agent_tool_call_id,
                            tool_name, tool_version, safety_level,
                            input_fingerprint, granted_input,
                            decision_actor_reference, decision_request_id,
                            decision_correlation_id, approved_at, created_at
                        ) VALUES (
                            :grant_id,
                            :workspace_id,
                            :ticket_id,
                            :run_id,
                            :attempt_id,
                            :approval_id,
                            :tool_call_id,
                            'escalate_ticket',
                            1,
                            'sensitive_write',
                            'NOT-A-FINGERPRINT',
                            CAST(:granted_input AS jsonb),
                            'operator:alice',
                            :decision_request_id,
                            :decision_correlation_id,
                            TIMESTAMPTZ '2026-08-03 00:05:00+00',
                            TIMESTAMPTZ '2026-08-03 00:06:00+00'
                        )
                        """
                    ),
                    {
                        "grant_id": uuid4(),
                        "workspace_id": workspace_id,
                        "ticket_id": ticket_id,
                        "run_id": run_id,
                        "attempt_id": attempt_id,
                        "approval_id": second_approval_id,
                        "tool_call_id": second_tool_call_id,
                        "granted_input": '{"target_queue":"support_operations"}',
                        "decision_request_id": uuid4(),
                        "decision_correlation_id": uuid4(),
                    },
                )

        async with engine.begin() as connection:
            with pytest.raises(IntegrityError):
                await connection.execute(
                    text(
                        """
                        INSERT INTO sensitive_execution_grants (
                            id, workspace_id, ticket_id, agent_run_id,
                            executed_by_agent_run_attempt_id,
                            approval_request_id, agent_tool_call_id,
                            tool_name, tool_version, safety_level,
                            input_fingerprint, granted_input,
                            decision_actor_reference, decision_request_id,
                            decision_correlation_id, approved_at, created_at
                        ) VALUES (
                            :grant_id,
                            :workspace_id,
                            :ticket_id,
                            :run_id,
                            :attempt_id,
                            :approval_id,
                            :tool_call_id,
                            'escalate_ticket',
                            1,
                            'sensitive_write',
                            :fingerprint,
                            CAST('"not-an-object"' AS jsonb),
                            'operator:alice',
                            :decision_request_id,
                            :decision_correlation_id,
                            TIMESTAMPTZ '2026-08-03 00:05:00+00',
                            TIMESTAMPTZ '2026-08-03 00:06:00+00'
                        )
                        """
                    ),
                    {
                        "grant_id": uuid4(),
                        "workspace_id": workspace_id,
                        "ticket_id": ticket_id,
                        "run_id": run_id,
                        "attempt_id": attempt_id,
                        "approval_id": second_approval_id,
                        "tool_call_id": second_tool_call_id,
                        "fingerprint": "b" * 64,
                        "decision_request_id": uuid4(),
                        "decision_correlation_id": uuid4(),
                    },
                )

        async with engine.begin() as connection:
            with pytest.raises(IntegrityError):
                await connection.execute(
                    text(
                        """
                        INSERT INTO sensitive_execution_grants (
                            id, workspace_id, ticket_id, agent_run_id,
                            executed_by_agent_run_attempt_id,
                            approval_request_id, agent_tool_call_id,
                            tool_name, tool_version, safety_level,
                            input_fingerprint, granted_input,
                            decision_actor_reference, decision_request_id,
                            decision_correlation_id, approved_at, created_at
                        ) VALUES (
                            :grant_id,
                            :workspace_id,
                            :ticket_id,
                            :run_id,
                            :attempt_id,
                            :approval_id,
                            :tool_call_id,
                            'escalate_ticket',
                            1,
                            'sensitive_write',
                            :fingerprint,
                            CAST(:granted_input AS jsonb),
                            'operator:alice',
                            :decision_request_id,
                            :decision_correlation_id,
                            TIMESTAMPTZ '2026-08-03 00:05:00+00',
                            TIMESTAMPTZ '2026-08-03 00:04:00+00'
                        )
                        """
                    ),
                    {
                        "grant_id": uuid4(),
                        "workspace_id": workspace_id,
                        "ticket_id": ticket_id,
                        "run_id": run_id,
                        "attempt_id": attempt_id,
                        "approval_id": second_approval_id,
                        "tool_call_id": second_tool_call_id,
                        "fingerprint": "b" * 64,
                        "granted_input": '{"target_queue":"support_operations"}',
                        "decision_request_id": uuid4(),
                        "decision_correlation_id": uuid4(),
                    },
                )

        async with engine.begin() as connection:
            with pytest.raises(IntegrityError):
                await connection.execute(
                    text(
                        """
                        INSERT INTO sensitive_execution_grants (
                            id, workspace_id, ticket_id, agent_run_id,
                            executed_by_agent_run_attempt_id,
                            approval_request_id, agent_tool_call_id,
                            tool_name, tool_version, safety_level,
                            input_fingerprint, granted_input,
                            decision_actor_reference, decision_request_id,
                            decision_correlation_id, approved_at, created_at
                        ) VALUES (
                            :grant_id,
                            :workspace_id,
                            :ticket_id,
                            :run_id,
                            :attempt_id,
                            :approval_id,
                            :tool_call_id,
                            'escalate_ticket',
                            1,
                            'sensitive_write',
                            :fingerprint,
                            CAST(:granted_input AS jsonb),
                            'operator:alice',
                            :decision_request_id,
                            :decision_correlation_id,
                            TIMESTAMPTZ '2026-08-03 00:05:00+00',
                            TIMESTAMPTZ '2026-08-03 00:06:00+00'
                        )
                        """
                    ),
                    {
                        "grant_id": uuid4(),
                        "workspace_id": workspace_id,
                        "ticket_id": ticket_id,
                        "run_id": run_id,
                        "attempt_id": attempt_id,
                        "approval_id": approval_id,
                        "tool_call_id": second_tool_call_id,
                        "fingerprint": fingerprint,
                        "granted_input": '{"target_queue":"engineering_support"}',
                        "decision_request_id": uuid4(),
                        "decision_correlation_id": uuid4(),
                    },
                )

        async with engine.begin() as connection:
            with pytest.raises(IntegrityError):
                await connection.execute(
                    text(
                        """
                        INSERT INTO sensitive_execution_grants (
                            id, workspace_id, ticket_id, agent_run_id,
                            executed_by_agent_run_attempt_id,
                            approval_request_id, agent_tool_call_id,
                            tool_name, tool_version, safety_level,
                            input_fingerprint, granted_input,
                            decision_actor_reference, decision_request_id,
                            decision_correlation_id, approved_at, created_at
                        ) VALUES (
                            :grant_id,
                            :workspace_id,
                            :ticket_id,
                            :run_id,
                            :attempt_id,
                            :approval_id,
                            :tool_call_id,
                            'escalate_ticket',
                            1,
                            'sensitive_write',
                            :fingerprint,
                            CAST(:granted_input AS jsonb),
                            'operator:alice',
                            :decision_request_id,
                            :decision_correlation_id,
                            TIMESTAMPTZ '2026-08-03 00:05:00+00',
                            TIMESTAMPTZ '2026-08-03 00:06:00+00'
                        )
                        """
                    ),
                    {
                        "grant_id": uuid4(),
                        "workspace_id": workspace_id,
                        "ticket_id": ticket_id,
                        "run_id": run_id,
                        "attempt_id": attempt_id,
                        "approval_id": second_approval_id,
                        "tool_call_id": tool_call_id,
                        "fingerprint": "b" * 64,
                        "granted_input": '{"target_queue":"support_operations"}',
                        "decision_request_id": uuid4(),
                        "decision_correlation_id": uuid4(),
                    },
                )

        async with engine.begin() as connection:
            with pytest.raises(IntegrityError):
                await connection.execute(
                    text(
                        """
                        INSERT INTO sensitive_execution_grants (
                            id, workspace_id, ticket_id, agent_run_id,
                            executed_by_agent_run_attempt_id,
                            approval_request_id, agent_tool_call_id,
                            tool_name, tool_version, safety_level,
                            input_fingerprint, granted_input,
                            decision_actor_reference, decision_request_id,
                            decision_correlation_id, approved_at, created_at
                        ) VALUES (
                            :grant_id,
                            :workspace_id,
                            :ticket_id,
                            :run_id,
                            :attempt_id,
                            :approval_id,
                            :tool_call_id,
                            'escalate_ticket',
                            1,
                            'sensitive_write',
                            :fingerprint,
                            CAST(:granted_input AS jsonb),
                            'operator:alice',
                            :decision_request_id,
                            :decision_correlation_id,
                            TIMESTAMPTZ '2026-08-03 00:05:00+00',
                            TIMESTAMPTZ '2026-08-03 00:06:00+00'
                        )
                        """
                    ),
                    {
                        "grant_id": grant_id,
                        "workspace_id": workspace_id,
                        "ticket_id": ticket_id,
                        "run_id": run_id,
                        "attempt_id": attempt_id,
                        "approval_id": second_approval_id,
                        "tool_call_id": second_tool_call_id,
                        "fingerprint": "b" * 64,
                        "granted_input": '{"target_queue":"support_operations"}',
                        "decision_request_id": uuid4(),
                        "decision_correlation_id": uuid4(),
                    },
                )

        async with engine.begin() as connection:
            with pytest.raises(IntegrityError):
                await connection.execute(
                    text(
                        """
                        INSERT INTO sensitive_execution_grants (
                            id, workspace_id, ticket_id, agent_run_id,
                            executed_by_agent_run_attempt_id,
                            approval_request_id, agent_tool_call_id,
                            tool_name, tool_version, safety_level,
                            input_fingerprint, granted_input,
                            decision_actor_reference, decision_request_id,
                            decision_correlation_id, approved_at, created_at
                        ) VALUES (
                            :grant_id,
                            :workspace_id,
                            :ticket_id,
                            :run_id,
                            :attempt_id,
                            :approval_id,
                            :tool_call_id,
                            'escalate_ticket',
                            1,
                            'sensitive_write',
                            :fingerprint,
                            CAST(:granted_input AS jsonb),
                            'operator:alice',
                            :decision_request_id,
                            :decision_correlation_id,
                            TIMESTAMPTZ '2026-08-03 00:05:00+00',
                            TIMESTAMPTZ '2026-08-03 00:06:00+00'
                        )
                        """
                    ),
                    {
                        "grant_id": uuid4(),
                        "workspace_id": workspace_id,
                        "ticket_id": ticket_id,
                        "run_id": run_id,
                        "attempt_id": uuid4(),
                        "approval_id": second_approval_id,
                        "tool_call_id": second_tool_call_id,
                        "fingerprint": "b" * 64,
                        "granted_input": '{"target_queue":"support_operations"}',
                        "decision_request_id": uuid4(),
                        "decision_correlation_id": uuid4(),
                    },
                )

        async with engine.begin() as connection:
            oversized = "{" + (",".join(f'"k{i}":"{i}"' for i in range(2000))) + "}"
            with pytest.raises(IntegrityError):
                await connection.execute(
                    text(
                        """
                        INSERT INTO sensitive_execution_grants (
                            id, workspace_id, ticket_id, agent_run_id,
                            executed_by_agent_run_attempt_id,
                            approval_request_id, agent_tool_call_id,
                            tool_name, tool_version, safety_level,
                            input_fingerprint, granted_input,
                            decision_actor_reference, decision_request_id,
                            decision_correlation_id, approved_at, created_at
                        ) VALUES (
                            :grant_id,
                            :workspace_id,
                            :ticket_id,
                            :run_id,
                            :attempt_id,
                            :approval_id,
                            :tool_call_id,
                            'escalate_ticket',
                            1,
                            'sensitive_write',
                            :fingerprint,
                            CAST(:granted_input AS jsonb),
                            'operator:alice',
                            :decision_request_id,
                            :decision_correlation_id,
                            TIMESTAMPTZ '2026-08-03 00:05:00+00',
                            TIMESTAMPTZ '2026-08-03 00:06:00+00'
                        )
                        """
                    ),
                    {
                        "grant_id": uuid4(),
                        "workspace_id": workspace_id,
                        "ticket_id": ticket_id,
                        "run_id": run_id,
                        "attempt_id": attempt_id,
                        "approval_id": second_approval_id,
                        "tool_call_id": second_tool_call_id,
                        "fingerprint": "b" * 64,
                        "granted_input": oversized,
                        "decision_request_id": uuid4(),
                        "decision_correlation_id": uuid4(),
                    },
                )

        downgrade = run_alembic_command(
            "downgrade",
            APPROVAL_REQUEST_REVISION,
        )
        assert downgrade.returncode == 0, downgrade.stderr
        assert not await relation_exists(
            engine,
            "public.sensitive_execution_grants",
        )

        reupgrade = run_alembic_command("upgrade", EXPECTED_HEAD)
        assert reupgrade.returncode == 0, reupgrade.stderr
        assert await relation_exists(
            engine,
            "public.sensitive_execution_grants",
        )

        table = cast(Table, SensitiveExecutionGrantRecord.__table__)
        column_names = {column.name for column in table.c}
        assert "executed_by_agent_run_attempt_id" in column_names
        assert "granted_input" in column_names
        assert "approved_at" in column_names
        constraint_names = {constraint.name for constraint in table.constraints}
        assert "ck_sensitive_execution_grants_safety_level" in constraint_names
        assert "uq_sensitive_execution_grants_approval_request" in constraint_names
        assert "fk_sensitive_execution_grants_execution_attempt" in constraint_names
        index_names = {index.name for index in table.indexes}
        assert "ix_sensitive_execution_grants_workspace_created_id" in index_names
    finally:
        await engine.dispose()
        run_alembic_command("upgrade", "head")
