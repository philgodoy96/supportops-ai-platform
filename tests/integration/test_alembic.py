"""Integration tests for Alembic migration configuration."""

import subprocess
import sys
from typing import cast
from uuid import uuid4

import pytest
from sqlalchemy import Table, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from supportops.core.settings import Settings
from supportops.modules.agent_runs.infrastructure.models import (
    AgentRunRecord,
)

pytestmark = pytest.mark.integration

EXPECTED_HEAD = "f3a9c1d7e5b2"
CONTROLLED_WORKFLOW_REVISION = "e8b7c6d5a4f3"
PRE_CONTROLLED_WORKFLOW_REVISION = "d4e8f2a6c901"


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

    downgrade = run_alembic_command("downgrade", "base")
    assert downgrade.returncode == 0, downgrade.stderr

    try:
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
        assert not await relation_exists(
            engine,
            "public.support_recommendations",
        )
        assert not await relation_exists(
            engine,
            "public.support_recommendation_citations",
        )
    finally:
        await engine.dispose()

    reupgrade = run_alembic_command("upgrade", "head")
    assert reupgrade.returncode == 0, reupgrade.stderr

    engine = create_async_engine(str(settings.postgresql_url))
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
        assert await relation_exists(engine, "public.support_recommendations")
        assert await relation_exists(
            engine,
            "public.support_recommendation_citations",
        )
    finally:
        await engine.dispose()


async def test_alembic_downgrade_controlled_workflow_revision_removes_only_those_tables(
    exclusive_integration_database: None,
) -> None:
    settings = Settings()
    engine = create_async_engine(str(settings.postgresql_url))

    upgrade = run_alembic_command("upgrade", "head")
    assert upgrade.returncode == 0, upgrade.stderr

    try:
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
