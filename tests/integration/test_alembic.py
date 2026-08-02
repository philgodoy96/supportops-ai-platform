"""Integration tests for Alembic migration configuration."""

import subprocess
import sys

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from supportops.core.settings import Settings

pytestmark = pytest.mark.integration

EXPECTED_HEAD = "e8b7c6d5a4f3"


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


async def test_alembic_downgrade_one_removes_only_controlled_workflow_tables_and_can_reupgrade(
    exclusive_integration_database: None,
) -> None:
    settings = Settings()
    engine = create_async_engine(str(settings.postgresql_url))

    upgrade = run_alembic_command("upgrade", "head")
    assert upgrade.returncode == 0, upgrade.stderr

    try:
        downgrade = run_alembic_command("downgrade", "-1")
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
