"""Integration tests for Alembic migration configuration."""

import subprocess
import sys

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from supportops.core.settings import Settings

pytestmark = pytest.mark.integration

EXPECTED_HEAD = "6d9f0a2b4c31"


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


def test_alembic_reports_expected_head() -> None:
    result = run_alembic_command("heads")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{EXPECTED_HEAD} (head)"


def test_alembic_current_connects_to_postgresql() -> None:
    result = run_alembic_command("current")

    assert result.returncode == 0, result.stderr


async def test_alembic_upgrade_creates_business_tables() -> None:
    settings = Settings()
    engine = create_async_engine(str(settings.postgresql_url))

    upgrade = run_alembic_command("upgrade", "head")
    assert upgrade.returncode == 0, upgrade.stderr

    try:
        assert await relation_exists(engine, "public.alembic_version")
        assert await relation_exists(engine, "public.workspaces")
        assert await relation_exists(engine, "public.tickets")
    finally:
        await engine.dispose()


async def test_alembic_downgrade_removes_business_tables_and_can_reupgrade() -> None:
    settings = Settings()
    engine = create_async_engine(str(settings.postgresql_url))

    upgrade = run_alembic_command("upgrade", "head")
    assert upgrade.returncode == 0, upgrade.stderr

    downgrade = run_alembic_command("downgrade", "base")
    assert downgrade.returncode == 0, downgrade.stderr

    try:
        assert not await relation_exists(engine, "public.tickets")
        assert not await relation_exists(engine, "public.workspaces")
    finally:
        await engine.dispose()

    reupgrade = run_alembic_command("upgrade", "head")
    assert reupgrade.returncode == 0, reupgrade.stderr

    engine = create_async_engine(str(settings.postgresql_url))
    try:
        assert await relation_exists(engine, "public.alembic_version")
        assert await relation_exists(engine, "public.workspaces")
        assert await relation_exists(engine, "public.tickets")
    finally:
        await engine.dispose()
