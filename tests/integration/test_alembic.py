"""Integration tests for Alembic migration configuration."""

import subprocess
import sys

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from supportops.core.settings import Settings
from supportops.infrastructure.postgresql import Base

pytestmark = pytest.mark.integration


def run_alembic_command(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run Alembic through the active Python environment."""

    return subprocess.run(
        [sys.executable, "-m", "alembic", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_alembic_has_no_migration_heads() -> None:
    result = run_alembic_command("heads")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_alembic_current_connects_to_postgresql() -> None:
    result = run_alembic_command("current")

    assert result.returncode == 0, result.stderr


async def test_shared_metadata_contains_no_business_tables() -> None:
    assert Base.metadata.tables == {}


async def test_alembic_version_table_is_not_created_without_revisions() -> None:
    settings = Settings()
    engine = create_async_engine(str(settings.postgresql_url))

    try:
        async with engine.connect() as connection:
            result = await connection.execute(
                text(
                    """
                    SELECT to_regclass('public.alembic_version')
                    """
                )
            )
            alembic_version_table = result.scalar_one()
    finally:
        await engine.dispose()

    assert alembic_version_table is None
