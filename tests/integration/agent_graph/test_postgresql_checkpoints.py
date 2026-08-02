"""PostgreSQL integration tests for graph checkpoint setup."""

from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from supportops.agent_graph.infrastructure.checkpoints import (
    create_postgres_checkpoint_runtime,
)
from supportops.core.settings import Settings

_FRAMEWORK_TABLES = (
    "checkpoint_migrations",
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
)


async def test_checkpoint_setup_is_idempotent_across_runtimes(
    exclusive_integration_database: None,
    postgresql_engine: AsyncEngine,
    integration_settings: Settings,
) -> None:
    """Create framework tables repeatedly without application migrations."""

    del exclusive_integration_database

    checkpoint_database_url = SecretStr(_to_psycopg_connection_url(integration_settings))

    first_runtime = await create_postgres_checkpoint_runtime(
        database_url=checkpoint_database_url,
    )

    try:
        await first_runtime.setup()
        await first_runtime.setup()
    finally:
        await first_runtime.close()

    second_runtime = await create_postgres_checkpoint_runtime(
        database_url=checkpoint_database_url,
    )

    try:
        await second_runtime.setup()
    finally:
        await second_runtime.close()

    async with postgresql_engine.connect() as connection:
        table_result = await connection.execute(
            text(
                """
                SELECT
                    to_regclass(
                        'public.checkpoint_migrations'
                    ) AS checkpoint_migrations,
                    to_regclass(
                        'public.checkpoints'
                    ) AS checkpoints,
                    to_regclass(
                        'public.checkpoint_blobs'
                    ) AS checkpoint_blobs,
                    to_regclass(
                        'public.checkpoint_writes'
                    ) AS checkpoint_writes
                """
            )
        )

        row = table_result.mappings().one()

    assert set(row) == set(_FRAMEWORK_TABLES)

    for table_name in _FRAMEWORK_TABLES:
        assert row[table_name] == table_name


def _to_psycopg_connection_url(
    settings: Settings,
) -> str:
    application_database_url = str(settings.postgresql_url)
    sqlalchemy_prefix = "postgresql+asyncpg://"

    if not application_database_url.startswith(sqlalchemy_prefix):
        raise AssertionError("Integration PostgreSQL URL must use the asyncpg SQLAlchemy scheme.")

    return "postgresql://" + application_database_url[len(sqlalchemy_prefix) :]
