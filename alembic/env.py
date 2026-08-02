"""Alembic migration environment."""

import asyncio
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import Connection, Index, pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from supportops.core.settings import Settings
from supportops.infrastructure.postgresql import Base
from supportops.infrastructure.postgresql.model_registry import (
    register_persistence_models,
)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

register_persistence_models()
target_metadata = Base.metadata

# LangGraph owns and migrates these tables through checkpointer setup.
LANGGRAPH_CHECKPOINT_TABLE_NAMES = frozenset(
    {
        "checkpoint_migrations",
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
    }
)


def include_object(
    object_: Any,
    name: str | None,
    type_: str,
    reflected: bool,
    compare_to: object | None,
) -> bool:
    del reflected, compare_to

    if type_ == "table" and name in LANGGRAPH_CHECKPOINT_TABLE_NAMES:
        return False

    return not (
        type_ == "index"
        and isinstance(object_, Index)
        and object_.table is not None
        and object_.table.name in LANGGRAPH_CHECKPOINT_TABLE_NAMES
    )


def get_database_url() -> str:
    """Return the validated PostgreSQL URL used for migrations."""

    settings = Settings()
    return str(settings.postgresql_url)


def configure_database_url() -> None:
    """Inject the environment-owned database URL into Alembic configuration."""

    config.set_main_option(
        "sqlalchemy.url",
        get_database_url().replace("%", "%%"),
    )


def run_migrations_offline() -> None:
    """Run migrations without creating a database connection."""

    configure_database_url()

    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        render_as_batch=False,
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_sync_migrations(connection: Connection) -> None:
    """Configure and execute migrations using a synchronous connection facade."""

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=False,
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and execute migrations."""

    configure_database_url()

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    try:
        async with connectable.connect() as connection:
            await connection.run_sync(run_sync_migrations)
    finally:
        await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations using an async SQLAlchemy engine."""

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
