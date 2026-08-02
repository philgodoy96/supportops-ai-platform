"""Unit tests for PostgreSQL graph checkpoint lifecycle."""

from typing import Any, cast

import pytest
from langgraph.checkpoint.postgres.aio import (
    AsyncPostgresSaver,
)
from langgraph.checkpoint.serde.base import (
    SerializerProtocol,
)
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import SecretStr

from supportops.agent_graph.infrastructure.checkpoints import (
    GraphCheckpointRuntimeClosedError,
    GraphCheckpointSetupError,
    GraphCheckpointUnavailableError,
    PostgresCheckpointRuntime,
    create_checkpoint_serializer,
    create_postgres_checkpoint_runtime,
)
from supportops.infrastructure.postgresql import Base

_CHECKPOINT_DATABASE_URL = (
    "postgresql://checkpoint-user:checkpoint-password@localhost:5432/supportops_checkpoints"
)


class UnsupportedCheckpointValue:
    """Represent one value outside the approved state boundary."""


class FakePool:
    """Record pool lifecycle without PostgreSQL access."""

    def __init__(self) -> None:
        self.open_calls: list[tuple[bool, float]] = []
        self.close_calls: list[float] = []
        self.open_error: Exception | None = None
        self.close_error: Exception | None = None

    async def open(
        self,
        *,
        wait: bool = False,
        timeout: float = 30.0,
    ) -> None:
        self.open_calls.append((wait, timeout))

        if self.open_error is not None:
            raise self.open_error

    async def close(
        self,
        timeout: float = 5.0,
    ) -> None:
        self.close_calls.append(timeout)

        if self.close_error is not None:
            raise self.close_error


class FakeSaver:
    """Record framework setup calls without database access."""

    def __init__(self) -> None:
        self.setup_calls = 0
        self.setup_error: Exception | None = None

    async def setup(self) -> None:
        self.setup_calls += 1

        if self.setup_error is not None:
            raise self.setup_error


def _cast_pool(
    pool: FakePool,
) -> AsyncConnectionPool[Any]:
    return cast(
        AsyncConnectionPool[Any],
        pool,
    )


def _cast_saver(
    saver: FakeSaver,
) -> AsyncPostgresSaver:
    return cast(
        AsyncPostgresSaver,
        saver,
    )


def test_checkpoint_serializer_round_trips_json_state() -> None:
    serializer = create_checkpoint_serializer()
    state = {
        "agent_run_id": ("11111111-1111-4111-8111-111111111111"),
        "tool_call_count": 2,
        "seen_fingerprints": [
            "a" * 64,
            "b" * 64,
        ],
        "analysis_complete": True,
        "recommendation_id": None,
    }

    serialized = serializer.dumps_typed(state)
    deserialized = serializer.loads_typed(serialized)

    assert deserialized == state


def test_checkpoint_serializer_has_no_pickle_fallback() -> None:
    serializer = create_checkpoint_serializer()

    with pytest.raises(TypeError):
        serializer.dumps_typed(UnsupportedCheckpointValue())


async def test_runtime_factory_opens_pool_and_builds_saver() -> None:
    pool = FakePool()
    saver = FakeSaver()
    observed_connection_info: list[str] = []
    observed_serializer: list[SerializerProtocol] = []

    def build_pool(
        connection_info: str,
    ) -> AsyncConnectionPool[Any]:
        observed_connection_info.append(connection_info)
        return _cast_pool(pool)

    def build_saver(
        actual_pool: AsyncConnectionPool[Any],
        serializer: SerializerProtocol,
    ) -> AsyncPostgresSaver:
        assert actual_pool is _cast_pool(pool)
        observed_serializer.append(serializer)
        return _cast_saver(saver)

    runtime = await create_postgres_checkpoint_runtime(
        database_url=SecretStr(_CHECKPOINT_DATABASE_URL),
        pool_builder=build_pool,
        saver_builder=build_saver,
    )

    assert observed_connection_info == [_CHECKPOINT_DATABASE_URL]
    assert pool.open_calls == [(True, 10.0)]
    assert len(observed_serializer) == 1
    assert runtime.checkpointer is _cast_saver(saver)
    assert runtime.is_closed is False

    await runtime.close()

    assert pool.close_calls == [10.0]
    assert runtime.is_closed is True


async def test_pool_open_failure_is_normalized_and_closed() -> None:
    pool = FakePool()
    pool.open_error = RuntimeError(_CHECKPOINT_DATABASE_URL)

    def build_pool(
        connection_info: str,
    ) -> AsyncConnectionPool[Any]:
        assert connection_info == _CHECKPOINT_DATABASE_URL
        return _cast_pool(pool)

    with pytest.raises(
        GraphCheckpointUnavailableError,
        match="checkpoint infrastructure is unavailable",
    ) as exc_info:
        await create_postgres_checkpoint_runtime(
            database_url=SecretStr(_CHECKPOINT_DATABASE_URL),
            pool_builder=build_pool,
        )

    assert _CHECKPOINT_DATABASE_URL not in str(exc_info.value)
    assert pool.close_calls == [10.0]


async def test_saver_construction_failure_closes_pool() -> None:
    pool = FakePool()

    def build_pool(
        connection_info: str,
    ) -> AsyncConnectionPool[Any]:
        assert connection_info == _CHECKPOINT_DATABASE_URL
        return _cast_pool(pool)

    def fail_saver(
        actual_pool: AsyncConnectionPool[Any],
        serializer: SerializerProtocol,
    ) -> AsyncPostgresSaver:
        del actual_pool, serializer
        raise RuntimeError("saver construction failed")

    with pytest.raises(GraphCheckpointUnavailableError):
        await create_postgres_checkpoint_runtime(
            database_url=SecretStr(_CHECKPOINT_DATABASE_URL),
            pool_builder=build_pool,
            saver_builder=fail_saver,
        )

    assert pool.open_calls == [(True, 10.0)]
    assert pool.close_calls == [10.0]


async def test_runtime_setup_delegates_idempotently() -> None:
    pool = FakePool()
    saver = FakeSaver()
    runtime = PostgresCheckpointRuntime(
        pool=_cast_pool(pool),
        checkpointer=_cast_saver(saver),
    )

    await runtime.setup()
    await runtime.setup()

    assert saver.setup_calls == 2


async def test_runtime_setup_failure_is_normalized() -> None:
    pool = FakePool()
    saver = FakeSaver()
    saver.setup_error = RuntimeError(_CHECKPOINT_DATABASE_URL)
    runtime = PostgresCheckpointRuntime(
        pool=_cast_pool(pool),
        checkpointer=_cast_saver(saver),
    )

    with pytest.raises(
        GraphCheckpointSetupError,
        match="setup could not be completed",
    ) as exc_info:
        await runtime.setup()

    assert _CHECKPOINT_DATABASE_URL not in str(exc_info.value)


async def test_runtime_close_is_idempotent() -> None:
    pool = FakePool()
    saver = FakeSaver()
    runtime = PostgresCheckpointRuntime(
        pool=_cast_pool(pool),
        checkpointer=_cast_saver(saver),
    )

    await runtime.close()
    await runtime.close()

    assert pool.close_calls == [10.0]


async def test_runtime_rejects_setup_after_close() -> None:
    pool = FakePool()
    saver = FakeSaver()
    runtime = PostgresCheckpointRuntime(
        pool=_cast_pool(pool),
        checkpointer=_cast_saver(saver),
    )

    await runtime.close()

    with pytest.raises(
        GraphCheckpointRuntimeClosedError,
        match="already closed",
    ):
        await runtime.setup()


async def test_close_failure_is_normalized() -> None:
    pool = FakePool()
    pool.close_error = RuntimeError(_CHECKPOINT_DATABASE_URL)
    saver = FakeSaver()
    runtime = PostgresCheckpointRuntime(
        pool=_cast_pool(pool),
        checkpointer=_cast_saver(saver),
    )

    with pytest.raises(
        GraphCheckpointUnavailableError,
        match="checkpoint infrastructure is unavailable",
    ) as exc_info:
        await runtime.close()

    assert _CHECKPOINT_DATABASE_URL not in str(exc_info.value)
    assert runtime.is_closed is True


def test_checkpoint_tables_are_not_application_metadata() -> None:
    framework_owned_tables = {
        "checkpoint_migrations",
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
    }

    assert framework_owned_tables.isdisjoint(Base.metadata.tables)


def test_default_pool_configuration_is_framework_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_arguments: dict[str, object] = {}

    class CapturingPool:
        def __init__(
            self,
            **arguments: object,
        ) -> None:
            observed_arguments.update(arguments)

    monkeypatch.setattr(
        "supportops.agent_graph.infrastructure.checkpoints.AsyncConnectionPool",
        CapturingPool,
    )

    from supportops.agent_graph.infrastructure import (
        checkpoints,
    )

    pool = checkpoints._build_checkpoint_pool(_CHECKPOINT_DATABASE_URL)

    assert isinstance(pool, CapturingPool)
    assert observed_arguments["conninfo"] == (_CHECKPOINT_DATABASE_URL)
    assert observed_arguments["min_size"] == 1
    assert observed_arguments["max_size"] == 5
    assert observed_arguments["timeout"] == 10.0
    assert observed_arguments["open"] is False

    connection_arguments = cast(
        dict[str, object],
        observed_arguments["kwargs"],
    )

    assert connection_arguments == {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row,
    }
