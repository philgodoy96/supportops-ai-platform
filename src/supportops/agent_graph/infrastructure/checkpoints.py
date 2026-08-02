"""PostgreSQL lifecycle for LangGraph operational checkpoints."""

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import SecretStr

_CHECKPOINT_POOL_MIN_SIZE = 1
_CHECKPOINT_POOL_MAX_SIZE = 5
_CHECKPOINT_POOL_TIMEOUT_SECONDS = 10.0
_CHECKPOINT_POOL_CLOSE_TIMEOUT_SECONDS = 10.0

type CheckpointPool = AsyncConnectionPool[Any]
type CheckpointPoolBuilder = Callable[[str], CheckpointPool]
type CheckpointSaverBuilder = Callable[
    [CheckpointPool, SerializerProtocol],
    AsyncPostgresSaver,
]


class GraphCheckpointError(RuntimeError):
    """Base application-owned checkpoint infrastructure error."""

    error_code = "graph_checkpoint_unavailable"
    retryable = True


class GraphCheckpointUnavailableError(GraphCheckpointError):
    """Raised when PostgreSQL checkpoint infrastructure is unavailable."""

    error_code = "graph_checkpoint_unavailable"
    retryable = True

    def __init__(self) -> None:
        super().__init__("PostgreSQL graph checkpoint infrastructure is unavailable.")


class GraphCheckpointSetupError(GraphCheckpointError):
    """Raised when framework-owned checkpoint setup cannot complete."""

    error_code = "graph_checkpoint_setup_failed"
    retryable = True

    def __init__(self) -> None:
        super().__init__("PostgreSQL graph checkpoint setup could not be completed.")


class GraphCheckpointRuntimeClosedError(GraphCheckpointError):
    """Raised when a closed checkpoint runtime is reused."""

    error_code = "graph_checkpoint_runtime_closed"
    retryable = False

    def __init__(self) -> None:
        super().__init__("PostgreSQL graph checkpoint runtime is already closed.")


@dataclass(slots=True)
class PostgresCheckpointRuntime:
    """Own one process-scoped checkpoint pool and checkpointer."""

    pool: CheckpointPool
    checkpointer: AsyncPostgresSaver
    _closed: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    @property
    def is_closed(self) -> bool:
        """Return whether process-owned resources were released."""

        return self._closed

    async def setup(self) -> None:
        """Create or migrate framework-owned checkpoint tables."""

        self._ensure_open()

        try:
            await self.checkpointer.setup()
        except Exception as exc:
            raise GraphCheckpointSetupError() from exc

    async def close(self) -> None:
        """Release the process-owned Psycopg connection pool once."""

        if self._closed:
            return

        self._closed = True

        try:
            await self.pool.close(
                timeout=_CHECKPOINT_POOL_CLOSE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise GraphCheckpointUnavailableError() from exc

    def _ensure_open(self) -> None:
        if self._closed:
            raise GraphCheckpointRuntimeClosedError()


def create_checkpoint_serializer() -> SerializerProtocol:
    """Create a restricted serializer for bounded graph state."""

    return JsonPlusSerializer(
        pickle_fallback=False,
        allowed_json_modules=(),
        allowed_msgpack_modules=(),
    )


async def create_postgres_checkpoint_runtime(
    *,
    database_url: SecretStr,
    pool_builder: CheckpointPoolBuilder | None = None,
    saver_builder: CheckpointSaverBuilder | None = None,
) -> PostgresCheckpointRuntime:
    """Open one process-scoped PostgreSQL checkpoint runtime."""

    resolved_pool_builder = pool_builder or _build_checkpoint_pool
    resolved_saver_builder = saver_builder or _build_checkpoint_saver

    pool = resolved_pool_builder(
        database_url.get_secret_value(),
    )

    try:
        await pool.open(
            wait=True,
            timeout=_CHECKPOINT_POOL_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        with suppress(Exception):
            await pool.close(
                timeout=_CHECKPOINT_POOL_CLOSE_TIMEOUT_SECONDS,
            )

        raise GraphCheckpointUnavailableError() from exc

    try:
        serializer = create_checkpoint_serializer()
        checkpointer = resolved_saver_builder(
            pool,
            serializer,
        )
    except Exception as exc:
        with suppress(Exception):
            await pool.close(
                timeout=_CHECKPOINT_POOL_CLOSE_TIMEOUT_SECONDS,
            )

        raise GraphCheckpointUnavailableError() from exc

    return PostgresCheckpointRuntime(
        pool=pool,
        checkpointer=checkpointer,
    )


def _build_checkpoint_pool(
    connection_info: str,
) -> CheckpointPool:
    return AsyncConnectionPool(
        conninfo=connection_info,
        min_size=_CHECKPOINT_POOL_MIN_SIZE,
        max_size=_CHECKPOINT_POOL_MAX_SIZE,
        timeout=_CHECKPOINT_POOL_TIMEOUT_SECONDS,
        open=False,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    )


def _build_checkpoint_saver(
    pool: CheckpointPool,
    serializer: SerializerProtocol,
) -> AsyncPostgresSaver:
    return AsyncPostgresSaver(
        conn=pool,
        serde=serializer,
    )
