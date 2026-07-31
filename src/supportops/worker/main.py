"""Executable process composition for the PostgreSQL AgentRun worker."""

import asyncio
import json
import logging
import os
import signal
import socket
from collections.abc import Callable
from contextlib import suppress
from types import FrameType
from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from supportops.core.settings import Settings
from supportops.modules.agent_runs.application.deterministic_executor import (
    DeterministicTicketProcessingExecutor,
)
from supportops.modules.agent_runs.application.retry_policy import (
    AgentRunRetryPolicy,
)
from supportops.modules.agent_runs.application.worker import (
    WorkerCycleResult,
)
from supportops.modules.agent_runs.application.worker_loop import (
    RunAgentWorkerLoop,
    WorkerCycleObserver,
)
from supportops.modules.agent_runs.infrastructure.worker_runtime import (
    PostgreSqlAgentWorkerCycleRunner,
)

_SignalHandler = Callable[[int, FrameType | None], Any] | int | signal.Handlers | None

_LOGGER = logging.getLogger("supportops.worker")

_EXIT_SUCCESS = 0
_EXIT_RUNTIME_FAILURE = 1


def main() -> None:
    """Run the worker process and exit with an operational status code."""

    raise SystemExit(asyncio.run(run_worker()))


async def run_worker(
    *,
    settings: Settings | None = None,
    stop_event: asyncio.Event | None = None,
    engine_factory: Callable[[str], AsyncEngine] | None = None,
) -> int:
    """Compose and run the worker until shutdown or runtime failure."""

    resolved_settings = settings or Settings()
    resolved_stop_event = stop_event or asyncio.Event()
    worker_id = resolve_worker_id(resolved_settings.worker_id)

    configure_logging()

    build_engine = engine_factory or create_worker_engine
    engine = build_engine(
        str(resolved_settings.postgresql_url),
    )
    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        autoflush=False,
        expire_on_commit=False,
    )

    retry_policy = AgentRunRetryPolicy(
        base_delay_seconds=(resolved_settings.worker_retry_base_seconds),
        maximum_delay_seconds=(resolved_settings.worker_retry_max_seconds),
    )
    executor = DeterministicTicketProcessingExecutor()
    cycle_runner = PostgreSqlAgentWorkerCycleRunner(
        session_factory=session_factory,
        worker_id=worker_id,
        executor=executor,
        retry_policy=retry_policy,
        lease_seconds=resolved_settings.worker_lease_seconds,
        execution_timeout_seconds=(resolved_settings.worker_execution_timeout_seconds),
    )
    worker_loop = RunAgentWorkerLoop(
        cycle=cycle_runner,
        stop_event=resolved_stop_event,
        poll_interval_seconds=(resolved_settings.worker_poll_interval_seconds),
        cycle_observer=create_cycle_observer(worker_id),
    )

    loop_task = asyncio.create_task(
        worker_loop.execute(),
        name="supportops-agent-worker-loop",
    )
    remove_shutdown_handlers = install_shutdown_handlers(
        stop_event=resolved_stop_event,
    )

    log_event(
        logging.INFO,
        "worker_started",
        worker_id=worker_id,
        executor=resolved_settings.worker_executor,
        poll_interval_seconds=(resolved_settings.worker_poll_interval_seconds),
        lease_seconds=resolved_settings.worker_lease_seconds,
        execution_timeout_seconds=(resolved_settings.worker_execution_timeout_seconds),
    )

    exit_code = _EXIT_SUCCESS

    try:
        await _wait_for_loop_or_shutdown(
            loop_task=loop_task,
            stop_event=resolved_stop_event,
            shutdown_grace_seconds=(resolved_settings.worker_shutdown_grace_seconds),
            worker_id=worker_id,
        )
    except asyncio.CancelledError:
        loop_task.cancel()

        with suppress(asyncio.CancelledError):
            await loop_task

        raise
    except Exception:
        exit_code = _EXIT_RUNTIME_FAILURE
        log_event(
            logging.ERROR,
            "worker_failed",
            worker_id=worker_id,
            include_exception=True,
        )
    finally:
        remove_shutdown_handlers()

        if not loop_task.done():
            loop_task.cancel()

            with suppress(asyncio.CancelledError):
                await loop_task

        await engine.dispose()

        log_event(
            logging.INFO,
            "worker_stopped",
            worker_id=worker_id,
            exit_code=exit_code,
        )

    return exit_code


async def _wait_for_loop_or_shutdown(
    *,
    loop_task: asyncio.Task[None],
    stop_event: asyncio.Event,
    shutdown_grace_seconds: float,
    worker_id: str,
) -> None:
    stop_task = asyncio.create_task(
        stop_event.wait(),
        name="supportops-agent-worker-stop",
    )

    try:
        done, _ = await asyncio.wait(
            {loop_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if loop_task in done:
            await loop_task
            return

        log_event(
            logging.INFO,
            "worker_shutdown_requested",
            worker_id=worker_id,
            shutdown_grace_seconds=shutdown_grace_seconds,
        )

        if shutdown_grace_seconds == 0:
            loop_task.cancel()
        else:
            try:
                await asyncio.wait_for(
                    asyncio.shield(loop_task),
                    timeout=shutdown_grace_seconds,
                )
            except TimeoutError:
                log_event(
                    logging.WARNING,
                    "worker_shutdown_grace_exceeded",
                    worker_id=worker_id,
                    shutdown_grace_seconds=shutdown_grace_seconds,
                )
                loop_task.cancel()

        with suppress(asyncio.CancelledError):
            await loop_task
    finally:
        stop_task.cancel()

        with suppress(asyncio.CancelledError):
            await stop_task


def install_shutdown_handlers(
    *,
    stop_event: asyncio.Event,
) -> Callable[[], None]:
    """Install portable SIGINT and SIGTERM handlers."""

    event_loop = asyncio.get_running_loop()
    installed_loop_signals: list[signal.Signals] = []
    previous_handlers: dict[signal.Signals, _SignalHandler] = {}

    def request_shutdown() -> None:
        stop_event.set()

    for shutdown_signal in (
        signal.SIGINT,
        signal.SIGTERM,
    ):
        try:
            event_loop.add_signal_handler(
                shutdown_signal,
                request_shutdown,
            )
            installed_loop_signals.append(shutdown_signal)
        except (NotImplementedError, RuntimeError):
            previous_handlers[shutdown_signal] = signal.getsignal(
                shutdown_signal,
            )

            def synchronous_handler(
                signum: int,
                frame: FrameType | None,
                *,
                callback: Callable[[], None] = request_shutdown,
            ) -> None:
                del signum, frame
                event_loop.call_soon_threadsafe(callback)

            signal.signal(
                shutdown_signal,
                synchronous_handler,
            )

    def remove_handlers() -> None:
        for shutdown_signal in installed_loop_signals:
            event_loop.remove_signal_handler(shutdown_signal)

        for shutdown_signal, previous_handler in previous_handlers.items():
            signal.signal(
                shutdown_signal,
                previous_handler,
            )

    return remove_handlers


def resolve_worker_id(configured_worker_id: str | None) -> str:
    """Return a configured or process-generated worker identity."""

    if configured_worker_id is not None:
        return configured_worker_id

    hostname = socket.gethostname().strip() or "unknown-host"
    generated_id = f"{hostname}-{os.getpid()}-{uuid4().hex[:8]}"

    return generated_id[:128]


def create_worker_engine(database_url: str) -> AsyncEngine:
    """Create the worker's PostgreSQL async engine."""

    return create_async_engine(
        database_url,
        pool_pre_ping=True,
    )


def create_cycle_observer(worker_id: str) -> WorkerCycleObserver:
    """Create the worker cycle structured-log observer."""

    async def observe(result: WorkerCycleResult) -> None:
        log_event(
            logging.INFO,
            "worker_cycle_completed",
            worker_id=worker_id,
            outcome=result.outcome.value,
            recovered_expired_run=result.recovered_expired_run,
            agent_run_id=(str(result.agent_run_id) if result.agent_run_id is not None else None),
        )

    return observe


def configure_logging() -> None:
    """Configure process-level console logging when not already configured."""

    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


def log_event(
    level: int,
    event: str,
    *,
    include_exception: bool = False,
    **fields: object,
) -> None:
    """Emit one structured operational log event."""

    payload = {
        "event": event,
        **fields,
    }

    _LOGGER.log(
        level,
        json.dumps(
            payload,
            default=str,
            sort_keys=True,
        ),
        exc_info=include_exception,
    )


if __name__ == "__main__":
    main()
