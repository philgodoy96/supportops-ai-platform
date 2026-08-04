"""Executable process composition for the PostgreSQL AgentRun worker."""

import asyncio
import json
import logging
import os
import signal
import socket
from collections.abc import Awaitable, Callable
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
from supportops.core.transactions import TransactionManager
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutor,
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
    AgentRunExecutorFactory,
    PostgreSqlAgentWorkerCycleRunner,
)
from supportops.observability.composition import create_observability_client
from supportops.observability.contracts import ObservabilityClient
from supportops.worker.composition import (
    WorkerControlledSupportRuntime,
    WorkerLLMRuntime,
    create_session_scoped_executor_registry,
    create_worker_controlled_support_runtime,
    create_worker_llm_runtime,
)

_SignalHandler = Callable[[int, FrameType | None], Any] | int | signal.Handlers | None

_LOGGER = logging.getLogger("supportops.worker")

_EXIT_SUCCESS = 0
_EXIT_RUNTIME_FAILURE = 1

type ControlledSupportRuntimeFactory = Callable[
    ...,
    Awaitable[WorkerControlledSupportRuntime],
]


def main() -> None:
    """Run the worker process and exit with an operational status code."""

    raise SystemExit(asyncio.run(run_worker()))


async def run_worker(
    *,
    settings: Settings | None = None,
    stop_event: asyncio.Event | None = None,
    engine_factory: Callable[[str], AsyncEngine] | None = None,
    controlled_runtime_factory: ControlledSupportRuntimeFactory | None = None,
) -> int:
    """Compose and run the worker until shutdown or runtime failure."""

    resolved_settings = settings or Settings()
    resolved_stop_event = stop_event or asyncio.Event()
    worker_id = resolve_worker_id(resolved_settings.worker_id)
    build_controlled_runtime = (
        controlled_runtime_factory or create_worker_controlled_support_runtime
    )

    configure_logging()

    provider_name = str(resolved_settings.llm_provider)
    openai_api_key = (
        resolved_settings.openai_api_key.get_secret_value()
        if resolved_settings.openai_api_key is not None
        else None
    )
    openai_base_url = (
        str(resolved_settings.openai_base_url)
        if resolved_settings.openai_base_url is not None
        else None
    )
    observability_client: ObservabilityClient = create_observability_client(
        resolved_settings,
    )

    try:
        llm_runtime = create_worker_llm_runtime(
            provider_name=provider_name,
            openai_api_key=openai_api_key,
            openai_model=resolved_settings.openai_model,
            openai_base_url=openai_base_url,
            request_timeout_seconds=(resolved_settings.llm_request_timeout_seconds),
            transport_max_retries=(resolved_settings.llm_transport_max_retries),
            max_repair_attempts=(resolved_settings.llm_max_repair_attempts),
            observability_client=observability_client,
        )
    except Exception:
        with suppress(Exception):
            observability_client.shutdown()
        raise

    try:
        controlled_runtime = await build_controlled_runtime(
            settings=resolved_settings,
            observability_client=observability_client,
        )
    except Exception:
        await llm_runtime.close()
        with suppress(Exception):
            observability_client.shutdown()
        raise

    build_engine = engine_factory or create_worker_engine
    try:
        engine = build_engine(
            str(resolved_settings.postgresql_url),
        )
    except Exception:
        await _close_startup_resources(
            controlled_runtime=controlled_runtime,
            llm_runtime=llm_runtime,
        )
        raise

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

    def executor_factory(
        session: AsyncSession,
        transaction_manager: TransactionManager,
    ) -> AgentRunExecutor:
        return create_session_scoped_executor_registry(
            session=session,
            transaction_manager=transaction_manager,
            gateway=llm_runtime.gateway,
            provider=llm_runtime.provider,
            model=llm_runtime.model,
            request_timeout_seconds=(resolved_settings.llm_request_timeout_seconds),
            controlled_runtime=controlled_runtime,
            embedding_timeout_seconds=(resolved_settings.embedding_request_timeout_seconds),
        )

    session_scoped_executor_factory: AgentRunExecutorFactory = executor_factory

    cycle_runner = PostgreSqlAgentWorkerCycleRunner(
        session_factory=session_factory,
        worker_id=worker_id,
        executor_factory=session_scoped_executor_factory,
        retry_policy=retry_policy,
        lease_seconds=resolved_settings.worker_lease_seconds,
        execution_timeout_seconds=(resolved_settings.worker_execution_timeout_seconds),
        approval_expiration_batch_size=(resolved_settings.approval_expiration_batch_size),
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
        executor_registry="versioned-workflow-registry",
        llm_provider=llm_runtime.provider.provider_name,
        llm_model=llm_runtime.model,
        ticket_processing_workflow_version=(resolved_settings.ticket_processing_workflow_version),
        poll_interval_seconds=(resolved_settings.worker_poll_interval_seconds),
        lease_seconds=resolved_settings.worker_lease_seconds,
        execution_timeout_seconds=(resolved_settings.worker_execution_timeout_seconds),
        approval_ttl_seconds=resolved_settings.approval_ttl_seconds,
        approval_expiration_batch_size=(resolved_settings.approval_expiration_batch_size),
        observability_provider=(resolved_settings.ai_observability_provider.value),
        observability_enabled=(controlled_runtime.observability_client.enabled),
        observability_capture_mode=(resolved_settings.langfuse_capture_mode.value),
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

        try:
            await _close_shutdown_resources(
                controlled_runtime=controlled_runtime,
                llm_runtime=llm_runtime,
                engine=engine,
            )
        finally:
            log_event(
                logging.INFO,
                "worker_stopped",
                worker_id=worker_id,
                exit_code=exit_code,
            )

    return exit_code


async def _close_startup_resources(
    *,
    controlled_runtime: WorkerControlledSupportRuntime,
    llm_runtime: WorkerLLMRuntime,
) -> None:
    failures: list[Exception] = []

    try:
        await controlled_runtime.close()
    except Exception as error:
        failures.append(error)

    try:
        await llm_runtime.close()
    except Exception as error:
        failures.append(error)

    if failures:
        primary_failure = failures[0]
        for secondary_failure in failures[1:]:
            primary_failure.add_note(
                "An additional startup resource failed to close: "
                f"{type(secondary_failure).__name__}."
            )
        raise primary_failure


async def _close_shutdown_resources(
    *,
    controlled_runtime: WorkerControlledSupportRuntime,
    llm_runtime: WorkerLLMRuntime,
    engine: AsyncEngine,
) -> None:
    failures: list[Exception] = []

    try:
        await controlled_runtime.close()
    except Exception as error:
        failures.append(error)

    try:
        await llm_runtime.close()
    except Exception as error:
        failures.append(error)

    try:
        await engine.dispose()
    except Exception as error:
        failures.append(error)

    if failures:
        primary_failure = failures[0]
        for secondary_failure in failures[1:]:
            primary_failure.add_note(
                "An additional shutdown resource failed to close: "
                f"{type(secondary_failure).__name__}."
            )
        raise primary_failure


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
