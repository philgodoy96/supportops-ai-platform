"""Unit tests for the executable AgentRun worker process."""

import asyncio
import json
import logging
from collections.abc import Callable
from contextlib import suppress
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from supportops.core.settings import Settings
from supportops.worker.main import (
    _wait_for_loop_or_shutdown,
    log_event,
    resolve_worker_id,
    run_worker,
)


def _create_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "postgresql_url": (
            "postgresql+asyncpg://supportops:supportops-local@localhost:5432/supportops"
        ),
        "qdrant_url": "http://localhost:6333",
        "worker_id": "worker-test-1",
        "worker_shutdown_grace_seconds": 0,
        "worker_poll_interval_seconds": 0.01,
    }
    values.update(overrides)
    settings_type = cast(Any, Settings)
    return cast(Settings, settings_type(_env_file=None, **values))


def test_default_ticket_processing_workflow_remains_controlled_support() -> None:
    settings = _create_settings()

    assert settings.ticket_processing_workflow_version == ("controlled-support-v1")


class FakeObservabilityClient:
    """Process-scoped observability stand-in for worker tests."""

    def __init__(self) -> None:
        self.enabled = False
        self.shutdown_calls = 0
        self.shutdown_error: Exception | None = None

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error


class FakeProvider:
    """Minimal provider exposing a stable process-scoped name."""

    def __init__(self, provider_name: str = "mock") -> None:
        self.provider_name = provider_name


class FakeLLMRuntime:
    """Process-scoped LLM runtime stand-in for composition tests."""

    def __init__(
        self,
        *,
        provider_name: str = "mock",
        model: str = "mock-ticket-classifier-v1",
    ) -> None:
        self.provider = FakeProvider(provider_name)
        self.gateway = object()
        self.model = model
        self.close_calls = 0
        self.close_error: Exception | None = None

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class FakeControlledRuntime:
    """Process-scoped controlled-support runtime stand-in."""

    def __init__(self) -> None:
        self.observability_client = FakeObservabilityClient()
        self.close_calls = 0
        self.close_error: Exception | None = None

    async def close(self) -> None:
        self.close_calls += 1
        with suppress(Exception):
            self.observability_client.shutdown()
        if self.close_error is not None:
            raise self.close_error


class FakeEngine:
    """Async engine stand-in that records disposal."""

    def __init__(self) -> None:
        self.dispose_calls = 0

    async def dispose(self) -> None:
        self.dispose_calls += 1


def _as_engine_factory(
    engine: FakeEngine,
) -> Callable[[str], AsyncEngine]:
    return cast(
        Callable[[str], AsyncEngine],
        lambda database_url: engine,
    )


def _reset_capturing_cycle_runner() -> None:
    CapturingCycleRunner.last_kwargs = cast(
        dict[str, object] | None,
        None,
    )


def _captured_cycle_runner_kwargs() -> dict[str, object]:
    last_kwargs = CapturingCycleRunner.last_kwargs
    assert last_kwargs is not None
    return last_kwargs


class CapturingCycleRunner:
    """Capture constructor kwargs without running database work."""

    last_kwargs: dict[str, object] | None = None

    def __init__(self, **kwargs: object) -> None:
        CapturingCycleRunner.last_kwargs = kwargs

    async def execute(self) -> object:
        return None


class ImmediateWorkerLoop:
    """Worker loop that returns immediately after construction."""

    def __init__(self, **kwargs: object) -> None:
        del kwargs

    async def execute(self) -> None:
        return None


class FailingWorkerLoop:
    """Worker loop that fails during execution."""

    def __init__(self, **kwargs: object) -> None:
        del kwargs

    async def execute(self) -> None:
        raise RuntimeError("worker loop failed")


def test_resolve_worker_id_preserves_configured_value() -> None:
    assert resolve_worker_id("worker-local-1") == "worker-local-1"


def test_resolve_worker_id_generates_bounded_process_identity() -> None:
    with (
        patch(
            "supportops.worker.main.socket.gethostname",
            return_value="supportops-host",
        ),
        patch(
            "supportops.worker.main.os.getpid",
            return_value=4312,
        ),
        patch(
            "supportops.worker.main.uuid4",
        ) as uuid_factory,
    ):
        uuid_factory.return_value.hex = "12345678abcdef00"

        worker_id = resolve_worker_id(None)

    assert worker_id == "supportops-host-4312-12345678"
    assert len(worker_id) <= 128


def test_resolve_worker_id_handles_empty_hostname() -> None:
    with (
        patch(
            "supportops.worker.main.socket.gethostname",
            return_value=" ",
        ),
        patch(
            "supportops.worker.main.os.getpid",
            return_value=4312,
        ),
        patch(
            "supportops.worker.main.uuid4",
        ) as uuid_factory,
    ):
        uuid_factory.return_value.hex = "12345678abcdef00"

        worker_id = resolve_worker_id(None)

    assert worker_id == "unknown-host-4312-12345678"


async def test_wait_returns_when_worker_loop_finishes() -> None:
    stop_event = asyncio.Event()
    loop_task = asyncio.create_task(
        asyncio.sleep(0),
    )

    await _wait_for_loop_or_shutdown(
        loop_task=loop_task,
        stop_event=stop_event,
        shutdown_grace_seconds=10.0,
        worker_id="worker-a",
    )

    assert loop_task.done()
    assert loop_task.cancelled() is False


async def test_shutdown_allows_loop_to_finish_within_grace() -> None:
    stop_event = asyncio.Event()
    loop_finished = asyncio.Event()

    async def finish_after_shutdown() -> None:
        await stop_event.wait()
        loop_finished.set()

    loop_task = asyncio.create_task(
        finish_after_shutdown(),
    )
    stop_event.set()

    await _wait_for_loop_or_shutdown(
        loop_task=loop_task,
        stop_event=stop_event,
        shutdown_grace_seconds=1.0,
        worker_id="worker-a",
    )

    assert loop_finished.is_set()
    assert loop_task.done()
    assert loop_task.cancelled() is False


async def test_zero_grace_cancels_active_loop() -> None:
    stop_event = asyncio.Event()

    async def block_forever() -> None:
        await asyncio.Event().wait()

    loop_task = asyncio.create_task(
        block_forever(),
    )
    stop_event.set()

    await _wait_for_loop_or_shutdown(
        loop_task=loop_task,
        stop_event=stop_event,
        shutdown_grace_seconds=0.0,
        worker_id="worker-a",
    )

    assert loop_task.cancelled()


async def test_expired_grace_cancels_active_loop() -> None:
    stop_event = asyncio.Event()

    async def block_forever() -> None:
        await asyncio.Event().wait()

    loop_task = asyncio.create_task(
        block_forever(),
    )
    stop_event.set()

    await _wait_for_loop_or_shutdown(
        loop_task=loop_task,
        stop_event=stop_event,
        shutdown_grace_seconds=0.001,
        worker_id="worker-a",
    )

    assert loop_task.cancelled()


def test_log_event_emits_json_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(
        logging.INFO,
        logger="supportops.worker",
    )

    log_event(
        logging.INFO,
        "worker_cycle_completed",
        worker_id="worker-a",
        outcome="processed",
        agent_run_id="run-1",
    )

    payload = json.loads(caplog.records[-1].message)

    assert payload == {
        "agent_run_id": "run-1",
        "event": "worker_cycle_completed",
        "outcome": "processed",
        "worker_id": "worker-a",
    }


def test_log_event_can_include_exception_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(
        logging.ERROR,
        logger="supportops.worker",
    )

    try:
        raise RuntimeError("runtime failure")
    except RuntimeError:
        log_event(
            logging.ERROR,
            "worker_failed",
            worker_id="worker-a",
            include_exception=True,
        )

    record = caplog.records[-1]

    assert record.exc_info is not None
    assert json.loads(record.message)["event"] == "worker_failed"


async def test_run_worker_composes_llm_runtime_and_closes_resources(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _create_settings(
        llm_provider="openai",
        openai_api_key="secret-openai-key",
        openai_model="gpt-5-nano",
        openai_base_url="https://api.example.com/v1",
        llm_request_timeout_seconds=15,
        llm_transport_max_retries=2,
        llm_max_repair_attempts=0,
        ticket_processing_workflow_version="ticket-classification-v1",
        worker_execution_timeout_seconds=50,
        worker_lease_seconds=65,
    )
    llm_runtime = FakeLLMRuntime(
        provider_name="openai",
        model="gpt-5-nano",
    )
    controlled_runtime = FakeControlledRuntime()
    engine = FakeEngine()
    _reset_capturing_cycle_runner()
    create_runtime = MagicMock(return_value=llm_runtime)
    create_controlled_runtime = AsyncMock(return_value=controlled_runtime)

    caplog.set_level(logging.INFO, logger="supportops.worker")

    with (
        patch(
            "supportops.worker.main.create_worker_llm_runtime",
            create_runtime,
        ),
        patch(
            "supportops.worker.main.async_sessionmaker",
            return_value=object(),
        ),
        patch(
            "supportops.worker.main.PostgreSqlAgentWorkerCycleRunner",
            CapturingCycleRunner,
        ),
        patch(
            "supportops.worker.main.RunAgentWorkerLoop",
            ImmediateWorkerLoop,
        ),
        patch(
            "supportops.worker.main.install_shutdown_handlers",
            return_value=lambda: None,
        ),
    ):
        exit_code = await run_worker(
            settings=settings,
            engine_factory=_as_engine_factory(engine),
            controlled_runtime_factory=create_controlled_runtime,
        )

    assert exit_code == 0
    create_runtime.assert_called_once_with(
        provider_name="openai",
        openai_api_key="secret-openai-key",
        openai_model="gpt-5-nano",
        openai_base_url="https://api.example.com/v1",
        request_timeout_seconds=15.0,
        transport_max_retries=2,
        max_repair_attempts=0,
    )
    create_controlled_runtime.assert_awaited_once_with(settings=settings)
    assert callable(_captured_cycle_runner_kwargs()["executor_factory"])
    assert controlled_runtime.close_calls == 1
    assert controlled_runtime.observability_client.shutdown_calls == 1
    assert llm_runtime.close_calls == 1
    assert engine.dispose_calls == 1

    started_payloads = [
        json.loads(record.message)
        for record in caplog.records
        if '"event": "worker_started"' in record.message
        or '"event":"worker_started"' in record.message
    ]
    assert len(started_payloads) == 1
    started = started_payloads[0]
    assert started["executor_registry"] == "versioned-workflow-registry"
    assert started["llm_provider"] == "openai"
    assert started["llm_model"] == "gpt-5-nano"
    assert started["ticket_processing_workflow_version"] == ("ticket-classification-v1")
    assert started["approval_ttl_seconds"] == 86400.0
    assert started["approval_expiration_batch_size"] == 100
    assert started["observability_provider"] == "noop"
    assert started["observability_enabled"] is False
    assert started["observability_capture_mode"] == "metadata_only"
    assert "secret-openai-key" not in json.dumps(started)
    assert "openai_api_key" not in started

    assert _captured_cycle_runner_kwargs()["approval_expiration_batch_size"] == 100


async def test_run_worker_executor_factory_builds_session_scoped_registry() -> None:
    settings = _create_settings()
    llm_runtime = FakeLLMRuntime()
    controlled_runtime = FakeControlledRuntime()
    engine = FakeEngine()
    _reset_capturing_cycle_runner()
    registry = object()
    create_registry = MagicMock(return_value=registry)
    create_controlled_runtime = AsyncMock(return_value=controlled_runtime)

    with (
        patch(
            "supportops.worker.main.create_worker_llm_runtime",
            return_value=llm_runtime,
        ),
        patch(
            "supportops.worker.main.create_session_scoped_executor_registry",
            create_registry,
        ),
        patch(
            "supportops.worker.main.async_sessionmaker",
            return_value=object(),
        ),
        patch(
            "supportops.worker.main.PostgreSqlAgentWorkerCycleRunner",
            CapturingCycleRunner,
        ),
        patch(
            "supportops.worker.main.RunAgentWorkerLoop",
            ImmediateWorkerLoop,
        ),
        patch(
            "supportops.worker.main.install_shutdown_handlers",
            return_value=lambda: None,
        ),
    ):
        await run_worker(
            settings=settings,
            engine_factory=_as_engine_factory(engine),
            controlled_runtime_factory=create_controlled_runtime,
        )

        executor_factory = _captured_cycle_runner_kwargs()["executor_factory"]
        assert callable(executor_factory)

        session = object()
        transaction_manager = object()
        result = executor_factory(session, transaction_manager)

        assert result is registry
        create_controlled_runtime.assert_awaited_once_with(settings=settings)
        create_registry.assert_called_once_with(
            session=session,
            transaction_manager=transaction_manager,
            gateway=llm_runtime.gateway,
            provider=llm_runtime.provider,
            model=llm_runtime.model,
            request_timeout_seconds=settings.llm_request_timeout_seconds,
            controlled_runtime=controlled_runtime,
            embedding_timeout_seconds=(settings.embedding_request_timeout_seconds),
        )


async def test_run_worker_closes_controlled_runtime_when_engine_construction_fails() -> None:
    settings = _create_settings()
    llm_runtime = FakeLLMRuntime()
    controlled_runtime = FakeControlledRuntime()
    create_controlled_runtime = AsyncMock(return_value=controlled_runtime)

    with (
        patch(
            "supportops.worker.main.create_worker_llm_runtime",
            return_value=llm_runtime,
        ),
        patch(
            "supportops.worker.main.install_shutdown_handlers",
            return_value=lambda: None,
        ),
        pytest.raises(RuntimeError, match="engine unavailable"),
    ):

        def failing_engine_factory(database_url: str) -> AsyncEngine:
            del database_url
            raise RuntimeError("engine unavailable")

        await run_worker(
            settings=settings,
            engine_factory=failing_engine_factory,
            controlled_runtime_factory=create_controlled_runtime,
        )

    assert controlled_runtime.close_calls == 1
    assert llm_runtime.close_calls == 1


async def test_run_worker_closes_llm_runtime_when_controlled_runtime_creation_fails() -> None:
    settings = _create_settings()
    llm_runtime = FakeLLMRuntime()

    async def failing_controlled_runtime_factory(
        *,
        settings: Settings,
    ) -> FakeControlledRuntime:
        del settings
        raise RuntimeError("controlled runtime unavailable")

    with (
        patch(
            "supportops.worker.main.create_worker_llm_runtime",
            return_value=llm_runtime,
        ),
        patch(
            "supportops.worker.main.install_shutdown_handlers",
            return_value=lambda: None,
        ),
        pytest.raises(RuntimeError, match="controlled runtime unavailable"),
    ):
        await run_worker(
            settings=settings,
            controlled_runtime_factory=cast(
                Any,
                failing_controlled_runtime_factory,
            ),
        )

    assert llm_runtime.close_calls == 1


async def test_run_worker_closes_controlled_runtime_when_worker_execution_fails() -> None:
    settings = _create_settings()
    llm_runtime = FakeLLMRuntime()
    controlled_runtime = FakeControlledRuntime()
    engine = FakeEngine()
    create_controlled_runtime = AsyncMock(return_value=controlled_runtime)

    with (
        patch(
            "supportops.worker.main.create_worker_llm_runtime",
            return_value=llm_runtime,
        ),
        patch(
            "supportops.worker.main.async_sessionmaker",
            return_value=object(),
        ),
        patch(
            "supportops.worker.main.PostgreSqlAgentWorkerCycleRunner",
            CapturingCycleRunner,
        ),
        patch(
            "supportops.worker.main.RunAgentWorkerLoop",
            FailingWorkerLoop,
        ),
        patch(
            "supportops.worker.main.install_shutdown_handlers",
            return_value=lambda: None,
        ),
    ):
        exit_code = await run_worker(
            settings=settings,
            engine_factory=_as_engine_factory(engine),
            controlled_runtime_factory=create_controlled_runtime,
        )

    assert exit_code == 1
    assert controlled_runtime.close_calls == 1
    assert llm_runtime.close_calls == 1
    assert engine.dispose_calls == 1


async def test_run_worker_disposes_engine_when_provider_close_fails() -> None:
    settings = _create_settings()
    llm_runtime = FakeLLMRuntime()
    llm_runtime.close_error = RuntimeError("provider close failed")
    controlled_runtime = FakeControlledRuntime()
    engine = FakeEngine()
    create_controlled_runtime = AsyncMock(return_value=controlled_runtime)

    with (
        patch(
            "supportops.worker.main.create_worker_llm_runtime",
            return_value=llm_runtime,
        ),
        patch(
            "supportops.worker.main.async_sessionmaker",
            return_value=object(),
        ),
        patch(
            "supportops.worker.main.PostgreSqlAgentWorkerCycleRunner",
            CapturingCycleRunner,
        ),
        patch(
            "supportops.worker.main.RunAgentWorkerLoop",
            ImmediateWorkerLoop,
        ),
        patch(
            "supportops.worker.main.install_shutdown_handlers",
            return_value=lambda: None,
        ),
        pytest.raises(RuntimeError, match="provider close failed"),
    ):
        await run_worker(
            settings=settings,
            engine_factory=_as_engine_factory(engine),
            controlled_runtime_factory=create_controlled_runtime,
        )

    assert controlled_runtime.close_calls == 1
    assert controlled_runtime.observability_client.shutdown_calls == 1
    assert llm_runtime.close_calls == 1
    assert engine.dispose_calls == 1


async def test_run_worker_observability_shutdown_failure_preserves_exit_code() -> None:
    settings = _create_settings()
    llm_runtime = FakeLLMRuntime()
    controlled_runtime = FakeControlledRuntime()
    controlled_runtime.observability_client.shutdown_error = RuntimeError(
        "observability shutdown failed",
    )
    engine = FakeEngine()
    create_controlled_runtime = AsyncMock(return_value=controlled_runtime)

    with (
        patch(
            "supportops.worker.main.create_worker_llm_runtime",
            return_value=llm_runtime,
        ),
        patch(
            "supportops.worker.main.async_sessionmaker",
            return_value=object(),
        ),
        patch(
            "supportops.worker.main.PostgreSqlAgentWorkerCycleRunner",
            CapturingCycleRunner,
        ),
        patch(
            "supportops.worker.main.RunAgentWorkerLoop",
            ImmediateWorkerLoop,
        ),
        patch(
            "supportops.worker.main.install_shutdown_handlers",
            return_value=lambda: None,
        ),
    ):
        exit_code = await run_worker(
            settings=settings,
            engine_factory=_as_engine_factory(engine),
            controlled_runtime_factory=create_controlled_runtime,
        )

    assert exit_code == 0
    assert controlled_runtime.close_calls == 1
    assert controlled_runtime.observability_client.shutdown_calls == 1
    assert llm_runtime.close_calls == 1
    assert engine.dispose_calls == 1


async def test_run_worker_holds_one_process_observability_client() -> None:
    settings = _create_settings()
    llm_runtime = FakeLLMRuntime()
    controlled_runtime = FakeControlledRuntime()
    engine = FakeEngine()
    create_controlled_runtime = AsyncMock(return_value=controlled_runtime)
    _reset_capturing_cycle_runner()

    with (
        patch(
            "supportops.worker.main.create_worker_llm_runtime",
            return_value=llm_runtime,
        ),
        patch(
            "supportops.worker.main.async_sessionmaker",
            return_value=object(),
        ),
        patch(
            "supportops.worker.main.PostgreSqlAgentWorkerCycleRunner",
            CapturingCycleRunner,
        ),
        patch(
            "supportops.worker.main.RunAgentWorkerLoop",
            ImmediateWorkerLoop,
        ),
        patch(
            "supportops.worker.main.install_shutdown_handlers",
            return_value=lambda: None,
        ),
    ):
        exit_code = await run_worker(
            settings=settings,
            engine_factory=_as_engine_factory(engine),
            controlled_runtime_factory=create_controlled_runtime,
        )

        executor_factory = _captured_cycle_runner_kwargs()["executor_factory"]
        assert callable(executor_factory)

    assert exit_code == 0
    create_controlled_runtime.assert_awaited_once_with(settings=settings)
    assert controlled_runtime.observability_client.shutdown_calls == 1
