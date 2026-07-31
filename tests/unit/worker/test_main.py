"""Unit tests for the executable AgentRun worker process."""

import asyncio
import json
import logging
from unittest.mock import patch

import pytest

from supportops.worker.main import (
    _wait_for_loop_or_shutdown,
    log_event,
    resolve_worker_id,
)


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
