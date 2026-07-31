"""Unit tests for the continuous AgentRun worker loop."""

import asyncio
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from supportops.modules.agent_runs.application.worker import (
    WorkerCycleOutcome,
    WorkerCycleResult,
)
from supportops.modules.agent_runs.application.worker_loop import (
    RunAgentWorkerLoop,
)

_RUN_ID = UUID(
    "69184ef1-4d71-452e-8070-0b784c29368e",
)


def idle_result() -> WorkerCycleResult:
    return WorkerCycleResult(
        outcome=WorkerCycleOutcome.IDLE,
        recovered_expired_run=False,
        agent_run_id=None,
    )


def processed_result() -> WorkerCycleResult:
    return WorkerCycleResult(
        outcome=WorkerCycleOutcome.PROCESSED,
        recovered_expired_run=False,
        agent_run_id=_RUN_ID,
    )


def lease_lost_result() -> WorkerCycleResult:
    return WorkerCycleResult(
        outcome=WorkerCycleOutcome.LEASE_LOST,
        recovered_expired_run=False,
        agent_run_id=_RUN_ID,
    )


async def test_loop_does_not_start_when_stop_is_already_requested() -> None:
    cycle = AsyncMock()
    stop_event = asyncio.Event()
    stop_event.set()

    worker_loop = RunAgentWorkerLoop(
        cycle=cycle,
        stop_event=stop_event,
        poll_interval_seconds=1.0,
    )

    await worker_loop.execute()

    cycle.execute.assert_not_awaited()


async def test_loop_runs_one_cycle_then_stops() -> None:
    stop_event = asyncio.Event()
    cycle = AsyncMock()

    async def execute_cycle() -> WorkerCycleResult:
        stop_event.set()
        return processed_result()

    cycle.execute.side_effect = execute_cycle

    worker_loop = RunAgentWorkerLoop(
        cycle=cycle,
        stop_event=stop_event,
        poll_interval_seconds=1.0,
    )

    await worker_loop.execute()

    cycle.execute.assert_awaited_once_with()


async def test_loop_publishes_cycle_result() -> None:
    stop_event = asyncio.Event()
    cycle = AsyncMock()
    observer = AsyncMock()

    async def execute_cycle() -> WorkerCycleResult:
        stop_event.set()
        return lease_lost_result()

    cycle.execute.side_effect = execute_cycle

    worker_loop = RunAgentWorkerLoop(
        cycle=cycle,
        stop_event=stop_event,
        poll_interval_seconds=1.0,
        cycle_observer=observer,
    )

    await worker_loop.execute()

    observer.assert_awaited_once_with(
        lease_lost_result(),
    )


async def test_processed_cycle_repeats_without_idle_wait() -> None:
    stop_event = asyncio.Event()
    cycle = AsyncMock()
    results = iter(
        [
            processed_result(),
            processed_result(),
            processed_result(),
        ],
    )

    async def execute_cycle() -> WorkerCycleResult:
        result = next(results)

        if cycle.execute.await_count == 3:
            stop_event.set()

        return result

    cycle.execute.side_effect = execute_cycle

    worker_loop = RunAgentWorkerLoop(
        cycle=cycle,
        stop_event=stop_event,
        poll_interval_seconds=10.0,
    )

    await asyncio.wait_for(
        worker_loop.execute(),
        timeout=0.5,
    )

    assert cycle.execute.await_count == 3


async def test_idle_wait_is_interrupted_by_stop_request() -> None:
    stop_event = asyncio.Event()
    cycle = AsyncMock()
    cycle.execute.return_value = idle_result()

    worker_loop = RunAgentWorkerLoop(
        cycle=cycle,
        stop_event=stop_event,
        poll_interval_seconds=60.0,
    )

    task = asyncio.create_task(
        worker_loop.execute(),
    )

    await asyncio.sleep(0)
    stop_event.set()

    await asyncio.wait_for(
        task,
        timeout=0.5,
    )

    cycle.execute.assert_awaited_once_with()


async def test_idle_cycle_repeats_after_poll_interval() -> None:
    stop_event = asyncio.Event()
    cycle = AsyncMock()

    async def execute_cycle() -> WorkerCycleResult:
        if cycle.execute.await_count == 2:
            stop_event.set()

        return idle_result()

    cycle.execute.side_effect = execute_cycle

    worker_loop = RunAgentWorkerLoop(
        cycle=cycle,
        stop_event=stop_event,
        poll_interval_seconds=0.001,
    )

    await asyncio.wait_for(
        worker_loop.execute(),
        timeout=0.5,
    )

    assert cycle.execute.await_count == 2


async def test_observer_runs_after_each_cycle() -> None:
    stop_event = asyncio.Event()
    cycle = AsyncMock()
    observer = AsyncMock()
    results = iter(
        [
            processed_result(),
            lease_lost_result(),
        ],
    )

    async def execute_cycle() -> WorkerCycleResult:
        result = next(results)

        if cycle.execute.await_count == 2:
            stop_event.set()

        return result

    cycle.execute.side_effect = execute_cycle

    worker_loop = RunAgentWorkerLoop(
        cycle=cycle,
        stop_event=stop_event,
        poll_interval_seconds=1.0,
        cycle_observer=observer,
    )

    await worker_loop.execute()

    assert observer.await_count == 2
    assert observer.await_args_list[0].args == (processed_result(),)
    assert observer.await_args_list[1].args == (lease_lost_result(),)


async def test_cycle_cancellation_is_propagated() -> None:
    cycle = AsyncMock()
    cycle.execute.side_effect = asyncio.CancelledError()
    stop_event = asyncio.Event()

    worker_loop = RunAgentWorkerLoop(
        cycle=cycle,
        stop_event=stop_event,
        poll_interval_seconds=1.0,
    )

    with pytest.raises(asyncio.CancelledError):
        await worker_loop.execute()


async def test_observer_cancellation_is_propagated() -> None:
    stop_event = asyncio.Event()
    cycle = AsyncMock()
    cycle.execute.return_value = processed_result()

    observer = AsyncMock()
    observer.side_effect = asyncio.CancelledError()

    worker_loop = RunAgentWorkerLoop(
        cycle=cycle,
        stop_event=stop_event,
        poll_interval_seconds=1.0,
        cycle_observer=observer,
    )

    with pytest.raises(asyncio.CancelledError):
        await worker_loop.execute()


@pytest.mark.parametrize(
    "poll_interval_seconds",
    [
        0.0,
        -1.0,
    ],
)
def test_loop_requires_positive_poll_interval(
    poll_interval_seconds: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="poll_interval_seconds must be greater than zero",
    ):
        RunAgentWorkerLoop(
            cycle=AsyncMock(),
            stop_event=asyncio.Event(),
            poll_interval_seconds=poll_interval_seconds,
        )
