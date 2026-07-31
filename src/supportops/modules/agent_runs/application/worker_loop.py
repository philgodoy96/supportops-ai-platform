"""Continuous polling loop for the PostgreSQL AgentRun worker."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

from supportops.modules.agent_runs.application.worker import (
    WorkerCycleOutcome,
    WorkerCycleResult,
)

WorkerCycleObserver = Callable[[WorkerCycleResult], Awaitable[None]]


class WorkerCycleRunner(Protocol):
    """Run one worker cycle and return its observable result."""

    async def execute(self) -> WorkerCycleResult:
        """Run one worker cycle."""

        ...


class RunAgentWorkerLoop:
    """Run worker cycles until cooperative shutdown is requested."""

    def __init__(
        self,
        *,
        cycle: WorkerCycleRunner,
        stop_event: asyncio.Event,
        poll_interval_seconds: float,
        cycle_observer: WorkerCycleObserver | None = None,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError(
                "poll_interval_seconds must be greater than zero.",
            )

        self._cycle = cycle
        self._stop_event = stop_event
        self._poll_interval_seconds = poll_interval_seconds
        self._cycle_observer = cycle_observer

    async def execute(self) -> None:
        """Run cycles until shutdown is requested."""

        while not self._stop_event.is_set():
            result = await self._cycle.execute()

            if self._cycle_observer is not None:
                await self._cycle_observer(result)

            if self._stop_event.is_set():
                break

            if result.outcome is WorkerCycleOutcome.IDLE:
                await self._wait_for_work_or_stop()

    async def _wait_for_work_or_stop(self) -> None:
        try:
            await asyncio.wait_for(
                self._stop_event.wait(),
                timeout=self._poll_interval_seconds,
            )
        except TimeoutError:
            return
