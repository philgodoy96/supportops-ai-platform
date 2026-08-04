"""Unit tests for task-local observability context."""

import asyncio

import pytest

from supportops.observability.context import (
    ActiveObservationContext,
    ActiveTraceContext,
    current_observation_context,
    current_trace_context,
    observation_context_scope,
    trace_context_scope,
)


def test_observability_context_is_empty_by_default() -> None:
    assert current_trace_context() is None
    assert current_observation_context() is None


def test_trace_context_scope_binds_and_restores_context() -> None:
    context = ActiveTraceContext(
        trace_seed="agent-run:run-1",
        trace_id="a" * 32,
        session_id="ticket:ticket-1",
    )

    with trace_context_scope(context):
        assert current_trace_context() == context

    assert current_trace_context() is None


def test_nested_trace_context_restores_parent() -> None:
    parent = ActiveTraceContext(
        trace_seed="agent-run:parent",
        trace_id="a" * 32,
    )
    child = ActiveTraceContext(
        trace_seed="semantic-search:child",
        trace_id="b" * 32,
    )

    with trace_context_scope(parent):
        assert current_trace_context() == parent

        with trace_context_scope(child):
            assert current_trace_context() == child

        assert current_trace_context() == parent

    assert current_trace_context() is None


def test_trace_context_is_restored_when_business_exception_escapes() -> None:
    context = ActiveTraceContext(
        trace_seed="agent-run:run-1",
        trace_id="a" * 32,
    )

    with pytest.raises(RuntimeError, match="business failure"), trace_context_scope(context):
        raise RuntimeError("business failure")

    assert current_trace_context() is None


def test_observation_context_scope_binds_and_restores_context() -> None:
    context = ActiveObservationContext(
        name="classification",
        observation_id="c" * 16,
    )

    with observation_context_scope(context):
        assert current_observation_context() == context

    assert current_observation_context() is None


def test_nested_observation_context_restores_parent() -> None:
    parent = ActiveObservationContext(
        name="classification",
        observation_id="a" * 16,
    )
    child = ActiveObservationContext(
        name="generation",
        observation_id="b" * 16,
    )

    with observation_context_scope(parent):
        with observation_context_scope(child):
            assert current_observation_context() == child

        assert current_observation_context() == parent

    assert current_observation_context() is None


@pytest.mark.asyncio
async def test_trace_context_is_isolated_between_async_tasks() -> None:
    async def capture_trace(seed: str, trace_id: str) -> str:
        context = ActiveTraceContext(
            trace_seed=seed,
            trace_id=trace_id,
        )

        with trace_context_scope(context):
            await asyncio.sleep(0)

            active_context = current_trace_context()
            assert active_context is not None
            return active_context.trace_seed

    results = await asyncio.gather(
        capture_trace("agent-run:one", "a" * 32),
        capture_trace("agent-run:two", "b" * 32),
    )

    assert list(results) == ["agent-run:one", "agent-run:two"]
    assert current_trace_context() is None


def test_active_trace_context_rejects_blank_trace_seed() -> None:
    with pytest.raises(ValueError, match="trace_seed must not be blank"):
        ActiveTraceContext(trace_seed=" ")


def test_active_observation_context_rejects_blank_name() -> None:
    with pytest.raises(ValueError, match="name must not be blank"):
        ActiveObservationContext(name=" ")
