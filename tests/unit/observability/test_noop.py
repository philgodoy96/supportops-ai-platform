"""Unit tests for the no-op observability adapter."""

import pytest

from supportops.observability.context import (
    current_observation_context,
    current_trace_context,
)
from supportops.observability.identity import TraceIdentity
from supportops.observability.models import (
    EventObservation,
    ObservationAttributes,
    ObservationType,
    ObservationUpdate,
    TraceAttributes,
)
from supportops.observability.noop import NoOpObservabilityClient


def test_noop_client_satisfies_contract_without_export() -> None:
    client = NoOpObservabilityClient()

    assert client.enabled is False
    assert client.provider.value == "noop"

    with client.start_trace(
        TraceAttributes(
            trace_seed="agent-run:1",
            name="agent-run",
            session_id="ticket:1",
        )
    ) as trace:
        assert trace.trace_seed == "agent-run:1"
        assert trace.trace_id is None
        assert current_trace_context() is not None

        with trace.start_observation(
            ObservationAttributes(
                name="worker-attempt",
                observation_type=ObservationType.SPAN,
            )
        ) as observation:
            assert observation.observation_id is not None
            assert current_observation_context() is not None

            observation.update(ObservationUpdate())
            observation.record_event(EventObservation(name="attempt_started"))

        trace.update(ObservationUpdate())

    assert current_trace_context() is None
    assert current_observation_context() is None

    client.record_trace_event(
        identity=TraceIdentity(
            trace_seed="agent-run:1",
            trace_name="agent-run",
            session_id="ticket:1",
            tags=("supportops", "agent-run"),
        ),
        event=EventObservation(name="approval.granted"),
    )
    client.flush()
    client.shutdown()


def test_noop_trace_update_is_safe() -> None:
    client = NoOpObservabilityClient()

    with client.start_trace(
        TraceAttributes(
            trace_seed="agent-run:1",
            name="agent-run",
        )
    ) as trace:
        trace.update(
            ObservationUpdate(
                status=None,
                metadata={"agent_run_status": "succeeded"},
            )
        )


def test_noop_identity_scoped_event_is_safe() -> None:
    client = NoOpObservabilityClient()

    client.record_trace_event(
        identity=TraceIdentity(
            trace_seed="agent-run:1",
            trace_name="agent-run",
        ),
        event=EventObservation(name="approval.expired"),
    )


def test_noop_context_cleanup_preserves_business_exception() -> None:
    client = NoOpObservabilityClient()

    with (
        pytest.raises(RuntimeError, match="business failure"),
        client.start_trace(
            TraceAttributes(
                trace_seed="agent-run:1",
                name="agent-run",
            )
        ),
    ):
        raise RuntimeError("business failure")

    assert current_trace_context() is None
