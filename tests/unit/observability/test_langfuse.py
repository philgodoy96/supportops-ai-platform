"""Unit tests for the fail-open Langfuse adapter."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from decimal import Decimal
from types import TracebackType
from typing import Literal
from unittest.mock import patch

import pytest

from supportops.observability.contracts import ObservabilityClient, TraceScope
from supportops.observability.identity import TraceIdentity
from supportops.observability.langfuse import (
    LangfuseObservabilityClient,
)
from supportops.observability.models import (
    CostDetails,
    EventObservation,
    ObservationAttributes,
    ObservationStatus,
    ObservationType,
    ObservationUpdate,
    PricingStatus,
    TraceAttributes,
    UsageDetails,
)
from supportops.observability.privacy import (
    MetadataOnlyExportPolicy,
)


class FakeObservation:
    def __init__(
        self,
        *,
        trace_id: str,
        observation_id: str,
    ) -> None:
        self.trace_id = trace_id
        self.id = observation_id
        self.updates: list[dict[str, object]] = []

    def update(self, **kwargs: object) -> object:
        self.updates.append(kwargs)
        return self


class FakeManager(AbstractContextManager[FakeObservation]):
    def __init__(self, observation: FakeObservation) -> None:
        self.observation = observation
        self.exit_args: tuple[object, object, object] | None = None

    def __enter__(self) -> FakeObservation:
        return self.observation

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self.exit_args = (exc_type, exc, traceback)
        return False


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.starts: list[dict[str, object]] = []
        self.managers: list[FakeManager] = []
        self.events: list[dict[str, object]] = []
        self.flush_calls = 0
        self.shutdown_calls = 0
        self.fail_start = False
        self.fail_event = False

    def create_trace_id(self, *, seed: str) -> str:
        return (seed.encode().hex() + ("0" * 32))[:32]

    def start_as_current_observation(
        self,
        **kwargs: object,
    ) -> AbstractContextManager[FakeObservation]:
        if self.fail_start:
            raise RuntimeError("sdk unavailable")

        self.starts.append(kwargs)

        trace_context = kwargs.get("trace_context")
        trace_id = trace_context["trace_id"] if isinstance(trace_context, dict) else "f" * 32

        manager = FakeManager(
            FakeObservation(
                trace_id=str(trace_id),
                observation_id=(f"{len(self.managers) + 1:016x}"),
            )
        )
        self.managers.append(manager)

        return manager

    def create_event(
        self,
        **kwargs: object,
    ) -> FakeObservation:
        if self.fail_event:
            raise RuntimeError("event export unavailable")

        self.events.append(kwargs)

        return FakeObservation(
            trace_id="f" * 32,
            observation_id=f"{len(self.events):016x}",
        )

    def flush(self) -> None:
        self.flush_calls += 1

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _client(
    sdk: FakeLangfuseClient,
) -> LangfuseObservabilityClient:
    return LangfuseObservabilityClient(
        sdk_client=sdk,
        export_policy=MetadataOnlyExportPolicy(),
    )


def test_langfuse_trace_uses_deterministic_seed_and_safe_metadata() -> None:
    sdk = FakeLangfuseClient()
    client = _client(sdk)

    attributes = TraceAttributes(
        trace_seed="agent-run:1",
        name="agent-run",
        session_id="ticket:1",
        metadata={
            "workspace_id": "workspace-1",
            "secret": "remove",
        },
        metadata_paths=frozenset(
            {
                ("workspace_id",),
                ("secret",),
            }
        ),
        tags=("supportops", "agent-run"),
    )

    with client.start_trace(attributes) as trace:
        assert trace.trace_id is not None

    start = sdk.starts[0]

    assert start["as_type"] == "agent"
    assert start["metadata"] == {"workspace_id": "workspace-1"}

    trace_context = start["trace_context"]
    assert isinstance(trace_context, dict)
    assert trace_context["trace_id"] == trace.trace_id


def test_langfuse_trace_update_reaches_root_observation() -> None:
    sdk = FakeLangfuseClient()
    client = _client(sdk)

    with client.start_trace(
        TraceAttributes(
            trace_seed="agent-run:1",
            name="agent-run",
            metadata_paths=frozenset({("agent_run_status",)}),
        )
    ) as trace:
        assert isinstance(trace, TraceScope)
        trace.update(
            ObservationUpdate(
                status=ObservationStatus.OK,
                metadata={"agent_run_status": "succeeded"},
            )
        )

    assert sdk.managers[0].observation.updates[0]["metadata"] == {
        "agent_run_status": "succeeded",
    }


def test_langfuse_trace_update_failure_fails_open() -> None:
    sdk = FakeLangfuseClient()
    client = _client(sdk)

    with client.start_trace(
        TraceAttributes(
            trace_seed="agent-run:1",
            name="agent-run",
        )
    ) as trace:
        root = sdk.managers[0].observation

        def failing_update(**kwargs: object) -> object:
            del kwargs
            raise RuntimeError("update unavailable")

        root.update = failing_update  # type: ignore[method-assign]
        trace.update(
            ObservationUpdate(
                status=ObservationStatus.ERROR,
                error_code="retryable_failure",
            )
        )


def test_generation_mapping_avoids_total_double_counting() -> None:
    sdk = FakeLangfuseClient()
    client = _client(sdk)

    with client.start_observation(
        ObservationAttributes(
            name="generation",
            observation_type=ObservationType.GENERATION,
        )
    ) as observation:
        observation.update(
            ObservationUpdate(
                usage=UsageDetails(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                ),
                cost=CostDetails(
                    pricing_status=PricingStatus.KNOWN,
                    input_cost=Decimal("0"),
                    output_cost=Decimal("0"),
                    total_cost=Decimal("0"),
                    pricing_catalog_version="catalog-v1",
                ),
            )
        )

    update = sdk.managers[0].observation.updates[0]

    assert update["usage_details"] == {
        "input": 10,
        "output": 5,
    }
    assert update["cost_details"] == {
        "input": 0.0,
        "output": 0.0,
    }


def test_unknown_pricing_omits_cost() -> None:
    sdk = FakeLangfuseClient()
    client = _client(sdk)

    with client.start_observation(
        ObservationAttributes(
            name="embedding",
            observation_type=ObservationType.EMBEDDING,
        )
    ) as observation:
        observation.update(
            ObservationUpdate(
                usage=UsageDetails(total_tokens=7),
                cost=CostDetails(pricing_status=PricingStatus.UNKNOWN),
            )
        )

    update = sdk.managers[0].observation.updates[0]

    assert update["usage_details"] == {"total": 7}
    assert "cost_details" not in update


def test_sdk_start_failure_fails_open() -> None:
    sdk = FakeLangfuseClient()
    sdk.fail_start = True
    client = _client(sdk)

    with client.start_observation(
        ObservationAttributes(
            name="worker-attempt",
            observation_type=ObservationType.SPAN,
        )
    ) as observation:
        assert observation.observation_id is None


def test_business_exception_is_preserved_and_not_given_to_sdk() -> None:
    sdk = FakeLangfuseClient()
    client = _client(sdk)

    with (
        pytest.raises(RuntimeError, match="business failure"),
        client.start_observation(
            ObservationAttributes(
                name="worker-attempt",
                observation_type=ObservationType.SPAN,
            )
        ),
    ):
        raise RuntimeError("business failure")

    assert sdk.managers[0].exit_args == (
        None,
        None,
        None,
    )
    assert sdk.managers[0].observation.updates[-1]["status_message"] == "unhandled_business_error"


def test_event_and_lifecycle_calls_are_forwarded() -> None:
    sdk = FakeLangfuseClient()
    client = _client(sdk)

    client.record_event(
        EventObservation(
            name="workflow_paused",
            metadata={"approval_request_id": "approval-1"},
            metadata_paths=frozenset({("approval_request_id",)}),
            status=ObservationStatus.OK,
        )
    )
    client.flush()
    client.shutdown()

    assert sdk.events[0]["name"] == "workflow_paused"
    assert sdk.events[0]["metadata"] == {"approval_request_id": "approval-1"}
    assert sdk.flush_calls == 1
    assert sdk.shutdown_calls == 1


def test_identity_scoped_event_derives_deterministic_trace_id() -> None:
    sdk = FakeLangfuseClient()
    client = _client(sdk)
    identity = TraceIdentity(
        trace_seed="agent-run:1",
        trace_name="agent-run",
        session_id="ticket:1",
        tags=("supportops", "agent-run"),
    )

    with patch(
        "supportops.observability.langfuse.propagate_attributes",
        return_value=nullcontext(),
    ) as propagate:
        client.record_trace_event(
            identity=identity,
            event=EventObservation(
                name="approval.granted",
                metadata={
                    "approval_request_id": "approval-1",
                    "secret": "remove",
                },
                metadata_paths=frozenset(
                    {
                        ("approval_request_id",),
                        ("secret",),
                    }
                ),
            ),
        )

    assert len(sdk.starts) == 0
    assert len(sdk.events) == 1

    event = sdk.events[0]
    expected_trace_id = sdk.create_trace_id(seed=identity.trace_seed)

    assert event["name"] == "approval.granted"
    assert event["metadata"] == {"approval_request_id": "approval-1"}
    assert event["trace_context"] == {"trace_id": expected_trace_id}
    assert "user_id" not in event

    propagate.assert_called_once()
    propagate_kwargs = propagate.call_args.kwargs
    assert propagate_kwargs["session_id"] == "ticket:1"
    assert propagate_kwargs["tags"] == ["supportops", "agent-run"]
    assert propagate_kwargs["trace_name"] == "agent-run"
    assert "user_id" not in propagate_kwargs


def test_identity_scoped_event_export_failure_fails_open() -> None:
    sdk = FakeLangfuseClient()
    sdk.fail_event = True
    client = _client(sdk)

    client.record_trace_event(
        identity=TraceIdentity(
            trace_seed="agent-run:1",
            trace_name="agent-run",
        ),
        event=EventObservation(name="approval.expired"),
    )


def test_langfuse_client_satisfies_application_owned_contracts() -> None:
    sdk = FakeLangfuseClient()
    client = _client(sdk)

    assert isinstance(client, ObservabilityClient)

    with client.start_trace(
        TraceAttributes(
            trace_seed="agent-run:1",
            name="agent-run",
        )
    ) as trace:
        assert isinstance(trace, TraceScope)
        assert not hasattr(trace, "create_trace_id")
