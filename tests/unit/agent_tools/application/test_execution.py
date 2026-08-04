"""Unit tests for bounded read-only tool execution."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Literal
from uuid import UUID

import pytest

from supportops.agent_tools.application.bindings import (
    ExecutableToolBinding,
    ExecutableToolRegistry,
)
from supportops.agent_tools.application.execution import (
    BoundedReadOnlyToolExecutor,
    ExecuteToolCommand,
    ToolExecutionContext,
)
from supportops.agent_tools.domain.audit import (
    AgentToolCallStatus,
)
from supportops.agent_tools.domain.contracts import (
    StrictToolSchema,
    ToolAuditPolicy,
    ToolDefinition,
    ToolFailurePolicy,
    ToolSafetyLevel,
)
from supportops.agent_tools.domain.errors import (
    ToolDependencyUnavailableError,
    ToolInputValidationError,
    ToolNotFoundError,
)
from supportops.agent_tools.domain.fingerprints import (
    create_tool_call_fingerprint,
)
from supportops.observability.context import (
    ActiveObservationContext,
    current_observation_context,
    observation_context_scope,
)
from supportops.observability.contracts import TraceScope
from supportops.observability.models import (
    EventObservation,
    ObservabilityProvider,
    ObservationAttributes,
    ObservationStatus,
    ObservationType,
    ObservationUpdate,
    TraceAttributes,
)

_WORKSPACE_ID = UUID("10000000-0000-4000-8000-000000000001")
_TICKET_ID = UUID("20000000-0000-4000-8000-000000000002")
_AGENT_RUN_ID = UUID("30000000-0000-4000-8000-000000000003")
_ATTEMPT_ID = UUID("40000000-0000-4000-8000-000000000004")
_TOOL_CALL_ID = UUID("50000000-0000-4000-8000-000000000005")

_STARTED_AT = datetime(
    2026,
    8,
    2,
    18,
    0,
    tzinfo=UTC,
)
_FINISHED_AT = _STARTED_AT + timedelta(milliseconds=25)


class ExampleInput(StrictToolSchema):
    """Synthetic controlled input."""

    query: str
    top_k: int


class OtherInput(StrictToolSchema):
    """Incorrect validated input type."""

    service_name: str


class ExampleOutput(StrictToolSchema):
    """Synthetic controlled output."""

    result_count: int
    chunk_ids: tuple[str, ...]


class SuccessfulHandler:
    """Return one valid synthetic output."""

    def __init__(self) -> None:
        self.calls = 0
        self.received_context: object | None = None
        self.received_arguments: StrictToolSchema | None = None

    async def execute(
        self,
        context: object,
        arguments: StrictToolSchema,
    ) -> object:
        self.calls += 1
        self.received_context = context
        self.received_arguments = arguments

        return {
            "result_count": 1,
            "chunk_ids": [
                "chunk-1",
            ],
        }


class DependencyFailureHandler:
    """Raise one normalized dependency failure."""

    async def execute(
        self,
        context: object,
        arguments: StrictToolSchema,
    ) -> object:
        del context, arguments

        raise ToolDependencyUnavailableError()


class UnexpectedFailureHandler:
    """Raise one exception that must not escape."""

    async def execute(
        self,
        context: object,
        arguments: StrictToolSchema,
    ) -> object:
        del context, arguments

        raise RuntimeError("secret dependency details")


class InvalidOutputHandler:
    """Return output violating the approved schema."""

    async def execute(
        self,
        context: object,
        arguments: StrictToolSchema,
    ) -> object:
        del context, arguments

        return {
            "result_count": "invalid",
            "chunk_ids": [],
        }


class BlockingHandler:
    """Remain blocked until cancelled by the timeout."""

    def __init__(self) -> None:
        self.cancelled = False

    async def execute(
        self,
        context: object,
        arguments: StrictToolSchema,
    ) -> object:
        del context, arguments

        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled = True
            raise

        raise AssertionError("unreachable")


class SequenceClock:
    """Return deterministic sequential values."""

    def __init__(
        self,
        values: tuple[object, ...],
    ) -> None:
        self._values: Iterator[object] = iter(values)

    def __call__(self) -> object:
        return next(self._values)


def _definition(
    *,
    timeout_seconds: float = 5,
) -> ToolDefinition:
    return ToolDefinition(
        name="search_knowledge",
        version=1,
        description="Search workspace-scoped knowledge.",
        input_schema=ExampleInput,
        output_schema=ExampleOutput,
        safety_level=ToolSafetyLevel.READ_ONLY,
        timeout_seconds=timeout_seconds,
        failure_policy=(ToolFailurePolicy.RETRY_AGENT_RUN),
        audit_policy=ToolAuditPolicy.SAFE_PROJECTION,
    )


def _binding(
    handler: object,
    *,
    timeout_seconds: float = 5,
    output_projector: object | None = None,
) -> ExecutableToolBinding:
    if output_projector is None:

        def output_projector(value: object) -> object:
            return value.model_dump(mode="json")  # type: ignore[attr-defined]

    return ExecutableToolBinding(
        definition=_definition(timeout_seconds=timeout_seconds),
        handler=handler,  # type: ignore[arg-type]
        safe_input_projector=lambda value: {
            "query_length": len(str(value.model_dump(mode="json")["query"])),
            "top_k": value.model_dump(mode="json")["top_k"],
        },
        safe_output_projector=output_projector,  # type: ignore[arg-type]
    )


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
    )


def _arguments() -> ExampleInput:
    return ExampleInput(
        query="account access reset",
        top_k=5,
    )


def _command(
    *,
    prior_fingerprints: frozenset[str] = frozenset(),
    arguments: StrictToolSchema | None = None,
) -> ExecuteToolCommand:
    return ExecuteToolCommand(
        context=_context(),
        sequence=1,
        provider_tool_call_id="provider-tool-call-1",
        tool_name="search_knowledge",
        tool_version=1,
        arguments=arguments or _arguments(),
        prior_fingerprints=prior_fingerprints,
    )


def _executor(
    binding: ExecutableToolBinding,
    *,
    observability_client: object | None = None,
) -> BoundedReadOnlyToolExecutor:
    datetime_clock = SequenceClock(
        (
            _STARTED_AT,
            _FINISHED_AT,
        )
    )
    monotonic_clock = SequenceClock(
        (
            10.0,
            10.025,
        )
    )

    return BoundedReadOnlyToolExecutor(
        registry=ExecutableToolRegistry((binding,)),
        clock=datetime_clock,  # type: ignore[arg-type]
        monotonic_clock=monotonic_clock,  # type: ignore[arg-type]
        tool_call_id_factory=lambda: _TOOL_CALL_ID,
        observability_client=observability_client,  # type: ignore[arg-type]
    )


async def test_executes_and_returns_terminal_success_audit() -> None:
    handler = SuccessfulHandler()
    executor = _executor(_binding(handler))

    result = await executor.execute(_command())

    assert result.succeeded is True
    assert result.retryable is False
    assert result.output == ExampleOutput(
        result_count=1,
        chunk_ids=("chunk-1",),
    )
    assert handler.calls == 1
    assert handler.received_context == _context()
    assert handler.received_arguments == _arguments()

    assert result.audit.id == _TOOL_CALL_ID
    assert result.audit.workspace_id == _WORKSPACE_ID
    assert result.audit.ticket_id == _TICKET_ID
    assert result.audit.agent_run_id == _AGENT_RUN_ID
    assert result.audit.proposed_by_agent_run_attempt_id == _ATTEMPT_ID
    assert result.audit.executed_by_agent_run_attempt_id == _ATTEMPT_ID
    assert result.audit.sequence == 1
    assert result.audit.status is (AgentToolCallStatus.SUCCEEDED)
    assert result.audit.safe_input == {
        "query_length": 20,
        "top_k": 5,
    }
    assert result.audit.safe_output == {
        "result_count": 1,
        "chunk_ids": [
            "chunk-1",
        ],
    }
    assert result.audit.latency_ms == 25
    assert result.audit.error_code is None
    assert result.audit.proposed_at == _STARTED_AT
    assert result.audit.execution_started_at == _STARTED_AT
    assert result.audit.finished_at == _FINISHED_AT


async def test_rejects_repeated_fingerprint_without_execution() -> None:
    handler = SuccessfulHandler()
    binding = _binding(handler)
    fingerprint = create_tool_call_fingerprint(
        definition=binding.definition,
        arguments=_arguments(),
    )
    executor = _executor(binding)

    result = await executor.execute(
        _command(
            prior_fingerprints=frozenset(
                {
                    fingerprint,
                }
            )
        )
    )

    assert handler.calls == 0
    assert result.succeeded is False
    assert result.retryable is False
    assert result.output is None
    assert result.audit.status is (AgentToolCallStatus.REJECTED)
    assert result.audit.error_code == ("tool_call_repeated")


async def test_times_out_and_cancels_handler() -> None:
    handler = BlockingHandler()
    executor = _executor(
        _binding(
            handler,
            timeout_seconds=0.001,
        )
    )

    result = await executor.execute(_command())

    assert handler.cancelled is True
    assert result.succeeded is False
    assert result.retryable is True
    assert result.audit.status is (AgentToolCallStatus.TIMED_OUT)
    assert result.audit.error_code == "tool_timeout"
    assert result.audit.safe_output is None


async def test_normalizes_dependency_failure() -> None:
    executor = _executor(_binding(DependencyFailureHandler()))

    result = await executor.execute(_command())

    assert result.succeeded is False
    assert result.retryable is True
    assert result.audit.status is (AgentToolCallStatus.FAILED)
    assert result.audit.error_code == ("tool_dependency_unavailable")


async def test_rejects_invalid_dependency_output() -> None:
    executor = _executor(_binding(InvalidOutputHandler()))

    result = await executor.execute(_command())

    assert result.succeeded is False
    assert result.retryable is False
    assert result.audit.status is (AgentToolCallStatus.FAILED)
    assert result.audit.error_code == ("tool_output_invalid")


async def test_normalizes_unexpected_handler_failure() -> None:
    executor = _executor(_binding(UnexpectedFailureHandler()))

    result = await executor.execute(_command())

    assert result.succeeded is False
    assert result.retryable is True
    assert result.audit.status is (AgentToolCallStatus.FAILED)
    assert result.audit.error_code == "tool_unexpected"
    assert "secret dependency details" not in str(result.audit)


async def test_normalizes_safe_output_projection_failure() -> None:
    def fail_projection(
        value: StrictToolSchema,
    ) -> dict[str, object]:
        del value

        raise RuntimeError("unsafe projection failure")

    executor = _executor(
        _binding(
            SuccessfulHandler(),
            output_projector=fail_projection,
        )
    )

    result = await executor.execute(_command())

    assert result.succeeded is False
    assert result.retryable is True
    assert result.audit.status is (AgentToolCallStatus.FAILED)
    assert result.audit.error_code == "tool_unexpected"
    assert result.audit.safe_output is None


async def test_rejects_wrong_validated_input_schema() -> None:
    executor = _executor(_binding(SuccessfulHandler()))

    with pytest.raises(
        ToolInputValidationError,
        match="arguments are invalid",
    ):
        await executor.execute(_command(arguments=OtherInput(service_name="payments")))


def test_command_validates_prior_fingerprints() -> None:
    with pytest.raises(
        ValueError,
        match="lowercase SHA-256",
    ):
        _command(
            prior_fingerprints=frozenset(
                {
                    "invalid",
                }
            )
        )


def test_execution_context_is_immutable() -> None:
    context = _context()

    with pytest.raises(FrozenInstanceError):
        context.workspace_id = UUID(  # type: ignore[misc]
            "60000000-0000-4000-8000-000000000006",
        )


class RecordingObservationScope:
    def __init__(
        self,
        *,
        attributes: ObservationAttributes,
        fail_update: bool = False,
    ) -> None:
        self.attributes = attributes
        self._fail_update = fail_update
        self.updates: list[ObservationUpdate] = []

    @property
    def observation_id(self) -> str | None:
        return "tool-observation-1"

    def update(self, update: ObservationUpdate) -> None:
        if self._fail_update:
            raise RuntimeError("synthetic update failure")

        self.updates.append(update)

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[RecordingObservationScope]:
        del attributes
        raise AssertionError("Nested observations are not expected.")

    def record_event(self, event: EventObservation) -> None:
        del event
        raise AssertionError("Events are not expected.")


class RecordingObservationManager(AbstractContextManager[RecordingObservationScope]):
    def __init__(
        self,
        *,
        scope: RecordingObservationScope,
        fail_enter: bool = False,
        fail_exit: bool = False,
    ) -> None:
        self._scope = scope
        self._fail_enter = fail_enter
        self._fail_exit = fail_exit
        self.exit_calls = 0
        self._context_manager = observation_context_scope(
            ActiveObservationContext(
                name=scope.attributes.name,
                observation_id=scope.observation_id,
            )
        )

    def __enter__(self) -> RecordingObservationScope:
        if self._fail_enter:
            raise RuntimeError("synthetic enter failure")

        self._context_manager.__enter__()
        return self._scope

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self.exit_calls += 1
        self._context_manager.__exit__(exc_type, exc, traceback)

        if self._fail_exit:
            raise RuntimeError("synthetic exit failure")

        return False


class RecordingObservabilityClient:
    def __init__(
        self,
        *,
        fail_start: bool = False,
        fail_enter: bool = False,
        fail_update: bool = False,
        fail_exit: bool = False,
    ) -> None:
        self._fail_start = fail_start
        self._fail_enter = fail_enter
        self._fail_update = fail_update
        self._fail_exit = fail_exit
        self.attributes: list[ObservationAttributes] = []
        self.scopes: list[RecordingObservationScope] = []
        self.managers: list[RecordingObservationManager] = []
        self.parent_observation_names: list[str | None] = []

    @property
    def provider(self) -> ObservabilityProvider:
        return ObservabilityProvider.NOOP

    @property
    def enabled(self) -> bool:
        return True

    def start_trace(
        self,
        attributes: TraceAttributes,
    ) -> AbstractContextManager[TraceScope]:
        del attributes
        raise AssertionError("Tool tracing must not create roots.")

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[RecordingObservationScope]:
        if self._fail_start:
            raise RuntimeError("synthetic start failure")

        parent = current_observation_context()
        self.parent_observation_names.append(
            None if parent is None else parent.name,
        )

        scope = RecordingObservationScope(
            attributes=attributes,
            fail_update=self._fail_update,
        )
        manager = RecordingObservationManager(
            scope=scope,
            fail_enter=self._fail_enter,
            fail_exit=self._fail_exit,
        )
        self.attributes.append(attributes)
        self.scopes.append(scope)
        self.managers.append(manager)
        return manager

    def record_event(self, event: EventObservation) -> None:
        del event
        raise AssertionError("Tool tracing must not emit events.")

    def record_trace_event(self, *, identity: object, event: EventObservation) -> None:
        del identity, event
        raise AssertionError("Tool tracing must not emit events.")

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class NestedRetrievalHandler:
    """Simulate retrieval nesting under the active TOOL observation."""

    def __init__(self) -> None:
        self.parent_names: list[str | None] = []
        self.nested_names: list[str] = []

    async def execute(
        self,
        context: object,
        arguments: StrictToolSchema,
    ) -> object:
        del context, arguments
        parent = current_observation_context()
        self.parent_names.append(None if parent is None else parent.name)
        self.nested_names.append("knowledge.search")
        return {
            "result_count": 1,
            "chunk_ids": ["chunk-1"],
        }


async def test_records_one_tool_observation_for_actual_execution() -> None:
    observability = RecordingObservabilityClient()
    executor = _executor(
        _binding(SuccessfulHandler()),
        observability_client=observability,
    )

    result = await executor.execute(_command())

    assert result.succeeded is True
    assert len(observability.attributes) == 1
    attributes = observability.attributes[0]
    assert attributes.name == "tool.execute"
    assert attributes.observation_type is ObservationType.TOOL
    assert attributes.input_data is None
    assert attributes.input_paths == frozenset()
    assert attributes.output_paths == frozenset()
    assert attributes.metadata == {
        "tool_name": "search_knowledge",
        "tool_safety": "read_only",
        "requires_approval": False,
        "agent_run_id": str(_AGENT_RUN_ID),
        "agent_run_attempt_id": str(_ATTEMPT_ID),
        "tool_call_id": str(_TOOL_CALL_ID),
        "workspace_id": str(_WORKSPACE_ID),
        "ticket_id": str(_TICKET_ID),
    }
    update = observability.scopes[0].updates[0]
    assert update.status is ObservationStatus.OK
    assert update.metadata == {
        "status": "ok",
        "tool_outcome": "succeeded",
        "latency_ms": 25,
    }
    assert update.output_data is None
    assert current_observation_context() is None
    assert observability.managers[0].exit_calls == 1


async def test_maps_normalized_and_unexpected_execution_failures() -> None:
    dependency_observability = RecordingObservabilityClient()
    dependency_result = await _executor(
        _binding(DependencyFailureHandler()),
        observability_client=dependency_observability,
    ).execute(_command())

    assert dependency_result.audit.error_code == "tool_dependency_unavailable"
    dependency_update = dependency_observability.scopes[0].updates[0]
    assert dependency_update.status is ObservationStatus.ERROR
    assert dependency_update.metadata["tool_outcome"] == "failed"
    assert dependency_update.metadata["error_code"] == ("tool_dependency_unavailable")

    unexpected_observability = RecordingObservabilityClient()
    unexpected_result = await _executor(
        _binding(UnexpectedFailureHandler()),
        observability_client=unexpected_observability,
    ).execute(_command())

    assert unexpected_result.audit.error_code == "tool_unexpected"
    unexpected_update = unexpected_observability.scopes[0].updates[0]
    assert unexpected_update.status is ObservationStatus.ERROR
    assert unexpected_update.metadata["tool_outcome"] == ("unexpected_failure")
    assert unexpected_update.metadata["error_code"] == "tool_unexpected"
    assert "secret dependency details" not in repr(unexpected_observability.attributes)
    assert "secret dependency details" not in repr(unexpected_update.metadata)


async def test_unknown_and_invalid_arguments_create_no_tool_observation() -> None:
    observability = RecordingObservabilityClient()
    executor = _executor(
        _binding(SuccessfulHandler()),
        observability_client=observability,
    )

    with pytest.raises(ToolNotFoundError):
        await executor.execute(
            ExecuteToolCommand(
                context=_context(),
                sequence=1,
                provider_tool_call_id="provider-tool-call-1",
                tool_name="lookup_service_status",
                tool_version=1,
                arguments=_arguments(),
            )
        )

    with pytest.raises(ToolInputValidationError):
        await executor.execute(_command(arguments=OtherInput(service_name="payments")))

    assert observability.attributes == []


async def test_pre_execution_rejection_creates_no_tool_observation() -> None:
    handler = SuccessfulHandler()
    binding = _binding(handler)
    fingerprint = create_tool_call_fingerprint(
        definition=binding.definition,
        arguments=_arguments(),
    )
    observability = RecordingObservabilityClient()
    executor = _executor(
        binding,
        observability_client=observability,
    )

    result = await executor.execute(
        _command(prior_fingerprints=frozenset({fingerprint})),
    )

    assert handler.calls == 0
    assert result.audit.status is AgentToolCallStatus.REJECTED
    assert observability.attributes == []


async def test_retrieval_backed_execution_does_not_duplicate_knowledge_search() -> None:
    handler = NestedRetrievalHandler()
    observability = RecordingObservabilityClient()
    executor = _executor(
        _binding(handler),
        observability_client=observability,
    )

    await executor.execute(_command())

    assert len(observability.attributes) == 1
    assert observability.attributes[0].name == "tool.execute"
    assert handler.parent_names == ["tool.execute"]
    assert handler.nested_names == ["knowledge.search"]
    assert sum(1 for name in handler.nested_names if name == "knowledge.search") == 1


async def test_tool_observation_nests_under_active_parent_context() -> None:
    observability = RecordingObservabilityClient()
    executor = _executor(
        _binding(SuccessfulHandler()),
        observability_client=observability,
    )

    with observation_context_scope(
        ActiveObservationContext(
            name="graph-node.decide_and_execute",
            observation_id="node-1",
        )
    ):
        await executor.execute(_command())

    assert observability.parent_observation_names == ["graph-node.decide_and_execute"]
    assert current_observation_context() is None


async def test_observability_failures_fail_open_for_tool_execution() -> None:
    for kwargs in (
        {"fail_start": True},
        {"fail_enter": True},
        {"fail_update": True},
        {"fail_exit": True},
    ):
        observability = RecordingObservabilityClient(**kwargs)
        result = await _executor(
            _binding(SuccessfulHandler()),
            observability_client=observability,
        ).execute(_command())

        assert result.succeeded is True
        assert current_observation_context() is None


async def test_observability_failures_preserve_normalized_business_result() -> None:
    observability = RecordingObservabilityClient(fail_update=True)

    result = await _executor(
        _binding(DependencyFailureHandler()),
        observability_client=observability,
    ).execute(_command())

    assert result.audit.error_code == "tool_dependency_unavailable"
    assert result.succeeded is False
    assert current_observation_context() is None


async def test_exports_no_tool_arguments_or_outputs() -> None:
    observability = RecordingObservabilityClient()
    await _executor(
        _binding(SuccessfulHandler()),
        observability_client=observability,
    ).execute(_command())

    attributes = observability.attributes[0]
    update = observability.scopes[0].updates[0]
    exported = repr(attributes) + repr(update)
    forbidden = {
        "ticket_subject",
        "ticket_description",
        "conversation_content",
        "graph_state",
        "checkpoint_payload",
        "prompt_content",
        "model_output",
        "classification_text",
        "tool_arguments",
        "tool_output",
        "proposed_input",
        "approval_comment",
        "approver_identity",
        "escalation_reason",
        "recommendation_text",
        "decision_summary",
        "citation_text",
        "evidence_content",
        "document_content",
        "chunk_content",
        "embedding_vectors",
        "lease_token",
        "execution_grant",
        "authorization_headers",
        "credentials",
        "traceback",
        "user_id",
    }

    assert "account access reset" not in exported
    assert "chunk-1" not in exported
    assert "query" not in attributes.metadata
    assert "arguments" not in attributes.metadata
    assert attributes.input_data is None
    assert update.output_data is None
    assert forbidden.isdisjoint(attributes.metadata)
    assert forbidden.isdisjoint(update.metadata)
