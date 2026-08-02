"""Unit tests for bounded read-only tool execution."""

import asyncio
from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
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
)
from supportops.agent_tools.domain.fingerprints import (
    create_tool_call_fingerprint,
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
    assert result.audit.agent_run_attempt_id == _ATTEMPT_ID
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
    assert result.audit.started_at == _STARTED_AT
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
