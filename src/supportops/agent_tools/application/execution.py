"""Bounded application-owned execution for read-only tools."""

import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import JsonValue, ValidationError

from supportops.agent_tools.application.bindings import (
    ExecutableToolBinding,
    ExecutableToolRegistry,
)
from supportops.agent_tools.domain.audit import (
    AgentToolCall,
    AgentToolCallStatus,
    utc_now,
)
from supportops.agent_tools.domain.contracts import (
    StrictToolSchema,
    ToolFailurePolicy,
)
from supportops.agent_tools.domain.errors import (
    ToolError,
    ToolInputValidationError,
    ToolOutputValidationError,
    ToolRepeatedCallError,
    ToolSafetyViolationError,
    ToolTimeoutError,
    ToolUnexpectedError,
)
from supportops.agent_tools.domain.fingerprints import (
    create_tool_call_fingerprint,
)

type DateTimeClock = Callable[[], datetime]
type MonotonicClock = Callable[[], float]
type ToolCallIdFactory = Callable[[], UUID]


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """Trusted application context unavailable to the model."""

    workspace_id: UUID
    ticket_id: UUID
    agent_run_id: UUID
    agent_run_attempt_id: UUID

    def __post_init__(self) -> None:
        values = (
            self.workspace_id,
            self.ticket_id,
            self.agent_run_id,
            self.agent_run_attempt_id,
        )

        if not all(isinstance(value, UUID) for value in values):
            raise TypeError("Tool execution context identifiers must be UUID values.")


@dataclass(frozen=True, slots=True)
class ExecuteToolCommand:
    """Execute one previously validated provider tool decision."""

    context: ToolExecutionContext
    sequence: int
    provider_tool_call_id: str | None
    tool_name: str
    tool_version: int
    arguments: StrictToolSchema
    prior_fingerprints: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.sequence <= 0:
            raise ValueError("sequence must be positive.")

        _validate_optional_identifier(
            self.provider_tool_call_id,
            field_name="provider_tool_call_id",
            maximum_length=255,
        )
        _validate_required_identifier(
            self.tool_name,
            field_name="tool_name",
            maximum_length=64,
        )

        if self.tool_version <= 0:
            raise ValueError("tool_version must be positive.")

        if not isinstance(
            self.arguments,
            StrictToolSchema,
        ):
            raise TypeError("arguments must be a validated StrictToolSchema.")

        for fingerprint in self.prior_fingerprints:
            _validate_fingerprint(fingerprint)


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Terminal execution result prepared for fenced persistence."""

    audit: AgentToolCall
    output: StrictToolSchema | None
    retryable: bool
    failure_policy: ToolFailurePolicy

    def __post_init__(self) -> None:
        if self.audit.status is AgentToolCallStatus.SUCCEEDED:
            if self.output is None:
                raise ValueError("Successful tool execution requires output.")

            if self.retryable:
                raise ValueError("Successful tool execution cannot be retryable.")

            return

        if self.output is not None:
            raise ValueError("Unsuccessful tool execution cannot expose output.")

    @property
    def succeeded(self) -> bool:
        """Return whether the tool produced an accepted output."""

        return self.audit.status is AgentToolCallStatus.SUCCEEDED


class BoundedReadOnlyToolExecutor:
    """Execute exact registered tools under bounded policy."""

    def __init__(
        self,
        *,
        registry: ExecutableToolRegistry,
        clock: DateTimeClock = utc_now,
        monotonic_clock: MonotonicClock = (time.perf_counter),
        tool_call_id_factory: ToolCallIdFactory = uuid4,
    ) -> None:
        self._registry = registry
        self._clock = clock
        self._monotonic_clock = monotonic_clock
        self._tool_call_id_factory = tool_call_id_factory

    async def execute(
        self,
        command: ExecuteToolCommand,
    ) -> ToolExecutionResult:
        """Execute one exact read-only binding."""

        binding = self._registry.lookup(
            name=command.tool_name,
            version=command.tool_version,
        )
        definition = binding.definition

        if not isinstance(
            command.arguments,
            definition.input_schema,
        ):
            raise ToolInputValidationError()

        fingerprint = create_tool_call_fingerprint(
            definition=definition,
            arguments=command.arguments,
        )
        started_at = self._clock()
        started_monotonic = self._monotonic_clock()

        try:
            safe_input = _materialize_projection(binding.safe_input_projector(command.arguments))
        except Exception:
            return self._failure_result(
                command=command,
                binding=binding,
                fingerprint=fingerprint,
                safe_input={},
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=ToolUnexpectedError(),
            )

        if fingerprint in command.prior_fingerprints:
            return self._failure_result(
                command=command,
                binding=binding,
                fingerprint=fingerprint,
                safe_input=safe_input,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=ToolRepeatedCallError(),
            )

        try:
            async with asyncio.timeout(definition.timeout_seconds):
                raw_output = await binding.handler.execute(
                    command.context,
                    command.arguments,
                )

            validated_output = definition.output_schema.model_validate(raw_output)
            safe_output = _materialize_projection(binding.safe_output_projector(validated_output))
        except TimeoutError:
            return self._failure_result(
                command=command,
                binding=binding,
                fingerprint=fingerprint,
                safe_input=safe_input,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=ToolTimeoutError(),
            )
        except ValidationError:
            return self._failure_result(
                command=command,
                binding=binding,
                fingerprint=fingerprint,
                safe_input=safe_input,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=ToolOutputValidationError(),
            )
        except ToolError as error:
            return self._failure_result(
                command=command,
                binding=binding,
                fingerprint=fingerprint,
                safe_input=safe_input,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=error,
            )
        except Exception:
            return self._failure_result(
                command=command,
                binding=binding,
                fingerprint=fingerprint,
                safe_input=safe_input,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=ToolUnexpectedError(),
            )

        finished_at, latency_ms = self._finish_timing(
            started_at=started_at,
            started_monotonic=started_monotonic,
        )
        audit = AgentToolCall.create(
            tool_call_id=self._tool_call_id_factory(),
            workspace_id=command.context.workspace_id,
            ticket_id=command.context.ticket_id,
            agent_run_id=command.context.agent_run_id,
            agent_run_attempt_id=(command.context.agent_run_attempt_id),
            sequence=command.sequence,
            provider_tool_call_id=(command.provider_tool_call_id),
            tool_name=definition.name,
            tool_version=definition.version,
            safety_level=definition.safety_level,
            status=AgentToolCallStatus.SUCCEEDED,
            input_fingerprint=fingerprint,
            safe_input=safe_input,
            safe_output=safe_output,
            latency_ms=latency_ms,
            error_code=None,
            started_at=started_at,
            finished_at=finished_at,
        )

        return ToolExecutionResult(
            audit=audit,
            output=validated_output,
            retryable=False,
            failure_policy=definition.failure_policy,
        )

    def _failure_result(
        self,
        *,
        command: ExecuteToolCommand,
        binding: ExecutableToolBinding,
        fingerprint: str,
        safe_input: Mapping[str, JsonValue],
        started_at: datetime,
        started_monotonic: float,
        error: ToolError,
    ) -> ToolExecutionResult:
        finished_at, latency_ms = self._finish_timing(
            started_at=started_at,
            started_monotonic=started_monotonic,
        )
        status = _status_for_error(error)
        definition = binding.definition
        audit = AgentToolCall.create(
            tool_call_id=self._tool_call_id_factory(),
            workspace_id=command.context.workspace_id,
            ticket_id=command.context.ticket_id,
            agent_run_id=command.context.agent_run_id,
            agent_run_attempt_id=(command.context.agent_run_attempt_id),
            sequence=command.sequence,
            provider_tool_call_id=(command.provider_tool_call_id),
            tool_name=definition.name,
            tool_version=definition.version,
            safety_level=definition.safety_level,
            status=status,
            input_fingerprint=fingerprint,
            safe_input=safe_input,
            safe_output=None,
            latency_ms=latency_ms,
            error_code=error.error_code,
            started_at=started_at,
            finished_at=finished_at,
        )

        return ToolExecutionResult(
            audit=audit,
            output=None,
            retryable=error.retryable,
            failure_policy=definition.failure_policy,
        )

    def _finish_timing(
        self,
        *,
        started_at: datetime,
        started_monotonic: float,
    ) -> tuple[datetime, int]:
        finished_at = self._clock()
        finished_monotonic = self._monotonic_clock()
        elapsed_seconds = finished_monotonic - started_monotonic

        if elapsed_seconds < 0:
            raise RuntimeError("The tool execution clock moved backwards.")

        if finished_at < started_at:
            raise RuntimeError("The tool execution timestamp clock moved backwards.")

        return (
            finished_at,
            round(elapsed_seconds * 1_000),
        )


def _status_for_error(
    error: ToolError,
) -> AgentToolCallStatus:
    if isinstance(
        error,
        (
            ToolRepeatedCallError,
            ToolInputValidationError,
            ToolSafetyViolationError,
        ),
    ):
        return AgentToolCallStatus.REJECTED

    if isinstance(
        error,
        ToolTimeoutError,
    ):
        return AgentToolCallStatus.TIMED_OUT

    return AgentToolCallStatus.FAILED


def _materialize_projection(
    value: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    return dict(value)


def _validate_fingerprint(
    value: str,
) -> None:
    if len(value) != 64:
        raise ValueError("prior_fingerprints must contain lowercase SHA-256 hashes.")

    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError("prior_fingerprints must contain lowercase SHA-256 hashes.")


def _validate_required_identifier(
    value: str,
    *,
    field_name: str,
    maximum_length: int,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")

    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")

    if len(value) > maximum_length:
        raise ValueError(f"{field_name} exceeds the maximum length.")


def _validate_optional_identifier(
    value: str | None,
    *,
    field_name: str,
    maximum_length: int,
) -> None:
    if value is None:
        return

    _validate_required_identifier(
        value,
        field_name=field_name,
        maximum_length=maximum_length,
    )
