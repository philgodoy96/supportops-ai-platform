"""Bounded application-owned execution for read-only tools."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final
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
    ToolSafetyLevel,
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
from supportops.observability.contracts import (
    ObservabilityClient,
    ObservationScope,
)
from supportops.observability.models import (
    ObservationAttributes,
    ObservationStatus,
    ObservationType,
    ObservationUpdate,
)
from supportops.observability.noop import NoOpObservabilityClient

type DateTimeClock = Callable[[], datetime]
type MonotonicClock = Callable[[], float]
type ToolCallIdFactory = Callable[[], UUID]

_OBSERVATION_NAME: Final = "tool.execute"

_OBSERVATION_METADATA_KEYS: Final = frozenset(
    {
        "tool_name",
        "tool_safety",
        "requires_approval",
        "agent_run_id",
        "agent_run_attempt_id",
        "tool_call_id",
        "workspace_id",
        "ticket_id",
        "execution_request_id",
        "correlation_id",
        "status",
        "tool_outcome",
        "error_code",
        "latency_ms",
        "idempotent_replay",
    }
)
_OBSERVATION_METADATA_PATHS: Final = frozenset((key,) for key in _OBSERVATION_METADATA_KEYS)


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
        observability_client: ObservabilityClient | None = None,
    ) -> None:
        self._registry = registry
        self._clock = clock
        self._monotonic_clock = monotonic_clock
        self._tool_call_id_factory = tool_call_id_factory
        self._observability_client = observability_client or NoOpObservabilityClient()

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

        tool_call_id = self._tool_call_id_factory()
        observation = _SafeToolObservation(
            client=self._observability_client,
            attributes=_start_attributes(
                command=command,
                binding=binding,
                tool_call_id=tool_call_id,
            ),
        )
        observation.start()

        try:
            try:
                async with asyncio.timeout(definition.timeout_seconds):
                    raw_output = await binding.handler.execute(
                        command.context,
                        command.arguments,
                    )

                validated_output = definition.output_schema.model_validate(raw_output)
                safe_output = _materialize_projection(
                    binding.safe_output_projector(validated_output)
                )
            except TimeoutError:
                result = self._failure_result(
                    command=command,
                    binding=binding,
                    fingerprint=fingerprint,
                    safe_input=safe_input,
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error=ToolTimeoutError(),
                    tool_call_id=tool_call_id,
                )
                observation.update(_failure_update(result=result))
                return result
            except ValidationError:
                result = self._failure_result(
                    command=command,
                    binding=binding,
                    fingerprint=fingerprint,
                    safe_input=safe_input,
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error=ToolOutputValidationError(),
                    tool_call_id=tool_call_id,
                )
                observation.update(_failure_update(result=result))
                return result
            except ToolError as error:
                result = self._failure_result(
                    command=command,
                    binding=binding,
                    fingerprint=fingerprint,
                    safe_input=safe_input,
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error=error,
                    tool_call_id=tool_call_id,
                )
                observation.update(_failure_update(result=result))
                return result
            except Exception:
                result = self._failure_result(
                    command=command,
                    binding=binding,
                    fingerprint=fingerprint,
                    safe_input=safe_input,
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    error=ToolUnexpectedError(),
                    tool_call_id=tool_call_id,
                )
                observation.update(_failure_update(result=result))
                return result

            finished_at, latency_ms = self._finish_timing(
                started_at=started_at,
                started_monotonic=started_monotonic,
            )
            audit = AgentToolCall.create_terminal(
                tool_call_id=tool_call_id,
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

            result = ToolExecutionResult(
                audit=audit,
                output=validated_output,
                retryable=False,
                failure_policy=definition.failure_policy,
            )
            observation.update(_success_update(result=result))
            return result
        finally:
            observation.close()

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
        tool_call_id: UUID | None = None,
    ) -> ToolExecutionResult:
        finished_at, latency_ms = self._finish_timing(
            started_at=started_at,
            started_monotonic=started_monotonic,
        )
        status = _status_for_error(error)
        definition = binding.definition
        audit = AgentToolCall.create_terminal(
            tool_call_id=tool_call_id or self._tool_call_id_factory(),
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


class _SafeToolObservation:
    """Isolate observability failures from tool-execution behavior."""

    def __init__(
        self,
        *,
        client: ObservabilityClient,
        attributes: ObservationAttributes,
    ) -> None:
        self._client = client
        self._attributes = attributes
        self._manager: AbstractContextManager[ObservationScope] | None = None
        self._scope: ObservationScope | None = None

    def start(self) -> None:
        try:
            self._manager = self._client.start_observation(self._attributes)
            self._scope = self._manager.__enter__()
        except Exception:
            self._manager = None
            self._scope = None

    def update(self, update: ObservationUpdate | None) -> None:
        if self._scope is None or update is None:
            return

        try:
            self._scope.update(update)
        except Exception:
            return

    def close(self) -> None:
        if self._manager is None:
            return

        try:
            self._manager.__exit__(None, None, None)
        except Exception:
            return
        finally:
            self._manager = None
            self._scope = None


def _start_attributes(
    *,
    command: ExecuteToolCommand,
    binding: ExecutableToolBinding,
    tool_call_id: UUID,
) -> ObservationAttributes:
    try:
        return ObservationAttributes(
            name=_OBSERVATION_NAME,
            observation_type=ObservationType.TOOL,
            metadata=_start_metadata(
                command=command,
                binding=binding,
                tool_call_id=tool_call_id,
            ),
            metadata_paths=_OBSERVATION_METADATA_PATHS,
            input_data=None,
            input_paths=frozenset(),
            output_paths=frozenset(),
        )
    except Exception:
        return ObservationAttributes(
            name=_OBSERVATION_NAME,
            observation_type=ObservationType.TOOL,
            input_data=None,
            input_paths=frozenset(),
            output_paths=frozenset(),
        )


def _start_metadata(
    *,
    command: ExecuteToolCommand,
    binding: ExecutableToolBinding,
    tool_call_id: UUID,
) -> dict[str, JsonValue]:
    definition = binding.definition

    return {
        "tool_name": definition.name,
        "tool_safety": definition.safety_level.value,
        "requires_approval": _requires_approval(definition.safety_level),
        "agent_run_id": str(command.context.agent_run_id),
        "agent_run_attempt_id": str(command.context.agent_run_attempt_id),
        "tool_call_id": str(tool_call_id),
        "workspace_id": str(command.context.workspace_id),
        "ticket_id": str(command.context.ticket_id),
    }


def _success_update(
    *,
    result: ToolExecutionResult,
) -> ObservationUpdate | None:
    try:
        return ObservationUpdate(
            status=ObservationStatus.OK,
            metadata={
                "status": ObservationStatus.OK.value,
                "tool_outcome": "succeeded",
                "latency_ms": result.audit.latency_ms,
            },
        )
    except Exception:
        return None


def _failure_update(
    *,
    result: ToolExecutionResult,
) -> ObservationUpdate | None:
    try:
        error_code = result.audit.error_code
        if error_code == ToolUnexpectedError.error_code:
            tool_outcome = "unexpected_failure"
        else:
            tool_outcome = "failed"

        metadata: dict[str, JsonValue] = {
            "status": ObservationStatus.ERROR.value,
            "tool_outcome": tool_outcome,
            "latency_ms": result.audit.latency_ms,
        }
        if error_code is not None:
            metadata["error_code"] = error_code

        return ObservationUpdate(
            status=ObservationStatus.ERROR,
            metadata=metadata,
            error_code=error_code,
        )
    except Exception:
        return None


def _requires_approval(safety_level: ToolSafetyLevel) -> bool:
    return safety_level is not ToolSafetyLevel.READ_ONLY


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
