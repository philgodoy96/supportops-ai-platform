"""Provider-independent contracts for controlled LLM tool decisions."""

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol

from pydantic import ValidationError

from supportops.agent_tools.domain.contracts import (
    ProviderToolDefinition,
    StrictToolSchema,
    ToolDefinition,
    ToolSafetyLevel,
)
from supportops.ai.gateway.contracts import (
    LLMOperation,
    LLMTokenUsage,
)
from supportops.ai.gateway.errors import (
    LLMError,
    LLMToolDecisionValidationError,
)
from supportops.ai.gateway.results import (
    LLMGatewayFailure,
    LLMInvocationStatus,
    LLMInvocationTrace,
)

COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME = "complete_support_analysis"

_MAX_PROVIDER_ARGUMENTS_JSON_LENGTH = 20_000
_MAX_MODEL_VISIBLE_FUNCTIONS = 11

type Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class LLMTerminalControlDefinition:
    """Non-executable function controlling graph termination."""

    name: str
    version: int
    description: str
    input_schema: type[StrictToolSchema]

    def __post_init__(self) -> None:
        _validate_required_text(
            self.name,
            field_name="name",
        )
        _validate_required_text(
            self.description,
            field_name="description",
        )

        if self.version <= 0:
            raise ValueError("version must be positive.")

        if not isinstance(self.input_schema, type) or not issubclass(
            self.input_schema,
            StrictToolSchema,
        ):
            raise TypeError("input_schema must inherit StrictToolSchema.")

        _validate_closed_function_schema(self.input_schema.model_json_schema())

        self.to_provider_definition()

    def to_provider_definition(
        self,
    ) -> ProviderToolDefinition:
        """Project the terminal control into model-visible metadata."""

        return ProviderToolDefinition(
            name=self.name,
            version=self.version,
            description=self.description,
            input_schema=self.input_schema.model_json_schema(),
            strict=True,
        )


@dataclass(frozen=True, slots=True)
class LLMProviderToolDecisionRequest:
    """Provider-only request containing strict function definitions."""

    operation: LLMOperation
    model: str
    instructions: str
    input: str
    functions: tuple[ProviderToolDefinition, ...]
    timeout_seconds: float
    metadata: Mapping[str, str] = field(default_factory=dict)
    tool_choice: Literal["required"] = "required"
    parallel_tool_calls: Literal[False] = False

    def __post_init__(self) -> None:
        if self.operation is not LLMOperation.SUPPORT_ACTION_DECISION:
            raise ValueError(
                "Tool decision requests require the support_action_decision operation."
            )

        _validate_required_text(
            self.model,
            field_name="model",
        )
        _validate_required_text(
            self.instructions,
            field_name="instructions",
        )
        _validate_required_text(
            self.input,
            field_name="input",
        )

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        normalized_functions = tuple(self.functions)

        if not normalized_functions:
            raise ValueError("At least one function definition is required.")

        if len(normalized_functions) > _MAX_MODEL_VISIBLE_FUNCTIONS:
            raise ValueError("Too many model-visible function definitions.")

        function_names: set[str] = set()

        for function in normalized_functions:
            if not function.strict:
                raise ValueError("All provider function definitions must be strict.")

            if function.name in function_names:
                raise ValueError("Provider function names must be unique.")

            function_names.add(function.name)

        normalized_metadata = dict(self.metadata)

        for key, value in normalized_metadata.items():
            _validate_required_text(
                key,
                field_name="metadata key",
            )
            _validate_required_text(
                value,
                field_name=f"metadata[{key!r}]",
            )

        object.__setattr__(
            self,
            "functions",
            normalized_functions,
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(normalized_metadata),
        )


@dataclass(frozen=True, slots=True)
class LLMToolDecisionRequest:
    """Application request with complete tool policy metadata."""

    operation: LLMOperation
    model: str
    instructions: str
    input: str
    tools: tuple[ToolDefinition, ...]
    terminal_control: LLMTerminalControlDefinition
    timeout_seconds: float
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.operation is not LLMOperation.SUPPORT_ACTION_DECISION:
            raise ValueError(
                "Tool decision requests require the support_action_decision operation."
            )

        _validate_required_text(
            self.model,
            field_name="model",
        )
        _validate_required_text(
            self.instructions,
            field_name="instructions",
        )
        _validate_required_text(
            self.input,
            field_name="input",
        )

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        normalized_tools = tuple(
            sorted(
                self.tools,
                key=lambda definition: (
                    definition.name,
                    definition.version,
                ),
            )
        )

        if len(normalized_tools) > 10:
            raise ValueError("Too many executable tools were provided.")

        tool_names: set[str] = set()

        for definition in normalized_tools:
            if definition.safety_level is not ToolSafetyLevel.READ_ONLY:
                raise ValueError("Tool decision requests may expose only read-only tools.")

            if definition.name in tool_names:
                raise ValueError("Only one version of each tool may be model-visible.")

            if definition.name == self.terminal_control.name:
                raise ValueError("Terminal control name must not collide with an executable tool.")

            tool_names.add(definition.name)

        normalized_metadata = dict(self.metadata)

        for key, value in normalized_metadata.items():
            _validate_required_text(
                key,
                field_name="metadata key",
            )
            _validate_required_text(
                value,
                field_name=f"metadata[{key!r}]",
            )

        object.__setattr__(
            self,
            "tools",
            normalized_tools,
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(normalized_metadata),
        )

    def to_provider_request(
        self,
    ) -> LLMProviderToolDecisionRequest:
        """Project application policy into provider-only metadata."""

        functions = (
            *(definition.to_provider_definition() for definition in self.tools),
            self.terminal_control.to_provider_definition(),
        )

        return LLMProviderToolDecisionRequest(
            operation=self.operation,
            model=self.model,
            instructions=self.instructions,
            input=self.input,
            functions=functions,
            timeout_seconds=self.timeout_seconds,
            metadata=self.metadata,
            tool_choice="required",
            parallel_tool_calls=False,
        )


@dataclass(frozen=True, slots=True)
class LLMProviderFunctionCallResponse:
    """One provider-selected function without SDK-specific types."""

    provider_tool_call_id: str
    function_name: str
    arguments_json: str
    provider: str
    model: str
    provider_request_id: str | None = None
    usage: LLMTokenUsage | None = None
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        _validate_required_text(
            self.provider_tool_call_id,
            field_name="provider_tool_call_id",
        )
        _validate_required_text(
            self.function_name,
            field_name="function_name",
        )
        _validate_required_text(
            self.arguments_json,
            field_name="arguments_json",
        )
        _validate_required_text(
            self.provider,
            field_name="provider",
        )
        _validate_required_text(
            self.model,
            field_name="model",
        )
        _validate_optional_text(
            self.provider_request_id,
            field_name="provider_request_id",
        )
        _validate_optional_text(
            self.finish_reason,
            field_name="finish_reason",
        )

        if len(self.arguments_json) > _MAX_PROVIDER_ARGUMENTS_JSON_LENGTH:
            raise ValueError("arguments_json exceeds the supported size.")


class LLMToolDecisionProvider(Protocol):
    """Provider adapter for one native function-call decision."""

    @property
    def provider_name(self) -> str:
        """Return the stable provider identifier."""
        ...

    async def decide(
        self,
        request: LLMProviderToolDecisionRequest,
    ) -> LLMProviderFunctionCallResponse:
        """Return exactly one provider-selected function call."""
        ...

    async def close(self) -> None:
        """Release provider-owned process resources."""
        ...


@dataclass(frozen=True, slots=True)
class LLMExecutableToolCallDecision:
    """Validated decision to execute one registered read-only tool."""

    provider_tool_call_id: str
    tool_name: str
    tool_version: int
    arguments: StrictToolSchema

    def __post_init__(self) -> None:
        _validate_required_text(
            self.provider_tool_call_id,
            field_name="provider_tool_call_id",
        )
        _validate_required_text(
            self.tool_name,
            field_name="tool_name",
        )

        if self.tool_version <= 0:
            raise ValueError("tool_version must be positive.")


@dataclass(frozen=True, slots=True)
class LLMTerminalControlDecision:
    """Validated non-executable graph control decision."""

    provider_tool_call_id: str
    control_name: str
    control_version: int
    output: StrictToolSchema

    def __post_init__(self) -> None:
        _validate_required_text(
            self.provider_tool_call_id,
            field_name="provider_tool_call_id",
        )
        _validate_required_text(
            self.control_name,
            field_name="control_name",
        )

        if self.control_version <= 0:
            raise ValueError("control_version must be positive.")


@dataclass(frozen=True, slots=True)
class LLMToolDecisionGatewayResult:
    """Accepted tool decision and provider invocation provenance."""

    decision: LLMExecutableToolCallDecision | LLMTerminalControlDecision
    invocations: tuple[LLMInvocationTrace, ...]
    accepted_invocation_sequence: int

    def __post_init__(self) -> None:
        if not self.invocations:
            raise ValueError("A successful decision requires invocations.")

        accepted_invocation = next(
            (
                invocation
                for invocation in self.invocations
                if invocation.invocation_sequence == self.accepted_invocation_sequence
            ),
            None,
        )

        if accepted_invocation is None:
            raise ValueError("accepted_invocation_sequence must reference an invocation.")

        if accepted_invocation.status is not LLMInvocationStatus.SUCCEEDED:
            raise ValueError("The accepted invocation must have succeeded.")


class LLMToolDecisionGateway:
    """Own validation for one provider-native tool decision."""

    def __init__(
        self,
        *,
        provider: LLMToolDecisionProvider,
        clock: Clock = time.perf_counter,
    ) -> None:
        self._provider = provider
        self._clock = clock

    async def decide(
        self,
        request: LLMToolDecisionRequest,
    ) -> LLMToolDecisionGatewayResult:
        """Request and validate exactly one control decision."""

        provider_request = request.to_provider_request()
        invocation_sequence = 1
        started_at = self._clock()

        try:
            response = await self._provider.decide(provider_request)
        except LLMError as error:
            latency_ms = _elapsed_milliseconds(
                started_at,
                self._clock(),
            )
            trace = _trace_from_error(
                request=provider_request,
                provider_name=self._provider.provider_name,
                invocation_sequence=invocation_sequence,
                latency_ms=latency_ms,
                error=error,
            )

            raise LLMGatewayFailure(
                error=error,
                invocations=(trace,),
            ) from error

        latency_ms = _elapsed_milliseconds(
            started_at,
            self._clock(),
        )

        _validate_response_provenance(
            request=provider_request,
            expected_provider=self._provider.provider_name,
            response=response,
        )

        try:
            decision = _validate_function_call(
                request=request,
                response=response,
            )
        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
            ValidationError,
        ) as error:
            normalized_error = LLMToolDecisionValidationError(
                provider_request_id=(response.provider_request_id),
            )
            trace = LLMInvocationTrace(
                invocation_sequence=invocation_sequence,
                status=(LLMInvocationStatus.VALIDATION_FAILED),
                provider=response.provider,
                model=response.model,
                provider_request_id=(response.provider_request_id),
                usage=response.usage,
                latency_ms=latency_ms,
                error_code=normalized_error.error_code,
            )

            raise LLMGatewayFailure(
                error=normalized_error,
                invocations=(trace,),
            ) from error

        trace = LLMInvocationTrace(
            invocation_sequence=invocation_sequence,
            status=LLMInvocationStatus.SUCCEEDED,
            provider=response.provider,
            model=response.model,
            provider_request_id=(response.provider_request_id),
            usage=response.usage,
            latency_ms=latency_ms,
            error_code=None,
        )

        return LLMToolDecisionGatewayResult(
            decision=decision,
            invocations=(trace,),
            accepted_invocation_sequence=(invocation_sequence),
        )


def _validate_function_call(
    *,
    request: LLMToolDecisionRequest,
    response: LLMProviderFunctionCallResponse,
) -> LLMExecutableToolCallDecision | LLMTerminalControlDecision:
    arguments = json.loads(response.arguments_json)

    if not isinstance(arguments, dict):
        raise ValueError("Function arguments must be a JSON object.")

    if response.function_name == request.terminal_control.name:
        validated_output = request.terminal_control.input_schema.model_validate(arguments)

        return LLMTerminalControlDecision(
            provider_tool_call_id=(response.provider_tool_call_id),
            control_name=request.terminal_control.name,
            control_version=(request.terminal_control.version),
            output=validated_output,
        )

    definition = next(
        (tool for tool in request.tools if tool.name == response.function_name),
        None,
    )

    if definition is None:
        raise ValueError("Provider selected an unknown function.")

    validated_arguments = definition.input_schema.model_validate(arguments)

    return LLMExecutableToolCallDecision(
        provider_tool_call_id=(response.provider_tool_call_id),
        tool_name=definition.name,
        tool_version=definition.version,
        arguments=validated_arguments,
    )


def _trace_from_error(
    *,
    request: LLMProviderToolDecisionRequest,
    provider_name: str,
    invocation_sequence: int,
    latency_ms: int,
    error: LLMError,
) -> LLMInvocationTrace:
    if error.error_code.value == "llm_timeout":
        status = LLMInvocationStatus.TIMED_OUT
    elif error.error_code.value == "llm_refusal":
        status = LLMInvocationStatus.REFUSED
    elif error.error_code.value == "llm_incomplete_response":
        status = LLMInvocationStatus.INCOMPLETE
    elif error.error_code.value in {
        "llm_output_validation_failed",
        "llm_tool_decision_validation_failed",
    }:
        status = LLMInvocationStatus.VALIDATION_FAILED
    else:
        status = LLMInvocationStatus.PROVIDER_FAILED

    return LLMInvocationTrace(
        invocation_sequence=invocation_sequence,
        status=status,
        provider=provider_name,
        model=request.model,
        provider_request_id=(error.provider_request_id),
        usage=None,
        latency_ms=latency_ms,
        error_code=error.error_code,
    )


def _validate_response_provenance(
    *,
    request: LLMProviderToolDecisionRequest,
    expected_provider: str,
    response: LLMProviderFunctionCallResponse,
) -> None:
    if response.provider != expected_provider:
        raise RuntimeError(
            "LLM provider response provenance does not match the configured provider."
        )

    if response.model != request.model:
        raise RuntimeError("LLM provider response model does not match the requested model.")


def _validate_closed_function_schema(
    schema: object,
) -> None:
    if isinstance(schema, list):
        for item in schema:
            _validate_closed_function_schema(item)

        return

    if not isinstance(schema, dict):
        return

    if schema.get("type") == "object" or "properties" in schema:
        if schema.get("additionalProperties") is not False:
            raise ValueError("Terminal control schema must reject additional properties.")

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        if not isinstance(properties, dict):
            raise ValueError("Terminal control properties must be a JSON object.")

        if not isinstance(required, list):
            raise ValueError("Terminal control required fields must be an array.")

        required_names = {value for value in required if isinstance(value, str)}

        if set(properties) != required_names:
            raise ValueError("Terminal control must declare every property as required.")

    for value in schema.values():
        _validate_closed_function_schema(value)


def _elapsed_milliseconds(
    started_at: float,
    completed_at: float,
) -> int:
    elapsed_seconds = completed_at - started_at

    if elapsed_seconds < 0:
        raise RuntimeError("The LLM Gateway clock moved backwards.")

    return round(elapsed_seconds * 1_000)


def _validate_required_text(
    value: str,
    *,
    field_name: str,
) -> None:
    if not value:
        raise ValueError(f"{field_name} is required.")

    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace.")


def _validate_optional_text(
    value: str | None,
    *,
    field_name: str,
) -> None:
    if value is not None:
        _validate_required_text(
            value,
            field_name=field_name,
        )
