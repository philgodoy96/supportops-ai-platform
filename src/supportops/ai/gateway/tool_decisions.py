"""Provider-independent contracts for controlled LLM tool decisions."""

import json
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from types import MappingProxyType, TracebackType
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
from supportops.ai.pricing.catalog import DEFAULT_PRICING_CATALOG
from supportops.ai.pricing.estimation import estimate_llm_cost
from supportops.observability.contracts import (
    ObservabilityClient,
    ObservationScope,
)
from supportops.observability.models import (
    CostDetails,
    JsonValue,
    ObservationAttributes,
    ObservationStatus,
    ObservationType,
    ObservationUpdate,
    PricingStatus,
    UsageDetails,
)
from supportops.observability.noop import NoOpObservabilityClient

COMPLETE_SUPPORT_ANALYSIS_CONTROL_NAME = "complete_support_analysis"

_MAX_PROVIDER_ARGUMENTS_JSON_LENGTH = 20_000
_MAX_MODEL_VISIBLE_FUNCTIONS = 11

_GENERATION_OBSERVATION_NAME = "llm.tool_decision"
_UNHANDLED_BUSINESS_ERROR_CODE = "unhandled_business_error"
_DECISION_MODE_CONTROLLED = "controlled"
_DECISION_MODE_HUMAN_APPROVED = "human_approved"
_DECISION_KIND_EXECUTABLE_TOOL_CALL = "executable_tool_call"
_DECISION_KIND_TERMINAL_CONTROL = "terminal_control"

_REQUEST_METADATA_TO_OBSERVATION: Mapping[str, str] = {
    "supportops_prompt_id": "prompt_id",
    "supportops_prompt_version": "prompt_version",
    "supportops_prompt_content_hash": "prompt_hash",
    "supportops_schema_version": "schema_version",
    "supportops_agent_run_id": "agent_run_id",
    "supportops_agent_run_attempt_id": "agent_run_attempt_id",
    "supportops_workspace_id": "workspace_id",
    "supportops_correlation_id": "correlation_id",
}

_GENERATION_METADATA_PATHS = frozenset(
    {
        ("operation",),
        ("decision_mode",),
        ("invocation_sequence",),
        ("is_repair",),
        ("provider",),
        ("model",),
        ("prompt_id",),
        ("prompt_version",),
        ("prompt_hash",),
        ("schema_version",),
        ("agent_run_id",),
        ("agent_run_attempt_id",),
        ("workspace_id",),
        ("correlation_id",),
        ("tool_count",),
        ("sensitive_tool_count",),
        ("provider_request_id",),
        ("latency_ms",),
        ("pricing_catalog_version",),
        ("pricing_found",),
        ("decision_kind",),
        ("selected_tool_safety",),
    },
)

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
class LLMHumanApprovedToolDecisionRequest:
    """Application request for human-approved terminal or sensitive decisions."""

    operation: LLMOperation
    model: str
    instructions: str
    input: str
    sensitive_tools: tuple[ToolDefinition, ...]
    terminal_control: LLMTerminalControlDefinition
    timeout_seconds: float
    prompt_id: str
    prompt_version: int
    metadata: Mapping[str, str] = field(default_factory=dict)
    read_only_tools: tuple[ToolDefinition, ...] = ()

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
        _validate_required_text(
            self.prompt_id,
            field_name="prompt_id",
        )

        if self.prompt_version <= 0:
            raise ValueError("prompt_version must be positive.")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        normalized_sensitive_tools = tuple(
            sorted(
                self.sensitive_tools,
                key=lambda definition: (
                    definition.name,
                    definition.version,
                ),
            )
        )
        normalized_read_only_tools = tuple(
            sorted(
                self.read_only_tools,
                key=lambda definition: (
                    definition.name,
                    definition.version,
                ),
            )
        )

        if len(normalized_sensitive_tools) + len(normalized_read_only_tools) > 10:
            raise ValueError("Too many executable tools were provided.")

        tool_names: set[str] = set()

        for definition in normalized_sensitive_tools:
            if definition.safety_level is not ToolSafetyLevel.SENSITIVE_WRITE:
                raise ValueError(
                    "Human-approved sensitive tool requests may expose only sensitive_write tools."
                )

            if definition.name in tool_names:
                raise ValueError("Only one version of each tool may be model-visible.")

            if definition.name == self.terminal_control.name:
                raise ValueError("Terminal control name must not collide with an executable tool.")

            tool_names.add(definition.name)

        for definition in normalized_read_only_tools:
            if definition.safety_level is not ToolSafetyLevel.READ_ONLY:
                raise ValueError(
                    "Human-approved read-only tool requests may expose only read-only tools."
                )

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
            "sensitive_tools",
            normalized_sensitive_tools,
        )
        object.__setattr__(
            self,
            "read_only_tools",
            normalized_read_only_tools,
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
            *(definition.to_provider_definition() for definition in self.read_only_tools),
            *(definition.to_provider_definition() for definition in self.sensitive_tools),
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
        observability_client: ObservabilityClient | None = None,
    ) -> None:
        self._provider = provider
        self._clock = clock
        self._observability_client = (
            observability_client if observability_client is not None else NoOpObservabilityClient()
        )

    async def decide(
        self,
        request: LLMToolDecisionRequest,
    ) -> LLMToolDecisionGatewayResult:
        """Request and validate exactly one control decision."""

        provider_request = request.to_provider_request()
        invocation_sequence = 1
        is_repair = False
        started_at = self._clock()

        with _FailOpenGenerationObservation(
            client=self._observability_client,
            attributes=_build_generation_attributes(
                operation=request.operation,
                model=request.model,
                provider_name=self._provider.provider_name,
                metadata=request.metadata,
                decision_mode=_DECISION_MODE_CONTROLLED,
                invocation_sequence=invocation_sequence,
                is_repair=is_repair,
                tool_count=len(request.tools),
                sensitive_tool_count=0,
            ),
        ) as generation:
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
                generation.complete(
                    _build_error_update(
                        error_code=error.error_code.value,
                        provider_request_id=(error.provider_request_id),
                        latency_ms=latency_ms,
                    ),
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
                generation.complete(
                    _build_completion_update(
                        status=ObservationStatus.ERROR,
                        error_code=normalized_error.error_code.value,
                        provider=response.provider,
                        model=response.model,
                        provider_request_id=(response.provider_request_id),
                        usage=response.usage,
                        latency_ms=latency_ms,
                        decision_kind=None,
                        selected_tool_safety=None,
                    ),
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
            generation.complete(
                _build_completion_update(
                    status=ObservationStatus.OK,
                    error_code=None,
                    provider=response.provider,
                    model=response.model,
                    provider_request_id=(response.provider_request_id),
                    usage=response.usage,
                    latency_ms=latency_ms,
                    decision_kind=_decision_kind(decision),
                    selected_tool_safety=_selected_tool_safety(
                        decision=decision,
                        tools=request.tools,
                    ),
                ),
            )

            return LLMToolDecisionGatewayResult(
                decision=decision,
                invocations=(trace,),
                accepted_invocation_sequence=(invocation_sequence),
            )

    async def decide_human_approved(
        self,
        request: LLMHumanApprovedToolDecisionRequest,
    ) -> LLMToolDecisionGatewayResult:
        """Request and validate one human-approved control decision."""

        provider_request = request.to_provider_request()
        invocation_sequence = 1
        is_repair = False
        started_at = self._clock()

        with _FailOpenGenerationObservation(
            client=self._observability_client,
            attributes=_build_generation_attributes(
                operation=request.operation,
                model=request.model,
                provider_name=self._provider.provider_name,
                metadata=request.metadata,
                decision_mode=_DECISION_MODE_HUMAN_APPROVED,
                invocation_sequence=invocation_sequence,
                is_repair=is_repair,
                tool_count=(len(request.read_only_tools) + len(request.sensitive_tools)),
                sensitive_tool_count=len(request.sensitive_tools),
                prompt_id=request.prompt_id,
                prompt_version=request.prompt_version,
            ),
        ) as generation:
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
                generation.complete(
                    _build_error_update(
                        error_code=error.error_code.value,
                        provider_request_id=(error.provider_request_id),
                        latency_ms=latency_ms,
                    ),
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
                decision = _validate_human_approved_function_call(
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
                generation.complete(
                    _build_completion_update(
                        status=ObservationStatus.ERROR,
                        error_code=normalized_error.error_code.value,
                        provider=response.provider,
                        model=response.model,
                        provider_request_id=(response.provider_request_id),
                        usage=response.usage,
                        latency_ms=latency_ms,
                        decision_kind=None,
                        selected_tool_safety=None,
                    ),
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
            generation.complete(
                _build_completion_update(
                    status=ObservationStatus.OK,
                    error_code=None,
                    provider=response.provider,
                    model=response.model,
                    provider_request_id=(response.provider_request_id),
                    usage=response.usage,
                    latency_ms=latency_ms,
                    decision_kind=_decision_kind(decision),
                    selected_tool_safety=_selected_tool_safety(
                        decision=decision,
                        tools=(
                            *request.read_only_tools,
                            *request.sensitive_tools,
                        ),
                    ),
                ),
            )

            return LLMToolDecisionGatewayResult(
                decision=decision,
                invocations=(trace,),
                accepted_invocation_sequence=(invocation_sequence),
            )


class _FailOpenGenerationObservation:
    """Gateway-owned fail-open boundary around one generation observation."""

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
        self._completed = False

    def __enter__(self) -> "_FailOpenGenerationObservation":
        try:
            self._manager = self._client.start_observation(
                self._attributes,
            )
            self._scope = self._manager.__enter__()
        except Exception:
            self._manager = None
            self._scope = None

        return self

    def complete(self, update: ObservationUpdate) -> None:
        if self._scope is None or self._completed:
            return

        try:
            self._scope.update(update)
            self._completed = True
        except Exception:
            return

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exc, traceback

        if self._manager is None:
            return False

        try:
            if exc_type is not None and not self._completed:
                self.complete(
                    ObservationUpdate(
                        status=ObservationStatus.ERROR,
                        error_code=_UNHANDLED_BUSINESS_ERROR_CODE,
                    ),
                )

            self._manager.__exit__(None, None, None)
        except Exception:
            return False

        return False


def _build_generation_attributes(
    *,
    operation: LLMOperation,
    model: str,
    provider_name: str,
    metadata: Mapping[str, str],
    decision_mode: str,
    invocation_sequence: int,
    is_repair: bool,
    tool_count: int,
    sensitive_tool_count: int,
    prompt_id: str | None = None,
    prompt_version: int | None = None,
) -> ObservationAttributes:
    observation_metadata: dict[str, JsonValue] = {
        "operation": operation.value,
        "decision_mode": decision_mode,
        "invocation_sequence": invocation_sequence,
        "is_repair": is_repair,
        "provider": provider_name,
        "model": model,
        "tool_count": tool_count,
        "sensitive_tool_count": sensitive_tool_count,
    }

    if prompt_id is not None:
        observation_metadata["prompt_id"] = prompt_id

    if prompt_version is not None:
        observation_metadata["prompt_version"] = prompt_version

    for source_key, target_key in _REQUEST_METADATA_TO_OBSERVATION.items():
        if target_key in observation_metadata:
            continue

        value = metadata.get(source_key)
        if value is not None:
            observation_metadata[target_key] = value

    return ObservationAttributes(
        name=_GENERATION_OBSERVATION_NAME,
        observation_type=ObservationType.GENERATION,
        metadata=observation_metadata,
        metadata_paths=_GENERATION_METADATA_PATHS,
        input_paths=frozenset(),
        output_paths=frozenset(),
        provider=provider_name,
        model=model,
    )


def _build_error_update(
    *,
    error_code: str,
    provider_request_id: str | None,
    latency_ms: int,
) -> ObservationUpdate:
    metadata: dict[str, JsonValue] = {
        "latency_ms": latency_ms,
    }

    if provider_request_id is not None:
        metadata["provider_request_id"] = provider_request_id

    return ObservationUpdate(
        status=ObservationStatus.ERROR,
        metadata=metadata,
        error_code=error_code,
    )


def _build_completion_update(
    *,
    status: ObservationStatus,
    error_code: str | None,
    provider: str,
    model: str,
    provider_request_id: str | None,
    usage: LLMTokenUsage | None,
    latency_ms: int,
    decision_kind: str | None,
    selected_tool_safety: str | None,
) -> ObservationUpdate:
    metadata: dict[str, JsonValue] = {
        "latency_ms": latency_ms,
    }

    if provider_request_id is not None:
        metadata["provider_request_id"] = provider_request_id

    if decision_kind is not None:
        metadata["decision_kind"] = decision_kind

    if selected_tool_safety is not None:
        metadata["selected_tool_safety"] = selected_tool_safety

    usage_details, cost_details, pricing_metadata = _map_usage_and_cost(
        provider=provider,
        model=model,
        usage=usage,
    )
    metadata.update(pricing_metadata)

    return ObservationUpdate(
        status=status,
        metadata=metadata,
        usage=usage_details,
        cost=cost_details,
        error_code=error_code,
    )


def _map_usage_and_cost(
    *,
    provider: str,
    model: str,
    usage: LLMTokenUsage | None,
) -> tuple[
    UsageDetails | None,
    CostDetails | None,
    dict[str, JsonValue],
]:
    if usage is None:
        return (
            None,
            None,
            {},
        )

    usage_details = _to_usage_details(usage)

    if usage_details is None:
        return (
            None,
            None,
            {},
        )

    cost_estimate = estimate_llm_cost(
        provider=provider,
        model=model,
        usage=usage,
        catalog=DEFAULT_PRICING_CATALOG,
    )
    pricing_metadata: dict[str, JsonValue] = {
        "pricing_catalog_version": (cost_estimate.pricing_catalog_version),
        "pricing_found": cost_estimate.pricing_found,
    }

    if not cost_estimate.pricing_found:
        return (
            usage_details,
            CostDetails(
                pricing_status=PricingStatus.UNKNOWN,
                pricing_catalog_version=(cost_estimate.pricing_catalog_version),
            ),
            pricing_metadata,
        )

    return (
        usage_details,
        CostDetails(
            pricing_status=PricingStatus.KNOWN,
            input_cost=cost_estimate.estimated_input_cost_usd,
            cached_input_cost=(cost_estimate.estimated_cached_input_cost_usd),
            output_cost=cost_estimate.estimated_output_cost_usd,
            total_cost=None,
            pricing_catalog_version=(cost_estimate.pricing_catalog_version),
        ),
        pricing_metadata,
    )


def _to_usage_details(
    usage: LLMTokenUsage,
) -> UsageDetails | None:
    try:
        input_tokens = usage.input_tokens
        cached_input_tokens = usage.cached_input_tokens
        output_tokens = usage.output_tokens
        reasoning_tokens = usage.reasoning_tokens

        if (
            cached_input_tokens is not None
            and input_tokens is not None
            and cached_input_tokens > input_tokens
        ):
            return None

        if (
            reasoning_tokens is not None
            and output_tokens is not None
            and reasoning_tokens > output_tokens
        ):
            return None

        mapped_input_tokens = input_tokens

        if cached_input_tokens is not None and input_tokens is not None:
            mapped_input_tokens = input_tokens - cached_input_tokens

        mapped_output_tokens = output_tokens

        if reasoning_tokens is not None and output_tokens is not None:
            mapped_output_tokens = output_tokens - reasoning_tokens

        component_values = tuple(
            value
            for value in (
                mapped_input_tokens,
                cached_input_tokens,
                mapped_output_tokens,
                reasoning_tokens,
            )
            if value is not None
        )

        return UsageDetails(
            input_tokens=mapped_input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=mapped_output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=None if component_values else usage.total_tokens,
        )
    except Exception:
        return None


def _decision_kind(
    decision: LLMExecutableToolCallDecision | LLMTerminalControlDecision,
) -> str:
    if isinstance(decision, LLMTerminalControlDecision):
        return _DECISION_KIND_TERMINAL_CONTROL

    return _DECISION_KIND_EXECUTABLE_TOOL_CALL


def _selected_tool_safety(
    *,
    decision: LLMExecutableToolCallDecision | LLMTerminalControlDecision,
    tools: tuple[ToolDefinition, ...],
) -> str | None:
    if not isinstance(decision, LLMExecutableToolCallDecision):
        return None

    definition = next(
        (tool for tool in tools if tool.name == decision.tool_name),
        None,
    )

    if definition is None:
        return None

    return definition.safety_level.value


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


def _validate_human_approved_function_call(
    *,
    request: LLMHumanApprovedToolDecisionRequest,
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

    sensitive_definition = next(
        (tool for tool in request.sensitive_tools if tool.name == response.function_name),
        None,
    )

    if sensitive_definition is not None:
        validated_arguments = sensitive_definition.input_schema.model_validate(arguments)

        return LLMExecutableToolCallDecision(
            provider_tool_call_id=(response.provider_tool_call_id),
            tool_name=sensitive_definition.name,
            tool_version=sensitive_definition.version,
            arguments=validated_arguments,
        )

    read_only_definition = next(
        (tool for tool in request.read_only_tools if tool.name == response.function_name),
        None,
    )

    if read_only_definition is None:
        raise ValueError("Provider selected an unknown function.")

    raise ValueError(
        "Human-approved read-only decisions are unavailable in the current workflow surface."
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
