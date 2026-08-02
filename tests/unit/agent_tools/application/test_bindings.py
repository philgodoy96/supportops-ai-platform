"""Unit tests for immutable executable tool bindings."""

from typing import cast

import pytest

from supportops.agent_tools.application.bindings import (
    ExecutableToolBinding,
    ExecutableToolRegistry,
    ToolHandler,
)
from supportops.agent_tools.domain.contracts import (
    StrictToolSchema,
    ToolAuditPolicy,
    ToolDefinition,
    ToolFailurePolicy,
    ToolSafetyLevel,
)
from supportops.agent_tools.domain.errors import (
    ToolDuplicateDefinitionError,
    ToolNotFoundError,
    ToolSafetyViolationError,
    ToolVersionNotFoundError,
)


class ExampleInput(StrictToolSchema):
    """Synthetic strict tool input."""

    query: str


class ExampleOutput(StrictToolSchema):
    """Synthetic strict tool output."""

    result: str


class ExampleHandler:
    """Synthetic no-op tool handler."""

    async def execute(
        self,
        context: object,
        arguments: StrictToolSchema,
    ) -> object:
        del context, arguments

        return ExampleOutput(result="ok")


def _definition(
    *,
    name: str = "example_tool",
    version: int = 1,
    safety_level: ToolSafetyLevel = (ToolSafetyLevel.READ_ONLY),
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        version=version,
        description="Execute one synthetic read-only lookup.",
        input_schema=ExampleInput,
        output_schema=ExampleOutput,
        safety_level=safety_level,
        timeout_seconds=5,
        failure_policy=(ToolFailurePolicy.RETRY_AGENT_RUN),
        audit_policy=ToolAuditPolicy.SAFE_PROJECTION,
    )


def _binding(
    *,
    name: str = "example_tool",
    version: int = 1,
) -> ExecutableToolBinding:
    return ExecutableToolBinding(
        definition=_definition(
            name=name,
            version=version,
        ),
        handler=ExampleHandler(),
        safe_input_projector=lambda value: value.model_dump(mode="json"),
        safe_output_projector=lambda value: value.model_dump(mode="json"),
    )


def test_binding_requires_read_only_definition() -> None:
    with pytest.raises(
        ToolSafetyViolationError,
        match="not permitted",
    ):
        ExecutableToolBinding(
            definition=_definition(safety_level=(ToolSafetyLevel.SENSITIVE_WRITE)),
            handler=ExampleHandler(),
            safe_input_projector=lambda value: value.model_dump(mode="json"),
            safe_output_projector=lambda value: value.model_dump(mode="json"),
        )


def test_binding_requires_handler_execute_method() -> None:
    with pytest.raises(
        TypeError,
        match="execute method",
    ):
        ExecutableToolBinding(
            definition=_definition(),
            handler=cast(
                ToolHandler,
                object(),
            ),
            safe_input_projector=lambda value: value.model_dump(mode="json"),
            safe_output_projector=lambda value: value.model_dump(mode="json"),
        )


def test_registry_orders_bindings_deterministically() -> None:
    registry = ExecutableToolRegistry(
        (
            _binding(name="search_knowledge"),
            _binding(name="lookup_service_status"),
        )
    )

    assert [binding.definition.name for binding in registry.bindings] == [
        "lookup_service_status",
        "search_knowledge",
    ]
    assert [definition.name for definition in registry.definitions] == [
        "lookup_service_status",
        "search_knowledge",
    ]


def test_registry_requires_exact_identity() -> None:
    binding = _binding(
        name="search_knowledge",
        version=2,
    )
    registry = ExecutableToolRegistry((binding,))

    assert (
        registry.lookup(
            name="search_knowledge",
            version=2,
        )
        is binding
    )

    with pytest.raises(ToolNotFoundError):
        registry.lookup(
            name="unknown_tool",
            version=1,
        )

    with pytest.raises(ToolVersionNotFoundError):
        registry.lookup(
            name="search_knowledge",
            version=1,
        )


def test_registry_rejects_duplicate_exact_identity() -> None:
    binding = _binding()

    with pytest.raises(
        ToolDuplicateDefinitionError,
        match="duplicate definition",
    ):
        ExecutableToolRegistry(
            (
                binding,
                binding,
            )
        )


def test_registry_has_no_mutation_api() -> None:
    registry = ExecutableToolRegistry((_binding(),))

    assert not hasattr(registry, "register")
    assert not hasattr(registry, "replace")
    assert not hasattr(registry, "load")
