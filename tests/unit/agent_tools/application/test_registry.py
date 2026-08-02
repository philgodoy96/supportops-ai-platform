"""Unit tests for the immutable controlled tool registry."""

from typing import Annotated
from uuid import UUID

import pytest
from pydantic import Field

from supportops.agent_tools.application.registry import (
    ToolRegistry,
)
from supportops.agent_tools.domain.contracts import (
    StrictToolSchema,
    ToolAuditPolicy,
    ToolDefinition,
    ToolFailurePolicy,
    ToolReference,
    ToolSafetyLevel,
)
from supportops.agent_tools.domain.errors import (
    ToolDuplicateDefinitionError,
    ToolNotFoundError,
    ToolProviderSelectionError,
    ToolSafetyViolationError,
    ToolVersionNotFoundError,
)


class SearchInput(StrictToolSchema):
    """Synthetic strict knowledge-search input."""

    top_k: Annotated[
        int,
        Field(
            strict=True,
            ge=1,
            le=10,
        ),
    ]
    document_ids: tuple[UUID, ...] | None


class SearchOutput(StrictToolSchema):
    """Synthetic strict knowledge-search output."""

    result_count: Annotated[
        int,
        Field(
            strict=True,
            ge=0,
        ),
    ]


class StatusInput(StrictToolSchema):
    """Synthetic strict service-status input."""

    service_name: str


class StatusOutput(StrictToolSchema):
    """Synthetic strict service-status output."""

    status: str


def create_search_definition(
    *,
    version: int = 1,
    safety_level: ToolSafetyLevel = (ToolSafetyLevel.READ_ONLY),
) -> ToolDefinition:
    """Create one exact synthetic knowledge tool."""

    return ToolDefinition(
        name="search_knowledge",
        version=version,
        description="Search active workspace knowledge.",
        input_schema=SearchInput,
        output_schema=SearchOutput,
        safety_level=safety_level,
        timeout_seconds=15,
        failure_policy=(ToolFailurePolicy.RETRY_AGENT_RUN),
        audit_policy=ToolAuditPolicy.SAFE_PROJECTION,
    )


def create_status_definition() -> ToolDefinition:
    """Create one exact synthetic status tool."""

    return ToolDefinition(
        name="lookup_service_status",
        version=1,
        description="Look up deterministic service status.",
        input_schema=StatusInput,
        output_schema=StatusOutput,
        safety_level=ToolSafetyLevel.READ_ONLY,
        timeout_seconds=5,
        failure_policy=(ToolFailurePolicy.RETRY_AGENT_RUN),
        audit_policy=ToolAuditPolicy.SAFE_PROJECTION,
    )


def test_registry_orders_definitions_deterministically() -> None:
    registry = ToolRegistry(
        [
            create_search_definition(),
            create_status_definition(),
        ]
    )

    assert [
        (
            definition.name,
            definition.version,
        )
        for definition in registry.definitions
    ] == [
        (
            "lookup_service_status",
            1,
        ),
        (
            "search_knowledge",
            1,
        ),
    ]


def test_registry_rejects_duplicate_name_and_version() -> None:
    definition = create_search_definition()

    with pytest.raises(
        ToolDuplicateDefinitionError,
        match="duplicate definition",
    ):
        ToolRegistry(
            [
                definition,
                definition,
            ]
        )


def test_registry_allows_distinct_historical_versions() -> None:
    registry = ToolRegistry(
        [
            create_search_definition(version=1),
            create_search_definition(version=2),
        ]
    )

    assert (
        registry.lookup(
            name="search_knowledge",
            version=1,
        ).version
        == 1
    )
    assert (
        registry.lookup(
            name="search_knowledge",
            version=2,
        ).version
        == 2
    )


def test_registry_requires_exact_name() -> None:
    registry = ToolRegistry(
        [
            create_search_definition(),
        ]
    )

    with pytest.raises(
        ToolNotFoundError,
        match="not registered",
    ):
        registry.lookup(
            name="invented_tool",
            version=1,
        )


def test_registry_requires_exact_version() -> None:
    registry = ToolRegistry(
        [
            create_search_definition(),
        ]
    )

    with pytest.raises(
        ToolVersionNotFoundError,
        match="version is not registered",
    ):
        registry.lookup(
            name="search_knowledge",
            version=2,
        )


def test_registry_definitions_are_immutable_tuple() -> None:
    registry = ToolRegistry(
        [
            create_search_definition(),
        ]
    )

    assert isinstance(
        registry.definitions,
        tuple,
    )
    assert not hasattr(
        registry,
        "register",
    )
    assert not hasattr(
        registry,
        "load",
    )


def test_provider_projection_requires_explicit_selection() -> None:
    registry = ToolRegistry(
        [
            create_search_definition(),
            create_status_definition(),
        ]
    )

    provider_definitions = registry.project_provider_definitions(
        [
            ToolReference(
                name="search_knowledge",
                version=1,
            ),
        ]
    )

    assert len(provider_definitions) == 1
    assert provider_definitions[0].name == "search_knowledge"
    assert provider_definitions[0].version == 1
    assert provider_definitions[0].strict is True


def test_provider_projection_is_deterministic() -> None:
    registry = ToolRegistry(
        [
            create_search_definition(),
            create_status_definition(),
        ]
    )
    references = [
        ToolReference(
            name="search_knowledge",
            version=1,
        ),
        ToolReference(
            name="lookup_service_status",
            version=1,
        ),
    ]

    provider_definitions = registry.project_provider_definitions(reversed(references))

    assert [definition.name for definition in provider_definitions] == [
        "lookup_service_status",
        "search_knowledge",
    ]


def test_provider_projection_rejects_duplicate_reference() -> None:
    registry = ToolRegistry(
        [
            create_search_definition(),
        ]
    )
    reference = ToolReference(
        name="search_knowledge",
        version=1,
    )

    with pytest.raises(
        ToolProviderSelectionError,
        match="selected provider",
    ):
        registry.project_provider_definitions(
            [
                reference,
                reference,
            ]
        )


def test_provider_projection_rejects_multiple_versions() -> None:
    registry = ToolRegistry(
        [
            create_search_definition(version=1),
            create_search_definition(version=2),
        ]
    )

    with pytest.raises(
        ToolProviderSelectionError,
        match="selected provider",
    ):
        registry.project_provider_definitions(
            [
                ToolReference(
                    name="search_knowledge",
                    version=1,
                ),
                ToolReference(
                    name="search_knowledge",
                    version=2,
                ),
            ]
        )


def test_provider_projection_rejects_non_read_only_tool() -> None:
    sensitive_definition = create_search_definition(safety_level=ToolSafetyLevel.SENSITIVE_WRITE)
    registry = ToolRegistry(
        [
            sensitive_definition,
        ]
    )

    with pytest.raises(
        ToolSafetyViolationError,
        match="not permitted",
    ):
        registry.project_provider_definitions(
            [
                sensitive_definition.reference,
            ]
        )


def test_provider_projection_does_not_expose_execution_policy() -> None:
    registry = ToolRegistry(
        [
            create_search_definition(),
        ]
    )

    provider_definition = registry.project_provider_definitions(
        [
            ToolReference(
                name="search_knowledge",
                version=1,
            )
        ]
    )[0]
    payload = provider_definition.model_dump(mode="json")

    assert "safety_level" not in payload
    assert "timeout_seconds" not in payload
    assert "failure_policy" not in payload
    assert "audit_policy" not in payload
    assert "executor" not in payload
