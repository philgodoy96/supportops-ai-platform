"""Unit tests for immutable controlled tool contracts."""

from typing import Annotated, Any, cast
from uuid import UUID

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from supportops.agent_tools.domain.contracts import (
    StrictToolSchema,
    ToolAuditPolicy,
    ToolDefinition,
    ToolFailurePolicy,
    ToolSafetyLevel,
)


class DocumentFilter(StrictToolSchema):
    """Strict nested input used by contract tests."""

    document_id: UUID
    include_archived: bool


class ExampleToolInput(StrictToolSchema):
    """Strict complete input used by contract tests."""

    query: str
    top_k: Annotated[
        int,
        Field(
            strict=True,
            ge=1,
            le=10,
        ),
    ]
    document_filter: DocumentFilter | None


class ExampleToolOutput(StrictToolSchema):
    """Strict complete output used by contract tests."""

    result_count: Annotated[
        int,
        Field(
            strict=True,
            ge=0,
        ),
    ]
    chunk_ids: tuple[UUID, ...]


class InputWithDefault(StrictToolSchema):
    """Input that violates complete provider declaration."""

    top_k: int = 5


class InputWithOpenNestedObject(StrictToolSchema):
    """Input containing an uncontrolled object schema."""

    metadata: dict[str, str]


class NonStrictInput(BaseModel):
    """Schema that does not reject unknown fields."""

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )

    query: str


def create_definition(
    *,
    safety_level: ToolSafetyLevel = (ToolSafetyLevel.READ_ONLY),
) -> ToolDefinition:
    """Create a valid immutable synthetic tool definition."""

    return ToolDefinition(
        name="search_example",
        version=1,
        description="Search deterministic example evidence.",
        input_schema=ExampleToolInput,
        output_schema=ExampleToolOutput,
        safety_level=safety_level,
        timeout_seconds=5,
        failure_policy=(ToolFailurePolicy.RETRY_AGENT_RUN),
        audit_policy=ToolAuditPolicy.SAFE_PROJECTION,
    )


def test_strict_tool_schema_rejects_additional_fields() -> None:
    with pytest.raises(ValidationError):
        ExampleToolInput.model_validate(
            {
                "query": "reset procedure",
                "top_k": 5,
                "document_filter": None,
                "workspace_id": ("11111111-1111-4111-8111-111111111111"),
            }
        )


def test_tool_definition_is_immutable() -> None:
    definition = create_definition()

    with pytest.raises(ValidationError):
        definition.version = 2  # type: ignore[misc]


def test_tool_definition_exposes_exact_reference() -> None:
    definition = create_definition()

    assert definition.reference.name == "search_example"
    assert definition.reference.version == 1


def test_provider_projection_contains_only_model_metadata() -> None:
    definition = create_definition()

    provider_definition = definition.to_provider_definition()
    required = cast(
        list[str],
        provider_definition.input_schema["required"],
    )

    assert provider_definition.name == "search_example"
    assert provider_definition.version == 1
    assert provider_definition.strict is True
    assert provider_definition.input_schema["additionalProperties"] is False
    assert set(required) == {
        "query",
        "top_k",
        "document_filter",
    }

    serialized = provider_definition.model_dump(mode="json")

    assert "output_schema" not in serialized
    assert "safety_level" not in serialized
    assert "timeout_seconds" not in serialized
    assert "failure_policy" not in serialized
    assert "audit_policy" not in serialized
    assert "executor" not in serialized


def test_provider_projection_keeps_nested_objects_closed() -> None:
    definition = create_definition()

    schema = definition.to_provider_definition().input_schema
    nested_defs = cast(dict[str, Any], schema["$defs"])
    nested_schema = cast(dict[str, Any], nested_defs["DocumentFilter"])
    required = cast(list[str], nested_schema["required"])

    assert nested_schema["additionalProperties"] is False
    assert set(required) == {
        "document_id",
        "include_archived",
    }


def test_definition_rejects_input_with_defaulted_property() -> None:
    with pytest.raises(
        ValidationError,
        match="must declare every property as required",
    ):
        ToolDefinition(
            name="invalid_default",
            version=1,
            description="Invalid defaulted input.",
            input_schema=InputWithDefault,
            output_schema=ExampleToolOutput,
            safety_level=ToolSafetyLevel.READ_ONLY,
            timeout_seconds=5,
            failure_policy=(ToolFailurePolicy.RETRY_AGENT_RUN),
            audit_policy=(ToolAuditPolicy.SAFE_PROJECTION),
        )


def test_definition_rejects_open_nested_object() -> None:
    with pytest.raises(
        ValidationError,
        match="must reject additional properties",
    ):
        ToolDefinition(
            name="invalid_object",
            version=1,
            description="Invalid open nested object.",
            input_schema=InputWithOpenNestedObject,
            output_schema=ExampleToolOutput,
            safety_level=ToolSafetyLevel.READ_ONLY,
            timeout_seconds=5,
            failure_policy=(ToolFailurePolicy.RETRY_AGENT_RUN),
            audit_policy=(ToolAuditPolicy.SAFE_PROJECTION),
        )


def test_definition_rejects_non_strict_schema_type() -> None:
    with pytest.raises(ValidationError):
        ToolDefinition(
            name="invalid_schema",
            version=1,
            description="Invalid schema base type.",
            input_schema=cast(
                Any,
                NonStrictInput,
            ),
            output_schema=ExampleToolOutput,
            safety_level=ToolSafetyLevel.READ_ONLY,
            timeout_seconds=5,
            failure_policy=(ToolFailurePolicy.RETRY_AGENT_RUN),
            audit_policy=(ToolAuditPolicy.SAFE_PROJECTION),
        )


@pytest.mark.parametrize(
    "invalid_name",
    [
        "",
        "SearchExample",
        "search-example",
        "1_search",
        "search example",
    ],
)
def test_definition_rejects_invalid_name(
    invalid_name: str,
) -> None:
    with pytest.raises(ValidationError):
        ToolDefinition(
            name=invalid_name,
            version=1,
            description="Invalid name example.",
            input_schema=ExampleToolInput,
            output_schema=ExampleToolOutput,
            safety_level=ToolSafetyLevel.READ_ONLY,
            timeout_seconds=5,
            failure_policy=(ToolFailurePolicy.RETRY_AGENT_RUN),
            audit_policy=(ToolAuditPolicy.SAFE_PROJECTION),
        )


@pytest.mark.parametrize(
    "invalid_timeout",
    [
        0,
        -1,
        61,
    ],
)
def test_definition_rejects_invalid_timeout(
    invalid_timeout: float,
) -> None:
    with pytest.raises(ValidationError):
        ToolDefinition(
            name="invalid_timeout",
            version=1,
            description="Invalid timeout example.",
            input_schema=ExampleToolInput,
            output_schema=ExampleToolOutput,
            safety_level=ToolSafetyLevel.READ_ONLY,
            timeout_seconds=invalid_timeout,
            failure_policy=(ToolFailurePolicy.RETRY_AGENT_RUN),
            audit_policy=(ToolAuditPolicy.SAFE_PROJECTION),
        )
