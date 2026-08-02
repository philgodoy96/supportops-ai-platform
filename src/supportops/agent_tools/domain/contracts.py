"""Immutable contracts for application-owned controlled tools."""

from enum import StrEnum
from typing import Annotated, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

ToolName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    ),
]
ToolDescription = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=1000,
    ),
]
ToolVersion = Annotated[
    int,
    Field(
        strict=True,
        ge=1,
        le=1000,
    ),
]
ToolTimeoutSeconds = Annotated[
    float,
    Field(
        gt=0,
        le=60,
    ),
]


class ToolSafetyLevel(StrEnum):
    """Safety classification enforced by application policy."""

    READ_ONLY = "read_only"
    SENSITIVE_WRITE = "sensitive_write"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"


class ToolFailurePolicy(StrEnum):
    """AgentRun behavior for retryable tool dependency failures."""

    RETRY_AGENT_RUN = "retry_agent_run"
    FAIL_AGENT_RUN = "fail_agent_run"


class ToolAuditPolicy(StrEnum):
    """Audit projection behavior required from executable tools."""

    SAFE_PROJECTION = "safe_projection"


class StrictToolSchema(BaseModel):
    """Base model for strict tool input and output contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


class ToolReference(BaseModel):
    """Exact application-owned tool identity."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    name: ToolName
    version: ToolVersion


class ProviderToolDefinition(BaseModel):
    """Provider-independent model-visible function definition."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    name: ToolName
    version: ToolVersion
    description: ToolDescription
    input_schema: dict[str, JsonValue]
    strict: bool = True


class ToolDefinition(BaseModel):
    """Immutable application execution and provider projection metadata."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: ToolName
    version: ToolVersion
    description: ToolDescription
    input_schema: type[StrictToolSchema]
    output_schema: type[StrictToolSchema]
    safety_level: ToolSafetyLevel
    timeout_seconds: ToolTimeoutSeconds
    failure_policy: ToolFailurePolicy
    audit_policy: ToolAuditPolicy

    @model_validator(mode="after")
    def validate_schema_contracts(self) -> Self:
        """Require closed, fully declared JSON object schemas."""

        _validate_strict_tool_schema(
            schema_type=self.input_schema,
            schema_role="input",
            require_all_properties=True,
        )
        _validate_strict_tool_schema(
            schema_type=self.output_schema,
            schema_role="output",
            require_all_properties=True,
        )

        return self

    @property
    def reference(self) -> ToolReference:
        """Return the exact immutable tool identity."""

        return ToolReference(
            name=self.name,
            version=self.version,
        )

    def to_provider_definition(self) -> ProviderToolDefinition:
        """Project only model-visible metadata."""

        schema = cast(
            dict[str, JsonValue],
            self.input_schema.model_json_schema(),
        )

        return ProviderToolDefinition(
            name=self.name,
            version=self.version,
            description=self.description,
            input_schema=schema,
            strict=True,
        )


def _validate_strict_tool_schema(
    *,
    schema_type: type[StrictToolSchema],
    schema_role: str,
    require_all_properties: bool,
) -> None:
    schema = schema_type.model_json_schema()

    if schema.get("type") != "object":
        raise ValueError(f"Tool {schema_role} schema must be a JSON object.")

    _validate_object_nodes(
        node=schema,
        schema_role=schema_role,
        require_all_properties=require_all_properties,
        path="$",
    )


def _validate_object_nodes(
    *,
    node: object,
    schema_role: str,
    require_all_properties: bool,
    path: str,
) -> None:
    if isinstance(node, list):
        for index, item in enumerate(node):
            _validate_object_nodes(
                node=item,
                schema_role=schema_role,
                require_all_properties=require_all_properties,
                path=f"{path}[{index}]",
            )

        return

    if not isinstance(node, dict):
        return

    node_type = node.get("type")
    has_properties = "properties" in node

    if node_type == "object" or has_properties:
        if node.get("additionalProperties") is not False:
            raise ValueError(
                f"Tool {schema_role} schema object at {path} must reject additional properties."
            )

        properties = node.get("properties", {})

        if not isinstance(properties, dict):
            raise ValueError(f"Tool {schema_role} schema properties at {path} must be an object.")

        if require_all_properties:
            required = node.get("required", [])

            if not isinstance(required, list):
                raise ValueError(
                    f"Tool {schema_role} schema required fields at {path} must be an array."
                )

            property_names = set(properties)
            required_names = {value for value in required if isinstance(value, str)}

            if property_names != required_names:
                raise ValueError(
                    f"Tool {schema_role} schema object at {path} "
                    "must declare every property as required."
                )

    for key, value in node.items():
        _validate_object_nodes(
            node=value,
            schema_role=schema_role,
            require_all_properties=require_all_properties,
            path=f"{path}.{key}",
        )
