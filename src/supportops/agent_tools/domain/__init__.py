"""Domain contracts for controlled tool definitions and failures."""

from supportops.agent_tools.domain.contracts import (
    ProviderToolDefinition,
    StrictToolSchema,
    ToolAuditPolicy,
    ToolDefinition,
    ToolFailurePolicy,
    ToolReference,
    ToolSafetyLevel,
)
from supportops.agent_tools.domain.errors import (
    ToolCallLimitExceededError,
    ToolDependencyUnavailableError,
    ToolDuplicateDefinitionError,
    ToolError,
    ToolInputValidationError,
    ToolNotFoundError,
    ToolOutputValidationError,
    ToolProviderSelectionError,
    ToolRepeatedCallError,
    ToolSafetyViolationError,
    ToolTimeoutError,
    ToolUnexpectedError,
    ToolVersionNotFoundError,
)
from supportops.agent_tools.domain.fingerprints import (
    canonicalize_validated_tool_arguments,
    create_tool_call_fingerprint,
)

__all__ = [
    "ProviderToolDefinition",
    "StrictToolSchema",
    "ToolAuditPolicy",
    "ToolCallLimitExceededError",
    "ToolDefinition",
    "ToolDependencyUnavailableError",
    "ToolDuplicateDefinitionError",
    "ToolError",
    "ToolFailurePolicy",
    "ToolInputValidationError",
    "ToolNotFoundError",
    "ToolOutputValidationError",
    "ToolProviderSelectionError",
    "ToolReference",
    "ToolRepeatedCallError",
    "ToolSafetyLevel",
    "ToolSafetyViolationError",
    "ToolTimeoutError",
    "ToolUnexpectedError",
    "ToolVersionNotFoundError",
    "canonicalize_validated_tool_arguments",
    "create_tool_call_fingerprint",
]
