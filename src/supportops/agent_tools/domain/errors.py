"""Typed application-owned failures for controlled tool behavior."""


class ToolError(RuntimeError):
    """Base failure normalized at the application tool boundary."""

    error_code = "tool_unexpected"
    retryable = False


class ToolDuplicateDefinitionError(ToolError):
    """Raised when a registry contains the same exact tool twice."""

    error_code = "tool_definition_duplicate"
    retryable = False

    def __init__(self) -> None:
        super().__init__("The tool registry contains a duplicate definition.")


class ToolNotFoundError(ToolError):
    """Raised when a model requests an unregistered tool name."""

    error_code = "tool_not_found"
    retryable = False

    def __init__(self) -> None:
        super().__init__("The requested tool is not registered.")


class ToolVersionNotFoundError(ToolError):
    """Raised when a registered name lacks the requested version."""

    error_code = "tool_version_not_found"
    retryable = False

    def __init__(self) -> None:
        super().__init__("The requested tool version is not registered.")


class ToolProviderSelectionError(ToolError):
    """Raised when provider exposure is ambiguous or duplicated."""

    error_code = "tool_provider_selection_invalid"
    retryable = False

    def __init__(self) -> None:
        super().__init__("The selected provider tool definitions are invalid.")


class ToolSafetyViolationError(ToolError):
    """Raised before a disallowed tool can be exposed or executed."""

    error_code = "tool_safety_violation"
    retryable = False

    def __init__(self) -> None:
        super().__init__("The requested tool is not permitted by the active safety policy.")


class ToolInputValidationError(ToolError):
    """Raised when provider arguments violate a tool input contract."""

    error_code = "tool_input_invalid"
    retryable = False

    def __init__(self) -> None:
        super().__init__("The requested tool arguments are invalid.")


class ToolOutputValidationError(ToolError):
    """Raised when a tool dependency violates its output contract."""

    error_code = "tool_output_invalid"
    retryable = False

    def __init__(self) -> None:
        super().__init__("The tool returned an invalid output.")


class ToolTimeoutError(ToolError):
    """Raised when bounded tool execution exceeds its timeout."""

    error_code = "tool_timeout"
    retryable = True

    def __init__(self) -> None:
        super().__init__("The tool did not complete within its execution timeout.")


class ToolDependencyUnavailableError(ToolError):
    """Raised when a read-only tool dependency is unavailable."""

    error_code = "tool_dependency_unavailable"
    retryable = True

    def __init__(self) -> None:
        super().__init__("The tool dependency is unavailable.")


class ToolRepeatedCallError(ToolError):
    """Raised when a validated tool call fingerprint repeats."""

    error_code = "tool_call_repeated"
    retryable = False

    def __init__(self) -> None:
        super().__init__("The workflow attempted a repeated tool call.")


class ToolCallLimitExceededError(ToolError):
    """Raised before a tool call can exceed the configured limit."""

    error_code = "tool_call_limit_exceeded"
    retryable = False

    def __init__(self) -> None:
        super().__init__("The workflow exceeded the configured tool-call limit.")


class ToolUnexpectedError(ToolError):
    """Raised for normalized internal tool execution failures."""

    error_code = "tool_unexpected"
    retryable = True

    def __init__(self) -> None:
        super().__init__("The tool failed unexpectedly.")
