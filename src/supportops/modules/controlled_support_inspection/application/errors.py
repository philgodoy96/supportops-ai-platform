"""Expected controlled-support inspection application errors."""

from typing import ClassVar


class ControlledSupportInspectionError(Exception):
    """Base class for expected inspection failures."""

    error_code: ClassVar[str]
    safe_summary: ClassVar[str]

    def __init__(self) -> None:
        super().__init__(self.safe_summary)


class ControlledSupportInspectionNotFoundError(ControlledSupportInspectionError):
    """Raised when a scoped inspection lookup does not resolve."""

    error_code = "controlled_support_inspection_not_found"
    safe_summary = "The controlled support inspection was not found."


class UnsupportedAgentRunInspectionError(ControlledSupportInspectionError):
    """Raised when an AgentRun uses an unsupported workflow."""

    error_code = "unsupported_agent_run_inspection"
    safe_summary = "The AgentRun workflow does not support this inspection view."


class ControlledSupportInspectionInconsistentError(ControlledSupportInspectionError):
    """Raised when persisted inspection data violates invariants."""

    error_code = "controlled_support_inspection_inconsistent"
    safe_summary = "The controlled support inspection data is internally inconsistent."
