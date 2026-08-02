"""Unit tests for controlled-support inspection errors."""

from supportops.modules.controlled_support_inspection.application.errors import (
    ControlledSupportInspectionInconsistentError,
    ControlledSupportInspectionNotFoundError,
    UnsupportedAgentRunInspectionError,
)


def test_not_found_error_has_stable_contract() -> None:
    error = ControlledSupportInspectionNotFoundError()

    assert error.error_code == ("controlled_support_inspection_not_found")
    assert str(error) == ("The controlled support inspection was not found.")


def test_unsupported_workflow_error_has_stable_contract() -> None:
    error = UnsupportedAgentRunInspectionError()

    assert error.error_code == ("unsupported_agent_run_inspection")
    assert str(error) == ("The AgentRun workflow does not support this inspection view.")


def test_inconsistency_error_has_stable_contract() -> None:
    error = ControlledSupportInspectionInconsistentError()

    assert error.error_code == ("controlled_support_inspection_inconsistent")
    assert str(error) == ("The controlled support inspection data is internally inconsistent.")
