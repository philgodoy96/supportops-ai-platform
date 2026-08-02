"""Unit tests for the deterministic service-status tool."""

from uuid import UUID

import pytest

from supportops.agent_tools.application.execution import (
    ToolExecutionContext,
)
from supportops.agent_tools.domain.contracts import (
    ToolFailurePolicy,
    ToolSafetyLevel,
)
from supportops.agent_tools.tools.service_status import (
    LOOKUP_SERVICE_STATUS_TOOL_NAME,
    LOOKUP_SERVICE_STATUS_TOOL_VERSION,
    DeterministicServiceStatusCatalog,
    LookupServiceStatusInput,
    LookupServiceStatusOutput,
    LookupServiceStatusToolHandler,
    ServiceOperationalStatus,
    create_lookup_service_status_binding,
    project_service_status_safe_input,
    project_service_status_safe_output,
)


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        workspace_id=UUID("10000000-0000-4000-8000-000000000001"),
        ticket_id=UUID("20000000-0000-4000-8000-000000000002"),
        agent_run_id=UUID("30000000-0000-4000-8000-000000000003"),
        agent_run_attempt_id=UUID("40000000-0000-4000-8000-000000000004"),
    )


def _payments_snapshot() -> LookupServiceStatusOutput:
    return LookupServiceStatusOutput(
        service_name="payments-api",
        status=ServiceOperationalStatus.DEGRADED,
        summary=("Elevated latency is configured for the deterministic test scenario."),
        incident_reference="incident-local-001",
        source="deterministic_catalog",
    )


def test_catalog_orders_snapshots_deterministically() -> None:
    catalog = DeterministicServiceStatusCatalog(
        (
            LookupServiceStatusOutput(
                service_name="worker",
                status=(ServiceOperationalStatus.OPERATIONAL),
                summary=("The deterministic worker snapshot is operational."),
                incident_reference=None,
                source="deterministic_catalog",
            ),
            _payments_snapshot(),
        )
    )

    assert [snapshot.service_name for snapshot in catalog.snapshots] == [
        "payments-api",
        "worker",
    ]


def test_catalog_rejects_duplicate_service_name() -> None:
    snapshot = _payments_snapshot()

    with pytest.raises(
        ValueError,
        match="duplicate service name",
    ):
        DeterministicServiceStatusCatalog(
            (
                snapshot,
                snapshot,
            )
        )


async def test_handler_returns_configured_snapshot() -> None:
    snapshot = _payments_snapshot()
    handler = LookupServiceStatusToolHandler(DeterministicServiceStatusCatalog((snapshot,)))

    output = await handler.execute(
        _context(),
        LookupServiceStatusInput(service_name="PAYMENTS-API"),
    )

    assert output == snapshot


async def test_handler_returns_unknown_for_unregistered_service() -> None:
    handler = LookupServiceStatusToolHandler(DeterministicServiceStatusCatalog(()))

    output = await handler.execute(
        _context(),
        LookupServiceStatusInput(service_name="identity-api"),
    )

    assert output == LookupServiceStatusOutput(
        service_name="identity-api",
        status=ServiceOperationalStatus.UNKNOWN,
        summary=("No deterministic status snapshot is configured for this service."),
        incident_reference=None,
        source="deterministic_catalog",
    )


def test_safe_projections_are_bounded() -> None:
    snapshot = _payments_snapshot()

    assert project_service_status_safe_input(
        LookupServiceStatusInput(service_name="payments-api")
    ) == {
        "service_name": "payments-api",
    }
    assert project_service_status_safe_output(snapshot) == {
        "service_name": "payments-api",
        "status": "degraded",
        "incident_reference": "incident-local-001",
        "has_incident": True,
        "source": "deterministic_catalog",
    }


def test_binding_has_exact_read_only_policy() -> None:
    binding = create_lookup_service_status_binding(catalog=DeterministicServiceStatusCatalog(()))

    assert binding.definition.name == (LOOKUP_SERVICE_STATUS_TOOL_NAME)
    assert binding.definition.version == (LOOKUP_SERVICE_STATUS_TOOL_VERSION)
    assert binding.definition.input_schema is (LookupServiceStatusInput)
    assert binding.definition.output_schema is (LookupServiceStatusOutput)
    assert binding.definition.safety_level is (ToolSafetyLevel.READ_ONLY)
    assert binding.definition.failure_policy is (ToolFailurePolicy.FAIL_AGENT_RUN)
    assert binding.definition.timeout_seconds == 5


async def test_handler_requires_trusted_context() -> None:
    handler = LookupServiceStatusToolHandler(DeterministicServiceStatusCatalog(()))

    with pytest.raises(
        TypeError,
        match="requires ToolExecutionContext",
    ):
        await handler.execute(
            object(),
            LookupServiceStatusInput(service_name="payments-api"),
        )


def test_context_import_remains_concrete() -> None:
    context = _context()

    assert isinstance(context, ToolExecutionContext)
