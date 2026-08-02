"""Deterministic local service-status lookup tool."""

from collections.abc import Iterable, Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import (
    BeforeValidator,
    JsonValue,
    StringConstraints,
)

from supportops.agent_tools.application.bindings import (
    ExecutableToolBinding,
)
from supportops.agent_tools.application.execution import (
    ToolExecutionContext,
)
from supportops.agent_tools.domain.contracts import (
    StrictToolSchema,
    ToolAuditPolicy,
    ToolDefinition,
    ToolFailurePolicy,
    ToolSafetyLevel,
)

LOOKUP_SERVICE_STATUS_TOOL_NAME = "lookup_service_status"
LOOKUP_SERVICE_STATUS_TOOL_VERSION = 1


def _normalize_service_name(value: object) -> object:
    """Lowercase and trim before pattern validation."""

    if isinstance(value, str):
        return value.strip().lower()

    return value


ServiceName = Annotated[
    str,
    BeforeValidator(_normalize_service_name),
    StringConstraints(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
    ),
]
ServiceSummary = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=500,
    ),
]
IncidentReference = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
    ),
]


class ServiceOperationalStatus(StrEnum):
    """Controlled service states understood by the workflow."""

    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    MAJOR_OUTAGE = "major_outage"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class LookupServiceStatusInput(StrictToolSchema):
    """Strict model-visible service-status arguments."""

    service_name: ServiceName


class LookupServiceStatusOutput(StrictToolSchema):
    """One deterministic service-status snapshot."""

    service_name: ServiceName
    status: ServiceOperationalStatus
    summary: ServiceSummary
    incident_reference: IncidentReference | None
    source: Literal["deterministic_catalog"]


class DeterministicServiceStatusCatalog:
    """Immutable explicitly configured local status catalog."""

    def __init__(
        self,
        snapshots: Iterable[LookupServiceStatusOutput],
    ) -> None:
        snapshots_by_name: dict[
            str,
            LookupServiceStatusOutput,
        ] = {}

        for snapshot in snapshots:
            if snapshot.service_name in snapshots_by_name:
                raise ValueError("The service-status catalog contains a duplicate service name.")

            snapshots_by_name[snapshot.service_name] = snapshot

        self._snapshots: Mapping[
            str,
            LookupServiceStatusOutput,
        ] = MappingProxyType(snapshots_by_name)

    @property
    def snapshots(
        self,
    ) -> tuple[LookupServiceStatusOutput, ...]:
        """Return snapshots in deterministic service-name order."""

        return tuple(self._snapshots[name] for name in sorted(self._snapshots))

    def lookup(
        self,
        service_name: str,
    ) -> LookupServiceStatusOutput:
        """Return an explicit snapshot or a deterministic unknown."""

        snapshot = self._snapshots.get(service_name)

        if snapshot is not None:
            return snapshot

        return LookupServiceStatusOutput(
            service_name=service_name,
            status=ServiceOperationalStatus.UNKNOWN,
            summary=("No deterministic status snapshot is configured for this service."),
            incident_reference=None,
            source="deterministic_catalog",
        )


class LookupServiceStatusToolHandler:
    """Resolve one service status from the injected local catalog."""

    def __init__(
        self,
        catalog: DeterministicServiceStatusCatalog,
    ) -> None:
        self._catalog = catalog

    async def execute(
        self,
        context: object,
        arguments: StrictToolSchema,
    ) -> object:
        """Return one deterministic local status snapshot."""

        if not isinstance(context, ToolExecutionContext):
            raise TypeError("lookup_service_status requires ToolExecutionContext.")

        if not isinstance(
            arguments,
            LookupServiceStatusInput,
        ):
            raise TypeError("lookup_service_status requires LookupServiceStatusInput.")

        return self._catalog.lookup(arguments.service_name)


def create_lookup_service_status_binding(
    *,
    catalog: DeterministicServiceStatusCatalog,
    timeout_seconds: float = 5,
) -> ExecutableToolBinding:
    """Create the immutable service-status runtime binding."""

    return ExecutableToolBinding(
        definition=ToolDefinition(
            name=LOOKUP_SERVICE_STATUS_TOOL_NAME,
            version=LOOKUP_SERVICE_STATUS_TOOL_VERSION,
            description=("Look up an explicitly configured deterministic service-status snapshot."),
            input_schema=LookupServiceStatusInput,
            output_schema=LookupServiceStatusOutput,
            safety_level=ToolSafetyLevel.READ_ONLY,
            timeout_seconds=timeout_seconds,
            failure_policy=(ToolFailurePolicy.FAIL_AGENT_RUN),
            audit_policy=ToolAuditPolicy.SAFE_PROJECTION,
        ),
        handler=LookupServiceStatusToolHandler(catalog),
        safe_input_projector=(project_service_status_safe_input),
        safe_output_projector=(project_service_status_safe_output),
    )


def project_service_status_safe_input(
    value: StrictToolSchema,
) -> Mapping[str, JsonValue]:
    """Project the normalized service identifier."""

    if not isinstance(
        value,
        LookupServiceStatusInput,
    ):
        raise TypeError("Expected LookupServiceStatusInput.")

    return {
        "service_name": value.service_name,
    }


def project_service_status_safe_output(
    value: StrictToolSchema,
) -> Mapping[str, JsonValue]:
    """Project non-sensitive deterministic status metadata."""

    if not isinstance(
        value,
        LookupServiceStatusOutput,
    ):
        raise TypeError("Expected LookupServiceStatusOutput.")

    return {
        "service_name": value.service_name,
        "status": value.status.value,
        "incident_reference": value.incident_reference,
        "has_incident": (value.incident_reference is not None),
        "source": value.source,
    }
