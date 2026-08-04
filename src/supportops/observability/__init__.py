"""Application-owned AI observability contracts and models."""

from typing import TYPE_CHECKING

from supportops.observability.context import (
    ActiveObservationContext,
    ActiveTraceContext,
    current_observation_context,
    current_trace_context,
    observation_context_scope,
    trace_context_scope,
)
from supportops.observability.contracts import (
    ObservabilityClient,
    ObservationContainer,
    ObservationScope,
    TraceScope,
)
from supportops.observability.errors import (
    ObservabilityConfigurationError,
    ObservabilityError,
    ObservabilityExportError,
    ObservabilityLifecycleError,
    ObservabilityPrivacyPolicyError,
    ObservabilitySerializationError,
)
from supportops.observability.identity import (
    TraceIdentity,
    agent_run_trace_identity,
    knowledge_index_trace_identity,
    semantic_search_trace_identity,
    ticket_session_id,
)
from supportops.observability.models import (
    CostDetails,
    EventObservation,
    FieldPath,
    FieldPaths,
    ObservabilityCaptureMode,
    ObservabilityProvider,
    ObservationAttributes,
    ObservationStatus,
    ObservationType,
    ObservationUpdate,
    PricingStatus,
    TraceAttributes,
    UsageDetails,
)
from supportops.observability.privacy import (
    ExportFieldPolicy,
    MetadataOnlyExportPolicy,
    ObservabilityExportPolicy,
    PrivacySanitizer,
    RedactedContentExportPolicy,
    SanitizationLimits,
    SanitizedObservationPayload,
)

if TYPE_CHECKING:
    from supportops.observability.composition import create_observability_client
    from supportops.observability.langfuse import LangfuseObservabilityClient
    from supportops.observability.noop import NoOpObservabilityClient

__all__ = [
    "ActiveObservationContext",
    "ActiveTraceContext",
    "CostDetails",
    "EventObservation",
    "ExportFieldPolicy",
    "FieldPath",
    "FieldPaths",
    "LangfuseObservabilityClient",
    "MetadataOnlyExportPolicy",
    "NoOpObservabilityClient",
    "ObservabilityCaptureMode",
    "ObservabilityClient",
    "ObservabilityConfigurationError",
    "ObservabilityError",
    "ObservabilityExportError",
    "ObservabilityExportPolicy",
    "ObservabilityLifecycleError",
    "ObservabilityPrivacyPolicyError",
    "ObservabilityProvider",
    "ObservabilitySerializationError",
    "ObservationAttributes",
    "ObservationContainer",
    "ObservationScope",
    "ObservationStatus",
    "ObservationType",
    "ObservationUpdate",
    "PricingStatus",
    "PrivacySanitizer",
    "RedactedContentExportPolicy",
    "SanitizationLimits",
    "SanitizedObservationPayload",
    "TraceAttributes",
    "TraceIdentity",
    "TraceScope",
    "UsageDetails",
    "agent_run_trace_identity",
    "create_observability_client",
    "current_observation_context",
    "current_trace_context",
    "knowledge_index_trace_identity",
    "observation_context_scope",
    "semantic_search_trace_identity",
    "ticket_session_id",
    "trace_context_scope",
]


def __getattr__(name: str) -> object:
    if name == "create_observability_client":
        from supportops.observability.composition import (
            create_observability_client,
        )

        return create_observability_client

    if name == "LangfuseObservabilityClient":
        from supportops.observability.langfuse import (
            LangfuseObservabilityClient,
        )

        return LangfuseObservabilityClient

    if name == "NoOpObservabilityClient":
        from supportops.observability.noop import NoOpObservabilityClient

        return NoOpObservabilityClient

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
