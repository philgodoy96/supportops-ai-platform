"""Privacy-aware fail-open Langfuse observability adapter."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from contextlib import AbstractContextManager
from decimal import Decimal
from threading import Lock
from types import TracebackType
from typing import Any, Literal, Protocol, cast

from langfuse import Langfuse, propagate_attributes

from supportops.observability.context import (
    ActiveObservationContext,
    ActiveTraceContext,
    observation_context_scope,
    trace_context_scope,
)
from supportops.observability.contracts import ObservationScope, TraceScope
from supportops.observability.identity import TraceIdentity
from supportops.observability.models import (
    CostDetails,
    EventObservation,
    FieldPaths,
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
    ObservabilityExportPolicy,
)

_LOGGER = logging.getLogger(__name__)


class _SdkObservation(Protocol):
    id: str
    trace_id: str

    def update(self, **kwargs: object) -> object:
        """Update one SDK observation."""


class _SdkClient(Protocol):
    def create_trace_id(self, *, seed: str) -> str:
        """Derive a provider-compatible deterministic trace ID."""

    def start_as_current_observation(
        self,
        **kwargs: object,
    ) -> AbstractContextManager[_SdkObservation]:
        """Start one current SDK observation."""

    def create_event(self, **kwargs: object) -> _SdkObservation:
        """Create one discrete SDK event."""

    def flush(self) -> None:
        """Flush buffered observations."""

    def shutdown(self) -> None:
        """Shut down the SDK client."""


def create_langfuse_sdk_client(**kwargs: Any) -> _SdkClient:
    """Create the concrete SDK client inside the provider-specific module."""

    return cast(_SdkClient, Langfuse(**kwargs))


class LangfuseObservabilityClient:
    """Adapt Langfuse SDK tracing to application-owned contracts."""

    def __init__(
        self,
        *,
        sdk_client: _SdkClient,
        export_policy: ObservabilityExportPolicy,
        warning_logger: logging.Logger | None = None,
    ) -> None:
        self._sdk_client = sdk_client
        self._export_policy = export_policy
        self._logger = warning_logger or _LOGGER
        self._warning_codes: set[str] = set()
        self._warning_lock = Lock()

    @property
    def provider(self) -> ObservabilityProvider:
        return ObservabilityProvider.LANGFUSE

    @property
    def enabled(self) -> bool:
        return True

    def start_trace(
        self,
        attributes: TraceAttributes,
    ) -> AbstractContextManager[TraceScope]:
        return _LangfuseTraceManager(
            client=self,
            attributes=attributes,
        )

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[ObservationScope]:
        return _LangfuseObservationManager(
            client=self,
            attributes=attributes,
        )

    def record_event(self, event: EventObservation) -> None:
        try:
            payload = self._export_policy.sanitize(
                metadata=cast(
                    Mapping[str, object],
                    event.metadata,
                ),
                field_policy=_event_policy(event.metadata_paths),
            )
            self._sdk_client.create_event(
                name=event.name,
                metadata=payload.metadata,
                **_event_update(event),
            )
        except Exception:
            self._warn_once("event_export_failed")

    def record_trace_event(
        self,
        *,
        identity: TraceIdentity,
        event: EventObservation,
    ) -> None:
        trace_id = self._create_trace_id(identity.trace_seed)
        if trace_id is None:
            return

        try:
            payload = self._export_policy.sanitize(
                metadata=cast(
                    Mapping[str, object],
                    event.metadata,
                ),
                field_policy=_event_policy(event.metadata_paths),
            )
            with propagate_attributes(
                session_id=identity.session_id,
                tags=list(identity.tags),
                trace_name=identity.trace_name,
            ):
                self._sdk_client.create_event(
                    name=event.name,
                    metadata=payload.metadata,
                    trace_context={"trace_id": trace_id},
                    **_event_update(event),
                )
        except Exception:
            self._warn_once("event_export_failed")

    def flush(self) -> None:
        try:
            self._sdk_client.flush()
        except Exception:
            self._warn_once("flush_failed")

    def shutdown(self) -> None:
        try:
            self._sdk_client.shutdown()
        except Exception:
            self._warn_once("shutdown_failed")

    def _create_trace_id(self, seed: str) -> str | None:
        try:
            return self._sdk_client.create_trace_id(seed=seed)
        except Exception:
            self._warn_once("trace_id_derivation_failed")
            return None

    def _warn_once(self, code: str) -> None:
        with self._warning_lock:
            if code in self._warning_codes:
                return

            self._warning_codes.add(code)

        self._logger.warning(
            "observability export failed",
            extra={
                "observability_provider": "langfuse",
                "telemetry_export_error_code": code,
            },
        )


class _LangfuseTraceManager(AbstractContextManager[TraceScope]):
    def __init__(
        self,
        *,
        client: LangfuseObservabilityClient,
        attributes: TraceAttributes,
    ) -> None:
        self._client = client
        self._attributes = attributes
        self._sdk_manager: AbstractContextManager[_SdkObservation] | None = None
        self._trace_context_manager: AbstractContextManager[ActiveTraceContext] | None = None
        self._attribute_manager: AbstractContextManager[Any] | None = None
        self._scope: _LangfuseTraceScope | None = None

    def __enter__(self) -> TraceScope:
        trace_id = self._client._create_trace_id(self._attributes.trace_seed)

        if trace_id is None:
            self._scope = _LangfuseTraceScope(
                client=self._client,
                attributes=self._attributes,
                trace_id=None,
                sdk_observation=None,
            )
            self._trace_context_manager = trace_context_scope(
                ActiveTraceContext(
                    trace_seed=self._attributes.trace_seed,
                    session_id=self._attributes.session_id,
                )
            )
            self._trace_context_manager.__enter__()
            return self._scope

        try:
            payload = self._client._export_policy.sanitize(
                metadata=cast(
                    Mapping[str, object],
                    self._attributes.metadata,
                ),
                field_policy=_trace_policy(self._attributes.metadata_paths),
            )
            self._sdk_manager = self._client._sdk_client.start_as_current_observation(
                as_type=ObservationType.AGENT.value,
                name=self._attributes.name,
                metadata=payload.metadata,
                trace_context={
                    "trace_id": trace_id,
                    "parent_span_id": _root_parent_span_id(self._attributes.trace_seed),
                },
            )
            sdk_observation = self._sdk_manager.__enter__()

            self._attribute_manager = propagate_attributes(
                session_id=self._attributes.session_id,
                metadata=payload.metadata,
                tags=list(self._attributes.tags),
                trace_name=self._attributes.name,
            )
            self._attribute_manager.__enter__()
        except Exception:
            self._client._warn_once("trace_start_failed")
            sdk_observation = None

        self._scope = _LangfuseTraceScope(
            client=self._client,
            attributes=self._attributes,
            trace_id=trace_id,
            sdk_observation=sdk_observation,
        )
        self._trace_context_manager = trace_context_scope(
            ActiveTraceContext(
                trace_seed=self._attributes.trace_seed,
                trace_id=trace_id,
                session_id=self._attributes.session_id,
            )
        )
        self._trace_context_manager.__enter__()

        return self._scope

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if self._scope is not None and exc_type is not None:
            self._scope._safe_update(
                ObservationUpdate(
                    status=ObservationStatus.ERROR,
                    error_code="unhandled_business_error",
                )
            )

        if self._trace_context_manager is not None:
            self._trace_context_manager.__exit__(
                exc_type,
                exc,
                traceback,
            )

        if self._attribute_manager is not None:
            try:
                self._attribute_manager.__exit__(None, None, None)
            except Exception:
                self._client._warn_once("trace_attributes_end_failed")

        if self._sdk_manager is not None:
            try:
                self._sdk_manager.__exit__(None, None, None)
            except Exception:
                self._client._warn_once("trace_end_failed")

        return False


class _LangfuseObservationManager(AbstractContextManager[ObservationScope]):
    def __init__(
        self,
        *,
        client: LangfuseObservabilityClient,
        attributes: ObservationAttributes,
    ) -> None:
        self._client = client
        self._attributes = attributes
        self._sdk_manager: AbstractContextManager[_SdkObservation] | None = None
        self._context_manager: AbstractContextManager[ActiveObservationContext] | None = None
        self._scope: _LangfuseObservationScope | None = None

    def __enter__(self) -> ObservationScope:
        try:
            payload = self._client._export_policy.sanitize(
                metadata=cast(
                    Mapping[str, object],
                    self._attributes.metadata,
                ),
                field_policy=_observation_policy(self._attributes),
                input_data=self._attributes.input_data,
            )

            kwargs: dict[str, object] = {
                "as_type": self._attributes.observation_type.value,
                "name": self._attributes.name,
                "metadata": payload.metadata,
            }

            if payload.input_data is not None:
                kwargs["input"] = payload.input_data

            if self._attributes.model is not None:
                kwargs["model"] = self._attributes.model

            self._sdk_manager = self._client._sdk_client.start_as_current_observation(**kwargs)
            sdk_observation = self._sdk_manager.__enter__()
        except Exception:
            self._client._warn_once("observation_start_failed")
            sdk_observation = None

        self._scope = _LangfuseObservationScope(
            client=self._client,
            attributes=self._attributes,
            sdk_observation=sdk_observation,
        )
        self._context_manager = observation_context_scope(
            ActiveObservationContext(
                name=self._attributes.name,
                observation_id=(None if sdk_observation is None else sdk_observation.id),
            )
        )
        self._context_manager.__enter__()

        return self._scope

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if self._scope is not None and exc_type is not None:
            self._scope._safe_update(
                ObservationUpdate(
                    status=ObservationStatus.ERROR,
                    error_code="unhandled_business_error",
                )
            )

        if self._context_manager is not None:
            self._context_manager.__exit__(
                exc_type,
                exc,
                traceback,
            )

        if self._sdk_manager is not None:
            try:
                self._sdk_manager.__exit__(None, None, None)
            except Exception:
                self._client._warn_once("observation_end_failed")

        return False


class _LangfuseTraceScope:
    def __init__(
        self,
        *,
        client: LangfuseObservabilityClient,
        attributes: TraceAttributes,
        trace_id: str | None,
        sdk_observation: _SdkObservation | None,
    ) -> None:
        self._client = client
        self._attributes = attributes
        self._trace_id = trace_id
        self._sdk_observation = sdk_observation

    @property
    def trace_seed(self) -> str:
        return self._attributes.trace_seed

    @property
    def trace_id(self) -> str | None:
        return self._trace_id

    @property
    def session_id(self) -> str | None:
        return self._attributes.session_id

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[ObservationScope]:
        return _LangfuseObservationManager(
            client=self._client,
            attributes=attributes,
        )

    def update(self, update: ObservationUpdate) -> None:
        self._safe_update(update)

    def record_event(self, event: EventObservation) -> None:
        self._client.record_event(event)

    def _safe_update(self, update: ObservationUpdate) -> None:
        if self._sdk_observation is None:
            return

        try:
            payload = self._client._export_policy.sanitize(
                metadata=cast(
                    Mapping[str, object],
                    update.metadata,
                ),
                field_policy=_trace_policy(self._attributes.metadata_paths),
            )

            kwargs = _observation_update(update)
            kwargs["metadata"] = payload.metadata

            self._sdk_observation.update(**kwargs)
        except Exception:
            self._client._warn_once("trace_update_failed")


class _LangfuseObservationScope:
    def __init__(
        self,
        *,
        client: LangfuseObservabilityClient,
        attributes: ObservationAttributes,
        sdk_observation: _SdkObservation | None,
    ) -> None:
        self._client = client
        self._attributes = attributes
        self._sdk_observation = sdk_observation

    @property
    def observation_id(self) -> str | None:
        if self._sdk_observation is None:
            return None

        return self._sdk_observation.id

    def update(self, update: ObservationUpdate) -> None:
        self._safe_update(update)

    def start_observation(
        self,
        attributes: ObservationAttributes,
    ) -> AbstractContextManager[ObservationScope]:
        return _LangfuseObservationManager(
            client=self._client,
            attributes=attributes,
        )

    def record_event(self, event: EventObservation) -> None:
        self._client.record_event(event)

    def _safe_update(self, update: ObservationUpdate) -> None:
        if self._sdk_observation is None:
            return

        try:
            payload = self._client._export_policy.sanitize(
                metadata=cast(
                    Mapping[str, object],
                    update.metadata,
                ),
                field_policy=_observation_policy(self._attributes),
                output_data=update.output_data,
            )

            kwargs = _observation_update(update)
            kwargs["metadata"] = payload.metadata

            if payload.output_data is not None:
                kwargs["output"] = payload.output_data

            self._sdk_observation.update(**kwargs)
        except Exception:
            self._client._warn_once("observation_update_failed")


def _trace_policy(paths: FieldPaths) -> ExportFieldPolicy:
    return ExportFieldPolicy(metadata_paths=paths)


def _event_policy(paths: FieldPaths) -> ExportFieldPolicy:
    return ExportFieldPolicy(metadata_paths=paths)


def _observation_policy(
    attributes: ObservationAttributes,
) -> ExportFieldPolicy:
    return ExportFieldPolicy(
        metadata_paths=attributes.metadata_paths,
        input_paths=attributes.input_paths,
        output_paths=attributes.output_paths,
    )


def _event_update(
    event: EventObservation,
) -> dict[str, object]:
    kwargs: dict[str, object] = {}

    if event.status is ObservationStatus.ERROR:
        kwargs["level"] = "ERROR"

    if event.error_code is not None:
        kwargs["status_message"] = event.error_code

    return kwargs


def _observation_update(
    update: ObservationUpdate,
) -> dict[str, object]:
    kwargs: dict[str, object] = {}

    if update.status is ObservationStatus.ERROR:
        kwargs["level"] = "ERROR"
    elif update.status is ObservationStatus.CANCELLED:
        kwargs["level"] = "WARNING"

    if update.status_message is not None:
        kwargs["status_message"] = update.status_message
    elif update.error_code is not None:
        kwargs["status_message"] = update.error_code

    usage_details = _usage_details(update.usage)
    if usage_details is not None:
        kwargs["usage_details"] = usage_details

    cost_details = _cost_details(update.cost)
    if cost_details is not None:
        kwargs["cost_details"] = cost_details

    return kwargs


def _usage_details(
    usage: UsageDetails | None,
) -> dict[str, int] | None:
    if usage is None:
        return None

    components = {
        key: value
        for key, value in {
            "input": usage.input_tokens,
            "cached_input": usage.cached_input_tokens,
            "output": usage.output_tokens,
            "reasoning": usage.reasoning_tokens,
        }.items()
        if value is not None
    }

    if components:
        return components

    if usage.total_tokens is not None:
        return {"total": usage.total_tokens}

    return None


def _cost_details(
    cost: CostDetails | None,
) -> dict[str, float] | None:
    if cost is None:
        return None

    if cost.pricing_status is PricingStatus.UNKNOWN:
        return None

    components = {
        key: _decimal_to_float(value)
        for key, value in {
            "input": cost.input_cost,
            "cached_input": cost.cached_input_cost,
            "output": cost.output_cost,
            "reasoning": cost.reasoning_cost,
        }.items()
        if value is not None
    }

    if components:
        return components

    if cost.total_cost is not None:
        return {"total": _decimal_to_float(cost.total_cost)}

    return None


def _decimal_to_float(value: Decimal) -> float:
    return float(value)


def _root_parent_span_id(trace_seed: str) -> str:
    return hashlib.sha256(f"{trace_seed}:root-parent".encode()).hexdigest()[:16]
