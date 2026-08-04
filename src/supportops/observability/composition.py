"""Composition of process-owned observability clients."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from supportops.core.settings import Settings
from supportops.observability.contracts import ObservabilityClient
from supportops.observability.langfuse import (
    LangfuseObservabilityClient,
    create_langfuse_sdk_client,
)
from supportops.observability.models import (
    ObservabilityCaptureMode,
    ObservabilityProvider,
)
from supportops.observability.noop import NoOpObservabilityClient
from supportops.observability.privacy import (
    MetadataOnlyExportPolicy,
    ObservabilityExportPolicy,
    PrivacySanitizer,
    RedactedContentExportPolicy,
)


def create_observability_client(
    settings: Settings,
    *,
    sdk_factory: Callable[..., Any] = create_langfuse_sdk_client,
) -> ObservabilityClient:
    """Create one process-owned observability client."""

    if settings.ai_observability_provider is ObservabilityProvider.NOOP:
        return NoOpObservabilityClient()

    public_key = settings.langfuse_public_key
    secret_key = settings.langfuse_secret_key

    if public_key is None or secret_key is None:
        raise ValueError("Langfuse credentials must be validated before composition")

    sdk_client = sdk_factory(
        public_key=public_key.get_secret_value(),
        secret_key=secret_key.get_secret_value(),
        base_url=str(settings.langfuse_base_url),
        environment=settings.langfuse_environment,
        release=settings.langfuse_release,
        timeout=settings.langfuse_timeout_seconds,
        mask_otel_spans=_defense_in_depth_mask,
    )

    return LangfuseObservabilityClient(
        sdk_client=sdk_client,
        export_policy=_create_export_policy(settings),
    )


def _create_export_policy(
    settings: Settings,
) -> ObservabilityExportPolicy:
    sanitizer = PrivacySanitizer()

    if settings.langfuse_capture_mode is ObservabilityCaptureMode.REDACTED_CONTENT:
        return RedactedContentExportPolicy(sanitizer=sanitizer)

    return MetadataOnlyExportPolicy(sanitizer=sanitizer)


def _defense_in_depth_mask(*, params: object) -> None:
    """Leave the batch unchanged after application sanitization.

    Application-owned privacy policies are the primary boundary. The SDK
    export-stage hook is configured deliberately so additional defensive
    patches can be introduced without changing process composition.
    """

    del params
    return None
