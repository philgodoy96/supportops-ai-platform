"""Internal observability error taxonomy."""


class ObservabilityError(Exception):
    """Base class for internal observability failures."""


class ObservabilityConfigurationError(ObservabilityError):
    """Raised when explicitly enabled observability is configured incorrectly."""


class ObservabilityExportError(ObservabilityError):
    """Raised internally when telemetry export fails."""


class ObservabilitySerializationError(ObservabilityError):
    """Raised internally when an observation cannot be serialized safely."""


class ObservabilityPrivacyPolicyError(ObservabilityError):
    """Raised internally when data violates the configured export policy."""


class ObservabilityLifecycleError(ObservabilityError):
    """Raised internally when flush or shutdown lifecycle handling fails."""
