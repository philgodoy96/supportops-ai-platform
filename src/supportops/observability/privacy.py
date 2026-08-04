"""Privacy-aware export policies for AI observability."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from supportops.observability.errors import (
    ObservabilityPrivacyPolicyError,
    ObservabilitySerializationError,
)
from supportops.observability.models import (
    JsonValue,
    ObservabilityCaptureMode,
)

type FieldPath = tuple[str, ...]

_EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)
_PHONE_CANDIDATE_PATTERN = re.compile(r"(?<![\w-])\+?\d[\d().\-\s]{8,}\d(?![\w-])")
_DATABASE_URL_PATTERN = re.compile(
    r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|"
    r"redis|rediss)://[^\s]+",
    re.IGNORECASE,
)
_BEARER_TOKEN_PATTERN = re.compile(
    r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}",
    re.IGNORECASE,
)
_PREFIXED_API_KEY_PATTERN = re.compile(
    r"\b(?:sk|pk)-(?:[a-z0-9]+-){0,4}[a-z0-9_]{8,}\b",
    re.IGNORECASE,
)
_AWS_ACCESS_KEY_PATTERN = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")

_EXACT_FORBIDDEN_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "authorization_header",
        "connection_string",
        "database_url",
        "db_url",
        "embedding_vector",
        "embedding_vectors",
        "exception",
        "exception_payload",
        "grant",
        "grant_token",
        "lease_token",
        "password",
        "postgresql_url",
        "proxy_authorization",
        "raw_checkpoint",
        "raw_exception",
        "refresh_token",
        "secret",
        "secret_key",
        "stack_trace",
        "token",
        "traceback",
        "vector",
        "vectors",
    }
)

_FORBIDDEN_SUFFIXES = (
    "_access_token",
    "_api_key",
    "_authorization",
    "_authorization_header",
    "_database_url",
    "_lease_token",
    "_password",
    "_refresh_token",
    "_secret",
    "_secret_key",
)


class _OmittedValue:
    """Internal sentinel for values removed by the privacy policy."""


_OMIT = _OmittedValue()


@dataclass(frozen=True, slots=True)
class SanitizationLimits:
    """Structural limits applied before telemetry export."""

    max_string_length: int = 512
    max_collection_length: int = 32
    max_depth: int = 8

    def __post_init__(self) -> None:
        if self.max_string_length < 16:
            raise ValueError("max_string_length must be at least 16")

        if self.max_collection_length < 1:
            raise ValueError("max_collection_length must be positive")

        if self.max_depth < 1:
            raise ValueError("max_depth must be positive")


@dataclass(frozen=True, slots=True)
class ExportFieldPolicy:
    """Allowlisted field paths for one observation payload."""

    metadata_paths: frozenset[FieldPath]
    input_paths: frozenset[FieldPath] = field(default_factory=frozenset)
    output_paths: frozenset[FieldPath] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        _validate_paths("metadata_paths", self.metadata_paths)
        _validate_paths("input_paths", self.input_paths)
        _validate_paths("output_paths", self.output_paths)


@dataclass(frozen=True, slots=True)
class SanitizedObservationPayload:
    """Privacy-filtered data ready for an observability adapter."""

    metadata: dict[str, JsonValue]
    input_data: JsonValue = None
    output_data: JsonValue = None


@runtime_checkable
class ObservabilityExportPolicy(Protocol):
    """Application-owned policy applied before telemetry export."""

    @property
    def capture_mode(self) -> ObservabilityCaptureMode:
        """Return the configured content-capture mode."""

    def sanitize(
        self,
        *,
        metadata: Mapping[str, object],
        field_policy: ExportFieldPolicy,
        input_data: object | None = None,
        output_data: object | None = None,
    ) -> SanitizedObservationPayload:
        """Return a bounded payload safe for the configured mode."""


@dataclass(frozen=True, slots=True)
class PrivacySanitizer:
    """Allowlist, bound, and mask telemetry data recursively."""

    limits: SanitizationLimits = field(default_factory=SanitizationLimits)

    def sanitize_metadata(
        self,
        metadata: Mapping[str, object],
        *,
        allowed_paths: frozenset[FieldPath],
    ) -> dict[str, JsonValue]:
        """Sanitize metadata using explicit recursive paths."""

        return self._sanitize_root_mapping(
            metadata,
            allowed_paths=allowed_paths,
        )

    def sanitize_content(
        self,
        content: object | None,
        *,
        allowed_paths: frozenset[FieldPath],
    ) -> JsonValue:
        """Sanitize explicitly allowlisted structured content."""

        if content is None or not allowed_paths:
            return None

        if not isinstance(content, Mapping):
            raise ObservabilityPrivacyPolicyError("redacted content must be a structured mapping")

        return self._sanitize_root_mapping(
            content,
            allowed_paths=allowed_paths,
        )

    def _sanitize_root_mapping(
        self,
        value: Mapping[Any, object],
        *,
        allowed_paths: frozenset[FieldPath],
    ) -> dict[str, JsonValue]:
        sanitized = self._sanitize_mapping(
            value,
            path=(),
            allowed_paths=allowed_paths,
            depth=0,
        )

        if isinstance(sanitized, _OmittedValue):
            return {}

        return sanitized

    def _sanitize_mapping(
        self,
        value: Mapping[Any, object],
        *,
        path: FieldPath,
        allowed_paths: frozenset[FieldPath],
        depth: int,
    ) -> dict[str, JsonValue] | _OmittedValue:
        if depth > self.limits.max_depth:
            return _OMIT

        sanitized: dict[str, JsonValue] = {}

        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                raise ObservabilitySerializationError("observability mapping keys must be strings")

            normalized_key = _normalize_key(raw_key)

            if _is_forbidden_key(normalized_key):
                continue

            child_path = (*path, raw_key)

            if not _path_is_relevant(child_path, allowed_paths):
                continue

            child_value = self._sanitize_value(
                raw_value,
                path=child_path,
                allowed_paths=allowed_paths,
                depth=depth + 1,
            )

            if isinstance(child_value, _OmittedValue):
                continue

            sanitized[raw_key] = child_value

            if len(sanitized) >= self.limits.max_collection_length:
                break

        return sanitized

    def _sanitize_sequence(
        self,
        value: Sequence[object],
        *,
        path: FieldPath,
        allowed_paths: frozenset[FieldPath],
        depth: int,
    ) -> list[JsonValue] | _OmittedValue:
        if depth > self.limits.max_depth:
            return _OMIT

        sanitized: list[JsonValue] = []

        for raw_item in value[: self.limits.max_collection_length]:
            item = self._sanitize_value(
                raw_item,
                path=path,
                allowed_paths=allowed_paths,
                depth=depth + 1,
            )

            if isinstance(item, _OmittedValue):
                continue

            sanitized.append(item)

        return sanitized

    def _sanitize_value(
        self,
        value: object,
        *,
        path: FieldPath,
        allowed_paths: frozenset[FieldPath],
        depth: int,
    ) -> JsonValue | _OmittedValue:
        if depth > self.limits.max_depth:
            return _OMIT

        if isinstance(value, BaseException):
            raise ObservabilityPrivacyPolicyError("exception objects must not be exported")

        if isinstance(value, Enum):
            return self._sanitize_value(
                value.value,
                path=path,
                allowed_paths=allowed_paths,
                depth=depth,
            )

        if isinstance(value, UUID):
            return str(value)

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, date):
            return value.isoformat()

        if isinstance(value, Decimal):
            return str(value)

        if value is None or isinstance(value, bool | int):
            return value

        if isinstance(value, float):
            if not math.isfinite(value):
                raise ObservabilitySerializationError("non-finite floats must not be exported")
            return value

        if isinstance(value, str):
            if path not in allowed_paths:
                return _OMIT
            return self._sanitize_string(value)

        if isinstance(value, bytes | bytearray | memoryview):
            raise ObservabilitySerializationError("binary values must not be exported")

        if isinstance(value, Mapping):
            if not _path_is_relevant(path, allowed_paths):
                return _OMIT

            return self._sanitize_mapping(
                value,
                path=path,
                allowed_paths=allowed_paths,
                depth=depth,
            )

        if isinstance(value, Sequence):
            if path not in allowed_paths and not _path_has_children(
                path,
                allowed_paths,
            ):
                return _OMIT

            return self._sanitize_sequence(
                value,
                path=path,
                allowed_paths=allowed_paths,
                depth=depth,
            )

        raise ObservabilitySerializationError(
            f"unsupported observability value type: {type(value).__name__}"
        )

    def _sanitize_string(self, value: str) -> str:
        if _is_uuid_string(value):
            return value

        masked = _DATABASE_URL_PATTERN.sub(
            "<redacted-database-url>",
            value,
        )
        masked = _BEARER_TOKEN_PATTERN.sub(
            "Bearer <redacted-credential>",
            masked,
        )
        masked = _PREFIXED_API_KEY_PATTERN.sub(
            "<redacted-credential>",
            masked,
        )
        masked = _AWS_ACCESS_KEY_PATTERN.sub(
            "<redacted-credential>",
            masked,
        )
        masked = _EMAIL_PATTERN.sub(
            "<redacted-email>",
            masked,
        )
        masked = _PHONE_CANDIDATE_PATTERN.sub(
            _mask_phone_candidate,
            masked,
        )

        if len(masked) <= self.limits.max_string_length:
            return masked

        return masked[: self.limits.max_string_length - 1] + "…"


@dataclass(frozen=True, slots=True)
class MetadataOnlyExportPolicy:
    """Export allowlisted metadata while omitting all content."""

    sanitizer: PrivacySanitizer = field(default_factory=PrivacySanitizer)

    @property
    def capture_mode(self) -> ObservabilityCaptureMode:
        return ObservabilityCaptureMode.METADATA_ONLY

    def sanitize(
        self,
        *,
        metadata: Mapping[str, object],
        field_policy: ExportFieldPolicy,
        input_data: object | None = None,
        output_data: object | None = None,
    ) -> SanitizedObservationPayload:
        del input_data
        del output_data

        return SanitizedObservationPayload(
            metadata=self.sanitizer.sanitize_metadata(
                metadata,
                allowed_paths=field_policy.metadata_paths,
            )
        )


@dataclass(frozen=True, slots=True)
class RedactedContentExportPolicy:
    """Export allowlisted content only after masking and bounding."""

    sanitizer: PrivacySanitizer = field(default_factory=PrivacySanitizer)

    @property
    def capture_mode(self) -> ObservabilityCaptureMode:
        return ObservabilityCaptureMode.REDACTED_CONTENT

    def sanitize(
        self,
        *,
        metadata: Mapping[str, object],
        field_policy: ExportFieldPolicy,
        input_data: object | None = None,
        output_data: object | None = None,
    ) -> SanitizedObservationPayload:
        return SanitizedObservationPayload(
            metadata=self.sanitizer.sanitize_metadata(
                metadata,
                allowed_paths=field_policy.metadata_paths,
            ),
            input_data=self.sanitizer.sanitize_content(
                input_data,
                allowed_paths=field_policy.input_paths,
            ),
            output_data=self.sanitizer.sanitize_content(
                output_data,
                allowed_paths=field_policy.output_paths,
            ),
        )


def _validate_paths(
    field_name: str,
    paths: frozenset[FieldPath],
) -> None:
    for path in paths:
        if not path:
            raise ValueError(f"{field_name} must not contain an empty path")

        for segment in path:
            if not segment.strip():
                raise ValueError(f"{field_name} must not contain blank path segments")


def _path_is_relevant(
    path: FieldPath,
    allowed_paths: frozenset[FieldPath],
) -> bool:
    return path in allowed_paths or _path_has_children(
        path,
        allowed_paths,
    )


def _path_has_children(
    path: FieldPath,
    allowed_paths: frozenset[FieldPath],
) -> bool:
    return any(
        len(candidate) > len(path) and candidate[: len(path)] == path for candidate in allowed_paths
    )


def _normalize_key(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        value.strip().lower(),
    ).strip("_")


def _is_forbidden_key(normalized_key: str) -> bool:
    if normalized_key in _EXACT_FORBIDDEN_KEYS:
        return True

    if normalized_key.startswith("checkpoint"):
        return True

    if "_checkpoint_" in normalized_key:
        return True

    if normalized_key.endswith("_vector"):
        return True

    if normalized_key.endswith("_vectors"):
        return True

    if "grant" in normalized_key.split("_"):
        return True

    return normalized_key.endswith(_FORBIDDEN_SUFFIXES)


def _is_uuid_string(value: str) -> bool:
    try:
        UUID(value)
    except ValueError:
        return False

    return True


def _mask_phone_candidate(match: re.Match[str]) -> str:
    candidate = match.group(0)
    digit_count = sum(character.isdigit() for character in candidate)

    if 10 <= digit_count <= 15:
        return "<redacted-phone>"

    return candidate
