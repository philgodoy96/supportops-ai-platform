from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel

type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


class CanonicalSerializationError(ValueError):
    """Raised when a value cannot be represented as canonical JSON."""


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a supported value to deterministic UTF-8 JSON bytes."""

    normalized = _to_json_value(value)
    try:
        serialized = json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalSerializationError(str(exc)) from exc
    return serialized.encode("utf-8")


def sha256_hexdigest(value: object) -> str:
    """Return the SHA-256 digest of a canonical JSON representation."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(content: bytes) -> str:
    """Return the SHA-256 digest of raw bytes."""

    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it entirely."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        while chunk := artifact.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _to_json_value(value: object) -> JsonValue:
    if isinstance(value, BaseModel):
        return _to_json_value(value.model_dump(mode="python", exclude_none=False))
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalSerializationError("Non-finite floating-point values are not supported")
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise CanonicalSerializationError("Non-finite Decimal values are not supported")
        return format(value, "f")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return _to_json_value(value.value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalSerializationError("Canonical JSON mappings require string keys")
            normalized_mapping[key] = _to_json_value(item)
        return normalized_mapping
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_to_json_value(item) for item in value]

    raise CanonicalSerializationError(
        f"Unsupported canonical JSON value type: {type(value).__name__}"
    )
