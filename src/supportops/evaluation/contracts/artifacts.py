from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from supportops.evaluation.contracts.hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


class ArtifactWriteError(RuntimeError):
    """Base error for canonical artifact write failures."""


class ArtifactHashMismatchError(ArtifactWriteError):
    """Raised when staged artifact content does not match the expected hash."""


@dataclass(frozen=True, slots=True)
class ArtifactWriteResult:
    """Metadata returned after a successful atomic artifact write."""

    path: Path
    content_hash: str
    size_bytes: int


def write_bytes_atomically(
    destination: Path,
    content: bytes,
    *,
    expected_hash: str | None = None,
) -> ArtifactWriteResult:
    """Write bytes without exposing a partial canonical artifact."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        actual_hash = sha256_file(temporary_path)
        if expected_hash is not None and actual_hash != expected_hash:
            raise ArtifactHashMismatchError("Staged artifact hash does not match the expected hash")

        os.replace(temporary_path, destination)
        return ArtifactWriteResult(
            path=destination,
            content_hash=actual_hash,
            size_bytes=len(content),
        )
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_text_atomically(
    destination: Path,
    content: str,
    *,
    encoding: str = "utf-8",
    expected_hash: str | None = None,
) -> ArtifactWriteResult:
    """Write encoded text through the atomic byte writer."""

    return write_bytes_atomically(
        destination,
        content.encode(encoding),
        expected_hash=expected_hash,
    )


def write_canonical_json_atomically(
    destination: Path,
    value: object,
    *,
    expected_hash: str | None = None,
) -> ArtifactWriteResult:
    """Write deterministic JSON followed by one newline."""

    content = canonical_json_bytes(value) + b"\n"
    actual_hash = sha256_bytes(content)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ArtifactHashMismatchError("Canonical JSON content does not match the expected hash")
    return write_bytes_atomically(
        destination,
        content,
        expected_hash=actual_hash,
    )
