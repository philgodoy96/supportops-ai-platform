from pathlib import Path

import pytest

from supportops.evaluation.contracts.artifacts import (
    ArtifactHashMismatchError,
    write_bytes_atomically,
    write_canonical_json_atomically,
)
from supportops.evaluation.contracts.hashing import sha256_bytes


def test_atomic_write_creates_complete_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "canonical" / "artifact.json"
    content = b'{"status":"complete"}\n'

    result = write_bytes_atomically(destination, content)

    assert destination.read_bytes() == content
    assert result.content_hash == sha256_bytes(content)
    assert result.size_bytes == len(content)


def test_hash_mismatch_preserves_existing_canonical_artifact(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_bytes(b"existing")

    with pytest.raises(ArtifactHashMismatchError):
        write_bytes_atomically(
            destination,
            b"candidate",
            expected_hash="0" * 64,
        )

    assert destination.read_bytes() == b"existing"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_canonical_json_write_is_stable_across_mapping_order(
    tmp_path: Path,
) -> None:
    first_destination = tmp_path / "first.json"
    second_destination = tmp_path / "second.json"

    first = write_canonical_json_atomically(
        first_destination,
        {"b": 2, "a": 1},
    )
    second = write_canonical_json_atomically(
        second_destination,
        {"a": 1, "b": 2},
    )

    assert first_destination.read_bytes() == b'{"a":1,"b":2}\n'
    assert first.content_hash == second.content_hash
