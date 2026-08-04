from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import BaseModel

from supportops.evaluation.contracts.hashing import (
    CanonicalSerializationError,
    canonical_json_bytes,
    sha256_hexdigest,
)


class ExampleModel(BaseModel):
    amount: Decimal
    captured_at: datetime


def test_canonical_json_is_independent_of_mapping_order() -> None:
    first = {"b": 2, "a": {"value": 1}}
    second = {"a": {"value": 1}, "b": 2}

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert sha256_hexdigest(first) == sha256_hexdigest(second)


def test_canonical_json_normalizes_supported_domain_values() -> None:
    model = ExampleModel(
        amount=Decimal("1.2300"),
        captured_at=datetime(2026, 8, 4, 23, 0, tzinfo=UTC),
    )

    assert canonical_json_bytes(model) == (
        b'{"amount":"1.2300","captured_at":"2026-08-04T23:00:00+00:00"}'
    )


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_canonical_json_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(CanonicalSerializationError):
        canonical_json_bytes(value)


def test_canonical_json_rejects_non_string_mapping_keys() -> None:
    with pytest.raises(CanonicalSerializationError):
        canonical_json_bytes({1: "unsupported"})
