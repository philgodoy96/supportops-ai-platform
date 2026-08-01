"""Unit tests for classification prediction artifacts."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from supportops.evaluation.ticket_classification.predictions import (
    DuplicateTicketClassificationPredictionError,
    TicketClassificationPredictionError,
    load_ticket_classification_predictions,
)

_PROMPT_HASH = "a" * 64


def _success_payload(
    *,
    case_id: str = "billing-case-001",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "status": "succeeded",
        "provenance": {
            "prompt_id": "ticket-classification",
            "prompt_version": 1,
            "prompt_content_hash": _PROMPT_HASH,
            "provider": "mock",
            "model": "mock-ticket-classifier-v1",
        },
        "output": {
            "category": "billing",
            "intent": "ask_question",
            "urgency": "normal",
            "sentiment": "neutral",
            "requires_human_review": False,
            "summary": "The customer asks about a charge.",
            "schema_version": "ticket-classification-v1",
        },
        "invocations": [
            {
                "invocation_sequence": 1,
                "status": "succeeded",
                "provider": "mock",
                "model": "mock-ticket-classifier-v1",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 0,
                    "output_tokens": 20,
                    "reasoning_tokens": None,
                    "total_tokens": 120,
                },
                "cost": {
                    "pricing_catalog_version": "pricing-v1",
                    "pricing_found": True,
                    "estimated_input_cost_usd": "0",
                    "estimated_cached_input_cost_usd": "0",
                    "estimated_output_cost_usd": "0",
                    "estimated_total_cost_usd": "0",
                },
                "latency_ms": 25,
                "error_code": None,
            },
        ],
    }


def _write_jsonl(
    path: Path,
    *payloads: dict[str, object],
) -> None:
    path.write_text(
        "\n".join(json.dumps(payload) for payload in payloads) + "\n",
        encoding="utf-8",
    )


def test_loads_successful_prediction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        path,
        _success_payload(),
    )

    predictions = load_ticket_classification_predictions(
        path,
    )

    assert len(predictions.predictions) == 1
    assert predictions.predictions[0].case_id == ("billing-case-001")
    assert len(predictions.content_hash) == 64


def test_prediction_hash_ignores_json_formatting(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    payload = _success_payload()

    first_path.write_text(
        json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    second_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ).replace("\n", "")
        + "\n",
        encoding="utf-8",
    )

    first = load_ticket_classification_predictions(
        first_path,
    )
    second = load_ticket_classification_predictions(
        second_path,
    )

    assert first.content_hash == second.content_hash


def test_rejects_duplicate_prediction_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        path,
        _success_payload(),
        _success_payload(),
    )

    with pytest.raises(
        DuplicateTicketClassificationPredictionError,
        match="Duplicate ticket classification prediction",
    ):
        load_ticket_classification_predictions(path)


def test_rejects_non_contiguous_invocation_sequences(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.jsonl"
    payload = _success_payload()
    invocations = deepcopy(payload["invocations"])

    assert isinstance(invocations, list)
    assert isinstance(invocations[0], dict)

    invocations[0]["invocation_sequence"] = 2
    payload["invocations"] = invocations
    _write_jsonl(path, payload)

    with pytest.raises(
        TicketClassificationPredictionError,
        match="does not match the prediction contract",
    ):
        load_ticket_classification_predictions(path)


def test_rejects_invocation_provider_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.jsonl"
    payload = _success_payload()
    invocations = deepcopy(payload["invocations"])

    assert isinstance(invocations, list)
    assert isinstance(invocations[0], dict)

    invocations[0]["provider"] = "openai"
    payload["invocations"] = invocations
    _write_jsonl(path, payload)

    with pytest.raises(
        TicketClassificationPredictionError,
        match="does not match the prediction contract",
    ):
        load_ticket_classification_predictions(path)


def test_rejects_unknown_pricing_with_cost(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.jsonl"
    payload = _success_payload()
    invocations = deepcopy(payload["invocations"])

    assert isinstance(invocations, list)
    assert isinstance(invocations[0], dict)

    cost = invocations[0]["cost"]
    assert isinstance(cost, dict)

    cost["pricing_found"] = False
    payload["invocations"] = invocations
    _write_jsonl(path, payload)

    with pytest.raises(
        TicketClassificationPredictionError,
        match="does not match the prediction contract",
    ):
        load_ticket_classification_predictions(path)


def test_rejects_blank_prediction_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps(_success_payload()) + "\n\n",
        encoding="utf-8",
    )

    with pytest.raises(
        TicketClassificationPredictionError,
        match="line 2 must not be blank",
    ):
        load_ticket_classification_predictions(path)


def test_rejects_missing_prediction_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TicketClassificationPredictionError,
        match="could not be read",
    ):
        load_ticket_classification_predictions(
            tmp_path / "missing.jsonl",
        )
