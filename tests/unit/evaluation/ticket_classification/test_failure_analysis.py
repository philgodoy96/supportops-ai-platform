"""Unit tests for ticket-classification failure analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from supportops.evaluation.ticket_classification.dataset import (
    TICKET_CLASSIFICATION_EVALUATION_DATASET_ID,
    TICKET_CLASSIFICATION_EVALUATION_DATASET_VERSION,
    TicketClassificationEvaluationDataset,
    load_ticket_classification_dataset,
)
from supportops.evaluation.ticket_classification.failure_analysis import (
    TicketClassificationFailureAnalysis,
    TicketClassificationFailureAnalysisContent,
    TicketClassificationFailureAnalysisError,
    TicketClassificationFailureEvidenceKind,
    TicketClassificationFailureEvidenceSummary,
    TicketClassificationFailureObservation,
    TicketClassificationFailureSafetyImpact,
    TicketClassificationFailureType,
    build_ticket_classification_failure_analysis,
    load_ticket_classification_failure_analysis,
    validate_ticket_classification_failure_analysis_against_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATASET_PATH = (
    PROJECT_ROOT
    / "evals"
    / "ticket-classification"
    / "datasets"
    / "ticket-classification-eval-v1.jsonl"
)
ANALYSIS_PATH = (
    PROJECT_ROOT
    / "evals"
    / "ticket-classification"
    / "analyses"
    / "classification-prompt-v1-failure-analysis.json"
)

ANALYSIS_HASH = "18f485339df2f4a47233f5475850ea9d323a28a76d31bc3ea78f6822bd8c695c"
DEVELOPMENT_CASE_IDS = {
    "account-access-password-reset-001",
    "account-access-payroll-permission-002",
    "service-incident-latency-004",
    "billing-angry-low-impact-007",
    "product-bug-export-failure-008",
    "product-bug-cosmetic-alignment-009",
    "how-to-audit-log-export-010",
    "security-suspicious-login-013",
    "feature-request-dark-mode-014",
    "other-ambiguous-problem-017",
    "billing-prompt-injection-018",
    "product-bug-mixed-sentiment-022",
}
HOLDOUT_CASE_IDS = {
    "billing-duplicate-charge-005",
    "billing-refund-request-006",
    "how-to-positive-automation-011",
    "feature-request-bulk-update-015",
    "other-sales-inquiry-016",
    "account-access-executive-lockout-020",
    "product-bug-single-user-dashboard-021",
    "billing-positive-feedback-023",
}
SAFETY_GATE_CASE_IDS = {
    "service-incident-global-outage-003",
    "security-exposed-api-key-012",
    "security-prompt-injection-019",
    "security-data-deletion-request-024",
}


def _load_dataset() -> TicketClassificationEvaluationDataset:
    return load_ticket_classification_dataset(
        DATASET_PATH,
        dataset_id=TICKET_CLASSIFICATION_EVALUATION_DATASET_ID,
        version=TICKET_CLASSIFICATION_EVALUATION_DATASET_VERSION,
    )


def _load_analysis() -> TicketClassificationFailureAnalysis:
    return load_ticket_classification_failure_analysis(ANALYSIS_PATH)


def test_committed_failure_analysis_is_valid_and_immutable() -> None:
    analysis = _load_analysis()

    assert analysis.analysis_id == ("classification-prompt-v1-failure-analysis")
    assert analysis.analysis_version == 1
    assert analysis.analyzed_split == "development"
    assert analysis.analysis_content_hash == ANALYSIS_HASH
    assert len(analysis.analyzed_case_ids) == 12
    assert len(analysis.observations) == 8


def test_analysis_provenance_matches_committed_dataset() -> None:
    analysis = _load_analysis()
    dataset = _load_dataset()

    validate_ticket_classification_failure_analysis_against_dataset(
        analysis=analysis,
        dataset=dataset,
    )

    assert analysis.dataset_content_hash == dataset.content_hash


def test_analysis_uses_development_cases_only() -> None:
    analysis = _load_analysis()
    analyzed_case_ids = set(analysis.analyzed_case_ids)
    observed_case_ids = {
        case_id
        for observation in analysis.observations
        for case_id in observation.affected_case_ids
    }

    assert analyzed_case_ids == DEVELOPMENT_CASE_IDS
    assert analyzed_case_ids.isdisjoint(HOLDOUT_CASE_IDS)
    assert analyzed_case_ids.isdisjoint(SAFETY_GATE_CASE_IDS)
    assert observed_case_ids <= analyzed_case_ids


def test_analysis_does_not_claim_observed_failures() -> None:
    analysis = _load_analysis()

    assert analysis.evidence_summary == (
        TicketClassificationFailureEvidenceSummary(
            provider_observation_count=0,
            static_fixture_observation_count=0,
            dataset_design_hypothesis_count=8,
        )
    )
    assert all(
        observation.evidence_kind
        is TicketClassificationFailureEvidenceKind.DATASET_DESIGN_HYPOTHESIS
        for observation in analysis.observations
    )


def test_failure_taxonomy_covers_prompt_v2_risk_targets() -> None:
    observed_types = {observation.failure_type for observation in _load_analysis().observations}

    assert observed_types == {
        TicketClassificationFailureType.CATEGORY_CONFUSION,
        TicketClassificationFailureType.INTENT_CONFUSION,
        TicketClassificationFailureType.URGENCY_UNDER_CLASSIFICATION,
        TicketClassificationFailureType.URGENCY_OVER_CLASSIFICATION,
        TicketClassificationFailureType.SENTIMENT_MISMATCH,
        TicketClassificationFailureType.HUMAN_REVIEW_FALSE_NEGATIVE,
        TicketClassificationFailureType.CROSS_LABEL_INTERACTION,
        TicketClassificationFailureType.AMBIGUOUS_INPUT,
    }


def test_builder_produces_same_canonical_hash() -> None:
    analysis = _load_analysis()
    content = TicketClassificationFailureAnalysisContent.model_validate(
        analysis.model_dump(
            mode="python",
            exclude={"analysis_content_hash"},
        )
    )

    rebuilt = build_ticket_classification_failure_analysis(content)

    assert rebuilt == analysis


def test_rejects_observation_outside_analyzed_split() -> None:
    analysis = _load_analysis()
    payload = analysis.model_dump(
        mode="python",
        exclude={"analysis_content_hash"},
    )
    first_observation = dict(payload["observations"][0])
    first_observation["affected_case_ids"] = ("service-incident-global-outage-003",)
    payload["observations"] = (
        first_observation,
        *payload["observations"][1:],
    )

    with pytest.raises(
        ValidationError,
        match="outside the analyzed split",
    ):
        TicketClassificationFailureAnalysisContent.model_validate(payload)


def test_rejects_duplicate_failure_types() -> None:
    analysis = _load_analysis()
    payload = analysis.model_dump(
        mode="python",
        exclude={"analysis_content_hash"},
    )
    payload["observations"] = (
        payload["observations"][0],
        payload["observations"][0],
    )
    payload["evidence_summary"] = {
        "provider_observation_count": 0,
        "static_fixture_observation_count": 0,
        "dataset_design_hypothesis_count": 2,
    }

    with pytest.raises(
        ValidationError,
        match="Failure analysis types must be unique",
    ):
        TicketClassificationFailureAnalysisContent.model_validate(payload)


def test_rejects_inconsistent_evidence_summary() -> None:
    analysis = _load_analysis()
    payload = analysis.model_dump(
        mode="python",
        exclude={"analysis_content_hash"},
    )
    payload["evidence_summary"] = {
        "provider_observation_count": 1,
        "static_fixture_observation_count": 0,
        "dataset_design_hypothesis_count": 7,
    }

    with pytest.raises(
        ValidationError,
        match="evidence summary does not match",
    ):
        TicketClassificationFailureAnalysisContent.model_validate(payload)


def test_rejects_prompt_only_remediation_with_schema_change() -> None:
    with pytest.raises(
        ValidationError,
        match="Prompt-only remediation cannot require",
    ):
        TicketClassificationFailureObservation(
            failure_type=TicketClassificationFailureType.CATEGORY_CONFUSION,
            evidence_kind=(TicketClassificationFailureEvidenceKind.DATASET_DESIGN_HYPOTHESIS),
            title="Category boundary risk.",
            description="Category boundary risk.",
            affected_case_ids=("other-ambiguous-problem-017",),
            metric_impacts=("category_accuracy.rate",),
            safety_impact=TicketClassificationFailureSafetyImpact.MEDIUM,
            prompt_only_remediation_appropriate=True,
            dataset_change_required=False,
            schema_change_required=True,
            rationale="The schema would need to change.",
        )


def test_rejects_tampered_content_hash(tmp_path: Path) -> None:
    payload = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    payload["analysis_content_hash"] = "0" * 64
    tampered_path = tmp_path / "tampered-analysis.json"
    tampered_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        TicketClassificationFailureAnalysisError,
        match="does not match the contract",
    ):
        load_ticket_classification_failure_analysis(tampered_path)


def test_rejects_dataset_hash_mismatch() -> None:
    analysis = _load_analysis().model_copy(update={"dataset_content_hash": "0" * 64})

    with pytest.raises(
        TicketClassificationFailureAnalysisError,
        match="dataset hash does not match",
    ):
        validate_ticket_classification_failure_analysis_against_dataset(
            analysis=analysis,
            dataset=_load_dataset(),
        )


def test_rejects_invalid_json(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid-analysis.json"
    invalid_path.write_text("not-json", encoding="utf-8")

    with pytest.raises(
        TicketClassificationFailureAnalysisError,
        match="could not be read",
    ):
        load_ticket_classification_failure_analysis(invalid_path)
