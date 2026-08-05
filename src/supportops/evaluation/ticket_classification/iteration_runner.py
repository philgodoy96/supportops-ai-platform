"""Offline orchestration for ticket-classification prompt iteration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    StringConstraints,
    model_validator,
)

from supportops.evaluation.contracts.artifacts import (
    write_canonical_json_atomically,
)
from supportops.evaluation.contracts.hashing import sha256_hexdigest
from supportops.evaluation.contracts.manifest import (
    EvaluationManifest,
    EvaluationRunStatus,
)
from supportops.evaluation.ticket_classification.comparison import (
    TicketClassificationComparisonEvidenceKind,
    TicketClassificationPairedComparison,
    compare_ticket_classification_prediction_sets,
    load_ticket_classification_paired_comparison,
    write_ticket_classification_paired_comparison,
)
from supportops.evaluation.ticket_classification.dataset import (
    TicketClassificationEvaluationDataset,
    load_ticket_classification_dataset,
)
from supportops.evaluation.ticket_classification.decision import (
    TicketClassificationPromptDecision,
    decide_ticket_classification_prompt_candidate,
    load_ticket_classification_prompt_decision,
    write_ticket_classification_prompt_decision,
)
from supportops.evaluation.ticket_classification.evaluator import (
    TicketClassificationEvaluationReport,
    evaluate_ticket_classification_predictions,
)
from supportops.evaluation.ticket_classification.failure_analysis import (
    TicketClassificationFailureAnalysis,
    load_ticket_classification_failure_analysis,
    validate_ticket_classification_failure_analysis_against_dataset,
)
from supportops.evaluation.ticket_classification.predictions import (
    TicketClassificationPredictionSet,
    load_ticket_classification_predictions,
)
from supportops.evaluation.ticket_classification.split_manifest import (
    TicketClassificationSplitManifest,
    load_ticket_classification_split_manifest,
)

Sha256Hex = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]


class TicketClassificationPromptIterationError(ValueError):
    """Raised when prompt-iteration evidence cannot be composed safely."""


class TicketClassificationPromptPairManifestContent(BaseModel):
    """Canonical provenance binding for one paired prompt comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_manifest: EvaluationManifest
    baseline_manifest_content_hash: Sha256Hex
    candidate_manifest: EvaluationManifest
    candidate_manifest_content_hash: Sha256Hex

    baseline_report_content_hash: Sha256Hex
    candidate_report_content_hash: Sha256Hex
    comparison_content_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_manifest_binding(self) -> Self:
        if self.baseline_manifest_content_hash != self.baseline_manifest.content_hash():
            raise ValueError("Baseline manifest hash does not match its content.")
        if self.candidate_manifest_content_hash != self.candidate_manifest.content_hash():
            raise ValueError("Candidate manifest hash does not match its content.")
        if (
            self.baseline_manifest.dataset_id != self.candidate_manifest.dataset_id
            or self.baseline_manifest.dataset_version != self.candidate_manifest.dataset_version
            or self.baseline_manifest.dataset_hash != self.candidate_manifest.dataset_hash
        ):
            raise ValueError("Paired manifests must bind the same dataset.")
        if (
            self.baseline_manifest.split_manifest_id != self.candidate_manifest.split_manifest_id
            or self.baseline_manifest.split_manifest_version
            != self.candidate_manifest.split_manifest_version
            or self.baseline_manifest.split_manifest_hash
            != self.candidate_manifest.split_manifest_hash
        ):
            raise ValueError("Paired manifests must bind the same split manifest.")
        if (
            self.baseline_manifest.system_provider != self.candidate_manifest.system_provider
            or self.baseline_manifest.system_model != self.candidate_manifest.system_model
        ):
            raise ValueError("Paired manifests must bind the same provider and model.")
        if self.baseline_manifest.prompt_id != self.candidate_manifest.prompt_id:
            raise ValueError("Paired manifests must bind the same prompt ID.")
        if self.baseline_manifest.prompt_version == self.candidate_manifest.prompt_version:
            raise ValueError("Paired manifests must bind distinct prompt versions.")
        return self


class TicketClassificationPromptPairManifest(TicketClassificationPromptPairManifestContent):
    """Immutable pair-manifest artifact."""

    content_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_content_hash(self) -> Self:
        content = TicketClassificationPromptPairManifestContent.model_validate(
            self.model_dump(
                mode="python",
                exclude={"content_hash"},
            )
        )
        if self.content_hash != sha256_hexdigest(content):
            raise ValueError("Pair-manifest hash does not match canonical content.")
        return self


@dataclass(frozen=True, slots=True)
class TicketClassificationPromptComparisonRunResult:
    """Artifacts produced by one no-network paired comparison."""

    comparison: TicketClassificationPairedComparison
    baseline_manifest: EvaluationManifest
    candidate_manifest: EvaluationManifest
    pair_manifest: TicketClassificationPromptPairManifest


def validate_ticket_classification_failure_analysis_artifact(
    *,
    dataset_path: Path,
    dataset_id: str,
    dataset_version: int,
    split_manifest_path: Path,
    analysis_path: Path,
) -> TicketClassificationFailureAnalysis:
    """Validate a committed development-only failure analysis."""

    dataset = load_ticket_classification_dataset(
        dataset_path,
        dataset_id=dataset_id,
        version=dataset_version,
    )
    split_manifest = load_ticket_classification_split_manifest(split_manifest_path)
    split_manifest.validate_dataset_binding(
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        dataset_hash=dataset.content_hash,
        dataset_case_ids=tuple(case.case_id for case in dataset.cases),
    )

    analysis = load_ticket_classification_failure_analysis(analysis_path)
    validate_ticket_classification_failure_analysis_against_dataset(
        analysis=analysis,
        dataset=dataset,
    )

    if (
        analysis.split_manifest_id != split_manifest.split_manifest_id
        or analysis.split_manifest_version != split_manifest.split_manifest_version
        or analysis.split_manifest_content_hash != split_manifest.content_hash()
    ):
        raise TicketClassificationPromptIterationError(
            "Failure analysis split-manifest provenance does not match."
        )
    if analysis.analyzed_case_ids != (split_manifest.assignments.development):
        raise TicketClassificationPromptIterationError(
            "Failure analysis must cover the development split exactly and in split-manifest order."
        )

    return analysis


def run_ticket_classification_prompt_comparison(
    *,
    dataset_path: Path,
    dataset_id: str,
    dataset_version: int,
    split_manifest_path: Path,
    baseline_predictions_path: Path,
    candidate_predictions_path: Path,
    evidence_kind: TicketClassificationComparisonEvidenceKind,
    capture_timestamp: AwareDatetime,
    git_commit: str,
) -> TicketClassificationPromptComparisonRunResult:
    """Compose comparison and provenance artifacts without provider access."""

    dataset = load_ticket_classification_dataset(
        dataset_path,
        dataset_id=dataset_id,
        version=dataset_version,
    )
    split_manifest = load_ticket_classification_split_manifest(split_manifest_path)
    baseline_predictions = load_ticket_classification_predictions(baseline_predictions_path)
    candidate_predictions = load_ticket_classification_predictions(candidate_predictions_path)

    comparison = compare_ticket_classification_prediction_sets(
        dataset=dataset,
        split_manifest=split_manifest,
        baseline_predictions=baseline_predictions,
        candidate_predictions=candidate_predictions,
        evidence_kind=evidence_kind,
    )
    baseline_report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=baseline_predictions,
    )
    candidate_report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=candidate_predictions,
    )

    baseline_manifest = _build_evaluation_manifest(
        dataset=dataset,
        split_manifest=split_manifest,
        predictions=baseline_predictions,
        report=baseline_report,
        capture_timestamp=capture_timestamp,
        git_commit=git_commit,
    )
    candidate_manifest = _build_evaluation_manifest(
        dataset=dataset,
        split_manifest=split_manifest,
        predictions=candidate_predictions,
        report=candidate_report,
        capture_timestamp=capture_timestamp,
        git_commit=git_commit,
    )
    pair_content = TicketClassificationPromptPairManifestContent(
        baseline_manifest=baseline_manifest,
        baseline_manifest_content_hash=baseline_manifest.content_hash(),
        candidate_manifest=candidate_manifest,
        candidate_manifest_content_hash=candidate_manifest.content_hash(),
        baseline_report_content_hash=(baseline_report.report_content_hash),
        candidate_report_content_hash=(candidate_report.report_content_hash),
        comparison_content_hash=comparison.comparison_content_hash,
    )
    pair_manifest = TicketClassificationPromptPairManifest(
        **pair_content.model_dump(),
        content_hash=sha256_hexdigest(pair_content),
    )
    return TicketClassificationPromptComparisonRunResult(
        comparison=comparison,
        baseline_manifest=baseline_manifest,
        candidate_manifest=candidate_manifest,
        pair_manifest=pair_manifest,
    )


def run_ticket_classification_prompt_decision(
    *,
    comparison_path: Path,
    decision_template_path: Path,
) -> TicketClassificationPromptDecision:
    """Rebuild a governed decision from comparison and review evidence."""

    comparison = load_ticket_classification_paired_comparison(comparison_path)
    template = load_ticket_classification_prompt_decision(decision_template_path)

    if (
        template.comparison_id != comparison.comparison_id
        or template.comparison_version != comparison.comparison_version
        or template.comparison_content_hash != comparison.comparison_content_hash
    ):
        raise TicketClassificationPromptIterationError(
            "Decision template does not bind the supplied comparison."
        )

    decision = decide_ticket_classification_prompt_candidate(
        comparison=comparison,
        review=template.review,
    )
    if decision != template:
        raise TicketClassificationPromptIterationError(
            "Decision template does not match the deterministic decision."
        )
    return decision


def write_ticket_classification_prompt_comparison_run(
    *,
    comparison_output: Path,
    baseline_manifest_output: Path,
    candidate_manifest_output: Path,
    pair_manifest_output: Path,
    result: TicketClassificationPromptComparisonRunResult,
) -> None:
    """Write comparison and provenance outputs atomically per artifact."""

    write_ticket_classification_paired_comparison(
        comparison_output,
        result.comparison,
    )
    write_canonical_json_atomically(
        baseline_manifest_output,
        result.baseline_manifest.canonical_payload(),
    )
    write_canonical_json_atomically(
        candidate_manifest_output,
        result.candidate_manifest.canonical_payload(),
    )
    write_canonical_json_atomically(
        pair_manifest_output,
        result.pair_manifest.model_dump(mode="python"),
    )


def write_ticket_classification_prompt_decision_run(
    *,
    output: Path,
    decision: TicketClassificationPromptDecision,
) -> None:
    """Write one deterministic prompt decision."""

    write_ticket_classification_prompt_decision(
        output,
        decision,
    )


def _build_evaluation_manifest(
    *,
    dataset: TicketClassificationEvaluationDataset,
    split_manifest: TicketClassificationSplitManifest,
    predictions: TicketClassificationPredictionSet,
    report: TicketClassificationEvaluationReport,
    capture_timestamp: AwareDatetime,
    git_commit: str,
) -> EvaluationManifest:
    prediction_identity = {
        (
            prediction.provenance.prompt_id,
            prediction.provenance.prompt_version,
            prediction.provenance.prompt_content_hash,
            prediction.provenance.provider,
            prediction.provenance.model,
        )
        for prediction in predictions.predictions
    }
    if len(prediction_identity) != 1:
        raise TicketClassificationPromptIterationError(
            "A comparison manifest requires one prediction provenance identity."
        )
    (
        prompt_id,
        prompt_version,
        prompt_hash,
        provider,
        model,
    ) = next(iter(prediction_identity))

    pricing_catalog_versions = set(report.pricing_catalog_versions)
    if len(pricing_catalog_versions) != 1:
        raise TicketClassificationPromptIterationError(
            "A comparison manifest requires one pricing catalog version."
        )

    return EvaluationManifest(
        evaluation_id="ticket-classification-prompt-comparison",
        evaluation_version=1,
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        dataset_hash=dataset.content_hash,
        split_manifest_id=split_manifest.split_manifest_id,
        split_manifest_version=split_manifest.split_manifest_version,
        split_manifest_hash=split_manifest.content_hash(),
        system_provider=provider,
        system_model=model,
        workflow_name="ticket-classification-evaluation",
        workflow_version="ticket-classification-prompt-comparison-v1",
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        prompt_hash=prompt_hash,
        schema_version="ticket-classification-v1",
        pricing_catalog_version=next(iter(pricing_catalog_versions)),
        capture_timestamp=capture_timestamp,
        git_commit=git_commit,
        prediction_hash=predictions.content_hash,
        run_status=EvaluationRunStatus.COMPLETE,
    )
