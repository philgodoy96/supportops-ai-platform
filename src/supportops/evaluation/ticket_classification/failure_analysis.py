"""Failure-analysis contracts for ticket classification prompt iteration."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from supportops.evaluation.contracts.hashing import sha256_hexdigest
from supportops.evaluation.ticket_classification.dataset import (
    TicketClassificationEvaluationDataset,
)
from supportops.evaluation.ticket_classification.models import EvaluationCaseId

FailureAnalysisIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    ),
]
FailureAnalysisContentHash = Annotated[
    str,
    StringConstraints(pattern=r"^[a-f0-9]{64}$"),
]
FailureAnalysisText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1000),
]
FailureAnalysisMetricName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9_]+(?:\.[a-z0-9_]+)*$",
    ),
]


class TicketClassificationFailureAnalysisError(ValueError):
    """Raised when a failure-analysis artifact cannot be trusted."""


class TicketClassificationFailureEvidenceKind(StrEnum):
    """Evidence authority behind one failure-analysis observation."""

    PROVIDER_OBSERVATION = "provider_observation"
    STATIC_FIXTURE = "static_fixture"
    DATASET_DESIGN_HYPOTHESIS = "dataset_design_hypothesis"


class TicketClassificationFailureType(StrEnum):
    """Supported classification failure taxonomy."""

    CATEGORY_CONFUSION = "category_confusion"
    INTENT_CONFUSION = "intent_confusion"
    URGENCY_UNDER_CLASSIFICATION = "urgency_under_classification"
    URGENCY_OVER_CLASSIFICATION = "urgency_over_classification"
    SENTIMENT_MISMATCH = "sentiment_mismatch"
    HUMAN_REVIEW_FALSE_NEGATIVE = "human_review_false_negative"
    HUMAN_REVIEW_FALSE_POSITIVE = "human_review_false_positive"
    STRUCTURED_OUTPUT_FAILURE = "structured_output_failure"
    PROVIDER_FAILURE = "provider_failure"
    MISSING_PREDICTION = "missing_prediction"
    AMBIGUOUS_INPUT = "ambiguous_input"
    CROSS_LABEL_INTERACTION = "cross_label_interaction"


class TicketClassificationFailureSafetyImpact(StrEnum):
    """Qualitative safety consequence of one failure class."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketClassificationFailureObservation(BaseModel):
    """One grouped classification failure or risk observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    failure_type: TicketClassificationFailureType
    evidence_kind: TicketClassificationFailureEvidenceKind
    title: FailureAnalysisText
    description: FailureAnalysisText
    affected_case_ids: tuple[EvaluationCaseId, ...] = Field(min_length=1)
    metric_impacts: tuple[FailureAnalysisMetricName, ...] = Field(min_length=1)
    safety_impact: TicketClassificationFailureSafetyImpact
    prompt_only_remediation_appropriate: bool
    dataset_change_required: bool
    schema_change_required: bool
    rationale: FailureAnalysisText

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        if len(self.affected_case_ids) != len(set(self.affected_case_ids)):
            raise ValueError("Failure observation case IDs must be unique.")

        if len(self.metric_impacts) != len(set(self.metric_impacts)):
            raise ValueError("Failure observation metric impacts must be unique.")

        if self.prompt_only_remediation_appropriate and (
            self.dataset_change_required or self.schema_change_required
        ):
            raise ValueError("Prompt-only remediation cannot require dataset or schema changes.")

        return self


class TicketClassificationFailureEvidenceSummary(BaseModel):
    """Counts of observations by evidence authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_observation_count: int = Field(ge=0)
    static_fixture_observation_count: int = Field(ge=0)
    dataset_design_hypothesis_count: int = Field(ge=0)


class TicketClassificationFailureAnalysisContent(BaseModel):
    """Canonical failure-analysis content before hashing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_id: FailureAnalysisIdentifier
    analysis_version: int = Field(ge=1)
    schema_version: Literal["ticket-classification-failure-analysis-v1"]

    dataset_id: FailureAnalysisIdentifier
    dataset_version: int = Field(ge=1)
    dataset_content_hash: FailureAnalysisContentHash

    split_manifest_id: FailureAnalysisIdentifier
    split_manifest_version: int = Field(ge=1)
    split_manifest_content_hash: FailureAnalysisContentHash
    analyzed_split: Literal["development"]
    analyzed_case_ids: tuple[EvaluationCaseId, ...] = Field(min_length=1)

    prompt_id: FailureAnalysisIdentifier
    prompt_version: int = Field(ge=1)
    prompt_content_hash: FailureAnalysisContentHash

    evidence_summary: TicketClassificationFailureEvidenceSummary
    observations: tuple[TicketClassificationFailureObservation, ...] = Field(min_length=1)
    prompt_revision_constraints: tuple[FailureAnalysisText, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_analysis_content(self) -> Self:
        if len(self.analyzed_case_ids) != len(set(self.analyzed_case_ids)):
            raise ValueError("Analyzed case IDs must be unique.")

        failure_types = tuple(observation.failure_type for observation in self.observations)
        if len(failure_types) != len(set(failure_types)):
            raise ValueError("Failure analysis types must be unique.")

        analyzed_case_ids = set(self.analyzed_case_ids)
        for observation in self.observations:
            unknown_case_ids = set(observation.affected_case_ids) - analyzed_case_ids
            if unknown_case_ids:
                formatted_case_ids = ", ".join(sorted(unknown_case_ids))
                raise ValueError(
                    "Failure observations reference cases outside the analyzed "
                    f"split: {formatted_case_ids}."
                )

        expected_summary = TicketClassificationFailureEvidenceSummary(
            provider_observation_count=sum(
                observation.evidence_kind
                is TicketClassificationFailureEvidenceKind.PROVIDER_OBSERVATION
                for observation in self.observations
            ),
            static_fixture_observation_count=sum(
                observation.evidence_kind is TicketClassificationFailureEvidenceKind.STATIC_FIXTURE
                for observation in self.observations
            ),
            dataset_design_hypothesis_count=sum(
                observation.evidence_kind
                is TicketClassificationFailureEvidenceKind.DATASET_DESIGN_HYPOTHESIS
                for observation in self.observations
            ),
        )
        if self.evidence_summary != expected_summary:
            raise ValueError("Failure evidence summary does not match the observations.")

        if len(self.prompt_revision_constraints) != len(set(self.prompt_revision_constraints)):
            raise ValueError("Prompt revision constraints must be unique.")

        return self


class TicketClassificationFailureAnalysis(TicketClassificationFailureAnalysisContent):
    """Complete deterministic failure-analysis artifact."""

    analysis_content_hash: FailureAnalysisContentHash

    @model_validator(mode="after")
    def validate_content_hash(self) -> Self:
        content = TicketClassificationFailureAnalysisContent.model_validate(
            self.model_dump(
                mode="python",
                exclude={"analysis_content_hash"},
            )
        )
        if self.analysis_content_hash != sha256_hexdigest(content):
            raise ValueError("Failure analysis content hash does not match canonical content.")

        return self


def build_ticket_classification_failure_analysis(
    content: TicketClassificationFailureAnalysisContent,
) -> TicketClassificationFailureAnalysis:
    """Build a canonically hashed failure-analysis artifact."""

    return TicketClassificationFailureAnalysis(
        **content.model_dump(),
        analysis_content_hash=sha256_hexdigest(content),
    )


def validate_ticket_classification_failure_analysis_against_dataset(
    *,
    analysis: TicketClassificationFailureAnalysis,
    dataset: TicketClassificationEvaluationDataset,
) -> None:
    """Verify that analysis provenance and case identities match a dataset."""

    if analysis.dataset_id != dataset.dataset_id:
        raise TicketClassificationFailureAnalysisError(
            "Failure analysis dataset ID does not match the loaded dataset."
        )
    if analysis.dataset_version != dataset.version:
        raise TicketClassificationFailureAnalysisError(
            "Failure analysis dataset version does not match the loaded dataset."
        )
    if analysis.dataset_content_hash != dataset.content_hash:
        raise TicketClassificationFailureAnalysisError(
            "Failure analysis dataset hash does not match the loaded dataset."
        )

    dataset_case_ids = {case.case_id for case in dataset.cases}
    unknown_case_ids = set(analysis.analyzed_case_ids) - dataset_case_ids
    if unknown_case_ids:
        formatted_case_ids = ", ".join(sorted(unknown_case_ids))
        raise TicketClassificationFailureAnalysisError(
            f"Failure analysis references unknown dataset cases: {formatted_case_ids}."
        )


def load_ticket_classification_failure_analysis(
    path: Path,
) -> TicketClassificationFailureAnalysis:
    """Load and validate one committed failure-analysis artifact."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TicketClassificationFailureAnalysisError(
            "Ticket classification failure analysis could not be read."
        ) from error

    try:
        return TicketClassificationFailureAnalysis.model_validate(payload)
    except ValidationError as error:
        raise TicketClassificationFailureAnalysisError(
            "Ticket classification failure analysis does not match the contract."
        ) from error
