"""Shared vocabulary and result types for repository regression gates."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from supportops.evaluation.contracts.hashing import sha256_hexdigest

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
Sha256Hex = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]

REPOSITORY_REGRESSION_SCHEMA_VERSION = "1"

DOMAIN_TICKET_CLASSIFICATION = "ticket-classification"
DOMAIN_SEMANTIC_RETRIEVAL = "semantic-retrieval"
DOMAIN_CONTROLLED_SUPPORT = "controlled-support"
DOMAIN_HUMAN_APPROVAL = "human-approval"

STABLE_DOMAIN_ORDER: tuple[str, ...] = (
    DOMAIN_TICKET_CLASSIFICATION,
    DOMAIN_SEMANTIC_RETRIEVAL,
    DOMAIN_CONTROLLED_SUPPORT,
    DOMAIN_HUMAN_APPROVAL,
)

SUPPORTED_REGRESSION_DOMAINS: frozenset[str] = frozenset(STABLE_DOMAIN_ORDER)


class RegressionGateCategory(StrEnum):
    """Release-gate category for repository regression profiles."""

    SAFETY = "safety"
    RELIABILITY = "reliability"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"


class RegressionGateOutcome(StrEnum):
    """Explicit outcome for one regression gate."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class RegressionGateOperator(StrEnum):
    """Supported comparison operators for absolute regression gates."""

    EQUAL = "equal"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"


class RegressionAggregateStatus(StrEnum):
    """Aggregate status for a domain profile or repository result."""

    PASSED = "passed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class RegressionGateResult(BaseModel):
    """Deterministic result for one domain release gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gate_id: NonEmptyString
    domain: NonEmptyString
    category: RegressionGateCategory
    outcome: RegressionGateOutcome
    blocking: bool
    metric_name: NonEmptyString
    operator: RegressionGateOperator
    actual_value: Decimal | int | None
    threshold_value: Decimal | int | None
    reason: NonEmptyString


class RegressionDomainProfileResultContent(BaseModel):
    """Reproducible domain gate-profile content before hashing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: NonEmptyString
    profile_version: int = Field(ge=1)
    domain: NonEmptyString
    source_report_hash: Sha256Hex
    gate_results: tuple[RegressionGateResult, ...]
    blocking_failure_count: int = Field(ge=0)
    not_applicable_count: int = Field(ge=0)
    status: RegressionAggregateStatus


class RegressionDomainProfileResult(RegressionDomainProfileResultContent):
    """Complete deterministic domain gate-profile evaluation."""

    content_hash: Sha256Hex


class RepositoryRegressionResultContent(BaseModel):
    """Reproducible repository regression content before hashing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: NonEmptyString
    domain_results: tuple[RegressionDomainProfileResult, ...]
    not_provided_domains: tuple[NonEmptyString, ...]
    blocking_failure_count: int = Field(ge=0)
    incomplete_domain_count: int = Field(ge=0)
    status: RegressionAggregateStatus

    @model_validator(mode="after")
    def validate_domain_ordering(self) -> Self:
        domain_order = {domain: index for index, domain in enumerate(STABLE_DOMAIN_ORDER)}
        domains = tuple(result.domain for result in self.domain_results)
        if len(domains) != len(set(domains)):
            raise ValueError("domain_results contains duplicate domains")

        if any(domain not in domain_order for domain in domains):
            raise ValueError("domain_results contains an unsupported domain")

        ordered = tuple(sorted(domains, key=lambda domain: domain_order[domain]))
        if domains != ordered:
            raise ValueError("domain_results must follow stable domain order")

        not_provided = self.not_provided_domains
        if len(not_provided) != len(set(not_provided)):
            raise ValueError("not_provided_domains contains duplicates")

        if any(domain not in domain_order for domain in not_provided):
            raise ValueError("not_provided_domains contains an unsupported domain")

        ordered_not_provided = tuple(sorted(not_provided, key=lambda domain: domain_order[domain]))
        if not_provided != ordered_not_provided:
            raise ValueError("not_provided_domains must follow stable domain order")

        overlap = set(domains) & set(not_provided)
        if overlap:
            raise ValueError("domains cannot be both supplied and not provided")

        return self


class RepositoryRegressionResult(RepositoryRegressionResultContent):
    """Complete deterministic repository regression evaluation."""

    content_hash: Sha256Hex


def build_domain_profile_result(
    *,
    profile_id: str,
    profile_version: int,
    domain: str,
    source_report_hash: str,
    gate_results: tuple[RegressionGateResult, ...],
) -> RegressionDomainProfileResult:
    """Aggregate gate outcomes into one hashed domain profile result."""

    blocking_failure_count = sum(
        1
        for result in gate_results
        if result.blocking and result.outcome is RegressionGateOutcome.FAILED
    )
    not_applicable_count = sum(
        1 for result in gate_results if result.outcome is RegressionGateOutcome.NOT_APPLICABLE
    )
    status = aggregate_domain_status(gate_results)

    content = RegressionDomainProfileResultContent(
        profile_id=profile_id,
        profile_version=profile_version,
        domain=domain,
        source_report_hash=source_report_hash,
        gate_results=gate_results,
        blocking_failure_count=blocking_failure_count,
        not_applicable_count=not_applicable_count,
        status=status,
    )
    return RegressionDomainProfileResult(
        **content.model_dump(),
        content_hash=sha256_hexdigest(content),
    )


def build_repository_regression_result(
    *,
    domain_results: tuple[RegressionDomainProfileResult, ...],
    not_provided_domains: tuple[str, ...] = (),
) -> RepositoryRegressionResult:
    """Aggregate supplied domain profiles into one hashed repository result."""

    if not domain_results:
        raise ValueError("at least one domain result must be supplied")

    domain_order = {domain: index for index, domain in enumerate(STABLE_DOMAIN_ORDER)}
    ordered_domain_results = tuple(
        sorted(
            domain_results,
            key=lambda result: domain_order[result.domain],
        )
    )
    ordered_not_provided = tuple(
        sorted(
            not_provided_domains,
            key=lambda domain: domain_order[domain],
        )
    )

    blocking_failure_count = sum(
        1 for result in ordered_domain_results if result.status is RegressionAggregateStatus.FAILED
    )
    incomplete_domain_count = sum(
        1
        for result in ordered_domain_results
        if result.status is RegressionAggregateStatus.INCOMPLETE
    )
    status = aggregate_repository_status(ordered_domain_results)

    content = RepositoryRegressionResultContent(
        schema_version=REPOSITORY_REGRESSION_SCHEMA_VERSION,
        domain_results=ordered_domain_results,
        not_provided_domains=ordered_not_provided,
        blocking_failure_count=blocking_failure_count,
        incomplete_domain_count=incomplete_domain_count,
        status=status,
    )
    return RepositoryRegressionResult(
        **content.model_dump(),
        content_hash=sha256_hexdigest(content),
    )


def aggregate_domain_status(
    gate_results: tuple[RegressionGateResult, ...],
) -> RegressionAggregateStatus:
    """Derive domain status from blocking gate outcomes."""

    blocking_results = tuple(result for result in gate_results if result.blocking)

    if any(result.outcome is RegressionGateOutcome.FAILED for result in blocking_results):
        return RegressionAggregateStatus.FAILED

    if any(result.outcome is RegressionGateOutcome.NOT_APPLICABLE for result in blocking_results):
        return RegressionAggregateStatus.INCOMPLETE

    return RegressionAggregateStatus.PASSED


def aggregate_repository_status(
    domain_results: tuple[RegressionDomainProfileResult, ...],
) -> RegressionAggregateStatus:
    """Derive repository status from supplied domain profile statuses."""

    if any(result.status is RegressionAggregateStatus.FAILED for result in domain_results):
        return RegressionAggregateStatus.FAILED

    if any(result.status is RegressionAggregateStatus.INCOMPLETE for result in domain_results):
        return RegressionAggregateStatus.INCOMPLETE

    return RegressionAggregateStatus.PASSED
