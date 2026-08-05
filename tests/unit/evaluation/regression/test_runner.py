"""Unit tests for the repository regression runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import pytest

from supportops.evaluation.contracts.hashing import sha256_hexdigest
from supportops.evaluation.regression.models import (
    DOMAIN_CONTROLLED_SUPPORT,
    DOMAIN_HUMAN_APPROVAL,
    DOMAIN_SEMANTIC_RETRIEVAL,
    DOMAIN_TICKET_CLASSIFICATION,
    STABLE_DOMAIN_ORDER,
    RegressionAggregateStatus,
    RegressionDomainProfileResult,
    RepositoryRegressionResultContent,
)
from supportops.evaluation.regression.runner import (
    UnknownRegressionDomainError,
    run_repository_regression,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]


class _DefaultArtifactPaths(TypedDict):
    semantic_retrieval_dataset: Path
    semantic_retrieval_predictions: Path
    controlled_support_dataset: Path
    controlled_support_predictions: Path
    human_approval_dataset: Path
    human_approval_predictions: Path


def _default_paths() -> _DefaultArtifactPaths:
    return {
        "semantic_retrieval_dataset": (
            PROJECT_ROOT
            / "evals"
            / "semantic-retrieval"
            / "datasets"
            / "semantic-retrieval-eval-v1.jsonl"
        ),
        "semantic_retrieval_predictions": (
            PROJECT_ROOT
            / "evals"
            / "semantic-retrieval"
            / "predictions"
            / "semantic-retrieval-eval-v1.static.jsonl"
        ),
        "controlled_support_dataset": (
            PROJECT_ROOT
            / "evals"
            / "controlled-support"
            / "datasets"
            / "controlled-support-eval-v1.jsonl"
        ),
        "controlled_support_predictions": (
            PROJECT_ROOT
            / "evals"
            / "controlled-support"
            / "predictions"
            / "controlled-support-eval-v1.static.jsonl"
        ),
        "human_approval_dataset": (
            PROJECT_ROOT / "evals" / "human-approval" / "datasets" / "human-approval-eval-v1.jsonl"
        ),
        "human_approval_predictions": (
            PROJECT_ROOT
            / "evals"
            / "human-approval"
            / "predictions"
            / "human-approval-eval-v1.static.jsonl"
        ),
    }


def test_default_three_domain_execution_is_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    result = run_repository_regression()

    assert tuple(item.domain for item in result.domain_results) == (
        DOMAIN_SEMANTIC_RETRIEVAL,
        DOMAIN_CONTROLLED_SUPPORT,
        DOMAIN_HUMAN_APPROVAL,
    )
    assert result.not_provided_domains == (DOMAIN_TICKET_CLASSIFICATION,)
    assert result.status is RegressionAggregateStatus.INCOMPLETE
    assert result.blocking_failure_count == 0
    assert result.incomplete_domain_count == 3


def test_committed_fixtures_score_without_network() -> None:
    result = run_repository_regression(**_default_paths())

    assert result.status is RegressionAggregateStatus.INCOMPLETE
    assert all(
        domain_result.status is RegressionAggregateStatus.INCOMPLETE
        for domain_result in result.domain_results
    )


def test_stable_domain_order_and_repository_hash() -> None:
    first = run_repository_regression(**_default_paths())
    second = run_repository_regression(**_default_paths())

    assert tuple(item.domain for item in first.domain_results) == tuple(
        domain for domain in STABLE_DOMAIN_ORDER if domain != DOMAIN_TICKET_CLASSIFICATION
    )
    assert first.content_hash == second.content_hash
    content = RepositoryRegressionResultContent.model_validate(
        first.model_dump(exclude={"content_hash"})
    )
    assert first.content_hash == sha256_hexdigest(content)


def test_one_domain_failure_makes_repository_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _default_paths()
    original = run_repository_regression(**paths)
    failed = original.domain_results[0].model_copy(
        update={
            "status": RegressionAggregateStatus.FAILED,
            "blocking_failure_count": 1,
        }
    )

    def _fake_score_semantic_retrieval(**_kwargs: object) -> RegressionDomainProfileResult:
        return failed

    monkeypatch.setattr(
        "supportops.evaluation.regression.runner._score_semantic_retrieval",
        _fake_score_semantic_retrieval,
    )
    result = run_repository_regression(**paths)

    assert result.status is RegressionAggregateStatus.FAILED
    assert result.blocking_failure_count == 1


def test_optional_classification_omitted_and_recorded() -> None:
    result = run_repository_regression(
        domains=(DOMAIN_SEMANTIC_RETRIEVAL,),
        semantic_retrieval_dataset=_default_paths()["semantic_retrieval_dataset"],
        semantic_retrieval_predictions=_default_paths()["semantic_retrieval_predictions"],
    )

    assert result.not_provided_domains == (DOMAIN_TICKET_CLASSIFICATION,)
    assert DOMAIN_TICKET_CLASSIFICATION not in {item.domain for item in result.domain_results}


def test_selected_domain_execution() -> None:
    result = run_repository_regression(
        domains=(DOMAIN_HUMAN_APPROVAL, DOMAIN_SEMANTIC_RETRIEVAL),
        semantic_retrieval_dataset=_default_paths()["semantic_retrieval_dataset"],
        semantic_retrieval_predictions=_default_paths()["semantic_retrieval_predictions"],
        human_approval_dataset=_default_paths()["human_approval_dataset"],
        human_approval_predictions=_default_paths()["human_approval_predictions"],
    )

    assert tuple(item.domain for item in result.domain_results) == (
        DOMAIN_SEMANTIC_RETRIEVAL,
        DOMAIN_HUMAN_APPROVAL,
    )


def test_unknown_domain_rejected() -> None:
    with pytest.raises(UnknownRegressionDomainError):
        run_repository_regression(domains=("unknown-domain",))


def test_artifact_validation_failure_does_not_overwrite_output(tmp_path: Path) -> None:
    output_path = tmp_path / "repository-regression.json"
    output_path.write_text('{"preserved":true}\n', encoding="utf-8")
    original = output_path.read_text(encoding="utf-8")

    with pytest.raises(OSError):
        run_repository_regression(
            domains=(DOMAIN_SEMANTIC_RETRIEVAL,),
            semantic_retrieval_dataset=tmp_path / "missing-dataset.jsonl",
            semantic_retrieval_predictions=tmp_path / "missing-predictions.jsonl",
            output_path=output_path,
        )

    assert output_path.read_text(encoding="utf-8") == original


def test_successful_output_is_canonical_and_atomic(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "repository-regression.json"
    result = run_repository_regression(
        domains=(DOMAIN_SEMANTIC_RETRIEVAL,),
        semantic_retrieval_dataset=_default_paths()["semantic_retrieval_dataset"],
        semantic_retrieval_predictions=_default_paths()["semantic_retrieval_predictions"],
        output_path=output_path,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["content_hash"] == result.content_hash
    assert payload["status"] == RegressionAggregateStatus.INCOMPLETE.value
    assert "created_at" not in payload
    assert list(payload.keys()) == sorted(payload.keys())
