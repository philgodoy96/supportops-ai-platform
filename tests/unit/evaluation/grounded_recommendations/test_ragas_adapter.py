from decimal import Decimal
from types import ModuleType

import pytest

from supportops.evaluation.grounded_recommendations.ragas_adapter import (
    PINNED_RAGAS_VERSION,
    RagasDependencyError,
    RagasEvaluationResult,
    RagasEvaluationSample,
    RagasMetricName,
    RagasMetricResult,
    load_ragas_runtime,
)


def test_load_ragas_runtime_returns_verified_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = ModuleType("ragas")

    monkeypatch.setattr(
        "supportops.evaluation.grounded_recommendations.ragas_adapter.importlib.metadata.version",
        lambda package_name: PINNED_RAGAS_VERSION,
    )
    monkeypatch.setattr(
        "supportops.evaluation.grounded_recommendations.ragas_adapter.importlib.import_module",
        lambda package_name: fake_module,
    )

    runtime = load_ragas_runtime()

    assert runtime.package_version == PINNED_RAGAS_VERSION
    assert runtime.module is fake_module


def test_load_ragas_runtime_rejects_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_missing(package_name: str) -> str:
        raise __import__(
            "importlib.metadata",
            fromlist=["PackageNotFoundError"],
        ).PackageNotFoundError(package_name)

    monkeypatch.setattr(
        "supportops.evaluation.grounded_recommendations.ragas_adapter.importlib.metadata.version",
        raise_missing,
    )

    with pytest.raises(
        RagasDependencyError,
        match="evaluation dependency group",
    ):
        load_ragas_runtime()


def test_load_ragas_runtime_rejects_unexpected_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "supportops.evaluation.grounded_recommendations.ragas_adapter.importlib.metadata.version",
        lambda package_name: "0.0.0",
    )

    with pytest.raises(
        RagasDependencyError,
        match="unexpected RAGAS version",
    ):
        load_ragas_runtime()


def test_load_ragas_runtime_wraps_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "supportops.evaluation.grounded_recommendations.ragas_adapter.importlib.metadata.version",
        lambda package_name: PINNED_RAGAS_VERSION,
    )

    def raise_import_error(package_name: str) -> ModuleType:
        raise ImportError(package_name)

    monkeypatch.setattr(
        "supportops.evaluation.grounded_recommendations.ragas_adapter.importlib.import_module",
        raise_import_error,
    )

    with pytest.raises(
        RagasDependencyError,
        match="could not be imported",
    ):
        load_ragas_runtime()


def test_evaluation_sample_preserves_provider_neutral_inputs() -> None:
    sample = RagasEvaluationSample(
        case_id="grounded-recommendation-001",
        question="How should the incident be handled?",
        response="Follow the documented escalation procedure.",
        contexts=("Escalate severity-one incidents immediately.",),
        reference_answer="Escalate the incident immediately.",
        reference_claims=("Severity-one incidents require escalation.",),
    )

    assert sample.case_id == "grounded-recommendation-001"
    assert len(sample.contexts) == 1
    assert len(sample.reference_claims) == 1


def test_metric_result_accepts_normalized_score() -> None:
    result = RagasMetricResult(
        metric=RagasMetricName.FAITHFULNESS,
        score=Decimal("0.875"),
    )

    assert result.score == Decimal("0.875")
    assert result.error_code is None


def test_metric_result_accepts_explicit_failure() -> None:
    result = RagasMetricResult(
        metric=RagasMetricName.ANSWER_RELEVANCY,
        score=None,
        error_code="evaluator_timeout",
    )

    assert result.score is None
    assert result.error_code == "evaluator_timeout"


@pytest.mark.parametrize(
    ("score", "error_code"),
    [
        (None, None),
        (Decimal("0.5"), "unexpected_error"),
        (Decimal("-0.1"), None),
        (Decimal("1.1"), None),
    ],
)
def test_metric_result_rejects_invalid_state(
    score: Decimal | None,
    error_code: str | None,
) -> None:
    with pytest.raises(ValueError):
        RagasMetricResult(
            metric=RagasMetricName.CONTEXT_PRECISION,
            score=score,
            error_code=error_code,
        )


def test_case_result_rejects_duplicate_metrics() -> None:
    metric = RagasMetricResult(
        metric=RagasMetricName.CONTEXT_RECALL,
        score=Decimal("1"),
    )

    with pytest.raises(
        ValueError,
        match="duplicate metric names",
    ):
        RagasEvaluationResult(
            case_id="grounded-recommendation-001",
            metrics=(metric, metric),
        )
