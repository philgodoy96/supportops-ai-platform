from __future__ import annotations

from decimal import Decimal
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from supportops.evaluation.grounded_recommendations.ragas_adapter import (
    PINNED_RAGAS_VERSION,
    RagasEvaluationSample,
    RagasMetricName,
    RagasRuntime,
)
from supportops.evaluation.grounded_recommendations.ragas_execution import (
    OpenAIRagasAdapter,
    OpenAIRagasConfiguration,
)


class _FakeMetricResult:
    def __init__(self, value: object) -> None:
        self.value = value


class _RecordingMetric:
    def __init__(
        self,
        name: str,
        *,
        value: object = "0.75",
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def ascore(self, **kwargs: Any) -> _FakeMetricResult:
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return _FakeMetricResult(self.value)


def _sample() -> RagasEvaluationSample:
    return RagasEvaluationSample(
        case_id="grounded-recommendation-fully-grounded-001",
        question="Reset MFA\nUser cannot reset MFA.",
        response="Follow the MFA reset procedure.",
        contexts=("MFA resets require verified ownership.",),
        reference_answer="Guide the user through MFA reset.",
        reference_claims=("MFA resets require verified ownership.",),
    )


def _build_adapter(
    monkeypatch: pytest.MonkeyPatch,
    *,
    metrics: dict[RagasMetricName, _RecordingMetric] | None = None,
) -> tuple[OpenAIRagasAdapter, dict[RagasMetricName, _RecordingMetric]]:
    recorded = metrics or {
        RagasMetricName.FAITHFULNESS: _RecordingMetric("faithfulness", value="0.91"),
        RagasMetricName.ANSWER_RELEVANCY: _RecordingMetric(
            "answer_relevancy",
            value=0.82,
        ),
        RagasMetricName.CONTEXT_PRECISION: _RecordingMetric(
            "context_precision",
            value="0.77",
        ),
        RagasMetricName.CONTEXT_RECALL: _RecordingMetric(
            "context_recall",
            value="0.66",
        ),
    }

    openai_module = ModuleType("openai")

    class _AsyncOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    openai_module.AsyncOpenAI = _AsyncOpenAI  # type: ignore[attr-defined]

    llms_module = ModuleType("ragas.llms")
    embeddings_module = ModuleType("ragas.embeddings.base")
    collections_module = ModuleType("ragas.metrics.collections")

    constructed: list[str] = []

    def llm_factory(*args: object, **kwargs: object) -> object:
        constructed.append("llm")
        return SimpleNamespace(args=args, kwargs=kwargs)

    def embedding_factory(*args: object, **kwargs: object) -> object:
        constructed.append("embeddings")
        return SimpleNamespace(args=args, kwargs=kwargs)

    llms_module.llm_factory = llm_factory  # type: ignore[attr-defined]
    embeddings_module.embedding_factory = embedding_factory  # type: ignore[attr-defined]

    collections_module.Faithfulness = (  # type: ignore[attr-defined]
        lambda **kwargs: recorded[RagasMetricName.FAITHFULNESS]
    )
    collections_module.AnswerRelevancy = (  # type: ignore[attr-defined]
        lambda **kwargs: recorded[RagasMetricName.ANSWER_RELEVANCY]
    )
    collections_module.ContextPrecision = (  # type: ignore[attr-defined]
        lambda **kwargs: recorded[RagasMetricName.CONTEXT_PRECISION]
    )
    collections_module.ContextRecall = (  # type: ignore[attr-defined]
        lambda **kwargs: recorded[RagasMetricName.CONTEXT_RECALL]
    )

    modules = {
        "openai": openai_module,
        "ragas.llms": llms_module,
        "ragas.embeddings.base": embeddings_module,
        "ragas.metrics.collections": collections_module,
    }

    def import_module(name: str) -> ModuleType:
        if name not in modules:
            raise ImportError(name)
        return modules[name]

    monkeypatch.setattr(
        "supportops.evaluation.grounded_recommendations.ragas_execution._import_module",
        import_module,
    )
    monkeypatch.setattr(
        "supportops.evaluation.grounded_recommendations.ragas_execution.load_ragas_runtime",
        lambda: RagasRuntime(
            package_version=PINNED_RAGAS_VERSION,
            module=ModuleType("ragas"),
        ),
    )

    adapter = OpenAIRagasAdapter(
        configuration=OpenAIRagasConfiguration(
            evaluator_model="gpt-4.1-mini",
            evaluator_embedding_model="text-embedding-3-small",
        ),
        api_key="test-key",
    )
    adapter._constructed = constructed  # type: ignore[attr-defined]
    return adapter, recorded


def test_runtime_version_is_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _ = _build_adapter(monkeypatch)
    assert adapter.runtime_version == PINNED_RAGAS_VERSION
    assert adapter.runtime_version == "0.4.3"


def test_adapter_construction_does_not_import_ragas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def deny_import(name: str) -> ModuleType:
        calls.append(name)
        raise AssertionError(f"unexpected import during construction: {name}")

    monkeypatch.setattr(
        "supportops.evaluation.grounded_recommendations.ragas_execution._import_module",
        deny_import,
    )

    adapter = OpenAIRagasAdapter(
        configuration=OpenAIRagasConfiguration(
            evaluator_model="gpt-4.1-mini",
            evaluator_embedding_model="text-embedding-3-small",
        ),
        api_key="test-key",
    )

    assert adapter is not None
    assert calls == []


def test_requested_metric_order_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, recorded = _build_adapter(monkeypatch)
    metrics = (
        RagasMetricName.CONTEXT_RECALL,
        RagasMetricName.FAITHFULNESS,
        RagasMetricName.ANSWER_RELEVANCY,
        RagasMetricName.CONTEXT_PRECISION,
    )

    results = adapter.evaluate(samples=(_sample(),), metrics=metrics)

    assert tuple(metric.metric for metric in results[0].metrics) == metrics
    assert all(metric.score is not None for metric in results[0].metrics)
    assert recorded[RagasMetricName.FAITHFULNESS].calls
    assert recorded[RagasMetricName.CONTEXT_RECALL].calls


def test_metric_result_value_converts_to_decimal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _ = _build_adapter(monkeypatch)
    results = adapter.evaluate(
        samples=(_sample(),),
        metrics=(RagasMetricName.ANSWER_RELEVANCY,),
    )

    assert results[0].metrics[0].score == Decimal("0.82")


def test_faithfulness_input_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, recorded = _build_adapter(monkeypatch)
    sample = _sample()

    adapter.evaluate(
        samples=(sample,),
        metrics=(RagasMetricName.FAITHFULNESS,),
    )

    assert recorded[RagasMetricName.FAITHFULNESS].calls == [
        {
            "user_input": sample.question,
            "response": sample.response,
            "retrieved_contexts": list(sample.contexts),
        }
    ]


def test_answer_relevancy_input_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, recorded = _build_adapter(monkeypatch)
    sample = _sample()

    adapter.evaluate(
        samples=(sample,),
        metrics=(RagasMetricName.ANSWER_RELEVANCY,),
    )

    assert recorded[RagasMetricName.ANSWER_RELEVANCY].calls == [
        {
            "user_input": sample.question,
            "response": sample.response,
        }
    ]


def test_context_precision_input_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, recorded = _build_adapter(monkeypatch)
    sample = _sample()

    adapter.evaluate(
        samples=(sample,),
        metrics=(RagasMetricName.CONTEXT_PRECISION,),
    )

    assert recorded[RagasMetricName.CONTEXT_PRECISION].calls == [
        {
            "user_input": sample.question,
            "reference": sample.reference_answer,
            "retrieved_contexts": list(sample.contexts),
        }
    ]


def test_context_recall_input_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, recorded = _build_adapter(monkeypatch)
    sample = _sample()

    adapter.evaluate(
        samples=(sample,),
        metrics=(RagasMetricName.CONTEXT_RECALL,),
    )

    assert recorded[RagasMetricName.CONTEXT_RECALL].calls == [
        {
            "user_input": sample.question,
            "retrieved_contexts": list(sample.contexts),
            "reference": sample.reference_answer,
        }
    ]


def test_one_metric_failure_does_not_drop_other_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = {
        RagasMetricName.FAITHFULNESS: _RecordingMetric(
            "faithfulness",
            error=RuntimeError("provider boom with secret sk-secret"),
        ),
        RagasMetricName.ANSWER_RELEVANCY: _RecordingMetric(
            "answer_relevancy",
            value="0.5",
        ),
        RagasMetricName.CONTEXT_PRECISION: _RecordingMetric(
            "context_precision",
            value="0.4",
        ),
        RagasMetricName.CONTEXT_RECALL: _RecordingMetric(
            "context_recall",
            value="0.3",
        ),
    }
    adapter, _ = _build_adapter(monkeypatch, metrics=metrics)

    results = adapter.evaluate(
        samples=(_sample(),),
        metrics=tuple(RagasMetricName),
    )

    by_name = {metric.metric: metric for metric in results[0].metrics}
    assert by_name[RagasMetricName.FAITHFULNESS].score is None
    assert by_name[RagasMetricName.FAITHFULNESS].error_code == ("ragas_faithfulness_failed")
    assert by_name[RagasMetricName.ANSWER_RELEVANCY].score == Decimal("0.5")
    assert by_name[RagasMetricName.CONTEXT_PRECISION].score == Decimal("0.4")
    assert by_name[RagasMetricName.CONTEXT_RECALL].score == Decimal("0.3")


def test_stable_error_codes_and_no_raw_response_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = {
        RagasMetricName.FAITHFULNESS: _RecordingMetric(
            "faithfulness",
            error=RuntimeError('{"choices":[{"message":{"content":"secret"}}]}'),
        ),
        RagasMetricName.ANSWER_RELEVANCY: _RecordingMetric(
            "answer_relevancy",
            error=RuntimeError("sk-live-should-not-leak"),
        ),
        RagasMetricName.CONTEXT_PRECISION: _RecordingMetric(
            "context_precision",
            value="1",
        ),
        RagasMetricName.CONTEXT_RECALL: _RecordingMetric(
            "context_recall",
            value="1",
        ),
    }
    adapter, _ = _build_adapter(monkeypatch, metrics=metrics)

    results = adapter.evaluate(
        samples=(_sample(),),
        metrics=tuple(RagasMetricName),
    )

    serialized = repr(results)
    assert "sk-live-should-not-leak" not in serialized
    assert "choices" not in serialized
    assert results[0].metrics[0].error_code == "ragas_faithfulness_failed"
    assert results[0].metrics[1].error_code == "ragas_answer_relevancy_failed"
