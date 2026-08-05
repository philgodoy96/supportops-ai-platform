"""OpenAI-backed RAGAS execution for grounded recommendation evaluation."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from types import ModuleType
from typing import Any

from supportops.evaluation.grounded_recommendations.ragas_adapter import (
    RagasEvaluationResult,
    RagasEvaluationSample,
    RagasMetricName,
    RagasMetricResult,
    load_ragas_runtime,
)


def _import_module(name: str) -> ModuleType:
    """Import a module through a patchable boundary for tests."""

    return importlib.import_module(name)


_METRIC_ERROR_CODES: Mapping[RagasMetricName, str] = {
    RagasMetricName.FAITHFULNESS: "ragas_faithfulness_failed",
    RagasMetricName.ANSWER_RELEVANCY: "ragas_answer_relevancy_failed",
    RagasMetricName.CONTEXT_PRECISION: "ragas_context_precision_failed",
    RagasMetricName.CONTEXT_RECALL: "ragas_context_recall_failed",
}


class GroundedRecommendationRagasExecutionError(RuntimeError):
    """Raised when OpenAI-backed RAGAS execution cannot be prepared."""


@dataclass(frozen=True, slots=True)
class OpenAIRagasConfiguration:
    """Explicit OpenAI evaluator identities for one RAGAS run."""

    evaluator_model: str
    evaluator_embedding_model: str
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.evaluator_model.strip():
            raise ValueError("evaluator_model must not be empty")
        if not self.evaluator_embedding_model.strip():
            raise ValueError("evaluator_embedding_model must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


class OpenAIRagasAdapter:
    """Evaluation-only RAGAS adapter using explicit OpenAI clients."""

    def __init__(
        self,
        *,
        configuration: OpenAIRagasConfiguration,
        api_key: str,
    ) -> None:
        if not api_key.strip():
            raise GroundedRecommendationRagasExecutionError(
                "an OpenAI API key is required for RAGAS evaluation"
            )

        self._configuration = configuration
        self._api_key = api_key
        self._runtime_version: str | None = None

    @property
    def runtime_version(self) -> str:
        """Return the pinned RAGAS package version after lazy verification."""

        if self._runtime_version is None:
            runtime = load_ragas_runtime()
            self._runtime_version = runtime.package_version
        return self._runtime_version

    def evaluate(
        self,
        *,
        samples: tuple[RagasEvaluationSample, ...],
        metrics: tuple[RagasMetricName, ...],
    ) -> tuple[RagasEvaluationResult, ...]:
        """Evaluate samples with the requested metrics in stable order."""

        if not metrics:
            raise GroundedRecommendationRagasExecutionError(
                "at least one RAGAS metric must be requested"
            )

        if len(metrics) != len(set(metrics)):
            raise GroundedRecommendationRagasExecutionError(
                "requested RAGAS metrics must be unique"
            )

        # Ensure the pinned runtime is verified before importing RAGAS APIs.
        _ = self.runtime_version

        return asyncio.run(
            self._evaluate_async(
                samples=samples,
                metrics=metrics,
            )
        )

    async def _evaluate_async(
        self,
        *,
        samples: tuple[RagasEvaluationSample, ...],
        metrics: tuple[RagasMetricName, ...],
    ) -> tuple[RagasEvaluationResult, ...]:
        metric_instances = self._build_metric_instances(metrics)
        results: list[RagasEvaluationResult] = []

        for sample in samples:
            metric_results: list[RagasMetricResult] = []
            for metric_name in metrics:
                metric_results.append(
                    await self._score_metric(
                        metric_name=metric_name,
                        metric=metric_instances[metric_name],
                        sample=sample,
                    )
                )
            results.append(
                RagasEvaluationResult(
                    case_id=sample.case_id,
                    metrics=tuple(metric_results),
                )
            )

        return tuple(results)

    def _build_metric_instances(
        self,
        metrics: tuple[RagasMetricName, ...],
    ) -> dict[RagasMetricName, Any]:
        openai_module = _import_module("openai")
        llm_factory = _import_module("ragas.llms").llm_factory
        embedding_factory = _import_module("ragas.embeddings.base").embedding_factory
        collections = _import_module("ragas.metrics.collections")

        async_client = openai_module.AsyncOpenAI(
            api_key=self._api_key,
            timeout=self._configuration.timeout_seconds,
        )
        llm = llm_factory(
            self._configuration.evaluator_model,
            provider="openai",
            client=async_client,
        )
        embeddings = embedding_factory(
            "openai",
            model=self._configuration.evaluator_embedding_model,
            client=async_client,
            interface="modern",
        )

        instances: dict[RagasMetricName, Any] = {}
        for metric_name in metrics:
            if metric_name is RagasMetricName.FAITHFULNESS:
                instances[metric_name] = collections.Faithfulness(llm=llm)
            elif metric_name is RagasMetricName.ANSWER_RELEVANCY:
                instances[metric_name] = collections.AnswerRelevancy(
                    llm=llm,
                    embeddings=embeddings,
                )
            elif metric_name is RagasMetricName.CONTEXT_PRECISION:
                instances[metric_name] = collections.ContextPrecision(llm=llm)
            elif metric_name is RagasMetricName.CONTEXT_RECALL:
                instances[metric_name] = collections.ContextRecall(llm=llm)
            else:
                raise GroundedRecommendationRagasExecutionError(
                    f"unsupported RAGAS metric: {metric_name}"
                )

        return instances

    async def _score_metric(
        self,
        *,
        metric_name: RagasMetricName,
        metric: Any,
        sample: RagasEvaluationSample,
    ) -> RagasMetricResult:
        try:
            score_coro = _metric_ascore(
                metric_name=metric_name,
                metric=metric,
                sample=sample,
            )
            raw_result = await score_coro
            score = _normalize_metric_value(getattr(raw_result, "value", raw_result))
            return RagasMetricResult(
                metric=metric_name,
                score=score,
            )
        except Exception:
            return RagasMetricResult(
                metric=metric_name,
                score=None,
                error_code=_METRIC_ERROR_CODES[metric_name],
            )


def _metric_ascore(
    *,
    metric_name: RagasMetricName,
    metric: Any,
    sample: RagasEvaluationSample,
) -> Awaitable[Any]:
    ascore: Callable[..., Awaitable[Any]] = metric.ascore
    contexts = list(sample.contexts)

    if metric_name is RagasMetricName.FAITHFULNESS:
        return ascore(
            user_input=sample.question,
            response=sample.response,
            retrieved_contexts=contexts,
        )

    if metric_name is RagasMetricName.ANSWER_RELEVANCY:
        return ascore(
            user_input=sample.question,
            response=sample.response,
        )

    if metric_name is RagasMetricName.CONTEXT_PRECISION:
        if sample.reference_answer is None:
            raise ValueError("reference answer is required for context precision")
        return ascore(
            user_input=sample.question,
            reference=sample.reference_answer,
            retrieved_contexts=contexts,
        )

    if metric_name is RagasMetricName.CONTEXT_RECALL:
        if sample.reference_answer is None:
            raise ValueError("reference answer is required for context recall")
        return ascore(
            user_input=sample.question,
            retrieved_contexts=contexts,
            reference=sample.reference_answer,
        )

    raise GroundedRecommendationRagasExecutionError(f"unsupported RAGAS metric: {metric_name}")


def _normalize_metric_value(value: object) -> Decimal:
    score = Decimal(str(value))
    if not score.is_finite():
        raise ValueError("RAGAS metric value must be finite")
    return score


__all__ = [
    "GroundedRecommendationRagasExecutionError",
    "OpenAIRagasAdapter",
    "OpenAIRagasConfiguration",
]
