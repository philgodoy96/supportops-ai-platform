from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from types import ModuleType
from typing import Protocol, runtime_checkable

PINNED_RAGAS_VERSION = "0.4.3"


class RagasDependencyError(RuntimeError):
    """Raised when the isolated RAGAS dependency cannot be loaded."""


class RagasMetricName(StrEnum):
    """Grounded-recommendation metrics supported by the adapter boundary."""

    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"


@dataclass(frozen=True, slots=True)
class RagasEvaluationSample:
    """Provider-neutral input for one grounded recommendation evaluation."""

    case_id: str
    question: str
    response: str
    contexts: tuple[str, ...]
    reference_answer: str | None = None
    reference_claims: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RagasMetricResult:
    """One normalized metric result emitted by a RAGAS adapter."""

    metric: RagasMetricName
    score: Decimal | None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.score is None and self.error_code is None:
            raise ValueError("a metric result without a score must include an error code")

        if self.score is not None and self.error_code is not None:
            raise ValueError("a successful metric result cannot include an error code")

        if self.score is not None and not Decimal("0") <= self.score <= Decimal("1"):
            raise ValueError("RAGAS metric scores must be between zero and one")


@dataclass(frozen=True, slots=True)
class RagasEvaluationResult:
    """Normalized RAGAS results for one evaluation case."""

    case_id: str
    metrics: tuple[RagasMetricResult, ...]

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")

        metric_names = tuple(metric.metric for metric in self.metrics)
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("a case result cannot contain duplicate metric names")


@dataclass(frozen=True, slots=True)
class RagasRuntime:
    """Loaded and version-verified RAGAS runtime."""

    package_version: str
    module: ModuleType


@runtime_checkable
class RagasAdapter(Protocol):
    """Evaluation-only boundary for executing grounded RAGAS metrics."""

    @property
    def runtime_version(self) -> str:
        """Return the loaded RAGAS package version."""

    def evaluate(
        self,
        *,
        samples: tuple[RagasEvaluationSample, ...],
        metrics: tuple[RagasMetricName, ...],
    ) -> tuple[RagasEvaluationResult, ...]:
        """Evaluate samples and return normalized, provider-neutral results."""


def load_ragas_runtime(
    *,
    expected_version: str = PINNED_RAGAS_VERSION,
) -> RagasRuntime:
    """Load RAGAS lazily and enforce the repository-owned package pin."""

    try:
        installed_version = importlib.metadata.version("ragas")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RagasDependencyError(
            "RAGAS is not installed. Install the evaluation dependency group."
        ) from exc

    if installed_version != expected_version:
        raise RagasDependencyError(
            f"unexpected RAGAS version: expected {expected_version}, found {installed_version}"
        )

    try:
        ragas_module = importlib.import_module("ragas")
    except ImportError as exc:
        raise RagasDependencyError("the installed RAGAS package could not be imported") from exc

    return RagasRuntime(
        package_version=installed_version,
        module=ragas_module,
    )
