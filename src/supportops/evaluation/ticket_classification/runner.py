"""Sequential execution and artifact output for classification evaluation."""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from supportops.evaluation.ticket_classification.dataset import (
    TicketClassificationEvaluationDataset,
)
from supportops.evaluation.ticket_classification.evaluator import (
    TicketClassificationEvaluationReport,
    evaluate_ticket_classification_predictions,
)
from supportops.evaluation.ticket_classification.predictions import (
    TicketClassificationEvaluationPrediction,
    TicketClassificationPredictionSet,
    compute_ticket_classification_predictions_content_hash,
)
from supportops.evaluation.ticket_classification.predictor import (
    TicketClassificationEvaluationPredictor,
)


@dataclass(frozen=True, slots=True)
class TicketClassificationEvaluationRunResult:
    """Prediction and report artifacts from one sequential run."""

    predictions: TicketClassificationPredictionSet
    report: TicketClassificationEvaluationReport


async def run_ticket_classification_evaluation(
    *,
    dataset: TicketClassificationEvaluationDataset,
    predictor: TicketClassificationEvaluationPredictor,
) -> TicketClassificationEvaluationRunResult:
    """Predict every dataset case sequentially and evaluate the results."""

    predictions: list[TicketClassificationEvaluationPrediction] = []

    for case in dataset.cases:
        prediction = await predictor.predict(
            case=case,
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.version,
        )
        predictions.append(prediction)

    immutable_predictions = tuple(predictions)
    prediction_set = TicketClassificationPredictionSet(
        content_hash=(
            compute_ticket_classification_predictions_content_hash(
                immutable_predictions,
            )
        ),
        predictions=immutable_predictions,
    )
    report = evaluate_ticket_classification_predictions(
        dataset=dataset,
        predictions=prediction_set,
    )

    return TicketClassificationEvaluationRunResult(
        predictions=prediction_set,
        report=report,
    )


def write_ticket_classification_predictions(
    path: Path,
    predictions: TicketClassificationPredictionSet,
) -> None:
    """Write canonical JSONL predictions through an atomic replacement."""

    content = "".join(
        f"{_canonical_prediction_json(prediction)}\n" for prediction in predictions.predictions
    )

    _write_text_atomically(
        path=path,
        content=content,
    )


def write_ticket_classification_evaluation_report(
    path: Path,
    report: TicketClassificationEvaluationReport,
) -> None:
    """Write one deterministic human-readable JSON report atomically."""

    content = (
        json.dumps(
            report.model_dump(
                mode="json",
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    _write_text_atomically(
        path=path,
        content=content,
    )


def _canonical_prediction_json(
    prediction: TicketClassificationEvaluationPrediction,
) -> str:
    return json.dumps(
        prediction.model_dump(
            mode="json",
        ),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _write_text_atomically(
    *,
    path: Path,
    content: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(
                temporary_file.fileno(),
            )
            temporary_path = Path(
                temporary_file.name,
            )

        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

        raise
