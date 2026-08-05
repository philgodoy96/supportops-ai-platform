from pathlib import Path

import pytest

from supportops.evaluation.grounded_recommendations.predictions import (
    GroundedRecommendationPredictionError,
    load_grounded_recommendation_predictions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "evals"
    / "grounded-recommendations"
    / "predictions"
    / "grounded-recommendations-eval-v1.static.jsonl"
)

PREDICTION_HASH = "a91baea6243c3edcce33e8004e79b8d1c709e3e84c83bd1d1e47c324edee2b72"


def test_committed_predictions_are_complete_and_immutable() -> None:
    predictions, prediction_hash = load_grounded_recommendation_predictions(PREDICTIONS_PATH)

    assert len(predictions) == 14
    assert len({prediction.case_id for prediction in predictions}) == 14
    assert prediction_hash == PREDICTION_HASH


def test_prediction_payloads_are_structured() -> None:
    predictions, _ = load_grounded_recommendation_predictions(PREDICTIONS_PATH)

    for prediction in predictions:
        assert prediction.payload is not None
        assert prediction.payload.response_text
        assert prediction.payload.prompt_id
        assert prediction.payload.prompt_version == 1
        assert prediction.payload.schema_version == "support-recommendation-v1"


def test_duplicate_prediction_case_id_is_rejected(
    tmp_path: Path,
) -> None:
    lines = PREDICTIONS_PATH.read_text(encoding="utf-8").splitlines()

    duplicate_file = tmp_path / "duplicate.jsonl"
    duplicate_file.write_text(
        "\n".join([lines[0], lines[0]]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        GroundedRecommendationPredictionError,
        match="duplicate prediction case_id",
    ):
        load_grounded_recommendation_predictions(duplicate_file)


def test_empty_prediction_file_is_rejected(
    tmp_path: Path,
) -> None:
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(
        GroundedRecommendationPredictionError,
        match="prediction set must not be empty",
    ):
        load_grounded_recommendation_predictions(empty_file)
