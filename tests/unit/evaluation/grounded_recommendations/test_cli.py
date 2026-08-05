from __future__ import annotations

import socket
import tomllib
from decimal import Decimal
from io import StringIO
from pathlib import Path

import pytest

from supportops.evaluation.contracts.manifest import EvaluationRunStatus
from supportops.evaluation.grounded_recommendations import cli as grounded_cli
from supportops.evaluation.grounded_recommendations.cli import (
    build_parser,
    main,
    run_cli,
)
from supportops.evaluation.grounded_recommendations.ragas_adapter import (
    RagasEvaluationResult,
    RagasEvaluationSample,
    RagasMetricName,
    RagasMetricResult,
)
from supportops.evaluation.grounded_recommendations.runner import (
    DEFAULT_GROUNDED_DATASET_PATH,
    DEFAULT_GROUNDED_PREDICTIONS_PATH,
    DEFAULT_GROUNDED_RAGAS_SCORES_PATH,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
_PROMPT_HASH = "a" * 64
_GIT_COMMIT = "c" * 40


class _FakeRagasAdapter:
    @property
    def runtime_version(self) -> str:
        return "0.4.3"

    def evaluate(
        self,
        *,
        samples: tuple[RagasEvaluationSample, ...],
        metrics: tuple[RagasMetricName, ...],
    ) -> tuple[RagasEvaluationResult, ...]:
        return tuple(
            RagasEvaluationResult(
                case_id=sample.case_id,
                metrics=tuple(
                    RagasMetricResult(metric=metric, score=Decimal("0.9")) for metric in metrics
                ),
            )
            for sample in samples
        )


def _run_argv(tmp_path: Path, **overrides: str) -> list[str]:
    values = {
        "dataset": str(PROJECT_ROOT / DEFAULT_GROUNDED_DATASET_PATH),
        "predictions": str(PROJECT_ROOT / DEFAULT_GROUNDED_PREDICTIONS_PATH),
        "output_dir": str(tmp_path / "artifacts" / "cli-run"),
        "system_provider": "openai",
        "system_model": "system-model",
        "evaluator_provider": "openai",
        "evaluator_model": "evaluator-model",
        "evaluator_embedding_model": "text-embedding-3-small",
        "prompt_id": "grounded-recommendation",
        "prompt_version": "1",
        "prompt_hash": _PROMPT_HASH,
        "git_commit": _GIT_COMMIT,
    }
    values.update(overrides)
    argv = [
        "run",
        "--dataset",
        values["dataset"],
        "--predictions",
        values["predictions"],
        "--output-dir",
        values["output_dir"],
        "--allow-external-provider",
        "--system-provider",
        values["system_provider"],
        "--system-model",
        values["system_model"],
        "--evaluator-provider",
        values["evaluator_provider"],
        "--evaluator-model",
        values["evaluator_model"],
        "--evaluator-embedding-model",
        values["evaluator_embedding_model"],
        "--prompt-id",
        values["prompt_id"],
        "--prompt-version",
        values["prompt_version"],
        "--prompt-hash",
        values["prompt_hash"],
        "--git-commit",
        values["git_commit"],
    ]
    return argv


def test_validate_succeeds_with_default_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = run_cli(["validate"], stdout=stdout, stderr=stderr)

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert "status=valid" in stdout.getvalue()
    assert "case_count=14" in stdout.getvalue()


def test_validate_is_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(PROJECT_ROOT)

    def _deny_network(*_args: object, **_kwargs: object) -> None:
        raise OSError("network access is forbidden during validate")

    monkeypatch.setattr(socket, "create_connection", _deny_network)

    exit_code = run_cli(["validate"], stdout=StringIO(), stderr=StringIO())
    assert exit_code == 0


def test_score_succeeds_with_committed_static_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    stdout = StringIO()

    exit_code = run_cli(["score"], stdout=stdout, stderr=StringIO())

    assert exit_code == 0
    assert "status=ok" in stdout.getvalue()
    assert "case_count=14" in stdout.getvalue()


def test_score_is_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(PROJECT_ROOT)

    def _deny_network(*_args: object, **_kwargs: object) -> None:
        raise OSError("network access is forbidden during score")

    monkeypatch.setattr(socket, "create_connection", _deny_network)

    exit_code = run_cli(["score"], stdout=StringIO(), stderr=StringIO())
    assert exit_code == 0


def test_score_with_ragas_scores_aggregates_static_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    stdout = StringIO()

    exit_code = run_cli(
        [
            "score",
            "--ragas-scores",
            str(DEFAULT_GROUNDED_RAGAS_SCORES_PATH),
        ],
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    assert "ragas_report" in stdout.getvalue()
    assert "scored_case_count=14" in stdout.getvalue()


def test_validate_and_score_expose_no_provider_flags() -> None:
    parser = build_parser()
    validate_help = None
    score_help = None
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):
            validate_help = choices["validate"].format_help()
            score_help = choices["score"].format_help()
            break

    assert validate_help is not None
    assert score_help is not None
    for help_text in (validate_help, score_help):
        assert "--allow-external-provider" not in help_text
        assert "--evaluator-provider" not in help_text
        assert "--system-provider" not in help_text
        assert "SUPPORTOPS_EVALUATION_OPENAI_API_KEY" not in help_text


def test_run_without_acknowledgement_exits_4(tmp_path: Path) -> None:
    argv = _run_argv(tmp_path)
    argv.remove("--allow-external-provider")
    stderr = StringIO()

    exit_code = run_cli(argv, stdout=StringIO(), stderr=stderr)

    assert exit_code == 4
    assert "allow-external-provider" in stderr.getvalue()


def test_run_without_evaluation_api_key_fails_without_printing_secrets(
    tmp_path: Path,
) -> None:
    stderr = StringIO()
    exit_code = run_cli(
        _run_argv(tmp_path),
        stdout=StringIO(),
        stderr=stderr,
        environ={},
    )

    assert exit_code == 3
    message = stderr.getvalue()
    assert "SUPPORTOPS_EVALUATION_OPENAI_API_KEY" in message
    assert "sk-" not in message


def test_unsupported_evaluator_provider_fails(tmp_path: Path) -> None:
    stderr = StringIO()
    exit_code = run_cli(
        _run_argv(tmp_path, evaluator_provider="anthropic"),
        stdout=StringIO(),
        stderr=stderr,
        environ={"SUPPORTOPS_EVALUATION_OPENAI_API_KEY": "test-key"},
    )

    assert exit_code == 3
    assert "openai" in stderr.getvalue()


def test_fake_backed_run_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        grounded_cli,
        "OpenAIRagasAdapter",
        lambda **_kwargs: _FakeRagasAdapter(),
    )
    stdout = StringIO()

    exit_code = run_cli(
        _run_argv(tmp_path),
        stdout=stdout,
        stderr=StringIO(),
        environ={"SUPPORTOPS_EVALUATION_OPENAI_API_KEY": "test-key"},
    )

    assert exit_code == 0
    assert f"status={EvaluationRunStatus.COMPLETE.value}" in stdout.getvalue()
    assert (tmp_path / "artifacts" / "cli-run" / "manifest.json").exists()


def test_same_model_warning_goes_to_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        grounded_cli,
        "OpenAIRagasAdapter",
        lambda **_kwargs: _FakeRagasAdapter(),
    )
    stderr = StringIO()

    exit_code = run_cli(
        _run_argv(
            tmp_path,
            system_model="shared-model",
            evaluator_model="shared-model",
        ),
        stdout=StringIO(),
        stderr=stderr,
        environ={"SUPPORTOPS_EVALUATION_OPENAI_API_KEY": "test-key"},
    )

    assert exit_code == 0
    assert "system model and evaluator model identities are equal" in stderr.getvalue()


def test_malformed_artifact_exits_3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    bad_dataset = tmp_path / "bad.jsonl"
    bad_dataset.write_text("{not-json\n", encoding="utf-8")
    stderr = StringIO()

    exit_code = run_cli(
        ["validate", "--dataset", str(bad_dataset)],
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 3
    assert stderr.getvalue().strip()


def test_script_entry_point_calls_main() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    assert (
        scripts["supportops-evaluate-grounded-recommendations"]
        == "supportops.evaluation.grounded_recommendations.cli:main"
    )
    assert callable(main)


def test_main_raises_system_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    with pytest.raises(SystemExit) as exit_info:
        main(["validate"])
    assert exit_info.value.code == 0
