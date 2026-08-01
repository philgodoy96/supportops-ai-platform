"""Unit tests for evaluation-only environment settings."""

from typing import Any, cast

import pytest
from pydantic import ValidationError

from supportops.evaluation.ticket_classification.settings import (
    TicketClassificationEvaluationSettings,
)

_ENVIRONMENT_VARIABLES = (
    "SUPPORTOPS_OPENAI_API_KEY",
    "SUPPORTOPS_OPENAI_MODEL",
    "SUPPORTOPS_OPENAI_BASE_URL",
    "SUPPORTOPS_LLM_REQUEST_TIMEOUT_SECONDS",
    "SUPPORTOPS_LLM_TRANSPORT_MAX_RETRIES",
    "SUPPORTOPS_LLM_MAX_REPAIR_ATTEMPTS",
)


@pytest.fixture(autouse=True)
def isolate_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable_name in _ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(
            variable_name,
            raising=False,
        )


def _settings(
    **overrides: object,
) -> TicketClassificationEvaluationSettings:
    settings_type = cast(
        Any,
        TicketClassificationEvaluationSettings,
    )

    return cast(
        TicketClassificationEvaluationSettings,
        settings_type(
            _env_file=None,
            **overrides,
        ),
    )


def test_evaluation_settings_use_safe_defaults() -> None:
    settings = _settings()

    assert settings.openai_api_key is None
    assert settings.openai_model == "gpt-5-nano"
    assert settings.openai_base_url is None
    assert settings.llm_request_timeout_seconds == 12.0
    assert settings.llm_transport_max_retries == 1
    assert settings.llm_max_repair_attempts == 1


def test_evaluation_settings_do_not_require_infrastructure() -> None:
    settings = _settings()

    assert not hasattr(
        settings,
        "postgresql_url",
    )
    assert not hasattr(
        settings,
        "qdrant_url",
    )
    assert not hasattr(
        settings,
        "worker_lease_seconds",
    )


def test_evaluation_settings_load_llm_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SUPPORTOPS_OPENAI_API_KEY",
        "evaluation-key",
    )
    monkeypatch.setenv(
        "SUPPORTOPS_OPENAI_MODEL",
        "evaluation-model",
    )
    monkeypatch.setenv(
        "SUPPORTOPS_OPENAI_BASE_URL",
        "https://example.invalid/v1",
    )
    monkeypatch.setenv(
        "SUPPORTOPS_LLM_REQUEST_TIMEOUT_SECONDS",
        "20",
    )
    monkeypatch.setenv(
        "SUPPORTOPS_LLM_TRANSPORT_MAX_RETRIES",
        "2",
    )
    monkeypatch.setenv(
        "SUPPORTOPS_LLM_MAX_REPAIR_ATTEMPTS",
        "0",
    )

    settings = TicketClassificationEvaluationSettings(
        _env_file=None,
    )

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "evaluation-key"
    assert settings.openai_model == "evaluation-model"
    assert settings.openai_base_url == ("https://example.invalid/v1")
    assert settings.llm_request_timeout_seconds == 20
    assert settings.llm_transport_max_retries == 2
    assert settings.llm_max_repair_attempts == 0


def test_evaluation_settings_normalize_optional_openai_values() -> None:
    settings = _settings(
        openai_api_key="  evaluation-key  ",
        openai_base_url="  ",
    )

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "evaluation-key"
    assert settings.openai_base_url is None


@pytest.mark.parametrize(
    (
        "field_name",
        "value",
    ),
    [
        (
            "openai_model",
            " padded-model ",
        ),
        (
            "llm_request_timeout_seconds",
            0,
        ),
        (
            "llm_transport_max_retries",
            3,
        ),
        (
            "llm_max_repair_attempts",
            2,
        ),
    ],
)
def test_evaluation_settings_reject_invalid_values(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError):
        _settings(
            **{
                field_name: value,
            },
        )
