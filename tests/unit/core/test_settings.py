"""Unit tests for application settings."""

from typing import Any, cast

import pytest
from pydantic import ValidationError

from supportops.core.settings import (
    ApplicationEnvironment,
    LogLevel,
    Settings,
)

SUPPORTOPS_ENVIRONMENT_VARIABLES = (
    "SUPPORTOPS_ENVIRONMENT",
    "SUPPORTOPS_APPLICATION_NAME",
    "SUPPORTOPS_APPLICATION_VERSION",
    "SUPPORTOPS_LOG_LEVEL",
    "SUPPORTOPS_API_HOST",
    "SUPPORTOPS_API_PORT",
    "SUPPORTOPS_POSTGRESQL_URL",
    "SUPPORTOPS_POSTGRESQL_POOL_SIZE",
    "SUPPORTOPS_POSTGRESQL_MAX_OVERFLOW",
    "SUPPORTOPS_POSTGRESQL_POOL_TIMEOUT_SECONDS",
    "SUPPORTOPS_WORKER_ID",
    "SUPPORTOPS_WORKER_EXECUTOR",
    "SUPPORTOPS_WORKER_POLL_INTERVAL_SECONDS",
    "SUPPORTOPS_WORKER_LEASE_SECONDS",
    "SUPPORTOPS_WORKER_EXECUTION_TIMEOUT_SECONDS",
    "SUPPORTOPS_WORKER_SHUTDOWN_GRACE_SECONDS",
    "SUPPORTOPS_WORKER_MAX_ATTEMPTS",
    "SUPPORTOPS_WORKER_RETRY_BASE_SECONDS",
    "SUPPORTOPS_WORKER_RETRY_MAX_SECONDS",
    "SUPPORTOPS_QDRANT_URL",
    "SUPPORTOPS_QDRANT_API_KEY",
    "SUPPORTOPS_DEPENDENCY_HEALTH_TIMEOUT_SECONDS",
)


@pytest.fixture(autouse=True)
def isolate_settings_from_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent process environment variables from affecting settings unit tests."""

    for variable_name in SUPPORTOPS_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable_name, raising=False)


def create_required_settings(**overrides: object) -> Settings:
    """Create settings with the required infrastructure values."""

    values: dict[str, object] = {
        "postgresql_url": (
            "postgresql+asyncpg://supportops:supportops-local@localhost:5432/supportops"
        ),
        "qdrant_url": "http://localhost:6333",
    }
    values.update(overrides)

    settings_type = cast(Any, Settings)
    return cast(Settings, settings_type(_env_file=None, **values))


def test_settings_use_safe_local_defaults() -> None:
    settings = create_required_settings()

    assert settings.environment is ApplicationEnvironment.LOCAL
    assert settings.application_name == "SupportOps AI Platform"
    assert settings.application_version == "0.1.0"
    assert settings.log_level is LogLevel.INFO
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 8000
    assert settings.postgresql_pool_size == 5
    assert settings.postgresql_max_overflow == 10
    assert settings.postgresql_pool_timeout_seconds == 10.0
    assert settings.worker_id is None
    assert settings.worker_executor == "deterministic-ticket-processing"
    assert settings.worker_poll_interval_seconds == 1.0
    assert settings.worker_lease_seconds == 45.0
    assert settings.worker_execution_timeout_seconds == 30.0
    assert settings.worker_shutdown_grace_seconds == 10.0
    assert settings.worker_max_attempts == 3
    assert settings.worker_retry_base_seconds == 2.0
    assert settings.worker_retry_max_seconds == 60.0
    assert settings.qdrant_api_key is None
    assert settings.dependency_health_timeout_seconds == 2.0


def test_settings_load_worker_configuration_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUPPORTOPS_WORKER_ID", "worker-local-1")
    monkeypatch.setenv(
        "SUPPORTOPS_WORKER_EXECUTOR",
        "deterministic-ticket-processing",
    )
    monkeypatch.setenv("SUPPORTOPS_WORKER_POLL_INTERVAL_SECONDS", "0.5")
    monkeypatch.setenv("SUPPORTOPS_WORKER_LEASE_SECONDS", "70")
    monkeypatch.setenv("SUPPORTOPS_WORKER_EXECUTION_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("SUPPORTOPS_WORKER_SHUTDOWN_GRACE_SECONDS", "15")
    monkeypatch.setenv("SUPPORTOPS_WORKER_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("SUPPORTOPS_WORKER_RETRY_BASE_SECONDS", "4")
    monkeypatch.setenv("SUPPORTOPS_WORKER_RETRY_MAX_SECONDS", "120")

    settings_type = cast(Any, Settings)
    settings = cast(
        Settings,
        settings_type(
            _env_file=None,
            postgresql_url=(
                "postgresql+asyncpg://supportops:supportops-local@localhost:5432/supportops"
            ),
            qdrant_url="http://localhost:6333",
        ),
    )

    assert settings.worker_id == "worker-local-1"
    assert settings.worker_executor == "deterministic-ticket-processing"
    assert settings.worker_poll_interval_seconds == 0.5
    assert settings.worker_lease_seconds == 70.0
    assert settings.worker_execution_timeout_seconds == 60.0
    assert settings.worker_shutdown_grace_seconds == 15.0
    assert settings.worker_max_attempts == 5
    assert settings.worker_retry_base_seconds == 4.0
    assert settings.worker_retry_max_seconds == 120.0


def test_settings_reject_empty_worker_id() -> None:
    with pytest.raises(ValidationError):
        create_required_settings(worker_id="")


def test_settings_reject_worker_id_with_surrounding_whitespace() -> None:
    with pytest.raises(ValidationError):
        create_required_settings(worker_id="  worker-local-1  ")


def test_settings_reject_worker_id_longer_than_128_characters() -> None:
    with pytest.raises(ValidationError):
        create_required_settings(worker_id="w" * 129)


def test_settings_accept_worker_id_with_exactly_128_characters() -> None:
    worker_id = "w" * 128
    settings = create_required_settings(worker_id=worker_id)

    assert settings.worker_id == worker_id


def test_settings_reject_unsupported_worker_executor() -> None:
    with pytest.raises(ValidationError):
        create_required_settings(worker_executor="unsupported-executor")


def test_settings_normalize_trimmed_values() -> None:
    settings = create_required_settings(
        application_name="  SupportOps AI Platform  ",
        qdrant_url="  http://localhost:6333  ",
        qdrant_api_key="  local-api-key  ",
    )

    assert settings.application_name == "SupportOps AI Platform"
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_api_key == "local-api-key"


def test_settings_normalize_blank_optional_api_key() -> None:
    settings = create_required_settings(qdrant_api_key="   ")

    assert settings.qdrant_api_key is None


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("application_name", "   "),
        ("application_version", ""),
        ("api_host", "\t"),
        ("qdrant_url", "\n"),
    ],
)
def test_settings_reject_blank_required_strings(
    field_name: str,
    field_value: str,
) -> None:
    with pytest.raises(ValidationError):
        create_required_settings(**{field_name: field_value})


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("api_port", 0),
        ("api_port", 65536),
        ("postgresql_pool_size", 0),
        ("postgresql_max_overflow", -1),
        ("postgresql_pool_timeout_seconds", 0),
        ("worker_poll_interval_seconds", 0),
        ("worker_poll_interval_seconds", 61),
        ("worker_lease_seconds", 0),
        ("worker_lease_seconds", 3601),
        ("worker_execution_timeout_seconds", 0),
        ("worker_execution_timeout_seconds", 1801),
        ("worker_shutdown_grace_seconds", -1),
        ("worker_shutdown_grace_seconds", 301),
        ("worker_max_attempts", 0),
        ("worker_max_attempts", 101),
        ("worker_retry_base_seconds", 0),
        ("worker_retry_base_seconds", 3601),
        ("worker_retry_max_seconds", 0),
        ("worker_retry_max_seconds", 86401),
        ("dependency_health_timeout_seconds", 0),
        ("dependency_health_timeout_seconds", 31),
    ],
)
def test_settings_reject_invalid_numeric_configuration(
    field_name: str,
    field_value: int,
) -> None:
    with pytest.raises(ValidationError):
        create_required_settings(**{field_name: field_value})


def test_settings_reject_lease_shorter_than_execution_timeout_margin() -> None:
    with pytest.raises(ValidationError):
        create_required_settings(
            worker_lease_seconds=34,
            worker_execution_timeout_seconds=30,
        )


def test_settings_accept_lease_equal_to_execution_timeout_margin() -> None:
    settings = create_required_settings(
        worker_lease_seconds=35,
        worker_execution_timeout_seconds=30,
    )

    assert settings.worker_lease_seconds == 35.0
    assert settings.worker_execution_timeout_seconds == 30.0


def test_settings_reject_retry_max_smaller_than_retry_base() -> None:
    with pytest.raises(ValidationError):
        create_required_settings(
            worker_retry_base_seconds=2,
            worker_retry_max_seconds=1,
        )


def test_settings_accept_retry_max_equal_to_retry_base() -> None:
    settings = create_required_settings(
        worker_retry_base_seconds=2,
        worker_retry_max_seconds=2,
    )

    assert settings.worker_retry_base_seconds == 2.0
    assert settings.worker_retry_max_seconds == 2.0


def test_settings_require_postgresql_url() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, qdrant_url="http://localhost:6333")


def test_settings_require_qdrant_url() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            postgresql_url=(
                "postgresql+asyncpg://supportops:supportops-local@localhost:5432/supportops"
            ),
        )


def test_settings_reject_invalid_postgresql_url() -> None:
    with pytest.raises(ValidationError):
        create_required_settings(postgresql_url="not-a-postgresql-url")
