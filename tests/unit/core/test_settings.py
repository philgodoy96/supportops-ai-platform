"""Unit tests for application settings."""

from typing import Any, cast

import pytest
from pydantic import ValidationError

from supportops.core.settings import (
    AgentGraphDurability,
    ApplicationEnvironment,
    EmbeddingProviderName,
    LLMProviderName,
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
    "SUPPORTOPS_WORKER_POLL_INTERVAL_SECONDS",
    "SUPPORTOPS_WORKER_LEASE_SECONDS",
    "SUPPORTOPS_WORKER_EXECUTION_TIMEOUT_SECONDS",
    "SUPPORTOPS_WORKER_SHUTDOWN_GRACE_SECONDS",
    "SUPPORTOPS_WORKER_MAX_RETRYABLE_FAILURES",
    "SUPPORTOPS_WORKER_RETRY_BASE_SECONDS",
    "SUPPORTOPS_WORKER_RETRY_MAX_SECONDS",
    "SUPPORTOPS_APPROVAL_TTL_SECONDS",
    "SUPPORTOPS_APPROVAL_EXPIRATION_BATCH_SIZE",
    "SUPPORTOPS_TICKET_PROCESSING_WORKFLOW_VERSION",
    "SUPPORTOPS_AGENT_GRAPH_MAX_STEPS",
    "SUPPORTOPS_AGENT_GRAPH_MAX_TOOL_CALLS",
    "SUPPORTOPS_AGENT_GRAPH_MAX_DECISION_TURNS",
    "SUPPORTOPS_AGENT_GRAPH_TOOL_TIMEOUT_SECONDS",
    "SUPPORTOPS_AGENT_GRAPH_CHECKPOINT_DATABASE_URL",
    "SUPPORTOPS_AGENT_GRAPH_DURABILITY",
    "SUPPORTOPS_SUPPORT_WORKFLOW_VERSION",
    "SUPPORTOPS_LLM_PROVIDER",
    "SUPPORTOPS_OPENAI_API_KEY",
    "SUPPORTOPS_OPENAI_MODEL",
    "SUPPORTOPS_OPENAI_BASE_URL",
    "SUPPORTOPS_LLM_REQUEST_TIMEOUT_SECONDS",
    "SUPPORTOPS_LLM_TRANSPORT_MAX_RETRIES",
    "SUPPORTOPS_LLM_MAX_REPAIR_ATTEMPTS",
    "SUPPORTOPS_EMBEDDING_PROVIDER",
    "SUPPORTOPS_EMBEDDING_MODEL",
    "SUPPORTOPS_EMBEDDING_DIMENSIONS",
    "SUPPORTOPS_EMBEDDING_REQUEST_TIMEOUT_SECONDS",
    "SUPPORTOPS_EMBEDDING_TRANSPORT_MAX_RETRIES",
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
    assert settings.worker_poll_interval_seconds == 1.0
    assert settings.worker_lease_seconds == 150.0
    assert settings.worker_execution_timeout_seconds == 135.0
    assert settings.worker_shutdown_grace_seconds == 10.0
    assert settings.worker_max_retryable_failures == 3
    assert settings.worker_retry_base_seconds == 2.0
    assert settings.worker_retry_max_seconds == 60.0
    assert settings.approval_ttl_seconds == 86400.0
    assert settings.approval_expiration_batch_size == 100
    assert settings.ticket_processing_workflow_version == "controlled-support-v1"
    assert settings.agent_graph_max_steps == 16
    assert settings.agent_graph_max_tool_calls == 3
    assert settings.agent_graph_max_decision_turns == 4
    assert settings.agent_graph_tool_timeout_seconds == 15.0
    assert settings.agent_graph_checkpoint_database_url is None
    assert settings.agent_graph_durability is AgentGraphDurability.SYNC
    assert settings.support_workflow_version == "controlled-support-v1"
    assert settings.llm_provider is LLMProviderName.MOCK
    assert settings.openai_api_key is None
    assert settings.openai_model == "gpt-5-nano"
    assert settings.openai_base_url is None
    assert settings.llm_request_timeout_seconds == 10.0
    assert settings.llm_transport_max_retries == 1
    assert settings.llm_max_repair_attempts == 1
    assert settings.embedding_provider is EmbeddingProviderName.MOCK
    assert settings.embedding_model == "mock-hashing-embedding-v1"
    assert settings.embedding_dimensions == 64
    assert settings.embedding_request_timeout_seconds == 12.0
    assert settings.embedding_transport_max_retries == 1
    assert settings.qdrant_api_key is None
    assert settings.dependency_health_timeout_seconds == 2.0


def test_settings_load_worker_configuration_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUPPORTOPS_WORKER_ID", "worker-local-1")
    monkeypatch.setenv("SUPPORTOPS_WORKER_POLL_INTERVAL_SECONDS", "0.5")
    monkeypatch.setenv("SUPPORTOPS_WORKER_LEASE_SECONDS", "80")
    monkeypatch.setenv("SUPPORTOPS_WORKER_EXECUTION_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("SUPPORTOPS_WORKER_SHUTDOWN_GRACE_SECONDS", "15")
    monkeypatch.setenv("SUPPORTOPS_WORKER_MAX_ATTEMPTS", "9")
    monkeypatch.setenv("SUPPORTOPS_WORKER_MAX_RETRYABLE_FAILURES", "5")
    monkeypatch.setenv("SUPPORTOPS_WORKER_RETRY_BASE_SECONDS", "4")
    monkeypatch.setenv("SUPPORTOPS_WORKER_RETRY_MAX_SECONDS", "120")
    monkeypatch.setenv("SUPPORTOPS_APPROVAL_TTL_SECONDS", "3600")
    monkeypatch.setenv("SUPPORTOPS_APPROVAL_EXPIRATION_BATCH_SIZE", "25")
    monkeypatch.setenv(
        "SUPPORTOPS_TICKET_PROCESSING_WORKFLOW_VERSION",
        "ticket-classification-v1",
    )
    monkeypatch.setenv("SUPPORTOPS_LLM_REQUEST_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("SUPPORTOPS_LLM_MAX_REPAIR_ATTEMPTS", "0")

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
    assert settings.worker_poll_interval_seconds == 0.5
    assert settings.worker_lease_seconds == 80.0
    assert settings.worker_execution_timeout_seconds == 60.0
    assert settings.worker_shutdown_grace_seconds == 15.0
    assert settings.worker_max_retryable_failures == 5
    assert settings.worker_retry_base_seconds == 4.0
    assert settings.worker_retry_max_seconds == 120.0
    assert settings.approval_ttl_seconds == 3600.0
    assert settings.approval_expiration_batch_size == 25


def test_settings_load_agent_graph_configuration_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_database_url = (
        "postgresql://checkpoint-user:checkpoint-password@localhost:5432/supportops_checkpoints"
    )
    monkeypatch.setenv("SUPPORTOPS_AGENT_GRAPH_MAX_STEPS", "24")
    monkeypatch.setenv("SUPPORTOPS_AGENT_GRAPH_MAX_TOOL_CALLS", "5")
    monkeypatch.setenv("SUPPORTOPS_AGENT_GRAPH_MAX_DECISION_TURNS", "7")
    monkeypatch.setenv("SUPPORTOPS_AGENT_GRAPH_TOOL_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv(
        "SUPPORTOPS_AGENT_GRAPH_CHECKPOINT_DATABASE_URL",
        checkpoint_database_url,
    )
    monkeypatch.setenv("SUPPORTOPS_AGENT_GRAPH_DURABILITY", "sync")
    monkeypatch.setenv("SUPPORTOPS_SUPPORT_WORKFLOW_VERSION", "controlled-support-v1")
    monkeypatch.setenv("SUPPORTOPS_WORKER_EXECUTION_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("SUPPORTOPS_WORKER_LEASE_SECONDS", "75")
    monkeypatch.setenv(
        "SUPPORTOPS_TICKET_PROCESSING_WORKFLOW_VERSION",
        "ticket-classification-v1",
    )
    monkeypatch.setenv("SUPPORTOPS_LLM_REQUEST_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("SUPPORTOPS_LLM_MAX_REPAIR_ATTEMPTS", "0")

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

    assert settings.agent_graph_max_steps == 24
    assert settings.agent_graph_max_tool_calls == 5
    assert settings.agent_graph_max_decision_turns == 7
    assert settings.agent_graph_tool_timeout_seconds == 20.0
    assert settings.agent_graph_checkpoint_database_url is not None
    assert (
        settings.agent_graph_checkpoint_database_url.get_secret_value() == checkpoint_database_url
    )
    assert settings.agent_graph_durability is AgentGraphDurability.SYNC
    assert settings.support_workflow_version == "controlled-support-v1"
    assert settings.worker_execution_timeout_seconds == 60.0
    assert settings.worker_lease_seconds == 75.0


def test_settings_load_llm_configuration_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SUPPORTOPS_TICKET_PROCESSING_WORKFLOW_VERSION",
        "deterministic-baseline-v1",
    )
    monkeypatch.setenv("SUPPORTOPS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("SUPPORTOPS_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("SUPPORTOPS_OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("SUPPORTOPS_OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("SUPPORTOPS_LLM_REQUEST_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("SUPPORTOPS_LLM_TRANSPORT_MAX_RETRIES", "2")
    monkeypatch.setenv("SUPPORTOPS_LLM_MAX_REPAIR_ATTEMPTS", "0")
    monkeypatch.setenv("SUPPORTOPS_WORKER_EXECUTION_TIMEOUT_SECONDS", "50")
    monkeypatch.setenv("SUPPORTOPS_WORKER_LEASE_SECONDS", "65")

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

    assert settings.llm_provider is LLMProviderName.OPENAI
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "test-openai-key"
    assert settings.openai_model == "gpt-4.1-mini"
    assert settings.openai_base_url == "https://api.example.com/v1"
    assert settings.llm_request_timeout_seconds == 15.0
    assert settings.llm_transport_max_retries == 2
    assert settings.llm_max_repair_attempts == 0
    assert settings.ticket_processing_workflow_version == "deterministic-baseline-v1"
    assert settings.worker_execution_timeout_seconds == 50.0
    assert settings.worker_lease_seconds == 65.0


def test_settings_load_embedding_configuration_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUPPORTOPS_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("SUPPORTOPS_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("SUPPORTOPS_EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv("SUPPORTOPS_EMBEDDING_REQUEST_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("SUPPORTOPS_EMBEDDING_TRANSPORT_MAX_RETRIES", "2")
    monkeypatch.setenv("SUPPORTOPS_OPENAI_API_KEY", "test-openai-key")

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

    assert settings.embedding_provider is EmbeddingProviderName.OPENAI
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 1536
    assert settings.embedding_request_timeout_seconds == 20.0
    assert settings.embedding_transport_max_retries == 2
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "test-openai-key"


def test_settings_mock_provider_does_not_require_openai_api_key() -> None:
    settings = create_required_settings(llm_provider=LLMProviderName.MOCK)

    assert settings.llm_provider is LLMProviderName.MOCK
    assert settings.openai_api_key is None


def test_settings_mock_embedding_provider_does_not_require_openai_api_key() -> None:
    settings = create_required_settings(
        llm_provider=LLMProviderName.MOCK,
        embedding_provider=EmbeddingProviderName.MOCK,
    )

    assert settings.llm_provider is LLMProviderName.MOCK
    assert settings.embedding_provider is EmbeddingProviderName.MOCK
    assert settings.openai_api_key is None


def test_settings_reject_openai_provider_without_api_key() -> None:
    with pytest.raises(
        ValidationError,
        match=r"openai_api_key is required when an OpenAI provider is configured\.",
    ):
        create_required_settings(
            llm_provider=LLMProviderName.OPENAI,
            openai_api_key=None,
            ticket_processing_workflow_version="ticket-classification-v1",
            worker_execution_timeout_seconds=50,
            worker_lease_seconds=65,
            llm_request_timeout_seconds=12,
        )


def test_settings_blank_openai_api_key_becomes_none_and_is_rejected_for_openai() -> None:
    with pytest.raises(
        ValidationError,
        match=r"openai_api_key is required when an OpenAI provider is configured\.",
    ):
        create_required_settings(
            llm_provider=LLMProviderName.OPENAI,
            openai_api_key="   ",
            ticket_processing_workflow_version="ticket-classification-v1",
            worker_execution_timeout_seconds=50,
            worker_lease_seconds=65,
            llm_request_timeout_seconds=12,
        )


def test_settings_reject_openai_embedding_provider_without_api_key() -> None:
    with pytest.raises(
        ValidationError,
        match=r"openai_api_key is required when an OpenAI provider is configured\.",
    ):
        create_required_settings(
            embedding_provider=EmbeddingProviderName.OPENAI,
            openai_api_key=None,
        )


def test_settings_reject_unsupported_embedding_provider() -> None:
    with pytest.raises(ValidationError):
        create_required_settings(embedding_provider="unsupported-provider")


def test_settings_reject_blank_embedding_model() -> None:
    with pytest.raises(ValidationError):
        create_required_settings(embedding_model="   ")


def test_settings_strip_surrounding_whitespace_from_openai_api_key() -> None:
    settings = create_required_settings(
        llm_provider=LLMProviderName.OPENAI,
        openai_api_key="  test-openai-key  ",
        ticket_processing_workflow_version="ticket-classification-v1",
        worker_execution_timeout_seconds=50,
        worker_lease_seconds=65,
        llm_request_timeout_seconds=12,
    )

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "test-openai-key"


def test_settings_openai_api_key_absent_from_repr() -> None:
    settings = create_required_settings(
        llm_provider=LLMProviderName.OPENAI,
        openai_api_key="super-secret-key",
        ticket_processing_workflow_version="ticket-classification-v1",
        worker_execution_timeout_seconds=50,
        worker_lease_seconds=65,
        llm_request_timeout_seconds=12,
    )

    assert "super-secret-key" not in repr(settings)


def test_settings_strip_openai_base_url() -> None:
    settings = create_required_settings(
        openai_base_url="  https://api.example.com/v1  ",
    )

    assert settings.openai_base_url == "https://api.example.com/v1"


def test_settings_blank_openai_base_url_becomes_none() -> None:
    settings = create_required_settings(openai_base_url="   ")

    assert settings.openai_base_url is None


def test_settings_reject_unsupported_llm_provider() -> None:
    with pytest.raises(ValidationError):
        create_required_settings(llm_provider="unsupported-provider")


def test_settings_reject_unsupported_ticket_processing_workflow_version() -> None:
    with pytest.raises(ValidationError):
        create_required_settings(
            ticket_processing_workflow_version="unsupported-workflow",
        )


def test_settings_accept_controlled_support_as_ticket_processing_workflow_version() -> None:
    settings = create_required_settings(
        ticket_processing_workflow_version="controlled-support-v1",
    )

    assert settings.ticket_processing_workflow_version == "controlled-support-v1"


def test_settings_accept_human_approved_support_as_ticket_processing_workflow_version() -> None:
    settings = create_required_settings(
        ticket_processing_workflow_version="human-approved-support-v1",
    )

    assert settings.ticket_processing_workflow_version == "human-approved-support-v1"
    assert create_required_settings().ticket_processing_workflow_version == (
        "controlled-support-v1"
    )


def test_settings_reject_unsupported_agent_graph_durability() -> None:
    with pytest.raises(ValidationError):
        create_required_settings(agent_graph_durability="async")


def test_settings_reject_unsupported_support_workflow_version() -> None:
    with pytest.raises(ValidationError):
        create_required_settings(support_workflow_version="unsupported-workflow")


def test_settings_strip_agent_graph_checkpoint_database_url() -> None:
    checkpoint_database_url = (
        "postgresql://checkpoint-user:checkpoint-password@localhost:5432/supportops_checkpoints"
    )
    settings = create_required_settings(
        agent_graph_checkpoint_database_url=f"  {checkpoint_database_url}  ",
    )

    assert settings.agent_graph_checkpoint_database_url is not None
    assert (
        settings.agent_graph_checkpoint_database_url.get_secret_value() == checkpoint_database_url
    )


def test_settings_blank_agent_graph_checkpoint_database_url_becomes_none() -> None:
    settings = create_required_settings(agent_graph_checkpoint_database_url="   ")

    assert settings.agent_graph_checkpoint_database_url is None


def test_settings_agent_graph_checkpoint_database_url_absent_from_repr() -> None:
    checkpoint_database_url = (
        "postgresql://checkpoint-user:checkpoint-password@localhost:5432/supportops_checkpoints"
    )
    settings = create_required_settings(
        agent_graph_checkpoint_database_url=checkpoint_database_url,
    )

    assert "checkpoint-password" not in repr(settings)
    assert checkpoint_database_url not in repr(settings)


@pytest.mark.parametrize(
    "checkpoint_database_url",
    [
        "mysql://checkpoint-user:checkpoint-password@localhost:5432/supportops_checkpoints",
        "postgresql://checkpoint-user:checkpoint-password@/supportops_checkpoints",
        "postgresql://checkpoint-user:checkpoint-password@localhost:5432/",
    ],
)
def test_settings_reject_invalid_agent_graph_checkpoint_database_url(
    checkpoint_database_url: str,
) -> None:
    with pytest.raises(
        ValidationError,
        match=(
            r"agent_graph_checkpoint_database_url must be a valid "
            r"PostgreSQL connection URL\."
        ),
    ) as raised_error:
        create_required_settings(
            agent_graph_checkpoint_database_url=checkpoint_database_url,
        )

    error_text = str(raised_error.value)
    assert "checkpoint-password" not in error_text
    assert checkpoint_database_url not in error_text


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
        ("worker_max_retryable_failures", 0),
        ("worker_max_retryable_failures", 101),
        ("worker_retry_base_seconds", 0),
        ("worker_retry_base_seconds", 3601),
        ("worker_retry_max_seconds", 0),
        ("worker_retry_max_seconds", 86401),
        ("approval_ttl_seconds", 0),
        ("approval_ttl_seconds", 2592001),
        ("approval_expiration_batch_size", 0),
        ("approval_expiration_batch_size", 1001),
        ("llm_request_timeout_seconds", 0),
        ("llm_request_timeout_seconds", 301),
        ("llm_transport_max_retries", -1),
        ("llm_transport_max_retries", 3),
        ("llm_max_repair_attempts", -1),
        ("llm_max_repair_attempts", 2),
        ("embedding_dimensions", 0),
        ("embedding_dimensions", 4097),
        ("embedding_request_timeout_seconds", 0),
        ("embedding_request_timeout_seconds", 301),
        ("embedding_transport_max_retries", -1),
        ("embedding_transport_max_retries", 3),
        ("dependency_health_timeout_seconds", 0),
        ("dependency_health_timeout_seconds", 31),
        ("agent_graph_max_steps", 7),
        ("agent_graph_max_steps", 65),
        ("agent_graph_max_tool_calls", 0),
        ("agent_graph_max_tool_calls", 11),
        ("agent_graph_max_decision_turns", 1),
        ("agent_graph_max_decision_turns", 12),
        ("agent_graph_tool_timeout_seconds", 0),
        ("agent_graph_tool_timeout_seconds", 61),
    ],
)
def test_settings_reject_invalid_numeric_configuration(
    field_name: str,
    field_value: int,
) -> None:
    with pytest.raises(ValidationError):
        create_required_settings(**{field_name: field_value})


def test_settings_accept_approval_policy_boundaries() -> None:
    settings = create_required_settings(
        approval_ttl_seconds=1,
        approval_expiration_batch_size=1,
    )
    assert settings.approval_ttl_seconds == 1.0
    assert settings.approval_expiration_batch_size == 1

    maximum = create_required_settings(
        approval_ttl_seconds=2592000,
        approval_expiration_batch_size=1000,
    )
    assert maximum.approval_ttl_seconds == 2592000.0
    assert maximum.approval_expiration_batch_size == 1000


def test_settings_ignore_unrelated_legacy_approval_environment_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUPPORTOPS_APPROVAL_TTL", "30")
    monkeypatch.setenv("SUPPORTOPS_APPROVAL_BATCH_SIZE", "9")
    monkeypatch.setenv("APPROVAL_TTL_SECONDS", "120")
    monkeypatch.setenv("APPROVAL_EXPIRATION_BATCH_SIZE", "50")

    settings = create_required_settings()

    assert settings.approval_ttl_seconds == 86400.0
    assert settings.approval_expiration_batch_size == 100


def test_settings_reject_max_steps_below_tool_loop_budget() -> None:
    with pytest.raises(
        ValidationError,
        match=r"agent_graph_max_steps must cover the configured maximum tool loop\.",
    ):
        create_required_settings(
            agent_graph_max_tool_calls=3,
            agent_graph_max_steps=11,
        )


def test_settings_accept_max_steps_equal_to_tool_loop_budget() -> None:
    settings = create_required_settings(
        agent_graph_max_tool_calls=3,
        agent_graph_max_steps=12,
    )

    assert settings.agent_graph_max_tool_calls == 3
    assert settings.agent_graph_max_steps == 12


def test_settings_reject_decision_turns_without_terminal_decision() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            r"agent_graph_max_decision_turns must allow one terminal decision "
            r"after all tool calls\."
        ),
    ):
        create_required_settings(
            agent_graph_max_tool_calls=3,
            agent_graph_max_decision_turns=3,
            agent_graph_max_steps=16,
        )


def test_settings_accept_decision_turns_with_terminal_decision() -> None:
    settings = create_required_settings(
        agent_graph_max_tool_calls=3,
        agent_graph_max_decision_turns=4,
        agent_graph_max_steps=16,
    )

    assert settings.agent_graph_max_tool_calls == 3
    assert settings.agent_graph_max_decision_turns == 4


def test_settings_reject_tool_timeout_equal_to_worker_execution_timeout() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            r"agent_graph_tool_timeout_seconds must be smaller than "
            r"worker_execution_timeout_seconds\."
        ),
    ):
        create_required_settings(
            ticket_processing_workflow_version="ticket-classification-v1",
            agent_graph_tool_timeout_seconds=30,
            worker_execution_timeout_seconds=30,
            worker_lease_seconds=45,
            llm_request_timeout_seconds=12,
            llm_max_repair_attempts=0,
        )


def test_settings_accept_tool_timeout_below_worker_execution_timeout() -> None:
    settings = create_required_settings(
        ticket_processing_workflow_version="ticket-classification-v1",
        agent_graph_tool_timeout_seconds=29,
        worker_execution_timeout_seconds=30,
        worker_lease_seconds=45,
        llm_request_timeout_seconds=12,
        llm_max_repair_attempts=0,
    )

    assert settings.agent_graph_tool_timeout_seconds == 29.0
    assert settings.worker_execution_timeout_seconds == 30.0


def test_settings_reject_lease_shorter_than_execution_timeout_margin() -> None:
    with pytest.raises(ValidationError):
        create_required_settings(
            ticket_processing_workflow_version="ticket-classification-v1",
            worker_lease_seconds=44,
            worker_execution_timeout_seconds=30,
            llm_request_timeout_seconds=12,
            llm_max_repair_attempts=0,
        )


def test_settings_accept_lease_equal_to_execution_timeout_margin() -> None:
    settings = create_required_settings(
        ticket_processing_workflow_version="ticket-classification-v1",
        worker_lease_seconds=45,
        worker_execution_timeout_seconds=30,
        llm_request_timeout_seconds=12,
        llm_max_repair_attempts=0,
    )

    assert settings.worker_lease_seconds == 45.0
    assert settings.worker_execution_timeout_seconds == 30.0


def test_settings_reject_execution_timeout_shorter_than_llm_budget() -> None:
    with pytest.raises(ValidationError):
        create_required_settings(
            ticket_processing_workflow_version="ticket-classification-v1",
            worker_execution_timeout_seconds=38,
            worker_lease_seconds=53,
            llm_request_timeout_seconds=12,
            llm_max_repair_attempts=1,
        )


def test_settings_accept_execution_timeout_equal_to_llm_budget_margin() -> None:
    settings = create_required_settings(
        ticket_processing_workflow_version="ticket-classification-v1",
        worker_execution_timeout_seconds=39,
        worker_lease_seconds=54,
        llm_request_timeout_seconds=12,
        llm_max_repair_attempts=1,
    )

    assert settings.worker_execution_timeout_seconds == 39.0
    assert settings.worker_lease_seconds == 54.0
    assert settings.llm_request_timeout_seconds == 12.0
    assert settings.llm_max_repair_attempts == 1


def test_settings_accept_zero_repair_attempts_reduced_llm_budget() -> None:
    settings = create_required_settings(
        ticket_processing_workflow_version="ticket-classification-v1",
        worker_execution_timeout_seconds=27,
        worker_lease_seconds=42,
        llm_request_timeout_seconds=12,
        llm_max_repair_attempts=0,
    )

    assert settings.worker_execution_timeout_seconds == 27.0
    assert settings.worker_lease_seconds == 42.0
    assert settings.llm_request_timeout_seconds == 12.0
    assert settings.llm_max_repair_attempts == 0


def test_settings_reject_controlled_workflow_llm_budget_exceeding_execution_timeout() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            r"worker_execution_timeout_seconds must cover all configured LLM "
            r"invocations plus the safety margin\."
        ),
    ):
        create_required_settings(
            ticket_processing_workflow_version="controlled-support-v1",
            worker_execution_timeout_seconds=134,
            worker_lease_seconds=149,
            llm_request_timeout_seconds=10,
            llm_max_repair_attempts=1,
        )


def test_settings_accept_baseline_and_classification_single_generation_budgets() -> None:
    baseline = create_required_settings(
        ticket_processing_workflow_version="deterministic-baseline-v1",
        worker_execution_timeout_seconds=39,
        worker_lease_seconds=54,
        llm_request_timeout_seconds=12,
        llm_max_repair_attempts=1,
    )
    classification = create_required_settings(
        ticket_processing_workflow_version="ticket-classification-v1",
        worker_execution_timeout_seconds=39,
        worker_lease_seconds=54,
        llm_request_timeout_seconds=12,
        llm_max_repair_attempts=1,
    )

    assert baseline.ticket_processing_workflow_version == "deterministic-baseline-v1"
    assert classification.ticket_processing_workflow_version == "ticket-classification-v1"


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
