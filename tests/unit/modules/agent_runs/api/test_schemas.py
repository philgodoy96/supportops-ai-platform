"""Unit tests for AgentRun HTTP response projections."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest

from supportops.ai.gateway.errors import LLMErrorCode
from supportops.ai.gateway.results import LLMInvocationStatus
from supportops.ai.schemas.ticket_classification import (
    TICKET_CLASSIFICATION_SCHEMA_VERSION,
)
from supportops.modules.agent_runs.api.schemas import (
    AgentRunAttemptResponse,
    AgentRunLLMInvocationResponse,
    AgentRunResponse,
)
from supportops.modules.agent_runs.domain.models import (
    DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
    AgentRun,
    AgentRunAttempt,
    AgentRunAttemptOutcome,
    AgentRunStatus,
)
from supportops.modules.ticket_classifications.domain.inspection import (
    AgentRunClassificationReference,
    LLMInvocationInspection,
)

_NOW = datetime(
    2026,
    7,
    31,
    21,
    0,
    tzinfo=UTC,
)
_WORKSPACE_ID = UUID(
    "032c8c87-57cc-4d14-bfbd-04968b4e8cd4",
)
_TICKET_ID = UUID(
    "38bb60fe-d2ea-4615-b499-91aa45069019",
)
_AGENT_RUN_ID = UUID(
    "69184ef1-4d71-452e-8070-0b784c29368e",
)
_ATTEMPT_ID = UUID(
    "2b39f5b7-b2a4-48d0-b079-fdad286d5315",
)
_INGESTION_REQUEST_ID = UUID(
    "725eec8a-c504-4071-ac96-c78cc907f26c",
)
_CORRELATION_ID = UUID(
    "1038c98e-62fd-45df-9839-138f7105cb78",
)
_LEASE_TOKEN = UUID(
    "dd0ae456-3467-41db-93d1-a908f40e8365",
)
_EXECUTION_REQUEST_ID = UUID(
    "d1fa068f-2278-47a8-b3c9-39ccf91f0a5e",
)
_CLASSIFICATION_ID = UUID(
    "66666666-ffff-4666-8666-666666666666",
)
_INVOCATION_ID = UUID(
    "55555555-eeee-4555-8555-555555555555",
)
_PROMPT_HASH = "a" * 64
_ZERO_COST = Decimal("0.000000000000")


def create_agent_run() -> AgentRun:
    """Create one deterministic AgentRun."""

    return AgentRun.create_initial(
        agent_run_id=_AGENT_RUN_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        ingestion_request_id=_INGESTION_REQUEST_ID,
        correlation_id=_CORRELATION_ID,
        workflow_version=DETERMINISTIC_BASELINE_WORKFLOW_VERSION,
        max_retryable_failures=3,
        now=_NOW,
    )


def create_attempt() -> AgentRunAttempt:
    """Create one deterministic AgentRun attempt."""

    return AgentRunAttempt.start(
        attempt_id=_ATTEMPT_ID,
        agent_run_id=_AGENT_RUN_ID,
        attempt_number=1,
        worker_id="worker-a",
        lease_token=_LEASE_TOKEN,
        execution_request_id=_EXECUTION_REQUEST_ID,
        now=_NOW,
    )


def create_classification_reference() -> AgentRunClassificationReference:
    """Create one deterministic accepted-classification reference."""

    return AgentRunClassificationReference(
        id=_CLASSIFICATION_ID,
        schema_version=TICKET_CLASSIFICATION_SCHEMA_VERSION,
        created_at=_NOW,
    )


def create_succeeded_invocation() -> LLMInvocationInspection:
    """Create one deterministic successful logical invocation."""

    return LLMInvocationInspection(
        id=_INVOCATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        attempt_number=1,
        invocation_sequence=1,
        status=LLMInvocationStatus.SUCCEEDED,
        provider="mock",
        model="mock-ticket-classifier-v1",
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        schema_version=TICKET_CLASSIFICATION_SCHEMA_VERSION,
        input_tokens=100,
        cached_input_tokens=0,
        output_tokens=20,
        reasoning_tokens=None,
        total_tokens=120,
        pricing_catalog_version="pricing-v1",
        pricing_found=True,
        estimated_input_cost_usd=_ZERO_COST,
        estimated_cached_input_cost_usd=_ZERO_COST,
        estimated_output_cost_usd=_ZERO_COST,
        estimated_total_cost_usd=_ZERO_COST,
        latency_ms=25,
        error_code=None,
        created_at=_NOW,
    )


def create_failed_invocation() -> LLMInvocationInspection:
    """Create one deterministic failed logical invocation."""

    return LLMInvocationInspection(
        id=_INVOCATION_ID,
        workspace_id=_WORKSPACE_ID,
        ticket_id=_TICKET_ID,
        agent_run_id=_AGENT_RUN_ID,
        agent_run_attempt_id=_ATTEMPT_ID,
        attempt_number=1,
        invocation_sequence=1,
        status=LLMInvocationStatus.TIMED_OUT,
        provider="mock",
        model="mock-ticket-classifier-v1",
        prompt_id="ticket-classification",
        prompt_version=1,
        prompt_content_hash=_PROMPT_HASH,
        schema_version=TICKET_CLASSIFICATION_SCHEMA_VERSION,
        input_tokens=None,
        cached_input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        total_tokens=None,
        pricing_catalog_version="pricing-v1",
        pricing_found=False,
        estimated_input_cost_usd=None,
        estimated_cached_input_cost_usd=None,
        estimated_output_cost_usd=None,
        estimated_total_cost_usd=None,
        latency_ms=10,
        error_code=LLMErrorCode.TIMEOUT,
        created_at=_NOW,
    )


def test_agent_run_response_projects_safe_fields() -> None:
    agent_run = replace(
        create_agent_run(),
        status=AgentRunStatus.RETRY_SCHEDULED,
        attempt_count=1,
        available_at=_NOW + timedelta(seconds=2),
        first_started_at=_NOW,
        last_error_code="unexpected_executor_failure",
        last_error_summary=("The executor failed unexpectedly and may be retried."),
        updated_at=_NOW + timedelta(seconds=1),
    )
    classification = create_classification_reference()

    response = AgentRunResponse.from_domain(
        agent_run,
        classification=classification,
    )

    assert response.id == _AGENT_RUN_ID
    assert response.workspace_id == _WORKSPACE_ID
    assert response.ticket_id == _TICKET_ID
    assert response.status is AgentRunStatus.RETRY_SCHEDULED
    assert response.workflow.name == "ticket-processing"
    assert response.workflow.version == "deterministic-baseline-v1"
    assert response.workflow.trigger_key == "initial-ticket-processing"
    assert response.classification is not None
    assert response.classification.id == _CLASSIFICATION_ID
    assert response.classification.schema_version == TICKET_CLASSIFICATION_SCHEMA_VERSION
    assert response.classification.created_at == _NOW
    assert response.attempt_count == 1
    assert response.retryable_failure_count == 0
    assert response.max_retryable_failures == 3
    assert response.correlation_id == _CORRELATION_ID
    assert response.last_error is not None
    assert response.last_error.code == "unexpected_executor_failure"


def test_agent_run_response_supports_no_classification() -> None:
    response = AgentRunResponse.from_domain(create_agent_run())

    assert response.classification is None


def test_agent_run_response_omits_internal_fields() -> None:
    agent_run = replace(
        create_agent_run(),
        status=AgentRunStatus.RUNNING,
        attempt_count=1,
        first_started_at=_NOW,
        lease_owner="worker-a",
        lease_token=_LEASE_TOKEN,
        lease_expires_at=_NOW + timedelta(seconds=45),
    )

    payload = AgentRunResponse.from_domain(
        agent_run,
    ).model_dump()

    assert "lease_owner" not in payload
    assert "lease_token" not in payload
    assert "lease_expires_at" not in payload
    assert "ingestion_request_id" not in payload
    assert "max_attempts" not in payload
    assert "execution_request_id" not in payload
    assert "approval_request_id" not in payload
    assert "graph_thread_id" not in payload


def test_agent_run_response_serializes_waiting_for_approval_with_null_available_at() -> None:
    agent_run = replace(
        create_agent_run(),
        status=AgentRunStatus.WAITING_FOR_APPROVAL,
        available_at=None,
        attempt_count=1,
        first_started_at=_NOW,
        updated_at=_NOW,
    )

    response = AgentRunResponse.from_domain(agent_run)
    payload = response.model_dump(mode="json")

    assert response.status is AgentRunStatus.WAITING_FOR_APPROVAL
    assert response.available_at is None
    assert response.completed_at is None
    assert response.last_error is None
    assert payload["status"] == "waiting_for_approval"
    assert payload["available_at"] is None
    assert "lease_token" not in payload
    assert "approval_request_id" not in payload
    assert "graph_thread_id" not in payload


def test_agent_run_response_requires_complete_safe_error_pair() -> None:
    agent_run = create_agent_run()
    # Bypass AgentRun invariants to assert defensive projection behavior.
    object.__setattr__(agent_run, "last_error_code", "retryable_failure")

    response = AgentRunResponse.from_domain(agent_run)

    assert response.last_error is None


def test_attempt_response_projects_safe_fields() -> None:
    attempt = replace(
        create_attempt(),
        finished_at=_NOW + timedelta(seconds=1),
        outcome=AgentRunAttemptOutcome.RETRYABLE_FAILURE,
        error_code="unexpected_executor_failure",
        error_summary=("The executor failed unexpectedly and may be retried."),
    )

    response = AgentRunAttemptResponse.from_domain(attempt)

    assert response.id == _ATTEMPT_ID
    assert response.attempt_number == 1
    assert response.worker_id == "worker-a"
    assert response.outcome is AgentRunAttemptOutcome.RETRYABLE_FAILURE
    assert response.error is not None
    assert response.error.code == "unexpected_executor_failure"


def test_attempt_response_omits_internal_fencing_fields() -> None:
    payload = AgentRunAttemptResponse.from_domain(
        create_attempt(),
    ).model_dump()

    assert "agent_run_id" not in payload
    assert "lease_token" not in payload
    assert "execution_request_id" not in payload


def test_attempt_response_supports_active_attempt() -> None:
    response = AgentRunAttemptResponse.from_domain(
        create_attempt(),
    )

    assert response.finished_at is None
    assert response.outcome is None
    assert response.error is None


def test_attempt_response_serializes_awaiting_approval() -> None:
    attempt = replace(
        create_attempt(),
        finished_at=_NOW + timedelta(seconds=1),
        outcome=AgentRunAttemptOutcome.AWAITING_APPROVAL,
    )

    response = AgentRunAttemptResponse.from_domain(attempt)
    payload = response.model_dump(mode="json")

    assert response.outcome is AgentRunAttemptOutcome.AWAITING_APPROVAL
    assert response.error is None
    assert payload["outcome"] == "awaiting_approval"
    assert "lease_token" not in payload
    assert "approval_request_id" not in payload


def test_llm_invocation_response_projects_successful_invocation() -> None:
    response = AgentRunLLMInvocationResponse.from_domain(
        create_succeeded_invocation(),
    )

    assert response.id == _INVOCATION_ID
    assert response.agent_run_attempt_id == _ATTEMPT_ID
    assert response.attempt_number == 1
    assert response.invocation_sequence == 1
    assert response.status is LLMInvocationStatus.SUCCEEDED
    assert response.provider == "mock"
    assert response.model == "mock-ticket-classifier-v1"
    assert response.prompt.id == "ticket-classification"
    assert response.prompt.version == 1
    assert response.prompt.content_hash == _PROMPT_HASH
    assert response.usage is not None
    assert response.usage.input_tokens == 100
    assert response.usage.cached_input_tokens == 0
    assert response.usage.output_tokens == 20
    assert response.usage.reasoning_tokens is None
    assert response.usage.total_tokens == 120
    assert response.estimated_cost.pricing_catalog_version == "pricing-v1"
    assert response.estimated_cost.pricing_found is True
    assert response.estimated_cost.input_cost_usd == _ZERO_COST
    assert response.estimated_cost.cached_input_cost_usd == _ZERO_COST
    assert response.estimated_cost.output_cost_usd == _ZERO_COST
    assert response.estimated_cost.total_cost_usd == _ZERO_COST
    assert response.latency_ms == 25
    assert response.error_code is None


def test_llm_invocation_response_projects_failed_invocation() -> None:
    response = AgentRunLLMInvocationResponse.from_domain(
        create_failed_invocation(),
    )

    assert response.status is LLMInvocationStatus.TIMED_OUT
    assert response.usage is None
    assert response.estimated_cost.pricing_found is False
    assert response.estimated_cost.input_cost_usd is None
    assert response.estimated_cost.cached_input_cost_usd is None
    assert response.estimated_cost.output_cost_usd is None
    assert response.estimated_cost.total_cost_usd is None
    assert response.error_code is LLMErrorCode.TIMEOUT


def test_llm_invocation_response_omits_scoped_and_sensitive_fields() -> None:
    payload = AgentRunLLMInvocationResponse.from_domain(
        create_succeeded_invocation(),
    ).model_dump()

    assert "workspace_id" not in payload
    assert "ticket_id" not in payload
    assert "agent_run_id" not in payload
    assert "provider_request_id" not in payload
    assert "raw_prompt" not in payload
    assert "raw_response" not in payload
    assert "lease_token" not in payload
    assert "execution_request_id" not in payload
    assert "worker_id" not in payload


def test_llm_invocation_response_rejects_partial_token_usage() -> None:
    invocation = replace(
        create_succeeded_invocation(),
        cached_input_tokens=None,
    )

    with pytest.raises(
        RuntimeError,
        match=r"LLM invocation inspection contains partial token usage\.",
    ):
        AgentRunLLMInvocationResponse.from_domain(invocation)
