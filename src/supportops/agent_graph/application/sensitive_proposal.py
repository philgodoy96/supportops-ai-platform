"""Durable preparation of sensitive tool proposals and approvals."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from supportops.agent_tools.application.persistence import (
    AgentToolCallExecutionRepository,
    AgentToolCallPersistenceResult,
    PersistAgentToolCallCommand,
)
from supportops.agent_tools.application.queries import (
    AgentToolCallQueryRepository,
    SensitiveAgentToolCallLookup,
)
from supportops.agent_tools.application.sensitive_bindings import (
    SensitiveToolRegistry,
)
from supportops.agent_tools.domain.audit import AgentToolCall
from supportops.agent_tools.domain.contracts import StrictToolSchema
from supportops.agent_tools.domain.fingerprints import (
    create_tool_call_fingerprint,
)
from supportops.core.transactions import TransactionManager
from supportops.modules.agent_runs.application.execution import (
    AgentRunExecutionContext,
    RetryableAgentRunExecutionError,
)
from supportops.modules.approvals.domain.models import ApprovalRequest
from supportops.modules.approvals.domain.repositories import (
    ApprovalRequestPersistenceResult,
    ApprovalRequestRepository,
)
from supportops.observability.contracts import ObservabilityClient
from supportops.observability.identity import agent_run_trace_identity
from supportops.observability.models import (
    EventObservation,
    FieldPaths,
    ObservationStatus,
)
from supportops.observability.noop import NoOpObservabilityClient

type UtcNowProvider = Callable[[], datetime]
type UuidFactory = Callable[[], UUID]

_APPROVAL_REQUESTED_EVENT_METADATA_PATHS: FieldPaths = frozenset(
    {
        ("approval_request_id",),
        ("agent_run_id",),
        ("agent_run_attempt_id",),
        ("workspace_id",),
        ("ticket_id",),
        ("tool_name",),
        ("approval_status",),
        ("expires_at",),
    }
)


@dataclass(frozen=True, slots=True)
class SensitiveProposalCommand:
    """Validated model decision prepared for durable persistence."""

    provider_tool_call_id: str
    tool_name: str
    tool_version: int
    arguments: StrictToolSchema
    requested_by_llm_invocation_id: UUID
    sequence: int

    def __post_init__(self) -> None:
        if not self.provider_tool_call_id:
            raise ValueError("provider_tool_call_id is required.")
        if self.provider_tool_call_id != (self.provider_tool_call_id.strip()):
            raise ValueError(
                "provider_tool_call_id must not contain surrounding whitespace.",
            )
        if not self.tool_name:
            raise ValueError("tool_name is required.")
        if self.tool_name != self.tool_name.strip():
            raise ValueError(
                "tool_name must not contain surrounding whitespace.",
            )
        if self.tool_version < 1:
            raise ValueError("tool_version must be positive.")
        if not isinstance(self.arguments, StrictToolSchema):
            raise TypeError(
                "arguments must inherit StrictToolSchema.",
            )
        if not isinstance(
            self.requested_by_llm_invocation_id,
            UUID,
        ):
            raise TypeError(
                "requested_by_llm_invocation_id must be a UUID.",
            )
        if type(self.sequence) is not int:
            raise TypeError("sequence must be an integer.")
        if self.sequence < 1:
            raise ValueError("sequence must be positive.")


@dataclass(frozen=True, slots=True)
class SensitiveProposalOutcome:
    """Durable proposal and approval selected for interruption."""

    tool_call: AgentToolCall
    approval_request: ApprovalRequest
    tool_call_created: bool
    approval_request_created: bool


class SensitiveProposalConsistencyError(RuntimeError):
    """Raised when durable proposal state conflicts with replay."""


class SensitiveProposalService:
    """Persist or reuse one sensitive proposal and its approval."""

    def __init__(
        self,
        *,
        transaction_manager: TransactionManager,
        sensitive_tool_registry: SensitiveToolRegistry,
        tool_call_execution_repository: (AgentToolCallExecutionRepository),
        tool_call_query_repository: AgentToolCallQueryRepository,
        approval_request_repository: ApprovalRequestRepository,
        approval_ttl_seconds: float,
        utc_now: UtcNowProvider | None = None,
        uuid_factory: UuidFactory = uuid4,
        observability_client: ObservabilityClient | None = None,
    ) -> None:
        if approval_ttl_seconds <= 0:
            raise ValueError(
                "approval_ttl_seconds must be positive.",
            )
        self._transaction_manager = transaction_manager
        self._sensitive_tool_registry = sensitive_tool_registry
        self._tool_call_execution_repository = tool_call_execution_repository
        self._tool_call_query_repository = tool_call_query_repository
        self._approval_request_repository = approval_request_repository
        self._approval_ttl_seconds = approval_ttl_seconds
        self._utc_now = utc_now or _utc_now
        self._uuid_factory = uuid_factory
        self._observability_client = observability_client or NoOpObservabilityClient()

    async def execute(
        self,
        *,
        context: AgentRunExecutionContext,
        command: SensitiveProposalCommand,
    ) -> SensitiveProposalOutcome:
        """Persist both durable records before graph interruption."""

        binding = self._sensitive_tool_registry.lookup(
            name=command.tool_name,
            version=command.tool_version,
        )
        if not isinstance(
            command.arguments,
            binding.definition.input_schema,
        ):
            raise ValueError(
                "Sensitive proposal arguments do not match the registered schema.",
            )

        safe_input = dict(
            binding.safe_input_projector(command.arguments),
        )
        request_reason = binding.approval_reason_projector(
            command.arguments,
        )
        fingerprint = create_tool_call_fingerprint(
            definition=binding.definition,
            arguments=command.arguments,
        )
        proposed_at = self._utc_now()
        candidate = AgentToolCall.propose_for_approval(
            workspace_id=context.agent_run.workspace_id,
            ticket_id=context.ticket.id,
            agent_run_id=context.agent_run.id,
            proposed_by_agent_run_attempt_id=context.attempt.id,
            sequence=command.sequence,
            provider_tool_call_id=(command.provider_tool_call_id),
            tool_name=binding.definition.name,
            tool_version=binding.definition.version,
            input_fingerprint=fingerprint,
            safe_input=safe_input,
            proposed_at=proposed_at,
            tool_call_id=self._uuid_factory(),
        )

        async with self._transaction_manager.transaction():
            tool_result = await self._tool_call_execution_repository.persist_fenced(
                PersistAgentToolCallCommand(
                    workspace_id=(context.agent_run.workspace_id),
                    ticket_id=context.ticket.id,
                    agent_run_id=context.agent_run.id,
                    agent_run_attempt_id=context.attempt.id,
                    lease_token=context.attempt.lease_token,
                    persisted_at=proposed_at,
                    tool_call=candidate,
                ),
            )
            durable_tool_call = await self._resolve_tool_call(
                context=context,
                candidate=candidate,
                persistence_result=tool_result,
            )
            expires_at = durable_tool_call.proposed_at + timedelta(
                seconds=self._approval_ttl_seconds,
            )
            approval_candidate = ApprovalRequest.create_pending(
                tool_call=durable_tool_call,
                requested_by_llm_invocation_id=(command.requested_by_llm_invocation_id),
                request_reason=request_reason,
                expires_at=expires_at,
                approval_request_id=self._uuid_factory(),
                now=durable_tool_call.proposed_at,
            )
            approval_result = await self._approval_request_repository.persist_pending(
                approval_candidate,
            )
            durable_approval = await self._resolve_approval(
                workspace_id=context.agent_run.workspace_id,
                candidate=approval_candidate,
                persistence_result=approval_result,
            )

        outcome = SensitiveProposalOutcome(
            tool_call=durable_tool_call,
            approval_request=durable_approval,
            tool_call_created=(tool_result is AgentToolCallPersistenceResult.APPLIED),
            approval_request_created=(approval_result is ApprovalRequestPersistenceResult.APPLIED),
        )
        if outcome.approval_request_created:
            _safe_record_approval_requested(
                client=self._observability_client,
                context=context,
                approval_request=durable_approval,
            )
        return outcome

    async def _resolve_tool_call(
        self,
        *,
        context: AgentRunExecutionContext,
        candidate: AgentToolCall,
        persistence_result: AgentToolCallPersistenceResult,
    ) -> AgentToolCall:
        if persistence_result is AgentToolCallPersistenceResult.LEASE_LOST:
            raise RetryableAgentRunExecutionError(
                error_code="sensitive_proposal_lease_lost",
                error_summary=(
                    "The AgentRun lease was lost before the sensitive proposal could be persisted."
                ),
            )
        if persistence_result is AgentToolCallPersistenceResult.APPLIED:
            return candidate
        if persistence_result is not AgentToolCallPersistenceResult.ALREADY_RECORDED:
            raise RuntimeError(
                "Sensitive proposal persistence returned an unsupported result.",
            )

        existing = await self._tool_call_query_repository.get_sensitive_by_identity(
            SensitiveAgentToolCallLookup(
                workspace_id=(context.agent_run.workspace_id),
                ticket_id=context.ticket.id,
                agent_run_id=context.agent_run.id,
                tool_name=candidate.tool_name,
                tool_version=candidate.tool_version,
                input_fingerprint=(candidate.input_fingerprint),
            ),
        )
        if existing is None:
            raise SensitiveProposalConsistencyError(
                "A recorded sensitive proposal could not be loaded.",
            )
        _validate_matching_tool_proposal(
            existing=existing,
            candidate=candidate,
        )
        return existing

    async def _resolve_approval(
        self,
        *,
        workspace_id: UUID,
        candidate: ApprovalRequest,
        persistence_result: ApprovalRequestPersistenceResult,
    ) -> ApprovalRequest:
        if persistence_result is ApprovalRequestPersistenceResult.APPLIED:
            return candidate
        if persistence_result is not ApprovalRequestPersistenceResult.ALREADY_RECORDED:
            raise RuntimeError(
                "Approval persistence returned an unsupported result.",
            )

        existing = await self._approval_request_repository.get_by_agent_tool_call_id(
            workspace_id=workspace_id,
            agent_tool_call_id=candidate.agent_tool_call_id,
        )
        if existing is None:
            raise SensitiveProposalConsistencyError(
                "A recorded approval request could not be loaded.",
            )
        if not existing.matches_pending_proposal(candidate):
            raise SensitiveProposalConsistencyError(
                "The existing approval request conflicts with the sensitive proposal replay.",
            )
        return existing


def _validate_matching_tool_proposal(
    *,
    existing: AgentToolCall,
    candidate: AgentToolCall,
) -> None:
    values = (
        existing.workspace_id == candidate.workspace_id,
        existing.ticket_id == candidate.ticket_id,
        existing.agent_run_id == candidate.agent_run_id,
        existing.tool_name == candidate.tool_name,
        existing.tool_version == candidate.tool_version,
        existing.safety_level is candidate.safety_level,
        existing.input_fingerprint == candidate.input_fingerprint,
        dict(existing.safe_input) == dict(candidate.safe_input),
    )
    if not all(values):
        raise SensitiveProposalConsistencyError(
            "The existing sensitive proposal conflicts with the replayed proposal.",
        )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _safe_record_approval_requested(
    *,
    client: ObservabilityClient,
    context: AgentRunExecutionContext,
    approval_request: ApprovalRequest,
) -> None:
    try:
        client.record_trace_event(
            identity=agent_run_trace_identity(
                agent_run_id=approval_request.agent_run_id,
                ticket_id=approval_request.ticket_id,
            ),
            event=EventObservation(
                name="approval.requested",
                status=ObservationStatus.OK,
                metadata={
                    "approval_request_id": str(approval_request.id),
                    "agent_run_id": str(approval_request.agent_run_id),
                    "agent_run_attempt_id": str(context.attempt.id),
                    "workspace_id": str(approval_request.workspace_id),
                    "ticket_id": str(approval_request.ticket_id),
                    "tool_name": approval_request.tool_name,
                    "approval_status": approval_request.status.value,
                    "expires_at": approval_request.expires_at.isoformat(),
                },
                metadata_paths=_APPROVAL_REQUESTED_EVENT_METADATA_PATHS,
            ),
        )
    except Exception:
        return
