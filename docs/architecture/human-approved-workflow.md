# Human-Approved Support Workflow

## Purpose

`human-approved-support-v1` adds a durable human approval boundary to sensitive support operations.

It introduces persisted sensitive proposals, durable approval requests, LangGraph interruption and resumption, PostgreSQL-backed authorization, idempotent sensitive execution, immutable ticket escalation records, and deterministic crash recovery.

The workflow is versioned separately from `controlled-support-v1`. Existing workflow identities, checkpoint threads, prompt versions, and historical behavior remain unchanged.

## Workflow Identity

```text
workflow_name: ticket-processing
workflow_version: human-approved-support-v1
state_schema_version: human-approved-support-state-v1
graph_version: graph-v1
```

Thread identity:

```text
ticket-processing:human-approved-support-v1:graph-v1:{agent_run_id}
```

`controlled-support-v1` remains the default workflow.

## Durable Proposal Boundary

Sensitive execution is never requested before business state is durable:

```text
AgentToolCall.pending_approval
-> ApprovalRequest.pending
-> checkpoint durable identifiers
-> LangGraph interrupt
```

Proposal and approval persistence are idempotent. Replays reuse matching durable records. Conflicting replay fails closed.

## Safe Interrupt Payload

The interrupt contains only:

```text
approval_request_id
agent_tool_call_id
agent_run_id
ticket_id
tool_name
tool_version
proposed_input
request_reason
expires_at
```

It excludes workspace authorization data, lease state, attempt identifiers, ORM objects, raw prompts, raw model output, provider clients, token and cost data, approval actor metadata, request metadata, correlation metadata, and secrets.

The interrupt payload is a navigation signal. It is not authorization.

## AgentRun Lifecycle

Initial execution:

```text
queued
-> running
-> waiting_for_approval
```

The active attempt ends with `awaiting_approval`.

After approval, rejection, or expiration:

```text
waiting_for_approval
-> queued
-> running
```

The resume claim creates a new `AgentRunAttempt`.

Every real claim increments `attempt_count`. Approval pause and resume do not increment `retryable_failure_count`.

## Resume Planning

The worker selects one execution plan:

```text
InitialGraphExecution
ContinueGraphExecution
ResumeGraphExecution
CompletedGraphExecution
IncompatibleGraphState
```

`ResumeGraphExecution` requires one valid interrupt plus matching PostgreSQL state for the workspace, ticket, AgentRun, ApprovalRequest, and AgentToolCall.

The executor resumes with:

```python
Command(
    resume={
        "approval_request_id": "...",
        "agent_tool_call_id": "...",
        "decision_status": "approved | rejected | expired",
    },
)
```

The same LangGraph thread identity is reused.

A pending approval, missing checkpoint, incompatible checkpoint, ownership mismatch, or proposal mismatch produces `IncompatibleGraphState` and fails closed.

## PostgreSQL as Business Authority

LangGraph checkpoints preserve workflow continuity.

PostgreSQL remains authoritative for:

- ownership;
- AgentRun lifecycle;
- approval status;
- approval decision metadata;
- AgentToolCall lifecycle;
- sensitive execution authorization;
- ticket escalation persistence;
- recommendation persistence.

Neither a checkpoint nor a resume payload authorizes sensitive execution.

## Approval Outcomes

### Approved

```text
ApprovalRequest.approved
-> SensitiveExecutionGrant
-> TicketEscalation
-> AgentToolCall.succeeded
-> grounded recommendation
```

### Rejected

Rejected approvals create no grant, no escalation, and no sensitive execution. The workflow produces a recommendation that explicitly states the action was not executed.

### Expired

Expired approvals follow the same non-execution guarantees as rejected approvals.

### Pending

Pending approvals cannot resume execution.

## Sensitive Execution Grant

`SensitiveExecutionGrant` is the only authorization accepted by sensitive execution.

It is immutable and bound to exactly one:

- workspace;
- ticket;
- AgentRun;
- resume AgentRunAttempt;
- ApprovalRequest;
- AgentToolCall;
- tool identity;
- input fingerprint;
- granted input;
- approval actor;
- decision request;
- decision correlation;
- approval timestamp.

It is unique per ApprovalRequest and per AgentToolCall.

A grant represents authorization, not successful execution.

## Ticket Escalation

`TicketEscalation` is an immutable side record under the tickets bounded context.

It records the workspace, ticket, AgentRun, execution attempt, approval, tool call, bounded target queue, reason, and creation timestamp.

It is unique per approval and per tool call.

`Ticket.status` is intentionally not mutated.

## Transaction Boundary

Approved internal execution uses one short transaction:

```text
lock ApprovalRequest
-> lock AgentToolCall
-> validate ownership and proposal identity
-> persist or reuse SensitiveExecutionGrant
-> persist or reuse TicketEscalation
-> persist AgentToolCall success
-> commit
```

The transaction does not span LLM calls, checkpoint I/O, external calls, sleeps, or retry delays.

## AgentToolCall Completion

Granted execution transitions:

```text
pending_approval
-> succeeded
```

The original proposal attempt remains unchanged. The resume attempt is stored as `executed_by_agent_run_attempt_id`.

Safe output contains only:

```text
escalation_id
ticket_id
target_queue
status
```

Terminal tool-call states are never overwritten.

## Idempotency and Recovery

The workflow safely recovers from:

- crash after tool proposal;
- crash after approval persistence;
- crash after interrupt checkpoint;
- crash after grant creation;
- crash after escalation persistence;
- crash after tool-call completion;
- duplicate `Command(resume=...)`;
- missing checkpoint with approved PostgreSQL state;
- checkpoint resume while PostgreSQL remains pending;
- concurrent duplicate execution.

Concurrent duplicate execution converges to:

```text
one SensitiveExecutionGrant
one TicketEscalation
one succeeded AgentToolCall
```

Conflicting replay fails closed and never overwrites durable state.

## Recommendation Integrity

The workflow uses:

```text
human-approved-support-recommendation / 1
```

It preserves invocation persistence, token accounting, estimated cost tracking, validation, and one-recommendation-per-AgentRun semantics.

An approved recommendation may claim escalation only when safe execution output confirms `status=escalated`.

Rejected and expired recommendations contain no execution output.

## Security Model

Current guarantees include:

- workspace-scoped reads;
- ownership validation;
- bounded sensitive input;
- safe interrupt payloads;
- PostgreSQL-backed authorization;
- immutable execution grants;
- no execution from approval API handlers;
- no external side-effect tools;
- fail-closed checkpoint mismatch handling.

Authentication and RBAC are intentionally deferred to a separate slice. Actor references remain asserted rather than verified.

## Intentional Scope Boundaries

This workflow does not include:

- approval inspection APIs;
- approval decision HTTP endpoints;
- ticket escalation inspection APIs;
- authentication or RBAC;
- external side-effect tools;
- notifications;
- mutable ticket escalation state;
- prompt version 2;
- evaluation datasets and scoring.

Operational APIs are introduced separately so the interrupt, authorization, and recovery design remains independent from inspection concerns.

## Operational Validation

Coverage includes:

- domain invariants;
- repository idempotency;
- rollback and savepoint behavior;
- concurrency;
- interrupt and resume;
- approved, rejected, and expired paths;
- crash recovery;
- AgentRun lifecycle;
- historical controlled-support regressions.

Current Alembic head:

```text
b8d3f6a1c9e4
```