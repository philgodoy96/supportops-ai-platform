# ADR 0012: Use Application-Owned Approval Records with LangGraph Interrupts

## Status

Accepted

## Context

SupportOps AI Platform uses LangGraph to coordinate bounded support-ticket workflows while PostgreSQL remains authoritative for business records such as tickets, classifications, agent runs, tool-call audits, and support recommendations.

The controlled support workflow introduced durable PostgreSQL-backed LangGraph checkpoints. Those checkpoints preserve graph progression, node state, and workflow continuity across process restarts. They are operational workflow state, not an application-owned business record.

Slice 6 introduces the first consequential internal write proposed by the model: creating an immutable ticket escalation record. The action must not execute until a human decision has been recorded. Human review may take minutes, hours, or days, so the worker cannot retain a lease, open execution attempt, database transaction, provider request, or process task while approval is pending.

A LangGraph interrupt can suspend graph execution, but interrupt state alone does not provide the business semantics required to:

- list pending approvals;
- display the exact safe proposed action;
- record asserted operator attribution;
- enforce concurrency between approval, rejection, and expiration;
- preserve terminal decisions independently of checkpoint inspection;
- audit non-execution after rejection or expiration;
- associate the decision with a sensitive state mutation;
- recover safely after worker or process failure.

The existing AgentRun retry model also uses total execution attempts as the retry-exhaustion budget. Human pauses require a new worker claim when the graph resumes, but approval waits and resumes are not operational failures. Charging those claims against the failure retry budget would allow healthy approval workflows to exhaust retry capacity.

Authentication, authorization, verified operator identity, multi-approver policies, and real external support-system integrations are intentionally outside this slice.

## Decision

### PostgreSQL owns approval semantics

The platform will introduce an application-owned `ApprovalRequest` record in PostgreSQL.

`ApprovalRequest` is authoritative for:

- workspace, ticket, and AgentRun ownership;
- the proposed `AgentToolCall`;
- tool name and version;
- tool safety level;
- a normalized input fingerprint;
- the immutable safe proposed input;
- the bounded request reason;
- approval status;
- expiration;
- asserted decision attribution;
- decision request and correlation identifiers;
- decision comments;
- decision timestamps.

The supported approval states are:

- `pending`;
- `approved`;
- `rejected`;
- `expired`.

Terminal approval decisions are immutable.

A proposed sensitive `AgentToolCall` is persisted before the graph interrupts. One proposed sensitive tool call may have at most one `ApprovalRequest`.

### LangGraph checkpoints own workflow suspension and continuity

The human-approved workflow will call LangGraph `interrupt()` at a dedicated approval boundary.

The interrupt payload contains only safe JSON-compatible identifiers and projections required to present the pending action. It does not contain secrets, raw provider payloads, hidden reasoning, database sessions, lease tokens, checkpoint internals, or an execution grant.

Checkpoint state remains authoritative only for:

- graph progression;
- the active node;
- bounded graph state;
- interrupt state;
- resume continuity.

The checkpoint does not authorize a sensitive action.

### Approval creation is idempotent before interruption

LangGraph may restart an interrupted node from its beginning. A process may also fail after PostgreSQL commits an approval record but before the next checkpoint is durably advanced.

Therefore:

- proposed sensitive tool-call persistence is idempotent;
- approval-request persistence is idempotent;
- repeated node execution loads the existing records;
- the same logical proposal fingerprint cannot create another approval;
- no consequential write occurs before `interrupt()`;
- the interrupt order remains stable.

### AgentRun enters a lease-free waiting state

When execution reaches the approval interrupt, the executor reports a typed paused result.

The AgentRun processor then atomically:

- closes the current `AgentRunAttempt` with the non-failure outcome `awaiting_approval`;
- transitions the AgentRun from `running` to `waiting_for_approval`;
- clears the worker lease owner, token, and expiry;
- leaves the run incomplete;
- leaves failure-only error fields unset.

A waiting run is not claimable and is not retry-scheduled.

The worker does not remain active while a human decision is pending.

### Approval decisions requeue the AgentRun

Approval and rejection endpoints persist decisions but do not initialize, compile, inspect, or resume LangGraph.

The API process owns:

- request validation;
- workspace-scoped approval lookup;
- asserted operator attribution;
- concurrency-safe decision persistence;
- expiration checks;
- atomic AgentRun requeue;
- response serialization.

The worker process owns:

- graph reconstruction;
- checkpoint inspection;
- resume planning;
- `Command(resume=...)`;
- sensitive tool execution;
- final recommendation completion.

Approval, rejection, and expiration transition the waiting AgentRun to `queued` in the same transaction that persists the terminal approval state. The next worker claim creates a new execution attempt.

Rejected and expired workflows are requeued because the graph must resume to record safe non-execution and produce an accurate final recommendation. They are not execution failures.

### Resume reuses the same workflow thread

The resumed workflow uses the same deterministic AgentRun-derived LangGraph thread identity as the interrupted execution.

Before issuing `Command(resume=...)`, the worker verifies:

- the AgentRun uses the human-approved workflow version;
- a terminal approval decision exists;
- the approval belongs to the AgentRun, workspace, and ticket;
- the approval references the expected proposed tool call;
- tool name, version, and input fingerprint match;
- the checkpoint thread belongs to the same AgentRun;
- the checkpoint is currently interrupted at the expected approval boundary;
- the interrupt payload references the same `ApprovalRequest`;
- no incompatible final recommendation already exists.

The resume payload is a small JSON-compatible projection derived from PostgreSQL. It is not authorization. The graph validates the persisted `ApprovalRequest` after receiving the resume value.

### Human pauses do not consume failure retry capacity

The AgentRun execution model will separate:

- total worker execution attempts;
- retryable failure count;
- maximum retryable failures.

Every worker claim remains auditable through the total attempt count.

Only retryable operational failures consume retry capacity. Approval pauses, approved resumes, rejected resumes, and expired resumes do not consume the failure budget.

The existing retry budget will not be increased dynamically to hide approval resumes.

### Sensitive execution requires a persisted grant

The model may propose only tools registered by the application.

Tool safety policy is:

- `read_only`: executable through the existing bounded read-only path;
- `sensitive_write`: executable only with a matching persisted approval grant;
- `external_side_effect`: unavailable in Slice 6.

A boolean such as `approved=True` is insufficient.

The application constructs an immutable sensitive execution grant only after validating the authoritative approval record. The grant binds:

- approval request;
- proposed tool call;
- tool name and version;
- normalized input fingerprint;
- workspace;
- ticket;
- AgentRun;
- terminal approved decision.

Any mismatch fails closed.

### The initial sensitive action is an internal escalation record

Slice 6 introduces exactly one `sensitive_write` tool: `escalate_ticket`.

The tool creates an immutable internal `TicketEscalation` record. It does not modify ticket status and does not call Jira, ServiceNow, Slack, email, webhooks, or another external system.

The escalation record is idempotent by approval identity. Repeated execution after a crash:

- returns the existing matching escalation;
- does not create another escalation;
- fails with a terminal consistency error if persisted data conflicts with the approved input.

This is idempotent state convergence, not exactly-once execution.

### Asserted operator attribution is not authentication

Approval and rejection requests require an `actor_reference`.

The value is caller-supplied, validated, and persisted for workflow auditability. It is not treated as a verified identity or authenticated principal.

The approval API is not described as a secure public multi-tenant authorization boundary.

A future authenticated principal and authorization policy layer will replace or supplement asserted attribution without changing PostgreSQL ownership of approval decisions.

### Workflow compatibility is versioned

The existing controlled workflow and its active checkpoint node names remain unchanged.

Human approval is introduced through a new exact workflow version. Interrupted threads depend on their workflow version, graph version, state schema, node names, and deterministic thread identity.

Future incompatible graph changes require another workflow version or an explicit migration strategy.

## Consequences

### Positive consequences

- Business approvals remain inspectable without reading checkpoint tables.
- Terminal decisions survive graph completion and checkpoint cleanup.
- Approval, rejection, expiration, and requeue can be transactionally coordinated.
- Workers remain available while humans review actions.
- Human review does not consume operational retry capacity.
- Sensitive writes require durable matching evidence rather than an in-memory flag.
- Repeated execution converges without duplicate escalation records.
- The API and worker retain separate process responsibilities.
- Historical workflows and interrupted checkpoints remain compatible.
- Authentication and external integrations can be added later without redefining approval ownership.

### Costs and trade-offs

- Approval and checkpoint persistence cannot share one database transaction, so idempotent recovery is required across the commit gap.
- AgentRun, AgentRunAttempt, and AgentToolCall lifecycles become more expressive.
- Resume planning must inspect and validate both checkpoint state and PostgreSQL business records.
- Approval expiration adds bounded worker-cycle recovery work.
- The platform must maintain workflow-version compatibility while interrupted runs exist.
- Concurrency tests are required for approval, rejection, and expiration races.
- Exactly-once graph resume and exactly-once side effects are not claimed.

### Intentionally deferred capabilities

The following capabilities remain outside Slice 6:

- authenticated operator identity;
- authorization policies and role-based approval permissions;
- workspace memberships;
- edited tool arguments during review;
- approval cancellation or delegation;
- multi-approver and quorum policies;
- real Jira, ServiceNow, Slack, or email actions;
- provider-side external idempotency adapters;
- arbitrary HTTP actions;
- frontend approval interfaces;
- Langfuse;
- RAGAS;
- prompt optimization.

## Alternatives considered

### Use LangGraph interrupt state as the approval record

Rejected because checkpoint state is operational workflow state. It does not provide the application-owned listing, attribution, concurrency, expiration, audit, and historical guarantees required for a business approval.

### Keep the worker lease while approval is pending

Rejected because human review is unbounded relative to worker execution. Holding the lease would waste worker capacity, create stale ownership, keep attempts artificially open, and complicate process recovery.

### Resume LangGraph directly from the approval API

Rejected because it would move worker responsibilities into the HTTP process, bypass AgentRun claims and attempt history, complicate provider and checkpoint resource lifecycles, and weaken retry and lease fencing.

### Treat rejection or expiration as an AgentRun failure

Rejected because neither outcome is an execution failure. The graph must resume to persist accurate non-execution state and produce a safe final recommendation.

### Increase `max_attempts` after every approval pause

Rejected because it conceals the semantic difference between worker executions and retryable failures, creates mutable budgets, and remains difficult to explain and test.

### Authorize sensitive execution with an in-memory boolean

Rejected because an in-memory flag does not bind the approval to tool identity, version, input fingerprint, workspace, ticket, AgentRun, or checkpoint interrupt.

### Execute a real external escalation integration

Deferred intentionally. An immutable internal escalation record demonstrates approval authority, sensitive-write gating, idempotency, and auditability without introducing provider credentials, network delivery ambiguity, or external idempotency contracts.

### Modify the existing controlled workflow in place

Rejected because active historical checkpoints depend on their workflow version, state schema, graph version, and node names. Human approval requires a new versioned workflow contract.