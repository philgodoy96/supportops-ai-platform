# ADR 0013: Use Optional Application-Owned Langfuse Observability

## Status

Accepted

## Context

SupportOps AI Platform already persists authoritative business, workflow, audit, usage, and estimated-cost records in PostgreSQL.

The platform also uses:

- LangGraph PostgreSQL checkpoints for workflow continuity;
- Qdrant as a rebuildable semantic-retrieval projection;
- application-owned LLM and embedding provider boundaries;
- versioned prompts and pricing catalogs;
- durable AgentRun, AgentRunAttempt, LLMInvocation, AgentToolCall, ApprovalRequest, TicketEscalation, and SupportRecommendation records.

The platform needs AI workflow observability that can correlate:

- durable AgentRuns and worker attempts;
- LangGraph workflow stages;
- LLM generation requests;
- embedding requests;
- semantic retrieval;
- tool execution;
- approval pause and resume;
- recommendation generation and grounding;
- token usage, estimated cost, latency, and normalized failures.

This observability must remain optional, privacy-aware, and isolated from business correctness.

The integration must not replace existing provider boundaries, persistence, workflow state, or audit records.

## Decision

The platform will introduce Langfuse as an optional derived observability projection behind an application-owned abstraction.

The application-owned observability boundary exposes provider-independent concepts for:

- logical traces;
- nested observations;
- discrete events;
- deterministic trace identities;
- observation status;
- usage details;
- cost details;
- privacy-aware export policies;
- explicit flush and shutdown lifecycle operations.

Two implementations are provided:

- a no-op adapter used by default;
- an optional Langfuse adapter enabled through validated configuration.

The no-op adapter requires no credentials and performs no network access.

The Langfuse adapter is the only application module allowed to import Langfuse SDK types.

### Manual instrumentation

The platform will use manual application-owned instrumentation.

The existing OpenAI client will not be replaced with a Langfuse OpenAI wrapper.

LangChain or LangGraph callbacks will not be used as the primary observability mechanism.

Manual instrumentation preserves:

- the existing provider boundary;
- application-owned retry and repair semantics;
- durable LLMInvocation persistence;
- application-calculated usage and estimated cost;
- precise privacy controls;
- support for mock and future providers;
- prevention of duplicate generation observations.

### Authoritative state

Langfuse is not authoritative for business, workflow, audit, usage, or billing state.

PostgreSQL remains the source of truth for durable business, audit, usage, and estimated-cost state.

LangGraph PostgreSQL checkpoints remain the source of truth for graph continuity and pause/resume state.

Qdrant remains a replaceable retrieval projection.

Langfuse receives optional derived telemetry for operational debugging and later evaluation workflows.

Langfuse may contain delayed, duplicated, incomplete, missing, or out-of-order telemetry.

Deterministic trace identity provides correlation. Telemetry delivery is best-effort. Retries and process failures may produce incomplete or repeated telemetry. The platform does not use Langfuse as an exactly-once ledger.

Langfuse is not an audit ledger and cannot reconstruct authoritative business state.

### Optional configuration

The default observability provider is `noop`.

Langfuse credentials are required only when the `langfuse` provider is explicitly selected.

Invalid explicitly enabled Langfuse configuration fails early.

Runtime export failures fail open.

Langfuse availability is not a readiness dependency.

### Deterministic durable-workflow correlation

One durable AgentRun maps to one logical trace.

```text
one AgentRun
→ one deterministic logical trace seed
→ re-entered across attempts, retries, pause, decision, and resume
```

The application derives the stable trace seed:

```text
agent-run:{agent_run_id}
```

The Langfuse adapter converts this seed into the provider-compatible deterministic trace ID.

No provider-specific trace IDs are persisted.

The same logical trace is reused across:

- initial worker execution;
- retries;
- approval pause;
- approval decision;
- resumed worker execution;
- final completion.

Ticket grouping uses the session identity:

```text
ticket:{ticket_id}
```

Workspace identity is stored as metadata and is not mapped to Langfuse `user_id`.

The application does not currently have an authenticated principal.

### Attempt and workflow hierarchy

Durable workflow telemetry uses nested observations:

```text
AgentRun trace
→ worker attempt
→ workflow invocation
→ graph node
→ provider/retrieval/tool observations
```

Each AgentRunAttempt is represented as a separate `worker-attempt` observation inside the logical AgentRun trace.

Controlled support uses workflow observation `workflow.controlled-support-v1`.

Human-approved support uses workflow observation `workflow.human-approved-support-v1`.

LangGraph stages are recorded as `graph-node.<node-name>` observations.

Provider generation, tool-decision, embedding, retrieval, and tool-execution observations nest under the active graph-node or workflow scope when present.

### Approval waiting

The platform does not keep an observation open while waiting for a human approval.

Approval waiting may last longer than a worker process lifetime.

Approval lifecycle transitions are represented through discrete events:

```text
approval.requested
workflow.paused
approval.approved
approval.rejected
approval.expired
workflow.resume_scheduled
workflow.resumed
```

The resumed worker attempt re-enters the same deterministic AgentRun trace.

### Outcome events

Durable workflow outcomes also emit discrete events:

```text
ticket.escalated
recommendation.generated
recommendation.persisted
recommendation.failed
```

These events remain metadata-only and do not replace PostgreSQL-authoritative escalation or recommendation records.

### Attempt-end flush

Buffered Langfuse export uses normal SDK batching by default.

An opt-in configuration flag enables one best-effort flush after each finalized AgentRun attempt:

```text
SUPPORTOPS_LANGFUSE_FLUSH_AT_ATTEMPT_END
```

Default value:

```text
false
```

When enabled, flush runs after attempt observability contexts close for finalized outcomes such as success, retryable failure, terminal failure, timeout, approval pause, and lease-lost finalization.

Flush does not run per provider call, graph node, tool call, or event, and it does not run during human waiting.

Flush is fail-open. It does not alter business outcome, retry scheduling, persistence, or worker exit behavior, and it does not replace process shutdown flushing.

### Privacy

Privacy is enforced before data reaches the Langfuse SDK.

The default capture mode is:

```text
metadata_only
```

The current implemented metadata-only policy exports safe operational metadata such as:

- technical IDs;
- workflow names and versions;
- node names;
- controlled tool identifiers;
- bounded status enums;
- counts;
- latency;
- safe normalized error codes.

Content-bearing fields are excluded, including:

- ticket content;
- graph state;
- checkpoint data;
- prompts;
- model outputs;
- tool arguments;
- tool outputs;
- approval payloads and comments;
- escalation reason;
- recommendation text;
- evidence content;
- lease tokens;
- execution grants;
- credentials;
- raw exceptions;
- document and chunk content;
- search queries;
- vectors.

An explicit opt-in mode is available:

```text
redacted_content
```

This mode accepts only structured, allowlisted fields after masking, truncation, and collection bounds.

Unrestricted raw-content capture is not supported.

Regex masking reduces exposure but does not prove de-identification.

Sensitive deployments require organization-specific data-governance, legal, and compliance review.

The Langfuse export-stage masking hook is configured only as defense in depth.

The application-owned sanitizer remains the primary privacy boundary.

### Usage and estimated cost

The application remains authoritative for provider-reported usage, pricing-catalog selection, and application-calculated estimated cost.

The Langfuse adapter receives an observability copy.

When pricing is known, the application-calculated cost is exported.

When pricing is unknown, usage may be exported while cost is omitted.

Unknown pricing is never represented as fabricated zero cost.

Catalog-defined zero cost remains valid for mock providers.

Usage and cost buckets are mutually exclusive to prevent double counting.

### Failure isolation

Observability failures do not change provider results, workflow results, persistence, retries, approval transitions, idempotency, HTTP behavior, worker outcomes, or process exit behavior.

Runtime observability failures therefore do not:

- fail an AgentRun;
- increment AgentRun retry counts;
- change ticket or approval state;
- block escalation persistence;
- roll back recommendation persistence;
- alter API response status;
- alter indexing results;
- alter process business exit codes.

Observability exceptions do not replace business exceptions.

Telemetry delivery does not provide exactly-once guarantees.

A hard process termination may lose buffered observations.

### Process lifecycle

Every process owns one observability client.

The FastAPI process:

- creates the client during lifespan startup;
- reuses it process-wide;
- shuts it down during lifespan cleanup;
- preserves partial-startup cleanup.

The worker process:

- creates the client during worker composition;
- reuses it across AgentRuns;
- shuts it down during graceful termination.

The knowledge-indexing CLI:

- creates one client per command execution;
- shuts it down in the command cleanup path;
- preserves the indexing operation exit code.

The application does not construct one client per request, AgentRun, provider call, or embedding batch.

### Deployment

The application integrates with Langfuse Cloud or an independently operated compatible Langfuse deployment.

Langfuse is not added to the local Docker Compose stack.

The application does not own Langfuse server deployment or migrations.

### Deferred capabilities

Slice 7 implements the observability foundation, Langfuse and no-op adapters, process lifecycle, privacy policy, provider observations, embedding observations, retrieval observations, indexing traces, AgentRun traces, worker-attempt observations, workflow and graph-node observations, tool observations, approval lifecycle events, escalation and recommendation outcome events, attempt-end flush policy, and trace-shape plus privacy tests.

This decision intentionally defers the following Slice 8 responsibilities:

- Langfuse evaluation workflows;
- RAGAS;
- prompt iteration based on evaluation results;
- prompt version 2;
- production feedback ingestion;
- evaluation dashboards and quality gates;
- Langfuse Prompt Management;
- remote prompt fetching;
- Langfuse datasets or experiments;
- Langfuse evaluators or scores;
- authenticated user attribution;
- general OpenTelemetry instrumentation;
- FastAPI, SQLAlchemy, or HTTP client auto-instrumentation;
- Prometheus, Grafana, Tempo, Loki, or OTLP collectors;
- invoice reconciliation or customer cost allocation.

External Langfuse smoke validation may remain an opt-in follow-up outside the normal unit suite.

## Consequences

### Positive consequences

- Observability remains replaceable and provider-independent.
- Business logic does not import Langfuse SDK types.
- Mock and future providers can use the same instrumentation boundary.
- Existing persistence and audit models remain authoritative.
- Privacy policy is explicit and testable.
- Local development and CI require no Langfuse credentials.
- Telemetry outages do not affect business correctness.
- Deterministic trace identity supports correlation across durable workflow attempts.
- Application-calculated usage and cost remain consistent with persisted records.

### Negative consequences

- Manual instrumentation requires explicit maintenance at architecture-significant boundaries.
- Telemetry may be duplicated or missing after retries and process failures.
- Buffered telemetry may be lost after hard termination.
- Metadata-only mode provides less debugging content than unrestricted tracing.
- Query-embedding and retrieval costs may be observable without being durably persisted.
- The application must maintain the mapping between its observation model and Langfuse SDK APIs.

### Operational consequences

- Langfuse credentials are needed only when explicitly enabled.
- Langfuse is not included in readiness checks.
- Long-running processes rely on SDK batching and graceful shutdown.
- Short-lived commands must shut down the client before process exit.
- Normal CI uses no-op or fake clients and performs no external Langfuse calls.

## Alternatives considered

### Replace the OpenAI client with the Langfuse OpenAI wrapper

Rejected because it would bypass or duplicate the application-owned provider boundary, persistence, retry, repair, privacy, and cost semantics.

### Use LangGraph or LangChain callbacks as the primary integration

Rejected because callbacks would not provide sufficient application ownership over durable identifiers, privacy policy, retries, repairs, tool audits, approvals, and application-calculated cost.

### Make Langfuse authoritative for usage and cost

Rejected because PostgreSQL already owns durable LLM invocation and indexing usage records, while Langfuse delivery is asynchronous and may be incomplete or duplicated.

### Make Langfuse a readiness dependency

Rejected because ticket acceptance and workflow execution must continue during telemetry outages.

### Keep a span open during human approval

Rejected because approval waits may last hours or days and outlive worker processes.

### Persist Langfuse trace IDs

Rejected because deterministic derivation and application-owned record identifiers provide sufficient correlation without adding provider-specific database state.

### Support unrestricted raw-content capture

Rejected because it creates excessive privacy, security, and compliance risk.

### Add Langfuse to the local Docker Compose stack

Rejected because self-hosting is a deployment decision rather than an application capability.

### Introduce a general OpenTelemetry platform

Rejected because Slice 7 focuses on manual AI workflow observability. General application and infrastructure observability remain separate future concerns.