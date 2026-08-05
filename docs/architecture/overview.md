# Architecture Overview

## Purpose

SupportOps AI Platform is a production-minded backend and AI systems engineering project designed to support reliable support operations, controlled AI orchestration, retrieval-augmented workflows, human approval, observability, and evaluation.

The platform is intentionally structured as an API-first modular monolith. This architecture keeps deployment and operational complexity controlled while preserving clear internal boundaries that can evolve as the system grows.

The current repository phase establishes the operational foundation, workspace-scoped persistence, the Slice 1 workspace and ticket HTTP API, durable AgentRun scheduling, the PostgreSQL-backed worker, workspace-scoped AgentRun inspection, the application-owned LLM Gateway, durable structured ticket classification with invocation and accepted classification persistence, classification inspection, logical invocation inspection, repository-owned offline deterministic evaluation with versioned datasets, split manifests, typed prediction envelopes, canonical hashing, atomic artifact writes, explicit prompt-version selection, and standalone classification release gates, opt-in provider evaluation, deterministic semantic-retrieval, controlled-support, and human-approval regression scoring over committed static fixtures, repository-level deterministic regression aggregation through `supportops-evaluate-regression score`, grounded recommendation evaluation with committed synthetic fixtures, deterministic complementary metrics, normalized static RAGAS score artifacts, offline RAGAS aggregation, an evaluation-only RAGAS dependency boundary, an explicit external RAGAS runner, and a committed human review rubric, PostgreSQL-authoritative immutable knowledge-document versioning, deterministic chunking, embedding providers, explicit Qdrant indexing, active-version semantic knowledge retrieval with authoritative PostgreSQL hydration, the controlled support workflow that combines LangGraph orchestration, bounded read-only tool execution, durable tool-call auditing, grounded recommendation drafting, recommendation and citation persistence, and workspace-scoped controlled support inspection, and the separately versioned human-approved support workflow that adds durable sensitive proposals, approval interruption and resume, grant-gated sensitive execution, immutable ticket escalation, and workspace-scoped approval list/detail, approve/reject command, and ticket escalation list/detail APIs with worker-owned resume. External side-effect tools, reranking, a real canonical external RAGAS baseline, classification prompt version 2, and paired prompt comparison remain later phases.

The controlled support workflow is documented in [`controlled-support-workflow.md`](controlled-support-workflow.md). The human-approved support workflow is documented in [`human-approved-workflow.md`](human-approved-workflow.md). Approval inspection, decision, and escalation inspection HTTP contracts are documented in [`../development/approval-workflow-api.md`](../development/approval-workflow-api.md). Durability boundaries for AgentRun and LangGraph checkpoint ownership are recorded in [`../decisions/0010-separate-agent-run-and-langgraph-durability.md`](../decisions/0010-separate-agent-run-and-langgraph-durability.md) and [`../decisions/0011-treat-langgraph-checkpoints-as-framework-owned-schema.md`](../decisions/0011-treat-langgraph-checkpoints-as-framework-owned-schema.md).

### Human-Approved Support Workflow

`human-approved-support-v1` is versioned separately from `controlled-support-v1`. `controlled-support-v1` remains the default ticket-processing workflow. Sensitive actions require durable approval before execution. PostgreSQL remains the business authority for ownership, approval status, grants, tool-call lifecycle, escalations, and recommendations. LangGraph checkpoints preserve workflow continuity across interrupt and resume. Sensitive execution is grant-gated through immutable `SensitiveExecutionGrant` records. `TicketEscalation` is an immutable side record and does not mutate `Ticket.status`. Workspace-scoped approval list/detail, approve/reject commands, and escalation list/detail endpoints exist. HTTP decisions requeue the waiting `AgentRun`; the worker owns claim, checkpoint validation, LangGraph resume, grant consumption, and sensitive execution. Missing and foreign resources share nondisclosing `404` contracts. External side-effect tools intentionally remain unavailable. Full interrupt, resume, recovery, and authorization semantics are documented in [`human-approved-workflow.md`](human-approved-workflow.md). Operational API contracts are documented in [`../development/approval-workflow-api.md`](../development/approval-workflow-api.md).

## Architectural principles

The architecture follows these principles:

- keep business logic independent from transport and infrastructure frameworks;
- preserve explicit dependency direction;
- treat PostgreSQL as the transactional source of truth;
- treat Qdrant as a rebuildable retrieval index;
- introduce modules only when they have concrete responsibilities;
- prefer simple operational models over premature distributed systems;
- isolate external providers behind application-owned adapters;
- design for testability and observable failure behavior;
- document intentional scope boundaries and trade-offs;
- evolve the system through small, reviewable implementation slices.

## System shape

The platform is a modular monolith with three deployable runtime entry points:

- an HTTP API process;
- a PostgreSQL AgentRun worker process;
- a one-shot `supportops-index-knowledge` indexing process.

These processes share:

- the same Python package;
- the same domain model;
- the same application services where required;
- the same PostgreSQL database;
- infrastructure adapters required by each process.

The processes have distinct runtime responsibilities while remaining part of one deployable codebase.

The API, worker, and indexing CLI remain separate runtime entry points with independently owned resources.

Current ownership:

- the API owns HTTP acceptance, transactional Ticket plus configured initial AgentRun scheduling, read-only AgentRun, classification, logical invocation, controlled support, and ticket escalation inspection, workspace-scoped approval inspection and approve/reject command submission with durable terminal decision persistence and AgentRun requeue, workspace-scoped knowledge-document registration, versioning, inspection, and explicit activation, and semantic knowledge retrieval through its own process-scoped embedding provider, Qdrant client, and immutable retrieval profile;
- the worker owns process-scoped LLM provider and Gateway composition, the PostgreSQL checkpoint runtime, its own embedding provider, Qdrant client, immutable knowledge index profile, and vector search adapter, expired lease recovery, claim, versioned executor dispatch, classification, controlled graph execution, bounded read-only tool execution, human-approved interrupt and resume, grant-gated sensitive execution, ticket escalation persistence, recommendation drafting, and fenced outcome persistence;
- the indexing CLI remains a separate one-shot process that owns its own PostgreSQL engine and session factory, Qdrant client, and embedding provider for explicit collection bootstrap and version indexing;
- PostgreSQL owns tickets, AgentRuns, attempts, leases, retry scheduling, execution history, logical invocations, accepted classifications, controlled tool-call audits, support recommendations, recommendation citations, knowledge documents, immutable source versions, immutable index profiles, authoritative chunk records, indexing lifecycle metadata, embedding usage and cost provenance, failure provenance, active-version pointers, and retrieval evidence content, and also stores LangGraph checkpoints whose schema remains framework-owned;
- repository-owned evaluation owns versioned synthetic datasets, split manifests, evaluation manifests, typed prediction envelopes, deterministic metrics, standalone release gates, committed static multi-domain prediction fixtures, repository regression aggregation, grounded recommendation fixtures and complementary metrics, normalized static RAGAS score artifacts, offline RAGAS aggregation, the evaluation-only RAGAS boundary and external runner, the grounded human review rubric, and reproducible reports outside the API and worker processes, and remains separate from runtime business authority and optional observability;
- Qdrant owns only the rebuildable dense-vector candidate projection used for indexing writes, public semantic search, and controlled workflow knowledge search, and is not required by ticket classification or the deterministic baseline;
- optional Langfuse observability receives derived telemetry only and is not the evaluation source of truth.

API document registration remains source-only. API business routes may use Qdrant for semantic search. The worker now owns the LLM, checkpoint, embedding, and Qdrant resources required by `controlled-support-v1`, while the API independently owns the retrieval resources used by the public semantic search endpoint. Indexing does not activate a document version.

The two durability boundaries are explicit. AgentRun owns outer durability, including scheduling, claiming, attempts, leases, timeout, retries, and final success or failure. LangGraph owns bounded inner orchestration, including node routing, the decision and tool loop, and checkpoint resume. LangGraph never transitions the AgentRun itself.

The platform separates three planes:

```text
runtime plane
→ worker execution and durable persistence

inspection plane
→ workspace-scoped read-only API

evaluation plane
→ repository-owned datasets, splits, static prediction fixtures, metrics, release gates, repository regression aggregates, grounded recommendation evaluation, and reports
```

The evaluation plane separates runtime recommendation generation, deterministic offline evaluation, external model-based evaluation, and human qualitative review. Evaluation remains separate from runtime business authority and from optional observability. Deterministic regression scoring consumes committed static fixtures for semantic retrieval, controlled support, and human approval and does not execute embeddings, Qdrant, LangGraph, providers, PostgreSQL mutations, approval services, or Langfuse. Grounded recommendation offline validation and scoring likewise consume committed fixtures without network access; external RAGAS runs evaluate existing predictions only after acknowledgement. PostgreSQL remains authoritative for durable business records. LangGraph PostgreSQL checkpoints remain authoritative for graph continuity. Qdrant remains a rebuildable retrieval projection. Langfuse remains optional derived telemetry and is not required to reproduce evaluation decisions. Git-owned evaluation artifacts remain the evaluation authority. Generated external artifacts under `artifacts/` are run-specific evidence. Evaluation architecture is documented in [`evaluation-and-regression.md`](evaluation-and-regression.md).

Delivery semantics are at-least-once execution. Lease-token fencing prevents stale workers from overwriting newer ownership. Exactly-once execution is not claimed. Classification recovery is idempotent after an accepted classification commits. Controlled runs add terminal tool-audit recovery, recommendation uniqueness per AgentRun, and checkpoint resume so committed progress is not repeated. Future executors and tools must make side effects idempotent or otherwise safely fenced.

## Package boundaries

The package structure is organized around framework composition, shared application configuration, infrastructure integration, vertical business modules, and the worker process entry point.

```text
src/supportops/
├── agent_graph/
├── agent_tools/
├── ai/
├── api/
├── application/
├── core/
├── evaluation/
├── infrastructure/
├── knowledge_index/
├── knowledge_retrieval/
├── modules/
│   ├── agent_runs/
│   ├── approvals/
│   ├── controlled_support_inspection/
│   ├── knowledge_documents/
│   ├── support_recommendations/
│   ├── ticket_classifications/
│   ├── workspaces/
│   └── tickets/
└── worker/
```

### API boundary

`supportops.api` owns the HTTP application boundary.

Its responsibilities include:

- creating and configuring the FastAPI application;
- composing routers;
- defining OpenAPI metadata;
- managing application startup and shutdown;
- exposing operational health endpoints;
- binding per-request trace context;
- returning trace response headers;
- translating application outcomes into HTTP responses;
- mounting versioned business routes under `/api/v1`;
- registering stable expected-error handlers.

The API boundary may depend on `supportops.core`, `supportops.infrastructure`, `supportops.application`, and module application services.

Infrastructure and core packages must not depend on the API package.

Operational health routes remain unversioned. Workspace, ticket, AgentRun, classification, controlled support inspection, approval inspection and decision, ticket escalation inspection, knowledge-document, and semantic knowledge-search business routes are versioned under `/api/v1`.

### Worker boundary

`supportops.worker` owns the executable worker process composition.

Its responsibilities include:

- loading and validating shared settings at process startup;
- creating a process-owned PostgreSQL engine and session factory;
- creating one process-scoped mock or OpenAI provider and one LLM Gateway;
- creating one process-scoped PostgreSQL checkpoint runtime for LangGraph;
- creating one process-scoped embedding provider, Qdrant client, immutable knowledge index profile, and vector search adapter for controlled knowledge search;
- resolving worker identity;
- composing a session-scoped versioned executor registry and retry policy per cycle;
- running the recovery, claim, and processing loop;
- emitting structured operational cycle logs;
- handling cooperative SIGINT and SIGTERM shutdown;
- closing the controlled runtime, closing the LLM runtime, and disposing the SQLAlchemy engine on exit, attempting every independent cleanup operation even after an earlier cleanup failure.

The worker now owns its own Qdrant client and embedding provider for the controlled workflow. Each polling cycle uses a new `AsyncSession`. Provider, Gateway, checkpoint, embedding, and Qdrant resources are process-scoped and reused across cycles. The checkpoint connection pool is separate from the SQLAlchemy engine. Repositories, workflow services, and the executor registry are session-scoped. Transactions remain short and do not span provider calls, embedding calls, tool execution, Qdrant queries, or idle waits.

The API process does not initialize the LLM provider or the checkpoint runtime.

### Application boundary

`supportops.application` owns cross-module application use cases that coordinate more than one business module.

Ticket intake is implemented as `CreateTicketWithInitialRun`. That use case coordinates Workspace, Ticket, and AgentRun repositories through one SQLAlchemy session and one application-owned transaction. The HTTP request returns after that transaction commits and does not execute the workflow.

AgentRun inspection composition is implemented as `GetAgentRunInspection`. That query coordinates AgentRun ownership lookup with an optional accepted-classification reference from the classification query boundary.

Module-local command and query use cases remain inside each module's application package. Cross-module orchestration belongs in `supportops.application` only when the use case spans multiple module boundaries.

### Core boundary

`supportops.core` contains cross-cutting capabilities that are independent from business modules and external providers.

Its responsibilities include:

- validated environment-based settings;
- logging configuration;
- provider-independent request and correlation context;
- application transaction boundary contracts;
- shared operational configuration types, including worker timing settings.

The core package is not a generic location for unrelated helpers. New code belongs in `supportops.core` only when its responsibility is genuinely cross-cutting and provider-independent.

### Infrastructure boundary

`supportops.infrastructure` owns integrations with external systems and technical persistence frameworks.

The infrastructure packages include:

```text
supportops.infrastructure.postgresql
supportops.infrastructure.qdrant
```

The PostgreSQL package owns:

- async SQLAlchemy engine configuration;
- async session creation;
- declarative metadata;
- persistence model registration;
- connection lifecycle;
- database connectivity checks;
- constraint-name inspection helpers;
- the SQLAlchemy transaction adapter.

The Qdrant package owns:

- async client construction;
- client lifecycle;
- connectivity checks;
- provider-specific configuration handling.

Infrastructure packages must not contain business rules.

## Business modules

Business modules live under:

```text
supportops.modules
```

The implemented vertical modules are:

```text
supportops.modules.workspaces
supportops.modules.tickets
supportops.modules.agent_runs
supportops.modules.ticket_classifications
supportops.modules.knowledge_documents
supportops.modules.support_recommendations
supportops.modules.controlled_support_inspection
```

Workspace and ticket modules follow the implemented internal layout:

```text
domain/
application/
infrastructure/
api/
```

The `agent_runs` module follows the same layout. It provides domain, application, infrastructure, and API layers for durable scheduling, claiming, versioned executor dispatch, execution, retry, recovery, and workspace-scoped inspection, including optional accepted-classification references and logical invocation history routes.

The `ticket_classifications` module provides domain, application, infrastructure, and API layers for durable classification execution, `LLMInvocation` and `TicketClassification` persistence, lease-fenced writes, Gateway failure translation, and workspace-scoped classification inspection.

The `knowledge_documents` module provides domain, application, infrastructure, and API layers for workspace-owned document identities, immutable text versions, deterministic source normalization and hashing, concurrency-safe version allocation, PostgreSQL-authoritative chunk records, explicit ready-state and active-version separation, and tenant-safe document inspection. It does not perform embeddings or Qdrant operations; those belong to the separate indexing CLI.

The `support_recommendations` module provides domain, application, and infrastructure layers for the single accepted `SupportRecommendation` per AgentRun, ordered `SupportRecommendationCitation` provenance, lease-fenced atomic persistence, and exact recovery queries that prevent duplicate recommendation writes after a retry.

The `controlled_support_inspection` module provides domain, application, infrastructure, and API layers for the workspace-scoped read-only aggregate inspection view of a controlled workflow run. It reads durable business records only. It does not read LangGraph checkpoint state and does not expose checkpoint identity or blobs.

The cross-cutting `supportops.ai` package owns provider-independent LLM contracts, provider adapters, prompt definitions, structured schemas, repair behavior, estimated-cost calculation, and embedding contracts.

The `supportops.agent_graph` package owns the controlled workflow graph: bounded versioned graph state, runtime context separation, node routing, the PostgreSQL checkpoint runtime, decision and recommendation execution, tool observation reconstruction from durable records, and typed retryable and terminal graph failures.

The `supportops.agent_tools` package owns the controlled read-only tool registry, argument validation, bounded tool execution, the deterministic service-status catalog, and durable `AgentToolCall` audit persistence.

The `supportops.knowledge_index` package owns deterministic chunking, Qdrant collection and point adapters, indexing orchestration, composition, and the `supportops-index-knowledge` CLI.

The `supportops.knowledge_retrieval` package owns provider-independent retrieval contracts, active-version resolution, Qdrant candidate search, PostgreSQL chunk hydration, provenance validation, deterministic ranking, citations, FastAPI dependency composition, and the workspace-scoped semantic search route.

The `supportops.evaluation.contracts` package owns shared evaluation manifests, typed prediction envelopes, deterministic canonical serialization and hashing, and atomic artifact writes.

The `supportops.evaluation.ticket_classification` package owns the immutable versioned synthetic dataset loader, split-manifest validation, prediction artifacts, deterministic evaluator, release-gate profile, Gateway predictor, sequential runner, and evaluation CLI with explicit prompt-version selection.

The `supportops.evaluation.semantic_retrieval`, `supportops.evaluation.controlled_support`, and `supportops.evaluation.human_approval` packages own immutable synthetic datasets, static prediction fixtures, typed envelopes, deterministic metrics, and domain release-gate profiles. The `supportops.evaluation.regression` package owns repository-level aggregation, domain ordering, optional classification inclusion, atomic optional output, and the `supportops-evaluate-regression` CLI. The `supportops.evaluation.grounded_recommendations` package owns the grounded recommendation dataset loader, static prediction contracts, deterministic complementary metrics, normalized RAGAS score artifacts, offline aggregation, evaluation-only RAGAS adapter isolation, the external runner and CLI, and the human review rubric loader.

### Domain

Domain packages own persistence-independent frozen entities, invariants, and repository protocols.

Workspace, Ticket, AgentRun, AgentRunAttempt, Document, DocumentVersion, and DocumentChunk entities are frozen dataclasses. They do not depend on SQLAlchemy session state.

Ticket status remains `open` after intake. AgentRun status reports workflow execution. An accepted `TicketClassification` records the model interpretation and does not mutate Ticket status. A persisted `SupportRecommendation` records the analysis outcome of the controlled workflow; it does not modify the ticket, send a customer response, or mutate external systems. New tickets are scheduled with the configured workflow version; the local default is `controlled-support-v1`. The classification workflow and the deterministic baseline remain registered for historical or explicitly scheduled runs.

Persisted AgentRun states are:

```text
queued
running
waiting_for_approval
retry_scheduled
succeeded
failed
```

Persisted AgentRunAttempt outcomes include:

```text
succeeded
retryable_failure
terminal_failure
timed_out
lease_expired
awaiting_approval
```

### Application

Application packages own use cases and transaction orchestration.

Command use cases such as workspace creation own the transaction boundary through the application-facing transaction contract. Knowledge-document creation, immutable version creation, and explicit activation also own application-level transaction boundaries. Query use cases such as workspace retrieval, ticket retrieval, ticket listing, document retrieval, document listing, version retrieval, version listing, AgentRun retrieval, AgentRunAttempt listing, classification detail, ticket classification history, and AgentRun logical invocation history execute reads without opening a write transaction.

`CreateTicketWithInitialRun` is the cross-module ticket intake command. It validates the workspace, persists the ticket, and persists the initial AgentRun in one shared SQLAlchemy session and one application-owned transaction. Repositories flush without committing independently.

The `agent_runs` application layer owns:

- bounded retry policy calculation;
- the versioned `AgentRunExecutorRegistry` with exact name and version dispatch;
- the deterministic baseline executor;
- claimed-run processing outside claim transactions;
- one-cycle worker orchestration;
- the continuous polling loop;
- workspace-scoped AgentRun inspection use cases.

The `ticket_classifications` application layer owns the classification executor that calls the Gateway outside database transactions, materializes durable invocations, persists accepted classifications under lease fencing, and translates Gateway failures into retryable or terminal AgentRun execution errors. The same executor serves both `ticket-classification-v1` and the classification step of `controlled-support-v1`.

The `agent_graph` application layer owns the controlled workflow executor registered as `controlled-support-v1`. It ensures the durable classification, recovers committed tool outcomes before requesting another decision, requests bounded provider decisions, executes validated read-only tools, reconstructs observations from durable records, drafts the recommendation, and returns successfully only after a persisted recommendation identity exists in validated graph state. The outer processor then performs the lease-fenced AgentRun completion.

The `knowledge_documents` application layer owns atomic document plus version-one creation, concurrency-safe version allocation under a document row lock, workspace-scoped queries, and explicit ready-version activation. These use cases perform no embedding-provider or Qdrant calls.

`GetAgentRun` performs workspace-scoped lookup through `AgentRunQueryRepository`. Cross-module `GetAgentRunInspection` composes that lookup with an optional accepted-classification reference from the classification query boundary. `ListAgentRunAttempts` first validates AgentRun ownership through the same workspace-scoped lookup, then lists attempts. Attempts do not carry `workspace_id` and therefore cannot establish ownership independently. Logical invocation history is available through a separate AgentRun-scoped endpoint after the same ownership validation.

`AgentRunQueryRepository` is a dedicated read-only contract. `AgentRunRepository` remains the mutation-oriented scheduling and worker contract. This separation applies interface segregation: inspection depends only on read operations, while claiming, transitions, and recovery remain on the mutation protocol.

Read inspection services use the request-scoped `AsyncSession` without an explicit `TransactionManager`. They do not flush, commit, or mutate state. PostgreSQL orders attempts by `attempt_number` ascending in SQL. Invocation history is ordered by attempt number ascending, then invocation sequence ascending. Classification history is ordered newest first and uses opaque keyset pagination.

Application services depend on domain repository protocols. They do not depend on FastAPI or SQLAlchemy session APIs directly.

Transactional AgentRun scheduling is documented in [`agent-run-scheduling.md`](agent-run-scheduling.md). Runtime process topology is documented in [`runtime-topology.md`](runtime-topology.md). LLM Gateway behavior is documented in [`llm-gateway.md`](llm-gateway.md). Durable classification is documented in [`ticket-classification.md`](ticket-classification.md). Classification inspection and evaluation are documented in [`classification-evaluation.md`](classification-evaluation.md). Repository-owned evaluation and regression architecture is documented in [`evaluation-and-regression.md`](evaluation-and-regression.md). Versioned knowledge-document ownership and rollout are documented in [`knowledge-documents.md`](knowledge-documents.md). Explicit knowledge indexing is documented in [`knowledge-indexing.md`](knowledge-indexing.md). Semantic knowledge retrieval is documented in [`semantic-knowledge-retrieval.md`](semantic-knowledge-retrieval.md). The controlled support workflow is documented in [`controlled-support-workflow.md`](controlled-support-workflow.md). The indexing decision is recorded in [`../decisions/0008-use-explicit-profiled-knowledge-indexing.md`](../decisions/0008-use-explicit-profiled-knowledge-indexing.md). The PostgreSQL hydration decision is recorded in [`../decisions/0009-hydrate-retrieval-evidence-from-postgresql.md`](../decisions/0009-hydrate-retrieval-evidence-from-postgresql.md). The separation of outer AgentRun durability from inner LangGraph orchestration is recorded in [`../decisions/0010-separate-agent-run-and-langgraph-durability.md`](../decisions/0010-separate-agent-run-and-langgraph-durability.md). Framework-owned checkpoint schema ownership is recorded in [`../decisions/0011-treat-langgraph-checkpoints-as-framework-owned-schema.md`](../decisions/0011-treat-langgraph-checkpoints-as-framework-owned-schema.md).

### Infrastructure

Module infrastructure packages own:

- SQLAlchemy persistence records;
- explicit domain-to-record and record-to-domain mapping;
- named PostgreSQL constraints and indexes;
- async repository implementations;
- session-scoped worker cycle composition.

Repositories flush pending changes and translate expected named constraints into stable domain-facing errors. They do not commit independently.

### API

Module API packages own HTTP routes, request and response schemas, dependency construction, and transport-level pagination encoding.

Implemented routes include:

```text
POST /api/v1/workspaces
GET  /api/v1/workspaces/{workspace_id}
POST /api/v1/workspaces/{workspace_id}/tickets
GET  /api/v1/workspaces/{workspace_id}/tickets/{ticket_id}
GET  /api/v1/workspaces/{workspace_id}/tickets
GET  /api/v1/workspaces/{workspace_id}/tickets/{ticket_id}/classifications
GET  /api/v1/workspaces/{workspace_id}/ticket-classifications/{classification_id}
GET  /api/v1/workspaces/{workspace_id}/agent-runs/{agent_run_id}
GET  /api/v1/workspaces/{workspace_id}/agent-runs/{agent_run_id}/attempts
GET  /api/v1/workspaces/{workspace_id}/agent-runs/{agent_run_id}/llm-invocations
GET  /api/v1/workspaces/{workspace_id}/tickets/{ticket_id}/agent-runs/{agent_run_id}/inspection
POST /api/v1/workspaces/{workspace_id}/documents
GET  /api/v1/workspaces/{workspace_id}/documents
GET  /api/v1/workspaces/{workspace_id}/documents/{document_id}
POST /api/v1/workspaces/{workspace_id}/documents/{document_id}/versions
GET  /api/v1/workspaces/{workspace_id}/documents/{document_id}/versions
GET  /api/v1/workspaces/{workspace_id}/documents/{document_id}/versions/{document_version_id}
POST /api/v1/workspaces/{workspace_id}/documents/{document_id}/versions/{document_version_id}/activate
POST /api/v1/workspaces/{workspace_id}/knowledge/search
```

Ticket creation returns the existing Ticket response shape. AgentRun records are persisted atomically and can be inspected through the workspace-scoped read-only endpoints when the AgentRun identifier is otherwise known. AgentRun detail includes an optional minimal accepted-classification reference. The creation response does not expose a full classification result.

Document creation returns document metadata and version-one metadata without source content. Version creation and listing also omit source content. Only the version detail route returns the authoritative normalized source. Activation remains an explicit operation and rejects pending or failed versions.

Semantic search returns ranked authoritative evidence and citations. The response contains no generated answer. Only active ready versions whose persisted profile matches the runtime retrieval profile participate.

Controlled support inspection is a single read-only aggregate view scoped by workspace, ticket, and AgentRun. It supports `controlled-support-v1` only and may return valid partial views for queued, running, retrying, and failed workflows. A completed controlled workflow requires a persisted recommendation. The view is assembled from durable business records: the safe AgentRun lifecycle summary, the accepted classification, attempt-ordered tool-call summaries, attempt-ordered logical invocation history, persisted token usage, persisted historical estimated cost, the recommendation, and ordered citation provenance. It does not read LangGraph checkpoint state, and it excludes checkpoint identity, raw prompts, raw provider payloads, raw tool arguments, input fingerprints, and retrieved document bodies.

HTTP schemas project safe operational fields only. Internal lease ownership, lease tokens, lease expiry, ingestion request IDs, execution request IDs, provider request IDs, raw prompts, and raw provider responses remain private. Attempt responses exclude `agent_run_id`, lease tokens, and execution request IDs.

Inspection endpoints are observational only. They do not perform mutation, retry, cancellation, or lease revocation.

Missing and cross-workspace AgentRun lookups both return `404` with `agent_run_not_found`. Missing and cross-workspace classification lookups both return `404` with `ticket_classification_not_found`. Missing, cross-ticket, and cross-workspace controlled support inspection lookups all return the same `404` contract with `controlled_support_inspection_not_found`. Missing and cross-workspace document and version lookups return the corresponding knowledge-document `404` contract. This prevents resource ownership disclosure across workspaces.

Expected application errors use a stable envelope with `code`, `message`, and `request_id`. Malformed identifiers and invalid request schemas use FastAPI validation responses.

## Dependency direction

The implemented dependency direction is:

```text
API / Worker
    ↓
Application use cases
    ↓
Domain protocols
    ↑
Infrastructure adapters
```

Concrete adapters implement interfaces owned by the domain layer. Application services depend on those protocols rather than on provider-specific repository classes.

The executable composition shape is:

```text
supportops.api
supportops.worker
    ↓
supportops.modules.*.api
    ↓
supportops.application
supportops.modules.*.application
    ↓
supportops.modules.*.domain
    ↑
supportops.modules.*.infrastructure
supportops.infrastructure.postgresql
supportops.core
```

Circular dependencies are not permitted.

## Command and query behavior

Write commands own the transaction:

- create workspace;
- create ticket with initial AgentRun.

Those use cases open an application-owned transaction, persist through repositories that flush without committing, and commit or roll back as one unit.

`CreateTicketWithInitialRun` coordinates Workspace, Ticket, and AgentRun repositories through the same session and the same transaction. A successful commit persists both the ticket and its initial run. An AgentRun insertion failure rolls back the ticket. A ticket conflict creates no additional run.

Worker write paths use short transactions for recovery, claim, ticket load, and fenced outcome persistence. Executor work runs outside those transactions. Controlled graph nodes follow the same rule: tool and provider calls execute outside database transactions, and each node persists its durable result through a short lease-fenced transaction.

Read queries do not open write transactions:

- get workspace;
- get ticket;
- list tickets;
- get AgentRun;
- list AgentRunAttempts;
- get TicketClassification;
- list ticket classifications;
- list AgentRun logical invocations;
- get controlled support inspection.

Ticket listing verifies that the workspace exists before returning an empty page. A missing workspace returns `workspace_not_found`. An empty workspace returns an empty `items` collection with a null `next_cursor`.

AgentRun lookup is always workspace-scoped. Attempt history is returned only after AgentRun ownership is validated. Queued runs may return an empty attempt `items` array. Attempt pagination is intentionally omitted because `max_attempts` is bounded.

## Versioned business API and error contract

Business routes are versioned under `/api/v1`.

Operational health routes remain outside that prefix:

```text
GET /health/live
GET /health/ready
```

Expected application errors use a stable response envelope:

```json
{
  "error": {
    "code": "ticket_not_found",
    "message": "Ticket was not found.",
    "request_id": "00000000-0000-0000-0000-000000000000"
  }
}
```

Implemented expected error codes include:

- `workspace_not_found`;
- `ticket_not_found`;
- `agent_run_not_found`;
- `ticket_classification_not_found`;
- `workspace_slug_conflict`;
- `ticket_external_reference_conflict`;
- `invalid_pagination_cursor`;
- `controlled_support_inspection_not_found`;
- `unsupported_agent_run_inspection`;
- `controlled_support_inspection_inconsistent`.

Cross-workspace ticket retrieval returns the same `ticket_not_found` contract as a missing ticket. Cross-workspace AgentRun retrieval returns the same `agent_run_not_found` contract as a missing AgentRun. Cross-workspace classification retrieval returns the same `ticket_classification_not_found` contract as a missing classification.

Controlled support inspection distinguishes three expected outcomes. An unresolved scoped lookup returns `controlled_support_inspection_not_found`. An AgentRun that ran a workflow version outside the controlled inspection contract returns `unsupported_agent_run_inspection`. Persisted records that violate inspection invariants return `controlled_support_inspection_inconsistent` rather than an unsafe partial projection.

## Data ownership

### PostgreSQL

PostgreSQL is the transactional source of truth and the durable AgentRun work queue.

Authoritative business state, workflow state, AgentRun leases, retry scheduling, attempt history, logical invocations, accepted classifications, controlled tool-call audits, support recommendations, recommendation citations, approval requests, sensitive execution grants, ticket escalations, knowledge documents, immutable source versions, immutable index profiles, authoritative chunks, indexing lifecycle, embedding usage and cost provenance, failure provenance, active-version pointers, and future operational cost reporting are persisted or planned for persistence in PostgreSQL.

The same PostgreSQL database also stores LangGraph checkpoint tables. Those tables remain framework-owned orchestration storage rather than business records, and they are not part of the application domain model or the inspection contract.

This decision supports:

- transactional consistency;
- explicit relational constraints;
- durable workflow state;
- reliable auditing;
- at-least-once execution with lease-token fencing;
- PostgreSQL-backed asynchronous processing without an external broker.

Business migrations create:

- `workspaces`;
- `tickets`;
- `agent_runs`;
- `agent_run_attempts`;
- `llm_invocations`;
- `ticket_classifications`;
- `agent_tool_calls`;
- `support_recommendations`;
- `support_recommendation_citations`.

Knowledge-document, version, and chunk tables are created by their own business migrations. LangGraph checkpoint tables are not. Checkpointer setup owns their creation and internal migration history, and Alembic autogenerate excludes those exact framework-owned table names and their indexes from schema comparison. The exclusion is name-exact, so unexpected drift in application-owned schema remains visible.

Implemented ownership and integrity rules include:

- persistence-independent frozen Workspace, Ticket, AgentRun, AgentRunAttempt, LLMInvocation, TicketClassification, AgentToolCall, SupportRecommendation, and SupportRecommendationCitation domain entities;
- SQLAlchemy records with explicit mapping;
- repository protocols, including workspace-scoped ticket `get` and `list`;
- `AgentRunQueryRepository` for workspace-scoped AgentRun inspection reads;
- `AgentRunRepository` for mutation-oriented scheduling and worker operations;
- application-owned transaction boundaries;
- repository flush without commit;
- PostgreSQL named constraints, including unique workspace slugs and workspace-scoped external references;
- a required foreign key from `tickets.workspace_id` to `workspaces.id`;
- composite workspace and ticket ownership for AgentRun, invocation, and classification records;
- unique initial trigger enforcement for duplicate initial scheduling prevention;
- one invocation sequence per AgentRunAttempt;
- one accepted classification per AgentRun with accepted-invocation provenance;
- one tool-call sequence per AgentRunAttempt;
- one support recommendation per AgentRun with accepted-invocation provenance;
- ordered citation records owned by one recommendation;
- a workspace-leading listing index for deterministic ticket ordering;
- query-driven indexes for claim and recovery paths.

Workspace scoping is a data ownership boundary. It is not authenticated tenant isolation. Workspace ownership identifies which tickets belong to which workspace. It does not establish trusted caller identity.

### Qdrant

Qdrant is a rebuildable retrieval index.

It stores the derived dense-vector projection used for semantic candidate search, but it does not own authoritative document or workflow state.

The same rebuildable candidate index serves both the public semantic search endpoint and the controlled workflow `search_knowledge` tool.

The retrieval index must remain reproducible from source content, authoritative chunks, and indexing provenance stored in PostgreSQL. Semantic retrieval hydrates final chunk content from PostgreSQL after Qdrant returns candidate identifiers. Qdrant is never used to reconstruct evidence during graph resume; PostgreSQL rebuilds authoritative observations from persisted tool audits and chunk records.

Implemented Qdrant indexing and retrieval behavior includes:

- compatible named-vector collections with vector name `dense`;
- cosine distance;
- separate mock and OpenAI collections selected by immutable index profile;
- ownership payload indexes for `workspace_id`, `document_id`, and `document_version_id`;
- deterministic chunk IDs as point IDs;
- payloads that exclude source and chunk content;
- bounded upserts with `wait=True`;
- exact version projection-count verification before ready;
- rejection of existing incompatible collections;
- filtered `query_points` candidate search for the API semantic search route and the controlled `search_knowledge` tool.

Qdrant is not involved in workspace, ticket, AgentRun, tool-audit, or recommendation persistence. The API owns a process-scoped Qdrant client for readiness and semantic retrieval. The worker owns a separate process-scoped Qdrant client used only by controlled workflows that execute knowledge search. The indexing CLI owns its own Qdrant client separately from the API and worker processes.

### Semantic retrieval flow

Implemented semantic evidence retrieval follows:

```text
workspace-scoped request
→ active ready versions from PostgreSQL
→ query embedding
→ exact-pair Qdrant search
→ PostgreSQL hydration
→ evidence response
```

Empty eligible scope short-circuits with HTTP `200`, zero searched versions, empty evidence, and no embedding or Qdrant call. Generated answers are not part of this boundary.

## Asynchronous processing direction

Durable AgentRun scheduling and the PostgreSQL worker are implemented.

Ticket intake atomically persists both the Ticket and its initial AgentRun using the configured workflow version. The API returns the existing Ticket response after the transaction commits. The HTTP request does not execute the workflow or call the model. Ticket acceptance and asynchronous processing success remain separate outcomes.

The worker process:

- uses PostgreSQL as its durable work queue and source of truth;
- recovers expired leases before claiming available work;
- claims eligible `queued` and `retry_scheduled` runs with due `available_at`;
- uses `FOR UPDATE SKIP LOCKED` so multiple worker processes can claim distinct runs safely;
- dispatches exact workflow name and version through the executor registry;
- executes provider, embedding, tool, and Qdrant calls outside database transactions;
- persists invocation, classification, tool-audit, recommendation, and AgentRun outcome records through separate short lease-token-fenced transactions.

The registry contains exactly four workflow versions:

```text
ticket-processing / deterministic-baseline-v1
ticket-processing / ticket-classification-v1
ticket-processing / controlled-support-v1
ticket-processing / human-approved-support-v1
```

The persisted initial workflow contract is:

```text
workflow_name = ticket-processing
workflow_version = configured value
trigger_key = initial-ticket-processing
```

The local default configured value is `controlled-support-v1`. The human-approved workflow is versioned separately and may be scheduled explicitly. The classification workflow and the deterministic baseline remain supported for historical or explicitly scheduled runs; the deterministic baseline performs no external I/O, LLM call, retrieval, or ticket classification. Unknown workflow or version values are terminal.

For controlled runs, the worker also compiles the graph with the process-scoped PostgreSQL checkpointer. Graph nodes persist business records through short lease-fenced transactions, while provider, embedding, tool, and Qdrant work happens outside those transactions. An outer attempt retry may resume the same graph thread rather than repeating completed nodes, and the outer AgentRun succeeds only after the recommendation is durably persisted.

The architecture intentionally avoids Redis, Celery, Kafka, and SQS in this phase because PostgreSQL provides transactional durability and adequate local and portfolio scope. An external queue or outbox is not required for the current worker model.

After scheduling, clients inspect persisted lifecycle state through workspace-scoped read-only endpoints when the AgentRun identifier is otherwise known. Inspection reports current durable status, retry budget, safe error metadata, ordered attempt history, optional accepted-classification reference, and logical invocation history. Controlled runs additionally expose the aggregate inspection view built from tool audits, invocations, the recommendation, and citation provenance. It does not alter retries, leases, or state transitions, and it does not guarantee future completion. Inspection remains read-only and business-record-based.

Scheduling details are documented in [`agent-run-scheduling.md`](agent-run-scheduling.md). Runtime topology details are documented in [`runtime-topology.md`](runtime-topology.md). Classification details are documented in [`ticket-classification.md`](ticket-classification.md). Inspection and evaluation details are documented in [`classification-evaluation.md`](classification-evaluation.md). Controlled workflow details are documented in [`controlled-support-workflow.md`](controlled-support-workflow.md).

## AI system boundaries

Durable structured ticket classification is implemented through the application-owned LLM Gateway and the `ticket-classification-v1` worker workflow.

The implemented classification boundary covers:

- process-scoped mock or OpenAI provider composition in the worker;
- Structured Outputs with application-side validation;
- prompt `ticket-classification` version 1;
- bounded repair;
- durable `LLMInvocation` and `TicketClassification` persistence;
- token usage and estimated-cost provenance;
- retryable and terminal failure translation;
- idempotent recovery after classification commit;
- durable classification inspection projections;
- an immutable versioned synthetic classification dataset;
- a versioned development, holdout, and safety-gate split manifest;
- repository-owned evaluation contracts and typed prediction envelopes;
- a deterministic classification evaluator with validity, safety-recall, latency, and token metrics;
- standalone classification release-gate evaluation;
- an offline evaluation CLI with explicit prompt-version selection and mock or opt-in OpenAI provider selection;
- deterministic semantic-retrieval, controlled-support, and human-approval regression over committed static fixtures;
- repository-level deterministic regression scoring through `supportops-evaluate-regression score`;
- grounded recommendation evaluation with committed synthetic fixtures, deterministic complementary metrics, static RAGAS score artifacts, offline aggregation, and an explicit external RAGAS runner;
- a committed human qualitative review rubric for grounded recommendations.

Classification does not mutate Ticket status and cannot execute tools or actions. Inspection exposes accepted classifications and logical invocation provenance through workspace-scoped read-only HTTP routes. Evaluation measures the same prompt and schema boundary offline without writing to PostgreSQL or Qdrant. Multi-domain regression scoring likewise consumes committed static fixtures and does not execute runtime services. Grounded recommendation evaluation consumes existing predictions and does not execute the controlled workflow. Runtime classification remains independently pinned; evaluation prompt selection does not change the production default. Standalone release-gate reports cannot authorize prompt promotion. Evaluation architecture is documented in [`evaluation-and-regression.md`](evaluation-and-regression.md).

The controlled support workflow extends that boundary into bounded, evidence-driven analysis through the `controlled-support-v1` worker workflow.

The implemented controlled boundary covers:

- LangGraph orchestration with PostgreSQL checkpoints inside the existing worker;
- application-owned, provider-independent decision contracts;
- a registry of exactly two read-only tools, `search_knowledge` and `lookup_service_status`;
- application-side validation of every tool call before execution;
- durable `AgentToolCall` audits with bounded safe input and output;
- observation reconstruction from durable records rather than process memory;
- grounded recommendation drafting over classification, retrieved evidence, and deterministic service status;
- durable recommendation and ordered citation persistence;
- workspace-scoped controlled support inspection.

Model selection never grants execution authority. The model cannot select unregistered tools, arbitrary Python functions, or infrastructure adapters, and write-capable safety levels are rejected by the registry. The workflow does not modify tickets, deliver customer responses, mutate external systems, approve sensitive actions, or make authorization decisions.

Future AI behavior is expected to remain behind application-owned boundaries for:

- reranking over retrieved evidence;
- multi-profile score fusion;
- external side-effect tools;
- paired prompt comparison and evidence-driven prompt promotion;
- a real canonical external RAGAS baseline;
- generation evaluation beyond the current classification and grounded recommendation evaluation boundaries.

External AI frameworks and providers must not become the source of business rules or workflow ownership. LangGraph orchestrates bounded internal steps; it does not own the AgentRun lifecycle, business records, or public workflow outcomes.

The OpenAI Python SDK exists only behind the OpenAI provider adapters and is used only when an OpenAI provider is explicitly selected. Deterministic chunking depends on pinned `tiktoken` for reproducible tokenization. LangGraph and its PostgreSQL checkpoint saver are pinned dependencies used only by the worker for controlled orchestration and checkpoint durability. `ragas==0.4.3` is isolated in the `evaluation` dependency group and is not a runtime dependency. The following dependencies remain intentionally absent until a concrete capability requires them:

- Anthropic SDK;
- LangChain as a runtime orchestration dependency;
- reranking libraries.

## Operational health model

The platform distinguishes process health from dependency readiness.

### Liveness

`GET /health/live` verifies that the application process is running.

Liveness does not depend on PostgreSQL or Qdrant.

### Readiness

`GET /health/ready` verifies that required infrastructure dependencies are available.

Readiness checks:

- PostgreSQL;
- Qdrant.

Each check uses a bounded timeout.

When a required dependency is unavailable, the application process may remain alive while readiness returns a structured non-success response.

This behavior preserves diagnostics and accurately communicates whether the application can accept workload.

The worker process does not expose these health endpoints. It now depends on Qdrant and the configured embedding provider whenever a controlled workflow executes `search_knowledge`. Dependency failures during execution become typed workflow execution failures handled by the outer processor rather than health-endpoint results.

## Configuration

Runtime configuration is environment-based and validated.

The settings model includes:

- application environment;
- application name;
- application version;
- log level;
- API host;
- API port;
- PostgreSQL connection URL;
- PostgreSQL pool configuration where justified;
- Qdrant URL;
- optional Qdrant API key;
- dependency health-check timeout;
- worker identity, polling, lease, timeout, shutdown, retry, and attempt settings;
- configured ticket-processing workflow version;
- controlled graph bounds for maximum steps, decision turns, tool calls, and tool timeout;
- controlled checkpoint durability mode;
- LLM provider, OpenAI model and credentials, request timeout, transport retry, and repair settings;
- embedding provider, model, dimensions, request timeout, and transport retry settings.

Worker, graph, and LLM settings are validated at process startup. Cross-field invariants require:

- worker lease duration to exceed execution timeout by at least fifteen seconds;
- retry maximum not to be smaller than retry base;
- the complete logical LLM invocation budget to fit inside the worker execution timeout with a fifteen-second safety margin;
- the controlled tool timeout to remain below the worker execution timeout.

For the controlled workflow, the budget reserves six logical generations before repair multiplication: one classification, up to four decisions, and one recommendation. A provider budget that cannot fit inside the execution timeout is rejected during settings validation.

Secrets and complete connection credentials must not be logged.

Invalid required configuration fails clearly during application, worker, or indexing-CLI construction. The API validates shared settings, creates the configured embedding provider and immutable retrieval profile for semantic search, and does not initialize the LLM provider. The worker creates the configured LLM provider, the checkpoint runtime, and its own embedding provider, Qdrant client, and index profile for controlled knowledge search. The indexing CLI creates its own embedding provider. LLM provider and embedding provider selections are independent. OpenAI credentials are required when either OpenAI generation or OpenAI embeddings are selected.

## Logging

The repository foundation establishes structured JSON logging.

The logging baseline supports:

- severity levels;
- application environment;
- event-oriented messages;
- exception information;
- active request and correlation identifiers.

HTTP request traceability uses the following semantics:

- every HTTP request receives a server-generated UUID v4 request ID;
- a valid incoming `X-Correlation-ID` UUID is propagated;
- an absent or invalid correlation ID defaults to the request ID;
- both identifiers are returned through `X-Request-ID` and `X-Correlation-ID`;
- active identifiers are added automatically to structured JSON logs;
- context is scoped with `contextvars` and reset after request completion;
- completion logs contain safe operational metadata only;
- raw incoming invalid identifiers and request bodies are not logged.

The worker emits structured operational cycle logs for start, cycle completion, shutdown, and stop events.

No external logging platform is introduced during the repository foundation phase.

## Testing boundaries

Tests are separated by infrastructure dependency.

### Unit tests

Unit tests do not require Docker or network services.

They validate behavior such as:

- settings validation, including worker timing invariants;
- application construction;
- request-context creation;
- async isolation and cleanup;
- response trace headers;
- correlation propagation;
- spoofing prevention;
- exception behavior;
- completion-log coverage;
- liveness behavior;
- readiness aggregation;
- dependency failure handling;
- domain invariants, including AgentRun and AgentRunAttempt;
- application service command and query behavior, including AgentRun inspection;
- transactional ticket-intake orchestration;
- retry policy, claiming, fencing, recovery, processor, and worker cycle behavior;
- versioned registry dispatch;
- deterministic executor contract validation;
- classification executor behavior with Gateway failure translation;
- worker process identity, composition, and graceful shutdown;
- ORM mapping and metadata;
- named constraints;
- persistence model registration;
- PostgreSQL constraint-name inspection;
- API schemas and opaque cursor encoding;
- AgentRun inspection schema projections that omit internal fencing identifiers;
- classification inspection projections, cursors, and query services;
- evaluation contract, split-manifest, dataset, prediction, metrics, predictor, runner, release-gate, and CLI safety coverage;
- semantic-retrieval, controlled-support, human-approval, repository regression, and grounded recommendation evaluator coverage;
- deterministic chunking and tokenizer adapter behavior;
- embedding contracts, pricing, mock provider, OpenAI fake provider, and normalized embedding errors;
- Qdrant collection compatibility, payload indexes, upsert, and exact-count adapter behavior;
- indexing orchestration, composition, and CLI safety coverage;
- knowledge-retrieval contracts, request bounds, provenance validation, ranking, schemas, route, and lifespan coverage;
- controlled graph state invariants, routing, and transition behavior;
- provider decision normalization and terminal control validation;
- bounded read-only tool execution and registry rejection rules;
- observation reconstruction from durable tool audits;
- controlled support inspection services and schema projections.

### Integration tests

Integration tests may depend on local PostgreSQL and Qdrant services.

They validate:

- real PostgreSQL connectivity;
- real Qdrant connectivity;
- readiness against live dependencies;
- dependency failure behavior;
- Alembic upgrade, downgrade, re-upgrade, and metadata parity for classification and knowledge-document tables;
- workspace, ticket, and knowledge-document repository behavior;
- concurrency-sensitive uniqueness enforcement, including document-version allocation and normalized-content conflicts;
- atomic ticket and AgentRun commit and rollback behavior;
- PostgreSQL claim ordering and `SKIP LOCKED` concurrency;
- fenced transitions and expired lease recovery;
- processor transaction separation against live PostgreSQL;
- mock classification workflow integration;
- fenced invocation and classification persistence;
- retry and recovery idempotency after classification commit;
- workspace, ticket, and knowledge-document HTTP API behavior;
- workspace-scoped AgentRun inspection HTTP behavior;
- classification detail and ticket classification history inspection;
- AgentRun classification reference and logical invocation ordering;
- stable expected-error responses;
- opaque cursor pagination, including separate document and version cursor kinds;
- workspace-scoped SQL predicates and attempt ordering;
- real Qdrant collection and payload-index creation;
- deterministic point upsert and exact scoped count;
- successful indexing against real PostgreSQL and Qdrant with mock embeddings;
- ready-version rerun no-op behavior;
- partial-projection failure and retry recovery;
- indexing without implicit activation;
- semantic retrieval against real PostgreSQL and Qdrant with deterministic mock embeddings;
- active-ready isolation, workspace isolation, document filters, stale-hash rejection, and rank promotion;
- LangGraph PostgreSQL checkpoint setup and resume without repeating completed nodes;
- lease-fenced tool-audit and recommendation persistence, including stale-worker rejection;
- post-commit and pre-checkpoint tool recovery;
- controlled support inspection HTTP behavior and cross-workspace isolation;
- Alembic parity that excludes framework-owned checkpoint tables from application schema comparison.

PostgreSQL integration tests are required for concurrency and row-locking behavior. Knowledge-index, knowledge-retrieval, and controlled workflow integration tests use real PostgreSQL and Qdrant with mock embeddings. Unit tests use fakes at provider boundaries. No paid external provider call belongs in the default automated suite. Shared business cleanup does not delete checkpoint rows; tests that create durable graph threads clean up their own threads.

The test suite must verify externally observable behavior rather than reproduce implementation details.

## Security considerations

The repository foundation establishes the following security baseline:

- no committed credentials;
- safe example environment values;
- validated configuration;
- no secret logging;
- health responses without credential disclosure;
- bounded dependency checks;
- explicit lifecycle handling;
- server-owned request IDs;
- strict UUID parsing for incoming correlation IDs;
- downstream response-header override protection;
- omission of request bodies from completion logs;
- sanitized worker failure summaries without raw exception text persistence.

Trace identifiers support observability and supportability. They are not authentication or authorization controls.

Workspace scoping is a data ownership boundary. It is not authentication, authorization, or authenticated tenant isolation.

Authentication, authorization, authenticated tenant isolation, and public deployment hardening are intentionally deferred to later implementation phases.

## Scaling considerations

The modular monolith is intended to support early and intermediate platform growth without introducing distributed coordination prematurely.

The architecture keeps room for:

- separate API and worker scaling;
- independent process deployment;
- multiple worker processes through `FOR UPDATE SKIP LOCKED`;
- PostgreSQL connection pool tuning;
- Qdrant deployment changes;
- provider adapter replacement;
- extraction of modules when operational evidence justifies it.

Service extraction is not a default objective. It should be driven by clear ownership, scaling, reliability, or deployment requirements.

## Repository foundation scope

The repository foundation, Slice 1, durable AgentRun scheduling, the PostgreSQL worker, AgentRun inspection, the LLM Gateway, durable ticket classification, classification inspection, offline evaluation, versioned knowledge-document management, explicit knowledge indexing, semantic knowledge retrieval, and the controlled support workflow establish:

- project and dependency management;
- local PostgreSQL and Qdrant infrastructure;
- application configuration;
- structured logging;
- FastAPI bootstrap;
- lifecycle management;
- liveness and readiness endpoints;
- infrastructure connectivity checks;
- SQLAlchemy and Alembic foundations;
- workspace and ticket domain persistence;
- immutable workspace-scoped knowledge-document and version persistence;
- PostgreSQL-authoritative document chunks, index profiles, and active-version state;
- deterministic chunking and embedding-provider composition;
- Qdrant collection bootstrap and verified vector-point indexing;
- explicit indexing CLI with idempotency and recovery behavior;
- semantic retrieval contracts, adapters, API, lifecycle, and tests;
- active-version resolution, query embeddings, exact-pair Qdrant search, and PostgreSQL hydration;
- AgentRun and AgentRunAttempt persistence;
- atomic Ticket and configured initial AgentRun scheduling;
- PostgreSQL claiming, leases, fencing, retries, and recovery;
- versioned workflow executor registry with four exact registered versions;
- deterministic baseline, classification, controlled support, and human-approved support workflow execution;
- process-scoped provider, Gateway, checkpoint, embedding, and Qdrant composition in the worker;
- durable `LLMInvocation` and `TicketClassification` persistence;
- LangGraph orchestration with PostgreSQL checkpoints inside the worker;
- bounded read-only tool execution with durable `AgentToolCall` audits;
- durable `SupportRecommendation` and citation persistence;
- a separate worker process with cooperative shutdown;
- workspace-scoped AgentRun inspection;
- workspace-scoped classification and logical invocation inspection;
- workspace-scoped controlled support inspection;
- repository-owned offline deterministic classification evaluation;
- explicit evaluation prompt-version selection;
- standalone classification release-gate evaluation;
- deterministic semantic-retrieval, controlled-support, and human-approval regression;
- repository-level deterministic regression scoring;
- grounded recommendation evaluation with deterministic complementary metrics, static RAGAS score artifacts, offline aggregation, and an explicit external RAGAS runner;
- a committed grounded recommendation human review rubric;
- opt-in external-provider evaluation;
- application services and versioned business APIs;
- workspace-scoped document and immutable version APIs;
- concurrency-safe document-version creation and explicit ready-version activation;
- stable expected-error contracts;
- opaque cursor pagination, including separate document and version cursor kinds;
- request and correlation identifier persistence;
- unit and integration testing;
- continuous integration;
- architecture and development documentation.

## Intentionally deferred scope

The current phase does not implement:

- authentication or authorization;
- authenticated tenant isolation;
- manual AgentRun retry or cancellation;
- lease revocation or worker administration;
- global AgentRun listing, status filtering, or pagination across runs;
- WebSockets or Server-Sent Events;
- frontend monitoring applications;
- Redis, Celery, Kafka, or SQS;
- classification prompt version 2;
- paired prompt comparison across versions;
- prompt promotion, rejection, or inconclusive decisions;
- automatic prompt optimization;
- scheduled or online evaluation;
- evaluation history persistence;
- evaluation database;
- production A/B testing;
- cross-provider fallback and automatic model routing;
- Anthropic provider;
- operational cost reporting and invoice reconciliation;
- automated document ingestion;
- automatic indexing scheduling;
- automatic version activation;
- reranking;
- generated answers in the public semantic search API;
- multi-profile score fusion;
- external side-effect tools;
- customer-response delivery;
- workflow cancellation and HTTP retry controls;
- checkpoint inspection endpoints and raw graph-state exposure;
- long-term checkpoint retention policy;
- Phoenix integration;
- Langfuse evaluation workflows and Langfuse datasets or experiments;
- a real canonical external RAGAS baseline;
- production feedback ingestion;
- a full annotation platform;
- generation evaluation beyond the current classification and grounded recommendation boundaries;
- frontend applications;
- public cloud deployment;
- infrastructure as code.

Ticket status remains `open` after intake. Durable AgentRun scheduling, the PostgreSQL worker, the application-owned LLM Gateway, durable ticket classification, workspace-scoped AgentRun, classification, controlled support, approval, and escalation inspection, approval decision commands with worker-owned resume, repository-owned offline classification evaluation with standalone release gates, deterministic semantic-retrieval, controlled-support, and human-approval regression over committed static fixtures, repository-level deterministic regression scoring, grounded recommendation evaluation with deterministic complementary metrics and an explicit external RAGAS boundary, versioned knowledge documents, explicit profiled knowledge indexing, active-version semantic knowledge retrieval, and the controlled support workflow with LangGraph orchestration, read-only tools, and durable recommendations are implemented. Redis, Celery, Kafka, and SQS remain intentionally deferred because PostgreSQL already provides transactional durability and adequate local and portfolio scope for this phase.

These capabilities are deferred to preserve clear scope, avoid speculative abstractions, and keep each implementation slice independently reviewable.
