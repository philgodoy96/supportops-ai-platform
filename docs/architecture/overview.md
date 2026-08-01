# Architecture Overview

## Purpose

SupportOps AI Platform is a production-minded backend and AI systems engineering project designed to support reliable support operations, controlled AI orchestration, retrieval-augmented workflows, human approval, observability, and evaluation.

The platform is intentionally structured as an API-first modular monolith. This architecture keeps deployment and operational complexity controlled while preserving clear internal boundaries that can evolve as the system grows.

The current repository phase establishes the operational foundation, workspace-scoped persistence, the Slice 1 workspace and ticket HTTP API, durable AgentRun scheduling, the PostgreSQL-backed worker, workspace-scoped AgentRun inspection, the application-owned LLM Gateway, durable structured ticket classification with invocation and accepted classification persistence, classification inspection, logical invocation inspection, offline deterministic evaluation, opt-in provider evaluation, and PostgreSQL-authoritative immutable knowledge-document versioning. Token-aware chunking, embeddings, Qdrant indexing, semantic retrieval, LangGraph orchestration, tools, approvals, AI observability integrations, and RAGAS remain later phases.

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

The platform is a modular monolith with two deployable runtime entry points:

- an HTTP API process;
- a PostgreSQL AgentRun worker process.

Both processes share:

- the same Python package;
- the same domain model;
- the same application services;
- the same PostgreSQL database;
- infrastructure adapters required by each process.

The processes have distinct runtime responsibilities while remaining part of one deployable codebase.

Current ownership:

- the API owns HTTP acceptance, transactional Ticket plus configured initial AgentRun scheduling, read-only AgentRun, classification, and logical invocation inspection, and workspace-scoped knowledge-document registration, versioning, inspection, and explicit activation;
- the worker owns process-scoped provider and Gateway composition, expired lease recovery, claim, versioned executor dispatch, classification execution, and fenced outcome persistence;
- PostgreSQL owns tickets, AgentRuns, attempts, leases, retry scheduling, execution history, logical invocations, accepted classifications, knowledge documents, immutable source versions, authoritative chunk records, indexing lifecycle metadata, and active-version pointers;
- offline evaluation owns versioned synthetic datasets, prediction artifacts, deterministic metrics, and reproducible reports outside the API and worker processes;
- Qdrant remains outside the worker flow and is used by the API only for its current connectivity lifecycle.

The platform separates three planes:

```text
runtime plane
→ worker execution and durable persistence

inspection plane
→ workspace-scoped read-only API

evaluation plane
→ offline datasets, predictions, metrics, and reports
```

Delivery semantics are at-least-once execution. Lease-token fencing prevents stale workers from overwriting newer ownership. Exactly-once execution is not claimed. Classification recovery is idempotent after an accepted classification commits. Future executors and tools must make side effects idempotent or otherwise safely fenced.

## Package boundaries

The package structure is organized around framework composition, shared application configuration, infrastructure integration, vertical business modules, and the worker process entry point.

```text
src/supportops/
├── ai/
├── api/
├── application/
├── core/
├── evaluation/
├── infrastructure/
├── modules/
│   ├── agent_runs/
│   ├── knowledge_documents/
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

Operational health routes remain unversioned. Workspace, ticket, AgentRun, classification, and knowledge-document business routes are versioned under `/api/v1`.

### Worker boundary

`supportops.worker` owns the executable worker process composition.

Its responsibilities include:

- loading and validating shared settings at process startup;
- creating a process-owned PostgreSQL engine and session factory;
- creating one process-scoped mock or OpenAI provider and one LLM Gateway;
- resolving worker identity;
- composing a session-scoped versioned executor registry and retry policy per cycle;
- running the recovery, claim, and processing loop;
- emitting structured operational cycle logs;
- handling cooperative SIGINT and SIGTERM shutdown;
- closing the provider and disposing the SQLAlchemy engine on exit, with engine disposal attempted after provider cleanup failure.

The worker does not initialize or depend on Qdrant. Each polling cycle uses a new `AsyncSession`. Provider and Gateway instances are reused across cycles. Repositories and the executor registry are session-scoped. Transactions remain short and do not span provider calls or idle waits.

The API process does not initialize the LLM provider.

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

The `knowledge_documents` module provides domain, application, infrastructure, and API layers for workspace-owned document identities, immutable text versions, deterministic source normalization and hashing, concurrency-safe version allocation, PostgreSQL-authoritative chunk records, explicit ready-state and active-version separation, and tenant-safe document inspection. It does not perform embeddings or Qdrant operations in the current phase.

The cross-cutting `supportops.ai` package owns provider-independent LLM contracts, provider adapters, prompt definitions, structured schemas, repair behavior, and estimated-cost calculation.

The `supportops.evaluation.ticket_classification` package owns the versioned synthetic dataset loader, prediction artifacts, deterministic evaluator, Gateway predictor, sequential runner, and evaluation CLI.

### Domain

Domain packages own persistence-independent frozen entities, invariants, and repository protocols.

Workspace, Ticket, AgentRun, AgentRunAttempt, Document, DocumentVersion, and DocumentChunk entities are frozen dataclasses. They do not depend on SQLAlchemy session state.

Ticket status remains `open` after intake. AgentRun status reports workflow execution. An accepted `TicketClassification` records the model interpretation and does not mutate Ticket status or execute tools. New tickets are scheduled with the configured workflow version; the local default is `ticket-classification-v1`. The deterministic baseline remains registered for historical or explicitly scheduled runs.

Persisted AgentRun states are:

```text
queued
running
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

The `ticket_classifications` application layer owns the classification executor that calls the Gateway outside database transactions, materializes durable invocations, persists accepted classifications under lease fencing, and translates Gateway failures into retryable or terminal AgentRun execution errors.

The `knowledge_documents` application layer owns atomic document plus version-one creation, concurrency-safe version allocation under a document row lock, workspace-scoped queries, and explicit ready-version activation. These use cases perform no embedding-provider or Qdrant calls.

`GetAgentRun` performs workspace-scoped lookup through `AgentRunQueryRepository`. Cross-module `GetAgentRunInspection` composes that lookup with an optional accepted-classification reference from the classification query boundary. `ListAgentRunAttempts` first validates AgentRun ownership through the same workspace-scoped lookup, then lists attempts. Attempts do not carry `workspace_id` and therefore cannot establish ownership independently. Logical invocation history is available through a separate AgentRun-scoped endpoint after the same ownership validation.

`AgentRunQueryRepository` is a dedicated read-only contract. `AgentRunRepository` remains the mutation-oriented scheduling and worker contract. This separation applies interface segregation: inspection depends only on read operations, while claiming, transitions, and recovery remain on the mutation protocol.

Read inspection services use the request-scoped `AsyncSession` without an explicit `TransactionManager`. They do not flush, commit, or mutate state. PostgreSQL orders attempts by `attempt_number` ascending in SQL. Invocation history is ordered by attempt number ascending, then invocation sequence ascending. Classification history is ordered newest first and uses opaque keyset pagination.

Application services depend on domain repository protocols. They do not depend on FastAPI or SQLAlchemy session APIs directly.

Transactional AgentRun scheduling is documented in [`agent-run-scheduling.md`](agent-run-scheduling.md). Runtime process topology is documented in [`runtime-topology.md`](runtime-topology.md). LLM Gateway behavior is documented in [`llm-gateway.md`](llm-gateway.md). Durable classification is documented in [`ticket-classification.md`](ticket-classification.md). Classification inspection and evaluation are documented in [`classification-evaluation.md`](classification-evaluation.md). Versioned knowledge-document ownership and rollout are documented in [`knowledge-documents.md`](knowledge-documents.md).

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
POST /api/v1/workspaces/{workspace_id}/documents
GET  /api/v1/workspaces/{workspace_id}/documents
GET  /api/v1/workspaces/{workspace_id}/documents/{document_id}
POST /api/v1/workspaces/{workspace_id}/documents/{document_id}/versions
GET  /api/v1/workspaces/{workspace_id}/documents/{document_id}/versions
GET  /api/v1/workspaces/{workspace_id}/documents/{document_id}/versions/{document_version_id}
POST /api/v1/workspaces/{workspace_id}/documents/{document_id}/versions/{document_version_id}/activate
```

Ticket creation returns the existing Ticket response shape. AgentRun records are persisted atomically and can be inspected through the workspace-scoped read-only endpoints when the AgentRun identifier is otherwise known. AgentRun detail includes an optional minimal accepted-classification reference. The creation response does not expose a full classification result.

Document creation returns document metadata and version-one metadata without source content. Version creation and listing also omit source content. Only the version detail route returns the authoritative normalized source. Activation remains an explicit operation and rejects pending or failed versions.

HTTP schemas project safe operational fields only. Internal lease ownership, lease tokens, lease expiry, ingestion request IDs, execution request IDs, provider request IDs, raw prompts, and raw provider responses remain private. Attempt responses exclude `agent_run_id`, lease tokens, and execution request IDs.

Inspection endpoints are observational only. They do not perform mutation, retry, cancellation, or lease revocation.

Missing and cross-workspace AgentRun lookups both return `404` with `agent_run_not_found`. Missing and cross-workspace classification lookups both return `404` with `ticket_classification_not_found`. Missing and cross-workspace document and version lookups return the corresponding knowledge-document `404` contract. This prevents resource ownership disclosure across workspaces.

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

Worker write paths use short transactions for recovery, claim, ticket load, and fenced outcome persistence. Executor work runs outside those transactions.

Read queries do not open write transactions:

- get workspace;
- get ticket;
- list tickets;
- get AgentRun;
- list AgentRunAttempts;
- get TicketClassification;
- list ticket classifications;
- list AgentRun logical invocations.

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
- `invalid_pagination_cursor`.

Cross-workspace ticket retrieval returns the same `ticket_not_found` contract as a missing ticket. Cross-workspace AgentRun retrieval returns the same `agent_run_not_found` contract as a missing AgentRun. Cross-workspace classification retrieval returns the same `ticket_classification_not_found` contract as a missing classification.

## Data ownership

### PostgreSQL

PostgreSQL is the transactional source of truth and the durable AgentRun work queue.

Authoritative business state, workflow state, AgentRun leases, retry scheduling, attempt history, logical invocations, accepted classifications, and future approvals, audit records, and operational cost reporting are persisted or planned for persistence in PostgreSQL.

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
- `ticket_classifications`.

Implemented ownership and integrity rules include:

- persistence-independent frozen Workspace, Ticket, AgentRun, AgentRunAttempt, LLMInvocation, and TicketClassification domain entities;
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
- a workspace-leading listing index for deterministic ticket ordering;
- query-driven indexes for claim and recovery paths.

Workspace scoping is a data ownership boundary. It is not authenticated tenant isolation. Workspace ownership identifies which tickets belong to which workspace. It does not establish trusted caller identity.

### Qdrant

Qdrant is a rebuildable retrieval index.

It will store vectorized representations required for semantic retrieval, but it will not own authoritative document or workflow state.

The retrieval index must remain reproducible from source content and ingestion metadata stored in authoritative systems.

Qdrant is limited to:

- local service configuration;
- environment-based client configuration;
- lifecycle management;
- bounded connectivity checks.

No collections, embeddings, ingestion pipelines, or retrieval behavior are implemented. Qdrant is not involved in workspace, ticket, or AgentRun persistence, and the worker does not connect to Qdrant.

## Asynchronous processing direction

Durable AgentRun scheduling and the PostgreSQL worker are implemented.

Ticket intake atomically persists both the Ticket and its initial AgentRun using the configured workflow version. The API returns the existing Ticket response after the transaction commits. The HTTP request does not execute the workflow or call the model. Ticket acceptance and asynchronous processing success remain separate outcomes.

The worker process:

- uses PostgreSQL as its durable work queue and source of truth;
- recovers expired leases before claiming available work;
- claims eligible `queued` and `retry_scheduled` runs with due `available_at`;
- uses `FOR UPDATE SKIP LOCKED` so multiple worker processes can claim distinct runs safely;
- dispatches exact workflow name and version through the executor registry;
- executes provider calls outside database transactions;
- persists invocation and classification records and AgentRun outcomes through separate lease-token-fenced transactions.

Registered workflow versions include:

```text
ticket-processing / deterministic-baseline-v1
ticket-processing / ticket-classification-v1
```

The persisted initial workflow contract is:

```text
workflow_name = ticket-processing
workflow_version = configured value
trigger_key = initial-ticket-processing
```

The local default configured value is `ticket-classification-v1`. The deterministic baseline remains supported for historical or explicitly scheduled runs and performs no external I/O, LLM call, retrieval, or ticket classification. Unknown workflow or version values are terminal.

The architecture intentionally avoids Redis, Celery, Kafka, and SQS in this phase because PostgreSQL provides transactional durability and adequate local and portfolio scope. An external queue or outbox is not required for the current worker model.

After scheduling, clients inspect persisted lifecycle state through workspace-scoped read-only endpoints when the AgentRun identifier is otherwise known. Inspection reports current durable status, retry budget, safe error metadata, ordered attempt history, optional accepted-classification reference, and logical invocation history. It does not alter retries, leases, or state transitions, and it does not guarantee future completion. Inspection remains read-only.

Scheduling details are documented in [`agent-run-scheduling.md`](agent-run-scheduling.md). Runtime topology details are documented in [`runtime-topology.md`](runtime-topology.md). Classification details are documented in [`ticket-classification.md`](ticket-classification.md). Inspection and evaluation details are documented in [`classification-evaluation.md`](classification-evaluation.md).

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
- a versioned synthetic classification dataset;
- a deterministic classification evaluator;
- an offline evaluation CLI with mock or opt-in OpenAI provider selection.

Classification does not mutate Ticket status and cannot execute tools or actions. Inspection exposes accepted classifications and logical invocation provenance through workspace-scoped read-only HTTP routes. Evaluation measures the same prompt and schema boundary offline without writing to PostgreSQL or Qdrant.

Future AI behavior is expected to remain behind application-owned boundaries for:

- retrieval and embeddings;
- LangGraph orchestration;
- tool execution;
- approvals;
- AI observability integrations;
- RAGAS;
- retrieval and generation evaluation beyond structured classification.

External AI frameworks and providers must not become the source of business rules or workflow ownership.

The OpenAI Python SDK exists only behind the OpenAI provider adapter and is used only when the OpenAI provider is explicitly selected. The following dependencies remain intentionally absent until a concrete capability requires them:

- Anthropic SDK;
- LangGraph;
- LangChain;
- Langfuse;
- RAGAS;
- embedding providers;
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

The worker process does not expose these health endpoints and does not require Qdrant.

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
- LLM provider, OpenAI model and credentials, request timeout, transport retry, and repair settings.

Worker and LLM settings are validated at process startup. Cross-field invariants require:

- worker lease duration to exceed execution timeout by at least five seconds;
- retry maximum not to be smaller than retry base;
- the complete logical LLM invocation budget to fit inside the worker execution timeout with a five-second safety margin.

Secrets and complete connection credentials must not be logged.

Invalid required configuration fails clearly during application or worker construction. The API validates shared settings but does not initialize the LLM provider.

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
- evaluation dataset, prediction, metrics, predictor, runner, and CLI safety coverage.

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
- workspace-scoped SQL predicates and attempt ordering.

PostgreSQL integration tests are required for concurrency and row-locking behavior. Qdrant-dependent tests are not worker tests.

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

The repository foundation, Slice 1, durable AgentRun scheduling, the PostgreSQL worker, AgentRun inspection, the LLM Gateway, durable ticket classification, classification inspection, offline evaluation, and versioned knowledge-document management establish:

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
- PostgreSQL-authoritative document chunks and active-version state;
- AgentRun and AgentRunAttempt persistence;
- atomic Ticket and configured initial AgentRun scheduling;
- PostgreSQL claiming, leases, fencing, retries, and recovery;
- versioned workflow executor registry;
- deterministic baseline and classification workflow execution;
- process-scoped provider and Gateway composition;
- durable `LLMInvocation` and `TicketClassification` persistence;
- a separate worker process with cooperative shutdown;
- workspace-scoped AgentRun inspection;
- workspace-scoped classification and logical invocation inspection;
- offline deterministic classification evaluation;
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
- evidence-driven prompt version 2;
- prompt regression comparison across versions;
- scheduled evaluation;
- evaluation history persistence;
- cross-provider fallback and automatic model routing;
- Anthropic provider;
- operational cost reporting and invoice reconciliation;
- automated document ingestion;
- token-aware chunk generation;
- embeddings;
- Qdrant collections or indexing;
- semantic retrieval;
- LangGraph orchestration;
- registered tools;
- approval workflows;
- AI observability integrations;
- Langfuse;
- RAGAS;
- retrieval and generation evaluation beyond structured classification;
- frontend applications;
- public cloud deployment;
- infrastructure as code.

Ticket status remains `open` after intake. Durable AgentRun scheduling, the PostgreSQL worker, the application-owned LLM Gateway, durable ticket classification, workspace-scoped AgentRun and classification inspection, and offline evaluation are implemented. Redis, Celery, Kafka, and SQS remain intentionally deferred because PostgreSQL already provides transactional durability and adequate local and portfolio scope for this phase.

These capabilities are deferred to preserve clear scope, avoid speculative abstractions, and keep each implementation slice independently reviewable.
