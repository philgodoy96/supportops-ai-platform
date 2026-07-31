# Architecture Overview

## Purpose

SupportOps AI Platform is a production-minded backend and AI systems engineering project designed to support reliable support operations, controlled AI orchestration, retrieval-augmented workflows, human approval, observability, and evaluation.

The platform is intentionally structured as an API-first modular monolith. This architecture keeps deployment and operational complexity controlled while preserving clear internal boundaries that can evolve as the system grows.

The current repository phase establishes the operational foundation, workspace-scoped persistence, the Slice 1 workspace and ticket HTTP API, durable AgentRun scheduling, and the PostgreSQL-backed worker. Retrieval and AI orchestration remain later phases.

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

- the API owns HTTP acceptance and transactional Ticket plus initial AgentRun scheduling;
- the worker owns expired lease recovery, claim, execution, and fenced outcome persistence;
- PostgreSQL owns tickets, AgentRuns, attempts, leases, retry scheduling, and execution history;
- Qdrant remains outside the worker flow and is used by the API only for its current connectivity lifecycle.

Delivery semantics are at-least-once execution. Lease-token fencing prevents stale workers from overwriting newer ownership. Exactly-once execution is not claimed. Future executors and tools must make side effects idempotent or otherwise safely fenced.

## Package boundaries

The package structure is organized around framework composition, shared application configuration, infrastructure integration, vertical business modules, and the worker process entry point.

```text
src/supportops/
├── api/
├── application/
├── core/
├── infrastructure/
├── modules/
│   ├── agent_runs/
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

Operational health routes remain unversioned. Workspace and ticket business routes are versioned under `/api/v1`.

### Worker boundary

`supportops.worker` owns the executable worker process composition.

Its responsibilities include:

- loading and validating shared settings at process startup;
- creating a process-owned PostgreSQL engine and session factory;
- resolving worker identity;
- composing one deterministic executor and retry policy;
- running the recovery, claim, and processing loop;
- emitting structured operational cycle logs;
- handling cooperative SIGINT and SIGTERM shutdown;
- disposing the SQLAlchemy engine on exit.

The worker does not initialize or depend on Qdrant. Each polling cycle uses a new `AsyncSession`. Transactions remain short and do not span executor work or idle waits.

### Application boundary

`supportops.application` owns cross-module application use cases that coordinate more than one business module.

Ticket intake is implemented as `CreateTicketWithInitialRun`. That use case coordinates Workspace, Ticket, and AgentRun repositories through one SQLAlchemy session and one application-owned transaction. The HTTP request returns after that transaction commits and does not execute the workflow.

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
```

Workspace and ticket modules follow the implemented internal layout:

```text
domain/
application/
infrastructure/
api/
```

The `agent_runs` module provides domain, application, and infrastructure layers for durable scheduling, claiming, execution, retry, and recovery. AgentRun inspection APIs remain planned.

### Domain

Domain packages own persistence-independent frozen entities, invariants, and repository protocols.

Workspace, Ticket, AgentRun, and AgentRunAttempt entities are frozen dataclasses. They do not depend on SQLAlchemy session state.

Ticket status remains `open` after intake. A queued deterministic-baseline AgentRun records that durable processing has been scheduled. It does not represent AI classification.

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

Command use cases such as workspace creation own the transaction boundary through the application-facing transaction contract. Query use cases such as workspace retrieval, ticket retrieval, and ticket listing execute reads without opening a write transaction.

`CreateTicketWithInitialRun` is the cross-module ticket intake command. It validates the workspace, persists the ticket, and persists the initial AgentRun in one shared SQLAlchemy session and one application-owned transaction. Repositories flush without committing independently.

The `agent_runs` application layer owns:

- bounded retry policy calculation;
- the deterministic baseline executor;
- claimed-run processing outside claim transactions;
- one-cycle worker orchestration;
- the continuous polling loop.

Application services depend on domain repository protocols. They do not depend on FastAPI or SQLAlchemy session APIs directly.

Transactional AgentRun scheduling is documented in [`agent-run-scheduling.md`](agent-run-scheduling.md). Runtime process topology is documented in [`runtime-topology.md`](runtime-topology.md).

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
```

Ticket creation returns a nested ticket object and a minimal processing-run reference. AgentRun inspection endpoints are not implemented yet.

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
- list tickets.

Ticket listing verifies that the workspace exists before returning an empty page. A missing workspace returns `workspace_not_found`. An empty workspace returns an empty `items` collection with a null `next_cursor`.

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
- `workspace_slug_conflict`;
- `ticket_external_reference_conflict`;
- `invalid_pagination_cursor`.

Cross-workspace ticket retrieval returns the same `ticket_not_found` contract as a missing ticket.

## Data ownership

### PostgreSQL

PostgreSQL is the transactional source of truth and the durable AgentRun work queue.

Authoritative business state, workflow state, AgentRun leases, retry scheduling, attempt history, and future approvals, audit records, and usage events are persisted or planned for persistence in PostgreSQL.

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
- `agent_run_attempts`.

Implemented ownership and integrity rules include:

- persistence-independent frozen Workspace, Ticket, AgentRun, and AgentRunAttempt domain entities;
- SQLAlchemy records with explicit mapping;
- repository protocols, including workspace-scoped ticket `get` and `list`;
- application-owned transaction boundaries;
- repository flush without commit;
- PostgreSQL named constraints, including unique workspace slugs and workspace-scoped external references;
- a required foreign key from `tickets.workspace_id` to `workspaces.id`;
- composite workspace and ticket ownership for AgentRun records;
- unique initial trigger enforcement for duplicate initial scheduling prevention;
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

Ticket intake atomically persists both the Ticket and its initial AgentRun. The API returns after the transaction commits. The HTTP request does not execute the workflow. The response includes a minimal `processing_run` projection. Ticket acceptance and asynchronous processing success remain separate outcomes.

The worker process:

- uses PostgreSQL as its durable work queue and source of truth;
- recovers expired leases before claiming available work;
- claims eligible `queued` and `retry_scheduled` runs with due `available_at`;
- uses `FOR UPDATE SKIP LOCKED` so multiple worker processes can claim distinct runs safely;
- executes the deterministic baseline outside database transactions;
- persists success, retry, or failure outcomes through lease-token fencing.

The current executor is `deterministic-ticket-processing`. The persisted workflow contract is:

```text
workflow_name = ticket-processing
workflow_version = deterministic-baseline-v1
trigger_key = initial-ticket-processing
```

The deterministic baseline validates that contract and performs no external I/O, LLM call, retrieval, or ticket classification. It exists to validate the durable execution architecture independently from future AI behavior.

The architecture intentionally avoids Redis, Celery, Kafka, and SQS in this phase because PostgreSQL provides transactional durability and adequate local and portfolio scope. An external queue or outbox is not required for the current worker model.

AgentRun inspection endpoints remain planned.

Scheduling details are documented in [`agent-run-scheduling.md`](agent-run-scheduling.md). Runtime topology details are documented in [`runtime-topology.md`](runtime-topology.md).

## AI system boundaries

AI capabilities are not part of the current implementation phase.

A queued or succeeded deterministic-baseline AgentRun does not indicate that AI classification has occurred. The deterministic executor validates the durable workflow contract only.

Future AI behavior is expected to remain behind application-owned boundaries for:

- LLM providers;
- embeddings;
- orchestration;
- tool execution;
- observability;
- evaluation.

External AI frameworks and providers must not become the source of business rules or workflow ownership.

The following dependencies are intentionally not installed during the repository foundation phase:

- OpenAI SDK;
- Anthropic SDK;
- LangGraph;
- LangChain;
- Langfuse;
- RAGAS;
- embedding providers;
- reranking libraries.

They will be introduced only when a concrete capability requires them.

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
- worker identity, executor, polling, lease, timeout, shutdown, retry, and attempt settings.

Worker settings are validated at process startup. Cross-field invariants require:

- worker lease duration to exceed execution timeout by at least five seconds;
- retry maximum not to be smaller than retry base.

Secrets and complete connection credentials must not be logged.

Invalid required configuration fails clearly during application or worker construction.

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
- application service command and query behavior;
- transactional ticket-intake orchestration;
- retry policy, claiming, fencing, recovery, processor, and worker cycle behavior;
- deterministic executor contract validation;
- worker process identity and graceful shutdown;
- ORM mapping and metadata;
- named constraints;
- persistence model registration;
- PostgreSQL constraint-name inspection;
- API schemas and opaque cursor encoding.

### Integration tests

Integration tests may depend on local PostgreSQL and Qdrant services.

They validate:

- real PostgreSQL connectivity;
- real Qdrant connectivity;
- readiness against live dependencies;
- dependency failure behavior;
- Alembic upgrade, downgrade, re-upgrade, and metadata parity;
- workspace and ticket repository behavior;
- concurrency-sensitive uniqueness enforcement;
- atomic ticket and AgentRun commit and rollback behavior;
- PostgreSQL claim ordering and `SKIP LOCKED` concurrency;
- fenced transitions and expired lease recovery;
- processor transaction separation against live PostgreSQL;
- workspace and ticket HTTP API behavior;
- stable expected-error responses;
- opaque cursor pagination.

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

The repository foundation, Slice 1, durable AgentRun scheduling, and the PostgreSQL worker establish:

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
- AgentRun and AgentRunAttempt persistence;
- atomic Ticket and initial AgentRun scheduling;
- PostgreSQL claiming, leases, fencing, retries, and recovery;
- deterministic baseline execution;
- a separate worker process with cooperative shutdown;
- application services and versioned business APIs;
- stable expected-error contracts;
- opaque cursor pagination;
- request and correlation identifier persistence;
- unit and integration testing;
- continuous integration;
- architecture and development documentation.

## Intentionally deferred scope

The current phase does not implement:

- authentication or authorization;
- authenticated tenant isolation;
- AgentRun inspection endpoints;
- Redis, Celery, Kafka, or SQS;
- LLM calls;
- AI classification;
- prompt execution or versioning;
- token or cost tracking;
- ingestion;
- embeddings;
- Qdrant collections or indexing;
- retrieval;
- LangGraph orchestration;
- registered tools;
- approval workflows;
- AI observability integrations;
- evaluation frameworks;
- frontend applications;
- public cloud deployment;
- infrastructure as code.

Ticket status remains `open` after intake. Durable AgentRun scheduling and the PostgreSQL worker are implemented. Redis, Celery, Kafka, and SQS remain intentionally deferred because PostgreSQL already provides transactional durability and adequate local and portfolio scope for this phase.

These capabilities are deferred to preserve clear scope, avoid speculative abstractions, and keep each implementation slice independently reviewable.
