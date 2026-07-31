# Architecture Overview

## Purpose

SupportOps AI Platform is a production-minded backend and AI systems engineering project designed to support reliable support operations, controlled AI orchestration, retrieval-augmented workflows, human approval, observability, and evaluation.

The platform is intentionally structured as an API-first modular monolith. This architecture keeps deployment and operational complexity controlled while preserving clear internal boundaries that can evolve as the system grows.

The current repository phase establishes the operational foundation, including HTTP request traceability, workspace-scoped persistence, and the Slice 1 workspace and ticket HTTP API. Asynchronous processing, retrieval, and AI orchestration remain later phases.

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

The platform is designed as a modular monolith with two eventual runtime entry points:

- an HTTP API process;
- an asynchronous worker process.

Both processes will share:

- the same Python package;
- the same domain model;
- the same application services;
- the same PostgreSQL database;
- the same infrastructure adapters.

The processes will have distinct runtime responsibilities, but they will remain part of one deployable codebase during the initial platform evolution.

The worker entry point and processing behavior are not implemented in the repository foundation phase.

## Package boundaries

The package structure is organized around framework composition, shared application configuration, infrastructure integration, and vertical business modules.

```text
src/supportops/
├── api/
├── core/
├── infrastructure/
└── modules/
    ├── workspaces/
    └── tickets/
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

The API boundary may depend on `supportops.core`, `supportops.infrastructure`, and module application services.

Infrastructure and core packages must not depend on the API package.

Operational health routes remain unversioned. Workspace and ticket business routes are versioned under `/api/v1`.

### Core boundary

`supportops.core` contains cross-cutting capabilities that are independent from business modules and external providers.

Its responsibilities include:

- validated environment-based settings;
- logging configuration;
- provider-independent request and correlation context;
- application transaction boundary contracts;
- shared operational configuration types.

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
```

Each module follows the implemented internal layout:

```text
domain/
application/
infrastructure/
api/
```

### Domain

Domain packages own persistence-independent frozen entities, invariants, and repository protocols.

Workspace and Ticket entities are frozen dataclasses. They do not depend on SQLAlchemy session state.

Ticket status remains `open` after intake. AI execution state is intentionally outside the ticket lifecycle and will belong to AgentRun in Slice 2.

### Application

Application packages own use cases and transaction orchestration.

Command use cases such as workspace creation and ticket intake own the transaction boundary through the application-facing transaction contract. Query use cases such as workspace retrieval, ticket retrieval, and ticket listing execute reads without opening a write transaction.

Application services depend on domain repository protocols. They do not depend on FastAPI or SQLAlchemy session APIs directly.

### Infrastructure

Module infrastructure packages own:

- SQLAlchemy persistence records;
- explicit domain-to-record and record-to-domain mapping;
- named PostgreSQL constraints and indexes;
- async repository implementations.

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

Expected application errors use a stable envelope with `code`, `message`, and `request_id`. Malformed identifiers and invalid request schemas use FastAPI validation responses.

## Dependency direction

The implemented dependency direction is:

```text
API
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
    ↓
supportops.modules.*.api
    ↓
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
- create ticket.

Those use cases open an application-owned transaction, persist through repositories that flush without committing, and commit or roll back as one unit.

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

PostgreSQL is the transactional source of truth.

Authoritative business state, workflow state, approvals, audit records, usage events, and future asynchronous job state will be persisted in PostgreSQL.

This decision supports:

- transactional consistency;
- explicit relational constraints;
- durable workflow state;
- reliable auditing;
- idempotency;
- future PostgreSQL-backed asynchronous processing.

The first business migration creates:

- `workspaces`;
- `tickets`.

Implemented ownership and integrity rules include:

- persistence-independent frozen Workspace and Ticket domain entities;
- SQLAlchemy records with explicit mapping;
- repository protocols, including workspace-scoped ticket `get` and `list`;
- application-owned transaction boundaries;
- repository flush without commit;
- PostgreSQL named constraints, including unique workspace slugs and workspace-scoped external references;
- a required foreign key from `tickets.workspace_id` to `workspaces.id`;
- a workspace-leading listing index for deterministic ticket ordering.

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

No collections, embeddings, ingestion pipelines, or retrieval behavior are implemented. Qdrant is not involved in workspace or ticket persistence.

## Asynchronous processing direction

A later phase will introduce a PostgreSQL-backed asynchronous worker model.

The initial architecture intentionally avoids Redis, Celery, Kafka, and SQS because the first platform version prioritizes:

- transactional reliability;
- a smaller operational surface;
- direct visibility into workflow state;
- controlled failure recovery;
- portfolio-grade demonstration of concurrency and idempotency fundamentals.

The API and worker will eventually run as separate processes while sharing the same package and persistence model.

The repository foundation does not implement:

- worker polling;
- job claiming;
- leases;
- retry processing;
- queue abstractions;
- worker entry points.

## AI system boundaries

AI capabilities are not part of the current implementation phase.

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

## Configuration

Runtime configuration is environment-based and validated.

The initial settings model includes:

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
- dependency health-check timeout.

Secrets and complete connection credentials must not be logged.

Invalid required configuration fails clearly during application construction.

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

No external logging platform is introduced during the repository foundation phase.

## Testing boundaries

Tests are separated by infrastructure dependency.

### Unit tests

Unit tests do not require Docker or network services.

They validate behavior such as:

- settings validation;
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
- domain invariants;
- application service command and query behavior;
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
- Alembic upgrade, downgrade, and re-upgrade;
- workspace and ticket repository behavior;
- concurrency-sensitive uniqueness enforcement;
- workspace and ticket HTTP API behavior;
- stable expected-error responses;
- opaque cursor pagination.

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
- omission of request bodies from completion logs.

Trace identifiers support observability and supportability. They are not authentication or authorization controls.

Workspace scoping is a data ownership boundary. It is not authentication, authorization, or authenticated tenant isolation.

Authentication, authorization, authenticated tenant isolation, and public deployment hardening are intentionally deferred to later implementation phases.

## Scaling considerations

The modular monolith is intended to support early and intermediate platform growth without introducing distributed coordination prematurely.

The architecture keeps room for:

- separate API and worker scaling;
- independent process deployment;
- PostgreSQL connection pool tuning;
- Qdrant deployment changes;
- provider adapter replacement;
- extraction of modules when operational evidence justifies it.

Service extraction is not a default objective. It should be driven by clear ownership, scaling, reliability, or deployment requirements.

## Repository foundation scope

The repository foundation and Slice 1 establish:

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
- asynchronous worker behavior;
- AgentRun;
- job queues, claiming, leases, retries, or worker recovery;
- LLM calls;
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

Ticket status remains `open` after intake. AI execution state will belong to AgentRun in Slice 2 rather than expanding the ticket lifecycle prematurely.

These capabilities are deferred to preserve clear scope, avoid speculative abstractions, and keep each implementation slice independently reviewable.
