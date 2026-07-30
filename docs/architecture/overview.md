# Architecture Overview

## Purpose

SupportOps AI Platform is a production-minded backend and AI systems engineering project designed to support reliable support operations, controlled AI orchestration, retrieval-augmented workflows, human approval, observability, and evaluation.

The platform is intentionally structured as an API-first modular monolith. This architecture keeps deployment and operational complexity controlled while preserving clear internal boundaries that can evolve as the system grows.

The current repository phase documents the architecture and runtime direction only. Business modules, asynchronous processing, retrieval, and AI orchestration are introduced in later phases.

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

## Planned foundation package boundaries

The initial package structure is intended to be organized around framework composition, shared application configuration, and infrastructure integration.

```text
src/supportops/
├── api/
├── core/
└── infrastructure/
```

### API boundary

`supportops.api` owns the HTTP application boundary.

Its responsibilities include:

- creating and configuring the FastAPI application;
- composing routers;
- defining OpenAPI metadata;
- managing application startup and shutdown;
- exposing operational health endpoints;
- translating application outcomes into HTTP responses.

The API boundary may depend on `supportops.core` and `supportops.infrastructure`.

Infrastructure and core packages must not depend on the API package.

### Core boundary

`supportops.core` contains cross-cutting capabilities that are independent from business modules and external providers.

Its initial responsibilities include:

- validated environment-based settings;
- logging configuration;
- shared operational configuration types.

The core package is not a generic location for unrelated helpers. New code belongs in `supportops.core` only when its responsibility is genuinely cross-cutting and provider-independent.

### Infrastructure boundary

`supportops.infrastructure` owns integrations with external systems and technical persistence frameworks.

The initial infrastructure packages are:

```text
supportops.infrastructure.postgresql
supportops.infrastructure.qdrant
```

The PostgreSQL package owns:

- async SQLAlchemy engine configuration;
- async session creation;
- declarative metadata;
- connection lifecycle;
- database connectivity checks.

The Qdrant package owns:

- async client construction;
- client lifecycle;
- connectivity checks;
- provider-specific configuration handling.

Infrastructure packages must not contain business rules.

## Future business modules

Business modules will be introduced under:

```text
supportops.modules
```

The package is intentionally not created during the repository foundation phase because no concrete business capability exists yet.

Future modules are expected to own cohesive business concerns such as:

- workspace-scoped support operations;
- support ticket processing;
- structured classification;
- retrieval workflows;
- controlled orchestration;
- approval workflows;
- usage accounting;
- evaluation.

Each module should contain only abstractions and behavior required by its responsibility. Empty modules and speculative interfaces are avoided.

## Dependency direction

The intended dependency direction is:

```text
API entry points
    ↓
Application services and business modules
    ↓
Domain behavior and ports
    ↓
Infrastructure adapters
```

During the repository foundation phase, business and application service layers do not yet exist. The first executable foundation commit is intended to implement this dependency direction:

```text
supportops.api
    ↓
supportops.core
supportops.infrastructure
```

Future infrastructure adapters may implement interfaces owned by application or domain layers, but application behavior must not become coupled to provider-specific implementations.

Circular dependencies are not permitted.

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

No business tables are introduced during the repository foundation phase.

### Qdrant

Qdrant is a rebuildable retrieval index.

It will store vectorized representations required for semantic retrieval, but it will not own authoritative document or workflow state.

The retrieval index must remain reproducible from source content and ingestion metadata stored in authoritative systems.

During the repository foundation phase, Qdrant is limited to:

- local service configuration;
- environment-based client configuration;
- lifecycle management;
- bounded connectivity checks.

No collections, embeddings, ingestion pipelines, or retrieval behavior are implemented.

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
- future request and correlation context.

Request identifiers and correlation identifiers are intentionally deferred.

No external logging platform is introduced during the repository foundation phase.

## Testing boundaries

Tests are separated by infrastructure dependency.

### Unit tests

Unit tests do not require Docker or network services.

They validate behavior such as:

- settings validation;
- application construction;
- liveness behavior;
- readiness aggregation;
- dependency failure handling.

### Integration tests

Integration tests may depend on local PostgreSQL and Qdrant services.

They validate:

- real PostgreSQL connectivity;
- real Qdrant connectivity;
- readiness against live dependencies;
- dependency failure behavior;
- Alembic configuration.

The test suite must verify externally observable behavior rather than reproduce implementation details.

## Security considerations

The repository foundation establishes the following security baseline:

- no committed credentials;
- safe example environment values;
- validated configuration;
- no secret logging;
- health responses without credential disclosure;
- bounded dependency checks;
- explicit lifecycle handling.

Authentication, authorization, tenant isolation, and public deployment hardening are intentionally deferred to later implementation phases.

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

The repository foundation phase is expected to establish:

- project and dependency management;
- local PostgreSQL and Qdrant infrastructure;
- application configuration;
- structured logging;
- FastAPI bootstrap;
- lifecycle management;
- liveness and readiness endpoints;
- infrastructure connectivity checks;
- SQLAlchemy and Alembic foundations;
- unit and integration testing;
- continuous integration;
- architecture and development documentation.

## Intentionally deferred scope

The current phase does not implement:

- business entities;
- business database tables;
- workspace isolation;
- authentication or authorization;
- asynchronous worker behavior;
- job queues;
- LLM calls;
- prompt execution;
- token or cost tracking;
- ingestion;
- embeddings;
- Qdrant collections;
- retrieval;
- orchestration;
- tool execution;
- approval workflows;
- AI observability integrations;
- evaluation frameworks;
- frontend applications;
- public cloud deployment;
- infrastructure as code.

These capabilities are deferred to preserve clear scope, avoid speculative abstractions, and keep the foundation independently reviewable.
