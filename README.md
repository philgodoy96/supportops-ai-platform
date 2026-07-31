# SupportOps AI Platform

SupportOps AI Platform is a production-minded backend and AI systems engineering project focused on reliable support operations, controlled AI orchestration, retrieval quality, human approval, observability, and evaluation.

The platform is designed as a portfolio-grade engineering system rather than a tutorial chatbot. Its architecture emphasizes clear boundaries, operational reliability, explicit trade-offs, testability, and incremental delivery.

## Project status

The repository foundation, Slice 1 workspace and ticket API, durable AgentRun scheduling, and the PostgreSQL-backed worker are implemented.

The current platform includes:

- reproducible Python dependency management with `uv`;
- a Python 3.12 `src` package layout;
- validated environment-based configuration;
- structured JSON logging with HTTP request traceability;
- a FastAPI application factory and explicit lifecycle management;
- a separate `supportops-worker` process for AgentRun execution;
- local PostgreSQL and Qdrant services through Docker Compose;
- async SQLAlchemy and Qdrant client lifecycle foundations;
- liveness and readiness endpoints;
- bounded and sanitized infrastructure health checks;
- persistence-independent Workspace and Ticket domain entities;
- SQLAlchemy persistence records with explicit domain mapping;
- PostgreSQL `workspaces` and `tickets` tables;
- workspace slug uniqueness;
- workspace-scoped ticket external-reference uniqueness;
- workspace-scoped ticket repository contracts;
- async PostgreSQL repository implementations;
- reversible Alembic migrations for workspace, ticket, and AgentRun tables;
- an application-owned transaction adapter;
- workspace creation and retrieval API;
- workspace-scoped ticket intake;
- atomic Ticket and initial AgentRun persistence;
- durable AgentRun and AgentRunAttempt persistence;
- PostgreSQL claiming with `FOR UPDATE SKIP LOCKED`;
- attempt history, leases, and lease-token fencing;
- bounded retries and expired lease recovery;
- deterministic baseline execution outside database transactions;
- cooperative worker shutdown with structured operational logs;
- database-enforced workspace and ticket ownership for AgentRun records;
- duplicate initial scheduling prevention;
- a minimal processing-run reference returned by ticket creation;
- workspace-scoped ticket retrieval and listing;
- versioned `/api/v1` business routes;
- stable expected-error responses;
- opaque cursor pagination;
- request and correlation identifier persistence;
- cross-workspace isolation behavior;
- application services with command and query use cases;
- repository, application, worker, and API tests;
- Ruff, mypy, and pytest quality gates;
- a reproducible application Docker image;
- GitHub Actions continuous integration;
- professional architecture and development documentation.

Workspace scoping is a data ownership boundary. It is not authentication or authorization, and it is not authenticated tenant isolation.

Ticket acceptance and asynchronous processing success are separate outcomes. A queued `deterministic-baseline-v1` processing run records that durable work has been scheduled. It does not represent AI classification. The deterministic baseline validates the durable execution architecture and performs no LLM call, retrieval, or ticket classification.

AgentRun inspection endpoints, LLM classification, retrieval and Qdrant indexing, LangGraph orchestration, registered tools, human approvals, cost tracking, AI observability, prompt versioning, and evaluation and regression testing remain planned and are not represented as complete.

## Engineering goals

The project is designed to demonstrate:

- production-minded backend architecture;
- explicit domain and infrastructure boundaries;
- reliable asynchronous processing;
- controlled LLM orchestration;
- retrieval-augmented generation over internal runbooks;
- human approval for sensitive actions;
- structured AI observability;
- token and cost accountability;
- retrieval and generation evaluation;
- professional testing, documentation, and Git practices.

## Architecture

The platform follows an API-first modular monolith architecture with separate deployable processes.

The FastAPI process and the PostgreSQL worker process share:

- the same Python package;
- the same application services;
- the same domain model;
- the same PostgreSQL database;
- the same infrastructure adapters where each process requires them.

The API owns HTTP acceptance and transactional AgentRun scheduling. The worker owns recovery, claim, execution, and fenced outcome persistence. PostgreSQL is the durable work queue and transactional source of truth. The worker does not initialize or depend on Qdrant.

Qdrant is treated as a rebuildable retrieval index. Retrieval data must remain reproducible from authoritative source content rather than becoming an independent system of record.

Delivery semantics are at-least-once execution. Lease-token fencing prevents stale workers from overwriting newer ownership. Exactly-once execution is not claimed. Future executors and tools must make side effects idempotent or otherwise safely fenced.

The current runtime foundation uses:

- Python 3.12;
- FastAPI;
- Uvicorn;
- Pydantic v2;
- pydantic-settings;
- SQLAlchemy 2.x with async support;
- asyncpg;
- Alembic;
- PostgreSQL;
- Qdrant;
- Docker Compose;
- pytest;
- pytest-asyncio;
- HTTPX;
- Ruff;
- mypy;
- uv;
- GitHub Actions.

Detailed architecture documentation is maintained under [`docs/architecture`](docs/architecture).

Accepted architectural decisions are recorded under [`docs/decisions`](docs/decisions).

## Current foundation capabilities

### Application runtime

The repository provides:

- explicit FastAPI application construction;
- OpenAPI project metadata;
- process-owned PostgreSQL and Qdrant resources;
- centralized startup and shutdown lifecycle;
- structured JSON logging;
- non-root container execution.

### Operational health

The application exposes:

```text
GET /health/live
GET /health/ready
```

Liveness verifies that the application process can respond.

Readiness verifies PostgreSQL and Qdrant connectivity using bounded timeouts.

When a required dependency is unavailable:

- liveness remains independent;
- readiness returns HTTP `503 Service Unavailable`;
- the unhealthy dependency is identified;
- credentials, connection URLs, and raw provider exceptions are not exposed.

### PostgreSQL foundation

The PostgreSQL integration includes:

- async SQLAlchemy engine construction;
- async session factory construction;
- process-owned connection lifecycle;
- pool configuration;
- `SELECT 1` connectivity checks;
- shared declarative metadata;
- deterministic constraint naming;
- Alembic async migration configuration;
- registered workspace, ticket, and AgentRun persistence models;
- reversible migrations that create `workspaces`, `tickets`, `agent_runs`, and `agent_run_attempts`.

### Workspace and ticket persistence

The first business modules provide:

- frozen Workspace and Ticket domain entities with validated invariants;
- SQLAlchemy records that own table definitions, constraints, indexes, and mapping;
- repository protocols with workspace-scoped ticket access;
- async SQLAlchemy repository implementations that flush without committing;
- named uniqueness constraints for workspace slugs and workspace-scoped external references;
- a minimal SQLAlchemy transaction adapter for application-owned boundaries;
- repository integration coverage, including concurrency-sensitive duplicate external-reference insertion.

### Durable AgentRun scheduling and PostgreSQL worker

Ticket intake schedules durable processing in the same application-owned transaction that creates the ticket. The HTTP request returns after that transaction commits and does not execute the workflow.

Implemented behavior includes:

- frozen AgentRun and AgentRunAttempt domain entities with validated invariants;
- PostgreSQL `agent_runs` and `agent_run_attempts` tables with query-driven indexes;
- composite workspace and ticket ownership enforcement;
- unique initial trigger enforcement for duplicate initial scheduling prevention;
- atomic Ticket and initial AgentRun creation;
- a persisted initial retry budget copied from configuration;
- a minimal processing-run reference in the ticket creation response;
- a separate `supportops-worker` process using PostgreSQL as its durable work queue;
- claim eligibility for `queued` and `retry_scheduled` runs with due `available_at`;
- PostgreSQL `FOR UPDATE SKIP LOCKED` claiming across multiple worker processes;
- attempt history, leases, and lease-token fencing;
- bounded exponential backoff retries;
- expired lease recovery before each claim cycle;
- deterministic baseline execution (`deterministic-ticket-processing`) outside transactions;
- cooperative SIGINT and SIGTERM shutdown with engine disposal.

A queued deterministic-baseline run does not represent AI classification. AgentRun inspection endpoints remain planned.

Scheduling and worker handoff behavior are documented in [`docs/architecture/agent-run-scheduling.md`](docs/architecture/agent-run-scheduling.md) and [`docs/architecture/runtime-topology.md`](docs/architecture/runtime-topology.md).

### Workspace and ticket API

Slice 1 exposes versioned business routes under `/api/v1`:

- workspace creation and retrieval;
- workspace-scoped ticket creation, retrieval, and listing;
- opaque cursor pagination for ticket listing;
- stable expected-error responses for missing resources, conflicts, and invalid cursors;
- persistence of request and correlation identifiers on accepted tickets;
- cross-workspace retrieval that returns the same `404` contract as a missing ticket.

Health routes remain unversioned. Workspace scoping is not authentication or authorization.

Reproducible request examples are documented in [`docs/development/api-examples.md`](docs/development/api-examples.md).

### Qdrant foundation

The Qdrant integration includes:

- async client construction;
- explicit client lifecycle;
- environment-based endpoint configuration;
- optional API key configuration;
- bounded read-only connectivity checks.

No collections, vectors, embeddings, ingestion pipelines, or retrieval behavior exist yet.

### Testing and quality

The repository includes:

- unit tests isolated from Docker and network services;
- integration tests against real PostgreSQL and Qdrant services;
- domain invariant tests, including AgentRun and AgentRunAttempt;
- application service unit coverage;
- transactional ticket-intake unit coverage;
- worker claim, retry, fencing, recovery, and process unit coverage;
- workspace and ticket API schema and pagination unit coverage;
- ORM mapping, named-constraint, and model-registration tests;
- repository integration and concurrency-sensitive tests, including SKIP LOCKED claiming;
- workspace and ticket API integration coverage;
- atomic ticket and AgentRun commit and rollback coverage;
- Alembic upgrade, downgrade, and metadata-parity coverage;
- settings validation tests;
- lifecycle tests;
- dependency failure-path tests;
- liveness and readiness tests;
- response sanitization tests;
- HTTP request traceability tests;
- Ruff linting and formatting checks;
- strict mypy validation;
- Docker image build validation;
- GitHub Actions quality gates.

## Planned platform modules

The repository already includes bounded `workspaces`, `tickets`, and `agent_runs` modules, plus a `supportops.worker` process entry point. Workspace and ticket modules expose domain entities, application services, repository contracts, PostgreSQL persistence, and versioned HTTP APIs. The `agent_runs` module provides durable scheduling, claiming, execution, retry, and recovery foundations coordinated during ticket intake and by the worker process.

Future implementation phases are expected to introduce or extend modules for:

- AgentRun inspection endpoints;
- structured LLM classification;
- internal runbook ingestion;
- semantic retrieval and Qdrant indexing;
- LangGraph orchestration;
- explicitly registered tools;
- human approval workflows;
- usage and cost tracking;
- AI observability;
- prompt versioning;
- retrieval and generation evaluation;
- regression testing.

Additional modules will be introduced only when they have concrete responsibilities and tested behavior.

## Repository structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yaml
├── alembic/
│   ├── versions/
│   ├── env.py
│   └── script.py.mako
├── docs/
│   ├── architecture/
│   │   ├── agent-run-scheduling.md
│   │   ├── overview.md
│   │   ├── runtime-topology.md
│   │   └── workspace-data-boundary.md
│   ├── decisions/
│   │   ├── 0001-use-a-modular-monolith.md
│   │   ├── 0002-use-postgresql-as-the-source-of-truth.md
│   │   ├── 0003-use-qdrant-as-a-rebuildable-retrieval-index.md
│   │   ├── 0004-use-a-postgresql-backed-worker-model.md
│   │   ├── 0005-keep-ai-observability-behind-an-adapter.md
│   │   └── 0006-establish-workspace-scoped-data-ownership.md
│   └── development/
│       ├── api-examples.md
│       ├── environment-variables.md
│       ├── local-setup.md
│       └── testing.md
├── src/
│   └── supportops/
│       ├── api/
│       │   ├── health/
│       │   ├── application.py
│       │   ├── lifespan.py
│       │   ├── main.py
│       │   ├── router.py
│       │   └── state.py
│       ├── application/
│       │   └── ticket_intake.py
│       ├── core/
│       │   ├── logging.py
│       │   ├── request_context.py
│       │   ├── settings.py
│       │   └── transactions.py
│       ├── infrastructure/
│       │   ├── postgresql/
│       │   └── qdrant/
│       ├── modules/
│       │   ├── agent_runs/
│       │   ├── tickets/
│       │   └── workspaces/
│       └── worker/
│           ├── __init__.py
│           └── main.py
├── tests/
│   ├── integration/
│   └── unit/
├── .env.example
├── .gitignore
├── .python-version
├── alembic.ini
├── compose.yaml
├── Dockerfile
├── pyproject.toml
├── README.md
└── uv.lock
```

Business modules are introduced when they have concrete responsibilities. The current `workspaces`, `tickets`, and `agent_runs` modules include domain and infrastructure layers. Workspace and ticket modules also expose application and API layers. The `agent_runs` module includes worker-facing application services. Cross-module ticket intake lives under `supportops.application`. The worker process entry point lives under `supportops.worker`.

## Local setup

Install locked dependencies:

```powershell
uv sync --frozen --all-groups
```

Create local configuration:

```powershell
Copy-Item .env.example .env
```

Start PostgreSQL and Qdrant:

```powershell
docker compose up -d
docker compose ps
```

Apply migrations before exercising business routes:

```powershell
uv run alembic upgrade head
```

Start the API in one terminal:

```powershell
uv run uvicorn supportops.api.main:app `
  --host 127.0.0.1 `
  --port 8000
```

Start the worker in another terminal:

```powershell
$env:SUPPORTOPS_WORKER_ID="worker-local-1"
uv run supportops-worker
```

Docker Compose provisions infrastructure only. A worker service is intentionally not added to Compose in this phase.

Validate liveness:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
```

Validate readiness:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

Workspace and ticket request examples are documented in [`docs/development/api-examples.md`](docs/development/api-examples.md).

The complete setup procedure is documented in [`docs/development/local-setup.md`](docs/development/local-setup.md).

## Configuration

Application configuration uses environment variables prefixed with:

```text
SUPPORTOPS_
```

The repository includes a safe local example:

```text
.env.example
```

Required application values include:

```text
SUPPORTOPS_POSTGRESQL_URL
SUPPORTOPS_QDRANT_URL
```

Worker timing and identity are controlled by `SUPPORTOPS_WORKER_*` variables. Defaults are validated at process startup, including lease-versus-timeout and retry-base-versus-max invariants.

The complete configuration contract is documented in [`docs/development/environment-variables.md`](docs/development/environment-variables.md).

## Quality commands

Run linting:

```powershell
uv run ruff check .
```

Verify formatting:

```powershell
uv run ruff format --check .
```

Run type checking:

```powershell
uv run mypy
```

Validate the lockfile:

```powershell
uv lock --check
```

Validate Docker Compose:

```powershell
docker compose config --quiet
```

## Test commands

Run unit tests:

```powershell
uv run pytest -m "not integration"
```

Run integration tests:

```powershell
uv run pytest -m integration
```

Run the complete test suite:

```powershell
uv run pytest
```

The complete testing strategy is documented in [`docs/development/testing.md`](docs/development/testing.md).

## Alembic commands

Apply the current migration head:

```powershell
uv run alembic upgrade head
```

Validate migration heads and connectivity:

```powershell
uv run alembic heads
uv run alembic current
uv run alembic check
```

Validate offline migration execution:

```powershell
uv run alembic upgrade head --sql
```

The current head creates the `workspaces`, `tickets`, `agent_runs`, and `agent_run_attempts` tables. Downgrade commands must run only against the local development or test database.

## Docker image

Build the application image:

```powershell
docker build `
  --tag supportops-ai-platform:local `
  .
```

The image:

- installs dependencies from the committed lockfile;
- excludes development dependencies;
- runs the FastAPI application;
- executes as a non-root user.

## Continuous integration

The GitHub Actions workflow validates pull requests and pushes to `main`.

The workflow executes:

- frozen dependency installation;
- lockfile validation;
- Ruff lint;
- Ruff formatting verification;
- mypy;
- unit tests;
- Alembic validation;
- integration tests with PostgreSQL and Qdrant service containers;
- Docker image build.

The workflow uses Python 3.12 and does not publish artifacts or images.

## Architecture decisions

The repository records the following accepted decisions:

- [Use a modular monolith](docs/decisions/0001-use-a-modular-monolith.md)
- [Use PostgreSQL as the source of truth](docs/decisions/0002-use-postgresql-as-the-source-of-truth.md)
- [Use Qdrant as a rebuildable retrieval index](docs/decisions/0003-use-qdrant-as-a-rebuildable-retrieval-index.md)
- [Use a PostgreSQL-backed worker model](docs/decisions/0004-use-a-postgresql-backed-worker-model.md)
- [Keep AI observability behind an application-owned adapter](docs/decisions/0005-keep-ai-observability-behind-an-adapter.md)
- [Establish workspace-scoped data ownership](docs/decisions/0006-establish-workspace-scoped-data-ownership.md)

## Roadmap

### Repository foundation

Implemented:

- architecture documentation;
- dependency management;
- environment configuration;
- local infrastructure;
- FastAPI bootstrap;
- structured logging;
- HTTP request traceability;
- health endpoints;
- Alembic;
- automated tests;
- Docker packaging;
- CI quality gates.

### Support operations

Implemented:

- workspace and ticket domain entities;
- PostgreSQL persistence and repository contracts;
- workspace-scoped data ownership;
- reversible workspace and ticket migration;
- application services and API error contracts;
- versioned workspace and ticket HTTP endpoints;
- opaque HTTP cursor pagination;
- request and correlation identifier persistence;
- atomic ticket intake with initial AgentRun scheduling.

Planned:

- structured classification;
- operational auditability beyond request and correlation identifiers.

### Asynchronous processing

Implemented:

- AgentRun and AgentRunAttempt domain and persistence foundations;
- atomic Ticket and initial AgentRun scheduling;
- database-enforced workspace and ticket ownership;
- duplicate initial scheduling prevention;
- persisted initial retry budget;
- minimal processing-run reference on ticket creation;
- PostgreSQL-backed worker execution;
- queue claiming with `FOR UPDATE SKIP LOCKED`;
- leases and lease-token fencing;
- bounded exponential backoff retries;
- expired lease recovery;
- deterministic baseline executor;
- separate worker process with cooperative shutdown.

Planned:

- AgentRun inspection endpoints;
- idempotent side effects for future executors and tools.

### Retrieval

Planned:

- runbook ingestion;
- chunking;
- embeddings;
- Qdrant collections and indexing;
- retrieval quality controls.

### Controlled orchestration

Planned:

- LangGraph workflows;
- registered tools;
- approval boundaries;
- failure recovery.

### Observability and evaluation

Planned:

- AI tracing;
- token and cost tracking;
- prompt versioning;
- retrieval evaluation;
- generation evaluation;
- prompt regression testing.

## Intentionally deferred capabilities

The following capabilities remain deferred to preserve architectural focus and avoid speculative abstractions:

- authentication and authorization;
- authenticated tenant isolation;
- AgentRun inspection endpoints;
- Redis, Celery, Kafka, and SQS;
- LLM provider integrations;
- AI classification;
- prompt execution and versioning;
- embeddings and retrieval;
- Qdrant collections and indexing;
- LangGraph orchestration;
- registered tools;
- human approval workflows;
- cost tracking;
- AI observability integrations;
- Langfuse integration;
- RAGAS evaluation;
- evaluation and regression testing frameworks;
- OpenTelemetry;
- Prometheus and Grafana;
- frontend applications;
- cloud deployment;
- infrastructure as code;
- Kubernetes.

Workspace scoping establishes data ownership. It is not authentication or authorization, and it does not establish caller identity or secure multi-tenancy.

Durable AgentRun scheduling and the PostgreSQL worker are implemented. Redis, Celery, Kafka, and SQS remain intentionally deferred because PostgreSQL already provides transactional durability and adequate local and portfolio scope for this phase. An external queue or outbox is not required for the current worker model.

The architecture keeps room for these capabilities without introducing dependencies or abstractions before they have concrete responsibilities.

## Documentation

- [Architecture overview](docs/architecture/overview.md)
- [Runtime topology](docs/architecture/runtime-topology.md)
- [Transactional AgentRun scheduling](docs/architecture/agent-run-scheduling.md)
- [Workspace-scoped data ownership](docs/architecture/workspace-data-boundary.md)
- [Architecture decision records](docs/decisions)
- [Local setup](docs/development/local-setup.md)
- [API examples](docs/development/api-examples.md)
- [Environment variables](docs/development/environment-variables.md)
- [Testing strategy](docs/development/testing.md)

## License

No open-source license has been selected for this repository.
