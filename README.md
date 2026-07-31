# SupportOps AI Platform

SupportOps AI Platform is a production-minded backend and AI systems engineering project focused on reliable support operations, controlled AI orchestration, retrieval quality, human approval, observability, and evaluation.

The platform is designed as a portfolio-grade engineering system rather than a tutorial chatbot. Its architecture emphasizes clear boundaries, operational reliability, explicit trade-offs, testability, and incremental delivery.

## Project status

The repository foundation and the first workspace-scoped persistence slice are implemented.

The current platform includes:

- reproducible Python dependency management with `uv`;
- a Python 3.12 `src` package layout;
- validated environment-based configuration;
- structured JSON logging with HTTP request traceability;
- a FastAPI application factory and explicit lifecycle management;
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
- a reversible Alembic migration for workspace and ticket tables;
- an application-owned transaction adapter;
- repository integration and concurrency-sensitive tests;
- unit and integration tests;
- Ruff, mypy, and pytest quality gates;
- a reproducible application Docker image;
- GitHub Actions continuous integration;
- professional architecture and development documentation.

Workspace scoping is a data ownership boundary. It is not authenticated tenant isolation.

Workspace and ticket HTTP endpoints, application services, API error contracts, HTTP cursor encoding, asynchronous processing, AgentRun, LLM classification, retrieval, LangGraph, tools, approvals, observability, and evaluation remain planned and are not represented as complete.

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

The platform follows an API-first modular monolith architecture.

The API process and future asynchronous worker will share:

- the same Python package;
- the same application services;
- the same domain model;
- the same PostgreSQL database;
- the same infrastructure adapters.

PostgreSQL is the transactional source of truth.

Qdrant is treated as a rebuildable retrieval index. Retrieval data must remain reproducible from authoritative source content rather than becoming an independent system of record.

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
- registered workspace and ticket persistence models;
- a reversible migration that creates `workspaces` and `tickets`.

### Workspace and ticket persistence

The first business modules provide:

- frozen Workspace and Ticket domain entities with validated invariants;
- SQLAlchemy records that own table definitions, constraints, indexes, and mapping;
- repository protocols with workspace-scoped ticket access;
- async SQLAlchemy repository implementations that flush without committing;
- named uniqueness constraints for workspace slugs and workspace-scoped external references;
- a minimal SQLAlchemy transaction adapter for application-owned boundaries;
- repository integration coverage, including concurrency-sensitive duplicate external-reference insertion.

Workspace and ticket HTTP endpoints are not implemented.

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
- domain invariant tests;
- ORM mapping, named-constraint, and model-registration tests;
- repository integration and concurrency-sensitive tests;
- Alembic upgrade and downgrade coverage;
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

The repository already includes bounded `workspaces` and `tickets` modules for domain entities, repository contracts, and PostgreSQL persistence.

Future implementation phases are expected to introduce or extend modules for:

- workspace and ticket HTTP endpoints;
- application services and API error contracts;
- HTTP cursor encoding at the API boundary;
- structured LLM classification;
- internal runbook ingestion;
- semantic retrieval;
- controlled orchestration;
- explicitly registered tools;
- human approval workflows;
- asynchronous processing and AgentRun;
- usage and cost tracking;
- AI observability;
- retrieval and generation evaluation;
- prompt versioning;
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
│       ├── core/
│       │   ├── logging.py
│       │   ├── request_context.py
│       │   ├── settings.py
│       │   └── transactions.py
│       ├── infrastructure/
│       │   ├── postgresql/
│       │   └── qdrant/
│       └── modules/
│           ├── tickets/
│           └── workspaces/
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

Business modules are introduced when they have concrete responsibilities. Application and API layers inside the current modules remain future work until HTTP use cases are implemented.

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

Start the API:

```powershell
uv run uvicorn supportops.api.main:app `
  --host 127.0.0.1 `
  --port 8000
```

Validate liveness:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
```

Validate readiness:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

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

The current head creates the `workspaces` and `tickets` tables. Downgrade commands must run only against the local development or test database.

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
- reversible workspace and ticket migration.

Planned:

- workspace and ticket HTTP endpoints;
- application services and API error contracts;
- HTTP cursor encoding;
- structured classification;
- operational auditability.

### Asynchronous processing

Planned:

- PostgreSQL-backed work records;
- atomic claiming;
- retry behavior;
- stale ownership recovery;
- idempotent processing.

### Retrieval

Planned:

- runbook ingestion;
- chunking;
- embeddings;
- Qdrant collections;
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
- retrieval evaluation;
- generation evaluation;
- prompt regression testing.

## Intentionally deferred capabilities

The following capabilities remain deferred to preserve architectural focus and avoid speculative abstractions:

- workspace and ticket HTTP endpoints;
- application services and API error contracts;
- HTTP cursor encoding;
- authentication and authorization;
- authenticated tenant isolation;
- worker polling and job claiming;
- AgentRun and asynchronous processing;
- Redis, Celery, Kafka, and SQS;
- LLM provider integrations;
- prompt execution;
- embeddings and retrieval;
- Qdrant collections;
- LangGraph orchestration;
- human approval workflows;
- Langfuse integration;
- RAGAS evaluation;
- OpenTelemetry;
- Prometheus and Grafana;
- frontend applications;
- cloud deployment;
- infrastructure as code;
- Kubernetes.

Workspace scoping establishes data ownership. It does not establish caller identity or secure multi-tenancy.

The architecture keeps room for these capabilities without introducing dependencies or abstractions before they have concrete responsibilities.

## Documentation

- [Architecture overview](docs/architecture/overview.md)
- [Runtime topology](docs/architecture/runtime-topology.md)
- [Workspace-scoped data ownership](docs/architecture/workspace-data-boundary.md)
- [Architecture decision records](docs/decisions)
- [Local setup](docs/development/local-setup.md)
- [Environment variables](docs/development/environment-variables.md)
- [Testing strategy](docs/development/testing.md)

## License

No open-source license has been selected for this repository.
