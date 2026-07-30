# SupportOps AI Platform

SupportOps AI Platform is a production-minded backend and AI systems engineering project focused on reliable support operations, controlled AI orchestration, retrieval quality, human approval, observability, and evaluation.

The platform is designed as a portfolio-grade engineering system rather than a tutorial chatbot. Its architecture emphasizes clear boundaries, operational reliability, explicit trade-offs, testability, and incremental delivery.

## Project status

The repository is currently in the architecture and repository foundation phase.

Implemented capabilities are limited to repository initialization and architecture documentation. The current phase defines the system architecture and intended development direction; executable application bootstrap, database migration tooling, automated tests, and continuous integration are planned for later foundation commits.

Business workflows and AI capabilities are planned for later implementation phases and are not represented as complete.

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

The API process and the future asynchronous worker will share:

- the same Python package;
- the same application services;
- the same domain model;
- the same PostgreSQL database;
- the same infrastructure adapters.

PostgreSQL is the transactional source of truth.

Qdrant is treated as a rebuildable retrieval index. Retrieval data must remain reproducible from authoritative source content rather than becoming an independent system of record.

The initial runtime foundation uses:

- Python 3.12;
- FastAPI;
- Pydantic v2;
- SQLAlchemy 2.x with async support;
- PostgreSQL;
- Qdrant;
- Alembic;
- Docker Compose;
- pytest;
- Ruff;
- mypy;
- uv;
- GitHub Actions.

Detailed architecture documentation is maintained under [`docs/architecture`](docs/architecture).

Accepted architectural decisions are recorded under [`docs/decisions`](docs/decisions).

## Current foundation scope

The repository foundation will provide:

- reproducible Python dependency management;
- environment-based configuration;
- local PostgreSQL and Qdrant services;
- explicit FastAPI application composition;
- structured JSON logging;
- application liveness and readiness endpoints;
- bounded dependency health checks;
- async SQLAlchemy connection management;
- Qdrant client lifecycle management;
- Alembic migration infrastructure;
- unit and integration testing;
- local quality commands;
- continuous integration quality gates;
- professional architecture and development documentation.

## Planned platform modules

Future implementation phases are expected to introduce bounded modules for:

- workspace-scoped support operations;
- support ticket intake and processing;
- structured LLM classification;
- internal runbook ingestion;
- semantic retrieval;
- controlled orchestration;
- explicitly registered tools;
- human approval workflows;
- usage and cost tracking;
- AI observability;
- retrieval and generation evaluation;
- prompt versioning;
- regression testing.

These modules will be introduced only when they have concrete responsibilities and tested behavior.

## Operational health

The application foundation distinguishes two operational health concepts:

- `GET /health/live` verifies that the application process is running;
- `GET /health/ready` verifies whether required infrastructure dependencies are available.

Liveness does not depend on PostgreSQL or Qdrant.

Readiness evaluates required dependencies using bounded timeouts and returns a structured non-success response when the application cannot safely accept workload.

## Repository structure

The planned foundation structure is:

```text
.
├── .github/
│   └── workflows/
├── alembic/
├── docs/
│   ├── architecture/
│   ├── decisions/
│   └── development/
├── src/
│   └── supportops/
│       ├── api/
│       ├── core/
│       └── infrastructure/
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

Business modules are intentionally not created as empty packages. They will be introduced when the first concrete domain capability is implemented.

## Local development

Detailed local setup, environment variable, and testing documentation will be added as the repository foundation becomes executable.

The expected local workflow will use `uv` for dependency management and Docker Compose for PostgreSQL and Qdrant.

## Quality strategy

The repository quality gates will include:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest -m "not integration"
uv run pytest -m integration
```

The exact commands will be validated and documented as the repository foundation is implemented.

## Roadmap

### Foundation

- repository architecture;
- dependency management;
- local infrastructure;
- FastAPI bootstrap;
- health endpoints;
- Alembic;
- automated tests;
- CI quality gates.

### Support operations

- workspace boundaries;
- support tickets;
- structured classification;
- operational auditability.

### Retrieval

- runbook ingestion;
- chunking;
- embeddings;
- Qdrant collections;
- retrieval quality controls.

### Controlled orchestration

- LangGraph workflows;
- registered tools;
- approval boundaries;
- failure recovery.

### Observability and evaluation

- AI tracing;
- token and cost tracking;
- retrieval evaluation;
- generation evaluation;
- prompt regression testing.

## Intentionally deferred capabilities

The initial foundation intentionally excludes business and AI implementation.

The following capabilities are deferred to preserve architectural focus and avoid speculative abstractions:

- authentication and authorization;
- tenant security enforcement;
- worker polling and job claiming;
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

The architecture keeps room for these capabilities without introducing dependencies or abstractions before they have concrete responsibilities.

## Documentation

- [Architecture overview](docs/architecture/overview.md)
- [Runtime topology](docs/architecture/runtime-topology.md)
- [Architecture decision records](docs/decisions)

## License

No open-source license has been selected for this repository.
