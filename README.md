# SupportOps AI Platform

SupportOps AI Platform is a production-minded backend and AI systems engineering project focused on reliable support operations, controlled AI orchestration, retrieval quality, human approval, observability, and evaluation.

The platform is designed as a portfolio-grade engineering system rather than a tutorial chatbot. Its architecture emphasizes clear boundaries, operational reliability, explicit trade-offs, testability, and incremental delivery.

## Project status

The repository foundation, Slice 1 workspace and ticket API, durable AgentRun scheduling, the PostgreSQL-backed worker, workspace-scoped AgentRun inspection, the application-owned LLM Gateway, durable structured ticket classification, durable logical invocation and accepted classification persistence, workspace-scoped classification and logical invocation inspection, and offline deterministic classification evaluation are implemented.

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
- reversible Alembic migrations for workspace, ticket, AgentRun, invocation, and classification tables;
- an application-owned transaction adapter;
- workspace creation and retrieval API;
- workspace-scoped ticket intake;
- atomic Ticket and initial AgentRun persistence;
- configured `ticket-classification-v1` scheduling for newly accepted tickets;
- durable AgentRun and AgentRunAttempt persistence;
- PostgreSQL claiming with `FOR UPDATE SKIP LOCKED`;
- attempt history, leases, and lease-token fencing;
- bounded retries and expired lease recovery;
- exact versioned workflow executor registry;
- deterministic baseline and classification workflow execution outside database transactions;
- process-scoped mock or OpenAI worker provider;
- Structured Outputs classification through the LLM Gateway;
- durable `LLMInvocation` history;
- durable `TicketClassification` records;
- lease-fenced classification persistence;
- token usage and estimated-cost provenance;
- idempotent recovery without another provider call after classification commit;
- cooperative worker shutdown with structured operational logs;
- database-enforced workspace and ticket ownership for AgentRun records;
- duplicate initial scheduling prevention;
- workspace-scoped AgentRun inspection;
- workspace-scoped AgentRunAttempt history inspection;
- workspace-scoped classification detail and ticket classification history inspection;
- optional minimal accepted-classification reference on AgentRun detail;
- AgentRun-scoped logical invocation inspection;
- safe operational AgentRun, classification, and invocation metadata without lease or execution fencing identifiers;
- tenant-safe `404` behavior for missing and cross-workspace AgentRuns, classifications, and invocation histories;
- workspace-scoped ticket retrieval and listing;
- versioned `/api/v1` business routes;
- stable expected-error responses;
- opaque cursor pagination;
- request and correlation identifier persistence;
- cross-workspace isolation behavior;
- application services with command and query use cases;
- application-owned provider-independent LLM Gateway contracts;
- a deterministic mock LLM provider;
- an OpenAI Responses API provider;
- prompt `ticket-classification` version 1 and deterministic prompt hashes;
- normalized provider failures and bounded repair;
- token usage mapping and versioned estimated-cost calculation;
- validated LLM runtime settings;
- a versioned synthetic classification dataset;
- a deterministic classification evaluator;
- offline scoring of prediction artifacts;
- an opt-in external-provider evaluation CLI;
- canonical dataset, prediction, and report provenance;
- classification inspection integration coverage;
- repository, application, worker, AI, evaluation, and API tests;
- Ruff, mypy, and pytest quality gates;
- a reproducible application Docker image;
- GitHub Actions continuous integration;
- professional architecture and development documentation.

Workspace scoping is a data ownership boundary. It is not authentication or authorization, and it is not authenticated tenant isolation.

Ticket acceptance and asynchronous processing success are separate outcomes. Ticket intake schedules the configured workflow version, with local default `ticket-classification-v1`. Ticket status remains `open`. AgentRun status reports workflow execution. An accepted `TicketClassification` records the model interpretation and does not mutate Ticket status or execute tools. The deterministic baseline remains registered for historical or explicitly scheduled runs and performs no LLM call.

Inspection endpoints report current persisted AgentRun, classification, and logical invocation state. They do not guarantee future completion, and they do not mutate retries, leases, or lifecycle transitions. Inspection is read-only. Evaluation measures the same prompt and schema boundary offline and does not write to PostgreSQL or Qdrant.

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
- OpenAI Python SDK;
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
- registered workspace, ticket, AgentRun, invocation, and classification persistence models;
- reversible migrations that create `workspaces`, `tickets`, `agent_runs`, `agent_run_attempts`, `llm_invocations`, and `ticket_classifications`;
- composite ownership constraints and accepted-invocation provenance for classification records.

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

Ticket intake schedules the configured workflow version in the same application-owned transaction that creates the ticket. The local default is `ticket-classification-v1`. The HTTP request returns the existing Ticket response after that transaction commits and does not execute the workflow or call the model.

Implemented behavior includes:

- frozen AgentRun and AgentRunAttempt domain entities with validated invariants;
- PostgreSQL `agent_runs` and `agent_run_attempts` tables with query-driven indexes;
- composite workspace and ticket ownership enforcement;
- unique initial trigger enforcement for duplicate initial scheduling prevention;
- atomic Ticket and initial AgentRun creation;
- a persisted initial retry budget copied from configuration;
- a separate `supportops-worker` process using PostgreSQL as its durable work queue;
- claim eligibility for `queued` and `retry_scheduled` runs with due `available_at`;
- PostgreSQL `FOR UPDATE SKIP LOCKED` claiming across multiple worker processes;
- attempt history, leases, and lease-token fencing;
- bounded exponential backoff retries;
- expired lease recovery before each claim cycle;
- a versioned executor registry with exact workflow name and version dispatch;
- deterministic baseline support for historical or explicitly scheduled runs;
- classification provider calls outside database transactions;
- separate fenced transactions for invocation and classification persistence and AgentRun completion;
- cooperative SIGINT and SIGTERM shutdown with provider cleanup and engine disposal.

The ticket creation response remains the Ticket response shape and does not report classification completion.

After ticket acceptance, the client can inspect the persisted AgentRun and its attempt history through workspace-scoped read-only endpoints when the AgentRun identifier is otherwise known. Inspection exposes status, retry budget, workflow identity, safe error metadata, and ordered attempt outcomes. It does not expose lease ownership, lease tokens, lease expiry, ingestion request IDs, or execution request IDs.

Scheduling and worker handoff behavior are documented in [`docs/architecture/agent-run-scheduling.md`](docs/architecture/agent-run-scheduling.md) and [`docs/architecture/runtime-topology.md`](docs/architecture/runtime-topology.md).

### Application-owned LLM Gateway and durable classification

The repository provides an application-owned LLM Gateway under `supportops.ai` and durable classification under `supportops.modules.ticket_classifications`:

- process-scoped mock or OpenAI provider and one Gateway per worker process;
- session-scoped classification executor and repositories;
- provider-independent asynchronous LLM contracts;
- OpenAI Responses API with Structured Outputs;
- application-side Pydantic validation of structured classification results;
- immutable prompt definitions with explicit ID, version, and SHA-256 content hashes;
- prompt `ticket-classification` version 1 with trusted instructions separated from untrusted ticket content;
- an application-owned provider failure taxonomy;
- bounded validation repair with at most one repair invocation;
- durable `LLMInvocation` and `TicketClassification` provenance;
- token usage mapping and versioned Decimal estimated-cost persistence;
- retryable and terminal Gateway failure translation;
- explicit provider selection with no cross-provider fallback;
- no LLM provider initialization in the API process;
- workspace-scoped classification detail and ticket classification history;
- optional minimal accepted-classification reference on AgentRun detail;
- AgentRun-scoped logical invocation history;
- offline deterministic evaluation against a versioned synthetic dataset.

Current classification flow:

```text
configured AgentRun scheduled
→ worker classifies and persists provenance
→ client inspects classification and logical invocation state
→ AI engineer evaluates the same prompt/schema boundary offline
```

Implemented inspection routes:

```text
GET /api/v1/workspaces/{workspace_id}/ticket-classifications/{classification_id}
GET /api/v1/workspaces/{workspace_id}/tickets/{ticket_id}/classifications
GET /api/v1/workspaces/{workspace_id}/agent-runs/{agent_run_id}/llm-invocations
```

AgentRun detail includes an optional minimal classification reference.

Gateway architecture is documented in [`docs/architecture/llm-gateway.md`](docs/architecture/llm-gateway.md). Durable classification behavior is documented in [`docs/architecture/ticket-classification.md`](docs/architecture/ticket-classification.md). Classification inspection and evaluation are documented in [`docs/architecture/classification-evaluation.md`](docs/architecture/classification-evaluation.md).

### Workspace, ticket, and AgentRun API

Slice 1 exposes versioned business routes under `/api/v1`:

- workspace creation and retrieval;
- workspace-scoped ticket creation, retrieval, and listing;
- workspace-scoped AgentRun retrieval;
- workspace-scoped AgentRunAttempt history listing;
- workspace-scoped classification detail and ticket classification history;
- workspace-scoped AgentRun logical invocation history;
- opaque cursor pagination for ticket listing and classification history;
- stable expected-error responses for missing resources, conflicts, and invalid cursors;
- persistence of request and correlation identifiers on accepted tickets;
- cross-workspace retrieval that returns the same `404` contract as a missing ticket, AgentRun, or classification.

Current operational flow:

```text
ticket accepted
→ AgentRun scheduled
→ worker processes
→ client inspects persisted state
```

Health routes remain unversioned. Workspace scoping is not authentication or authorization. Inspection endpoints are strictly read-only.

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
- workspace, ticket, and AgentRun API schema and pagination unit coverage;
- ORM mapping, named-constraint, and model-registration tests;
- repository integration and concurrency-sensitive tests, including SKIP LOCKED claiming;
- workspace-scoped AgentRun query repository coverage;
- workspace, ticket, and AgentRun API integration coverage;
- atomic ticket and AgentRun commit and rollback coverage;
- provider-independent LLM contract tests;
- prompt registry, prompt hash, and untrusted-input boundary tests;
- structured classification schema tests;
- deterministic mock-provider tests;
- OpenAI provider tests using injected fakes without network access;
- provider error normalization tests;
- gateway validation and bounded repair tests;
- Decimal pricing and unknown-pricing tests;
- LLM settings and secret-handling tests;
- classification domain and ORM tests;
- classification inspection projection and API tests;
- registry dispatch tests;
- worker composition and lifecycle tests;
- fenced classification repository tests;
- PostgreSQL mock classification workflow integration;
- retry and recovery idempotency coverage;
- classification inspection integration coverage;
- evaluation dataset, metrics, predictor, runner, and CLI unit coverage;
- Alembic upgrade, downgrade, and metadata-parity coverage for classification tables;
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

Normal unit and integration tests do not require an OpenAI API key or paid external requests. OpenAI evaluation remains an explicit manual operation.

## Planned platform modules

The repository already includes bounded `workspaces`, `tickets`, `agent_runs`, and `ticket_classifications` modules, a `supportops.worker` process entry point, the cross-cutting `supportops.ai` foundation, and the offline `supportops.evaluation.ticket_classification` package. Workspace and ticket modules expose domain entities, application services, repository contracts, PostgreSQL persistence, and versioned HTTP APIs. The `agent_runs` module provides durable scheduling, claiming, versioned executor dispatch, execution, retry, recovery, and workspace-scoped inspection foundations. The `supportops.modules.ticket_classifications` module is implemented for durable classification execution, persistence, and read-only inspection. The `supportops.ai` package owns provider-independent LLM contracts, provider adapters, prompt definitions, structured schemas, repair behavior, and estimated-cost calculation.

Future modules or extensions will introduce:

- evidence-driven prompt version 2;
- prompt regression comparison across versions;
- scheduled evaluation and evaluation history persistence;
- internal runbook ingestion;
- semantic retrieval and Qdrant indexing;
- LangGraph orchestration;
- registered tools;
- approval workflows;
- broader AI observability.

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
│   │   ├── classification-evaluation.md
│   │   ├── llm-gateway.md
│   │   ├── overview.md
│   │   ├── runtime-topology.md
│   │   ├── ticket-classification.md
│   │   └── workspace-data-boundary.md
│   ├── decisions/
│   │   ├── 0001-use-a-modular-monolith.md
│   │   ├── 0002-use-postgresql-as-the-source-of-truth.md
│   │   ├── 0003-use-qdrant-as-a-rebuildable-retrieval-index.md
│   │   ├── 0004-use-a-postgresql-backed-worker-model.md
│   │   ├── 0005-keep-ai-observability-behind-an-adapter.md
│   │   ├── 0006-establish-workspace-scoped-data-ownership.md
│   │   └── 0007-use-an-application-owned-llm-gateway.md
│   └── development/
│       ├── api-examples.md
│       ├── environment-variables.md
│       ├── local-setup.md
│       └── testing.md
├── evals/
│   └── ticket-classification/
│       ├── README.md
│       └── datasets/
│           └── ticket-classification-eval-v1.jsonl
├── src/
│   └── supportops/
│       ├── ai/
│       │   ├── gateway/
│       │   ├── pricing/
│       │   ├── prompts/
│       │   ├── providers/
│       │   └── schemas/
│       ├── api/
│       │   ├── health/
│       │   ├── application.py
│       │   ├── lifespan.py
│       │   ├── main.py
│       │   ├── router.py
│       │   └── state.py
│       ├── application/
│       │   ├── agent_run_inspection.py
│       │   └── ticket_intake.py
│       ├── core/
│       │   ├── logging.py
│       │   ├── request_context.py
│       │   ├── settings.py
│       │   └── transactions.py
│       ├── evaluation/
│       │   └── ticket_classification/
│       ├── infrastructure/
│       │   ├── postgresql/
│       │   └── qdrant/
│       ├── modules/
│       │   ├── agent_runs/
│       │   │   ├── api/
│       │   │   ├── application/
│       │   │   ├── domain/
│       │   │   └── infrastructure/
│       │   ├── ticket_classifications/
│       │   │   ├── api/
│       │   │   ├── application/
│       │   │   ├── domain/
│       │   │   └── infrastructure/
│       │   ├── tickets/
│       │   └── workspaces/
│       └── worker/
│           ├── __init__.py
│           ├── composition.py
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

Business modules are introduced when they have concrete responsibilities. The current `workspaces`, `tickets`, `agent_runs`, and `ticket_classifications` modules include domain, application, and infrastructure layers as required. The `agent_runs` and `ticket_classifications` modules include read-only inspection routes. Cross-module ticket intake and AgentRun inspection composition live under `supportops.application`. The worker process entry point and process-scoped LLM composition live under `supportops.worker`. The `supportops.ai` package owns provider-independent contracts, provider adapters, prompt definitions, structured schemas, repair behavior, and estimated-cost calculation. It is not a generic orchestration framework. The `supportops.evaluation.ticket_classification` package owns offline datasets, prediction artifacts, deterministic metrics, and the evaluation CLI. Alembic migrations create workspace, ticket, AgentRun, invocation, and classification tables. Versioned evaluation datasets remain committed under `evals/`. Generated evaluation outputs belong under ignored `artifacts/`.

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

## Classification evaluation

Offline classification evaluation uses the versioned synthetic dataset and the
`supportops-evaluate-classification` CLI. Evaluation accesses neither PostgreSQL
nor Qdrant. Generated outputs belong under ignored `artifacts/`. Versioned
datasets remain committed under `evals/`.

Mock pipeline:

```powershell
uv run supportops-evaluate-classification run `
  --provider mock `
  --dataset `
    evals/ticket-classification/datasets/ticket-classification-eval-v1.jsonl `
  --predictions-output `
    artifacts/classification-mock-predictions.jsonl `
  --output `
    artifacts/classification-mock-report.json
```

Mock evaluation validates pipeline wiring. It is not a model-quality benchmark.

Offline scoring:

```powershell
uv run supportops-evaluate-classification score `
  --dataset `
    evals/ticket-classification/datasets/ticket-classification-eval-v1.jsonl `
  --predictions `
    artifacts/classification-mock-predictions.jsonl `
  --output `
    artifacts/classification-mock-rescored-report.json
```

`score` initializes no provider and makes no network request.

OpenAI evaluation:

```powershell
uv run supportops-evaluate-classification run `
  --provider openai `
  --allow-external-provider `
  --dataset `
    evals/ticket-classification/datasets/ticket-classification-eval-v1.jsonl `
  --predictions-output `
    artifacts/classification-openai-predictions.jsonl `
  --output `
    artifacts/classification-openai-report.json
```

OpenAI evaluation requires `--allow-external-provider` and a configured API key.
Evaluation results do not alter production prompt selection automatically.

Classification inspection and evaluation architecture is documented in
[`docs/architecture/classification-evaluation.md`](docs/architecture/classification-evaluation.md).

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

Worker timing and identity are controlled by `SUPPORTOPS_WORKER_*` variables. Defaults are validated at process startup, including lease-versus-timeout and retry-base-versus-max invariants. The worker always composes the versioned executor registry. Executor selection is not a deployment switch.

AI runtime selection uses:

```text
SUPPORTOPS_LLM_PROVIDER
SUPPORTOPS_OPENAI_API_KEY
SUPPORTOPS_OPENAI_MODEL
SUPPORTOPS_OPENAI_BASE_URL
SUPPORTOPS_LLM_REQUEST_TIMEOUT_SECONDS
SUPPORTOPS_LLM_TRANSPORT_MAX_RETRIES
SUPPORTOPS_LLM_MAX_REPAIR_ATTEMPTS
SUPPORTOPS_TICKET_PROCESSING_WORKFLOW_VERSION
```

The local default provider is `mock`. An OpenAI API key is required only when `openai` is selected. `SUPPORTOPS_TICKET_PROCESSING_WORKFLOW_VERSION` controls newly scheduled runs; the local default is `ticket-classification-v1`. Request timeout, logical repair budget, worker timeout, and lease margins are validated at startup. Provider transport retry, gateway repair, and AgentRun retry are separate layers.

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

The current head creates the `workspaces`, `tickets`, `agent_runs`, `agent_run_attempts`, `llm_invocations`, and `ticket_classifications` tables. Downgrade commands must run only against the local development or test database.

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
- [Use an application-owned LLM Gateway](docs/decisions/0007-use-an-application-owned-llm-gateway.md)

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
- atomic ticket intake with initial AgentRun scheduling;
- durable structured classification;
- durable invocation and accepted classification persistence;
- workspace-scoped classification detail and history inspection;
- AgentRun classification reference and logical invocation inspection.

Planned:

- operational auditability beyond request and correlation identifiers.

### Asynchronous processing

Implemented:

- AgentRun and AgentRunAttempt domain and persistence foundations;
- atomic Ticket and initial AgentRun scheduling;
- database-enforced workspace and ticket ownership;
- duplicate initial scheduling prevention;
- persisted initial retry budget;
- PostgreSQL-backed worker execution;
- queue claiming with `FOR UPDATE SKIP LOCKED`;
- leases and lease-token fencing;
- bounded exponential backoff retries;
- expired lease recovery;
- versioned workflow executor registry;
- deterministic baseline executor;
- configured `ticket-classification-v1` scheduling and execution;
- process-scoped provider and Gateway composition;
- separate worker process with cooperative shutdown;
- workspace-scoped AgentRun inspection;
- workspace-scoped AgentRunAttempt history inspection;
- safe operational metadata projections;
- tenant-safe `agent_run_not_found` responses.

Planned:

- manual retry and cancellation;
- global AgentRun listing and status filtering;
- idempotent side effects for future executors and tools.

### LLM Gateway and structured classification

Implemented:

- provider-independent async contracts;
- deterministic mock provider;
- OpenAI Responses API provider;
- Structured Outputs;
- application-side Pydantic validation;
- bounded classification taxonomy;
- prompt `ticket-classification` version 1;
- deterministic prompt hashes;
- normalized provider failures;
- bounded repair;
- token usage mapping and persistence;
- versioned Decimal pricing catalog;
- estimated-cost calculation and persistence;
- validated AI runtime settings;
- worker provider composition;
- workflow executor registry;
- durable ticket-classification execution;
- `TicketClassification` persistence;
- `LLMInvocation` persistence;
- classification inspection API;
- synthetic classification dataset;
- deterministic classification evaluator;
- opt-in real-model evaluation.

Planned:

- evidence-driven prompt version 2;
- prompt regression comparison across versions;
- cross-provider fallback after baseline behavior is observable;
- operational cost reporting and invoice reconciliation.

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

Implemented:

- token usage and estimated-cost persistence with durable invocation provenance;
- prompt `ticket-classification` version 1;
- versioned synthetic classification dataset;
- deterministic classification evaluator;
- offline scoring;
- opt-in external-provider evaluation CLI;
- canonical dataset, prediction, and report provenance.

Planned:

- operational cost reporting and invoice reconciliation;
- AI tracing;
- evidence-driven prompt version 2;
- prompt regression comparison across versions;
- scheduled evaluation;
- evaluation history persistence;
- retrieval evaluation;
- generation evaluation beyond structured classification;
- RAGAS.

## Intentionally deferred capabilities

The following capabilities remain deferred to preserve architectural focus and avoid speculative abstractions:

- authentication and authorization;
- authenticated tenant isolation;
- manual AgentRun retry and cancellation;
- lease revocation and worker administration;
- global AgentRun listing, status filtering, and pagination across runs;
- WebSockets and Server-Sent Events;
- frontend monitoring applications;
- Redis, Celery, Kafka, and SQS;
- evidence-driven prompt version 2;
- prompt regression comparison across versions;
- scheduled evaluation;
- evaluation history persistence;
- cross-provider fallback and automatic model routing;
- Anthropic provider;
- operational cost reporting and invoice reconciliation;
- embeddings and retrieval;
- Qdrant collections and indexing;
- LangGraph orchestration;
- registered tools;
- human approval workflows;
- AI observability integrations;
- Langfuse integration;
- RAGAS evaluation;
- retrieval and generation evaluation beyond structured classification;
- OpenTelemetry;
- Prometheus and Grafana;
- frontend applications;
- cloud deployment;
- infrastructure as code;
- Kubernetes.

Workspace scoping establishes data ownership. It is not authentication or authorization, and it does not establish caller identity or secure multi-tenancy.

Durable AgentRun scheduling and the PostgreSQL worker are implemented. Redis, Celery, Kafka, and SQS remain intentionally deferred because PostgreSQL already provides transactional durability and adequate local and portfolio scope for this phase. An external queue or outbox is not required for the current worker model.

The application-owned LLM Gateway, durable ticket-classification workflow, classification inspection, and offline evaluation are implemented. Evidence-driven prompt version 2, prompt regression comparison, scheduled evaluation, evaluation history persistence, cross-provider fallback, operational cost reporting, RAGAS, and retrieval evaluation remain intentionally separated into later delivery boundaries.

The architecture keeps room for these capabilities without introducing dependencies or abstractions before they have concrete responsibilities.

## Documentation

- [Architecture overview](docs/architecture/overview.md)
- [Runtime topology](docs/architecture/runtime-topology.md)
- [Transactional AgentRun scheduling](docs/architecture/agent-run-scheduling.md)
- [Application-owned LLM Gateway](docs/architecture/llm-gateway.md)
- [Durable ticket classification](docs/architecture/ticket-classification.md)
- [Classification inspection and evaluation](docs/architecture/classification-evaluation.md)
- [Workspace-scoped data ownership](docs/architecture/workspace-data-boundary.md)
- [Architecture decision records](docs/decisions)
- [Local setup](docs/development/local-setup.md)
- [API examples](docs/development/api-examples.md)
- [Environment variables](docs/development/environment-variables.md)
- [Testing strategy](docs/development/testing.md)

## License

No open-source license has been selected for this repository.
