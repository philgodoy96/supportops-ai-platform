# Local Setup

## Purpose

This guide describes how to prepare and run the SupportOps AI Platform locally.

The current platform includes:

- Python dependency management with `uv`;
- PostgreSQL and Qdrant through Docker Compose;
- a FastAPI application process;
- a separate `supportops-worker` process;
- an explicit `supportops-index-knowledge` indexing CLI;
- liveness and readiness endpoints;
- structured JSON logging with HTTP request traceability;
- Alembic migrations for workspace, ticket, AgentRun, classification, and
  knowledge-document tables;
- versioned workspace, ticket, AgentRun, and knowledge-document HTTP APIs;
- durable AgentRun scheduling and PostgreSQL worker execution;
- workspace-scoped AgentRun inspection;
- explicit profiled knowledge indexing;
- active-version semantic knowledge retrieval;
- unit and integration tests;
- local quality checks.

Workspace scoping is not authentication or authorization. Docker Compose
provisions infrastructure only and intentionally does not run the worker or
indexing CLI.

## Prerequisites

The local environment requires:

- Git;
- Python 3.12;
- `uv`;
- Docker with Docker Compose;
- PowerShell.

Verify the required tools:

```powershell
git --version
python --version
uv --version
docker --version
docker compose version
```

Python must resolve to a supported Python 3.12 installation.

## Clone the repository

```powershell
git clone <repository-url>
Set-Location supportops-ai-platform
```

Use the repository URL provided by the hosting platform.

## Install dependencies

Create the project environment and install dependencies from the committed lockfile:

```powershell
uv sync --frozen --all-groups
```

This command:

- creates `.venv` when necessary;
- installs the project package;
- installs runtime and development dependencies;
- refuses to modify the lockfile.

Validate the lockfile:

```powershell
uv lock --check
```

## Create local environment configuration

Copy the safe example configuration:

```powershell
Copy-Item .env.example .env
```

The `.env` file is ignored by Git and must not be committed.

The default values are intended only for local development.

Review the environment variable reference before changing configuration:

```text
docs/development/environment-variables.md
```

## Start local infrastructure

Start PostgreSQL and Qdrant:

```powershell
docker compose up -d
```

Inspect service state:

```powershell
docker compose ps
```

Both services should become healthy.

Validate PostgreSQL directly:

```powershell
docker compose exec postgresql `
  pg_isready `
  -U supportops `
  -d supportops
```

Expected result:

```text
/var/run/postgresql:5432 - accepting connections
```

Validate Qdrant directly:

```powershell
Invoke-WebRequest http://localhost:6333/healthz
```

Expected response body:

```text
healthz check passed
```

## Start the API

Apply the current migration head before exercising business routes:

```powershell
uv run alembic upgrade head
```

Run the FastAPI application in one terminal:

```powershell
uv run uvicorn supportops.api.main:app `
  --host 127.0.0.1 `
  --port 8000
```

The process emits structured JSON logs.

The application may start even when PostgreSQL or Qdrant is unavailable. Dependency availability is represented by readiness.

Business routes under `/api/v1` require a migrated PostgreSQL database. Ticket creation schedules a durable `AgentRun` transactionally and does not execute the workflow. Semantic search routes use Qdrant for candidate search and the process-scoped embedding provider for request-driven query embeddings. The API does not perform indexing and does not use the LLM provider.

## Start the worker

In a separate terminal, start the PostgreSQL worker:

```powershell
$env:SUPPORTOPS_WORKER_ID="worker-local-1"
uv run supportops-worker
```

Expected startup behavior:

- settings validate successfully;
- the process emits a structured `worker_started` log;
- the worker begins recovery, claim, and processing cycles against PostgreSQL;
- the worker does not initialize or connect to Qdrant.

Docker Compose intentionally does not run the worker. Keep the API terminal and the worker terminal running while exercising ticket intake.

If `SUPPORTOPS_WORKER_ID` is omitted, the worker generates an identity from hostname, process ID, and a UUID suffix.

## Validate liveness

In another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
```

Expected result:

```text
status
------
healthy
```

Liveness verifies only that the application process can respond.

It does not call PostgreSQL or Qdrant.

## Validate readiness

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

Expected response:

```text
status      dependencies
------      ------------
healthy     ...
```

The serialized JSON response contains separate states for:

- PostgreSQL;
- Qdrant.

When either dependency is unavailable, readiness returns HTTP `503 Service Unavailable`.

## Validate workspace, ticket, and AgentRun APIs

Use three terminals:

1. API process;
2. worker process;
3. client PowerShell session for requests.

With infrastructure healthy, migrations applied, and the API running, create a
workspace:

```powershell
$workspace = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces" `
  -ContentType "application/json" `
  -Body (@{
    name = "Platform Support"
    slug = "platform-support"
  } | ConvertTo-Json)

$workspace
```

Create a ticket in that workspace:

```powershell
$ticketResponse = Invoke-WebRequest `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/$($workspace.id)/tickets" `
  -ContentType "application/json" `
  -Body (@{
    subject = "Unable to access the billing dashboard"
    description = "The dashboard returns an access error after sign-in."
  } | ConvertTo-Json)

$ticket = $ticketResponse.Content | ConvertFrom-Json
$ticket
$ticket.processing_run
$ticketResponse.Headers["X-Request-ID"]
$ticketResponse.Headers["X-Correlation-ID"]

$agentRunId = $ticket.processing_run.id
```

Expected behavior:

- workspace creation returns HTTP `201`;
- ticket creation returns HTTP `201`;
- ticket status is `open`;
- `processing_run.status` is `queued`;
- `processing_run.workflow_name` is `ticket-processing`;
- `processing_run.workflow_version` is `deterministic-baseline-v1`;
- `ingestion_request_id` matches `X-Request-ID`;
- `correlation_id` matches `X-Correlation-ID`;
- the response confirms ticket acceptance, not final processing success.

### Inspect the AgentRun while queued

If the worker has not yet claimed the run, inspect the queued state:

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/$($workspace.id)/agent-runs/$agentRunId"
```

Expected queued behavior:

- status is `queued`;
- `attempt_count` is `0`;
- `first_started_at` and `completed_at` are null;
- `last_error` is null.

Inspect attempt history for the same run:

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/$($workspace.id)/agent-runs/$agentRunId/attempts"
```

Expected queued attempt history:

```json
{
  "items": []
}
```

### Observe worker processing and succeeded state

With the worker running, observe structured worker logs such as
`worker_cycle_completed` for claimed and processed runs. The deterministic
baseline validates the workflow contract and does not perform AI
classification.

After processing completes, inspect the AgentRun again:

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/$($workspace.id)/agent-runs/$agentRunId"
```

Expected succeeded behavior for the deterministic baseline:

- status is `succeeded`;
- `attempt_count` is `1`;
- `first_started_at` and `completed_at` are populated;
- `last_error` is null.

Inspect ordered attempt history:

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/$($workspace.id)/agent-runs/$agentRunId/attempts"
```

Expected succeeded attempt history:

- `items` contains one attempt;
- `attempt_number` is `1`;
- `outcome` is `succeeded`;
- lease tokens and execution request IDs are not present.

Inspection reports current persisted state. It does not guarantee future
completion while a run is still in progress.

### Stop the worker

In the worker terminal, stop the process with `Ctrl+C`.

Expected graceful shutdown behavior:

- a structured `worker_shutdown_requested` log;
- the active cycle may finish within the configured shutdown grace period;
- a structured `worker_stopped` log after engine disposal.

If the grace period is exceeded, the process emits
`worker_shutdown_grace_exceeded` before cancelling the active loop task.

Complete request examples, conflict cases, cross-workspace isolation,
AgentRun inspection responses, and cursor pagination are documented in:

```text
docs/development/api-examples.md
```

## Index knowledge locally

Knowledge-document registration through the HTTP API persists source content
only. Indexing is a separate CLI operation.

With PostgreSQL and Qdrant healthy and migrations applied:

```powershell
docker compose up -d postgresql qdrant
uv run alembic upgrade head
uv run supportops-index-knowledge ensure-collection
uv run supportops-index-knowledge index-version `
  --workspace-id "<workspace-id>" `
  --document-id "<document-id>" `
  --document-version-id "<document-version-id>"
```

Use the UUIDs returned by the knowledge-document API. Do not execute the literal
placeholders.

Expected default mock behavior:

- ensure-collection and index-version are network-free with the mock embedding
  provider;
- successful indexing returns status `ready`;
- indexing does not activate the version;
- activate the ready version separately through the HTTP API;
- a ready-version rerun is a no-op.

Optional OpenAI opt-in using process environment variables:

```powershell
$env:SUPPORTOPS_EMBEDDING_PROVIDER="openai"
$env:SUPPORTOPS_EMBEDDING_MODEL="text-embedding-3-small"
$env:SUPPORTOPS_EMBEDDING_DIMENSIONS="1536"
$env:SUPPORTOPS_OPENAI_API_KEY="<temporary-secret>"

uv run supportops-index-knowledge ensure-collection --allow-external-provider

uv run supportops-index-knowledge index-version `
  --workspace-id "<workspace-id>" `
  --document-id "<document-id>" `
  --document-version-id "<document-version-id>" `
  --allow-external-provider

Remove-Item Env:SUPPORTOPS_OPENAI_API_KEY
```

Do not place a real API key in documentation or committed files. Remove the
temporary process key after the command.

## Search active knowledge locally

After indexing and activating a ready version, search through the API.

Start the API after setting the embedding profile variables. Indexing and API
retrieval must use matching profiles. The default mock profile is network-free.

```powershell
$searchResponse = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/<workspace-id>/knowledge/search" `
  -ContentType "application/json" `
  -Body (@{
    query = "How do I recover the database?"
    top_k = 5
  } | ConvertTo-Json)

$searchResponse.searched_version_count
$searchResponse.evidence
```

Use the workspace UUID returned by the knowledge-document API. Activate a ready
version before search. An empty active scope returns HTTP `200` with
`searched_version_count` `0` and empty evidence. No LLM answer is returned.

For OpenAI embeddings, start the API process with the OpenAI embedding
environment variables and key. The search endpoint itself does not accept
`--allow-external-provider`; that CLI flag applies only to explicit indexing
commands. Do not place a real API key in documentation or committed files.

Complete request examples are documented in
[`api-examples.md`](api-examples.md).

## Validate dependency failure behavior

Stop PostgreSQL:

```powershell
docker compose stop postgresql
```

Call readiness:

```powershell
try {
    Invoke-RestMethod http://127.0.0.1:8000/health/ready
} catch {
    $_.Exception.Response.StatusCode.value__
    $_.ErrorDetails.Message
}
```

Expected behavior:

- HTTP status is `503`;
- PostgreSQL is reported as unhealthy;
- Qdrant may remain healthy;
- no credential or connection string is returned.

Liveness must continue to succeed:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
```

Restart PostgreSQL:

```powershell
docker compose start postgresql
docker compose ps
```

Repeat the same validation for Qdrant:

```powershell
docker compose stop qdrant

try {
    Invoke-RestMethod http://127.0.0.1:8000/health/ready
} catch {
    $_.Exception.Response.StatusCode.value__
    $_.ErrorDetails.Message
}

Invoke-RestMethod http://127.0.0.1:8000/health/live

docker compose start qdrant
docker compose ps
```

## Run Alembic commands

Migration lifecycle commands:

```powershell
uv run alembic upgrade head
uv run alembic downgrade base
uv run alembic upgrade head
uv run alembic check
uv run alembic current
uv run alembic heads
```

The current head creates the `workspaces`, `tickets`, `agent_runs`, and
`agent_run_attempts` tables.

`alembic downgrade` must only run against the local development or test database. Do not run downgrades against shared or production databases.

Validate offline migration execution:

```powershell
uv run alembic upgrade head --sql
```

## Inspect the database schema

```powershell
docker compose exec postgresql `
  psql `
  -U supportops `
  -d supportops `
  -c "\dt"
```

After `alembic upgrade head`, the schema includes `workspaces`, `tickets`,
`agent_runs`, and `agent_run_attempts`.

## Run quality checks

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

## Run tests

Run unit tests without infrastructure:

```powershell
uv run pytest -m "not integration"
```

Run integration tests with PostgreSQL and Qdrant healthy:

```powershell
uv run pytest -m integration
```

Run the complete test suite:

```powershell
uv run pytest
```

The testing strategy is documented in:

```text
docs/development/testing.md
```

## Build the application image

```powershell
docker build `
  --tag supportops-ai-platform:local `
  .
```

The image installs dependencies from the lockfile and runs the API as a non-root user.

Run the image against infrastructure exposed by the host:

```powershell
docker run --rm `
  --env-file .env `
  -e SUPPORTOPS_API_HOST=0.0.0.0 `
  -e SUPPORTOPS_POSTGRESQL_URL=postgresql+asyncpg://supportops:supportops-local@host.docker.internal:5432/supportops `
  -e SUPPORTOPS_QDRANT_URL=http://host.docker.internal:6333 `
  -p 8001:8000 `
  supportops-ai-platform:local
```

Validate the containerized application:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health/live
Invoke-RestMethod http://127.0.0.1:8001/health/ready
```

Stop the foreground container with `Ctrl+C`.

## Full local validation

Run the complete repository validation sequence:

```powershell
uv sync --frozen --all-groups
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -m "not integration"
uv run alembic upgrade head
uv run alembic heads
uv run alembic current
uv run alembic check
uv run pytest -m integration
docker compose config --quiet
docker build --tag supportops-ai-platform:local .
git diff --check
```

## Stop local services

Stop containers without deleting local data:

```powershell
docker compose stop
```

Remove containers and the local network while preserving named volumes:

```powershell
docker compose down
```

Remove containers and local named volumes only when a full local reset is intentional:

```powershell
docker compose down -v
```

The `-v` option permanently deletes local PostgreSQL and Qdrant data.

## Troubleshooting

### The application reports invalid configuration

Confirm that `.env` exists and contains:

```text
SUPPORTOPS_POSTGRESQL_URL
SUPPORTOPS_QDRANT_URL
```

Validate the file against `.env.example`.

### Readiness returns HTTP 503

Inspect service state:

```powershell
docker compose ps
```

Then validate each dependency directly.

A healthy application process can remain live while readiness reports unavailable infrastructure.

### PostgreSQL port is already in use

Change `POSTGRES_PORT` in `.env`, then update `SUPPORTOPS_POSTGRESQL_URL` to use the same host port.

### Qdrant port is already in use

Change `QDRANT_HTTP_PORT` and `QDRANT_GRPC_PORT` in `.env`.

Update `SUPPORTOPS_QDRANT_URL` to use the configured HTTP port.

### The lockfile is out of date

Run:

```powershell
uv lock --check
```

Do not update the lockfile during routine installation. Dependency changes must be intentional and reviewed.

### Tests report duplicate module names

The project uses pytest importlib mode to support identical test basenames in separate unit and integration directories.

Do not remove the configured `--import-mode=importlib` option without restructuring the test packages.