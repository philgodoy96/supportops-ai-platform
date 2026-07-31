# Local Setup

## Purpose

This guide describes how to prepare and run the SupportOps AI Platform locally.

The current platform includes:

- Python dependency management with `uv`;
- PostgreSQL and Qdrant through Docker Compose;
- a FastAPI application process;
- liveness and readiness endpoints;
- structured JSON logging with HTTP request traceability;
- Alembic migrations for workspace and ticket tables;
- unit and integration tests;
- local quality checks.

Workspace and ticket HTTP endpoints, asynchronous processing, and AI capabilities are intentionally outside the current implementation scope.

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

Run the FastAPI application:

```powershell
uv run uvicorn supportops.api.main:app `
  --host 127.0.0.1 `
  --port 8000
```

The process emits structured JSON logs.

The application may start even when PostgreSQL or Qdrant is unavailable. Dependency availability is represented by readiness.

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

The current head creates the `workspaces` and `tickets` tables.

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

After `alembic upgrade head`, the schema includes `workspaces` and `tickets`.

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