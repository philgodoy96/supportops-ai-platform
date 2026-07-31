# Testing Strategy

## Purpose

The SupportOps AI Platform test suite verifies foundation behavior across configuration, application composition, infrastructure connectivity, lifecycle management, health semantics, HTTP request traceability, workspace and ticket persistence, migration tooling, and container packaging.

The strategy separates tests by dependency boundary:

- unit tests do not require Docker or network services;
- integration tests use real PostgreSQL and Qdrant services;
- manual validation covers destructive dependency outage scenarios;
- continuous integration reproduces the local quality and integration gates.

The suite is intended to verify externally observable behavior and architectural guarantees rather than mirror implementation lines.

## Test categories

## Unit tests

Unit tests run without PostgreSQL, Qdrant, Docker, or network access.

They validate:

- root package import and version;
- settings defaults and validation;
- invalid configuration behavior;
- structured JSON logging;
- request and correlation context primitives;
- contextvars cleanup and async-task isolation;
- structured logging enrichment;
- response trace headers;
- correlation propagation and invalid-value fallback;
- incoming request-ID rejection;
- downstream trace-header spoofing prevention;
- request completion logging;
- unexpected exception behavior;
- application construction;
- application lifecycle ownership;
- PostgreSQL engine and session factories;
- Qdrant client factory and cleanup;
- bounded dependency health checks;
- liveness behavior;
- readiness aggregation;
- readiness failure responses;
- response sanitization;
- workspace and ticket domain invariants;
- ORM mapping and metadata;
- named PostgreSQL constraints declared on persistence records;
- persistence model registration;
- PostgreSQL constraint-name inspection helpers.

Unit tests use mocks only at external boundaries.

They must not:

- open real network connections;
- require `.env`;
- depend on local containers;
- mutate Docker services;
- create Qdrant collections.

Run unit tests:

```powershell
uv run pytest -m "not integration"
```

Unit tests can also be run directly:

```powershell
uv run pytest tests/unit
```

Targeted domain and persistence unit coverage:

```powershell
uv run pytest tests/unit/modules/workspaces/domain tests/unit/modules/tickets/domain
uv run pytest tests/unit/modules/workspaces/infrastructure tests/unit/modules/tickets/infrastructure
uv run pytest tests/unit/infrastructure/postgresql
```
## Integration tests

Integration tests require live PostgreSQL and Qdrant services.

They validate:

- real PostgreSQL connectivity;
- real Qdrant connectivity;
- FastAPI lifecycle against live dependencies;
- liveness with the live application;
- readiness with healthy dependencies;
- Alembic import and configuration;
- Alembic connectivity to PostgreSQL;
- migration upgrade, downgrade, and re-upgrade;
- creation of `workspaces` and `tickets` tables;
- workspace persistence;
- duplicate workspace slug translation;
- transaction rollback;
- ticket foreign-key behavior;
- the same external reference across workspaces;
- duplicate external-reference rejection inside one workspace;
- cross-workspace ticket lookup behavior;
- deterministic ticket listing;
- keyset repository navigation;
- concurrent duplicate external-reference insertion.

Concurrency coverage uses independent sessions and synchronization primitives rather than arbitrary sleeps.

Integration tests are marked with:

```python
pytest.mark.integration
```

Run infrastructure before integration tests:

```powershell
docker compose up -d
docker compose ps
```

Both services must become healthy.

Apply migrations before repository integration tests when the local database is empty:

```powershell
uv run alembic upgrade head
```

Run integration tests:

```powershell
uv run pytest -m integration
```

Run only the integration directory:

```powershell
uv run pytest tests/integration
```

Targeted Alembic and repository integration coverage:

```powershell
uv run pytest tests/integration/test_alembic.py
uv run pytest tests/integration/modules/workspaces/infrastructure
uv run pytest tests/integration/modules/tickets/infrastructure
```
## Full test suite

Run all tests:

```powershell
uv run pytest
```

The project uses pytest importlib mode:

```text
--import-mode=importlib
```

This allows identical test basenames in separate unit and integration directories without module import collisions.

Do not remove this option without restructuring the test package layout.

## Pytest configuration

The repository configures pytest through `pyproject.toml`.

Current behavior includes:

- test discovery under `tests`;
- `src` layout import support;
- strict marker validation;
- strict configuration validation;
- automatic asyncio mode;
- function-scoped event loops for async fixtures;
- explicit integration marker;
- importlib-based module loading.

Strict marker configuration prevents accidental use of undeclared markers.

Strict configuration causes invalid pytest settings to fail instead of being ignored.

## Async test behavior

The project uses `pytest-asyncio`.

Async tests are declared with `async def`.

The configured asyncio mode automatically handles async test functions and fixtures.

Each async fixture uses a function-scoped event loop unless explicitly changed.

This prevents state leakage between tests and keeps infrastructure lifecycle ownership predictable.

## Unit test isolation

Unit tests must remain executable with all Docker services stopped.

Validate isolation:

```powershell
docker compose stop
uv run pytest tests/unit
```

Expected behavior:

- all unit tests pass;
- no connection attempts are made to PostgreSQL;
- no connection attempts are made to Qdrant.

Restart infrastructure after validation:

```powershell
docker compose start
docker compose ps
```

## Infrastructure connectivity tests

### PostgreSQL

The PostgreSQL integration test creates a real async SQLAlchemy engine and executes the application-owned connectivity check.

The check performs:

```sql
SELECT 1
```

The test confirms:

- successful connection;
- healthy dependency status;
- sanitized result;
- explicit engine disposal.

### Qdrant

The Qdrant integration test creates a real async client and performs the application-owned connectivity check.

The check uses a read-only collections request.

The test confirms:

- successful connection;
- healthy dependency status;
- sanitized result;
- explicit client closure;
- no collection creation.

## Health endpoint tests

### Liveness

Liveness tests verify:

- `GET /health/live` returns HTTP `200`;
- response status is `healthy`;
- PostgreSQL is not checked;
- Qdrant is not checked;
- dependency failure does not affect liveness.

### Readiness

Readiness tests verify:

- `GET /health/ready` returns HTTP `200` when all required dependencies are healthy;
- PostgreSQL and Qdrant are reported separately;
- any unhealthy dependency changes aggregate readiness to unhealthy;
- unhealthy readiness returns HTTP `503`;
- timeout and provider failures are sanitized;
- response bodies do not expose hostnames, ports, passwords, DSNs, or provider exception details.

Dependency checks execute concurrently.

The total readiness duration should not become the sum of sequential dependency timeouts.

## HTTP request traceability tests

HTTP request traceability tests verify externally observable guarantees:

- every request receives a server-generated UUID v4 `X-Request-ID`;
- inbound `X-Request-ID` values are ignored;
- a valid inbound `X-Correlation-ID` UUID is propagated;
- absent or invalid correlation values fall back to the request ID;
- both identifiers are returned as response headers;
- active identifiers enrich structured JSON logs;
- context is cleaned up after normal and exceptional completion;
- async tasks remain isolated from each other;
- downstream handlers cannot override trace response headers;
- completion logs include safe operational metadata only;
- unexpected exceptions retain safe `500` behavior with trace headers;
- request bodies and raw invalid header values are not logged.

These tests do not require Docker, PostgreSQL, Qdrant, or network services.

Run the focused request-traceability suite:

```powershell
uv run pytest tests/unit/core/test_request_context.py tests/unit/core/test_logging.py tests/unit/api/test_application.py tests/unit/api/test_request_context_middleware.py
```

## Failure-path testing

Unit tests simulate dependency failures through adapter-level exceptions.

Covered failure paths include:

- PostgreSQL operational failure;
- PostgreSQL timeout;
- Qdrant unexpected response;
- Qdrant response handling failure;
- Qdrant operating system failure;
- Qdrant timeout;
- application cleanup failure.

These tests verify predictable behavior without requiring destructive infrastructure manipulation.

## Manual outage validation

The test suite does not stop Docker services from inside pytest.

Automated service interruption would:

- require Docker socket access;
- mutate shared infrastructure;
- create race conditions;
- reduce CI reliability;
- make test order significant.

Dependency outage behavior is therefore validated manually.

### PostgreSQL outage

Start the API and services, then stop PostgreSQL:

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

- HTTP `503`;
- PostgreSQL is unhealthy;
- Qdrant remains independently evaluated;
- no secret is exposed.

Liveness must remain healthy:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
```

Restart PostgreSQL:

```powershell
docker compose start postgresql
docker compose ps
```

### Qdrant outage

Stop Qdrant:

```powershell
docker compose stop qdrant
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

- HTTP `503`;
- Qdrant is unhealthy;
- PostgreSQL remains independently evaluated;
- the API does not return HTTP `500`;
- no provider exception details are exposed.

Liveness must remain healthy.

Restart Qdrant:

```powershell
docker compose start qdrant
docker compose ps
```

## Alembic validation

The current migration head creates the `workspaces` and `tickets` tables.

Migration lifecycle commands:

```powershell
uv run alembic upgrade head
uv run alembic downgrade base
uv run alembic upgrade head
uv run alembic check
uv run alembic current
uv run alembic heads
```

Expected behavior:

- commands complete successfully;
- the expected workspace and ticket revision is reported as head;
- upgrade creates `workspaces` and `tickets`;
- downgrade removes those business tables;
- re-upgrade restores them.

Validate offline execution:

```powershell
uv run alembic upgrade head --sql
```

Inspect the database:

```powershell
docker compose exec postgresql `
  psql `
  -U supportops `
  -d supportops `
  -c "\dt"
```

After upgrade, the schema includes `workspaces` and `tickets`.

Downgrade commands must only run against the local development or test database.

## Quality checks

Run Ruff lint:

```powershell
uv run ruff check .
```

Run formatting verification:

```powershell
uv run ruff format --check .
```

Run mypy:

```powershell
uv run mypy
```

Validate the dependency lock:

```powershell
uv lock --check
```

Validate Docker Compose:

```powershell
docker compose config --quiet
```

Validate Git whitespace:

```powershell
git diff --check
```

## Docker validation

Build the application image:

```powershell
docker build `
  --tag supportops-ai-platform:local `
  .
```

Run the container against host-exposed PostgreSQL and Qdrant:

```powershell
docker run --rm `
  --env-file .env `
  -e SUPPORTOPS_API_HOST=0.0.0.0 `
  -e SUPPORTOPS_POSTGRESQL_URL=postgresql+asyncpg://supportops:supportops-local@host.docker.internal:5432/supportops `
  -e SUPPORTOPS_QDRANT_URL=http://host.docker.internal:6333 `
  -p 8001:8000 `
  supportops-ai-platform:local
```

Validate:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health/live
Invoke-RestMethod http://127.0.0.1:8001/health/ready
```

The container must:

- start successfully;
- run as a non-root user;
- expose liveness;
- expose readiness;
- connect to PostgreSQL and Qdrant;
- avoid installing development dependencies.

## Continuous integration parity

The GitHub Actions workflow executes the same core gates as local development:

```powershell
uv sync --frozen --all-groups
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -m "not integration"
uv run alembic heads
uv run alembic current
uv run alembic check
uv run pytest -m integration
docker build --tag supportops-ai-platform:ci .
```

Continuous integration provides PostgreSQL and Qdrant service containers.

The CI environment uses non-production credentials and a slightly higher dependency health timeout to reduce shared-runner flakiness.

CI must not update the lockfile or publish the application image.

## Full local validation sequence

Before committing a completed implementation slice, run:

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
uv run pytest
docker compose config --quiet
docker build --tag supportops-ai-platform:local .
git diff --check
```

## Test design guidelines

New tests should:

- verify behavior and invariants;
- isolate external boundaries;
- use live infrastructure only when integration value is real;
- assert safe failure behavior;
- verify cleanup where resources are owned;
- verify context cleanup after normal and exceptional completion;
- use fixed UUIDs where deterministic identifier assertions are needed;
- avoid implementation-only assertions;
- avoid arbitrary sleeps, including for async isolation;
- avoid ordering dependencies;
- avoid hidden reliance on `.env`;
- avoid shared mutable global state;
- preserve deterministic results.

New integration tests must declare the integration marker.

New failure scenarios should be automated when they can remain deterministic and non-destructive.

## Future testing direction

Later implementation phases are expected to add tests for:

- workspace and ticket HTTP endpoints;
- application services and API error contracts;
- HTTP cursor encoding;
- authorization boundaries;
- authenticated tenant isolation;
- asynchronous job claiming;
- AgentRun behavior;
- duplicate execution;
- idempotency;
- retry scheduling;
- stale lease recovery;
- structured LLM outputs;
- retrieval quality;
- tool authorization;
- approval enforcement;
- token and cost accounting;
- prompt regression;
- evaluation thresholds.

API tests remain planned because workspace and ticket HTTP business routes are not implemented.