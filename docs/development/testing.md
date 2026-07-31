# Testing Strategy

## Purpose

The SupportOps AI Platform test suite verifies foundation behavior across configuration, application composition, infrastructure connectivity, lifecycle management, health semantics, HTTP request traceability, workspace and ticket persistence, durable AgentRun scheduling, PostgreSQL worker claim and execution, workspace-scoped AgentRun inspection, application services, versioned HTTP APIs, migration tooling, and container packaging.

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
- AgentRun and AgentRunAttempt domain invariant tests;
- application service command and query behavior;
- transactional ticket-intake orchestration;
- retry policy calculation and attempt-budget gates;
- claim contracts and transition fencing contracts;
- deterministic executor workflow-contract validation;
- processor timeout, terminal, retryable, and sanitized unexpected-failure outcomes;
- worker cycle recovery-before-claim orchestration;
- polling-loop idle waits and interruptible shutdown behavior;
- scoped-session worker runtime composition;
- worker process identity resolution and graceful shutdown;
- SQLAlchemy mapping and metadata tests for AgentRun persistence;
- named PostgreSQL constraints declared on persistence records;
- persistence model registration;
- PostgreSQL constraint-name inspection helpers;
- workspace API schemas;
- ticket API schemas, including the nested processing-run response;
- AgentRun inspection schema projections;
- opaque ticket cursor encoding and invalid-cursor rejection.

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

Targeted domain, application, persistence, API, and worker unit coverage:

```powershell
uv run pytest tests/unit/modules/workspaces/domain tests/unit/modules/tickets/domain
uv run pytest tests/unit/modules/agent_runs/domain
uv run pytest tests/unit/modules/workspaces/application tests/unit/modules/tickets/application
uv run pytest tests/unit/application/test_ticket_intake.py
uv run pytest tests/unit/modules/workspaces/infrastructure tests/unit/modules/tickets/infrastructure
uv run pytest tests/unit/modules/agent_runs/infrastructure
uv run pytest tests/unit/modules/workspaces/api
uv run pytest tests/unit/modules/tickets/api
uv run pytest tests/unit/modules/agent_runs/api
uv run pytest tests/unit/infrastructure/postgresql
```

Worker unit coverage:

```powershell
uv run pytest tests/unit/modules/agent_runs/application
uv run pytest tests/unit/modules/agent_runs/infrastructure/test_worker_runtime.py
uv run pytest tests/unit/worker
```

Focused worker-related application unit tests:

```powershell
uv run pytest tests/unit/modules/agent_runs/application/test_retry_policy.py
uv run pytest tests/unit/modules/agent_runs/application/test_deterministic_executor.py
uv run pytest tests/unit/modules/agent_runs/application/test_processor.py
uv run pytest tests/unit/modules/agent_runs/application/test_worker.py
uv run pytest tests/unit/modules/agent_runs/application/test_worker_loop.py
uv run pytest tests/unit/worker/test_main.py
```

Application service unit coverage:

```powershell
uv run pytest tests/unit/modules/workspaces/application/test_services.py
uv run pytest tests/unit/modules/tickets/application/test_services.py
uv run pytest tests/unit/modules/agent_runs/application/test_services.py
uv run pytest tests/unit/application/test_ticket_intake.py
```

Workspace schema, ticket schema plus cursor, and AgentRun inspection schema unit coverage:

```powershell
uv run pytest tests/unit/modules/workspaces/api/test_schemas.py
uv run pytest tests/unit/modules/tickets/api/test_schemas.py
uv run pytest tests/unit/modules/tickets/api/test_pagination.py
uv run pytest tests/unit/modules/agent_runs/api/test_schemas.py
```

AgentRun inspection application coverage verifies:

- workspace-scoped AgentRun retrieval;
- missing AgentRun raising `AgentRunNotFoundError`;
- cross-workspace AgentRun lookups treated as not found;
- empty attempt history for queued runs;
- deterministic attempt ordering by `attempt_number`;
- ownership validation before attempt listing.

AgentRun inspection schema coverage verifies:

- safe public field projection for AgentRun responses;
- safe public field projection for attempt responses;
- omission of `lease_owner`, `lease_token`, `lease_expires_at`, and `ingestion_request_id`;
- omission of attempt `agent_run_id`, `lease_token`, and `execution_request_id`.

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
- migration upgrade, downgrade, and parity checks;
- creation of `workspaces`, `tickets`, `agent_runs`, and `agent_run_attempts` tables;
- workspace persistence;
- duplicate workspace slug translation;
- transaction rollback;
- ticket foreign-key behavior;
- the same external reference across workspaces;
- duplicate external-reference rejection inside one workspace;
- cross-workspace ticket lookup behavior;
- deterministic ticket listing;
- keyset repository navigation;
- concurrent duplicate external-reference insertion;
- atomic ticket and run commit;
- run insertion failure rolling back the ticket;
- duplicate ticket conflict creating no additional run;
- PostgreSQL claim ordering;
- `FOR UPDATE SKIP LOCKED` claim concurrency;
- fenced success and failure transitions;
- stale lease-token rejection;
- expired lease recovery;
- processor transaction separation against live PostgreSQL;
- workspace API creation and retrieval;
- duplicate slug conflict responses;
- ticket API intake, retrieval, and listing;
- API response and persistence verification for the processing-run reference;
- request and correlation identifier persistence;
- duplicate external-reference conflict responses;
- cross-workspace `404` behavior;
- empty ticket listing for an existing workspace;
- workspace-not-found behavior for ticket listing;
- opaque cursor pagination;
- invalid cursor responses;
- page-size validation;
- request schema validation errors;
- workspace-scoped AgentRun retrieval;
- empty and ordered AgentRunAttempt history responses;
- AgentRun HTTP `404` for missing and cross-workspace resources;
- invalid AgentRun UUID validation;
- absence of request bodies from completion logs, including ticket subject and description content.

Concurrency coverage uses independent sessions and synchronization primitives rather than arbitrary sleeps.

Full API integration tests require PostgreSQL and applied migrations. Qdrant remains required for readiness and shared integration fixtures, but business routes do not call Qdrant. Qdrant-dependent tests are not worker tests.

PostgreSQL integration tests are required for claim ordering, `SKIP LOCKED` concurrency, lease-token fencing, and expired lease recovery because those behaviors depend on real row locking and commit visibility.

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

Targeted Alembic, repository, AgentRun, and API integration coverage:

```powershell
uv run pytest tests/integration/test_alembic.py
uv run pytest tests/integration/modules/workspaces/infrastructure
uv run pytest tests/integration/modules/tickets/infrastructure
uv run pytest tests/integration/application/test_ticket_intake.py
uv run pytest tests/integration/modules/agent_runs
uv run pytest tests/integration/api/test_workspaces.py
uv run pytest tests/integration/api/test_tickets.py
uv run pytest tests/integration/api/test_agent_runs.py
```

The concurrency-sensitive duplicate external-reference repository test remains part of the ticket infrastructure integration suite and should continue to run against real PostgreSQL.

Atomic ticket and AgentRun scheduling coverage:

```powershell
uv run pytest tests/integration/application/test_ticket_intake.py
```

AgentRun worker integration coverage:

```powershell
uv run pytest tests/integration/modules/agent_runs/infrastructure/test_claim_repository.py
uv run pytest tests/integration/modules/agent_runs/infrastructure/test_transition_repository.py
uv run pytest tests/integration/modules/agent_runs/infrastructure/test_recovery_repository.py
uv run pytest tests/integration/modules/agent_runs/application/test_processor.py
```

These AgentRun integration tests verify:

- deterministic claim ordering;
- `SKIP LOCKED` concurrency across concurrent claimers;
- fenced transitions and stale-token rejection;
- expired lease recovery without incrementing attempt count;
- processor transaction separation for ticket load, executor work, and outcome persistence.

AgentRun query repository integration coverage:

```powershell
uv run pytest tests/integration/modules/agent_runs/infrastructure/test_query_repository.py
```

These query-repository tests verify:

- workspace-scoped PostgreSQL lookup predicates;
- missing and cross-workspace runs returning no row;
- empty attempt history for queued runs;
- actual SQL ordering by `attempt_number` ascending;
- attempt history scoped to the requested AgentRun.

Workspace, ticket, and AgentRun API integration coverage:

```powershell
uv run pytest tests/integration/api/test_workspaces.py tests/integration/api/test_tickets.py
uv run pytest tests/integration/api/test_agent_runs.py
```

AgentRun API integration coverage verifies:

- queued AgentRun inspection after ticket creation;
- omission of internal lease and execution identifiers from HTTP responses;
- HTTP `404` with `agent_run_not_found` for missing and cross-workspace runs;
- empty attempt-history envelopes;
- ordered attempt-history envelopes;
- FastAPI dependency composition for inspection routes;
- stable error envelope integration;
- invalid UUID validation responses;
- tenant isolation behavior that does not disclose cross-workspace ownership.

Integration tests are necessary for AgentRun inspection because unit tests cannot fully prove:

- workspace-scoped SQL predicates against PostgreSQL;
- actual SQL ordering of attempts;
- FastAPI dependency composition for the mounted routes;
- error envelope integration through registered handlers;
- end-to-end tenant isolation behavior across HTTP and persistence.

## Full test suite

Run all tests:

```powershell
uv run pytest
```

Run the complete unit suite:

```powershell
uv run pytest -m "not integration"
```

Run the complete integration suite:

```powershell
uv run pytest -m integration
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
- request bodies and raw invalid header values are not logged;
- completion logs omit ticket subject and description content because request bodies are not logged.

These tests do not require Docker, PostgreSQL, Qdrant, or network services.

Run the focused request-traceability suite:

```powershell
uv run pytest tests/unit/core/test_request_context.py tests/unit/core/test_logging.py tests/unit/api/test_application.py tests/unit/api/test_request_context_middleware.py
```

## Workspace, ticket, and AgentRun API tests

Workspace API integration coverage verifies:

- workspace creation and retrieval;
- duplicate slug conflict responses;
- missing workspace responses;
- malformed identifier validation errors;
- invalid create payload validation errors;
- health routes remaining outside `/api/v1`.

Ticket API integration coverage verifies:

- request and correlation identifier persistence on intake;
- nested ticket and processing-run response shape;
- queued `ticket-processing` / `deterministic-baseline-v1` processing reference;
- persistence of the referenced AgentRun after successful creation;
- missing workspace behavior for ticket creation and listing;
- duplicate external-reference conflicts within one workspace;
- reuse of the same external reference across workspaces;
- cross-workspace retrieval returning `ticket_not_found`;
- empty listing for an existing workspace;
- opaque cursor pagination;
- invalid cursor responses;
- page-size validation;
- request schema validation errors.

AgentRun inspection API integration coverage verifies:

- workspace-scoped AgentRun retrieval;
- empty and ordered attempt-history responses;
- safe schema projections without fencing identifiers;
- HTTP `404` with `agent_run_not_found` for missing and cross-workspace resources;
- invalid UUID validation;
- tenant-safe ownership behavior.

Transactional ticket-intake integration coverage verifies:

- atomic ticket and run commit;
- run insertion failure rolling back the ticket;
- duplicate ticket conflict creating no additional run.

Run the focused API and intake suites:

```powershell
uv run pytest tests/integration/api/test_workspaces.py
uv run pytest tests/integration/api/test_tickets.py
uv run pytest tests/integration/api/test_agent_runs.py
uv run pytest tests/integration/application/test_ticket_intake.py
```

Focused AgentRun inspection coverage:

```powershell
uv run pytest tests/unit/modules/agent_runs/application/test_services.py
uv run pytest tests/unit/modules/agent_runs/api/test_schemas.py
uv run pytest tests/integration/modules/agent_runs/infrastructure/test_query_repository.py
uv run pytest tests/integration/api/test_agent_runs.py
```

Full API integration tests require PostgreSQL and applied migrations.

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

The current migration head creates the `workspaces`, `tickets`, `agent_runs`, and `agent_run_attempts` tables.

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
- the expected revision is reported as head;
- upgrade creates `workspaces`, `tickets`, `agent_runs`, and `agent_run_attempts`;
- downgrade removes those business tables;
- re-upgrade restores them;
- migration metadata remains aligned with SQLAlchemy model metadata.

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

After upgrade, the schema includes `workspaces`, `tickets`, `agent_runs`, and `agent_run_attempts`.

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

- authorization boundaries;
- authenticated tenant isolation;
- manual AgentRun retry and cancellation;
- global AgentRun listing and status filtering;
- structured LLM outputs;
- retrieval quality and Qdrant indexing;
- LangGraph orchestration;
- tool authorization;
- approval enforcement;
- token and cost accounting;
- prompt versioning;
- prompt regression;
- evaluation thresholds;
- idempotent side effects for future executors and tools.

Authentication and AI classification remain intentional scope boundaries for the current suite. Durable AgentRun scheduling, PostgreSQL claiming, fencing, retries, recovery, deterministic execution, worker process coverage, and workspace-scoped AgentRun inspection are part of the current suite.