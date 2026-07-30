# Environment Variables

## Purpose

SupportOps AI Platform uses environment-based configuration validated by `pydantic-settings`.

Application variables use the `SUPPORTOPS_` prefix.

Docker Compose infrastructure variables remain unprefixed because they configure local containers rather than the application process.

The `.env.example` file contains safe local development values. Copy it to `.env` for local use:

```powershell
Copy-Item .env.example .env
```

The `.env` file must not be committed.

## Configuration loading

The application loads configuration from:

1. process environment variables;
2. `.env` when present;
3. validated defaults for optional values.

Explicit process environment variables take precedence over `.env`.

Variable names are case-insensitive.

Unknown environment variables are ignored by the application settings model.

## Application variables

### `SUPPORTOPS_ENVIRONMENT`

Runtime environment identifier.

Allowed values:

```text
local
test
development
staging
production
```

Default:

```text
local
```

Used for:

- structured log context;
- environment-specific operational visibility;
- future environment-aware policy decisions.

It must not be used as a substitute for explicit security configuration.

### `SUPPORTOPS_APPLICATION_NAME`

Human-readable application name.

Default:

```text
SupportOps AI Platform
```

Constraints:

- required to contain non-whitespace characters;
- maximum length of 100 characters.

Used for:

- OpenAPI metadata;
- structured startup logs.

### `SUPPORTOPS_APPLICATION_VERSION`

Application version exposed in OpenAPI metadata and startup logs.

Default:

```text
0.1.0
```

Constraints:

- required to contain non-whitespace characters;
- maximum length of 32 characters.

This value should remain aligned with the project metadata in `pyproject.toml`.

### `SUPPORTOPS_LOG_LEVEL`

Minimum application log level.

Allowed values:

```text
DEBUG
INFO
WARNING
ERROR
CRITICAL
```

Default:

```text
INFO
```

The repository foundation emits structured JSON logs through the Python standard library.

### `SUPPORTOPS_API_HOST`

Host interface used when starting the API through local commands or process configuration.

Default:

```text
127.0.0.1
```

For container execution, use:

```text
0.0.0.0
```

Constraints:

- required to contain non-whitespace characters.

The Dockerfile currently supplies the container host explicitly through the Uvicorn command.

### `SUPPORTOPS_API_PORT`

Application HTTP port.

Default:

```text
8000
```

Allowed range:

```text
1-65535
```

The Dockerfile exposes port `8000`.

## PostgreSQL application variables

### `SUPPORTOPS_POSTGRESQL_URL`

Required PostgreSQL connection URL used by:

- the FastAPI process;
- SQLAlchemy;
- Alembic;
- integration tests.

Local default from `.env.example`:

```text
postgresql+asyncpg://supportops:supportops-local@localhost:5432/supportops
```

Required scheme:

```text
postgresql+asyncpg
```

The application validates the value as a PostgreSQL DSN.

Security requirements:

- do not commit production credentials;
- do not include the URL in logs;
- do not expose the URL in health responses;
- use a managed secret mechanism before public deployment.

### `SUPPORTOPS_POSTGRESQL_POOL_SIZE`

Number of persistent connections maintained by the SQLAlchemy pool per process.

Default:

```text
5
```

Allowed range:

```text
1-50
```

API and future worker processes will own independent pools.

Total database connection planning must account for every running process.

### `SUPPORTOPS_POSTGRESQL_MAX_OVERFLOW`

Maximum number of temporary connections above the configured pool size.

Default:

```text
10
```

Allowed range:

```text
0-100
```

Overflow connections should absorb short bursts rather than replace deliberate capacity planning.

### `SUPPORTOPS_POSTGRESQL_POOL_TIMEOUT_SECONDS`

Maximum time to wait for a pooled database connection.

Default:

```text
10
```

Allowed range:

```text
greater than 0 and no more than 60 seconds
```

This timeout applies to pool acquisition.

It is separate from dependency health-check timeout behavior.

## Qdrant application variables

### `SUPPORTOPS_QDRANT_URL`

Required Qdrant HTTP endpoint.

Local default:

```text
http://localhost:6333
```

Constraints:

- required to contain non-whitespace characters.

The current foundation uses HTTP rather than gRPC for application connectivity.

The URL must not be returned in readiness responses.

### `SUPPORTOPS_QDRANT_API_KEY`

Optional Qdrant API key.

Local default:

```text
empty
```

Blank values are normalized to an absent value.

Security requirements:

- do not commit real API keys;
- do not log the value;
- do not return the value in health responses;
- use secret management before production deployment.

Local Qdrant does not require an API key.

## Dependency health configuration

### `SUPPORTOPS_DEPENDENCY_HEALTH_TIMEOUT_SECONDS`

Maximum duration of each PostgreSQL or Qdrant readiness check.

Default:

```text
2
```

Allowed range:

```text
greater than 0 and no more than 30 seconds
```

Each dependency check is independently bounded.

The readiness service executes checks concurrently, so total readiness latency is not the sum of sequential dependency timeouts.

Continuous integration uses a higher value to reduce shared-runner flakiness:

```text
5
```

This does not change the local default.

## Docker Compose variables

These variables configure local infrastructure containers.

They are not read by the application settings model unless separately mapped through a `SUPPORTOPS_` variable.

### `POSTGRES_DB`

PostgreSQL database created by the local container.

Local value:

```text
supportops
```

### `POSTGRES_USER`

PostgreSQL user created by the local container.

Local value:

```text
supportops
```

### `POSTGRES_PASSWORD`

PostgreSQL password used only by the local container.

Local value:

```text
supportops-local
```

This value is intentionally limited to local development.

It must not be reused in shared or production environments.

### `POSTGRES_PORT`

Host port mapped to PostgreSQL container port `5432`.

Local value:

```text
5432
```

When changing this value, update `SUPPORTOPS_POSTGRESQL_URL` to use the same host port.

### `QDRANT_HTTP_PORT`

Host port mapped to Qdrant HTTP port `6333`.

Local value:

```text
6333
```

When changing this value, update `SUPPORTOPS_QDRANT_URL`.

### `QDRANT_GRPC_PORT`

Host port mapped to Qdrant gRPC port `6334`.

Local value:

```text
6334
```

The current application foundation does not use gRPC, but the local service exposes the standard Qdrant port for future controlled evaluation.

## Local `.env` example

The repository-provided `.env.example` is:

```dotenv
# Application
SUPPORTOPS_ENVIRONMENT=local
SUPPORTOPS_APPLICATION_NAME=SupportOps AI Platform
SUPPORTOPS_APPLICATION_VERSION=0.1.0
SUPPORTOPS_LOG_LEVEL=INFO

# API
SUPPORTOPS_API_HOST=127.0.0.1
SUPPORTOPS_API_PORT=8000

# PostgreSQL
POSTGRES_DB=supportops
POSTGRES_USER=supportops
POSTGRES_PASSWORD=supportops-local
POSTGRES_PORT=5432

SUPPORTOPS_POSTGRESQL_URL=postgresql+asyncpg://supportops:supportops-local@localhost:5432/supportops
SUPPORTOPS_POSTGRESQL_POOL_SIZE=5
SUPPORTOPS_POSTGRESQL_MAX_OVERFLOW=10
SUPPORTOPS_POSTGRESQL_POOL_TIMEOUT_SECONDS=10

# Qdrant
QDRANT_HTTP_PORT=6333
QDRANT_GRPC_PORT=6334

SUPPORTOPS_QDRANT_URL=http://localhost:6333
SUPPORTOPS_QDRANT_API_KEY=

# Dependency health checks
SUPPORTOPS_DEPENDENCY_HEALTH_TIMEOUT_SECONDS=2
```

## Validation behavior

Missing required values fail during settings construction.

Required variables include:

```text
SUPPORTOPS_POSTGRESQL_URL
SUPPORTOPS_QDRANT_URL
```

Examples of invalid configuration include:

- malformed PostgreSQL DSN;
- blank Qdrant URL;
- API port outside the valid range;
- non-positive pool timeout;
- non-positive dependency health timeout;
- unsupported environment value;
- unsupported log level.

Validate application settings through the test suite:

```powershell
uv run pytest tests/unit/core/test_settings.py
```

Validate the full repository:

```powershell
uv run pytest
```

## Secret handling

The repository foundation follows these rules:

- `.env` is ignored by Git;
- `.env.example` contains local-only values;
- settings are not logged as a complete object;
- PostgreSQL credentials are not returned by health endpoints;
- Qdrant API keys are not returned by health endpoints;
- CI uses non-production service credentials;
- no external secret management system is introduced in the local foundation.

Before public production deployment, credentials must move to an environment-appropriate secret management service.

## Adding future variables

A new environment variable should be introduced only when a concrete runtime capability requires it.

Each addition must include:

- validated settings field;
- safe default or explicit required status;
- `.env.example` update when appropriate;
- unit tests;
- documentation in this file;
- CI configuration when integration tests require it;
- explicit secret-handling rules when sensitive.

Generic provider placeholders should not be added before the corresponding adapter or workflow exists.