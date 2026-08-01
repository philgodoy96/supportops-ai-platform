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

API and worker processes own independent pools.

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

## PostgreSQL worker configuration

Worker settings are part of the shared `Settings` model and are validated at
process startup for both API and worker construction. The worker process uses
these values for identity, polling, claiming, execution timeout, retry
scheduling, and cooperative shutdown. Ticket intake copies
`SUPPORTOPS_WORKER_MAX_ATTEMPTS` into newly scheduled `AgentRun` records.

Cross-field invariants:

- `SUPPORTOPS_WORKER_LEASE_SECONDS` must exceed
  `SUPPORTOPS_WORKER_EXECUTION_TIMEOUT_SECONDS` by at least five seconds;
- `SUPPORTOPS_WORKER_RETRY_MAX_SECONDS` must not be smaller than
  `SUPPORTOPS_WORKER_RETRY_BASE_SECONDS`;
- the complete logical LLM invocation budget must fit inside
  `SUPPORTOPS_WORKER_EXECUTION_TIMEOUT_SECONDS` with a five-second safety
  margin;
- logical invocation count is `1 + SUPPORTOPS_LLM_MAX_REPAIR_ATTEMPTS`;
- logical LLM budget is
  `SUPPORTOPS_LLM_REQUEST_TIMEOUT_SECONDS × logical invocation count`;
- provider transport retries are internal SDK behavior and are not separate
  application logical invocations.

The supported executor value is exactly:

```text
deterministic-ticket-processing
```

### `SUPPORTOPS_WORKER_ID`

Optional worker identity.

Settings field:

```text
worker_id
```

Type:

```text
string or omitted
```

Default:

```text
omitted
```

When omitted, the worker generates an identity from hostname, process ID, and a
UUID suffix, truncated to 128 characters.

Constraints:

- maximum length of 128 characters;
- must not be empty when provided;
- must not contain surrounding whitespace.

Applies to:

```text
worker
```

Purpose:

- identifies the lease owner and attempt worker ID;
- distinguishes concurrent worker processes in operational logs.

Example safe value:

```text
worker-local-1
```

### `SUPPORTOPS_WORKER_EXECUTOR`

Configured executor name.

Settings field:

```text
worker_executor
```

Type:

```text
literal string
```

Default:

```text
deterministic-ticket-processing
```

Accepted value:

```text
deterministic-ticket-processing
```

Applies to:

```text
worker
```

Purpose:

- selects the deterministic baseline executor;
- records the configured executor in worker startup logs.

Example safe value:

```text
deterministic-ticket-processing
```

### `SUPPORTOPS_WORKER_POLL_INTERVAL_SECONDS`

Idle polling interval between worker cycles.

Settings field:

```text
worker_poll_interval_seconds
```

Type:

```text
float seconds
```

Default:

```text
1.0
```

Accepted range:

```text
greater than 0 and no more than 60
```

Applies to:

```text
worker
```

Purpose:

- waits only after an idle cycle;
- remains interruptible by cooperative shutdown.

Example safe value:

```text
1.0
```

### `SUPPORTOPS_WORKER_LEASE_SECONDS`

Lease duration assigned when a run is claimed.

Settings field:

```text
worker_lease_seconds
```

Type:

```text
float seconds
```

Default:

```text
45.0
```

Accepted range:

```text
greater than 0 and no more than 3600
```

Applies to:

```text
worker
```

Purpose:

- bounds ownership of a claimed `AgentRun`;
- must leave headroom beyond the execution timeout for fenced completion.

Example safe value:

```text
45.0
```

### `SUPPORTOPS_WORKER_EXECUTION_TIMEOUT_SECONDS`

Maximum executor runtime for one claimed attempt.

Settings field:

```text
worker_execution_timeout_seconds
```

Type:

```text
float seconds
```

Default:

```text
30.0
```

Accepted range:

```text
greater than 0 and no more than 1800
```

Applies to:

```text
worker
```

Purpose:

- bounds executor work outside database transactions;
- timeout outcomes are persisted as `timed_out` and may retry while budget remains.

Example safe value:

```text
30.0
```

### `SUPPORTOPS_WORKER_SHUTDOWN_GRACE_SECONDS`

Grace period for cooperative worker shutdown.

Settings field:

```text
worker_shutdown_grace_seconds
```

Type:

```text
float seconds
```

Default:

```text
10.0
```

Accepted range:

```text
0 through 300
```

Applies to:

```text
worker
```

Purpose:

- allows the active cycle to finish after SIGINT or SIGTERM;
- cancels the loop task when the grace period is exceeded;
- a value of `0` cancels immediately after shutdown is requested.

Example safe value:

```text
10.0
```

### `SUPPORTOPS_WORKER_MAX_ATTEMPTS`

Integer retry budget copied into each newly created `AgentRun` during ticket
intake and enforced during worker retries and recovery.

Settings field:

```text
worker_max_attempts
```

Type:

```text
integer
```

Default:

```text
3
```

Accepted range:

```text
1 through 100
```

Applies to:

```text
API and worker
```

Purpose:

- defines the maximum number of attempts available to a newly scheduled AgentRun;
- is persisted on the AgentRun at ticket intake time;
- remains immutable for that run after scheduling;
- gates retry scheduling and exhausted recovery outcomes in the worker.

Existing runs retain their persisted retry budget after configuration changes.
Changing this setting affects only AgentRun records created after the new value
is loaded.

Example safe value:

```text
3
```

### `SUPPORTOPS_WORKER_RETRY_BASE_SECONDS`

Base delay used by bounded exponential backoff.

Settings field:

```text
worker_retry_base_seconds
```

Type:

```text
float seconds
```

Default:

```text
2.0
```

Accepted range:

```text
greater than 0 and no more than 3600
```

Applies to:

```text
worker
```

Purpose:

- calculates the initial retry delay after a retryable failure or timeout;
- also informs the delay used when recovering an expired lease with remaining budget.

Example safe value:

```text
2.0
```

### `SUPPORTOPS_WORKER_RETRY_MAX_SECONDS`

Maximum delay used by bounded exponential backoff.

Settings field:

```text
worker_retry_max_seconds
```

Type:

```text
float seconds
```

Default:

```text
60.0
```

Accepted range:

```text
greater than 0 and no more than 86400
```

Applies to:

```text
worker
```

Purpose:

- caps retry delays so backoff remains bounded;
- must not be smaller than `SUPPORTOPS_WORKER_RETRY_BASE_SECONDS`.

Example safe value:

```text
60.0
```

## AI runtime configuration

Provider choice is explicit. Local development defaults to `mock`, which does
not require a network connection or an OpenAI API key. OpenAI credentials are
required only when `openai` is selected.

Provider, model, workflow version, prompt version, and schema version remain
independent configuration and provenance dimensions. This pull request
introduces the provider adapters. Worker composition and durable classification
execution are introduced by the classification workflow and are not claimed as
complete here. No provider fallback exists.

### `SUPPORTOPS_TICKET_PROCESSING_WORKFLOW_VERSION`

Configured workflow version assigned to newly scheduled ticket-processing runs
when workflow-version-aware ticket intake is composed.

Settings field:

```text
ticket_processing_workflow_version
```

Type:

```text
literal string
```

Default:

```text
ticket-classification-v1
```

Accepted values:

```text
deterministic-baseline-v1
ticket-classification-v1
```

Applies to:

```text
API and worker configuration
```

Purpose:

- keeps workflow version independent from provider, model, prompt version, and
  schema version;
- allows deterministic historical runs to remain supported;
- controls only new scheduling after workflow-aware ticket intake is composed.

This setting is validated now. The actual scheduling change belongs to the
classification workflow integration and must not be treated as already complete
in this pull request.

Example safe value:

```text
ticket-classification-v1
```

### `SUPPORTOPS_LLM_PROVIDER`

Explicit LLM provider adapter selection.

Settings field:

```text
llm_provider
```

Type:

```text
enum string
```

Default:

```text
mock
```

Accepted values:

```text
mock
openai
```

Applies to:

```text
worker configuration
```

Purpose:

- selects one configured provider;
- keeps local development network-free by default;
- prevents implicit provider fallback.

OpenAI failure never selects `mock` automatically.

Example safe value:

```text
mock
```

### `SUPPORTOPS_OPENAI_API_KEY`

Optional OpenAI API credential required only when
`SUPPORTOPS_LLM_PROVIDER=openai`.

Settings field:

```text
openai_api_key
```

Type:

```text
secret string or omitted
```

Default:

```text
omitted
```

Constraints:

- blank values normalize to omitted;
- OpenAI provider selection requires a nonblank value;
- mock provider selection does not require a value.

Applies to:

```text
worker configuration when OpenAI is selected
```

Security requirements:

- never commit a real value;
- never include it in logs;
- never include complete settings in logs;
- use managed secret storage before production deployment.

Local `.env.example` leaves the value empty:

```text
SUPPORTOPS_OPENAI_API_KEY=
```

### `SUPPORTOPS_OPENAI_MODEL`

Deployment-configured OpenAI model identifier.

Settings field:

```text
openai_model
```

Type:

```text
string
```

Default:

```text
gpt-5-nano
```

Constraints:

- 1 through 128 characters after trimming;
- must not contain surrounding whitespace.

Applies to:

```text
worker configuration when OpenAI is selected
```

Purpose:

- centralizes model selection;
- prevents model identifiers from being distributed through business modules;
- preserves provider/model provenance for future invocation persistence.

Aliases may receive provider updates. The application does not switch models
automatically.

Example safe value:

```text
gpt-5-nano
```

### `SUPPORTOPS_OPENAI_BASE_URL`

Optional OpenAI-compatible base URL override.

Settings field:

```text
openai_base_url
```

Type:

```text
string or omitted
```

Default:

```text
omitted
```

Constraints:

- blank values normalize to omitted;
- maximum 2048 characters;
- surrounding whitespace is removed.

Applies to:

```text
worker configuration when OpenAI is selected
```

Purpose:

- supports controlled compatible endpoints and test environments;
- keeps endpoint configuration out of business code.

Setting a base URL does not create a provider client during settings
validation.

### `SUPPORTOPS_LLM_REQUEST_TIMEOUT_SECONDS`

Maximum duration of one logical provider invocation.

Settings field:

```text
llm_request_timeout_seconds
```

Type:

```text
float seconds
```

Default:

```text
12.0
```

Accepted range:

```text
greater than 0 and no more than 300
```

Applies to:

```text
worker configuration
```

Purpose:

- bounds one initial or repair provider invocation;
- contributes to the validated logical LLM execution budget.

This value is not the AgentRun execution timeout and does not include the
entire outer retry lifecycle.

Example safe value:

```text
12.0
```

### `SUPPORTOPS_LLM_TRANSPORT_MAX_RETRIES`

Maximum SDK-managed transport retries within one logical provider invocation.

Settings field:

```text
llm_transport_max_retries
```

Type:

```text
integer
```

Default:

```text
1
```

Accepted range:

```text
0 through 2
```

Applies to:

```text
OpenAI provider configuration
```

Purpose:

- makes SDK transport retry behavior explicit;
- handles eligible transient provider transport failures;
- avoids relying on the SDK default.

The application does not add another manual transport retry loop. This value is
separate from gateway repair and AgentRun retry. It is not used to multiply the
application logical invocation count.

Example safe value:

```text
1
```

### `SUPPORTOPS_LLM_MAX_REPAIR_ATTEMPTS`

Maximum application-owned structured-output repair invocations after the
initial provider invocation.

Settings field:

```text
llm_max_repair_attempts
```

Type:

```text
integer
```

Default:

```text
1
```

Accepted range:

```text
0 through 1
```

Applies to:

```text
LLM Gateway
```

Purpose:

- bounds replacement requests after repair-eligible structured-output failure;
- prevents indefinite repair;
- contributes to the worker execution-time budget.

Repair is a new logical provider invocation. Refusal is not repaired.
Authentication, quota, invalid request, timeout, and provider-unavailable
failures are not repaired. AgentRun retry is a separate outer layer.

Example safe value:

```text
1
```

### LLM timing relationship

Logical LLM timing relates to the worker execution timeout as follows:

```text
logical_invocation_count = 1 + llm_max_repair_attempts

logical_llm_budget_seconds =
    llm_request_timeout_seconds × logical_invocation_count

worker_execution_timeout_seconds
    >= logical_llm_budget_seconds + 5
```

Defaults:

```text
logical_invocation_count = 2
logical_llm_budget_seconds = 24
minimum worker execution timeout = 29
configured worker execution timeout = 30
configured worker lease = 45
```

The lease must independently exceed the worker execution timeout by at least
five seconds.

### Provider and retry semantics

- `mock` is explicit and deterministic;
- `mock` is not a fallback;
- OpenAI uses the configured model;
- transport retry is SDK-managed;
- repair is gateway-managed;
- AgentRun retry is processor-managed;
- no automatic model switching exists;
- no cross-provider fallback exists.

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

# AI runtime
SUPPORTOPS_TICKET_PROCESSING_WORKFLOW_VERSION=ticket-classification-v1
SUPPORTOPS_LLM_PROVIDER=mock
SUPPORTOPS_OPENAI_API_KEY=
SUPPORTOPS_OPENAI_MODEL=gpt-5-nano
SUPPORTOPS_OPENAI_BASE_URL=
SUPPORTOPS_LLM_REQUEST_TIMEOUT_SECONDS=12
SUPPORTOPS_LLM_TRANSPORT_MAX_RETRIES=1
SUPPORTOPS_LLM_MAX_REPAIR_ATTEMPTS=1

# Qdrant
QDRANT_HTTP_PORT=6333
QDRANT_GRPC_PORT=6334

SUPPORTOPS_QDRANT_URL=http://localhost:6333
SUPPORTOPS_QDRANT_API_KEY=

# Dependency health checks
SUPPORTOPS_DEPENDENCY_HEALTH_TIMEOUT_SECONDS=2
```

`SUPPORTOPS_WORKER_*` values use validated defaults when unset and do not need
to appear in the local example file for basic local development. Set
`SUPPORTOPS_WORKER_ID` explicitly when distinguishing concurrent local workers.
The AI runtime section above matches `.env.example`, including the empty
`SUPPORTOPS_OPENAI_API_KEY=` placeholder.

## Validation behavior

Missing required values fail during settings construction.

Required variables include:

```text
SUPPORTOPS_POSTGRESQL_URL
SUPPORTOPS_QDRANT_URL
```

The shared settings model still requires `SUPPORTOPS_QDRANT_URL` for process
construction. The worker validates that shared model at startup but does not
initialize or connect to Qdrant.

Examples of invalid configuration include:

- malformed PostgreSQL DSN;
- blank Qdrant URL;
- API port outside the valid range;
- worker maximum attempts outside the accepted range;
- worker lease duration that does not exceed execution timeout by at least five seconds;
- worker retry maximum smaller than retry base;
- unsupported worker executor value;
- non-positive pool timeout;
- non-positive dependency health timeout;
- unsupported environment value;
- unsupported log level;
- unsupported LLM provider;
- unsupported ticket-processing workflow version;
- OpenAI selected without an API key;
- request timeout outside its accepted range;
- transport retry count outside 0 through 2;
- repair attempt count outside 0 through 1;
- logical LLM budget exceeding the worker execution timeout after the required
  five-second safety margin.

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
- OpenAI API keys use a secret settings type;
- OpenAI API keys are not logged;
- `.env.example` leaves the OpenAI key empty;
- real OpenAI credentials require environment-appropriate secret management;
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