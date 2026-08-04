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

The worker always composes a versioned executor registry containing four exact
workflow versions: `deterministic-baseline-v1`, `ticket-classification-v1`,
`controlled-support-v1`, and `human-approved-support-v1`. Executor selection is
not deployment configuration. Each AgentRun's stored workflow version controls
dispatch. The local default configured version is `controlled-support-v1`. The
worker may compose embedding and Qdrant resources for controlled and
human-approved workflow knowledge search in addition to its LLM and checkpoint
runtimes.

Cross-field invariants:

- `SUPPORTOPS_WORKER_LEASE_SECONDS` must exceed
  `SUPPORTOPS_WORKER_EXECUTION_TIMEOUT_SECONDS` by at least 15 seconds;
- `SUPPORTOPS_WORKER_RETRY_MAX_SECONDS` must not be smaller than
  `SUPPORTOPS_WORKER_RETRY_BASE_SECONDS`;
- controlled workflow generation slots equal 6 before repair multiplication;
- each generation slot may use
  `1 + SUPPORTOPS_LLM_MAX_REPAIR_ATTEMPTS` logical provider requests;
- controlled budget equals
  `6 × SUPPORTOPS_LLM_REQUEST_TIMEOUT_SECONDS × (1 + SUPPORTOPS_LLM_MAX_REPAIR_ATTEMPTS)`;
- `SUPPORTOPS_WORKER_EXECUTION_TIMEOUT_SECONDS` must cover the controlled budget
  plus 15 seconds when the configured workflow is `controlled-support-v1`;
- direct classification and deterministic baseline retain their smaller
  single-generation validation behavior when selected;
- provider transport retries are internal SDK behavior and are not separate
  application logical invocations.
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
150.0
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
150.0
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
135.0
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
135.0
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

## Human approval configuration

Human-approved workflow interrupt and expiration reuse existing shared approval
settings. No additional secret is required for the durable approval boundary.
Approval decision and inspection HTTP APIs are introduced in a separate
operational API slice and are not configured here.

### `SUPPORTOPS_APPROVAL_TTL_SECONDS`

TTL applied when a durable pending approval is created for a sensitive proposal.

Settings field:

```text
approval_ttl_seconds
```

Type:

```text
float seconds
```

Default:

```text
86400.0
```

Accepted range:

```text
greater than 0 and no more than 2592000
```

Applies to:

```text
worker sensitive proposal persistence
```

Purpose:

- bounds how long a pending approval remains eligible before expiration;
- is copied into `ApprovalRequest.expires_at` at proposal time;
- is reused by the human-approved workflow without a separate secret.

Example safe value:

```text
86400.0
```

### `SUPPORTOPS_APPROVAL_EXPIRATION_BATCH_SIZE`

Maximum number of overdue pending approvals processed in one worker-cycle
expiration pass.

Settings field:

```text
approval_expiration_batch_size
```

Type:

```text
integer
```

Default:

```text
100
```

Accepted range:

```text
1 through 1000
```

Applies to:

```text
worker cycle approval expiration
```

Purpose:

- bounds expiration work per cycle after lease recovery and before claim;
- keeps expiration batches short and independent from claim transactions.

Example safe value:

```text
100
```

## AI runtime configuration

LLM provider and embedding provider selections are independent. Local
development defaults both to `mock`, which does not require a network connection
or an OpenAI API key. OpenAI credentials are required when either the OpenAI
generation adapter or the OpenAI embedding adapter is selected.

Provider, model, workflow version, prompt version, schema version, and embedding
profile remain independent configuration and provenance dimensions. The API
owns the embedding provider for public semantic retrieval and does not create
the LLM provider. The indexing CLI owns the embedding provider for indexing.
The worker owns the LLM provider and, for controlled workflows, also owns the
embedding provider used for controlled knowledge search. The worker converts
`SUPPORTOPS_POSTGRESQL_URL` internally to a Psycopg-compatible DSN for the
checkpoint runtime without logging the secret. No separate checkpoint DSN
environment variable exists. No provider fallback exists.

### `SUPPORTOPS_TICKET_PROCESSING_WORKFLOW_VERSION`

Configured workflow version assigned to newly scheduled ticket-processing runs
during ticket intake.

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
controlled-support-v1
```

Accepted values:

```text
deterministic-baseline-v1
ticket-classification-v1
controlled-support-v1
human-approved-support-v1
```

Applies to:

```text
API scheduling of newly accepted tickets
```

Purpose:

- assigns the workflow version persisted on newly created AgentRuns;
- keeps workflow version independent from provider, model, prompt version, and
  schema version;
- allows deterministic, classification, and human-approved historical or
  explicit runs to remain supported;
- requires the worker registry to contain the stored version for dispatch.

This setting does not select provider or model. Historical runs retain their
stored versions after configuration changes. The local default remains
`controlled-support-v1`.

Example safe value:

```text
controlled-support-v1
```

### `SUPPORTOPS_LLM_PROVIDER`

Explicit LLM provider adapter selection for worker composition.

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
worker provider composition
```

Purpose:

- selects one configured provider created once per worker process;
- keeps local development network-free by default;
- prevents implicit provider fallback.

OpenAI failure never selects `mock` automatically. The API process does not
create the provider.

Example safe value:

```text
mock
```

### `SUPPORTOPS_OPENAI_API_KEY`

Optional OpenAI API credential required when either
`SUPPORTOPS_LLM_PROVIDER=openai` or `SUPPORTOPS_EMBEDDING_PROVIDER=openai` is
selected.

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
- OpenAI LLM or embedding provider selection requires a nonblank value;
- mock provider selection does not require a value.

Applies to:

```text
API startup when OpenAI embeddings are selected
worker startup when OpenAI LLM is selected
indexing CLI when OpenAI embeddings are selected
```

API startup requires the key when OpenAI embeddings are selected. Worker startup
requires the key when OpenAI LLM is selected. The indexing CLI requires the key
when OpenAI embeddings are selected. The key is not required only for LLM
provider selection.

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
- preserves provider/model provenance on durable invocation records.

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
10.0
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
10.0
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

Logical LLM timing relates to the worker execution timeout as follows.

For `controlled-support-v1`:

```text
controlled_generation_slots = 6
logical_invocation_count_per_slot = 1 + llm_max_repair_attempts

controlled_budget_seconds =
    controlled_generation_slots
    × llm_request_timeout_seconds
    × logical_invocation_count_per_slot

worker_execution_timeout_seconds
    >= controlled_budget_seconds + 15

worker_lease_seconds
    >= worker_execution_timeout_seconds + 15
```

Defaults for the controlled workflow:

```text
controlled_generation_slots = 6
llm_request_timeout_seconds = 10
llm_max_repair_attempts = 1
logical_invocation_count_per_slot = 2
controlled_budget_seconds = 120
configured worker execution timeout = 135
configured worker lease = 150
```

For `ticket-classification-v1` and `deterministic-baseline-v1`, validation uses
the smaller single-generation budget:

```text
logical_invocation_count = 1 + llm_max_repair_attempts

logical_llm_budget_seconds =
    llm_request_timeout_seconds × logical_invocation_count

worker_execution_timeout_seconds
    >= logical_llm_budget_seconds + 15
```

The lease must independently exceed the worker execution timeout by at least
15 seconds.

### Provider and retry semantics

- `mock` is explicit and deterministic;
- `mock` is not a fallback;
- OpenAI uses the configured model;
- transport retry is SDK-managed;
- repair is gateway-managed;
- AgentRun retry is processor-managed;
- no automatic model switching exists;
- no cross-provider fallback exists.

### `SUPPORTOPS_EMBEDDING_PROVIDER`

Explicit embedding provider adapter selection for API semantic retrieval and
indexing composition.

Settings field:

```text
embedding_provider
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
API semantic retrieval
indexing CLI
worker controlled workflow knowledge search
```

Purpose:

- selects the embedding adapter used by API query embeddings, the indexing CLI,
  and the worker controlled `search_knowledge` path;
- keeps local development network-free by default;
- remains independent from `SUPPORTOPS_LLM_PROVIDER`;
- prevents implicit provider fallback.

OpenAI embedding failure never selects `mock` automatically. The API creates the
embedding provider at startup for semantic retrieval. The worker creates its own
embedding provider for controlled workflow retrieval. Provider construction
performs no embedding request at startup. OpenAI embedding mode means process
startup validates and constructs the client; query calls remain request-driven.

Example safe value:

```text
mock
```

### `SUPPORTOPS_EMBEDDING_MODEL`

Deployment-configured embedding model identifier.

Settings field:

```text
embedding_model
```

Type:

```text
string
```

Default:

```text
mock-hashing-embedding-v1
```

Constraints:

- the mock profile requires `mock-hashing-embedding-v1`;
- the OpenAI profile requires `text-embedding-3-small`.

Applies to:

```text
API semantic retrieval
indexing CLI embedding composition
```

Purpose:

- binds the selected embedding model into the immutable index and retrieval profiles;
- keeps model identity out of business modules;
- preserves provider and model provenance on indexed versions.

Example safe value:

```text
mock-hashing-embedding-v1
```

### `SUPPORTOPS_EMBEDDING_DIMENSIONS`

Expected dense-vector dimensionality for the selected embedding profile.

Settings field:

```text
embedding_dimensions
```

Type:

```text
integer
```

Default:

```text
64
```

Accepted range:

```text
1 through 4096
```

Constraints:

- the OpenAI profile requires `1536`.

Applies to:

```text
API semantic retrieval
indexing CLI embedding composition and Qdrant collection compatibility
```

Purpose:

- validates provider output dimensions;
- selects compatible Qdrant collection geometry;
- remains part of the immutable index and retrieval profiles.

Example safe value:

```text
64
```

### `SUPPORTOPS_EMBEDDING_REQUEST_TIMEOUT_SECONDS`

Maximum duration of one embedding provider request.

Settings field:

```text
embedding_request_timeout_seconds
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
API semantic retrieval query embeddings
indexing CLI embedding provider requests
```

Purpose:

- bounds one provider request;
- remains independent from worker execution timeout and LLM request timeout.

Example safe value:

```text
12.0
```

### `SUPPORTOPS_EMBEDDING_TRANSPORT_MAX_RETRIES`

Maximum SDK-managed transport retries within one embedding provider request.

Settings field:

```text
embedding_transport_max_retries
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
API OpenAI embedding provider configuration
indexing CLI OpenAI embedding provider configuration
```

Purpose:

- makes SDK transport retry behavior explicit;
- handles eligible transient provider transport failures;
- is not an application-level indexing or retrieval retry loop.

Example safe value:

```text
1
```

### Embedding-provider semantics

- LLM provider and embedding provider selections are independent;
- no cross-provider embedding fallback exists;
- API provider construction performs no embedding request at startup;
- query embeddings remain request-driven;
- OpenAI embeddings require `--allow-external-provider` at the indexing CLI;
- the search endpoint does not accept `--allow-external-provider`;
- the external-provider flag is rejected in mock mode;
- mock and OpenAI vectors use separate Qdrant collections;
- mock embeddings are network-free and zero-priced in the application catalog;
- mock vectors use deterministic lexical SHA-256 hashing and are not a
  semantic-quality benchmark;
- persisted index-profile compatibility is enforced on retry and retrieval;
- unknown pricing remains null rather than being treated as zero.

## AI observability configuration

AI observability is optional and application-owned. The default provider is
`noop`, which requires no credentials and performs no network access. Selecting
`langfuse` enables the optional Langfuse adapter. Langfuse is not a readiness
dependency and is not required for application correctness.

PostgreSQL remains the source of truth for durable business, audit, usage, and
estimated-cost state. LangGraph PostgreSQL checkpoints remain the source of
truth for graph continuity and pause/resume state. Qdrant remains a replaceable
retrieval projection. Langfuse receives optional derived telemetry for
operational debugging and later evaluation workflows.

The variables below configure the observability foundation, privacy policy, and
opt-in attempt-end flush behavior used by provider, retrieval, indexing, and
durable workflow instrumentation.

### `SUPPORTOPS_AI_OBSERVABILITY_PROVIDER`

Explicit observability provider adapter selection.

Settings field:

```text
ai_observability_provider
```

Type:

```text
enum string
```

Default:

```text
noop
```

Accepted values:

```text
noop
langfuse
```

Applies to:

```text
API process startup
worker process composition
knowledge-indexing CLI composition
```

Purpose:

- selects the process-scoped observability client;
- keeps local development and CI network-free by default;
- requires Langfuse credentials only when `langfuse` is selected.

Example safe value:

```text
noop
```

### `SUPPORTOPS_LANGFUSE_PUBLIC_KEY`

Optional Langfuse public key required when
`SUPPORTOPS_AI_OBSERVABILITY_PROVIDER=langfuse`.

Settings field:

```text
langfuse_public_key
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
- Langfuse provider selection requires a nonblank value;
- noop provider selection does not require a value.

Applies to:

```text
API, worker, and indexing CLI when Langfuse is selected
```

Security requirements:

- never commit a real value;
- never include it in logs;
- never include complete settings in logs;
- use managed secret storage before production deployment.

Synthetic placeholder only:

```text
pk-lf-example-public
```

### `SUPPORTOPS_LANGFUSE_SECRET_KEY`

Optional Langfuse secret key required when
`SUPPORTOPS_AI_OBSERVABILITY_PROVIDER=langfuse`.

Settings field:

```text
langfuse_secret_key
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
- Langfuse provider selection requires a nonblank value;
- noop provider selection does not require a value.

Applies to:

```text
API, worker, and indexing CLI when Langfuse is selected
```

Security requirements:

- never commit a real value;
- never include it in logs;
- never include complete settings in logs;
- use managed secret storage before production deployment.

Synthetic placeholder only:

```text
sk-lf-example-secret
```

### `SUPPORTOPS_LANGFUSE_BASE_URL`

HTTP or HTTPS base URL for Langfuse Cloud or an independently operated
compatible Langfuse deployment.

Settings field:

```text
langfuse_base_url
```

Type:

```text
HTTP or HTTPS URL
```

Default:

```text
https://cloud.langfuse.com
```

Applies to:

```text
Langfuse adapter construction when Langfuse is selected
```

Purpose:

- targets Langfuse Cloud or a compatible self-operated deployment;
- is not used when the provider is `noop`.

Langfuse is not added to the local Docker Compose stack.

Example safe value:

```text
https://cloud.langfuse.com
```

### `SUPPORTOPS_LANGFUSE_ENVIRONMENT`

Deployment environment label exported with Langfuse telemetry.

Settings field:

```text
langfuse_environment
```

Type:

```text
string
```

Default:

```text
local
```

Constraints:

- 1 through 64 characters after trimming;
- must start with an ASCII letter or digit;
- may contain ASCII letters, digits, `.`, `_`, and `-`.

Applies to:

```text
Langfuse adapter construction when Langfuse is selected
```

Example safe value:

```text
local
```

### `SUPPORTOPS_LANGFUSE_RELEASE`

Optional release identifier exported with Langfuse telemetry.

Settings field:

```text
langfuse_release
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
- maximum 128 characters;
- when provided, must start with an ASCII letter or digit and may contain
  ASCII letters, digits, `.`, `_`, `+`, and `-`.

Applies to:

```text
Langfuse adapter construction when Langfuse is selected
```

Release is optional. Omit it when no release label is available.

Example safe value:

```text
0.1.0
```

### `SUPPORTOPS_LANGFUSE_CAPTURE_MODE`

Privacy-aware export policy applied before data reaches the Langfuse SDK.

Settings field:

```text
langfuse_capture_mode
```

Type:

```text
enum string
```

Default:

```text
metadata_only
```

Accepted values:

```text
metadata_only
redacted_content
```

Applies to:

```text
Langfuse adapter privacy policy
```

Purpose:

- `metadata_only` omits business content and exports operational metadata;
- `redacted_content` accepts only structured, allowlisted fields after masking,
  truncation, and collection bounds;
- unrestricted raw-content capture is not supported.

Regex masking reduces exposure but does not prove de-identification.

Example safe value:

```text
metadata_only
```

### `SUPPORTOPS_LANGFUSE_FLUSH_AT_ATTEMPT_END`

Opt-in flag that performs one best-effort flush after each finalized
AgentRunAttempt when Langfuse or another flush-capable client is in use.

Settings field:

```text
langfuse_flush_at_attempt_end
```

Type:

```text
boolean
```

Default:

```text
false
```

Behavior:

```text
false:
use normal SDK batching

true:
perform one best-effort flush after each finalized AgentRun attempt
```

Applies to:

```text
worker AgentRun attempt finalization
```

Finalized attempt outcomes include:

```text
success
retryable failure
terminal failure
timeout
approval pause
lease-lost finalization
```

Flush semantics:

- runs after attempt observability contexts close;
- does not run per provider call, graph node, tool call, or event;
- does not run during human waiting;
- is fail-open;
- does not alter business outcome, retry scheduling, persistence, or worker
  exit behavior;
- does not replace shutdown flushing.

When the flag is `false`, the process relies on normal SDK batching and
graceful shutdown flush behavior.

Example safe value:

```text
false
```

### `SUPPORTOPS_LANGFUSE_TIMEOUT_SECONDS`

Timeout applied to Langfuse SDK client configuration.

Settings field:

```text
langfuse_timeout_seconds
```

Type:

```text
float seconds
```

Default:

```text
5.0
```

Accepted range:

```text
greater than 0 and no more than 30
```

Applies to:

```text
Langfuse SDK construction when Langfuse is selected
```

Purpose:

- bounds SDK network configuration;
- does not make Langfuse a readiness dependency;
- runtime export failures remain fail-open.

Example safe value:

```text
5.0
```

### Observability-provider semantics

- default provider is `noop`;
- Langfuse credentials are required only when `langfuse` is selected;
- Langfuse is not required in noop mode and is not required for application
  correctness;
- default capture mode is `metadata_only`;
- `redacted_content` is an explicit opt-in;
- unrestricted raw-content capture is not supported;
- release is optional;
- attempt-end flushing defaults to `false` and uses normal SDK batching;
- when attempt-end flushing is enabled, one best-effort flush runs after each
  finalized AgentRun attempt;
- timeout applies to SDK configuration only;
- Langfuse is not a readiness dependency;
- PostgreSQL remains the source of truth for durable business and usage records;
- Langfuse receives optional derived telemetry and may be incomplete after
  retries or hard process termination.

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
SUPPORTOPS_TICKET_PROCESSING_WORKFLOW_VERSION=controlled-support-v1
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
`SUPPORTOPS_EMBEDDING_*` values likewise use validated mock defaults when unset.
`SUPPORTOPS_AI_OBSERVABILITY_PROVIDER` and `SUPPORTOPS_LANGFUSE_*` values use
validated noop defaults when unset. Local development and CI require no Langfuse
credentials. The AI runtime section above matches `.env.example`, including the
empty `SUPPORTOPS_OPENAI_API_KEY=` placeholder. When overriding LLM request
timeout locally, keep worker execution timeout and lease values large enough for
the selected workflow budget and the 15-second safety margins.

## Validation behavior

Missing required values fail during settings construction.

Required variables include:

```text
SUPPORTOPS_POSTGRESQL_URL
SUPPORTOPS_QDRANT_URL
```

The shared settings model still requires `SUPPORTOPS_QDRANT_URL` for process
construction. The worker validates that shared model at startup and may
initialize Qdrant and embedding resources for controlled workflows. The worker
also converts `SUPPORTOPS_POSTGRESQL_URL` to a Psycopg-compatible checkpoint DSN
internally without logging the secret.

Examples of invalid configuration include:

- malformed PostgreSQL DSN;
- blank Qdrant URL;
- API port outside the valid range;
- worker maximum attempts outside the accepted range;
- worker lease duration that does not exceed execution timeout by at least 15 seconds;
- worker retry maximum smaller than retry base;
- approval TTL outside the accepted range;
- approval expiration batch size outside the accepted range;
- non-positive pool timeout;
- non-positive dependency health timeout;
- unsupported environment value;
- unsupported log level;
- unsupported LLM provider;
- unsupported embedding provider;
- unsupported ticket-processing workflow version;
- OpenAI LLM or embedding selection without an API key;
- unsupported AI observability provider;
- Langfuse provider selection without public or secret keys;
- unsupported Langfuse capture mode;
- invalid Langfuse base URL;
- invalid Langfuse environment or release characters;
- Langfuse timeout outside its accepted range;
- request timeout outside its accepted range;
- embedding request timeout outside its accepted range;
- transport retry count outside 0 through 2;
- embedding transport retry count outside 0 through 2;
- repair attempt count outside 0 through 1;
- logical LLM budget exceeding the worker execution timeout after the required
  15-second safety margin.

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
- Langfuse public and secret keys use a secret settings type;
- Langfuse keys are not logged and must not be committed;
- Langfuse credentials are required only when the Langfuse provider is selected;
- real Langfuse credentials require environment-appropriate secret management;
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