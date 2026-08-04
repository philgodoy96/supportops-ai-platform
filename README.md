# SupportOps AI Platform

SupportOps AI Platform is a production-minded backend and AI systems engineering project focused on reliable support operations, controlled AI orchestration, retrieval quality, human approval, observability, and evaluation.

The platform is designed as a portfolio-grade engineering system rather than a tutorial chatbot. Its architecture emphasizes clear boundaries, operational reliability, explicit trade-offs, testability, and incremental delivery.

## Project status

The repository foundation, Slice 1 workspace and ticket API, durable AgentRun scheduling, the PostgreSQL-backed worker, workspace-scoped AgentRun inspection, the application-owned LLM Gateway, durable structured ticket classification, durable logical invocation and accepted classification persistence, workspace-scoped classification and logical invocation inspection, offline deterministic classification evaluation, workspace-scoped immutable knowledge-document versioning, explicit profiled knowledge indexing, active-version semantic knowledge retrieval, the controlled support workflow with LangGraph orchestration, read-only tools, recommendation persistence, controlled support inspection, and the optional application-owned AI observability foundation with default no-op mode and optional Langfuse adapter are implemented.

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
- configured `controlled-support-v1` scheduling for newly accepted tickets as the local default ticket-processing workflow;
- exact registered workflow versions `deterministic-baseline-v1`, `ticket-classification-v1`, and `controlled-support-v1`;
- durable AgentRun and AgentRunAttempt persistence;
- PostgreSQL claiming with `FOR UPDATE SKIP LOCKED`;
- attempt history, leases, and lease-token fencing;
- bounded retries and expired lease recovery;
- exact versioned workflow executor registry;
- deterministic baseline, classification, and controlled-support workflow execution outside database transactions;
- LangGraph bounded orchestration inside AgentRun with PostgreSQL-backed checkpoint resume;
- process-scoped worker checkpoint, embedding, and Qdrant resources for controlled-support-v1;
- read-only controlled tools `search_knowledge` and `lookup_service_status`;
- durable `AgentToolCall` audits with post-commit/pre-checkpoint recovery;
- authoritative PostgreSQL reconstruction of knowledge observations;
- durable `SupportRecommendation` and `SupportRecommendationCitation` records;
- controlled support inspection endpoint over persisted business records;
- attempt-scoped tool and invocation ordering;
- historical estimated-cost aggregation from persisted invocations;
- exact Alembic exclusion of framework-owned LangGraph checkpoint tables;
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
- professional architecture and development documentation;
- workspace-scoped knowledge documents with optional workspace-local external references;
- immutable plain-text and Markdown document versions;
- deterministic source normalization and SHA-256 content hashing;
- PostgreSQL-backed authoritative document, version, and chunk records;
- concurrency-safe document version allocation under a document row lock;
- database-enforced duplicate-content rejection within one document;
- explicit `pending`, `failed`, and `ready` version states;
- explicit activation that permits only ready versions;
- opaque cursor pagination for document and version listings;
- source content exposure only through the workspace-scoped version detail route;
- cross-workspace document and version access that resolves as `404`;
- reversible migrations and concurrency-sensitive integration coverage for the knowledge-document domain;
- deterministic markdown-token chunking with `cl100k_base` tokenization;
- application-owned embedding contracts with mock and OpenAI providers;
- versioned embedding usage and estimated-cost provenance;
- Qdrant collection bootstrap, compatibility validation, and deterministic point upsert;
- explicit `supportops-index-knowledge` indexing CLI;
- indexing idempotency, partial-projection recovery, and ready-state verification;
- unit and integration coverage for chunking, embeddings, vector store, and indexing orchestration;
- active-version semantic knowledge retrieval through a workspace-scoped search API;
- query embedding composition with the process-scoped API embedding provider;
- exact Qdrant active-target filtering by workspace and document/version pairs;
- authoritative PostgreSQL hydration of candidate chunk content and token counts;
- stable citation metadata for document, version, and chunk provenance;
- workspace isolation and stale-projection discard coverage in retrieval tests;
- process-scoped API embedding provider and immutable retrieval profile lifecycle;
- application-owned AI observability abstraction with default no-op mode;
- optional Langfuse adapter behind the application-owned boundary;
- deterministic AgentRun and ticket trace-identity foundation;
- metadata-only default capture with redacted-content opt-in;
- fail-open observability client lifecycle for API, worker, and indexing CLI.

Workspace scoping is a data ownership boundary. It is not authentication or authorization, and it is not authenticated tenant isolation.

Ticket acceptance and asynchronous processing success are separate outcomes. Ticket intake schedules the configured workflow version, with local default `controlled-support-v1`. Ticket status remains `open`. AgentRun status reports workflow execution. An accepted `TicketClassification` records the model interpretation and does not mutate Ticket status. The controlled workflow may execute read-only tools and persist a support recommendation without mutating the ticket or executing write-capable actions. The deterministic baseline and direct classification workflows remain registered for historical or explicitly scheduled runs.

Inspection endpoints report current persisted AgentRun, classification, logical invocation, tool-call, recommendation, and controlled-support aggregate state. They do not guarantee future completion, and they do not mutate retries, leases, or lifecycle transitions. Inspection is read-only and does not deserialize LangGraph checkpoint state. Evaluation measures the same prompt and schema boundary offline and does not write to PostgreSQL or Qdrant. Slice 7 introduces the application-owned AI observability foundation. Detailed generation, embedding, retrieval, tool, approval, and workflow instrumentation is delivered incrementally in subsequent Slice 7 pull requests. PostgreSQL remains authoritative for business, workflow, audit, usage, and estimated-cost records.

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

The API owns HTTP acceptance, transactional AgentRun scheduling, semantic knowledge retrieval, and controlled support inspection. For public retrieval, the API process owns Qdrant client resources and a process-scoped embedding provider with an immutable retrieval profile. The worker owns recovery, claim, execution, fenced outcome persistence, and the process-scoped LLM, checkpoint, embedding, and Qdrant resources required by `controlled-support-v1`. PostgreSQL is the durable work queue and transactional source of truth for business records, and it also stores framework-owned LangGraph checkpoint tables. The indexing CLI remains a separate one-shot process.

PostgreSQL owns source content, authoritative chunks, indexing status, immutable index profile, active-version selection, embedding usage, estimated cost, failure provenance, retrieval evidence content, tool audits, recommendations, and citations. LangGraph checkpoint schema remains framework-owned and is excluded from application Alembic ownership. Qdrant owns only the rebuildable vector projection used for candidate search by both public retrieval and controlled workflow knowledge search. Retrieval data must remain reproducible from authoritative source content rather than becoming an independent system of record.

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
- LangGraph;
- Docker Compose;
- pytest;
- pytest-asyncio;
- HTTPX;
- Ruff;
- mypy;
- uv;
- tiktoken;
- GitHub Actions.

Detailed architecture documentation is maintained under [`docs/architecture`](docs/architecture).

Accepted architectural decisions are recorded under [`docs/decisions`](docs/decisions).

## Current foundation capabilities

### Application runtime

The repository provides:

- explicit FastAPI application construction;
- OpenAPI project metadata;
- process-owned PostgreSQL and Qdrant resources;
- process-scoped embedding provider and immutable retrieval profile for semantic search;
- process-scoped worker LLM, checkpoint, embedding, and Qdrant composition for controlled-support-v1;
- process-scoped observability client composition for API, worker, and indexing CLI;
- centralized startup and shutdown lifecycle with partial startup cleanup and independent shutdown cleanup;
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

Langfuse is not a readiness dependency.

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
- registered workspace, ticket, AgentRun, invocation, classification, and knowledge-document persistence models;
- reversible migrations that create `workspaces`, `tickets`, `agent_runs`, `agent_run_attempts`, `llm_invocations`, `ticket_classifications`, `knowledge_documents`, `knowledge_document_versions`, and `knowledge_document_chunks`;
- composite ownership constraints and accepted-invocation provenance for classification records;
- composite document-version and chunk-profile constraints for authoritative knowledge content;
- database triggers that permit activation only for ready versions and prevent ready-version rewrites.

### Workspace and ticket persistence

The first business modules provide:

- frozen Workspace and Ticket domain entities with validated invariants;
- SQLAlchemy records that own table definitions, constraints, indexes, and mapping;
- repository protocols with workspace-scoped ticket access;
- async SQLAlchemy repository implementations that flush without committing;
- named uniqueness constraints for workspace slugs and workspace-scoped external references;
- a minimal SQLAlchemy transaction adapter for application-owned boundaries;
- repository integration coverage, including concurrency-sensitive duplicate external-reference insertion.

### Versioned knowledge documents

The `knowledge_documents` module provides PostgreSQL-authoritative source registration and version rollout. Document registration and version creation make no embedding-provider or Qdrant calls.

Implemented behavior includes:

- workspace-owned document identities;
- optional external references unique within a workspace;
- immutable `text/plain` and `text/markdown` versions;
- deterministic line-ending normalization and UTF-8 SHA-256 content hashing;
- duplicate normalized content rejection within one document;
- authoritative source content stored on `DocumentVersion`;
- authoritative deterministic chunk records prepared for the indexing phase;
- `pending`, `failed`, and `ready` indexing lifecycle states;
- immutable persisted indexing-profile fields once assigned;
- concurrency-safe next-version allocation under a document row lock;
- explicit activation through an active-version pointer owned by PostgreSQL;
- database enforcement that only ready, owned versions can become active;
- database enforcement that ready versions cannot be rewritten;
- workspace-scoped document and version APIs with opaque keyset pagination;
- source content returned only from the version detail endpoint;
- tenant-safe `404` behavior for missing and cross-workspace documents and versions.

Current document lifecycle:

```text
create document and version 1
→ version remains pending
→ explicit indexing command produces chunks and vectors
→ successful indexing marks the version ready
→ explicit activation changes the document active-version pointer
```

Ready means the verified indexing projection completed. Ready does not mean active. Activation remains a separate explicit API operation.

The domain and transaction model are documented in [`docs/architecture/knowledge-documents.md`](docs/architecture/knowledge-documents.md).

### Knowledge indexing

The `supportops.knowledge_index` package provides explicit, profiled indexing from immutable PostgreSQL document versions to a rebuildable Qdrant projection.

Implemented behavior includes:

- deterministic `markdown-token` / `v1` chunking with `cl100k_base`, 500-token maximum, and 75-token overlap;
- Markdown-aware heading, paragraph, and fenced-code handling, plus plain-text support without Markdown interpretation;
- deterministic chunk IDs and content hashes persisted as authoritative PostgreSQL records;
- application-owned embedding contracts with ordered batches, dimensions, usage, provider, model, and request IDs;
- mock provider `mock-hashing-embedding-v1` at 64 dimensions using deterministic lexical SHA-256 hashing;
- OpenAI provider `text-embedding-3-small` at 1536 dimensions;
- versioned Decimal embedding pricing with known-zero mock cost and cataloged OpenAI pricing;
- separate mock and OpenAI Qdrant collections with named vector `dense`, cosine distance, and ownership payload indexes;
- Qdrant payloads that exclude source and chunk content;
- deterministic chunk IDs used as Qdrant point IDs;
- bounded upserts with `wait=True` and exact version projection-count verification before ready;
- short database transactions that do not span tokenization, provider calls, or Qdrant calls;
- failed compatible-version retry, ready-version no-op reruns, and partial-projection recovery;
- the `supportops-index-knowledge` CLI for `ensure-collection` and `index-version`.

Indexing does not activate a document version. Mock embeddings support local pipeline testing and are not a semantic-quality benchmark. OpenAI embeddings require explicit `--allow-external-provider` acknowledgement at the CLI.

Indexing architecture is documented in [`docs/architecture/knowledge-indexing.md`](docs/architecture/knowledge-indexing.md). The explicit profiled indexing decision is recorded in [`docs/decisions/0008-use-explicit-profiled-knowledge-indexing.md`](docs/decisions/0008-use-explicit-profiled-knowledge-indexing.md).

### Semantic knowledge retrieval

The `supportops.knowledge_retrieval` package provides workspace-scoped semantic search over explicitly active ready document versions.

Implemented behavior includes:

- `POST /api/v1/workspaces/{workspace_id}/knowledge/search`;
- PostgreSQL resolution of only the requested workspace's active ready versions whose complete persisted profile equals the runtime retrieval profile;
- exclusion of ready but inactive versions and of pending or failed versions;
- empty eligible scope returning HTTP `200` with `searched_version_count` `0`, empty evidence, and no embedding or Qdrant call;
- one normalized query embedding under operation `knowledge_query` with provider, model, dimensions, and count revalidation;
- Qdrant `query_points` against the persisted named vector with exact workspace plus document/version pair filters, selected metadata payload only, and no returned vectors or source content;
- candidate oversampling at `min(100, max(top_k, top_k * 4))`;
- unique candidate IDs bulk-hydrated from PostgreSQL for authoritative content and token counts;
- provenance validation that discards missing, stale, malformed, inactive, cross-workspace, and duplicate candidates;
- deterministic ranking by score descending with `chunk_id` ascending tie-break and contiguous one-based final ranks;
- stable citations identifying workspace, document, version, chunk, section path, and media type;
- process-scoped embedding provider, Qdrant client, PostgreSQL engine/session factory, and immutable retrieval profile, with request-scoped session and service composition;
- partial startup cleanup and independent shutdown cleanup;
- expected embedding and vector-store failures mapped to HTTP `503` with code `knowledge_retrieval_unavailable` and a sanitized message.

The evidence response contains ranked authoritative chunks and citations. It does not contain a generated answer. Cross-workspace document filters return empty evidence without disclosing foreign ownership.

Retrieval architecture is documented in [`docs/architecture/semantic-knowledge-retrieval.md`](docs/architecture/semantic-knowledge-retrieval.md). The PostgreSQL hydration decision is recorded in [`docs/decisions/0009-hydrate-retrieval-evidence-from-postgresql.md`](docs/decisions/0009-hydrate-retrieval-evidence-from-postgresql.md).

### Optional application-owned AI observability

Slice 7 introduces the application-owned AI observability foundation. Detailed generation, embedding, retrieval, tool, approval, and workflow instrumentation is delivered incrementally in subsequent Slice 7 pull requests.

Implemented foundation behavior includes:

- provider-independent observability contracts for traces, observations, events, usage, and cost;
- default no-op adapter with no credentials and no network access;
- optional Langfuse adapter enabled only through validated configuration;
- deterministic AgentRun and ticket trace-identity foundation;
- metadata-only default capture with redacted-content opt-in;
- no unrestricted raw-content capture mode;
- fail-open client lifecycle for the API, worker, and indexing CLI;
- privacy enforcement before data reaches the Langfuse SDK.

PostgreSQL remains authoritative for business, workflow, audit, usage, and estimated-cost records. Langfuse is a derived observability projection and is not a readiness dependency.

The optional Langfuse decision is recorded in [`docs/decisions/0013-use-optional-application-owned-langfuse-observability.md`](docs/decisions/0013-use-optional-application-owned-langfuse-observability.md).

### Durable AgentRun scheduling and PostgreSQL worker

Ticket intake schedules the configured workflow version in the same application-owned transaction that creates the ticket. The local default is `controlled-support-v1`. The HTTP request returns the existing Ticket response after that transaction commits and does not execute the workflow or call the model.

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
- a versioned executor registry with exact workflow name and version dispatch for three registered versions;
- deterministic baseline and direct classification support for historical or explicitly scheduled runs;
- controlled-support-v1 LangGraph orchestration with checkpoint resume inside the AgentRun boundary;
- provider, embedding, tool, and Qdrant calls outside database transactions;
- separate fenced transactions for invocation, classification, tool audit, recommendation, and AgentRun completion;
- cooperative SIGINT and SIGTERM shutdown with controlled runtime, provider, and engine cleanup.

The ticket creation response remains the Ticket response shape and does not report recommendation completion.

After ticket acceptance, the client can inspect the persisted AgentRun and its attempt history through workspace-scoped read-only endpoints when the AgentRun identifier is otherwise known. Controlled support runs can also be inspected through the aggregate controlled support inspection endpoint. Inspection exposes status, retry budget, workflow identity, safe error metadata, ordered attempt outcomes, tool calls, invocations, usage, and recommendations where present. It does not expose lease ownership, lease tokens, lease expiry, ingestion request IDs, execution request IDs, or checkpoint blobs.

Scheduling and worker handoff behavior are documented in [`docs/architecture/agent-run-scheduling.md`](docs/architecture/agent-run-scheduling.md) and [`docs/architecture/runtime-topology.md`](docs/architecture/runtime-topology.md). Controlled workflow behavior is documented in [`docs/architecture/controlled-support-workflow.md`](docs/architecture/controlled-support-workflow.md).

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

Gateway architecture is documented in [`docs/architecture/llm-gateway.md`](docs/architecture/llm-gateway.md). Durable classification behavior is documented in [`docs/architecture/ticket-classification.md`](docs/architecture/ticket-classification.md). Classification inspection and evaluation are documented in [`docs/architecture/classification-evaluation.md`](docs/architecture/classification-evaluation.md). Versioned knowledge-document ownership and rollout are documented in [`docs/architecture/knowledge-documents.md`](docs/architecture/knowledge-documents.md). Explicit knowledge indexing is documented in [`docs/architecture/knowledge-indexing.md`](docs/architecture/knowledge-indexing.md). Semantic knowledge retrieval is documented in [`docs/architecture/semantic-knowledge-retrieval.md`](docs/architecture/semantic-knowledge-retrieval.md). Controlled support workflow behavior is documented in [`docs/architecture/controlled-support-workflow.md`](docs/architecture/controlled-support-workflow.md). AgentRun and LangGraph durability ownership is recorded in [`docs/decisions/0010-separate-agent-run-and-langgraph-durability.md`](docs/decisions/0010-separate-agent-run-and-langgraph-durability.md). Framework-owned checkpoint schema ownership is recorded in [`docs/decisions/0011-treat-langgraph-checkpoints-as-framework-owned-schema.md`](docs/decisions/0011-treat-langgraph-checkpoints-as-framework-owned-schema.md).

### Controlled support workflow and inspection

The platform implements `controlled-support-v1` as the default ticket-processing workflow. AgentRun owns outer durability. LangGraph owns bounded inner orchestration with PostgreSQL checkpoint resume. The worker process owns LLM, checkpoint, embedding, and Qdrant resources required by the controlled workflow. The API independently owns retrieval resources for the public semantic search endpoint.

Implemented controlled-workflow behavior includes:

- classification reuse through the existing durable classification boundary;
- read-only tools `search_knowledge` and `lookup_service_status`;
- durable `AgentToolCall` audits under lease fencing;
- post-commit/pre-checkpoint recovery before requesting another model decision;
- authoritative PostgreSQL reconstruction of knowledge observations;
- durable `SupportRecommendation` and ordered `SupportRecommendationCitation` records;
- outer AgentRun success only after a persisted recommendation identity exists;
- attempt-scoped tool and invocation ordering;
- historical estimated-cost aggregation from persisted invocations;
- exact Alembic exclusion of framework-owned checkpoint tables.

Implemented inspection route:

```text
GET /api/v1/workspaces/{workspace_id}/tickets/{ticket_id}/agent-runs/{agent_run_id}/inspection
```

The inspection endpoint supports `controlled-support-v1` only. It reads persisted business records and does not deserialize LangGraph checkpoint state. Queued, running, retrying, and failed workflows may return valid partial inspection views.

### Workspace, ticket, AgentRun, and knowledge document API

The platform exposes versioned business routes under `/api/v1`:

- workspace creation and retrieval;
- workspace-scoped ticket creation, retrieval, and listing;
- workspace-scoped AgentRun retrieval;
- workspace-scoped AgentRunAttempt history listing;
- workspace-scoped classification detail and ticket classification history;
- workspace-scoped AgentRun logical invocation history;
- controlled support aggregate inspection for `controlled-support-v1` runs;
- workspace-scoped knowledge-document creation, retrieval, and listing;
- immutable document-version creation, retrieval, and listing;
- explicit ready-version activation;
- workspace-scoped semantic knowledge search returning ranked evidence;
- opaque cursor pagination for ticket, classification, document, and version listings;
- stable expected-error responses for missing resources, conflicts, invalid cursors, and temporary retrieval unavailability;
- persistence of request and correlation identifiers on accepted tickets;
- cross-workspace retrieval that returns the same `404` contract as a missing ticket, AgentRun, classification, document, or document version.

Semantic search returns evidence only. The response contains no generated answer field.

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
- bounded read-only connectivity checks;
- compatible named-vector knowledge collections with cosine distance;
- separate mock and OpenAI collections selected by immutable index profile;
- ownership payload indexes for `workspace_id`, `document_id`, and `document_version_id`;
- deterministic chunk IDs as point IDs;
- payloads that exclude source and chunk content;
- bounded upserts with exact version projection-count verification;
- filtered candidate search for semantic retrieval against the named dense vector.

Qdrant stores only the rebuildable vector projection. PostgreSQL remains authoritative for source content, chunks, status, profile, usage, cost, failure provenance, active-version selection, and retrieval evidence content. The API uses Qdrant for semantic candidate search and hydrates authoritative chunk content from PostgreSQL. The worker uses Qdrant for controlled `search_knowledge` candidate search and likewise hydrates authoritative content from PostgreSQL. Deterministic baseline and direct classification workflows do not use Qdrant. The HTTP API does not perform indexing.

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
- Alembic upgrade, downgrade, and metadata-parity coverage for classification and knowledge-document tables;
- knowledge-document domain, persistence mapping, repository, application-service, pagination, and API coverage;
- PostgreSQL concurrency tests for distinct and duplicate normalized document-version creation;
- database-trigger coverage for ready-only activation and ready-version immutability;
- deterministic chunking and tokenizer adapter coverage;
- embedding contract, pricing, mock, OpenAI fake, and error coverage;
- Qdrant collection, payload-index, upsert, and exact-count adapter coverage;
- indexing orchestration, composition, CLI, idempotency, and recovery coverage;
- knowledge-retrieval contracts, provenance validation, ranking, schemas, route, and lifespan coverage;
- knowledge-retrieval integration coverage against real PostgreSQL and Qdrant with mock embeddings;
- AI observability settings, models, identity, privacy, no-op adapter, and Langfuse fake-client coverage;
- API, worker, and indexing observability lifecycle coverage;
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

Normal unit and integration tests do not require an OpenAI API key, Langfuse credentials, or paid external requests. OpenAI evaluation remains an explicit manual operation.

## Planned platform modules

The repository already includes bounded `workspaces`, `tickets`, `agent_runs`, `ticket_classifications`, `knowledge_documents`, `support_recommendations`, and `controlled_support_inspection` modules, a `supportops.worker` process entry point, the cross-cutting `supportops.ai` foundation, the optional `supportops.observability` foundation, the offline `supportops.evaluation.ticket_classification` package, the explicit `supportops.knowledge_index` indexing package, the `supportops.knowledge_retrieval` semantic search package, the `supportops.agent_graph` controlled orchestration package, and the `supportops.agent_tools` bounded tool registry. Workspace and ticket modules expose domain entities, application services, repository contracts, PostgreSQL persistence, and versioned HTTP APIs. The `agent_runs` module provides durable scheduling, claiming, versioned executor dispatch, execution, retry, recovery, and workspace-scoped inspection foundations. The `supportops.modules.ticket_classifications` module is implemented for durable classification execution, persistence, and read-only inspection. The `supportops.modules.support_recommendations` module persists recommendations and citations. The `supportops.modules.controlled_support_inspection` module exposes the read-only controlled support inspection aggregate. The `supportops.ai` package owns provider-independent LLM contracts, provider adapters, prompt definitions, structured schemas, repair behavior, estimated-cost calculation, and embedding contracts. The `supportops.observability` package owns provider-independent observability contracts, deterministic identity, privacy policy, no-op and Langfuse adapters, and process-scoped client composition. The `supportops.knowledge_index` package owns deterministic chunking, Qdrant collection and point adapters, indexing orchestration, and the operator CLI. The `supportops.knowledge_retrieval` package owns active-version resolution, query embedding composition, Qdrant candidate search, PostgreSQL hydration, provenance validation, deterministic ranking, citations, and the workspace-scoped search route.

Future modules or extensions will introduce:

- reranking of retrieval candidates;
- retrieval evaluation datasets and deterministic scoring;
- evidence-driven prompt version 2;
- prompt regression comparison across versions;
- scheduled evaluation and evaluation history persistence;
- multi-profile score fusion;
- write-capable tools;
- approval workflows;
- detailed AI workflow instrumentation beyond the current observability foundation.

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
│   │   ├── controlled-support-workflow.md
│   │   ├── knowledge-documents.md
│   │   ├── knowledge-indexing.md
│   │   ├── llm-gateway.md
│   │   ├── overview.md
│   │   ├── runtime-topology.md
│   │   ├── semantic-knowledge-retrieval.md
│   │   ├── ticket-classification.md
│   │   └── workspace-data-boundary.md
│   ├── decisions/
│   │   ├── 0001-use-a-modular-monolith.md
│   │   ├── 0002-use-postgresql-as-the-source-of-truth.md
│   │   ├── 0003-use-qdrant-as-a-rebuildable-retrieval-index.md
│   │   ├── 0004-use-a-postgresql-backed-worker-model.md
│   │   ├── 0005-keep-ai-observability-behind-an-adapter.md
│   │   ├── 0006-establish-workspace-scoped-data-ownership.md
│   │   ├── 0007-use-an-application-owned-llm-gateway.md
│   │   ├── 0008-use-explicit-profiled-knowledge-indexing.md
│   │   ├── 0009-hydrate-retrieval-evidence-from-postgresql.md
│   │   ├── 0010-separate-agent-run-and-langgraph-durability.md
│   │   ├── 0011-treat-langgraph-checkpoints-as-framework-owned-schema.md
│   │   └── 0013-use-optional-application-owned-langfuse-observability.md
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
│       │   ├── embeddings/
│       │   ├── gateway/
│       │   ├── pricing/
│       │   ├── prompts/
│       │   ├── providers/
│       │   └── schemas/
│       ├── agent_graph/
│       ├── agent_tools/
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
│       ├── knowledge_index/
│       │   ├── chunking/
│       │   ├── indexing/
│       │   ├── vector_store/
│       │   ├── composition.py
│       │   └── cli.py
│       ├── knowledge_retrieval/
│       │   ├── api/
│       │   ├── contracts.py
│       │   ├── postgresql.py
│       │   ├── qdrant.py
│       │   └── service.py
│       ├── observability/
│       │   ├── composition.py
│       │   ├── contracts.py
│       │   ├── identity.py
│       │   ├── langfuse.py
│       │   ├── models.py
│       │   ├── noop.py
│       │   └── privacy.py
│       ├── modules/
│       │   ├── agent_runs/
│       │   │   ├── api/
│       │   │   ├── application/
│       │   │   ├── domain/
│       │   │   └── infrastructure/
│       │   ├── controlled_support_inspection/
│       │   ├── knowledge_documents/
│       │   │   ├── api/
│       │   │   ├── application/
│       │   │   ├── domain/
│       │   │   └── infrastructure/
│       │   ├── support_recommendations/
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
│   │   ├── knowledge_index/
│   │   ├── knowledge_retrieval/
│   │   └── ...
│   └── unit/
│       ├── ai/
│       │   └── embeddings/
│       ├── knowledge_index/
│       ├── knowledge_retrieval/
│       ├── observability/
│       └── ...
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

Business modules are introduced when they have concrete responsibilities. The current `workspaces`, `tickets`, `agent_runs`, `ticket_classifications`, `knowledge_documents`, `support_recommendations`, and `controlled_support_inspection` modules include domain, application, and infrastructure layers as required. The `agent_runs`, `ticket_classifications`, and `controlled_support_inspection` modules include read-only inspection routes. Cross-module ticket intake and AgentRun inspection composition live under `supportops.application`. The worker process entry point and process-scoped LLM, checkpoint, embedding, and Qdrant composition live under `supportops.worker`. The `supportops.ai` package owns provider-independent LLM and embedding contracts, provider adapters, prompt definitions, structured schemas, repair behavior, and estimated-cost calculation. It is not a generic orchestration framework. The `supportops.observability` package owns provider-independent observability contracts, deterministic identity, privacy policy, no-op and Langfuse adapters, and process-scoped client composition. The `supportops.agent_graph` package owns bounded LangGraph orchestration for controlled support. The `supportops.agent_tools` package owns the read-only controlled tool registry. The `supportops.knowledge_index` package owns deterministic chunking, Qdrant adapters, indexing orchestration, and the operator CLI. The `supportops.knowledge_retrieval` package owns semantic search contracts, adapters, orchestration, and the workspace-scoped HTTP search route. The `supportops.evaluation.ticket_classification` package owns offline datasets, prediction artifacts, deterministic metrics, and the evaluation CLI. Alembic migrations create workspace, ticket, AgentRun, invocation, classification, tool-call, recommendation, and versioned knowledge-document tables. Framework-owned LangGraph checkpoint tables are created by checkpointer setup and are excluded by exact name from Alembic comparison. Versioned evaluation datasets remain committed under `evals/`. Generated evaluation outputs belong under ignored `artifacts/`.

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

### Index a knowledge document version

Ensure the profile collection exists, then index a specific pending or failed version:

```powershell
uv run supportops-index-knowledge ensure-collection

uv run supportops-index-knowledge index-version `
  --workspace-id "<workspace-id>" `
  --document-id "<document-id>" `
  --document-version-id "<document-version-id>"
```

Use the UUIDs returned by the knowledge-document API. Do not execute the literal placeholders. Default mock mode is network-free. Successful indexing returns `ready` but does not activate the version. Activate separately through the HTTP API. A ready-version rerun is a no-op.

OpenAI embeddings require process configuration and explicit permission:

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

Do not place a real API key in documentation or committed files.

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
SUPPORTOPS_EMBEDDING_PROVIDER
SUPPORTOPS_EMBEDDING_MODEL
SUPPORTOPS_EMBEDDING_DIMENSIONS
SUPPORTOPS_EMBEDDING_REQUEST_TIMEOUT_SECONDS
SUPPORTOPS_EMBEDDING_TRANSPORT_MAX_RETRIES
```

The local default LLM and embedding providers are both `mock`. LLM provider and embedding provider selections are independent. The API creates the configured embedding provider at startup for semantic retrieval and does not create the LLM provider. The worker creates the configured LLM provider and, for controlled workflows, also creates the embedding provider, checkpoint runtime, and Qdrant client. The indexing CLI creates its own embedding provider. `SUPPORTOPS_OPENAI_API_KEY` is shared by the OpenAI generation and embedding adapters and is required when either OpenAI adapter is selected. `SUPPORTOPS_TICKET_PROCESSING_WORKFLOW_VERSION` controls newly scheduled runs; the local default is `controlled-support-v1`. Request timeout, logical repair budget, worker timeout, and lease margins are validated at startup. Provider transport retry, gateway repair, and AgentRun retry are separate layers. Embedding request timeout is independent from worker execution timeout. Query embeddings remain request-driven after API startup constructs the client. The worker converts `SUPPORTOPS_POSTGRESQL_URL` internally to a Psycopg-compatible DSN for the checkpoint runtime without logging the secret.

Optional AI observability selection uses:

```text
SUPPORTOPS_AI_OBSERVABILITY_PROVIDER
SUPPORTOPS_LANGFUSE_PUBLIC_KEY
SUPPORTOPS_LANGFUSE_SECRET_KEY
SUPPORTOPS_LANGFUSE_BASE_URL
SUPPORTOPS_LANGFUSE_ENVIRONMENT
SUPPORTOPS_LANGFUSE_RELEASE
SUPPORTOPS_LANGFUSE_CAPTURE_MODE
SUPPORTOPS_LANGFUSE_FLUSH_AT_ATTEMPT_END
SUPPORTOPS_LANGFUSE_TIMEOUT_SECONDS
```

The local default observability provider is `noop`. Langfuse credentials are required only when `langfuse` is selected. Default capture mode is `metadata_only`. Langfuse is not a readiness dependency.

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

The current head creates the `workspaces`, `tickets`, `agent_runs`, `agent_run_attempts`, `llm_invocations`, `ticket_classifications`, `agent_tool_calls`, `support_recommendations`, `support_recommendation_citations`, and versioned knowledge-document tables. Framework-owned LangGraph checkpoint tables are created by checkpointer setup and are excluded by exact name from Alembic comparison. Downgrade commands must run only against the local development or test database.

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
- [Use explicit, profiled knowledge indexing](docs/decisions/0008-use-explicit-profiled-knowledge-indexing.md)
- [Hydrate retrieval evidence from PostgreSQL](docs/decisions/0009-hydrate-retrieval-evidence-from-postgresql.md)
- [Separate AgentRun and LangGraph durability](docs/decisions/0010-separate-agent-run-and-langgraph-durability.md)
- [Treat LangGraph checkpoints as framework-owned schema](docs/decisions/0011-treat-langgraph-checkpoints-as-framework-owned-schema.md)
- [Use optional application-owned Langfuse observability](docs/decisions/0013-use-optional-application-owned-langfuse-observability.md)

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
- configured `controlled-support-v1` scheduling and execution as the local default;
- registered `deterministic-baseline-v1` and `ticket-classification-v1` retention;
- process-scoped provider, Gateway, checkpoint, embedding, and Qdrant composition;
- separate worker process with cooperative shutdown;
- workspace-scoped AgentRun inspection;
- workspace-scoped AgentRunAttempt history inspection;
- controlled support aggregate inspection;
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

### Retrieval foundation

Implemented:

- workspace-scoped knowledge-document identities;
- immutable plain-text and Markdown versions;
- deterministic source normalization and content hashing;
- duplicate normalized content protection;
- concurrency-safe version numbering;
- PostgreSQL-authoritative source content;
- deterministic token-aware chunk generation;
- authoritative PostgreSQL chunk persistence;
- mock and OpenAI embedding providers;
- embedding usage and cost accounting;
- Qdrant collection bootstrap and compatibility validation;
- deterministic vector-point indexing;
- exact projection-count verification before ready;
- explicit `supportops-index-knowledge` CLI;
- explicit ready-state and active-version separation;
- workspace-scoped document and version APIs;
- stable version detail access for source inspection;
- active-version semantic retrieval;
- query embeddings for retrieval requests;
- authoritative PostgreSQL chunk hydration for retrieved candidates;
- stable retrieval citations;
- process-scoped API embedding provider lifecycle for semantic search.

Planned:

- reranking;
- retrieval quality datasets and deterministic scoring;
- multi-profile score fusion.

### Controlled orchestration

Implemented:

- LangGraph bounded orchestration inside AgentRun;
- PostgreSQL-backed checkpoint resume;
- registered read-only tools `search_knowledge` and `lookup_service_status`;
- durable tool-call audits and recommendation persistence;
- controlled support inspection over business records;
- AgentRun outer durability with LangGraph inner orchestration.

Planned:

- write-capable tools;
- approval boundaries;
- failure recovery beyond the current post-commit/pre-checkpoint and checkpoint resume model for write-side effects.

### Observability and evaluation

Implemented:

- token usage and estimated-cost persistence with durable invocation provenance;
- prompt `ticket-classification` version 1;
- versioned synthetic classification dataset;
- deterministic classification evaluator;
- offline scoring;
- opt-in external-provider evaluation CLI;
- canonical dataset, prediction, and report provenance;
- application-owned AI observability abstraction;
- default no-op observability mode;
- optional Langfuse adapter;
- deterministic AgentRun and ticket trace-identity foundation;
- metadata-only default capture with redacted-content opt-in;
- fail-open observability client lifecycle for API, worker, and indexing CLI.

Slice 7 introduces the application-owned AI observability foundation. Detailed generation, embedding, retrieval, tool, approval, and workflow instrumentation is delivered incrementally in subsequent Slice 7 pull requests.

Planned:

- operational cost reporting and invoice reconciliation;
- detailed runtime AI workflow tracing instrumentation;
- opt-in live Langfuse smoke validation;
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
- automated and high-volume document ingestion;
- automatic indexing scheduling;
- automatic version activation;
- reranking;
- multi-profile score fusion;
- write-capable tools;
- human approval workflows;
- detailed runtime AI workflow tracing beyond the current observability foundation;
- opt-in live Langfuse smoke validation;
- Phoenix integration;
- RAGAS evaluation;
- prompt regression comparison across versions;
- retrieval and generation evaluation beyond structured classification;
- OpenTelemetry;
- Prometheus and Grafana;
- frontend applications;
- cloud deployment;
- infrastructure as code;
- Kubernetes.

Workspace scoping establishes data ownership. It is not authentication or authorization, and it does not establish caller identity or secure multi-tenancy.

Durable AgentRun scheduling and the PostgreSQL worker are implemented. Redis, Celery, Kafka, and SQS remain intentionally deferred because PostgreSQL already provides transactional durability and adequate local and portfolio scope for this phase. An external queue or outbox is not required for the current worker model.

The application-owned LLM Gateway, durable ticket-classification workflow, controlled-support-v1 workflow, classification inspection, controlled support inspection, offline evaluation, versioned knowledge documents, explicit profiled knowledge indexing, active-version semantic knowledge retrieval, and the optional application-owned AI observability foundation are implemented. Evidence-driven prompt version 2, prompt regression comparison, scheduled evaluation, evaluation history persistence, cross-provider fallback, operational cost reporting, RAGAS, reranking, retrieval evaluation, and detailed runtime AI workflow instrumentation remain intentionally separated into later delivery boundaries.

The architecture keeps room for these capabilities without introducing dependencies or abstractions before they have concrete responsibilities.

## Documentation

- [Architecture overview](docs/architecture/overview.md)
- [Runtime topology](docs/architecture/runtime-topology.md)
- [Transactional AgentRun scheduling](docs/architecture/agent-run-scheduling.md)
- [Controlled support workflow](docs/architecture/controlled-support-workflow.md)
- [Application-owned LLM Gateway](docs/architecture/llm-gateway.md)
- [Durable ticket classification](docs/architecture/ticket-classification.md)
- [Classification inspection and evaluation](docs/architecture/classification-evaluation.md)
- [Versioned knowledge documents](docs/architecture/knowledge-documents.md)
- [Knowledge indexing pipeline](docs/architecture/knowledge-indexing.md)
- [Semantic knowledge retrieval](docs/architecture/semantic-knowledge-retrieval.md)
- [Workspace-scoped data ownership](docs/architecture/workspace-data-boundary.md)
- [Architecture decision records](docs/decisions)
- [Local setup](docs/development/local-setup.md)
- [API examples](docs/development/api-examples.md)
- [Environment variables](docs/development/environment-variables.md)
- [Testing strategy](docs/development/testing.md)

## License

No open-source license has been selected for this repository.
