# Testing Strategy

## Purpose

The SupportOps AI Platform test suite verifies foundation behavior across configuration, application composition, infrastructure connectivity, lifecycle management, health semantics, HTTP request traceability, workspace and ticket persistence, immutable knowledge-document versioning, explicit knowledge indexing, semantic knowledge retrieval, durable AgentRun scheduling, PostgreSQL worker claim and execution, workspace-scoped AgentRun inspection, classification inspection, controlled support workflow orchestration, tool audits, recommendation persistence, controlled support inspection, human-approved interrupt and resume, approval inspection and decision APIs, ticket escalation inspection APIs, optional application-owned AI observability, offline classification evaluation, multi-domain deterministic regression scoring, grounded recommendation evaluation validation and offline scoring, application services, versioned HTTP APIs, migration tooling, and container packaging.

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
- Document, DocumentVersion, and DocumentChunk invariants;
- deterministic source normalization, hashing, and chunk identity;
- AgentRun and AgentRunAttempt domain invariant tests;
- application service command and query behavior;
- transactional document creation, immutable version creation, and ready-version activation;
- transactional ticket-intake orchestration;
- retry policy calculation and attempt-budget gates;
- claim contracts and transition fencing contracts;
- deterministic executor workflow-contract validation;
- processor timeout, terminal, retryable, and sanitized unexpected-failure outcomes;
- worker cycle recovery-before-claim orchestration;
- polling-loop idle waits and interruptible shutdown behavior;
- scoped-session worker runtime composition;
- worker process identity resolution and graceful shutdown;
- SQLAlchemy mapping and metadata tests for AgentRun and knowledge-document persistence;
- named PostgreSQL constraints declared on persistence records;
- persistence model registration;
- PostgreSQL constraint-name inspection helpers;
- workspace API schemas;
- knowledge-document API schemas and separate document/version cursor contracts;
- ticket API schemas, including the nested processing-run response;
- AgentRun inspection schema projections;
- opaque ticket cursor encoding and invalid-cursor rejection;
- classification inspection projections;
- classification cursor encoding;
- classification query services;
- classification query repository SQL shape;
- classification API schemas and routes;
- cross-module AgentRun inspection composition;
- evaluation dataset models and loader;
- evaluation contracts, manifests, prediction envelopes, hashing, and atomic
  artifact writes;
- split-manifest validation and frozen allocation;
- pinned dataset hash;
- prediction artifact validation;
- deterministic evaluation metrics, including validity, urgency recall,
  high-risk human-review recall, latency, and token aggregates;
- standalone release-gate evaluation;
- Gateway predictor behavior;
- sequential evaluation runner;
- explicit prompt-version selection;
- evaluation settings and composition;
- CLI provider safety gates;
- evaluation artifact reproducibility;
- deterministic markdown-token chunking;
- tokenizer adapter behavior;
- embedding contracts, pricing, mock provider, OpenAI fake provider, and normalized embedding errors;
- Qdrant adapter contracts, collection compatibility, payload indexes, upsert, and exact-count behavior;
- indexing orchestration, composition, and CLI safety gates;
- knowledge-retrieval contracts and request bounds;
- candidate provenance and Qdrant filter-pair validation;
- malformed candidate discard and active-scope short circuit;
- query embedding validation and authoritative hydration;
- deterministic ranking;
- retrieval API schemas, route composition, and `503` mappings;
- retrieval lifespan cleanup;
- AI observability settings validation;
- observability models and contracts;
- deterministic observability identity;
- observability ContextVar cleanup and async-task isolation;
- metadata-only and redacted-content privacy policies;
- observability sanitizer masking, truncation, bounds, and forbidden structures;
- no-op observability adapter behavior;
- Langfuse adapter behavior with an injected fake client;
- usage and cost payload mapping;
- fail-open Langfuse SDK behavior;
- gateway provider, embedding, retrieval, and indexing observability instrumentation;
- AgentRun, worker-attempt, workflow, graph-node, tool, approval, escalation, and recommendation observability instrumentation;
- attempt-end flush, privacy, fail-open, ContextVar isolation, and process-client ownership;
- API, worker, and indexing observability lifecycle ownership.

Unit tests use mocks only at external boundaries.

They must not:

- open real network connections;
- require `.env`;
- depend on local containers;
- mutate Docker services.

Focused knowledge-index and embedding unit coverage may exercise adapter logic against fakes without creating durable shared collections. Knowledge-document HTTP and module unit tests do not initialize embedding providers or create Qdrant collections.

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
uv run pytest tests/unit/modules/knowledge_documents/domain
uv run pytest tests/unit/modules/agent_runs/domain
uv run pytest tests/unit/modules/workspaces/application tests/unit/modules/tickets/application
uv run pytest tests/unit/modules/knowledge_documents/application
uv run pytest tests/unit/application/test_ticket_intake.py
uv run pytest tests/unit/modules/workspaces/infrastructure tests/unit/modules/tickets/infrastructure
uv run pytest tests/unit/modules/knowledge_documents/infrastructure
uv run pytest tests/unit/modules/agent_runs/infrastructure
uv run pytest tests/unit/modules/workspaces/api
uv run pytest tests/unit/modules/knowledge_documents/api
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
uv run pytest tests/unit/modules/ticket_classifications/api/test_schemas.py
```

Classification evaluation unit coverage:

```powershell
uv run pytest tests/unit/evaluation
```

Focused evaluation contract coverage:

```powershell
uv run pytest tests/unit/evaluation/contracts
```

Focused split-manifest coverage:

```powershell
uv run pytest `
  tests/unit/evaluation/ticket_classification/test_split_manifest.py
```

Focused ticket-classification evaluator coverage:

```powershell
uv run pytest `
  tests/unit/evaluation/ticket_classification/test_evaluator.py
```

Focused multi-domain deterministic evaluation coverage:

```powershell
uv run pytest `
  tests/unit/evaluation/semantic_retrieval `
  tests/unit/evaluation/controlled_support `
  tests/unit/evaluation/human_approval `
  tests/unit/evaluation/regression `
  -q
```

Focused grounded recommendation evaluation coverage:

```powershell
uv run pytest `
  tests/unit/evaluation/grounded_recommendations `
  -q
```

Focused prompt-version selection coverage:

```powershell
uv run pytest `
  tests/unit/evaluation/ticket_classification/test_predictor.py `
  tests/unit/evaluation/ticket_classification/test_runner.py `
  tests/unit/evaluation/ticket_classification/test_cli.py `
  -k prompt_version
```

Focused release-gate coverage:

```powershell
uv run pytest `
  tests/unit/evaluation/ticket_classification/test_evaluator.py `
  tests/unit/evaluation/ticket_classification/test_runner.py `
  -k "release_gate or not_applicable or coverage_gate or recall_gate or validity_gate"
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

Classification inspection unit coverage verifies:

- safe public field projection for classification responses;
- opaque classification history cursor behavior;
- query-service ownership validation;
- query-repository SQL shape for detail, history, and invocation reads;
- classification API schema and route composition;
- cross-module AgentRun inspection that attaches an optional classification
  reference and lists logical invocations.

Evaluation unit coverage verifies:

- shared evaluation contract hashing and atomic artifact writes;
- evaluation manifest and prediction-envelope validation;
- dataset model validation and loader behavior;
- pinned dataset content hash;
- split-manifest validation and frozen development, holdout, and safety-gate
  allocation;
- prediction artifact validation and hashing;
- deterministic full-label exact match and field accuracies;
- structured-output validity and invalid-output rate;
- high urgency, critical urgency, and high-risk human-review recall;
- average latency and average input, output, and total token metrics;
- human-review precision, recall, and F1;
- standalone release-gate outcomes and aggregate statuses;
- quality and efficiency gates remaining not applicable without paired
  baseline evidence;
- Gateway predictor and sequential runner behavior;
- explicit prompt-version selection with default version `1`;
- unsupported prompt versions failing without provider execution or artifact
  overwrite;
- evaluation settings and composition;
- CLI provider safety gates, including the external-provider permission flag;
- artifact reload and report reproducibility;
- semantic-retrieval, controlled-support, and human-approval deterministic
  evaluators over committed synthetic corpora;
- repository regression aggregation, optional classification omission, and
  deterministic domain ordering;
- grounded recommendation dataset, prediction, deterministic complementary
  metrics, static RAGAS score, offline aggregation, CLI, and fake-backed
  adapter coverage;
- grounded recommendation human review rubric loading and policy validation.

Normal pytest execution performs no paid provider calls. External provider
evaluation remains a manual operation that requires explicit acknowledgement.
Holdout discipline is procedural: holdout outcomes must not guide prompt
drafting.

## Knowledge-document tests

Focused unit coverage:

```powershell
uv run pytest tests/unit/modules/knowledge_documents
```

Focused PostgreSQL integration coverage:

```powershell
uv run pytest -m integration tests/integration/modules/knowledge_documents
```

Focused HTTP integration coverage:

```powershell
uv run pytest -m integration tests/integration/api/test_knowledge_documents.py
```

The knowledge-document suite verifies:

- title, external-reference, media-type, timestamp, and ownership invariants;
- canonical line-ending normalization and deterministic SHA-256 hashing;
- immutable source content and deterministic chunk UUIDs;
- complete-or-empty indexing-profile state;
- `pending`, `failed`, and `ready` lifecycle validation;
- SQLAlchemy domain round trips and named relational constraints;
- composite document, version, and chunk-profile ownership;
- migration upgrade, single-step downgrade, base downgrade, re-upgrade, and metadata parity;
- ready-only activation and ready-version rewrite protection;
- workspace-scoped repository reads and deterministic keyset ordering;
- normalized persistence conflict translation;
- idempotent deterministic chunk persistence;
- application-owned transaction boundaries;
- document-row locking before next-version allocation;
- concurrent distinct-content version creation producing unique numbers;
- concurrent equivalent-content creation producing one version and one stable conflict;
- separate opaque cursor kinds for documents and versions;
- source content omission from create and list responses;
- source content exposure only from the version detail endpoint;
- pending activation conflict and idempotent ready activation;
- cross-workspace document and version access returning `404`.

These tests do not initialize embedding providers, create Qdrant collections, or perform semantic retrieval. Knowledge-document API tests remain source-registration focused. Indexing behavior is covered by the dedicated knowledge-index suites below. Retrieval behavior is covered by the dedicated knowledge-retrieval suites.

## Knowledge-index and embedding tests

Focused unit coverage:

```powershell
uv run pytest tests/unit/knowledge_index
uv run pytest tests/unit/ai/embeddings
```

Focused integration coverage:

```powershell
uv run pytest -m integration tests/integration/knowledge_index
```

The knowledge-index and embedding suites verify:

- deterministic markdown-token chunking and `cl100k_base` tokenization;
- Markdown-aware heading, paragraph, and fenced-code handling;
- plain-text chunking without Markdown interpretation;
- embedding contracts, ordered batches, pricing, mock hashing, OpenAI fakes, and normalized errors;
- Qdrant collection bootstrap, payload indexes, deterministic upsert, and exact scoped count;
- indexing profile bind and validation;
- chunk persistence, batched embedding, Qdrant upsert, exact projection verification, and ready marking;
- stable failure recording;
- ready-version rerun no-op behavior;
- partial-projection failure and retry recovery;
- absence of implicit activation after successful indexing;
- composition and CLI resource ownership, external-provider permission gates, and exit codes.

Default automated tests use mock embeddings. OpenAI unit tests use injected fakes. No paid external provider call occurs in the automated suite. Integration tests create isolated Qdrant collections and delete them in `finally` blocks.

## Knowledge-retrieval tests

Focused unit coverage:

```powershell
uv run pytest tests/unit/knowledge_retrieval -vv
```

Focused integration coverage:

```powershell
uv run pytest -m integration tests/integration/knowledge_retrieval -vv
```

The knowledge-retrieval suites verify:

- retrieval contracts and request bounds;
- candidate provenance validation;
- Qdrant filter pairs for workspace and active document/version targets;
- malformed candidate discard;
- active-scope short circuit with no embedding or Qdrant call;
- query embedding validation;
- authoritative PostgreSQL hydration;
- deterministic ranking and rank promotion after invalid candidate discard;
- API schemas and route composition;
- `503` mappings for expected embedding and vector-store failures;
- lifespan cleanup, including partial API startup cleanup;
- real PostgreSQL and Qdrant integration with deterministic mock embeddings;
- active ready participation and ready inactive exclusion;
- workspace isolation and document filtering;
- cross-workspace nondisclosure through empty evidence;
- stale projection hash rejection;
- logs that exclude source and chunk content.

Default tests use mock embeddings and make no OpenAI network calls.

## AI observability tests

Slice 7 includes provider, embedding, retrieval, indexing, and durable
workflow observability coverage. Grounded recommendation evaluation with
deterministic complementary metrics, static RAGAS score artifacts, offline
aggregation, and an explicit external RAGAS boundary is implemented separately
from observability. Classification prompt version 2, paired prompt comparison,
and prompt promotion remain later follow-up work beyond the repository-owned
classification, multi-domain deterministic regression, and grounded
recommendation evaluation foundation.

### Provider, embedding, retrieval, and indexing coverage

Focused unit coverage:

```powershell
uv run pytest tests/unit/ai/gateway
uv run pytest tests/unit/ai/embeddings
uv run pytest tests/unit/knowledge_retrieval
uv run pytest tests/unit/knowledge_index
uv run pytest tests/unit/api/test_lifespan.py
uv run pytest tests/unit/worker/test_composition.py tests/unit/worker/test_main.py
uv run pytest tests/unit/observability
uv run pytest tests/unit/core/test_settings.py -k langfuse
uv run pytest `
  tests/unit/knowledge_index/test_composition.py `
  tests/unit/knowledge_index/test_cli.py
uv run pytest -m "not integration"
```

These suites verify:

- settings defaults and validation for provider, capture mode, credentials,
  base URL, environment, release, flush flag, and timeout;
- observability models and contracts;
- deterministic AgentRun and ticket identity seeds;
- ContextVar cleanup and async-task isolation;
- metadata-only privacy that omits business content;
- redacted-content privacy with allowlisted structured fields;
- sanitizer masking, truncation, collection bounds, and forbidden structures;
- no-op adapter behavior with no network access;
- Langfuse adapter behavior against an injected fake client;
- usage and cost payload mapping, including known and unknown pricing;
- fail-open SDK construction and export behavior;
- one provider request produces one provider observation at the
  application-owned gateway boundary;
- initial and repair requests produce separate generation observations;
- tool-decision modes remain distinguishable without exporting tool content;
- one embedding call produces one embedding observation through an
  application-owned wrapper;
- query and indexing embedding operations are distinguishable;
- semantic retrieval owns one `RETRIEVER` observation that exports counts and
  technical metadata, not query or evidence content;
- query embedding nests under retrieval;
- one indexing command owns one deterministic root trace;
- embedding observations nest under indexing traces;
- prompt, query, document, chunk, evidence, tool, and vector content remain
  absent from default metadata-only export;
- known mock zero cost remains explicit zero;
- unknown pricing remains unknown;
- missing usage does not fabricate cost;
- telemetry failures do not alter business results, exceptions, persistence, or
  exit codes;
- API, worker, and indexing each own one process-scoped observability client,
  including shutdown isolation and readiness independence from Langfuse.

### Durable workflow observability

Focused local validation:

```powershell
uv run pytest tests/unit/observability
uv run pytest tests/unit/modules/agent_runs
uv run pytest tests/unit/agent_graph
uv run pytest tests/unit/agent_tools
uv run pytest tests/unit/modules/approvals
uv run pytest tests/unit/worker
uv run pytest tests/unit/api/test_lifespan.py
```

Provider and retrieval instrumentation regression coverage:

```powershell
uv run pytest tests/unit/ai/gateway
uv run pytest tests/unit/ai/embeddings
uv run pytest tests/unit/knowledge_retrieval
uv run pytest tests/unit/knowledge_index
```

Static validation:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -m "not integration"
```

Durable workflow observability coverage verifies:

- deterministic AgentRun trace identity seeded as `agent-run:{agent_run_id}`;
- multiple AgentRunAttempts re-entering the same logical trace;
- `worker-attempt` outcome mapping for success, retryable failure, terminal
  failure, timeout, approval pause, and lease-lost finalization;
- controlled workflow hierarchy under `workflow.controlled-support-v1`;
- human-approved workflow hierarchy under `workflow.human-approved-support-v1`;
- graph-node observations for LangGraph stages;
- tool-execution observations named `tool.execute`;
- approval request, pause, decision, expiration, and resume events;
- escalation outcome event `ticket.escalated`;
- recommendation outcome events `recommendation.generated`,
  `recommendation.persisted`, and `recommendation.failed`;
- attempt-end flush when enabled, with no flush when disabled;
- privacy assertions that exclude content-bearing fields;
- duplicate prevention for idempotent approval decisions;
- fail-open behavior at observation and event boundaries;
- ContextVar restoration and isolation across nested scopes;
- process-client ownership for API, worker, and indexing composition.

Normal tests:

- use application-owned recording or no-op fakes;
- do not require Langfuse credentials;
- do not call Langfuse Cloud;
- validate trace shape and privacy at the application contract boundary.

External Langfuse smoke validation remains an opt-in follow-up. Langfuse
evaluation workflows, paired prompt comparison, and prompt promotion remain
deferred. A real canonical external RAGAS baseline remains deferred. Multi-domain
deterministic regression scoring and grounded recommendation offline validation
and scoring are implemented and covered by the evaluation suites above.
PostgreSQL remains authoritative for durable LLM invocation and indexing usage
and estimated-cost records. Query-embedding usage and cost remain ephemeral
observability data.

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
- creation of `workspaces`, `tickets`, `agent_runs`, `agent_run_attempts`, `llm_invocations`, `ticket_classifications`, `knowledge_documents`, `knowledge_document_versions`, and `knowledge_document_chunks` tables;
- workspace persistence;
- knowledge-document, immutable-version, and authoritative-chunk persistence;
- ready-only activation and ready-version immutability triggers;
- duplicate workspace slug translation;
- transaction rollback;
- ticket foreign-key behavior;
- the same external reference across workspaces;
- duplicate external-reference rejection inside one workspace;
- cross-workspace ticket lookup behavior;
- deterministic ticket listing;
- keyset repository navigation;
- concurrent duplicate external-reference insertion;
- concurrent document-version allocation for distinct content;
- concurrent duplicate normalized-content rejection;
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
- knowledge-document creation, listing, version detail, version creation, and activation APIs;
- cross-workspace document and version access returning stable `404` responses;
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
- classification detail inspection;
- ticket classification keyset pagination;
- AgentRun classification reference;
- invocation ordering across retries and repairs;
- safe usage and estimated-cost projection;
- empty classification and invocation histories;
- missing and cross-workspace classification and invocation behavior;
- absence of request bodies from completion logs, including ticket subject and description content;
- real Qdrant collection and payload-index creation for knowledge indexing;
- deterministic point upsert and exact scoped count;
- successful indexing with real PostgreSQL and Qdrant using mock embeddings;
- ready-version rerun no-op behavior;
- partial-projection failure and retry recovery;
- indexing without implicit activation;
- semantic retrieval against real PostgreSQL and Qdrant with mock embeddings;
- active-ready isolation, workspace isolation, document filters, stale-hash rejection, and rank promotion.

Concurrency coverage uses independent sessions and synchronization primitives rather than arbitrary sleeps.

Full API integration tests require PostgreSQL and applied migrations. Knowledge-document registration routes do not call Qdrant. Knowledge-index and knowledge-retrieval integration tests use real PostgreSQL and Qdrant with mock embeddings. Default automated tests make no OpenAI requests. Qdrant-dependent tests are not worker tests.

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
uv run pytest `
  tests/integration/api/test_ticket_classifications.py
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

Classification inspection API integration coverage verifies:

- classification detail responses;
- ticket classification history with opaque keyset pagination;
- optional AgentRun classification reference;
- invocation ordering across retries and repairs;
- safe usage and estimated-cost projection;
- empty histories for queued or unclassified runs;
- missing and cross-workspace `404` contracts for classifications and
  invocation histories;
- omission of provider request IDs and raw provider content.

Integration tests are necessary for AgentRun and classification inspection because unit tests cannot fully prove:

- workspace-scoped SQL predicates against PostgreSQL;
- actual SQL ordering of attempts;
- FastAPI dependency composition for the mounted routes;
- error envelope integration through registered handlers;
- end-to-end tenant isolation behavior across HTTP and persistence.

## Controlled support workflow

Controlled support coverage verifies the final Slice 5 architecture across graph
orchestration, tool audits, recommendation persistence, checkpoint resume, and
inspection.

### Unit coverage

Unit tests cover:

- graph state and routing;
- decision contracts;
- bounded tool registry and execution;
- observation reconstruction;
- recommendation execution;
- worker composition for controlled runtime resources;
- inspection contracts and schemas.

### Integration coverage

Integration tests cover:

- tool audit persistence and fencing;
- recommendation and citation persistence;
- checkpoint setup and resume;
- post-commit/pre-checkpoint recovery;
- attempt-scoped ordering for tools and invocations;
- controlled inspection repository behavior;
- vertical HTTP inspection;
- cross-workspace absence behavior;
- Alembic metadata parity with exact framework-owned checkpoint table
  exclusions.

Default automated tests use mock LLM and embedding providers. No paid provider
calls are required. PostgreSQL and Qdrant are required for integration tests.
Tests that create checkpoint threads clean their own checkpoint rows. Shared
business cleanup does not delete framework checkpoint tables.

## Human-approved support workflow

Human-approved coverage validates durable interrupt and resume, grant-gated
sensitive execution, immutable escalation, AgentRun waiting lifecycle, crash
recovery, concurrency convergence, historical controlled-support regressions,
and the operational approval and escalation HTTP APIs.

### Unit coverage

Focused unit groups include:

- human-approved identity, state, and routing;
- safe interrupt payload validation;
- resume planner taxonomy and fail-closed mismatch handling;
- approval decision handling for approved, rejected, and expired outcomes;
- sensitive proposal persistence before interrupt;
- human-approved workflow resume across approved, rejected, expired, and pending paths;
- grant-gated sensitive execution without external side effects;
- AgentRun processor transitions into and out of `waiting_for_approval`.

```powershell
uv run pytest `
  tests/unit/agent_graph/domain/test_human_approved_identity.py `
  tests/unit/agent_graph/domain/test_human_approved_state.py `
  tests/unit/agent_graph/domain/test_human_approved_routing.py `
  tests/unit/agent_graph/domain/test_resume_planning.py `
  tests/unit/agent_graph/application/test_approval_interrupt.py `
  tests/unit/agent_graph/application/test_resume_planning.py `
  tests/unit/agent_graph/application/test_approval_decision_handling.py `
  tests/unit/agent_graph/application/test_sensitive_proposal.py `
  tests/unit/agent_graph/application/test_human_approved_workflow_resume.py `
  tests/unit/agent_graph/application/test_human_approved_recommendation.py `
  tests/unit/agent_tools/application/test_sensitive_execution.py `
  tests/unit/agent_tools/domain/test_grants.py `
  tests/unit/modules/agent_runs/application/test_processor.py
```

### Approval and escalation API unit coverage

Focused API unit groups cover approval inspection schemas and routes, opaque
approval cursors, decision request schemas and routers, escalation schemas and
routes, and escalation cursors.

```powershell
uv run pytest `
  tests/unit/modules/approvals/api/test_schemas.py `
  tests/unit/modules/approvals/api/test_pagination.py `
  tests/unit/modules/approvals/api/test_router.py `
  tests/unit/modules/approvals/api/test_decision_schemas.py `
  tests/unit/modules/approvals/api/test_decision_router.py `
  tests/unit/modules/tickets/api/test_escalation_schemas.py `
  tests/unit/modules/tickets/api/test_escalation_pagination.py `
  tests/unit/modules/tickets/api/test_escalation_router.py
```

### Integration coverage

Integration tests require live PostgreSQL. They cover granted escalation
execution, concurrent duplicate execution convergence, crash recovery after
grant or escalation persistence, and durable approval repository behavior.

```powershell
uv run pytest -m integration `
  tests/integration/agent_tools/application/test_granted_escalation.py `
  tests/integration/agent_tools/infrastructure/test_grant_repository.py `
  tests/integration/modules/approvals/infrastructure/test_repository.py
```

### Approval and escalation API integration coverage

Focused HTTP integration groups cover approval decisions, approval API safety,
ticket escalation inspection, and escalation API safety. Coverage includes
approval inspection, approval decisions, escalation inspection, workspace
isolation, cursor validation, idempotent replay, conflicting terminal
decisions, concurrent decision safety, API/worker boundary, and read-only
escalation behavior.

```powershell
uv run pytest -m integration `
  tests/integration/api/test_approval_decisions.py `
  tests/integration/api/test_ticket_escalations.py `
  tests/integration/api/test_approval_api_safety.py `
  tests/integration/api/test_ticket_escalation_api_safety.py
```

Historical controlled-support regression:

```powershell
uv run pytest `
  tests/integration/application/test_controlled_support_workflow.py
```

### Migration head

Current Alembic head for the human-approved escalation tables:

```text
b8d3f6a1c9e4
```

```powershell
uv run alembic heads
uv run alembic check
```

Default automated tests use mock providers and make no paid provider calls.
External side-effect tools intentionally remain unavailable and are not
exercised by the suite.

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
- no durable shared collection mutation outside dedicated knowledge-index tests.

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

## Classification evaluation smoke commands

Mock evaluation validates pipeline wiring and does not measure model quality:

```powershell
uv run supportops-evaluate-classification run `
  --provider mock `
  --prompt-version 1 `
  --dataset `
    evals/ticket-classification/datasets/ticket-classification-eval-v1.jsonl `
  --predictions-output `
    artifacts/classification-mock-predictions.jsonl `
  --output `
    artifacts/classification-mock-report.json
```

Offline re-score initializes no provider and makes no network request:

```powershell
uv run supportops-evaluate-classification score `
  --dataset `
    evals/ticket-classification/datasets/ticket-classification-eval-v1.jsonl `
  --predictions `
    artifacts/classification-mock-predictions.jsonl `
  --output `
    artifacts/classification-mock-rescored-report.json
```

Normal pytest execution performs no paid provider calls. Unit and integration
tests do not call OpenAI. External provider evaluation is a manual operation
and requires `--allow-external-provider` plus a configured API key. Holdout
discipline is procedural and holdout outcomes must not guide prompt drafting.
Generated evaluation artifacts are ignored by Git. Versioned datasets and split
manifests remain committed under `evals/`.

## Multi-domain deterministic regression

Repository regression scoring uses committed synthetic datasets and static
prediction fixtures:

```powershell
uv run supportops-evaluate-regression score
```

Default domains are `semantic-retrieval`, `controlled-support`, and
`human-approval`. Classification remains optional when static classification
evidence is not supplied. The command performs no network calls and does not
require secrets or runtime services. Normal CI runs the command explicitly.

Standalone committed fixtures without paired quality and efficiency baselines
produce aggregate status `incomplete`. That status is valid deterministic
evidence and exits zero. Blocking gate failure exits one. Artifact validation
or scoring failure exits three. Usage errors exit two.

Focused unit coverage for the deterministic evaluators and regression runner:

```powershell
uv run pytest `
  tests/unit/evaluation/semantic_retrieval `
  tests/unit/evaluation/controlled_support `
  tests/unit/evaluation/human_approval `
  tests/unit/evaluation/regression `
  -q
```

## Grounded recommendation evaluation

Grounded recommendation evaluation uses committed synthetic fixtures and the
`supportops-evaluate-grounded-recommendations` CLI. Offline commands perform no
network access and do not require secrets or runtime services.

Focused unit coverage:

```powershell
uv run pytest `
  tests/unit/evaluation/grounded_recommendations `
  -q
```

Offline validation and scoring:

```powershell
uv run supportops-evaluate-grounded-recommendations validate

uv run supportops-evaluate-grounded-recommendations score

uv run supportops-evaluate-grounded-recommendations score `
  --ragas-scores "evals/grounded-recommendations/ragas-scores/grounded-recommendations-eval-v1.static.jsonl"
```

`validate` and `score` do not instantiate evaluator models or generate
embeddings. Normal CI runs these offline commands without API keys and without
`--allow-external-provider`.

External RAGAS evaluation of existing predictions is a manual operation. It
requires `--allow-external-provider`, credentials from
`SUPPORTOPS_EVALUATION_OPENAI_API_KEY`, and an output directory under
`artifacts/`. The command does not implicitly fall back to
`SUPPORTOPS_OPENAI_API_KEY`. Do not place a real API key in documentation or
committed files. No real external smoke tests are committed, and no `external`
pytest marker is introduced for this boundary.

Domain architecture is documented in
[`../architecture/evaluation-and-regression.md`](../architecture/evaluation-and-regression.md).
Committed fixtures are summarized in
[`../../evals/grounded-recommendations/README.md`](../../evals/grounded-recommendations/README.md).

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
- queued `ticket-processing` / `controlled-support-v1` processing reference;
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
uv run pytest `
  tests/integration/api/test_ticket_classifications.py
uv run pytest tests/integration/application/test_ticket_intake.py
```

Focused AgentRun inspection coverage:

```powershell
uv run pytest tests/unit/modules/agent_runs/application/test_services.py
uv run pytest tests/unit/modules/agent_runs/api/test_schemas.py
uv run pytest tests/integration/modules/agent_runs/infrastructure/test_query_repository.py
uv run pytest tests/integration/api/test_agent_runs.py
```

Focused classification inspection and evaluation coverage:

```powershell
uv run pytest tests/unit/modules/ticket_classifications
uv run pytest tests/unit/evaluation
uv run pytest tests/unit/evaluation/contracts
uv run pytest `
  tests/unit/evaluation/ticket_classification/test_split_manifest.py
uv run pytest `
  tests/unit/evaluation/ticket_classification/test_evaluator.py
uv run pytest `
  tests/integration/api/test_ticket_classifications.py
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
- application cleanup failure;
- embedding retrieval failure;
- vector-search failure;
- partial API startup cleanup.

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

The current migration head creates the `workspaces`, `tickets`, `agent_runs`,
`agent_run_attempts`, `llm_invocations`, `ticket_classifications`,
`agent_tool_calls`, `support_recommendations`,
`support_recommendation_citations`, `knowledge_documents`,
`knowledge_document_versions`, `knowledge_document_chunks`,
`approval_requests`, `sensitive_execution_grants`, and `ticket_escalations`
tables.
Framework-owned LangGraph checkpoint tables are created by checkpointer setup
and are excluded by exact name from Alembic comparison.

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
- upgrade creates the application-owned business tables listed above;
- downgrade removes those business tables;
- re-upgrade restores them;
- migration metadata remains aligned with SQLAlchemy model metadata;
- `alembic check` reports no new upgrade operations and does not propose
  removal of the exact framework-owned checkpoint tables.

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

After upgrade, the schema includes `workspaces`, `tickets`, `agent_runs`,
`agent_run_attempts`, `llm_invocations`, `ticket_classifications`,
`knowledge_documents`, `knowledge_document_versions`, and
`knowledge_document_chunks`.

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
uv run supportops-evaluate-regression score
uv run supportops-evaluate-grounded-recommendations validate
uv run supportops-evaluate-grounded-recommendations score
uv run supportops-evaluate-grounded-recommendations score --ragas-scores evals/grounded-recommendations/ragas-scores/grounded-recommendations-eval-v1.static.jsonl
uv run pytest -m "not integration"
uv run alembic heads
uv run alembic current
uv run alembic check
uv run pytest -m integration
docker build --tag supportops-ai-platform:ci .
```

Continuous integration provides PostgreSQL and Qdrant service containers.

The CI environment uses non-production credentials and a slightly higher dependency health timeout to reduce shared-runner flakiness.

CI must not update the lockfile or publish the application image. The regression command scores committed static fixtures only and does not require secrets or paid providers. Grounded recommendation CI commands likewise remain offline and do not perform paid external evaluation.

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
- retrieval reranking;
- write-capable tool authorization beyond grant-gated internal escalation;
- opt-in live Langfuse smoke validation;
- Langfuse evaluation workflows;
- prompt version 2 regression comparison;
- paired prompt comparison and promotion decision coverage;
- scheduled evaluation and evaluation history persistence;
- a real canonical external RAGAS baseline;
- production feedback ingestion;
- a full annotation platform;
- evaluation dashboards beyond standalone classification, multi-domain
  release gates, and grounded recommendation evaluation;
- idempotent side effects for future write-capable executors and tools.

Authentication remains an intentional scope boundary for the current suite.
Durable AgentRun scheduling, PostgreSQL claiming, fencing, retries, recovery,
deterministic execution, worker process coverage, workspace-scoped AgentRun and
classification inspection, controlled support workflow coverage, human-approved
workflow coverage, approval and escalation inspection and decision APIs,
immutable knowledge-document versioning, explicit knowledge indexing,
active-version semantic knowledge retrieval, optional application-owned AI
observability with provider, embedding, retrieval, indexing, and durable
workflow coverage, repository-owned offline classification evaluation with
contracts, split manifests, explicit prompt-version selection, and standalone
release gates, multi-domain deterministic regression for semantic retrieval,
controlled support, and human approval, and grounded recommendation evaluation
with offline validation, deterministic complementary metrics, static RAGAS score
aggregation, and fake-backed adapter tests are part of the current suite.
