# Knowledge Indexing Pipeline

## Purpose

This document defines the implemented knowledge indexing pipeline for SupportOps AI Platform.

The pipeline converts an immutable PostgreSQL-backed knowledge document version into:

- deterministic authoritative chunks stored in PostgreSQL;
- provider-generated embedding vectors;
- a rebuildable Qdrant projection;
- durable indexing status, usage, cost, and failure provenance.

Indexing is an explicit operator-controlled operation. Document registration through the HTTP API does not invoke an embedding provider, write to Qdrant, or activate a document version.

Semantic retrieval and grounded generation are separate capabilities and are not implemented by this pipeline.

## Architectural position

The indexing pipeline is implemented under:

```text
src/supportops/knowledge_index/
```

The package owns:

- deterministic chunking;
- embedding-provider composition;
- Qdrant collection and point management;
- indexing orchestration;
- the operator CLI.

Authoritative document state remains under:

```text
src/supportops/modules/knowledge_documents/
```

Provider-independent embedding contracts remain under:

```text
src/supportops/ai/embeddings/
```

This separation keeps document ownership, external AI integration, and derived index infrastructure independently testable.

## Source-of-truth boundaries

### PostgreSQL

PostgreSQL is authoritative for:

- document identity;
- immutable normalized source content;
- document version numbers;
- content hashes;
- indexing profile provenance;
- chunk identity and ordering;
- authoritative chunk text;
- chunk hashes and token counts;
- indexing status;
- provider usage;
- estimated embedding cost;
- pricing catalog version;
- failure code;
- active-version selection.

### Qdrant

Qdrant is a rebuildable projection used to store:

- deterministic chunk point identifiers;
- dense embedding vectors;
- workspace ownership identifiers;
- document ownership identifiers;
- document-version identifiers;
- chunk ordinal;
- chunk content hash;
- media type;
- chunking strategy and version.

Qdrant payloads do not contain authoritative source or chunk text.

Losing a Qdrant collection does not lose source content. The collection can be reconstructed from PostgreSQL, subject to the selected embedding provider being available and the corresponding model and pricing policy remaining operationally supported.

## Index profile

Every indexed document version is bound to one immutable `KnowledgeIndexProfile`.

The profile records:

- chunking strategy;
- chunking version;
- tokenizer encoding;
- embedding provider;
- embedding model;
- embedding dimensions;
- Qdrant collection name;
- Qdrant vector name.

Implemented profiles use:

```text
chunking_strategy = markdown-token
chunking_version = v1
tokenizer_encoding = cl100k_base
knowledge_vector_name = dense
```

The local mock profile uses:

```text
embedding_provider = mock
embedding_model = mock-hashing-embedding-v1
embedding_dimensions = 64
knowledge_collection = supportops-knowledge-mock-v1
```

The OpenAI profile uses:

```text
embedding_provider = openai
embedding_model = text-embedding-3-small
embedding_dimensions = 1536
knowledge_collection = supportops-knowledge-openai-v1
```

Mock and OpenAI vectors are stored in separate collections. A collection is never reused across incompatible dimensions or embedding identities.

A persisted profile cannot be silently replaced during retry. Runtime configuration that conflicts with the existing profile produces a stable profile-mismatch failure.

## Deterministic chunking

### Policy

The implemented policy is:

```text
maximum chunk size = 500 tokens
overlap = 75 tokens
tokenizer = cl100k_base
```

Chunking is deterministic for a fixed:

- normalized source document;
- document-version identity;
- media type;
- chunking profile.

### Markdown behavior

For Markdown sources, the chunker recognizes:

- ATX headings;
- heading hierarchy;
- paragraph boundaries;
- fenced code blocks.

Heading paths are persisted as ordered section metadata.

The chunker prefers semantic block boundaries before the hard token limit. A semantic block that exceeds the limit is split by token windows.

For plain-text sources, Markdown syntax is not interpreted as structural metadata.

### Chunk identity

Each chunk has:

- a zero-based ordinal;
- a deterministic UUID derived from the document-version identity and ordinal;
- a SHA-256 hash of its exact chunk content;
- token count;
- section path;
- chunking strategy and version.

A safe rerun produces the same chunk IDs and chunk content. Existing authoritative chunks are accepted only when they exactly match the deterministic result.

A mismatched existing chunk is a persistence conflict and blocks readiness.

## Embedding boundary

Embedding providers implement application-owned asynchronous contracts.

The indexing service submits ordered batches containing:

- operation identity;
- embedding model;
- ordered chunk text;
- expected dimensions;
- request timeout;
- ownership metadata.

Provider adapters return:

- ordered immutable vectors;
- provider identity;
- model identity;
- dimensions;
- provider-reported input-token usage;
- optional provider request identifier.

SDK-specific response objects and exceptions do not escape the adapter boundary.

### Mock provider

The mock provider is:

- deterministic;
- network-free;
- based on SHA-256 lexical feature hashing;
- fixed-dimensional;
- normalized;
- explicitly zero-cost in the pricing catalog.

The mock vectors model lexical overlap for local development and pipeline testing. They are not presented as a semantic-quality substitute for production embeddings.

### OpenAI provider

The OpenAI adapter uses:

```text
model = text-embedding-3-small
dimensions = 1536
encoding_format = float
```

The adapter:

- preserves batch order through provider indexes;
- validates model and dimensions;
- maps provider usage;
- retains provider request identifiers when available;
- normalizes timeout, rate-limit, authentication, quota, invalid-request, unavailable, invalid-response, and unexpected failures;
- uses bounded SDK transport retries;
- owns no business or persistence decisions.

OpenAI is selected explicitly. It is never used as an implicit fallback from the mock provider, and the mock provider is never used as fallback after an OpenAI failure.

## Usage and cost provenance

Embedding usage is provider-reported.

A document version becomes ready only when input-token usage is available for every embedding batch.

Estimated cost is calculated by the application through a versioned exact-match pricing catalog using `Decimal`.

The implemented catalog records:

```text
mock / mock-hashing-embedding-v1 = USD 0 per 1,000,000 input tokens
openai / text-embedding-3-small = USD 0.02 per 1,000,000 input tokens
```

The document version persists:

- total input tokens;
- estimated cost when known;
- pricing catalog version.

Unknown pricing is stored as an unknown estimate rather than being treated as zero.

Provider invoices remain the authoritative billing source.

## Qdrant collection model

Each profile owns a distinct collection.

A compatible knowledge collection requires:

- exactly one named dense vector;
- vector name `dense`;
- expected dimensions;
- cosine distance;
- UUID payload indexes for:
  - `workspace_id`;
  - `document_id`;
  - `document_version_id`.

Collection creation and payload index creation are idempotent.

An existing collection with incompatible:

- vector names;
- dimensions;
- distance metric;
- payload index types

is rejected rather than modified silently.

## Qdrant point model

Every Qdrant point uses the deterministic PostgreSQL chunk ID.

The point payload includes only non-authoritative metadata required for ownership filtering, reconciliation, and provenance.

The point payload intentionally excludes:

- source document content;
- authoritative chunk content;
- API keys;
- provider responses;
- raw provider errors;
- pricing secrets.

Point writes use `wait=True`.

Upserting the same deterministic point IDs replaces the existing projection rather than creating duplicates.

## Indexing state machine

Document versions use:

```text
pending
failed
ready
```

### Pending

A pending version may:

- have no profile before its first indexing attempt;
- receive the configured immutable profile;
- produce and persist deterministic chunks;
- invoke the embedding provider;
- write its Qdrant projection;
- become failed or ready.

### Failed

A failed version preserves:

- immutable source content;
- immutable profile;
- authoritative chunks already persisted;
- stable failure code;
- persisted chunk count.

A compatible retry returns the version to pending and re-executes the pipeline.

A retry does not assign a new profile or document-version identity.

### Ready

A ready version records:

- complete chunk count;
- embedding input-token usage;
- estimated embedding cost when known;
- pricing catalog version;
- indexed timestamp.

Ready versions are immutable.

Re-indexing a ready version is a safe no-op and does not invoke the provider or rewrite Qdrant.

## Ready versus active

Readiness and activation are independent.

```text
ready = the configured projection completed and passed verification
active = the document version is explicitly selected for retrieval
```

Successful indexing does not update `Document.active_version_id`.

Activation remains a separate PostgreSQL transaction through the knowledge-document application service and API.

This separation allows operators to:

- index a new version;
- inspect its provenance;
- run retrieval evaluation;
- activate it only after acceptance.

## Transaction boundaries

The indexing service uses short PostgreSQL transactions around authoritative state changes.

### Profile transaction

The service:

- locks the document version;
- confirms workspace and document ownership;
- returns immediately for a compatible ready version;
- prepares a failed version for retry;
- binds or validates the immutable profile;
- commits before external work.

### Chunk transaction

The service:

- locks the version again;
- validates the persisted profile;
- inserts or verifies deterministic chunks;
- verifies the PostgreSQL chunk count;
- commits before embedding calls.

### External work

The service performs outside PostgreSQL transactions:

- tokenization;
- embedding provider calls;
- Qdrant collection validation;
- vector upserts;
- Qdrant count verification.

No database row lock or transaction is held across provider or Qdrant latency.

### Finalization transaction

After successful projection verification, the service:

- locks the version;
- validates the profile again;
- marks the version ready;
- persists usage, cost, catalog, count, and timestamp.

### Failure transaction

For an owned operational failure, the service:

- locks the current version;
- does not overwrite a concurrently completed ready version;
- counts authoritative PostgreSQL chunks;
- persists a stable failure code;
- marks the version failed.

Failure persistence must not mask the original operational error.

## Projection verification

Readiness requires:

```text
exact Qdrant version point count
=
authoritative PostgreSQL chunk count
```

The count filter includes:

- workspace ID;
- document ID;
- document-version ID.

A partial vector projection cannot become ready.

## Idempotency and recovery

### Successful rerun

After a version becomes ready:

- no chunking occurs;
- no embedding provider call occurs;
- no Qdrant write occurs;
- the existing ready result is returned.

### Partial projection failure

A process may fail after some Qdrant batches are acknowledged.

In that case:

- all authoritative chunks remain in PostgreSQL;
- the partial Qdrant projection remains rebuildable;
- the version records a stable retryable failure;
- a retry regenerates the same chunks;
- the retry requests embeddings again;
- deterministic point IDs replace existing points and add missing points;
- exact count verification gates readiness.

The pipeline does not claim exactly-once external provider cost. A retry after an interrupted external-call boundary may incur repeated embedding usage.

### Chunk persistence conflict

If persisted chunks do not match deterministic regeneration, the pipeline fails terminally for that profile.

The service does not overwrite conflicting authoritative chunk state.

## Operator CLI

The implemented command is:

```text
supportops-index-knowledge
```

### Ensure collection

```powershell
uv run supportops-index-knowledge ensure-collection
```

This command:

- loads validated settings;
- builds the immutable profile;
- creates or validates the Qdrant collection;
- creates required payload indexes;
- writes a stable JSON summary.

### Index version

```powershell
uv run supportops-index-knowledge index-version `
  --workspace-id "<workspace-id>" `
  --document-id "<document-id>" `
  --document-version-id "<document-version-id>"
```

All three ownership identifiers are required.

The command writes a stable JSON result containing:

- document and version identity;
- ready status;
- already-ready flag;
- chunk count;
- embedding provider and model;
- input-token usage;
- estimated cost;
- pricing catalog version;
- collection;
- indexed timestamp.

Source content and secrets are not included.

### External-provider acknowledgement

When OpenAI embeddings are configured, both commands require:

```text
--allow-external-provider
```

Permission is validated before the OpenAI client is initialized.

The flag is rejected when the mock provider is configured so operator intent remains explicit.

### Exit codes

```text
0 = success
1 = operational or runtime failure
2 = invalid usage or configuration
```

Unexpected exceptions are sanitized.

## Resource lifecycle

The indexing command owns process-scoped:

- PostgreSQL engine;
- SQLAlchemy session factory;
- Qdrant client;
- embedding provider.

Every resource is closed after success or failure.

If composition fails after creating only some resources, the partial runtime is closed before the original failure is propagated.

A resource-close failure changes an otherwise successful process result to an operational failure.

## Failure model

Stable indexing failure categories include:

- deterministic chunking failure;
- chunk persistence conflict;
- index-profile mismatch;
- incomplete vector projection;
- incompatible Qdrant collection;
- vector-store unavailable;
- vector-store operation failure;
- normalized embedding provider failures.

Raw provider or infrastructure exceptions are not durable error contracts.

## Security and privacy

The pipeline follows these rules:

- source and chunk content remain authoritative in PostgreSQL;
- source text is not copied into Qdrant payloads;
- API keys use secret-backed settings;
- complete settings are not logged;
- provider request payloads are not logged by default;
- raw provider responses are not persisted;
- CLI summaries exclude content and secrets;
- unexpected errors are sanitized at the process boundary;
- workspace identifiers remain mandatory for every indexing operation.

Workspace scoping is an ownership boundary. It is not authentication or authorization.

## Testing strategy

### Unit coverage

Unit tests cover:

- chunking policy invariants;
- Markdown heading, paragraph, and fenced-code behavior;
- token limits and overlap;
- deterministic chunk IDs;
- embedding request and response contracts;
- mock vector determinism;
- OpenAI request mapping and error normalization;
- pricing and unknown-pricing behavior;
- Qdrant collection compatibility;
- payload indexes and point batches;
- indexing state transitions;
- transaction separation;
- CLI safety and lifecycle behavior.

OpenAI unit tests use injected fakes and make no network requests.

### Integration coverage

Integration tests use real PostgreSQL and Qdrant to prove:

- collection creation;
- payload index creation;
- idempotent point upserts;
- workspace/document/version-scoped counts;
- first-time indexing;
- ready-version no-op reruns;
- stable PostgreSQL chunks;
- stable Qdrant point identities;
- partial projection failure persistence;
- retry recovery without duplicate points;
- no implicit activation.

## Intentional scope boundaries

This pipeline does not implement:

- semantic search;
- query embeddings;
- active-version retrieval filtering;
- PostgreSQL hydration of Qdrant results;
- citations;
- reranking;
- grounded LLM generation;
- LangGraph orchestration;
- tool execution;
- human approval;
- automated ingestion from object storage or external connectors;
- scheduled indexing jobs;
- automatic activation;
- automatic provider fallback;
- multi-profile collection migration;
- deletion and retention automation.

These capabilities remain separate engineering slices because they introduce distinct API contracts, reliability requirements, evaluation needs, and operational policies.

The next retrieval capability will query only explicitly active ready versions, use Qdrant for candidate identifiers, and hydrate authoritative chunk text from PostgreSQL.