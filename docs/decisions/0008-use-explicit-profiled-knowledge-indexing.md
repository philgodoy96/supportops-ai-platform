# ADR 0008: Use Explicit, Profiled Knowledge Indexing

## Status

Accepted

## Context

SupportOps AI Platform requires a reproducible path from immutable knowledge
document versions to a vector projection suitable for later semantic retrieval.

PostgreSQL already owns document identity, normalized source content, version
history, indexing state, chunk persistence, and active-version selection.
ADR 0003 established Qdrant as a rebuildable retrieval index rather than a
source of truth.

The concrete indexing implementation must now decide:

- when indexing occurs;
- how chunking and embedding identity are persisted;
- how provider and vector-store failures affect document state;
- how partial Qdrant writes are recovered;
- whether indexing activates a version;
- how local deterministic execution differs from external-provider execution;
- which data may be copied into Qdrant;
- how collection compatibility is enforced.

Performing indexing inside document-registration HTTP requests would couple
request latency and availability to tokenization, embedding providers, and
Qdrant. It would also make provider cost an implicit consequence of source
registration.

Allowing runtime settings to replace a persisted profile during retry would
make a document version's chunks and vectors irreproducible.

Deleting an existing projection before every retry would create a destructive
availability window and would discard successfully acknowledged points after a
partial failure.

Marking a version ready before exact projection verification would allow
retrieval to depend on incomplete derived state.

Automatically activating a successfully indexed version would combine
technical projection completion with a separate product and operational
decision.

## Decision

SupportOps AI Platform will use explicit, operator-controlled, profiled
knowledge indexing.

### Explicit operation

Document and version registration will persist source state only.

Indexing will run through the dedicated command:

```text
supportops-index-knowledge
```

The command will support:

```text
ensure-collection
index-version
```

An external embedding provider requires explicit command-line acknowledgement
before its client is initialized.

### Immutable index profile

A document version will be bound to one immutable profile recording:

- chunking strategy;
- chunking version;
- tokenizer encoding;
- embedding provider;
- embedding model;
- embedding dimensions;
- Qdrant collection name;
- Qdrant vector name.

A retry must use the same profile.

A configuration mismatch will fail rather than silently migrate the version.

### Deterministic chunking

The initial chunking profile will use:

```text
strategy = markdown-token
version = v1
tokenizer = cl100k_base
maximum tokens = 500
overlap tokens = 75
```

Chunk IDs will be deterministic from document-version identity and ordinal.

Authoritative chunks will be persisted in PostgreSQL before embedding or
Qdrant operations.

### Provider boundary

Embedding providers will remain behind application-owned asynchronous
contracts.

The initial providers will be:

```text
mock
openai
```

The mock provider will use deterministic lexical hashing for local development
and pipeline tests. It will not be presented as equivalent to semantic
production embeddings.

The OpenAI profile will use:

```text
model = text-embedding-3-small
dimensions = 1536
```

Provider selection will be explicit. There will be no automatic fallback.

### Cost provenance

Provider-reported input-token usage will be required before a version can
become ready.

Estimated cost will use an immutable versioned pricing catalog and `Decimal`.

Unknown pricing will remain unknown rather than being treated as zero.

### Rebuildable Qdrant projection

Qdrant collections will use:

```text
named vector = dense
distance = cosine
```

Mock and OpenAI profiles will use separate collections.

Collection dimensions, vector names, distance, and ownership payload index
types will be validated before writes.

Existing incompatible collections will be rejected rather than mutated
silently.

Qdrant point IDs will equal deterministic PostgreSQL chunk IDs.

Qdrant payloads will exclude authoritative source and chunk content.

### Idempotent upsert

The normal indexing path will upsert deterministic points without deleting the
existing version projection first.

A retry will replace matching point IDs and add missing points.

Readiness will require an exact workspace-, document-, and version-scoped
Qdrant point count equal to the authoritative PostgreSQL chunk count.

### Short database transactions

The service will use separate short transactions for:

1. profile binding or retry preparation;
2. authoritative chunk persistence;
3. ready finalization;
4. failure persistence.

Tokenization, provider calls, and Qdrant operations will run outside database
transactions.

### Durable failure state

Owned provider, chunking, collection, vector-store, and projection failures will
produce stable failure codes.

A failure record will preserve the actual PostgreSQL chunk count.

Failure persistence will not overwrite a concurrently completed ready version.

The original operational error will remain primary if failure persistence
itself fails.

### Ready and active remain separate

Successful indexing will mark a version ready.

It will not update the document's active-version pointer.

Activation remains an explicit PostgreSQL operation after indexing and any
required evaluation or review.

### Ready reruns

A compatible ready version will return as an idempotent no-op.

The service will not re-chunk, call the provider, or rewrite Qdrant.

## Consequences

### Positive consequences

- document registration remains independent from provider and Qdrant latency;
- external provider cost requires explicit operator intent;
- chunk and vector provenance is reproducible;
- PostgreSQL remains authoritative;
- Qdrant collections remain replaceable;
- partial vector projections can be repaired;
- safe retries do not duplicate points;
- collection drift fails visibly;
- provider SDK types remain outside business and indexing orchestration;
- long external calls do not retain PostgreSQL transactions or row locks;
- readiness has a concrete completeness invariant;
- activation can be governed and evaluated independently;
- local development and CI remain network-free by default.

### Trade-offs

- indexing is not automatic after document registration;
- operators must run or later schedule a separate command;
- a failed retry may repeat embedding provider cost;
- rebuilding a large collection can require time and external-provider spend;
- persisted profiles require deliberate migration when chunking or embedding
  strategy changes;
- a ready no-op does not independently reconcile a subsequently damaged
  Qdrant collection;
- collection names are part of operational profile governance;
- separate mock and OpenAI collections increase local infrastructure metadata;
- exact Qdrant count verification adds an additional index operation.

### Required engineering practices

This decision requires:

- immutable normalized source versions;
- immutable persisted index profiles;
- deterministic chunk identifiers;
- authoritative chunk persistence in PostgreSQL;
- no source content in Qdrant payloads;
- separate collections for incompatible embedding profiles;
- collection compatibility checks before writes;
- payload indexes for ownership filters;
- bounded embedding and vector batches;
- provider calls outside database transactions;
- stable provider-independent errors;
- provider usage and pricing provenance;
- exact count verification before readiness;
- explicit activation after readiness;
- integration tests against real PostgreSQL and Qdrant;
- no paid provider calls in the default automated suite;
- explicit external-provider permission in the CLI.

## Alternatives considered

### Index inside the document creation API

Rejected because source registration should not depend on embedding-provider or
Qdrant availability, latency, or cost.

The API remains responsible for authoritative source persistence.

### Automatically enqueue indexing as a background AgentRun

Intentionally deferred.

The current slice establishes deterministic indexing semantics, idempotency,
failure recovery, and operator controls before introducing scheduling,
concurrency limits, leases, and queue retry policy for this workload.

The command can later become the execution core behind a scheduled job without
changing ownership boundaries.

### Delete the version projection before every retry

Rejected because a transient failure after deletion could remove previously
acknowledged valid points and reduce availability.

Deterministic upsert plus exact count verification repairs partial projections
without a destructive pre-delete step.

### Store chunk text in Qdrant payloads

Rejected because it would duplicate authoritative content and weaken
PostgreSQL ownership.

Retrieval will hydrate source text from PostgreSQL after Qdrant returns
candidate identifiers.

### Mark ready after Qdrant acknowledges the last batch

Rejected because acknowledgement of the last batch does not prove that the
complete version projection exists.

Exact version-scoped count verification is required.

### Activate every newly ready version automatically

Rejected because projection completion and retrieval selection are separate
decisions.

Explicit activation leaves room for retrieval evaluation, operational review,
and controlled rollback.

### Allow profile changes during retry

Rejected because changing tokenizer, chunking, embedding, dimensions, or
collection identity would make one document version represent multiple
incompatible derived states.

A changed profile requires a separate controlled rebuild or migration
capability.

### Use the mock provider as fallback after OpenAI failure

Rejected because synthetic lexical vectors would hide provider failure and
change retrieval behavior without explicit provenance or operator intent.

### Treat unknown pricing as zero

Rejected because missing price knowledge is not evidence that provider usage is
free.

Token usage remains durable while the cost estimate remains unknown.