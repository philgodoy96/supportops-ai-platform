# ADR 0009: Hydrate Retrieval Evidence from PostgreSQL

## Status

Accepted

## Context

SupportOps AI Platform stores immutable knowledge source versions and
deterministic chunks in PostgreSQL. Qdrant stores a rebuildable dense-vector
projection whose point identifiers correspond to PostgreSQL chunk identifiers.

The platform now requires a semantic retrieval API over internal runbooks.

The retrieval implementation must decide:

- which document versions are eligible;
- whether Qdrant payload content may be returned directly;
- how workspace ownership is enforced;
- how stale or malformed vector points are handled;
- when query embeddings are generated;
- how ranking behaves after inconsistent candidates are removed;
- whether retrieval should generate an LLM answer;
- whether one request may span multiple embedding profiles;
- how provider and vector-store failures are exposed through HTTP.

Returning text directly from Qdrant would make derived payload state compete
with PostgreSQL as an independent content source.

Searching every ready version would ignore the explicit active-version rollout
boundary.

Filtering document IDs and version IDs independently could permit cross-pair
combinations that were never active together.

Failing the complete request for every missing or stale point would allow
rebuildable projection drift to make all otherwise valid evidence unavailable.

Creating an embedding before resolving eligible versions would incur provider
work for empty scopes and cross-workspace filters.

Combining retrieval and LLM answer generation in the same delivery boundary
would introduce prompt, model, generation, citation-placement, refusal, cost,
and evaluation decisions before retrieval correctness was established.

Searching multiple embedding profiles in one request would require multiple
providers or models and a policy for comparing scores produced by different
vector spaces.

## Decision

SupportOps AI Platform will return semantic retrieval evidence only after
authoritative PostgreSQL hydration and provenance validation.

### Active ready versions define eligibility

PostgreSQL will resolve retrieval scope through the document's explicit
active-version pointer.

A version participates only when it:

- belongs to the requested workspace;
- belongs to the document;
- is the document's active version;
- has status `ready`;
- has a complete persisted index profile.

Ready but inactive versions will not participate.

Pending and failed versions will not participate.

### Workspace ownership is enforced at every layer

The requested workspace ID will scope:

- the HTTP route;
- active-version resolution;
- optional document filters;
- Qdrant search;
- chunk hydration;
- evidence validation.

A document identifier owned by another workspace will produce no eligible
scope and no evidence.

Workspace scoping remains a data ownership boundary rather than authentication
or authorization.

### Retrieval is profile-specific

The API process will build one immutable runtime retrieval profile.

Only active versions whose complete persisted profile equals that runtime
profile will participate.

The service will not silently switch providers, models, dimensions,
collections, tokenizers, or chunking versions.

Multi-profile search and score fusion will remain a separate capability.

### Empty scope avoids external work

The service will resolve eligible active versions before generating a query
embedding.

When no eligible version exists, it will return an empty successful result
without invoking the embedding provider or Qdrant.

### Query embeddings use an explicit operation

The application-owned embedding contract will use:

```text
operation = knowledge_query
```

One normalized query will produce exactly one vector.

Provider, model, dimensions, and vector count will be revalidated before
Qdrant search.

### Qdrant returns candidates, not evidence

The Qdrant adapter will request:

- point IDs;
- similarity scores;
- ownership metadata;
- chunk provenance metadata.

It will not request:

- stored vectors;
- source content;
- chunk content.

Qdrant candidates remain non-authoritative until PostgreSQL validation.

### Exact active target pairs

Qdrant filtering will combine:

```text
workspace
AND
(
    document A + active version A
    OR
    document B + active version B
)
```

Document IDs and version IDs will not be filtered as independent sets.

### Oversample before validation

Qdrant will return a bounded candidate set larger than the requested final
result count:

```text
min(100, max(top_k, top_k × 4))
```

This allows the service to discard invalid derived state while still returning
valid lower-ranked evidence.

### Bulk authoritative hydration

The service will bulk-load unique candidate chunk IDs from PostgreSQL using the
requested workspace.

Evidence content, hash, token count, section path, and source metadata will be
taken from PostgreSQL.

### Provenance must agree

A candidate will become evidence only when Qdrant metadata matches the active
version and PostgreSQL chunk for:

- workspace;
- document;
- document version;
- chunk ID;
- ordinal;
- content hash;
- media type;
- chunking strategy;
- chunking version;
- persisted index profile.

Missing or inconsistent candidates will be discarded and logged without
content.

### Deterministic ranking

Candidates will be ordered by:

```text
score descending
chunk_id ascending
```

Ranks will be assigned only after authoritative hydration and validation.

Final ranks will be contiguous and one-based.

### Retrieval returns evidence, not an answer

The HTTP endpoint will return:

- normalized query;
- eligible version count;
- ranked authoritative chunks;
- similarity scores;
- stable citation metadata.

It will not invoke an LLM or return a generated answer.

Grounded generation will consume this evidence through a later versioned
prompt and response contract.

### Process-scoped dependencies

The FastAPI lifespan will own one:

- embedding provider;
- Qdrant client;
- PostgreSQL engine and session factory;
- immutable retrieval profile.

SQLAlchemy sessions and retrieval services will remain request-scoped.

Every process resource will be independently released during shutdown and
partial-startup failure.

### Stable dependency errors

Expected embedding and vector-store failures will return HTTP 503 with:

```text
code = knowledge_retrieval_unavailable
message = Knowledge retrieval is temporarily unavailable.
```

Provider-specific and Qdrant-specific details will not cross the HTTP boundary.

## Consequences

### Positive consequences

- PostgreSQL remains the only authoritative content store;
- stale Qdrant payloads cannot silently become returned evidence;
- ready and active retain distinct operational meanings;
- inactive rollout candidates remain invisible to retrieval;
- cross-workspace filters reveal no content;
- empty scopes avoid provider cost and latency;
- exact document/version pairing prevents cross-pair leakage;
- projection inconsistencies degrade individual candidates rather than the
  complete request;
- deterministic tie-breaking makes output stable for equal scores;
- citation metadata identifies exact source versions and chunks;
- retrieval correctness can be evaluated independently from LLM generation;
- provider and vector-store failures use a stable API contract;
- process-scoped clients avoid per-request connection construction.

### Trade-offs

- retrieval performs PostgreSQL work before and after Qdrant search;
- oversampling increases the number of candidates hydrated and validated;
- invalid projection points can reduce returned evidence below `top_k`;
- incompatible active profiles are omitted rather than searched;
- a workspace with several supported profiles cannot search all of them in one
  request;
- cosine scores are exposed without application-defined relevance thresholds;
- evidence responses may include substantial chunk text;
- the API process now owns an embedding provider even when no search request is
  made;
- OpenAI retrieval configuration makes API startup depend on valid OpenAI
  credentials;
- returned evidence is not yet persisted as an auditable retrieval event;
- retrieval does not repair or delete stale Qdrant points.

### Required engineering practices

This decision requires:

- workspace-scoped active-version queries;
- direct active-pointer joins;
- ready-status filtering;
- complete profile equality;
- no embedding request for empty scope;
- one query vector per eligible request;
- exact active document/version Qdrant filters;
- payload indexes for workspace, document, and version identifiers;
- no source content in Qdrant payloads;
- bounded candidate oversampling;
- bulk PostgreSQL hydration;
- complete provenance validation;
- safe inconsistent-candidate discard;
- logs without source content or payload dumps;
- deterministic result ordering;
- stable HTTP dependency-error mapping;
- process lifecycle tests;
- real PostgreSQL and Qdrant isolation tests;
- no paid provider calls in the default automated suite.

## Alternatives considered

### Return chunk text from Qdrant payloads

Rejected because it would duplicate authoritative content and permit stale
derived state to become externally visible without PostgreSQL validation.

### Search every ready version

Rejected because readiness proves projection completeness but does not select
the version approved for retrieval.

The active pointer remains the rollout boundary.

### Search ready versions when no active version exists

Rejected because it would silently bypass explicit activation semantics.

An unactivated document contributes no evidence.

### Filter document IDs and version IDs with independent `MatchAny` clauses

Rejected because independent sets can match a document with another document's
active version.

Exact document/version pairs are required.

### Fail the complete request when one candidate is inconsistent

Rejected because Qdrant is rebuildable derived state.

A malformed, stale, or missing point should not suppress other valid evidence.

Expected infrastructure-wide failures still fail through the stable 503
contract.

### Generate query embeddings before resolving active scope

Rejected because empty scopes and cross-workspace document filters should not
incur provider cost or latency.

### Return Qdrant ranking without application normalization

Rejected because deterministic chunk-ID tie-breaking and post-validation rank
assignment are required for stable evidence output.

### Add a minimum similarity threshold now

Intentionally deferred.

A threshold requires retrieval evaluation data and a selected quality objective.
Applying an arbitrary value would create undocumented false-negative behavior.

### Add reranking now

Intentionally deferred.

Reranking introduces another model, latency budget, failure boundary, cost
dimension, and evaluation requirement.

### Search every embedding profile and merge results

Intentionally deferred.

Different models and vector spaces can produce scores that are not directly
comparable. Multi-profile retrieval requires explicit provider composition and
score-fusion policy.

### Generate an LLM answer in the search endpoint

Rejected for this delivery boundary.

Retrieval correctness, isolation, and evidence provenance must be established
before introducing prompt construction, generation failures, hallucination
controls, citation placement, token cost, and answer evaluation.

### Repair stale Qdrant points during read requests

Rejected because retrieval is read-only.

Projection repair remains an explicit indexing or future reconciliation
operation rather than a hidden side effect of search.