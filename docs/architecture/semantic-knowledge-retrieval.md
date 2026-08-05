# Semantic Knowledge Retrieval

## Purpose

This document defines the implemented semantic knowledge retrieval capability for SupportOps AI Platform.

The capability accepts a workspace-scoped natural-language query and returns ranked authoritative evidence from explicitly active, ready knowledge document versions.

The retrieval path performs:

- active-version resolution in PostgreSQL;
- query embedding generation;
- filtered dense-vector search in Qdrant;
- candidate provenance validation;
- authoritative chunk hydration from PostgreSQL;
- deterministic ranking;
- stable citation metadata projection.

The capability returns evidence. It does not generate an answer through an LLM.

## Architectural position

Semantic retrieval is implemented under:

```text
src/supportops/knowledge_retrieval/
```

The package owns:

- provider-independent retrieval contracts;
- PostgreSQL active-version resolution;
- PostgreSQL chunk hydration;
- Qdrant semantic candidate search;
- retrieval orchestration;
- HTTP request and response schemas;
- FastAPI dependency composition;
- the workspace-scoped search route.

Knowledge source ownership remains under:

```text
src/supportops/modules/knowledge_documents/
```

Embedding provider contracts and adapters remain under:

```text
src/supportops/ai/embeddings/
```

Knowledge indexing remains under:

```text
src/supportops/knowledge_index/
```

This separation keeps source management, indexing, retrieval, and later generation independently testable.

## HTTP contract

The implemented endpoint is:

```text
POST /api/v1/workspaces/{workspace_id}/knowledge/search
```

Example request:

```json
{
  "query": "How do I recover the database?",
  "top_k": 5,
  "document_ids": [
    "276046a2-28ec-4c33-884f-a4f36789a2ab"
  ]
}
```

Request fields:

- `query`: required nonblank text, normalized by trimming surrounding whitespace;
- `top_k`: optional result limit from 1 through 20, default 5;
- `document_ids`: optional list of at most 20 unique document UUIDs.

Unknown request fields are rejected.

The response includes:

- workspace ID;
- normalized query;
- requested `top_k`;
- document filters;
- number of compatible active versions searched;
- ranked evidence items.

Each evidence item includes:

- one-based rank;
- Qdrant similarity score;
- authoritative PostgreSQL chunk content;
- chunk content hash;
- token count;
- citation metadata.

The response does not include:

- a generated answer;
- an LLM prompt;
- chain-of-thought;
- Qdrant payload content;
- stored vectors;
- provider credentials;
- raw provider or Qdrant exceptions.

## End-to-end flow

```text
workspace-scoped request
→ resolve active ready document versions in PostgreSQL
→ omit versions incompatible with the runtime retrieval profile
→ return empty result if no eligible versions exist
→ generate one query embedding
→ search exact active document/version pairs in Qdrant
→ validate and deduplicate vector candidates
→ bulk-hydrate candidate chunks from PostgreSQL
→ compare vector provenance with authoritative chunk state
→ discard missing or inconsistent candidates
→ apply deterministic final ranking
→ return evidence and citations
```

## Source-of-truth boundary

### PostgreSQL

PostgreSQL is authoritative for:

- workspace ownership;
- document identity;
- document title and external reference;
- active-version selection;
- document-version readiness;
- media type;
- immutable index profile;
- chunk identity;
- chunk ordinal;
- chunk section path;
- chunk content;
- chunk content hash;
- chunk token count;
- chunking strategy and version.

### Qdrant

Qdrant is authoritative for no business or source-content state.

It provides a rebuildable candidate projection containing:

- point ID;
- embedding vector;
- workspace ID;
- document ID;
- document-version ID;
- chunk ID;
- chunk ordinal;
- chunk content hash;
- media type;
- chunking strategy;
- chunking version.

Qdrant payloads do not contain source or chunk content.

A Qdrant point is evidence only after its identifier and provenance match an authoritative PostgreSQL chunk belonging to an eligible active version.

## Active-ready retrieval scope

A document participates only when PostgreSQL confirms:

```text
document.workspace_id = requested workspace
document.active_version_id = document_version.id
document_version.document_id = document.id
document_version.workspace_id = requested workspace
document_version.status = ready
```

A ready version that is not active is excluded.

A pending or failed version is excluded.

A document without an active version is excluded.

Document filters are applied inside the same workspace-scoped PostgreSQL query.

A document ID owned by another workspace produces no eligible scope and reveals no evidence.

## Runtime retrieval profile

The API process builds one immutable `KnowledgeIndexProfile` at startup from validated settings.

The profile identifies:

- chunking strategy;
- chunking version;
- tokenizer encoding;
- embedding provider;
- embedding model;
- embedding dimensions;
- Qdrant collection;
- named vector.

A resolved active version participates only when its complete persisted profile equals the runtime profile.

This first retrieval implementation intentionally performs one profile-specific search per request.

It does not:

- instantiate multiple embedding providers;
- query multiple collections;
- normalize or merge scores across embedding models;
- fall back to another provider;
- silently migrate an active version to the runtime profile.

Incompatible active versions are omitted and logged using identifiers and a reason type without source content.

Multi-profile retrieval requires separate provider registry, score-comparability, cost, and failure-policy decisions.

## Query embeddings

The retrieval service submits one embedding request when at least one eligible active version exists.

The request uses:

```text
operation = knowledge_query
inputs = one normalized query
model = runtime profile embedding model
dimensions = runtime profile embedding dimensions
```

The response is revalidated for:

- provider identity;
- model identity;
- dimensions;
- exactly one vector.

Invalid provider responses fail through the application-owned embedding error boundary.

When no eligible version exists, retrieval returns an empty result without invoking the embedding provider or Qdrant.

## Qdrant semantic search

The Qdrant adapter uses:

```text
query_points
```

The search specifies:

- the persisted collection name;
- the persisted named vector;
- the validated query vector;
- exact workspace ownership;
- exact document/version target pairs;
- bounded candidate limit;
- selected metadata payload fields;
- no returned vectors.

The filter shape is logically:

```text
workspace_id = requested workspace

AND

(
    (document_id = document A AND document_version_id = active version A)
    OR
    (document_id = document B AND document_version_id = active version B)
)
```

Independent document and version `MatchAny` filters are not used because they would permit invalid cross-pair combinations.

Collection compatibility is verified before search.

## Candidate oversampling

The vector candidate limit is:

```text
min(100, max(top_k, top_k × 4))
```

Oversampling allows the service to discard invalid or stale candidates before producing the final `top_k` evidence items.

Examples of discarded candidates include:

- missing payload;
- invalid UUIDs;
- point ID different from chunk ID;
- other-workspace ownership;
- inactive document/version pair;
- incompatible chunking profile;
- duplicate chunk ID;
- missing PostgreSQL chunk;
- content-hash mismatch;
- ordinal mismatch;
- media-type mismatch;
- chunking provenance mismatch.

A discarded candidate does not fail the whole request.

## Authoritative hydration

After Qdrant search, the service bulk-loads unique candidate chunk IDs from PostgreSQL using the requested workspace ID.

The hydrator:

- returns only rows owned by the requested workspace;
- performs no Qdrant access;
- never reads source content from vector payloads;
- reconstructs candidate order by chunk ID in the service.

Missing rows are treated as inconsistent derived state and discarded.

## Provenance validation

An evidence item is created only when all of the following agree:

- candidate workspace;
- active-version workspace;
- PostgreSQL chunk workspace;
- candidate document ID;
- active-version document ID;
- PostgreSQL chunk document ID;
- candidate document-version ID;
- active document-version ID;
- PostgreSQL chunk document-version ID;
- candidate chunk ID;
- PostgreSQL chunk ID;
- candidate ordinal;
- PostgreSQL chunk ordinal;
- candidate content hash;
- PostgreSQL chunk content hash;
- candidate chunking strategy and version;
- PostgreSQL chunking strategy and version;
- active-version persisted profile;
- candidate media type;
- active-version media type.

The final evidence content and token count always come from PostgreSQL.

## Ranking

Qdrant candidates are normalized into deterministic order:

```text
score descending
chunk_id ascending
```

The chunk ID is a stable tie-breaker.

Ranks are assigned only after:

- candidate deduplication;
- active-target validation;
- PostgreSQL hydration;
- provenance validation.

Final ranks are contiguous and one-based:

```text
1, 2, 3, ...
```

An invalid higher-scoring candidate can therefore be removed while a valid lower-scoring candidate is promoted.

No application-side reranking model is implemented.

## Citation metadata

Every evidence item includes a citation containing:

- workspace ID;
- document ID;
- document title;
- document external reference when present;
- document-version ID;
- version number;
- chunk ID;
- chunk ordinal;
- section path;
- media type.

Citations identify the exact authoritative source fragment returned by retrieval.

They are retrieval evidence references. They are not generated-answer citation placement.

## API process lifecycle

The FastAPI process owns one process-scoped:

- PostgreSQL engine;
- SQLAlchemy session factory;
- Qdrant client;
- embedding provider;
- immutable retrieval profile.

Startup order is:

```text
configure logging
→ build retrieval profile
→ create embedding provider
→ create PostgreSQL engine and session factory
→ create Qdrant client
→ publish ApplicationState
```

The provider is created at startup but is not called during startup.

Per request:

- one SQLAlchemy session is created;
- one `SearchKnowledge` service is composed;
- the process-scoped provider is reused;
- the process-scoped Qdrant client is reused;
- the request-scoped session is closed after completion.

Shutdown attempts to release independently:

- embedding provider;
- Qdrant client;
- PostgreSQL engine.

A cleanup failure for one resource does not prevent cleanup attempts for the others.

Partially initialized startup resources are also released when startup fails.

## Error behavior

Expected embedding and vector-store failures return:

```text
503 Service Unavailable
```

Stable response:

```json
{
  "error": {
    "code": "knowledge_retrieval_unavailable",
    "message": "Knowledge retrieval is temporarily unavailable.",
    "request_id": "<request-id>"
  }
}
```

The API does not expose:

- provider-specific failure codes;
- provider request payloads;
- Qdrant response bodies;
- connection endpoints;
- credentials;
- raw exception messages.

Request validation failures use the existing FastAPI validation contract.

An empty eligible retrieval scope is not an error and returns HTTP 200 with empty evidence.

## Transaction behavior

Semantic retrieval is read-only.

The retrieval service does not open an application transaction around the complete request.

No database transaction spans:

- query embedding generation;
- Qdrant search;
- PostgreSQL hydration.

SQLAlchemy queries execute through the request-scoped session.

Retrieval does not mutate:

- document state;
- document-version state;
- active-version selection;
- chunks;
- vector points;
- ticket state;
- AgentRun state.

## Security and workspace ownership

The workspace ID is mandatory in:

- the HTTP route;
- active-version resolution;
- Qdrant filtering;
- chunk hydration;
- result validation.

Cross-workspace document filters produce no evidence.

Qdrant candidates with another workspace ID are discarded even when returned unexpectedly.

Citation workspace IDs must match the request workspace.

These controls establish data ownership boundaries. They do not establish caller identity or authorization.

The current endpoint is not suitable for public multi-tenant exposure until authentication and authorization verify that the caller may search the requested workspace.

## Logging

Candidate and active-version discard logs may contain:

- workspace ID;
- document ID;
- document-version ID;
- chunk ID;
- collection name;
- point ID;
- reason type.

Logs must not contain:

- source content;
- chunk content;
- complete Qdrant payloads;
- embedding vectors;
- embedding request text;
- API keys;
- complete settings.

## Testing strategy

### Unit coverage

Unit tests cover:

- query normalization and bounds;
- document-filter uniqueness and limits;
- profile and vector validation;
- active target identity;
- vector candidate provenance;
- authoritative evidence construction;
- result rank and score invariants;
- Qdrant filter construction;
- named-vector selection;
- payload field selection;
- malformed and duplicate candidate discard;
- empty-scope short-circuit;
- query embedding request construction;
- embedding response validation;
- oversampling;
- PostgreSQL hydration behavior;
- deterministic final ranking;
- API schemas;
- API route behavior;
- stable 503 mappings;
- API process lifecycle and partial-startup cleanup.

Provider and Qdrant unit tests use fakes and perform no network calls.

### Integration coverage

Integration tests use real PostgreSQL and Qdrant with the deterministic mock embedding provider.

They prove:

- workspace isolation;
- exact active-version filtering;
- ready-but-inactive exclusion;
- document filtering;
- cross-workspace filter nondisclosure;
- no embedding call for empty eligible scope;
- real vector search;
- authoritative PostgreSQL hydration;
- stale Qdrant hash rejection;
- safe candidate discard;
- rank reconstruction after discard;
- isolated collection cleanup.

The default automated suite does not call OpenAI.

## Deterministic evaluation

Deterministic semantic-retrieval regression is implemented against an immutable synthetic dataset and a committed static prediction fixture.

Within the committed synthetic regression corpus:

```text
evals/semantic-retrieval/datasets/semantic-retrieval-eval-v1.jsonl
evals/semantic-retrieval/predictions/semantic-retrieval-eval-v1.static.jsonl
```

The corpus contains 10 deterministic regression cases. Scoring consumes typed prediction envelopes and computes ranking and isolation metrics without live embeddings or Qdrant execution.

Deterministic metrics include:

- document hit rate at k;
- chunk hit rate at k;
- mean reciprocal rank;
- recall at k;
- no-result accuracy;
- workspace isolation rate;
- citation resolution rate;
- average latency;
- average query embedding tokens;
- estimated query cost.

Duplicate retrieved chunks count once. Cosine similarity score is ranking evidence only and is not treated as calibrated confidence.

Repository regression aggregation and release-gate profile `semantic-retrieval-release-gates / 1` are documented in [`evaluation-and-regression.md`](evaluation-and-regression.md).

## Intentional scope boundaries

Semantic retrieval currently does not implement:

- generated answers;
- RAG prompt construction;
- LLM invocation from retrieved evidence;
- generated-answer citation placement;
- reranking models;
- hybrid lexical and vector retrieval;
- query rewriting;
- multi-query retrieval;
- multi-profile search and score fusion;
- minimum score thresholds;
- MMR diversification;
- access-control policy evaluation;
- retrieval result persistence;
- retrieval usage or cost persistence;
- production relevance evaluation against live traffic;
- RAGAS;
- grounded-generation evaluation;
- LangGraph orchestration;
- tool calling;
- human approval;
- frontend search experiences.

These are separate engineering decisions because they introduce additional quality, reliability, cost, security, and observability contracts.

The next AI capability can consume the returned authoritative evidence through a versioned prompt and generated-answer schema without changing the implemented source-of-truth boundaries.