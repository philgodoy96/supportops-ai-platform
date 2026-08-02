# Versioned Knowledge Documents

## Purpose

The `knowledge_documents` module establishes the PostgreSQL-authoritative source
content required by the platform's indexing and semantic retrieval pipeline.

The module provides:

- workspace-owned document identities;
- immutable plain-text and Markdown versions;
- deterministic content normalization and hashing;
- concurrency-safe version creation;
- authoritative document chunks;
- explicit indexing lifecycle state;
- explicit ready-version activation;
- workspace-scoped HTTP management.

Document registration remains source-only. Token-aware chunk generation,
embeddings, and Qdrant indexing belong to the separate explicit indexing
pipeline documented in [`knowledge-indexing.md`](knowledge-indexing.md).
Semantic evidence retrieval over active ready versions is documented in
[`semantic-knowledge-retrieval.md`](semantic-knowledge-retrieval.md). Grounded
answer generation remains a later capability.

## Module boundary

The module lives under:

```text
src/supportops/modules/knowledge_documents/
├── api/
├── application/
├── domain/
└── infrastructure/
```

The internal dependency direction is:

```text
API
→ application
→ domain contracts

infrastructure
→ domain and application contracts

composition
→ concrete infrastructure adapters
```

The domain layer has no dependency on FastAPI, SQLAlchemy, OpenAI, Qdrant, or a
tokenizer library.

The application layer owns use-case orchestration and transaction boundaries.
Repositories query, add, update, flush, map, and lock; they do not commit.

## Domain responsibilities

### Document

`Document` owns stable workspace-scoped identity and rollout state.

Its responsibilities are:

- identify one logical runbook or knowledge source;
- retain the immutable `workspace_id`;
- retain a required bounded title;
- retain an optional workspace-local external reference;
- point to the explicitly selected active version;
- record creation and update timestamps.

Source content does not live on `Document`.

Titles may repeat. An external reference, when supplied, is unique within one
workspace and may be reused in another workspace.

Documents are not deleted in the current scope.

### DocumentVersion

`DocumentVersion` owns immutable accepted source content and indexing lifecycle
metadata.

Its responsibilities are:

- identify one immutable source revision;
- retain workspace and document ownership;
- retain a positive version number;
- retain the supported media type;
- retain normalized source content;
- retain the deterministic SHA-256 content hash;
- retain indexing state;
- retain an immutable indexing profile once assigned;
- retain aggregate indexing usage, cost, and completion provenance.

Supported media types are:

```text
text/plain
text/markdown
```

PDF, DOCX, HTML scraping, OCR, remote URL fetching, binary upload, and object
storage are intentionally outside the current boundary.

### DocumentChunk

`DocumentChunk` is the authoritative PostgreSQL representation of one
deterministic chunk.

Its responsibilities are:

- retain workspace, document, and version ownership;
- retain a zero-based ordinal;
- retain the ordered section path;
- retain authoritative chunk text;
- retain the deterministic chunk hash;
- retain token count;
- retain chunking strategy and version;
- retain a deterministic UUIDv5 identity.

Chunk rows are introduced with the source-of-truth model so the indexing
pipeline can persist and verify deterministic chunks without making Qdrant an
authoritative content store.

Chunk generation is performed by the explicit indexing command. The HTTP API
does not create chunks during document or version registration.

## Content normalization and hashing

Accepted source content is normalized before storage.

The normalization boundary:

1. removes one leading UTF-8 byte-order mark when present;
2. converts `CRLF` line endings to `LF`;
3. converts remaining `CR` line endings to `LF`;
4. rejects content containing only whitespace.

The boundary intentionally does not:

- trim the complete document;
- rewrite paragraph spacing;
- remove trailing spaces;
- reflow Markdown;
- normalize heading syntax;
- rewrite code-block contents;
- change Unicode case.

`content_sha256` is calculated from the UTF-8 bytes of the exact normalized
content stored in PostgreSQL.

Consequences:

- line endings do not produce environment-specific hashes;
- hashing and indexing consume the same authoritative input;
- meaningful Markdown and code formatting remain unchanged;
- duplicate normalized content can be rejected deterministically.

The database enforces uniqueness of:

```text
(document_id, content_sha256)
```

Equivalent content may exist in another document.

## Version numbering and concurrency

Version numbers begin at `1` and increase within a document.

Creating a new version uses this transaction:

```text
lock owning Document row
→ verify the document exists in the workspace
→ normalize and hash submitted content
→ reject an existing hash for the document
→ calculate the next version number
→ insert the immutable pending version
→ commit
```

The owning document row is the serialization point. Two concurrent requests for
distinct content receive different version numbers.

Database uniqueness on:

```text
(document_id, version_number)
```

remains the final concurrency backstop.

Two concurrent requests with equivalent normalized content produce one
persisted version and one stable content conflict.

No database transaction remains open for an external provider call because
document registration performs no external call.

## Indexing lifecycle

Persisted version states are:

```text
pending
failed
ready
```

There is no durable `indexing` state in this phase.

A new version begins as `pending`.

Pending versions may be indexed explicitly through the
`supportops-index-knowledge` CLI. That command binds or validates the immutable
indexing profile, persists or verifies deterministic chunks, performs embedding
and Qdrant operations outside database transactions, verifies the exact version
projection count, and then moves the version to `ready` or `failed`.

Failed versions retain their assigned profile and authoritative chunks. Compatible
failed versions may retry. Ready versions may be re-run as a no-op.

The indexing profile contains:

- chunking strategy;
- chunking version;
- tokenizer encoding;
- embedding provider;
- embedding model;
- embedding dimensions;
- Qdrant collection identity;
- named vector identity.

Profile fields must be all absent or all present.

Once assigned, the profile cannot change silently. A failed version may be
retried only with the same profile.

A ready version contains:

- positive chunk count;
- embedding input-token usage;
- pricing-catalog provenance;
- optional estimated cost;
- indexing completion timestamp;
- no failure code;
- a complete verified Qdrant projection for the version.

Unknown pricing remains `null`; it is never treated as zero.

Ready means the indexing projection completed and was verified. Ready does not
mean active.

## Ready state and active state

Readiness and activation represent different decisions.

`ready` means the version's indexing pipeline completed successfully and the
exact Qdrant projection count was verified.

`active` means the document currently exposes that ready version to semantic
retrieval.

The active state is stored only as:

```text
knowledge_documents.active_version_id
```

PostgreSQL owns this pointer.

Activation is explicit and transactional:

```text
lock Document row
→ load the owned DocumentVersion
→ require status = ready
→ update active_version_id
→ commit
```

A database trigger verifies that the selected version:

- exists;
- belongs to the same workspace;
- belongs to the same document;
- is ready.

A second trigger prevents any update to a ready version. Prior versions remain
immutable after another version becomes active.

Activation is idempotent when the selected version is already active.

A successfully indexed version may remain ready but inactive. Ready but inactive
versions are not searched. Activation remains explicit: changing the active
pointer updates retrieval eligibility but performs no indexing and no provider
call. This supports controlled rollout and validation before retrieval
eligibility changes.

## Indexing integration

Document registration remains source-only. Creating a document or version
through the HTTP API persists PostgreSQL state and does not call an embedding
provider or Qdrant.

The explicit indexing pipeline:

- binds or validates one immutable index profile per document version;
- persists authoritative chunks as PostgreSQL records;
- writes only derived vectors and non-content metadata into Qdrant;
- marks the version ready only after exact projection-count verification;
- does not activate the version.

Successful indexing therefore leaves `active_version_id` unchanged until an
operator activates a ready version through the API.

Indexing architecture is documented in
[`knowledge-indexing.md`](knowledge-indexing.md). Active-version semantic
retrieval is documented in
[`semantic-knowledge-retrieval.md`](semantic-knowledge-retrieval.md). The
explicit profiled indexing decision is recorded in
[`../decisions/0008-use-explicit-profiled-knowledge-indexing.md`](../decisions/0008-use-explicit-profiled-knowledge-indexing.md).

## Persistence model

PostgreSQL stores:

```text
knowledge_documents
knowledge_document_versions
knowledge_document_chunks
```

### Ownership constraints

The schema enforces:

- document ownership by workspace;
- version ownership by workspace and document;
- chunk ownership by workspace, document, and version;
- active-version ownership by workspace and document;
- chunking-profile agreement between a chunk and its version.

The chunk foreign key includes:

```text
workspace_id
document_id
document_version_id
chunking_strategy
chunking_version
```

This prevents a chunk from referencing the correct version with a mismatched
chunking profile.

### Query-driven indexes

The initial indexes support:

- document listing by workspace, creation timestamp, and ID;
- version listing by workspace, document, version number, and ID;
- active-version resolution by workspace;
- chunk hydration by workspace, version, and chunk ID.

No analytics indexes are introduced without an implemented query.

## API contract

Implemented routes:

```text
POST /api/v1/workspaces/{workspace_id}/documents
GET  /api/v1/workspaces/{workspace_id}/documents
GET  /api/v1/workspaces/{workspace_id}/documents/{document_id}

POST /api/v1/workspaces/{workspace_id}/documents/{document_id}/versions
GET  /api/v1/workspaces/{workspace_id}/documents/{document_id}/versions
GET  /api/v1/workspaces/{workspace_id}/documents/{document_id}/versions/{document_version_id}

POST /api/v1/workspaces/{workspace_id}/documents/{document_id}/versions/{document_version_id}/activate
```

### Source-content exposure

Document creation, document listing, document detail, version creation, version
listing, and activation responses do not return source content.

Only the version detail route returns the authoritative normalized source
content.

This keeps collection responses bounded while preserving an explicit inspection
endpoint.

### Pagination

Document and version listings use separate opaque keyset cursor contracts.

Document order:

```text
created_at DESC, id DESC
```

Version order:

```text
version_number DESC, id DESC
```

A document cursor cannot be used for a version listing and a version cursor
cannot be used for a document listing.

Page size defaults to `20` and is bounded at `100`.

### Error behavior

Expected error codes include:

```text
document_not_found
document_version_not_found
document_external_reference_conflict
document_version_content_conflict
document_version_number_conflict
document_version_not_ready
invalid_pagination_cursor
```

Missing and cross-workspace resources produce the same `404` behavior.

Raw PostgreSQL exceptions are translated before reaching the HTTP boundary.

## Transaction boundaries

Application services own transactions.

### Document creation

```text
verify workspace
→ add Document
→ add version 1
→ commit atomically
```

A failure inserting either record rolls back both.

### Version creation

```text
lock Document
→ verify duplicate content
→ allocate version number
→ insert version
→ commit
```

### Activation

```text
lock Document
→ lock owned DocumentVersion
→ require ready
→ update active pointer
→ commit
```

Read operations do not open application-owned write transactions.

## Security and privacy

Every document operation is scoped by `workspace_id`.

Every version operation is scoped by:

```text
workspace_id
document_id
document_version_id
```

Cross-workspace access returns `404` and does not reveal whether the identifier
exists elsewhere.

The current workspace boundary is data ownership, not authentication or
authorization.

Source content is untrusted data. The current module stores and returns it; it
does not:

- execute instructions from documents;
- alter application control flow;
- call an LLM;
- invoke tools;
- mutate tickets;
- activate versions automatically.

Document and version content must not appear in structured operational logs.

## Failure behavior

Stable conflicts are returned for:

- duplicate workspace-local external references;
- duplicate normalized content within a document;
- version-number conflicts;
- activation of pending or failed versions.

The database remains the final authority for uniqueness and ownership.

A failed transaction leaves no partial document/version pair and no partially
allocated version number.

## Testing guarantees

The test suite covers:

- domain invariants and immutable entities;
- normalization and SHA-256 hashing;
- deterministic chunk identity;
- SQLAlchemy mapping and metadata;
- migration upgrade, downgrade, and re-upgrade;
- named ownership, uniqueness, and lifecycle constraints;
- trigger-enforced ready-only activation;
- trigger-enforced ready-version immutability;
- repository workspace scoping;
- idempotent chunk persistence;
- application transaction boundaries;
- concurrency-safe distinct-content version creation;
- concurrent equivalent-content conflict behavior;
- HTTP schema validation;
- opaque document and version cursors;
- source-content response boundaries;
- cross-workspace `404` behavior;
- pending and ready activation behavior.

Normal tests make no paid provider calls.

## Intentional scope boundaries

The following capabilities are deliberately deferred:

- reranking;
- retrieval evaluation;
- grounded answer generation;
- document deletion;
- automatic activation;
- automatic indexing scheduling;
- automated high-volume ingestion;
- PDF, DOCX, OCR, image, or web ingestion;
- authentication and authorization.

Token-aware chunk generation, embedding providers, embedding usage and cost
calculation, Qdrant collection creation, and vector-point indexing are
implemented by the separate knowledge indexing pipeline. Active-version semantic
evidence retrieval with authoritative PostgreSQL hydration is implemented by the
separate knowledge retrieval package.

These deferred capabilities require the source-content, ownership, immutability,
indexing, rollout, and retrieval guarantees established by this module and the
adjacent indexing and retrieval packages.
