# ADR 0003: Use Qdrant as a Rebuildable Retrieval Index

## Status

Accepted

## Context

SupportOps AI Platform will eventually provide retrieval-augmented workflows over internal runbooks and operational knowledge.

Semantic retrieval requires infrastructure optimized for:

- vector storage;
- similarity search;
- metadata filtering;
- retrieval latency;
- index lifecycle management.

PostgreSQL remains the transactional source of truth for platform and workflow state. A vector database serves a different purpose: it stores derived representations that improve retrieval performance.

The architecture must prevent retrieval infrastructure from becoming an independent owner of authoritative content.

Without an explicit ownership rule, the platform could lose the ability to:

- reproduce retrieval data;
- recover from index corruption;
- change embedding models;
- rebuild collections;
- verify source provenance;
- audit ingestion behavior;
- migrate retrieval providers.

## Decision

Qdrant will be used as a rebuildable retrieval index.

Qdrant may store future derived retrieval data such as:

- vector embeddings;
- chunk identifiers;
- source references;
- retrieval metadata required for filtering;
- embedding model and index version identifiers.

Qdrant will not be the source of truth for:

- original runbook content;
- document ownership;
- ingestion workflow state;
- business workflow state;
- approval state;
- audit records;
- usage records.

Authoritative source content and the metadata required to reproduce the retrieval index must remain outside Qdrant in systems with explicit ownership.

The initial platform direction uses PostgreSQL for authoritative ingestion and workflow metadata. Original source files may later be stored in an appropriate object store when a concrete ingestion capability requires it.

Retrieval collections must be treated as replaceable infrastructure.

The platform must be able to:

- recreate collections;
- repopulate vectors;
- change embedding versions;
- rebuild indexes after corruption or migration;
- trace indexed records back to authoritative sources.

During the repository foundation phase, Qdrant integration is limited to:

- local service configuration;
- environment-based client configuration;
- optional API key support;
- client lifecycle management;
- bounded connectivity checks.

No collections, embeddings, ingestion pipeline, chunking strategy, or retrieval behavior are implemented.

## Consequences

### Positive consequences

- retrieval indexes can be rebuilt after corruption or configuration changes;
- embedding providers and models can evolve without changing source ownership;
- the platform can validate provenance from retrieved chunks back to authoritative content;
- Qdrant can be replaced if operational requirements change;
- retrieval-specific schema decisions do not redefine business data ownership;
- backup requirements for Qdrant can be evaluated independently from authoritative transactional recovery;
- index versioning and migration can be treated as controlled derived-data operations.

### Trade-offs

- the platform must maintain enough source and ingestion metadata to reproduce the index;
- rebuilding large collections can require significant time and provider cost;
- index freshness must be monitored relative to authoritative content;
- dual-write behavior must be avoided or coordinated carefully;
- deletion and update flows must propagate from authoritative state to derived indexes;
- retrieval availability may temporarily degrade during rebuilds;
- retrieval tests must account for embedding and index version compatibility.

### Required engineering practices

The decision requires:

- stable source identifiers;
- explicit index and embedding version metadata;
- idempotent ingestion operations;
- source-to-vector traceability;
- controlled collection creation and migration;
- reconciliation between authoritative content and indexed representations;
- failure handling for partial indexing;
- clear deletion and retention semantics;
- evaluation before changing chunking, embeddings, or retrieval configuration.

## Alternatives considered

### Store authoritative runbook content only in Qdrant

Rejected because vector storage is not the appropriate ownership boundary for original content, transactional ingestion state, or operational auditability.

A retrieval index should remain recoverable from authoritative data.

### Use PostgreSQL vector extensions as the initial retrieval engine

Not selected for the approved architecture because Qdrant provides a dedicated vector retrieval boundary and makes the derived-index role explicit.

PostgreSQL vector capabilities remain a valid alternative for systems that prioritize a smaller infrastructure footprint or tighter transactional coupling.

### Use an in-memory vector index

Rejected as the platform retrieval foundation because process-local indexes do not provide durable shared access across API and worker processes and are lost on restart.

In-memory indexes may remain useful in isolated tests.

### Add multiple vector providers behind a generic abstraction immediately

Rejected because no retrieval capability exists yet and provider-neutral interfaces would be speculative.

An application-owned retrieval boundary will be introduced when concrete ingestion and query behavior is defined.

### Treat Qdrant backups as the only recovery mechanism

Rejected because backups alone do not preserve a clear authoritative ownership model or guarantee that indexes can be regenerated after embedding and schema changes.

Backups may reduce recovery time, but rebuildability remains a required architectural property.
