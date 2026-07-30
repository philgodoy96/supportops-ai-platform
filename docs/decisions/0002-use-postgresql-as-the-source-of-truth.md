# ADR 0002: Use PostgreSQL as the Source of Truth

## Status

Accepted

## Context

SupportOps AI Platform will manage durable operational state across support tickets, classifications, approvals, audit events, usage records, asynchronous work, and future AI-assisted workflows.

The platform requires a persistence model that supports:

- transactional consistency;
- relational constraints;
- durable workflow state;
- explicit ownership of authoritative data;
- idempotent processing;
- concurrency control;
- operational auditing;
- reliable schema evolution;
- local and CI reproducibility.

The system will also use specialized infrastructure such as Qdrant for semantic retrieval. These systems provide capabilities that PostgreSQL is not intended to replace, but they must not become independent owners of authoritative business state.

The future asynchronous worker model will require durable coordination and recoverable processing state. That coordination must remain visible, queryable, and transactionally consistent with the business operations it supports.

## Decision

PostgreSQL will be the transactional source of truth for SupportOps AI Platform.

Authoritative platform state will be persisted in PostgreSQL, including future records for:

- workspaces and ownership boundaries;
- support tickets;
- classifications;
- workflow state;
- approval requests and decisions;
- audit events;
- usage events;
- asynchronous jobs;
- idempotency records;
- ingestion metadata required to rebuild retrieval indexes.

Specialized systems may hold derived or operationally optimized representations, but they must not become the only location of authoritative state.

The platform will use SQLAlchemy 2.x with async support and `asyncpg` for application database access.

Alembic will manage schema evolution.

Business invariants that can be enforced relationally should use appropriate database constraints in addition to application-level validation.

The repository foundation phase will configure PostgreSQL connectivity, metadata, session management, and migration tooling without introducing business tables.

## Consequences

### Positive consequences

- related state transitions can be committed atomically;
- relational constraints can protect critical invariants;
- asynchronous processing state can be coordinated with business state;
- audit and usage records remain queryable in one authoritative system;
- idempotency can be implemented using durable uniqueness guarantees;
- failure recovery does not depend on in-memory process state;
- schema evolution is explicit and reviewable;
- local development and CI can use the same database technology as the intended platform runtime;
- derived systems can be rebuilt from authoritative data.

### Trade-offs

- PostgreSQL becomes a critical operational dependency;
- schema and migration discipline are required;
- connection pool sizing must account for separate API and worker processes;
- long-running transactions can create contention and must be avoided;
- asynchronous workloads require careful claim, lease, and retry semantics;
- database-backed coordination may require future partitioning, indexing, or archival strategies as workload grows;
- specialized analytical and retrieval workloads should not be forced into PostgreSQL when another system has a clearer purpose.

### Required engineering practices

The decision requires:

- explicit transaction boundaries;
- migration review before deployment;
- constraints for durable invariants;
- idempotency keys and uniqueness where external retries are possible;
- concurrency tests for state transitions;
- bounded query execution;
- connection lifecycle ownership per process;
- backup and recovery planning before production deployment;
- no use of derived indexes as authoritative state.

## Alternatives considered

### Qdrant as an authoritative document store

Rejected because Qdrant is optimized for vector search rather than transactional business state, relational constraints, or general workflow coordination.

Qdrant remains appropriate for rebuildable retrieval indexes.

### Separate databases for each initial module

Rejected for the initial platform because the modular monolith does not yet have independent operational ownership or deployment boundaries that justify distributed transactions and cross-database coordination.

Clear table and module ownership can be maintained within one PostgreSQL database.

### Document database as the primary store

Rejected because the platform requires relational integrity, explicit workflow transitions, durable auditability, and transactional coordination across related records.

A document database may be appropriate for specific future workloads, but it is not the default source of truth.

### In-memory or process-local workflow state

Rejected because application and worker processes can restart independently.

Authoritative workflow state must survive process failure and remain visible for recovery and auditing.

### External message broker as the primary work ledger

Rejected for the initial platform because message delivery state alone does not provide the transactional relationship required between business state and asynchronous work.

A broker may be introduced later when throughput, fan-out, or integration requirements justify it, while PostgreSQL remains authoritative for workflow state.
