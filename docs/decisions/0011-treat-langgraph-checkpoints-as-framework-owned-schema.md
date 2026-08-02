# ADR 0011: Treat LangGraph Checkpoints as Framework-Owned Schema

## Status

Accepted

## Context

The controlled support workflow uses LangGraph's PostgreSQL checkpointer.

The PostgreSQL saver creates and evolves its internal tables through checkpointer setup:

```text
checkpoint_migrations
checkpoints
checkpoint_blobs
checkpoint_writes
```

SupportOps AI Platform also uses SQLAlchemy and Alembic for application-owned business schema.

Application-owned tables include durable records such as:

- `agent_runs`;
- `agent_run_attempts`;
- `llm_invocations`;
- `ticket_classifications`;
- `agent_tool_calls`;
- `support_recommendations`;
- `support_recommendation_citations`.

The checkpoint tables are intentionally absent from the application's SQLAlchemy metadata.

Without an explicit ownership rule, Alembic autogenerate sees reflected checkpoint tables that are absent from `target_metadata` and proposes destructive removal operations.

The architecture must decide whether to:

- map checkpoint tables into the application ORM;
- recreate them through application Alembic migrations;
- ignore all unknown database tables;
- isolate checkpoints in another database;
- or exclude the exact framework-owned objects from application schema comparison.

## Decision

LangGraph checkpoint tables will be treated as framework-owned schema.

### Checkpointer setup owns creation and internal migration

The PostgreSQL checkpoint runtime calls the framework setup operation before graph execution.

The framework owns the structure and internal migration history of:

```text
checkpoint_migrations
checkpoints
checkpoint_blobs
checkpoint_writes
```

Application Alembic migrations will not recreate or alter these tables.

### Application ORM metadata excludes checkpoint records

The application will not add SQLAlchemy business models for framework checkpoint tables.

Graph checkpoints are not business entities and do not participate in application repository contracts.

### Alembic excludes exact framework-owned objects

Alembic autogenerate uses an `include_object` callback that excludes:

- the four exact checkpoint table names;
- indexes whose owning table is one of those exact names.

The filter does not use broad prefixes and does not ignore arbitrary reflected tables.

All application-owned schema remains subject to Alembic comparison and migration checks.

### Business cleanup does not delete checkpoints

Shared integration cleanup removes application business rows in foreign-key-safe order.

It does not truncate or delete framework checkpoint tables.

Tests that create checkpoint threads are responsible for deleting their own thread-specific checkpoint rows when cleanup is required.

### Inspection does not expose checkpoint blobs

The controlled support inspection API reads durable business records.

It does not deserialize:

- checkpoint blobs;
- checkpoint metadata;
- checkpoint writes;
- internal task paths.

Checkpoint state remains an orchestration implementation detail.

## Consequences

### Positive consequences

- LangGraph can evolve its internal schema through its supported setup mechanism.
- Application migrations do not duplicate framework migration logic.
- `alembic check` remains meaningful for application-owned tables.
- Alembic no longer proposes dropping active checkpoint storage.
- Business ORM models remain focused on application concepts.
- Inspection contracts remain independent from framework serialization.
- Framework schema upgrades can be tested separately from business migrations.
- Exact-name filtering limits the exclusion boundary.

### Trade-offs

- Two schema-management mechanisms operate in the same PostgreSQL database.
- Operators must run both application migrations and checkpoint setup.
- Checkpoint backup and retention require explicit operational planning.
- Alembic alone does not describe the complete physical database schema.
- Framework upgrades may change checkpoint setup behavior.
- Integration tests must distinguish business cleanup from checkpoint cleanup.
- Database permissions must allow both Alembic-managed and checkpointer-managed objects.

### Required engineering practices

This decision requires:

- explicit checkpoint runtime setup;
- exact table-name documentation;
- exact Alembic `include_object` filtering;
- index exclusion tied to owning checkpoint tables;
- no broad unknown-table suppression;
- continued `alembic check` validation;
- checkpoint setup integration tests;
- thread-scoped checkpoint cleanup in tests that create durable graph state;
- no checkpoint ORM mapping;
- no application migration that drops or mutates framework-owned tables;
- independent resource cleanup for checkpoint connection pools;
- sanitized handling of checkpoint connection strings.

## Alternatives considered

### Add checkpoint tables to application SQLAlchemy metadata

Rejected because those tables are not application-owned domain records.

Mapping them would couple business persistence code to framework internals and could conflict with future LangGraph schema changes.

### Recreate checkpoint tables through Alembic migrations

Rejected because it would duplicate framework migration ownership.

The application would need to track implementation details and migration sequencing that belong to the checkpoint library.

### Ignore every reflected table missing from metadata

Rejected because broad exclusion would hide accidental or unauthorized schema drift.

Only the known framework-owned tables and their indexes are excluded.

### Place checkpoints in a separate PostgreSQL database

Deferred because the current portfolio and local runtime do not require a separate database lifecycle.

A separate database may become appropriate when retention, permissions, scaling, or operational ownership diverges materially from business persistence.

### Delete checkpoint tables before running Alembic checks

Rejected because it would destroy durable workflow progress and make schema validation unsafe.

### Expose checkpoint state through the inspection API

Rejected because checkpoint serialization is framework-specific and may contain internal orchestration details.

The API exposes stable application-owned records instead.

## Related documentation

- [`../architecture/controlled-support-workflow.md`](../architecture/controlled-support-workflow.md)
- [`../architecture/runtime-topology.md`](../architecture/runtime-topology.md)
- [`0010-separate-agent-run-and-langgraph-durability.md`](0010-separate-agent-run-and-langgraph-durability.md)