# Workspace-Scoped Data Ownership

## Purpose

This document defines the data ownership boundary introduced by the first
SupportOps business domain.

Every support ticket belongs to exactly one workspace. Application and
persistence operations require the workspace identifier whenever a ticket is
retrieved or listed.

This boundary prevents accidental unscoped access patterns and establishes the
ownership model that future authentication and authorization controls will
enforce.

## Ownership model

The current ownership relationship is:

```text
Workspace
└── Ticket
```

A workspace is the top-level owner of support ticket data.

Each ticket contains an immutable `workspace_id` that references an existing
workspace. Tickets cannot exist without that ownership relationship.

The current release does not support moving a ticket between workspaces.

## Repository boundary

Ticket repository operations require workspace scope explicitly.

Supported repository direction:

```text
add(ticket)
get(workspace_id, ticket_id)
list(workspace_id, ...)
```

An unscoped operation such as the following is intentionally absent:

```text
get(ticket_id)
```

This contract makes accidental global access more difficult and keeps workspace
ownership visible at the application boundary.

## Persistence guarantees

PostgreSQL is the transactional source of truth.

The schema enforces the following ownership and integrity rules:

- `tickets.workspace_id` is required.
- `tickets.workspace_id` references `workspaces.id`.
- workspace deletion is not implemented.
- the foreign key uses restrictive delete behavior.
- ticket external references are unique within one workspace.
- the same external reference may exist in different workspaces.
- ticket list queries are indexed by workspace and deterministic ordering.

The workspace-scoped external-reference constraint protects the invariant under
concurrent inserts. Application checks may improve error quality, but the
database remains the final concurrency-safe enforcement layer.

## Cross-workspace behavior

A ticket belongs to one workspace only.

A lookup using the correct ticket identifier and a different workspace
identifier produces the same application result as a missing ticket.

The HTTP API translates both conditions to `404 Not Found` with the
`ticket_not_found` error code. This avoids revealing that a resource exists
behind another ownership boundary.

## HTTP API boundary

Slice 1 exposes nested workspace-scoped routes:

```text
POST /api/v1/workspaces
GET  /api/v1/workspaces/{workspace_id}
POST /api/v1/workspaces/{workspace_id}/tickets
GET  /api/v1/workspaces/{workspace_id}/tickets/{ticket_id}
GET  /api/v1/workspaces/{workspace_id}/tickets
```

Ticket operations are always nested under a workspace identifier.

Listing tickets for a missing workspace returns `404 Not Found` with
`workspace_not_found`. Listing tickets for an existing workspace with no
tickets returns HTTP `200` with an empty `items` collection.

Opaque cursor pagination is encoded at the API boundary. Clients receive an
opaque `next_cursor` value and must not decode or modify it. Invalid cursors
return `400 Bad Request` with `invalid_pagination_cursor`.

Default page size is `20`. Maximum page size is `100`. Values outside that
range return FastAPI validation responses with status `422`.

Accepted tickets persist `ingestion_request_id` and `correlation_id` from the
active HTTP request context.

## Workspace scoping and tenant security

Workspace scoping is necessary, but it is not complete tenant isolation.

The current API does not establish trusted caller identity because
authentication and authorization are not part of this release. Workspace
scoping is not authentication or authorization.

The implemented boundary provides:

- explicit data ownership;
- repository-level resistance to unscoped ticket access;
- database-enforced ownership;
- nested HTTP routes that require a workspace identifier;
- cross-workspace behavior that avoids resource disclosure;
- a clear integration point for future authorization checks.

It does not provide:

- user authentication;
- workspace membership verification;
- role-based authorization;
- API key validation;
- caller-to-workspace trust establishment.

The current API is therefore appropriate for local and internal portfolio
demonstration. Public multi-tenant exposure requires an identity and
authorization layer that verifies whether the caller may act within the
requested workspace.

## Transaction ownership

Application use cases own transaction boundaries.

Repositories may:

- add records;
- execute queries;
- flush pending changes;
- map persistence records to domain entities;
- translate expected named constraints into stable domain-facing errors.

Repositories do not commit independently.

This keeps one use case atomic and allows all coordinated changes to roll back
together when any step fails.

The SQLAlchemy transaction adapter is intentionally minimal. It does not
introduce a generic Unit of Work framework or repository registry.

## Domain and persistence separation

Workspace and ticket entities are persistence-independent frozen dataclasses.

SQLAlchemy records own:

- table definitions;
- column types;
- constraints;
- indexes;
- domain-to-record mapping;
- record-to-domain mapping.

This separation prevents application code from depending on ORM session state
and keeps the domain reusable across the API and future worker processes.

The design intentionally avoids generic entity hierarchies, mapper frameworks,
and speculative domain abstractions.

## Ticket listing query

Ticket listing uses the following stable order:

```text
created_at DESC, id DESC
```

The supporting index is:

```text
workspace_id, created_at DESC, id DESC
```

The workspace identifier is the leading index column because every list query
is scoped to one workspace.

The identifier is used as a deterministic tie-breaker when multiple tickets
share the same creation timestamp.

Repository-level keyset navigation accepts both the last observed timestamp and
ticket identifier. The HTTP API encodes that position as an opaque cursor at
the transport boundary.

## Traceability fields

Each ticket persists:

- `ingestion_request_id`;
- `correlation_id`.

These values connect the accepted ticket to the HTTP request trace context.

They support operational investigation and future cross-process correlation.
They do not establish identity, authorization, or ownership by themselves.

## Intentional scope boundaries

This release does not introduce:

- authentication or authorization;
- workspace memberships;
- ticket mutation or deletion;
- workspace deletion;
- asynchronous processing;
- agent runs;
- queues, leases, retries, or workers;
- LLM classification;
- retrieval or vector indexing;
- external support integrations.

Ticket status remains `open` after intake. Asynchronous execution state will
be designed with the AgentRun and worker lifecycle so that claiming, leasing,
retries, crash recovery, idempotency, and terminal failure semantics are
defined together.