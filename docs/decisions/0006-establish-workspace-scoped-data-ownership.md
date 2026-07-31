# 0006 — Establish Workspace-Scoped Data Ownership

## Status

Accepted

## Context

SupportOps requires a clear ownership boundary before support tickets can be
accepted through the public application API.

Every ticket belongs to one workspace. Without an explicit workspace boundary,
repository and API designs could permit accidental global ticket access,
cross-workspace information disclosure, and unclear authorization integration
points.

The current release does not include authentication or authorization. The data
model must therefore distinguish application-level ownership scoping from
trusted tenant isolation.

## Decision

Every ticket operation requires workspace scope.

Ticket repository contracts require `workspace_id` for retrieval and listing:

```text
get(workspace_id, ticket_id)
list(workspace_id, ...)
```

No global ticket retrieval operation is provided.

Ticket routes will be nested under their workspace:

```text
/api/v1/workspaces/{workspace_id}/tickets
/api/v1/workspaces/{workspace_id}/tickets/{ticket_id}
```

A ticket lookup through the wrong workspace boundary will be treated as not
found. The API will return the same `404 Not Found` behavior for a missing ticket
and for a ticket owned by another workspace.

PostgreSQL enforces ticket ownership through a required foreign key from
`tickets.workspace_id` to `workspaces.id`.

External references are unique within one workspace rather than globally.

Workspace scoping is documented as a necessary data ownership boundary, not as
complete multi-tenant security.

Authentication and authorization remain a separate future boundary responsible
for establishing caller identity, workspace membership, and permitted actions.

## Consequences

### Positive

- Data ownership is explicit in the domain, persistence, repository, and future
  HTTP boundaries.
- Repository interfaces make accidental unscoped ticket access more difficult.
- Cross-workspace misses do not reveal resource existence.
- Database constraints enforce ownership and scoped uniqueness under
  concurrency.
- Future authorization checks have a clear workspace boundary to protect.
- Ticket listing queries can use workspace-leading indexes.

### Trade-offs

- Callers must always carry the workspace identifier when operating on tickets.
- Cross-workspace access cannot distinguish an unauthorized resource from a
  missing resource through the API response.
- Workspace scoping alone cannot verify whether a caller is trusted to act in a
  workspace.
- Future authentication and authorization must be integrated before public
  multi-tenant exposure.

## Alternatives considered

### Global ticket routes

A global route such as:

```text
/api/v1/tickets/{ticket_id}
```

was rejected because it hides ownership from the API shape and encourages
unscoped repository operations.

### Repository lookup by ticket identifier only

A repository contract such as:

```text
get(ticket_id)
```

was rejected because it allows application code to retrieve a ticket without
expressing the ownership boundary.

### Return `403 Forbidden` for cross-workspace access

Returning `403 Forbidden` was rejected for the current boundary because it
reveals that the ticket exists in another workspace.

The selected behavior returns `404 Not Found` for both missing and
cross-workspace resources.

### Treat workspace scoping as secure multi-tenancy

This was rejected because workspace identifiers do not establish trusted caller
identity.

Secure multi-tenancy requires authentication, authorization, membership checks,
and policy enforcement in addition to data scoping.