# Approval Workflow API

## Purpose

The approval workflow API exposes operational inspection and decision endpoints for the versioned `human-approved-support-v1` workflow.

The API supports:

- workspace-scoped approval request listing;
- workspace-scoped approval request detail;
- explicit approve and reject decisions;
- workspace-scoped ticket escalation listing;
- workspace-scoped ticket escalation detail.

The API does not execute sensitive tools. Approval decisions update durable business state and requeue the associated `AgentRun`. The worker remains the sole owner of LangGraph resume and sensitive execution.

## Base Path

All endpoints are served below:

```text
/api/v1
```

## Approval Inspection

### List Approval Requests

```http
GET /api/v1/workspaces/{workspace_id}/approvals
```

Supported query parameters:

| Parameter | Type | Required | Description |
|---|---|---:|---|
| `status` | `pending`, `approved`, `rejected`, `expired` | No | Filters by current approval status. |
| `cursor` | opaque string | No | Continues stable keyset pagination. |
| `page_size` | integer | No | Number of records, from 1 through 100. Defaults to 20. |

Ordering is stable:

```text
created_at DESC
id DESC
```

Example:

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://localhost:8000/api/v1/workspaces/$WorkspaceId/approvals?status=pending&page_size=20" `
  -Headers @{
    "X-Correlation-ID" = $CorrelationId
  }
```

Response:

```json
{
  "items": [
    {
      "id": "5f424ef6-e34c-489e-918b-5795238c9d51",
      "workspace_id": "52b088ac-e57c-4d0c-89ad-8c86949b90ea",
      "ticket_id": "8dfa091a-e5e0-46a2-b746-6b4c2d82a9d4",
      "agent_run_id": "2c4a94ba-88fa-4b0d-957f-e945d3599ac6",
      "agent_tool_call_id": "b65f5521-bd83-43a0-b5a1-0db1dfdf70f1",
      "requested_by_llm_invocation_id": "1494e98d-86b4-4c99-8d4c-8308958e83fa",
      "status": "pending",
      "tool_name": "escalate_ticket",
      "tool_version": 1,
      "input_fingerprint": "f4e4c932...",
      "proposed_input": {
        "target_queue": "support_operations",
        "reason": "Operational review required."
      },
      "request_reason": "Operational review required.",
      "expires_at": "2026-08-04T22:00:00Z",
      "decision_actor_reference": null,
      "decision_comment": null,
      "decision_request_id": null,
      "decision_correlation_id": null,
      "decided_at": null,
      "created_at": "2026-08-03T22:00:00Z",
      "updated_at": "2026-08-03T22:00:00Z"
    }
  ],
  "next_cursor": null
}
```

### Get Approval Request

```http
GET /api/v1/workspaces/{workspace_id}/approvals/{approval_request_id}
```

Pending and terminal approvals are inspectable.

Missing records and records owned by another workspace return the same nondisclosing response:

```text
HTTP 404
approval_request_not_found
```

## Approval Decisions

### Approve

```http
POST /api/v1/workspaces/{workspace_id}/approvals/{approval_request_id}/approve
```

Request:

```json
{
  "actor_reference": "operator:alice",
  "decision_request_id": "ff3b8d56-4c46-439c-8146-c14484e8c0c7",
  "comment": "Approved after operational review."
}
```

`comment` is optional for approval.

PowerShell example:

```powershell
$Body = @{
  actor_reference = "operator:alice"
  decision_request_id = [guid]::NewGuid().ToString()
  comment = "Approved after operational review."
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/api/v1/workspaces/$WorkspaceId/approvals/$ApprovalRequestId/approve" `
  -ContentType "application/json" `
  -Headers @{
    "X-Correlation-ID" = $CorrelationId
  } `
  -Body $Body
```

### Reject

```http
POST /api/v1/workspaces/{workspace_id}/approvals/{approval_request_id}/reject
```

Request:

```json
{
  "actor_reference": "operator:alice",
  "decision_request_id": "977441b4-75e0-4084-9d94-7640ba1fc9ba",
  "comment": "Escalation is not required for this ticket."
}
```

`comment` is required for rejection.

PowerShell example:

```powershell
$Body = @{
  actor_reference = "operator:alice"
  decision_request_id = [guid]::NewGuid().ToString()
  comment = "Escalation is not required for this ticket."
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/api/v1/workspaces/$WorkspaceId/approvals/$ApprovalRequestId/reject" `
  -ContentType "application/json" `
  -Headers @{
    "X-Correlation-ID" = $CorrelationId
  } `
  -Body $Body
```

### Decision Response

Both applied decisions and identical replays return `HTTP 200`.

```json
{
  "approval_request_id": "5f424ef6-e34c-489e-918b-5795238c9d51",
  "workspace_id": "52b088ac-e57c-4d0c-89ad-8c86949b90ea",
  "agent_run_id": "2c4a94ba-88fa-4b0d-957f-e945d3599ac6",
  "status": "approved",
  "decision_actor_reference": "operator:alice",
  "decision_comment": "Approved after operational review.",
  "decision_request_id": "ff3b8d56-4c46-439c-8146-c14484e8c0c7",
  "decision_correlation_id": "d7c393dc-fc5d-4678-a47a-b8867bad126c",
  "decided_at": "2026-08-03T22:05:00Z",
  "idempotent": false
}
```

Decision requests return after durable requeue, not after worker execution.

## Decision Idempotency

The current decision replay identity is:

```text
same terminal status
+ same actor_reference
+ same comment
= idempotent replay
```

`decision_request_id` and correlation ID are persisted audit metadata. They are not the idempotency key.

Consequences:

- the same terminal status, actor, and comment returns `idempotent=true`, even with a different `decision_request_id`;
- the same terminal status with a different actor or comment returns `HTTP 409` with `approval_decision_conflict`;
- approving an already rejected request returns `HTTP 409` with `approval_decision_conflict`;
- rejecting an already approved request returns `HTTP 409` with `approval_decision_conflict`;
- terminal state is never overwritten.

## Actor and Request Metadata

`actor_reference` is supplied in the request body.

It is currently an asserted operational identity, not an authenticated or verified principal.

`decision_request_id` is supplied by the client and persisted with the decision.

`X-Correlation-ID` is propagated through the existing request context. When absent or invalid, the middleware applies the current request-context fallback behavior.

`decided_at` is generated server-side as a timezone-aware UTC timestamp.

Clients cannot supply:

- terminal status;
- decision correlation ID;
- decision timestamp;
- workspace ownership;
- execution data.

## Worker Boundary

Command and query ownership:

```text
HTTP API
-> inspect or decide
-> requeue AgentRun

Worker
-> claim
-> validate checkpoint and PostgreSQL
-> resume graph
-> execute sensitive action
```

A successful decision request performs:

```text
ApprovalRequest terminal transition
-> AgentToolCall rejection outcome when rejecting
-> AgentRun waiting_for_approval to queued
-> commit
```

Approval leaves `AgentToolCall` in `pending_approval` until the worker consumes an execution grant. Rejection records the non-executed tool-call outcome during the decision transaction. Approval decisions always validate the linked pending tool call before commit.

The HTTP request does not perform:

- LangGraph resume;
- worker claim;
- AgentRunAttempt creation;
- SensitiveExecutionGrant creation;
- TicketEscalation creation;
- sensitive tool execution;
- recommendation generation.

The API owns validation, inspection, approval and rejection command submission, durable terminal decision persistence, and `AgentRun` requeue. The worker owns claim, new `AgentRunAttempt` creation, checkpoint validation, LangGraph resume, execution grant consumption, `TicketEscalation` creation, and recommendation completion. The API never invokes LangGraph or sensitive execution.

## Ticket Escalation Inspection

### List Ticket Escalations

```http
GET /api/v1/workspaces/{workspace_id}/ticket-escalations
```

Supported query parameters:

| Parameter | Type | Required | Description |
|---|---|---:|---|
| `ticket_id` | UUID | No | Limits results to one ticket in the workspace. |
| `cursor` | opaque string | No | Continues stable keyset pagination. |
| `page_size` | integer | No | Number of records, from 1 through 100. Defaults to 20. |

Ordering:

```text
created_at DESC
id DESC
```

Example:

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://localhost:8000/api/v1/workspaces/$WorkspaceId/ticket-escalations?ticket_id=$TicketId&page_size=20" `
  -Headers @{
    "X-Correlation-ID" = $CorrelationId
  }
```

### Get Ticket Escalation

```http
GET /api/v1/workspaces/{workspace_id}/ticket-escalations/{ticket_escalation_id}
```

Missing records and records owned by another workspace return the same nondisclosing response:

```text
HTTP 404
ticket_escalation_not_found
```

Response:

```json
{
  "id": "e2c14bf4-c352-40f6-bd29-bae5912192cb",
  "workspace_id": "52b088ac-e57c-4d0c-89ad-8c86949b90ea",
  "ticket_id": "8dfa091a-e5e0-46a2-b746-6b4c2d82a9d4",
  "agent_run_id": "2c4a94ba-88fa-4b0d-957f-e945d3599ac6",
  "executed_by_agent_run_attempt_id": "b1e67193-2913-41e5-b21b-78b849f77e65",
  "approval_request_id": "5f424ef6-e34c-489e-918b-5795238c9d51",
  "agent_tool_call_id": "b65f5521-bd83-43a0-b5a1-0db1dfdf70f1",
  "target_queue": "support_operations",
  "reason": "Operational review required.",
  "created_at": "2026-08-03T22:07:00Z"
}
```

`TicketEscalation` is immutable. Inspection endpoints do not mutate the escalation, ticket, approval, AgentRun, or tool call.

## Sensitive Data Boundary

Approval responses expose bounded proposed input because it is the content an operator must inspect before deciding.

Ticket escalation responses expose operational linkage identifiers.

The API intentionally does not expose:

- SensitiveExecutionGrant records;
- granted input;
- execution output;
- LangGraph checkpoint state;
- worker lease state;
- raw prompts;
- raw model responses;
- provider state;
- ORM internals.

Execution grants remain internal authorization records.

## Workspace Isolation

Every route is workspace-scoped.

The workspace identifier from the route is applied to all repository queries.

The API does not distinguish:

- a missing resource;
- a resource belonging to another workspace.

Both return the same `404` code and message.

List endpoints never return records from another workspace, including when a filter references a foreign ticket or resource identifier.

## Pagination

Approval and escalation list endpoints use opaque keyset cursors.

Cursor payloads are versioned and contain only:

```text
created_at
approval_request_id   # approval list cursors
ticket_escalation_id  # escalation list cursors
```

The cursor is URL-safe base64 over strict JSON.

Invalid cursors return:

```text
HTTP 400
invalid_pagination_cursor
```

Cursors reject:

- malformed base64;
- malformed JSON;
- unsupported versions;
- extra fields;
- invalid UUIDs;
- timestamps without timezone information.

The API does not use offset pagination or total-count queries.

## Error Semantics

| Condition | HTTP | Error code |
|---|---:|---|
| Approval missing or foreign | 404 | `approval_request_not_found` |
| Escalation missing or foreign | 404 | `ticket_escalation_not_found` |
| Conflicting terminal decision | 409 | `approval_decision_conflict` |
| Expired approval conflict | 409 | `approval_request_expired` |
| AgentRun not waiting for approval | 409 | `approval_run_state_conflict` |
| Pending tool-call state mismatch | 409 | `approval_tool_call_state_conflict` |
| Invalid pagination cursor | 400 | `invalid_pagination_cursor` |
| Invalid UUID or body | 422 | Existing FastAPI validation response |
| Unexpected consistency failure | 500 | Existing API error envelope |

All application error responses use the project error envelope and include the request ID generated by request context middleware.

## Security Considerations

Current protections include:

- mandatory workspace scoping;
- nondisclosing 404 behavior for missing and foreign resources;
- list queries that apply `workspace_id` before other filters;
- bounded pagination;
- strict request schemas that reject extra fields;
- asserted `actor_reference` values that are not verified identities;
- client-inaccessible, server-generated decision timestamps;
- serialized terminal decisions;
- immutable approval and escalation history;
- internal-only execution grants;
- bounded safe `proposed_input` on approval responses;
- escalation responses that omit approval actor, decision comment, and execution internals;
- no execution from HTTP handlers;
- no external side-effect endpoints.

Verified identities and RBAC remain a separate policy-enforcement concern. The current contract preserves room for replacing asserted actor references with verified principals without changing approval domain transitions.

## Intentional Scope Boundaries

The API intentionally does not include:

- authentication or RBAC;
- grant inspection endpoints;
- escalation mutation endpoints;
- manual LangGraph resume endpoints;
- worker execution endpoints;
- approval expiration endpoints;
- external side-effect tools;
- frontend operator interfaces;
- notification delivery.

Grant inspection intentionally remains internal. Manual resume endpoints remain unavailable to preserve worker ownership. Frontend operator workflows can be introduced without changing domain transitions. Keeping these concerns separate prevents inspection and decision endpoints from becoming alternate execution paths.