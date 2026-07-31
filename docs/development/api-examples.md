# SupportOps API Examples

## Purpose

This document provides reproducible examples for the implemented workspace and
support ticket HTTP API.

The examples use placeholder values suitable for local development and public
documentation.

The API currently supports:

- workspace creation and retrieval;
- workspace-scoped ticket creation with atomic initial AgentRun scheduling;
- a minimal processing-run reference on ticket creation;
- workspace-scoped ticket retrieval;
- workspace-scoped ticket listing;
- opaque cursor pagination;
- stable expected-error responses;
- request and correlation trace identifiers.

The current API does not include authentication or authorization. Workspace
scoping establishes an explicit data ownership boundary but does not establish
trusted tenant isolation.

A queued deterministic-baseline processing run records that durable work has
been scheduled. It does not represent AI classification. AgentRun inspection
endpoints are not implemented yet.

## Prerequisites

Start the local infrastructure and apply the current database migration:

```powershell
docker compose up -d
uv run alembic upgrade head
```

Start the API:

```powershell
uv run uvicorn supportops.api.main:app --host 127.0.0.1 --port 8000
```

The examples below use:

```text
http://127.0.0.1:8000
```

## Create workspace A

```powershell
$workspaceAResponse = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces" `
  -ContentType "application/json" `
  -Body (@{
    name = "Platform Support"
    slug = "platform-support"
  } | ConvertTo-Json)

$workspaceAResponse
$workspaceAId = $workspaceAResponse.id
```

Expected status:

```text
201 Created
```

## Create workspace B

```powershell
$workspaceBResponse = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces" `
  -ContentType "application/json" `
  -Body (@{
    name = "Customer Success"
    slug = "customer-success"
  } | ConvertTo-Json)

$workspaceBResponse
$workspaceBId = $workspaceBResponse.id
```

## Create a ticket in workspace A

```powershell
$ticketResponse = Invoke-WebRequest `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/$workspaceAId/tickets" `
  -ContentType "application/json" `
  -Body (@{
    subject = "Unable to access the billing dashboard"
    description = "The dashboard returns an access error after sign-in."
  } | ConvertTo-Json)

$payload = $ticketResponse.Content | ConvertFrom-Json
$ticket = $payload.ticket
$processingRun = $payload.processing_run
$ticketId = $ticket.id

$payload
$ticketResponse.Headers["X-Request-ID"]
$ticketResponse.Headers["X-Correlation-ID"]
```

Expected status:

```text
201 Created
```

`201 Created` means the ticket was created successfully. Processing is
scheduled asynchronously and is not executed by the API response path.

Expected response shape:

```json
{
  "ticket": {
    "id": "6e688ded-cf71-4c01-b87f-591cc014af03",
    "workspace_id": "59ecc675-bf00-4f3b-8284-876f226539d6",
    "subject": "Unable to access the billing dashboard",
    "description": "The dashboard returns an access error after sign-in.",
    "status": "open",
    "external_reference": null,
    "ingestion_request_id": "dfe63a63-031c-4ea9-89dd-d556bd51766a",
    "correlation_id": "db320c15-e7de-4b36-8b22-11b96b3c68de",
    "created_at": "2026-07-31T12:00:00Z",
    "updated_at": "2026-07-31T12:00:00Z"
  },
  "processing_run": {
    "id": "24f24172-f39c-4dcf-9722-b073e22944d0",
    "status": "queued",
    "workflow_name": "ticket-processing",
    "workflow_version": "deterministic-baseline-v1"
  }
}
```

Expected behavior:

- nested `ticket` object;
- nested `processing_run` object;
- initial ticket status `open`;
- processing-run status `queued`;
- workflow name `ticket-processing`;
- workflow version `deterministic-baseline-v1`;
- `workspace_id` equals workspace A;
- `ingestion_request_id` equals `X-Request-ID`;
- `correlation_id` equals `X-Correlation-ID`.

A queued deterministic-baseline run does not indicate that AI classification
has occurred.

## Retrieve the ticket through workspace A

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/$workspaceAId/tickets/$ticketId"
```

Expected status:

```text
200 OK
```

## Attempt cross-workspace retrieval

```powershell
try {
  Invoke-RestMethod `
    -Method Get `
    -Uri "http://127.0.0.1:8000/api/v1/workspaces/$workspaceBId/tickets/$ticketId"
} catch {
  $_.Exception.Response.StatusCode.value__
  $_.ErrorDetails.Message
}
```

Expected status:

```text
404 Not Found
```

Expected error code:

```text
ticket_not_found
```

The response does not reveal that the ticket exists in another workspace.

## List tickets in workspace A

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/$workspaceAId/tickets"
```

Expected response shape:

```json
{
  "items": [],
  "next_cursor": null
}
```

The actual `items` collection contains only tickets owned by workspace A.

## List tickets in workspace B

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/$workspaceBId/tickets"
```

Expected result:

```json
{
  "items": [],
  "next_cursor": null
}
```

Workspace B does not receive workspace A tickets.

## Create a ticket with an external reference

```powershell
$referencedTicketResponse = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/$workspaceAId/tickets" `
  -ContentType "application/json" `
  -Body (@{
    subject = "Billing dashboard access"
    description = "A support source supplied an upstream identifier."
    external_reference = "SUP-1042"
  } | ConvertTo-Json)

$referencedTicketResponse
$referencedTicketResponse.ticket
$referencedTicketResponse.processing_run
```

Expected status:

```text
201 Created
```

The response includes the nested ticket and a queued processing-run reference.
## Repeat the external reference in workspace A

```powershell
try {
  Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/api/v1/workspaces/$workspaceAId/tickets" `
    -ContentType "application/json" `
    -Body (@{
      subject = "Duplicate upstream identifier"
      description = "This request repeats the same workspace-scoped reference."
      external_reference = "SUP-1042"
    } | ConvertTo-Json)
} catch {
  $_.Exception.Response.StatusCode.value__
  $_.ErrorDetails.Message
}
```

Expected status:

```text
409 Conflict
```

Expected error code:

```text
ticket_external_reference_conflict
```

## Reuse the external reference in workspace B

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/$workspaceBId/tickets" `
  -ContentType "application/json" `
  -Body (@{
    subject = "Same reference in another workspace"
    description = "External references are scoped by workspace."
    external_reference = "SUP-1042"
  } | ConvertTo-Json)
```

Expected status:

```text
201 Created
```

## Propagate a correlation identifier

```powershell
$correlationId = [guid]::NewGuid().ToString()

$correlatedResponse = Invoke-WebRequest `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/$workspaceAId/tickets" `
  -Headers @{
    "X-Correlation-ID" = $correlationId
  } `
  -ContentType "application/json" `
  -Body (@{
    subject = "Correlated support request"
    description = "This request propagates a valid correlation identifier."
  } | ConvertTo-Json)

$correlatedTicket = $correlatedResponse.Content | ConvertFrom-Json

$correlationId
$correlatedResponse.Headers["X-Request-ID"]
$correlatedResponse.Headers["X-Correlation-ID"]
$correlatedTicket.ticket.ingestion_request_id
$correlatedTicket.ticket.correlation_id
$correlatedTicket.processing_run.status
```

Expected behavior:

- the response correlation ID equals `$correlationId`;
- the request ID is independently generated;
- the ticket persists both identifiers;
- the response includes a queued processing-run reference.

## Validate cursor pagination

Create additional workspace A tickets, then request a bounded page:

```powershell
$firstPage = Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/$workspaceAId/tickets?page_size=2"

$firstPage
```

When more records exist, `next_cursor` is non-null.

Request the next page:

```powershell
$encodedCursor = [System.Uri]::EscapeDataString(
  $firstPage.next_cursor
)

$secondPage = Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/$workspaceAId/tickets?page_size=2&cursor=$encodedCursor"

$secondPage
```

The cursor is opaque to API consumers and must not be decoded or modified by
clients.

## Invalid cursor response

```powershell
try {
  Invoke-RestMethod `
    -Method Get `
    -Uri "http://127.0.0.1:8000/api/v1/workspaces/$workspaceAId/tickets?cursor=invalid"
} catch {
  $_.Exception.Response.StatusCode.value__
  $_.ErrorDetails.Message
}
```

Expected status:

```text
400 Bad Request
```

Expected error code:

```text
invalid_pagination_cursor
```

## Page-size validation

The default page size is:

```text
20
```

The maximum page size is:

```text
100
```

Values below `1` or above `100` return:

```text
422 Unprocessable Entity
```

## Inspect structured logs

The API emits request completion events containing safe operational metadata:

```text
event
request_id
correlation_id
http_method
route_or_path
status_code
duration_ms
```

Completion logs do not include:

- ticket subject;
- ticket description;
- request bodies;
- authorization headers;
- raw invalid external identifiers;
- database URLs;
- raw infrastructure exception messages.

## Error response contract

Expected application errors use this envelope:

```json
{
  "error": {
    "code": "ticket_not_found",
    "message": "Ticket was not found.",
    "request_id": "00000000-0000-0000-0000-000000000000"
  }
}
```

Implemented expected error codes:

```text
workspace_not_found
ticket_not_found
workspace_slug_conflict
ticket_external_reference_conflict
invalid_pagination_cursor
```

Malformed UUIDs, invalid page sizes, and invalid request schemas use FastAPI
validation responses with status `422 Unprocessable Entity`.