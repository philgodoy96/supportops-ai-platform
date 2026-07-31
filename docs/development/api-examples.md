# SupportOps API Examples

## Purpose

This document provides reproducible examples for the implemented workspace and
support ticket HTTP API.

The examples use placeholder values suitable for local development and public
documentation.

The API currently supports:

- workspace creation and retrieval;
- workspace-scoped ticket creation;
- workspace-scoped ticket retrieval;
- workspace-scoped ticket listing;
- opaque cursor pagination;
- stable expected-error responses;
- request and correlation trace identifiers.

The current API does not include authentication or authorization. Workspace
scoping establishes an explicit data ownership boundary but does not establish
trusted tenant isolation.

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

$ticket = $ticketResponse.Content | ConvertFrom-Json
$ticketId = $ticket.id

$ticket
$ticketResponse.Headers["X-Request-ID"]
$ticketResponse.Headers["X-Correlation-ID"]
```

Expected behavior:

- status `201 Created`;
- initial ticket status `open`;
- `workspace_id` equals workspace A;
- `ingestion_request_id` equals `X-Request-ID`;
- `correlation_id` equals `X-Correlation-ID`.

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
$referencedTicket = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/$workspaceAId/tickets" `
  -ContentType "application/json" `
  -Body (@{
    subject = "Billing dashboard access"
    description = "A support source supplied an upstream identifier."
    external_reference = "SUP-1042"
  } | ConvertTo-Json)

$referencedTicket
```

Expected status:

```text
201 Created
```

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
  } | ConvertTo-Json
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
$correlatedTicket.ingestion_request_id
$correlatedTicket.correlation_id
```

Expected behavior:

- the response correlation ID equals `$correlationId`;
- the request ID is independently generated;
- the ticket persists both identifiers.

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