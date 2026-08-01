# SupportOps API Examples

## Purpose

This document provides reproducible examples for the implemented workspace,
support ticket, AgentRun, classification, and logical invocation inspection
HTTP API.

The examples use placeholder values suitable for local development and public
documentation.

The API currently supports:

- workspace creation and retrieval;
- workspace-scoped ticket creation with atomic initial AgentRun scheduling;
- a minimal processing-run reference on ticket creation;
- workspace-scoped ticket retrieval;
- workspace-scoped ticket listing;
- workspace-scoped AgentRun inspection;
- workspace-scoped AgentRunAttempt history inspection;
- AgentRun classification reference;
- classification detail;
- ticket classification history;
- AgentRun logical invocation history;
- opaque cursor pagination;
- stable expected-error responses;
- request and correlation trace identifiers.

The current API does not include authentication or authorization. Workspace
scoping establishes an explicit data ownership boundary but does not establish
trusted tenant isolation.

A queued `ticket-classification-v1` processing run records that durable
classification work has been scheduled. Classification completion remains an
asynchronous worker outcome.

AgentRun, classification, and logical invocation inspection endpoints are
strictly read-only. They report current persisted state and do not guarantee
future completion. They do not perform mutation, retry, cancellation, or lease
revocation.

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

$agentRunId = $processingRun.id
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
    "workflow_version": "ticket-classification-v1"
  }
}
```

Expected behavior:

- nested `ticket` object;
- nested `processing_run` object;
- initial ticket status `open`;
- processing-run status `queued`;
- workflow name `ticket-processing`;
- workflow version `ticket-classification-v1`;
- `workspace_id` equals workspace A;
- `ingestion_request_id` equals `X-Request-ID`;
- `correlation_id` equals `X-Correlation-ID`.

A queued `ticket-classification-v1` run does not indicate that AI classification
has completed.

Retain `$agentRunId` for the inspection examples below.

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

## Inspect a queued AgentRun

Immediately after ticket creation, before the worker claims the run:

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/$workspaceAId/agent-runs/$agentRunId"
```

Expected status:

```text
200 OK
```

Expected queued response:

```json
{
  "id": "24f24172-f39c-4dcf-9722-b073e22944d0",
  "workspace_id": "59ecc675-bf00-4f3b-8284-876f226539d6",
  "ticket_id": "6e688ded-cf71-4c01-b87f-591cc014af03",
  "status": "queued",
  "workflow": {
    "name": "ticket-processing",
    "version": "ticket-classification-v1",
    "trigger_key": "initial-ticket-processing"
  },
  "classification": null,
  "attempt_count": 0,
  "max_attempts": 3,
  "available_at": "2026-07-31T12:00:00Z",
  "first_started_at": null,
  "completed_at": null,
  "last_error": null,
  "correlation_id": "db320c15-e7de-4b36-8b22-11b96b3c68de",
  "created_at": "2026-07-31T12:00:00Z",
  "updated_at": "2026-07-31T12:00:00Z"
}
```

The response does not include lease ownership, lease tokens, lease expiry, or
ingestion request identifiers. Queued runs return `classification: null`.

## Inspect a succeeded AgentRun

After the worker completes classification:

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/$workspaceAId/agent-runs/$agentRunId"
```

Expected succeeded response:

```json
{
  "id": "24f24172-f39c-4dcf-9722-b073e22944d0",
  "workspace_id": "59ecc675-bf00-4f3b-8284-876f226539d6",
  "ticket_id": "6e688ded-cf71-4c01-b87f-591cc014af03",
  "status": "succeeded",
  "workflow": {
    "name": "ticket-processing",
    "version": "ticket-classification-v1",
    "trigger_key": "initial-ticket-processing"
  },
  "classification": {
    "id": "8f3c1b2a-4d5e-6f70-8192-a3b4c5d6e7f8",
    "schema_version": "ticket-classification-v1",
    "created_at": "2026-07-31T12:00:01Z"
  },
  "attempt_count": 1,
  "max_attempts": 3,
  "available_at": "2026-07-31T12:00:00Z",
  "first_started_at": "2026-07-31T12:00:01Z",
  "completed_at": "2026-07-31T12:00:01Z",
  "last_error": null,
  "correlation_id": "db320c15-e7de-4b36-8b22-11b96b3c68de",
  "created_at": "2026-07-31T12:00:00Z",
  "updated_at": "2026-07-31T12:00:01Z"
}
```

The `classification` object is a minimal accepted-classification reference.
Full labels and prompt provenance are available through the classification
detail route. Retain `$classificationId` from `classification.id` for the
examples below.

Inspection reports persisted state at the time of the request. It does not
guarantee future completion for runs that are still in progress.

## Inspect retry_scheduled or failed AgentRun state

When a run has safe error metadata after a retryable or terminal failure, the
`last_error` object contains only a stable code and summary:

```json
{
  "id": "24f24172-f39c-4dcf-9722-b073e22944d0",
  "workspace_id": "59ecc675-bf00-4f3b-8284-876f226539d6",
  "ticket_id": "6e688ded-cf71-4c01-b87f-591cc014af03",
  "status": "retry_scheduled",
  "workflow": {
    "name": "ticket-processing",
    "version": "ticket-classification-v1",
    "trigger_key": "initial-ticket-processing"
  },
  "classification": null,
  "attempt_count": 1,
  "max_attempts": 3,
  "available_at": "2026-07-31T12:00:05Z",
  "first_started_at": "2026-07-31T12:00:01Z",
  "completed_at": null,
  "last_error": {
    "code": "executor_timeout",
    "summary": "The configured executor exceeded its execution timeout."
  },
  "correlation_id": "db320c15-e7de-4b36-8b22-11b96b3c68de",
  "created_at": "2026-07-31T12:00:00Z",
  "updated_at": "2026-07-31T12:00:01Z"
}
```

A terminal exhaustion path uses status `failed` with the same safe
`last_error` shape. Raw exception text is not returned. Runs without an
accepted classification continue to return `classification: null`.

## Inspect empty attempt history

Queued runs have no attempts yet:

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/$workspaceAId/agent-runs/$agentRunId/attempts"
```

Expected empty history response:

```json
{
  "items": []
}
```

## Inspect ordered attempt history

After one or more claims:

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/api/v1/workspaces/$workspaceAId/agent-runs/$agentRunId/attempts"
```

Expected ordered history response:

```json
{
  "items": [
    {
      "id": "2b39f5b7-b2a4-48d0-b079-fdad286d5315",
      "attempt_number": 1,
      "worker_id": "worker-local-1",
      "started_at": "2026-07-31T12:00:01Z",
      "finished_at": "2026-07-31T12:00:01Z",
      "outcome": "succeeded",
      "error": null
    }
  ]
}
```

Attempts are ordered by `attempt_number` ascending. Possible outcomes are:

```text
succeeded
retryable_failure
terminal_failure
timed_out
lease_expired
```

Active attempts may return null `finished_at`, `outcome`, and `error`.

Attempt responses do not include `agent_run_id`, lease tokens, or execution
request identifiers. Attempt pagination is intentionally omitted because
`max_attempts` is bounded.

## Attempt cross-workspace AgentRun inspection

```powershell
try {
  Invoke-RestMethod `
    -Method Get `
    -Uri "http://127.0.0.1:8000/api/v1/workspaces/$workspaceBId/agent-runs/$agentRunId"
} catch {
  $_.Exception.Response.StatusCode.value__
  $_.ErrorDetails.Message
}
```

Expected status:

```text
404 Not Found
```

Expected error response:

```json
{
  "error": {
    "code": "agent_run_not_found",
    "message": "AgentRun was not found.",
    "request_id": "dfe63a63-031c-4ea9-89dd-d556bd51766a"
  }
}
```

Missing and cross-workspace AgentRuns both return this contract. The standard
error envelope includes the request ID. The response does not reveal that the
AgentRun exists in another workspace.

The same `agent_run_not_found` contract applies to attempt-history and
logical-invocation-history requests for missing or cross-workspace runs.

## Inspect classification detail

After a succeeded classification run:

```powershell
$classificationId = (
  Invoke-RestMethod `
    -Method Get `
    -Uri (
      "http://127.0.0.1:8000/api/v1/workspaces/" +
      "$workspaceAId/agent-runs/$agentRunId"
    )
).classification.id

Invoke-RestMethod `
  -Method Get `
  -Uri (
    "http://127.0.0.1:8000/api/v1/workspaces/" +
    "$workspaceAId/ticket-classifications/$classificationId"
  )
```

Expected status:

```text
200 OK
```

Expected response shape:

```json
{
  "id": "8f3c1b2a-4d5e-6f70-8192-a3b4c5d6e7f8",
  "workspace_id": "59ecc675-bf00-4f3b-8284-876f226539d6",
  "ticket_id": "6e688ded-cf71-4c01-b87f-591cc014af03",
  "agent_run_id": "24f24172-f39c-4dcf-9722-b073e22944d0",
  "accepted_invocation_id": "c1d2e3f4-5678-90ab-cdef-1234567890ab",
  "category": "billing",
  "intent": "ask_question",
  "urgency": "normal",
  "sentiment": "neutral",
  "requires_human_review": false,
  "summary": "The requester cannot access the billing dashboard after sign-in.",
  "schema_version": "ticket-classification-v1",
  "prompt": {
    "id": "ticket-classification",
    "version": 1,
    "content_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "provider": "mock",
  "model": "mock-ticket-classifier-v1",
  "created_at": "2026-07-31T12:00:01Z"
}
```

The response includes accepted invocation identity, bounded labels, nested
prompt provenance, provider, model, and creation timestamp. Provider request
IDs, raw prompts, raw responses, lease data, and execution request IDs are not
exposed.

## Inspect ticket classification history

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri (
    "http://127.0.0.1:8000/api/v1/workspaces/" +
    "$workspaceAId/tickets/$ticketId/classifications?page_size=20"
  )
```

Expected response shape:

```json
{
  "items": [
    {
      "id": "8f3c1b2a-4d5e-6f70-8192-a3b4c5d6e7f8",
      "workspace_id": "59ecc675-bf00-4f3b-8284-876f226539d6",
      "ticket_id": "6e688ded-cf71-4c01-b87f-591cc014af03",
      "agent_run_id": "24f24172-f39c-4dcf-9722-b073e22944d0",
      "accepted_invocation_id": "c1d2e3f4-5678-90ab-cdef-1234567890ab",
      "category": "billing",
      "intent": "ask_question",
      "urgency": "normal",
      "sentiment": "neutral",
      "requires_human_review": false,
      "summary": "The requester cannot access the billing dashboard after sign-in.",
      "schema_version": "ticket-classification-v1",
      "prompt": {
        "id": "ticket-classification",
        "version": 1,
        "content_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      },
      "provider": "mock",
      "model": "mock-ticket-classifier-v1",
      "created_at": "2026-07-31T12:00:01Z"
    }
  ],
  "next_cursor": null
}
```

Classification history is ordered newest first and uses opaque keyset
pagination through `page_size` and `cursor`. The route validates ticket
ownership before returning history. Missing and cross-workspace tickets return
`404` with `ticket_not_found`.

## Inspect AgentRun logical invocation history

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri (
    "http://127.0.0.1:8000/api/v1/workspaces/" +
    "$workspaceAId/agent-runs/$agentRunId/llm-invocations"
  )
```

Expected response shape:

```json
{
  "items": [
    {
      "id": "c1d2e3f4-5678-90ab-cdef-1234567890ab",
      "agent_run_attempt_id": "2b39f5b7-b2a4-48d0-b079-fdad286d5315",
      "attempt_number": 1,
      "invocation_sequence": 1,
      "status": "succeeded",
      "provider": "mock",
      "model": "mock-ticket-classifier-v1",
      "prompt": {
        "id": "ticket-classification",
        "version": 1,
        "content_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      },
      "schema_version": "ticket-classification-v1",
      "usage": {
        "input_tokens": 120,
        "cached_input_tokens": 0,
        "output_tokens": 24,
        "reasoning_tokens": null,
        "total_tokens": 144
      },
      "estimated_cost": {
        "pricing_catalog_version": "supportops-pricing-2026-08-01",
        "pricing_found": true,
        "input_cost_usd": "0",
        "cached_input_cost_usd": "0",
        "output_cost_usd": "0",
        "total_cost_usd": "0"
      },
      "latency_ms": 12,
      "error_code": null,
      "created_at": "2026-07-31T12:00:01Z"
    }
  ]
}
```

Invocation history is ordered by attempt number ascending, then sequence
ascending. The history is naturally bounded by retry and repair budgets and is
not paginated. Each item exposes attempt identity and number, sequence, status,
prompt provenance, usage when known, estimated cost, latency, and a safe
`error_code`. Provider request IDs and raw provider content are not exposed.

Queued runs may return:

```json
{
  "items": []
}
```

## Attempt cross-workspace classification inspection

```powershell
try {
  Invoke-RestMethod `
    -Method Get `
    -Uri (
      "http://127.0.0.1:8000/api/v1/workspaces/" +
      "$workspaceBId/ticket-classifications/$classificationId"
    )
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
ticket_classification_not_found
```

Missing and cross-workspace classifications use the same contract. The response
does not reveal that the classification exists in another workspace.

Cross-workspace AgentRun invocation history uses the same
`agent_run_not_found` contract as AgentRun detail and attempt history.

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
agent_run_not_found
ticket_classification_not_found
workspace_slug_conflict
ticket_external_reference_conflict
invalid_pagination_cursor
```

Malformed UUIDs, invalid page sizes, and invalid request schemas use FastAPI
validation responses with status `422 Unprocessable Entity`.

The API does not expose an endpoint that discovers AgentRuns by ticket. Clients
retain the AgentRun identifier from ticket creation or another known source.