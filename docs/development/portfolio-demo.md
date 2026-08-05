# End-to-End Demonstration

## Purpose

This walkthrough demonstrates the SupportOps AI Platform through a complete synthetic support scenario using the repository's local infrastructure and default mock providers.

The demonstration covers:

- workspace-scoped ticket intake;
- atomic Ticket and AgentRun creation;
- durable worker execution;
- controlled LangGraph orchestration;
- ticket classification through the application-owned LLM Gateway;
- knowledge ingestion and semantic retrieval;
- registered read-only tool execution;
- persisted recommendations and citations;
- AgentRun inspection;
- deterministic evaluation and prompt-decision governance.

The walkthrough does not require OpenAI credentials, Langfuse credentials, paid evaluation, external integrations, or production data.

## Demonstration boundaries

This scenario is intentionally local and synthetic.

It demonstrates application boundaries, durable execution, reproducibility, and AI release governance. It does not claim:

- secure authenticated public multi-tenancy;
- exactly-once execution;
- autonomous ticket resolution;
- real Jira, ServiceNow, Slack, or email writes;
- provider-backed prompt superiority;
- runtime adoption of `ticket-classification` prompt version 2.

PostgreSQL remains authoritative for business state. Qdrant remains a rebuildable retrieval projection. LangGraph runs inside the durable AgentRun boundary. Langfuse remains optional derived telemetry.

## Prerequisites

Run all commands from the repository root.

Required local tools:

- Python and `uv`;
- Docker with Docker Compose;
- PowerShell.

The default local configuration uses mock LLM and embedding providers.

## 1. Install dependencies and create local configuration

```powershell
uv sync --frozen --all-groups
Copy-Item .env.example .env
```

Confirm the local defaults before continuing:

```powershell
Select-String `
  -Path .env `
  -Pattern 'SUPPORTOPS_LLM_PROVIDER|SUPPORTOPS_EMBEDDING_PROVIDER|SUPPORTOPS_TICKET_PROCESSING_WORKFLOW_VERSION'
```

The default walkthrough should use:

```text
SUPPORTOPS_LLM_PROVIDER=mock
SUPPORTOPS_EMBEDDING_PROVIDER=mock
SUPPORTOPS_TICKET_PROCESSING_WORKFLOW_VERSION=controlled-support-v1
```

## 2. Start PostgreSQL and Qdrant

```powershell
docker compose up -d
docker compose ps
```

Validate the Compose configuration when needed:

```powershell
docker compose config --quiet
```

## 3. Apply application migrations

```powershell
uv run alembic upgrade head
uv run alembic current
uv run alembic heads
```

The worker initializes the PostgreSQL-backed LangGraph checkpoint structures during startup. No separate checkpoint initialization command is required for the supported local path.

## 4. Ensure the Qdrant knowledge collection

```powershell
uv run supportops-index-knowledge ensure-collection
```

This operation prepares the derived retrieval projection. PostgreSQL remains authoritative for knowledge documents, versions, chunks, and activation state.

## 5. Start the API

Open a dedicated PowerShell terminal from the repository root:

```powershell
uv run uvicorn supportops.api.main:app `
  --host 127.0.0.1 `
  --port 8000
```

In another terminal, verify health:

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/health/live"

Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:8000/health/ready"
```

## 6. Start the worker

Open another PowerShell terminal from the repository root:

```powershell
$env:SUPPORTOPS_WORKER_ID = "worker-demo-1"
uv run supportops-worker
```

Keep the API and worker processes running for the remaining runtime steps.

## 7. Create a synthetic workspace

In a separate PowerShell terminal:

```powershell
$baseUrl = "http://127.0.0.1:8000"

$workspaceBody = @{
  name = "Northstar Support Lab"
  slug = "northstar-support-lab"
} | ConvertTo-Json

$workspace = Invoke-RestMethod `
  -Method Post `
  -Uri "$baseUrl/api/v1/workspaces" `
  -ContentType "application/json" `
  -Body $workspaceBody

$workspace | ConvertTo-Json -Depth 10
$workspaceId = $workspace.id
```

Store the returned workspace identifier:

```powershell
$workspaceId
```

Use a unique slug when repeating the demonstration against the same database.

## 8. Create a synthetic runbook

```powershell
$runbookContent = @"
# Checkout API latency runbook

## Symptoms

- Checkout API p95 latency exceeds 1.5 seconds.
- Customers may report timeouts while submitting orders.
- Error rates may remain low while database saturation increases.

## Investigation

1. Inspect checkout API latency and error-rate telemetry.
2. Inspect database connection-pool utilization.
3. Check for a recent deployment affecting checkout requests.
4. Compare the incident window with dependency latency.
5. Do not perform external write actions automatically.

## Recommendation policy

- Prefer evidence-backed mitigation steps.
- Cite the active runbook version.
- Escalate sensitive operational actions for human approval.
- Do not claim that an external escalation has been delivered.
"@

$documentBody = @{
  title = "Checkout API latency runbook"
  external_reference = "synthetic://runbooks/checkout-api-latency"
  media_type = "text/markdown"
  content = $runbookContent
} | ConvertTo-Json

$documentResponse = Invoke-RestMethod `
  -Method Post `
  -Uri "$baseUrl/api/v1/workspaces/$workspaceId/documents" `
  -ContentType "application/json" `
  -Body $documentBody

$documentResponse | ConvertTo-Json -Depth 20

$documentId = $documentResponse.document.id
$documentVersionId = $documentResponse.version.id
```

## 9. Index the document version

```powershell
uv run supportops-index-knowledge index-version `
  --workspace-id "$workspaceId" `
  --document-id "$documentId" `
  --document-version-id "$documentVersionId"
```

The indexing process:

1. reads authoritative document content from PostgreSQL;
2. produces deterministic chunks;
3. generates embeddings through the configured provider;
4. writes the derived vector projection to Qdrant;
5. persists indexing state in PostgreSQL.

## 10. Activate the indexed version

```powershell
$activation = Invoke-RestMethod `
  -Method Post `
  -Uri "$baseUrl/api/v1/workspaces/$workspaceId/documents/$documentId/versions/$documentVersionId/activate"

$activation | ConvertTo-Json -Depth 20
```

Only active, ready document versions are eligible for runtime retrieval.

## 11. Verify semantic retrieval

```powershell
$searchBody = @{
  query = "What should support investigate when checkout latency increases?"
  limit = 5
} | ConvertTo-Json

$searchResults = Invoke-RestMethod `
  -Method Post `
  -Uri "$baseUrl/api/v1/workspaces/$workspaceId/knowledge/search" `
  -ContentType "application/json" `
  -Body $searchBody

$searchResults | ConvertTo-Json -Depth 20
```

Verify that the results reference the synthetic checkout runbook and contain stable document, version, and chunk identifiers.

## 12. Create a synthetic support ticket

```powershell
$ticketBody = @{
  subject = "Checkout requests are timing out"
  description = "Several customers report checkout timeouts. API latency increased after a deployment, and the database connection pool may be saturated."
} | ConvertTo-Json

$ticketResponse = Invoke-RestMethod `
  -Method Post `
  -Uri "$baseUrl/api/v1/workspaces/$workspaceId/tickets" `
  -ContentType "application/json" `
  -Body $ticketBody

$ticketResponse | ConvertTo-Json -Depth 20

$ticketId = $ticketResponse.ticket.id
$agentRunId = $ticketResponse.processing_run.id
```

The intake transaction should return both the persisted Ticket and its initial AgentRun.

## 13. Inspect durable processing

Allow the worker to process the run:

```powershell
Start-Sleep -Seconds 3
```

Inspect the AgentRun:

```powershell
$agentRun = Invoke-RestMethod `
  -Method Get `
  -Uri "$baseUrl/api/v1/workspaces/$workspaceId/agent-runs/$agentRunId"

$agentRun | ConvertTo-Json -Depth 20
```

Inspect execution attempts:

```powershell
$attempts = Invoke-RestMethod `
  -Method Get `
  -Uri "$baseUrl/api/v1/workspaces/$workspaceId/agent-runs/$agentRunId/attempts"

$attempts | ConvertTo-Json -Depth 20
```

When the run is still active, repeat the AgentRun request after a short delay.

## 14. Inspect the complete processing aggregate

```powershell
$inspection = Invoke-RestMethod `
  -Method Get `
  -Uri "$baseUrl/api/v1/workspaces/$workspaceId/tickets/$ticketId/agent-runs/$agentRunId/inspection"

$inspection | ConvertTo-Json -Depth 30
```

Review the available evidence for:

- AgentRun lifecycle and attempts;
- classification output;
- durable LLM invocation provenance;
- retrieved knowledge evidence;
- tool decisions and tool-call records;
- grounded recommendation;
- ordered citations;
- token and estimated-cost accounting.

The exact contents depend on the configured deterministic provider behavior, but all durable records exposed by the inspection endpoint remain application-owned.

## 15. Inspect the human-approval boundary

The default walkthrough uses `controlled-support-v1`. It does not fabricate a sensitive escalation or guarantee that every synthetic ticket produces an approval request.

The human-approved path is decision-driven and separately versioned as `human-approved-support-v1`.

To inspect approval APIs and lifecycle contracts, use:

- [`approval-workflow-api.md`](approval-workflow-api.md)
- [`human-approved-workflow.md`](../architecture/human-approved-workflow.md)

Pending approvals for a workspace can be inspected with:

```powershell
$pendingApprovals = Invoke-RestMethod `
  -Method Get `
  -Uri "$baseUrl/api/v1/workspaces/$workspaceId/approvals?status=pending&page_size=20"

$pendingApprovals | ConvertTo-Json -Depth 20
```

Ticket escalation records can be inspected with:

```powershell
$escalations = Invoke-RestMethod `
  -Method Get `
  -Uri "$baseUrl/api/v1/workspaces/$workspaceId/ticket-escalations?ticket_id=$ticketId&page_size=20"

$escalations | ConvertTo-Json -Depth 20
```

An empty result is valid for the default walkthrough. It means the controlled workflow did not create a sensitive action requiring approval.

For deterministic verification of approval, rejection, expiration, resume, execution-grant, and idempotent escalation semantics, run the repository-supported test suites:

```powershell
uv run pytest -m "not integration"
uv run pytest -m integration
```

This separates deterministic safety verification from a synthetic runtime path that may not request a sensitive action.

## 16. Run deterministic regression evaluation

```powershell
uv run supportops-evaluate-regression score
```

This command exercises repository-owned, credential-free regression evaluation.

## 17. Reproduce the static prompt comparison

Capture provenance:

```powershell
$gitCommit = git rev-parse HEAD
$captureTimestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

New-Item `
  -ItemType Directory `
  -Force `
  -Path "artifacts/evaluation/ticket-classification/prompt-v1-v2" |
  Out-Null
```

Run the paired static comparison:

```powershell
uv run supportops-evaluate-classification compare `
  --dataset evals/ticket-classification/datasets/ticket-classification-eval-v1.jsonl `
  --split-manifest evals/ticket-classification/splits/ticket-classification-eval-v1-splits-v1.json `
  --baseline-predictions evals/ticket-classification/predictions/ticket-classification-eval-v1.prompt-v1.static.jsonl `
  --candidate-predictions evals/ticket-classification/predictions/ticket-classification-eval-v1.prompt-v2.static.jsonl `
  --evidence-kind "static_fixture" `
  --capture-timestamp $captureTimestamp `
  --git-commit $gitCommit `
  --output artifacts/evaluation/ticket-classification/prompt-v1-v2/comparison.json `
  --baseline-manifest-output artifacts/evaluation/ticket-classification/prompt-v1-v2/baseline-manifest.json `
  --candidate-manifest-output artifacts/evaluation/ticket-classification/prompt-v1-v2/candidate-manifest.json `
  --pair-manifest-output artifacts/evaluation/ticket-classification/prompt-v1-v2/pair-manifest.json
```

Produce the explicit decision artifact:

```powershell
uv run supportops-evaluate-classification decide `
  --comparison artifacts/evaluation/ticket-classification/prompt-v1-v2/comparison.json `
  --decision-template evals/ticket-classification/decisions/ticket-classification-prompt-v2-decision.static.json `
  --output artifacts/evaluation/ticket-classification/prompt-v1-v2/decision.json
```

Inspect the result:

```powershell
Get-Content `
  artifacts/evaluation/ticket-classification/prompt-v1-v2/decision.json
```

The expected governance outcome is:

```text
outcome: inconclusive
run_status: incomplete
approved_for_runtime_adoption: false
separate_runtime_adoption_required: true
```

Prompt version 1 remains the runtime default. Prompt version 2 remains an immutable, non-adopted evaluation candidate.

Static fixtures validate comparison, provenance, safety-gate, and decision behavior. They do not establish provider-backed model superiority and cannot authorize runtime adoption.

## 18. Review the evidence

A reviewer should be able to verify:

| Evidence | Expected observation |
| --- | --- |
| Workspace | Synthetic workspace persisted in PostgreSQL |
| Knowledge document | Immutable document and version identifiers |
| Retrieval | Active runbook evidence returned through Qdrant and hydrated from PostgreSQL |
| Ticket intake | Ticket and initial AgentRun returned from one request |
| Worker reliability | AgentRun and attempt records available for inspection |
| AI invocation | Prompt, provider, model, tokens, and estimated-cost provenance persisted |
| Orchestration | LangGraph execution contained within the AgentRun boundary |
| Recommendation | Grounded output with ordered citations |
| Approval boundary | No sensitive action is assumed; approval records are durable when created |
| Observability | Optional telemetry does not replace durable state |
| Evaluation | Deterministic scoring and static paired comparison are reproducible |
| Prompt governance | Version 2 remains non-adopted after an inconclusive decision |

## 19. Clean up local infrastructure

Stop the API and worker processes with `Ctrl+C` in their terminals.

Then stop local containers:

```powershell
docker compose down
```

To remove local container volumes as well, use the repository's documented reset procedure rather than deleting data implicitly during the standard demonstration.

## Related documentation

- [Local setup](local-setup.md)
- [API examples](api-examples.md)
- [Approval workflow API](approval-workflow-api.md)
- [Architecture overview](../architecture/overview.md)
- [AgentRun reliability](../architecture/agent-run-reliability.md)
- [Knowledge ingestion and retrieval](../architecture/knowledge-ingestion-and-retrieval.md)
- [Controlled workflow](../architecture/controlled-workflow.md)
- [Human-approved workflow](../architecture/human-approved-workflow.md)
- [Evaluation and regression](../architecture/evaluation-and-regression.md)
- [Classification evaluation](../architecture/classification-evaluation.md)
- [Grounded recommendation evaluation](../architecture/grounded-recommendation-evaluation.md)