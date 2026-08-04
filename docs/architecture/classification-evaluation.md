# Classification Inspection and Evaluation

## Purpose

This document describes the read-only operational inspection and offline
evaluation capabilities for durable ticket classification.

The classification workflow produces durable application records:

```text
AgentRun
→ AgentRunAttempt
→ LLMInvocation
→ TicketClassification
```

Inspection makes those records safely observable through workspace-scoped HTTP
queries. Evaluation measures the structured classification behavior against a
versioned synthetic dataset without mutating production state.

These capabilities complete a deliberate separation:

```text
runtime execution
→ produces durable state

operational inspection
→ reads durable state

offline evaluation
→ measures model behavior
```

The three boundaries share taxonomy, prompt provenance, schema contracts, and
cost semantics, but they do not share transaction ownership or lifecycle
responsibilities.

## Implemented capabilities

The platform implements:

- workspace-scoped classification detail inspection;
- ticket-scoped classification history with opaque keyset pagination;
- optional accepted-classification references in AgentRun inspection;
- AgentRun-scoped logical LLM invocation history;
- safe prompt, provider, model, usage, latency, error, and estimated-cost
  projections;
- an immutable versioned synthetic classification dataset;
- a versioned sidecar split manifest with development, holdout, and
  safety-gate allocation;
- repository-owned evaluation manifest contracts;
- typed prediction envelopes;
- deterministic canonical serialization and SHA-256 hashing;
- canonical dataset and prediction artifact hashing;
- deterministic structured-label evaluation;
- per-field accuracy and full structured-label exact match;
- human-review precision, recall, and F1;
- structured-output validity and invalid-output rate;
- high urgency recall;
- critical urgency recall;
- high-risk human-review recall;
- average latency;
- average input, output, and total token metrics;
- failed-case and safe error-code accounting;
- token-usage and estimated-cost aggregation;
- standalone classification release-gate evaluation;
- gate categories safety, quality, reliability, and efficiency;
- explicit gate outcomes `passed`, `failed`, and `not_applicable`;
- standalone aggregate statuses `passed`, `failed`, and `incomplete`;
- offline scoring of existing prediction artifacts;
- sequential mock or OpenAI evaluation execution;
- explicit evaluation prompt-version selection with no implicit latest
  selection;
- an explicit external-provider permission gate;
- atomic prediction and report artifact writes.

## Architectural boundaries

### Runtime classification

Runtime classification is owned by the PostgreSQL-backed worker.

```text
configured AgentRun
→ exact workflow registry dispatch
→ ticket-classification-v1 executor
→ application-owned LLM Gateway
→ provider call outside database transaction
→ lease-fenced invocation and classification persistence
→ lease-fenced AgentRun completion
```

Runtime classification owns:

- workflow dispatch;
- provider invocation;
- durable logical invocation history;
- accepted classification persistence;
- lease fencing;
- retry and recovery semantics.

Runtime classification does not own:

- HTTP response projection;
- evaluation metric calculation;
- dataset management;
- prompt promotion.

### Operational inspection

Operational inspection is read-only.

```text
HTTP request
→ workspace-scoped application service
→ read-only query repository
→ safe response projection
```

Inspection owns:

- resource ownership validation;
- safe public projections;
- deterministic read ordering;
- opaque pagination;
- stable not-found behavior.

Inspection does not:

- claim AgentRuns;
- retry execution;
- revoke leases;
- mutate classifications;
- invoke an LLM provider;
- expose provider-internal request identifiers;
- expose raw prompts or responses.

### Offline evaluation

Offline evaluation is independent from the API and worker processes and remains
separate from runtime business authority and optional observability.

```text
immutable dataset + split sidecar
→ explicit prompt version
→ prediction artifact or evaluation predictor
→ deterministic evaluator
→ standalone release-gate evaluation
→ JSON report
```

Evaluation owns:

- synthetic case validation;
- split-manifest validation;
- artifact provenance;
- prediction alignment;
- deterministic quality, safety, validity, latency, and usage metrics;
- standalone release-gate evaluation;
- usage and estimated-cost aggregation;
- reproducible reports.

Evaluation does not:

- access PostgreSQL;
- access Qdrant;
- create AgentRuns;
- persist TicketClassification records;
- mutate Ticket status;
- change the runtime classification prompt pin;
- authorize prompt promotion from standalone reports;
- promote prompts automatically;
- execute tools or approvals.

Repository-owned evaluation and regression architecture is documented in
[`evaluation-and-regression.md`](evaluation-and-regression.md).

## Inspection API

All business routes are versioned under `/api/v1`.

### Classification detail

```http
GET /api/v1/workspaces/{workspace_id}/ticket-classifications/{classification_id}
```

The response includes:

- classification identity;
- workspace, ticket, and AgentRun ownership;
- accepted logical invocation identity;
- category;
- intent;
- urgency;
- sentiment;
- human-review recommendation;
- bounded summary;
- schema version;
- prompt ID, version, and content hash;
- provider and model identity;
- creation timestamp.

The response excludes:

- provider request ID;
- raw prompt;
- raw provider response;
- SDK exception text;
- worker identity;
- lease token;
- lease expiry;
- execution request ID;
- chain-of-thought.

### Ticket classification history

```http
GET /api/v1/workspaces/{workspace_id}/tickets/{ticket_id}/classifications
```

Supported query parameters:

```text
page_size
cursor
```

Ordering is deterministic:

```text
created_at DESC
id DESC
```

The route validates ticket ownership before reading classification history.

Although the current intake flow schedules one initial AgentRun, the API is
intentionally list-shaped. Future reclassification should create another
immutable AgentRun and TicketClassification rather than overwrite accepted
history.

### AgentRun classification reference

```http
GET /api/v1/workspaces/{workspace_id}/agent-runs/{agent_run_id}
```

AgentRun inspection includes:

```json
{
  "classification": {
    "id": "classification-uuid",
    "schema_version": "ticket-classification-v1",
    "created_at": "2026-08-01T21:45:00Z"
  }
}
```

Runs without an accepted classification return:

```json
{
  "classification": null
}
```

The AgentRun domain entity does not contain classification fields. The response
is composed through a cross-module application query. This preserves ownership:

```text
agent_runs
→ execution lifecycle

ticket_classifications
→ accepted model interpretation
```

### Logical invocation history

```http
GET /api/v1/workspaces/{workspace_id}/agent-runs/{agent_run_id}/llm-invocations
```

Ordering is:

```text
attempt_number ASC
invocation_sequence ASC
```

Each item exposes:

- invocation identity;
- AgentRunAttempt identity and number;
- logical invocation sequence;
- normalized status;
- provider and model;
- prompt provenance;
- output schema version;
- provider-reported token usage when known;
- versioned application-estimated cost;
- latency;
- safe application error code;
- creation timestamp.

The route is not paginated. Invocation volume is naturally bounded by the
AgentRun retry budget and the Gateway repair budget.

## Workspace ownership and not-found behavior

Every public classification read contains `workspace_id` in its persistence
predicate.

Ownership keys are:

```text
classification detail
= workspace_id + classification_id

ticket classification history
= workspace_id + ticket_id

AgentRun invocation history
= workspace_id + agent_run_id
```

Missing and cross-workspace resources use the same response contract.

Examples:

```text
unknown classification
→ 404 ticket_classification_not_found

classification owned by another workspace
→ 404 ticket_classification_not_found

unknown or cross-workspace ticket
→ 404 ticket_not_found

unknown or cross-workspace AgentRun
→ 404 agent_run_not_found
```

This behavior prevents the inspection API from revealing that a resource exists
under another workspace.

Workspace scoping is a data ownership boundary. It does not authenticate the
caller and does not establish secure tenant isolation without an authentication
and authorization layer.

## Read-model design

Write and read responsibilities remain separate.

```text
TicketClassificationRepository
→ workflow idempotency lookup
→ accepted classification persistence

LLMInvocationRepository
→ logical invocation persistence

TicketClassificationQueryRepository
→ classification detail
→ AgentRun classification lookup
→ lightweight AgentRun reference
→ ticket classification history
→ safe invocation history
```

The query repository:

- does not begin transactions;
- does not commit or roll back;
- does not flush;
- does not lock rows;
- does not return ORM records;
- does not modify write-path fencing behavior.

## Keyset pagination

Classification history uses an opaque cursor containing a canonical keyset
position:

```text
created_at
classification_id
```

The repository receives decoded values rather than the cursor string.

The query applies:

```text
(created_at, id) < (:after_created_at, :after_classification_id)
```

with descending ordering.

This avoids offset instability when new classifications are inserted between
requests.

The cursor is versioned. Invalid or unsupported cursors produce the stable
public error:

```json
{
  "error": {
    "code": "invalid_pagination_cursor",
    "message": "Pagination cursor is invalid.",
    "request_id": "request-uuid"
  }
}
```

## Evaluation dataset

The committed dataset is:

```text
dataset_id = ticket-classification-eval
version = 1
schema_version = ticket-classification-v1
file = evals/ticket-classification/datasets/ticket-classification-eval-v1.jsonl
```

Dataset version 1 remains immutable. It contains 24 synthetic cases.

Coverage includes:

- every category;
- every intent;
- every urgency level;
- every sentiment value;
- human-review true and false;
- ambiguous requests;
- security-sensitive requests;
- prompt-injection text inside untrusted ticket content;
- emotion-versus-operational-impact distinctions.

The dataset contains no customer data, production tickets, secrets, provider
responses, or chain-of-thought.

### Split sidecar

Split allocation is stored separately from behavioral content:

```text
file = evals/ticket-classification/splits/ticket-classification-eval-v1-splits-v1.json
split_manifest_id = ticket-classification-eval-splits
split_manifest_version = 1
```

The first classification split allocates:

```text
development: 12
holdout: 8
safety_gate: 4
```

Development cases may guide failure analysis and prompt-change hypotheses.
Holdout cases remain reserved for final paired evaluation. Holdout outcomes
must not guide prompt drafting. Safety-gate cases cover critical urgency,
credential exposure, privacy-sensitive behavior, prompt injection, and
mandatory human review.

A behavioral change requires a new dataset version. A procedural split change
requires a new split-manifest version.

### Dataset invariants

Each case has:

```text
case_id
tags
ticket.subject
ticket.description
expected structured labels
```

Validation requires:

- lowercase kebab-case case IDs and tags;
- unique case IDs;
- unique tags per case;
- production ticket length limits;
- production taxonomy enums;
- the current structured classification schema version;
- no unknown fields;
- at least one case.

### Dataset hash

The dataset content hash is SHA-256 over canonical JSONL.

Canonicalization:

- sorts JSON object keys;
- removes insignificant whitespace;
- preserves case order;
- preserves Unicode content;
- terminates every canonical record with a newline.

Formatting-only JSON changes do not alter the hash. Case reordering or semantic
changes do.

The dataset is immutable after it is used for a published evaluation result.
Changes require a new dataset version.

## Prediction artifacts

Prediction artifacts use validated JSONL.

Each prediction records:

- dataset case ID;
- successful or failed status;
- prompt provenance;
- provider and model;
- normalized structured output for success;
- safe final error code for failure;
- ordered logical invocation traces;
- usage, cost, latency, and safe error metadata.

Prediction artifacts do not contain:

- API keys;
- provider request IDs;
- raw provider responses;
- database identifiers;
- lease data;
- chain-of-thought.

### Prediction invariants

Prediction validation requires:

- unique case IDs;
- contiguous logical invocation sequences beginning at one;
- one provider and model matching prediction provenance;
- exactly one final successful invocation for successful predictions;
- no successful invocation for failed predictions;
- final invocation error matching the failed prediction error;
- valid token relationships;
- valid cost relationships;
- no invented cost when pricing is unknown.

Prediction artifacts also receive a canonical SHA-256 content hash.

## Deterministic metrics

The evaluator is provider-independent and performs no I/O after artifacts are
loaded.

The full structured-label exact match includes:

```text
category
intent
urgency
sentiment
requires_human_review
```

The evaluator also reports:

- category accuracy;
- intent accuracy;
- urgency accuracy;
- sentiment accuracy;
- human-review accuracy;
- human-review precision;
- human-review recall;
- human-review F1;
- structured-output validity;
- invalid-output rate;
- high urgency recall;
- critical urgency recall;
- high-risk human-review recall;
- average latency;
- average input tokens;
- average output tokens;
- average total tokens;
- successful prediction count;
- failed prediction count;
- failures by safe error code;
- known token totals;
- unknown-usage count;
- known estimated total cost;
- unknown-pricing count;
- pricing catalog versions;
- per-case results.

Rates use Decimal values quantized to six decimal places.

A structurally valid prediction may still be behaviorally incorrect. Structural
validity and label correctness remain separate. Critical urgency recall accepts
only an exact critical prediction for an expected critical case. High-risk
human-review recall uses explicit dataset expectations and an application-owned
set of high-risk tags.

## Standalone release gates

The initial classification gate profile is:

```text
profile_id = ticket-classification-release-gates
profile_version = 1
```

Gate categories are:

```text
safety
quality
reliability
efficiency
```

Individual gate outcomes are:

```text
passed
failed
not_applicable
```

Standalone aggregate statuses are:

```text
passed
failed
incomplete
```

Aggregate semantics:

```text
failed
→ at least one blocking gate failed

incomplete
→ no blocking gate failed, but at least one blocking gate is not applicable

passed
→ every blocking gate passed
```

Standalone safety and reliability gates evaluate absolute evidence such as
structured-output validity, critical urgency recall, high-risk human-review
recall, prediction artifact coverage, and successful deterministic report
generation. Quality and efficiency non-regression gates require paired baseline
evidence and remain `not_applicable` for a standalone report. A perfect
standalone report is therefore intentionally `incomplete`. Standalone reports
cannot authorize prompt promotion.

### Summary evaluation

Summary text is intentionally excluded from exact match.

Multiple summaries can be factually equivalent. Exact string comparison would
measure wording rather than quality. Introducing an LLM judge would add another
probabilistic dependency and require its own reliability and cost methodology.

Summary remains constrained by the production output schema. Semantic summary
evaluation remains a future evaluation capability.

### Failed and missing predictions

A normalized provider failure:

```text
counts as a failed prediction
→ all label matches are false
→ safe error code is aggregated
```

A missing prediction:

```text
counts as a failed prediction
→ error_code = prediction_missing
```

A prediction for an unknown dataset case is rejected because the artifact cannot
be aligned safely.

## Evaluation commands

### Explicit prompt-version selection

The evaluation CLI selects prompt version explicitly:

```powershell
uv run supportops-evaluate-classification run `
  --provider mock `
  --prompt-version 1 `
  --dataset `
    evals/ticket-classification/datasets/ticket-classification-eval-v1.jsonl `
  --predictions-output `
    artifacts/classification-mock-predictions.jsonl `
  --output `
    artifacts/classification-mock-report.json
```

The command remains domain-specific, so the stable prompt ID remains implicit:

```text
ticket-classification
```

The default evaluation prompt version remains `1`. There is no implicit latest
version. An unsupported version fails without provider execution, fallback, or
artifact replacement. Runtime classification remains independently pinned to
its approved prompt version. Evaluation selection does not change the
production default.

### Offline scoring

```powershell
uv run supportops-evaluate-classification score `
  --dataset `
    evals/ticket-classification/datasets/ticket-classification-eval-v1.jsonl `
  --predictions `
    artifacts/classification-predictions.jsonl `
  --output `
    artifacts/classification-report.json
```

Offline scoring:

- does not instantiate settings;
- does not instantiate a provider;
- does not require an API key;
- does not accept `--prompt-version`;
- does not access PostgreSQL;
- does not access Qdrant;
- performs no network requests.

### Mock pipeline run

```powershell
uv run supportops-evaluate-classification run `
  --provider mock `
  --prompt-version 1 `
  --dataset `
    evals/ticket-classification/datasets/ticket-classification-eval-v1.jsonl `
  --predictions-output `
    artifacts/classification-mock-predictions.jsonl `
  --output `
    artifacts/classification-mock-report.json
```

Mock execution validates:

- dataset loading;
- explicit prompt resolution;
- prompt rendering;
- Gateway integration;
- prediction generation;
- artifact writing;
- deterministic scoring;
- standalone release-gate evaluation.

Mock results are not presented as a model-quality baseline.

### OpenAI evaluation

```powershell
uv run supportops-evaluate-classification run `
  --provider openai `
  --allow-external-provider `
  --prompt-version 1 `
  --dataset `
    evals/ticket-classification/datasets/ticket-classification-eval-v1.jsonl `
  --predictions-output `
    artifacts/classification-openai-predictions.jsonl `
  --output `
    artifacts/classification-openai-report.json
```

OpenAI evaluation requires:

- explicit `--provider openai`;
- explicit `--allow-external-provider`;
- `SUPPORTOPS_OPENAI_API_KEY`;
- valid LLM timeout and retry settings.

The command fails closed if the external-provider permission flag is absent.

The flag is rejected for the mock provider so it cannot become a meaningless
habit or hidden default.

## Provider composition

Evaluation has a dedicated settings class containing only LLM configuration.

It does not require:

- PostgreSQL configuration;
- Qdrant configuration;
- API lifecycle configuration;
- worker lease settings;
- worker polling settings.

The evaluation runtime owns:

```text
provider
LLM Gateway
model identity
```

Provider selection is explicit:

```text
mock
→ MockLLMProvider only

openai
→ OpenAILLMProvider only
```

There is no cross-provider fallback.

The provider is closed after execution, including failure paths.

## Execution and artifact behavior

Evaluation cases run sequentially by default.

Sequential execution intentionally provides:

- predictable provider rate;
- predictable cost progression;
- straightforward failure attribution;
- deterministic artifact ordering;
- simpler operational review.

Parallel execution remains deferred until dataset size or evaluation latency
creates a measurable need.

Prediction and report artifacts are written through atomic replacement:

```text
write temporary file
→ flush
→ fsync
→ replace destination
```

Generated artifacts belong under `artifacts/`, which is ignored by Git.

Versioned datasets remain committed under `evals/`.

## Report provenance

Every report includes:

- dataset ID;
- dataset version;
- dataset content hash;
- predictions content hash;
- prompt ID;
- prompt version;
- prompt content hash;
- provider;
- model;
- pricing catalog versions;
- deterministic report content hash.

The report content hash is SHA-256 over canonical report content before the hash
field is added.

Re-scoring the same dataset and prediction artifact produces the same report
content hash.

## Failure semantics

### Input and configuration failures

Examples:

- unreadable dataset;
- invalid dataset record;
- duplicate case ID;
- invalid prediction record;
- unknown prediction case;
- mixed prompt or runtime provenance;
- unsupported evaluation prompt version;
- missing OpenAI API key;
- missing external-provider permission.

CLI exit code:

```text
2
```

### Artifact I/O and unexpected runtime failures

Examples:

- report write failure;
- unexpected provider adapter failure outside normalized Gateway behavior;
- unexpected programming error.

CLI exit code:

```text
1
```

### Successful execution

CLI exit code:

```text
0
```

Normalized per-case provider failures do not abort the full run. They become
failed prediction records and are represented in the report.

## Security model

The inspection and evaluation boundaries intentionally avoid sensitive runtime
content.

Controls include:

- synthetic committed datasets only;
- workspace predicates on every public classification read;
- stable missing and cross-workspace `404` responses;
- no raw provider responses in HTTP or evaluation artifacts;
- no provider request IDs in public projections;
- no API keys in reports or logs;
- no lease or execution fencing identifiers in responses;
- untrusted ticket content remains data, not instructions;
- no external provider initialization without explicit selection;
- no OpenAI execution without explicit permission;
- no database writes from evaluation.

## Cost semantics

Cost values are application estimates derived from:

```text
provider
model
provider-reported token usage
versioned pricing catalog
```

They are not invoices.

The evaluation report preserves:

- known estimated totals;
- unknown pricing counts;
- pricing catalog versions;
- exact Decimal arithmetic.

Provider invoice reconciliation and operational cost dashboards remain separate
future capabilities.

## Testing strategy

Inspection tests cover:

- query repository SQL shape;
- workspace predicates;
- classification detail;
- ticket-scoped keyset pagination;
- AgentRun classification reference;
- invocation ordering;
- safe response projection;
- empty histories;
- missing and cross-workspace behavior;
- real PostgreSQL integration.

Evaluation tests cover:

- evaluation contract hashing and atomic artifact writes;
- evaluation manifest and prediction-envelope validation;
- dataset validation and pinned hash;
- split-manifest validation and frozen allocation;
- prediction validation and hashing;
- exact-match and field-level metrics;
- structured-output validity and invalid-output rate;
- urgency and high-risk human-review recall;
- latency and token aggregates;
- human-review confusion matrix and F1;
- missing and failed predictions;
- usage and cost aggregation;
- standalone release-gate outcomes and aggregate statuses;
- quality and efficiency gates remaining not applicable without paired
  baseline evidence;
- prompt and runtime provenance consistency;
- explicit prompt-version selection and unsupported-version failure;
- mock Gateway prediction;
- repair trace preservation;
- provider lifecycle;
- CLI safety gates;
- offline scoring;
- artifact reload and deterministic report hashes.

Normal automated tests perform no paid external provider calls.

## Intentional trade-offs

### Read-only inspection

Classification mutation, manual override, and reclassification endpoints remain
deferred. The current boundary makes immutable accepted history observable
before introducing new state transitions.

### Filesystem evaluation artifacts

Evaluation results are stored as reproducible local files rather than production
database records. This keeps experimental quality analysis outside operational
transactions and avoids creating retention, tenancy, and migration requirements
before an evaluation history product exists.

### Sequential provider execution

Sequential execution prioritizes cost control, rate predictability, and
auditability over throughput.

### No summary exact match

The evaluator avoids a misleading metric until semantic summary quality has a
defensible methodology.

### No automatic prompt promotion

The platform records prompt provenance and generates evidence. It does not
automatically change production behavior. Standalone release-gate reports
cannot authorize promotion. Prompt version 2 and paired comparison remain
planned and should be introduced only after evaluation results identify
concrete failure patterns and the change can be reviewed explicitly.

### No RAGAS in classification evaluation

RAGAS remains planned for retrieval and grounded-generation evaluation. The
current capability evaluates a bounded structured classification task without a
retrieval context, so adding RAGAS would not serve a concrete architectural
responsibility.

## Intentionally deferred capabilities

The following remain outside this boundary:

- classification mutation and override APIs;
- reclassification scheduling;
- prompt version 2;
- paired prompt comparison;
- canonical baseline evidence for paired comparison;
- prompt promotion, rejection, or inconclusive decisions;
- automatic prompt promotion;
- automatic prompt optimization;
- evaluation database persistence;
- evaluation dashboards beyond standalone classification release gates;
- scheduled or online evaluation runs;
- parallel provider execution;
- cross-provider fallback;
- automatic model routing;
- Anthropic provider;
- LLM-as-judge summary scoring;
- operational invoice reconciliation;
- RAG;
- retrieval evaluation;
- controlled-support evaluation;
- approval-workflow evaluation;
- grounded recommendation evaluation;
- RAGAS;
- Langfuse datasets or experiments;
- production feedback ingestion;
- LangGraph orchestration;
- tools;
- human approval workflows;
- frontend inspection.

These capabilities remain possible without changing the ownership boundaries
established here.