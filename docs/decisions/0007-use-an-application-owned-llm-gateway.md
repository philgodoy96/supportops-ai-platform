# ADR 0007: Use an Application-Owned LLM Gateway

## Status

Accepted

## Context

SupportOps AI Platform requires model-assisted ticket classification while
preserving deterministic application control over workflow state, retries,
persistence, security, and cost accounting.

The OpenAI SDK provides request construction, transport behavior, Responses API
parsing, Structured Outputs, usage metadata, request identifiers, and typed
exceptions. Using those SDK objects directly across business modules would
create several risks:

- provider-specific types would leak into application and domain code;
- model-provider changes would require broad business-layer modifications;
- provider exceptions could become persistence or API contracts;
- retries could be duplicated across SDK, gateway, and worker layers;
- prompt selection could become coupled to provider selection;
- raw responses and sensitive content could spread across modules;
- tests would require provider-specific fixtures throughout the codebase;
- cost calculation could depend on provider response formatting;
- fallback behavior could become implicit and operationally unsafe.

Structured Outputs reduce malformed-output risk but do not remove application
responsibility for business validation. Provider schema enforcement cannot
decide the platform taxonomy, prompt version, repair policy, retryability,
persistence boundary, or terminal workflow behavior.

The platform also needs a deterministic network-free provider for local
development and automated testing. That provider must remain explicit and must
never become a silent fallback after a production provider failure.

## Decision

SupportOps AI Platform will isolate LLM providers behind an application-owned,
asynchronous LLM Gateway.

### Provider-independent contracts

Application and workflow modules will depend on platform-defined request,
response, usage, error, and result types.

Provider adapters must not expose:

- SDK response objects;
- SDK exception classes;
- SDK usage objects;
- provider-specific message types;
- HTTP clients;
- raw provider payloads.

Each provider adapter will translate its SDK behavior into the application-owned
contract.

### Responses API

The initial OpenAI adapter will use the Responses API.

The adapter will use Pydantic-compatible Structured Outputs through
`responses.parse`.

Deprecated completion APIs and provider calls inside business modules are not
permitted.

### Application validation

Structured Outputs do not replace application validation.

Every candidate result must pass the application-owned Pydantic schema before
it can become accepted persistence data.

Application validation owns:

- enum membership;
- required fields;
- strict field types;
- string normalization and limits;
- schema version;
- rejection of additional fields;
- rejection of unapproved classifications.

Unknown values must not be silently coerced to `other`.

### Prompt versioning

Prompts are immutable repository artifacts.

A prompt is selected by explicit prompt ID and version.

The prompt registry must:

- reject duplicate prompt ID and version pairs;
- provide explicit lookup;
- expose deterministic content hashes;
- remain independent from provider selection;
- remain independent from model selection;
- avoid a mutable `latest` alias.

The initial prompt is:

```text
prompt_id = ticket-classification
version = 1
output_schema_id = ticket-classification-v1
```

Prompt version 2 will not be created without evaluation evidence that motivates
a behavioral change.

### Provider implementations

The initial providers are:

```text
MockLLMProvider
OpenAILLMProvider
```

`MockLLMProvider` is deterministic, network-free, configurable through explicit
outcomes, and transparent in provenance.

`OpenAILLMProvider` uses the official asynchronous OpenAI SDK, the Responses API,
Structured Outputs, explicit timeout configuration, explicit SDK transport
retry configuration, provider request identifier extraction, usage extraction,
and exception normalization.

### Explicit provider selection

Provider selection is deployment configuration.

The mock provider is not a fallback.

An OpenAI failure must never invoke the mock provider automatically.

Cross-provider fallback is intentionally deferred until:

- baseline provider behavior is observable;
- retry semantics are stable;
- invocation persistence exists;
- cost accounting exists;
- failure classes are understood;
- cross-provider schema compatibility can be evaluated.

### Error normalization

Provider operational failures must be normalized into stable application-owned
error classes.

The error taxonomy must distinguish:

- timeout;
- temporary rate limit;
- authentication failure;
- quota exhaustion;
- invalid request;
- provider unavailability;
- refusal;
- incomplete response;
- output validation failure;
- unexpected provider failure.

Each error category declares:

- stable error code;
- retryability;
- terminal behavior;
- repair eligibility;
- safe summary.

Raw provider exception messages are not persistence or API contracts.

Programming defects and violated internal invariants must remain visible as
application failures rather than being masked as provider outages.

### Retry separation

The platform separates:

1. provider transport retry;
2. gateway repair;
3. AgentRun retry.

Provider transport retry remains inside the provider SDK and is configured
explicitly.

Gateway repair is a new logical provider invocation. It is bounded to at most
one attempt and is used only for repair-eligible structured-output failures.

AgentRun retry is the outer durable operational retry. It owns attempt
scheduling, backoff, exhaustion, and terminal workflow state.

The gateway must not add a manual transport retry loop around the SDK.

Refusals, authentication failures, quota exhaustion, and invalid requests are
not repaired.

### Provider lifecycle

Provider clients are process-scoped resources.

The worker process will:

- construct the selected provider during composition;
- reuse one provider client across executions;
- close the provider during shutdown.

The API process will not initialize OpenAI unless an API-owned capability
requires it.

A new provider HTTP client must not be created for every ticket.

### Prompt and ticket safety

Trusted classification instructions and untrusted ticket data remain separate.

Ticket content may contain instructions or prompt-injection-like text. It is
treated only as support-ticket data and cannot change:

- taxonomy;
- schema;
- provider;
- model;
- workflow;
- repair policy;
- tool availability;
- application behavior.

The prompt must not request chain-of-thought.

Raw prompts and complete ticket content are not logged by default.

### Usage and cost accounting

Provider usage metadata is translated into application-owned token fields.

Estimated cost is calculated by the application through an immutable,
versioned, provider-and-model-specific pricing catalog.

Monetary calculation uses `Decimal`.

Every future invocation record will preserve the pricing catalog version used.

When provider or model pricing is unknown:

- token usage remains available;
- estimated cost remains `null`;
- unknown pricing is not treated as zero;
- classification does not fail solely because pricing is unavailable.

The mock provider has an explicit known zero-cost catalog entry.

Provider invoices remain the authoritative billing source.

### External-call transaction boundary

Provider calls must execute outside database transactions.

Durable workflow integration will use short transactions to load required state
and to persist invocation and classification results.

A process may fail after the provider completes but before persistence commits.
The platform therefore treats model invocation across process crashes as an
at-least-once external-call boundary.

The platform does not claim exactly-once provider cost.

Database uniqueness can prevent duplicate accepted classification records for
one AgentRun but cannot eliminate every repeated provider call across the crash
gap.

## Consequences

### Positive consequences

- business modules remain independent from provider SDK types;
- provider-specific request and response logic is centralized;
- provider replacement requires a new adapter rather than broad business-layer
  changes;
- error semantics remain stable across providers;
- retry layers remain explicit and independently testable;
- mock behavior is deterministic and network-free;
- OpenAI failure cannot silently produce a mock result;
- prompt identity is reproducible;
- schema identity remains application-owned;
- application validation remains authoritative;
- token and cost interpretation remain provider-independent;
- secrets and raw provider payloads remain inside a narrow boundary;
- provider lifecycle is explicit;
- future RAG and orchestration can reuse the provider boundary without
  distributing SDK calls.

### Trade-offs

- the platform must maintain its own contracts and error taxonomy;
- provider-specific features require explicit adapter extension;
- not every SDK capability is exposed immediately;
- normalized errors can lose provider-specific detail that is unsafe or
  irrelevant to application behavior;
- prompt definitions require deliberate version governance;
- the versioned pricing catalog requires maintenance when provider prices
  change;
- a moving model alias may reduce reproducibility compared with a snapshot;
- bounded repair adds provider latency and cost;
- at-least-once external calls can repeat provider cost after a process crash;
- lack of fallback means a configured provider outage remains visible instead
  of being hidden by another provider.

### Required engineering practices

This decision requires:

- no direct provider SDK imports in business modules;
- application-owned request and response contracts;
- application-owned schemas;
- application-side validation after provider parsing;
- explicit provider and model provenance;
- explicit prompt ID, version, schema ID, and content hash;
- deterministic mock outcomes;
- explicit provider selection;
- no silent fallback;
- bounded provider transport retries;
- bounded repair attempts;
- explicit AgentRun retry translation;
- provider calls outside database transactions;
- safe stable error codes;
- no raw provider exception persistence;
- no API key or complete-settings logging;
- no complete prompt or ticket-content logging by default;
- `Decimal` for persisted monetary estimates;
- null estimated costs for unknown pricing;
- unit tests without provider network access;
- opt-in external-provider validation only.

## Alternatives considered

### Use the OpenAI SDK directly inside the classification service

Rejected because provider-specific request types, exceptions, usage objects, and
lifecycle behavior would leak into workflow code.

This would also make testing, provider replacement, and error governance more
difficult.

### Use LangChain or another model wrapper

Not selected because the initial workflow requires a small, explicit provider
boundary rather than a general orchestration abstraction.

Additional framework behavior would obscure request construction, retries,
structured-output parsing, error mapping, and lifecycle ownership.

LangGraph remains planned for controlled workflow orchestration after the LLM
boundary and classification persistence are established.

### Rely only on Structured Outputs

Rejected because provider schema enforcement does not replace application-owned
validation, business invariants, schema versioning, repair policy, or terminal
failure decisions.

### Make the mock provider an automatic fallback

Rejected because it would convert a real provider outage into a synthetic
business result while hiding provider provenance and operational failure.

The mock provider is selected explicitly for local development and tests.

### Add cross-provider fallback immediately

Deferred because fallback would introduce model compatibility, prompt
portability, retry ordering, cost, observability, and failure-precedence
decisions before the baseline provider is measurable.

### Store prompts in an external observability platform

Not selected as the source of truth for this phase.

Repository-versioned prompts provide code review, deterministic deployment
provenance, and reproducible hashes.

An observability platform may later mirror or annotate prompt execution without
becoming the authoritative prompt definition store.

### Calculate cost from provider-formatted monetary fields

Rejected because provider response formats are not a stable application
accounting contract.

The application owns a versioned pricing catalog and treats provider invoices
as authoritative.

### Hold a database transaction during the provider call

Rejected because model latency and provider failure would extend database
transaction duration, retain connections and locks, and reduce worker
reliability.

Short persistence transactions surround the external call instead.