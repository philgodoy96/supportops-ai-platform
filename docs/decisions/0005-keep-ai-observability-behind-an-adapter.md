# ADR 0005: Keep AI Observability Behind an Application-Owned Adapter

## Status

Accepted

## Context

SupportOps AI Platform will eventually require observability across AI-assisted workflows, including:

- prompt and model execution;
- retrieval context;
- tool calls;
- approval boundaries;
- token usage;
- cost attribution;
- latency;
- failures;
- evaluation signals;
- workflow traces.

External observability platforms can accelerate tracing, analysis, and debugging. However, direct provider integration throughout application code would create several risks:

- provider-specific APIs leaking into business and orchestration logic;
- inconsistent instrumentation across modules;
- difficult provider replacement;
- unclear ownership of sensitive payloads;
- uncontrolled capture of prompts, retrieved content, or user data;
- tests coupled to an external observability service;
- application behavior depending on telemetry availability.

The platform must retain control over what telemetry is emitted, how sensitive data is handled, and how observability failures affect business execution.

Langfuse is the approved initial observability platform direction, but it must remain an external adapter rather than an application dependency boundary.

## Decision

AI observability will be accessed through an application-owned adapter.

Application and orchestration code will depend on platform-defined observability contracts rather than importing Langfuse-specific clients directly.

The adapter boundary is expected to support future capabilities such as:

- workflow traces;
- spans or steps;
- prompt version identifiers;
- model metadata;
- token usage;
- cost metadata;
- retrieval metadata;
- tool execution metadata;
- evaluation results;
- error recording.

The application will define the semantic events and metadata it owns.

The provider adapter will translate those events into the selected observability platform.

Telemetry emission must not become the authoritative source of:

- workflow state;
- approval state;
- audit state;
- usage billing records;
- business outcomes.

Authoritative operational and accounting state will remain in PostgreSQL where required.

Observability failures must not corrupt or roll back successful business processing unless a future compliance requirement explicitly makes telemetry persistence mandatory.

Sensitive content must be filtered, redacted, or omitted according to explicit policy before leaving application-controlled boundaries.

Langfuse will not be self-hosted in the initial Docker Compose environment.

The repository foundation phase does not install or integrate:

- Langfuse;
- OpenTelemetry;
- external tracing SDKs;
- evaluation SDKs;
- AI provider SDKs.

The adapter will be introduced only when a concrete AI workflow requires observable behavior.

## Consequences

### Positive consequences

- business and orchestration logic remain independent from a telemetry provider;
- Langfuse can be replaced or supplemented without broad code changes;
- telemetry semantics remain consistent across modules;
- sensitive data handling can be centralized;
- tests can use in-memory or no-op adapters;
- telemetry outages do not automatically stop business workflows;
- authoritative usage and audit state remain under platform ownership;
- provider-specific configuration stays isolated.

### Trade-offs

- the platform must design and maintain its own observability contract;
- a custom adapter may not expose every provider feature immediately;
- provider-specific capabilities require explicit extension rather than direct use;
- semantic consistency requires governance across workflow implementations;
- asynchronous or buffered telemetry delivery may require later reliability decisions;
- no-op behavior can hide instrumentation gaps unless tests and reviews verify expected events.

### Required engineering practices

The decision requires:

- application-owned event names and metadata conventions;
- explicit redaction and data classification rules;
- tests for emitted observability events;
- no direct provider imports in business modules;
- bounded telemetry calls;
- safe failure handling;
- correlation with durable workflow identifiers;
- separate treatment of operational telemetry and authoritative audit records;
- documented retention and access policies before production use;
- cost and token records persisted authoritatively when required for accounting.

## Alternatives considered

### Import Langfuse directly throughout workflow code

Rejected because provider-specific concerns would spread through orchestration and business logic, making replacement, testing, and data governance more difficult.

### Self-host Langfuse in the initial Docker Compose stack

Rejected because the repository foundation focuses on PostgreSQL, Qdrant, application health, and testability.

Self-hosting observability would add infrastructure and operational responsibilities before any AI workflow exists.

A hosted or separately managed deployment can be evaluated when observability is introduced.

### Use only application logs for AI observability

Rejected as the long-term strategy because structured logs alone do not provide the trace relationships, prompt metadata, evaluation context, and workflow visualization required for production AI systems.

Application logs remain important for operational diagnostics.

### Use OpenTelemetry as the only application contract

Not selected as the initial AI observability boundary because general distributed tracing does not by itself define the AI-specific semantics required for prompts, retrieval, tools, tokens, costs, and evaluations.

OpenTelemetry may be introduced later beneath or alongside the application-owned adapter.

### Make observability delivery transactional with every workflow step

Rejected as the default because telemetry availability should not determine successful business execution.

Specific compliance or accounting records may require durable transactional persistence, but those records belong in authoritative platform storage rather than an external tracing system.
