# ADR 0001: Use a Modular Monolith

## Status

Accepted

## Context

SupportOps AI Platform will evolve across support operations, asynchronous processing, retrieval, controlled AI orchestration, human approval, observability, usage accounting, and evaluation.

These capabilities require clear ownership and internal boundaries, but the initial platform does not require the operational complexity of independently deployed services.

A distributed architecture would introduce additional concerns before they are justified by workload or organizational needs, including:

- service discovery;
- network failure handling;
- distributed tracing requirements;
- cross-service authentication;
- schema and contract versioning;
- deployment coordination;
- eventual consistency;
- duplicated operational tooling;
- more complex local development and testing.

The platform also requires the API and future asynchronous worker to share domain behavior, application services, persistence rules, and infrastructure adapters.

The architecture must preserve modularity without turning process boundaries into premature system boundaries.

## Decision

SupportOps AI Platform will use an API-first modular monolith.

The system will be maintained as a single Python codebase with explicit internal module boundaries.

The HTTP API and future asynchronous worker will run as separate processes while importing the same application package.

They will share:

- domain models;
- application services;
- PostgreSQL persistence;
- infrastructure adapters;
- configuration conventions;
- testing standards.

Business capabilities will be introduced as cohesive modules only when they have concrete responsibilities.

Framework composition will remain outside future business modules.

Infrastructure-specific code will remain isolated from business rules.

Module extraction into separately deployed services will be considered only when supported by clear operational evidence such as:

- independently scaling workloads;
- distinct reliability requirements;
- independent deployment ownership;
- strong data ownership boundaries;
- measurable coordination or performance constraints.

## Consequences

### Positive consequences

- deployment topology remains simple during early platform growth;
- local development remains reproducible;
- transactions can span related business operations without distributed coordination;
- API and worker behavior can reuse the same tested application logic;
- module boundaries can be reviewed directly in the codebase;
- integration testing remains practical;
- operational complexity is introduced only when justified;
- future service extraction remains possible around established module boundaries.

### Trade-offs

- internal boundaries must be enforced through code organization and review rather than network isolation;
- poorly controlled imports could create coupling between modules;
- the API and worker share release cadence while they remain in one repository;
- independent deployment of individual business modules is not available by default;
- connection pool and resource planning must account for multiple processes using the same infrastructure.

### Required engineering practices

The decision requires:

- explicit dependency direction;
- no circular module dependencies;
- infrastructure isolation;
- no generic shared package for unrelated behavior;
- business modules with clear ownership;
- tests around module contracts and invariants;
- architectural review before introducing cross-module dependencies.

## Alternatives considered

### Microservices from the initial release

Rejected because the platform does not yet have workload, ownership, or deployment requirements that justify distributed system complexity.

Starting with microservices would increase operational surface area while weakening the ability to evolve business boundaries through direct refactoring.

### Single-layer monolith

Rejected because placing transport, business logic, persistence, and provider integrations together would make the system difficult to test and evolve.

The decision is not to build an unstructured monolith. It is to build a modular monolith with enforceable internal boundaries.

### Separate repositories for the API and worker

Rejected for the initial platform because both processes are expected to share domain behavior, persistence models, and application services.

Separate repositories would create duplication or cross-repository version coordination before independent ownership is required.

### Serverless functions for each capability

Rejected as the default application architecture because the platform requires durable workflow state, cohesive business transactions, and controlled orchestration.

Serverless components may be evaluated later for isolated workloads when their operational characteristics provide a concrete advantage.
