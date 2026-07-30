# ADR 0004: Use a PostgreSQL-Backed Worker Model

## Status

Accepted

## Context

SupportOps AI Platform will eventually execute asynchronous work for support processing, classification, retrieval preparation, controlled orchestration, approval continuation, and evaluation.

This work must be:

- durable across process restarts;
- observable through persisted state;
- recoverable after partial failure;
- safe under retries;
- coordinated with transactional business changes;
- testable under concurrency;
- compatible with separate API and worker processes.

The platform could introduce an external queue or broker from the beginning. However, the initial system does not yet require the throughput, fan-out, or integration characteristics that would justify the additional operational surface.

A worker model backed by PostgreSQL keeps asynchronous work state close to the authoritative business data and allows job creation to participate in the same transaction as the business operation that requires it.

The architecture must still account for concurrency, duplicate execution, process crashes, and stale work ownership.

## Decision

SupportOps AI Platform will use a PostgreSQL-backed asynchronous worker model for the initial platform version.

The future worker will run as a process separate from the API while importing the same Python package.

PostgreSQL will own durable asynchronous work state.

The worker model is expected to define explicit behavior for:

- job creation;
- job availability;
- atomic claiming;
- ownership leases or equivalent bounded ownership;
- retries;
- retry scheduling;
- terminal failure;
- idempotent execution;
- stale claim recovery;
- concurrency limits;
- graceful shutdown;
- operational inspection.

Where a business state transition requires asynchronous follow-up, creation of the durable work record should occur within the same database transaction whenever practical.

The implementation must not rely on in-memory queues for authoritative work ownership.

Redis, Celery, Kafka, and SQS are intentionally not part of the initial worker foundation.

They may be introduced later if concrete requirements emerge for:

- higher throughput;
- broad event fan-out;
- cross-system integration;
- independent message retention;
- geographic distribution;
- workload isolation beyond what the database-backed model can provide.

The repository foundation phase establishes package boundaries that can support a separate worker process but does not implement:

- a worker entry point;
- job tables;
- polling;
- claiming;
- leases;
- retries;
- scheduling;
- worker health;
- queue abstractions.

## Consequences

### Positive consequences

- asynchronous work state remains durable and queryable;
- business state and required follow-up work can be committed atomically;
- local development requires fewer infrastructure services;
- operational debugging can use standard database queries;
- job ownership and retry behavior can be tested directly;
- API and worker share application and domain behavior;
- process crashes do not require in-memory recovery;
- idempotency and concurrency controls can use database constraints and transactional locking.

### Trade-offs

- worker polling adds database load;
- claim queries and indexes require careful design;
- long-running work must not hold database transactions open;
- lease duration and stale claim recovery require explicit semantics;
- high-throughput workloads may outgrow a database-backed polling model;
- database outages affect both transactional operations and asynchronous progress;
- fairness and prioritization require deliberate query and schema design;
- retry storms can increase database contention without backoff and limits.

### Required engineering practices

The decision requires:

- short transaction boundaries around claiming and state transitions;
- no database transaction held for the duration of external AI or network calls;
- idempotent handlers;
- durable attempt tracking;
- bounded retries;
- explicit terminal states;
- safe stale ownership recovery;
- concurrency and duplicate-execution tests;
- indexes that support claim queries;
- process-specific database connection pools;
- metrics and logs for queue depth, latency, attempts, and failures when observability is introduced.

## Alternatives considered

### Celery with Redis or RabbitMQ

Not selected for the initial platform because it would introduce additional infrastructure, delivery semantics, result tracking, and operational concerns before the worker requirements are concretely defined.

Celery may remain appropriate for systems that need its ecosystem and established task execution model.

### Kafka

Rejected as the initial worker foundation because the platform does not yet require high-volume event streaming, partitioned consumption, replay at scale, or broad consumer fan-out.

Kafka would add significant operational and conceptual complexity beyond the current workload.

### Amazon SQS

Not selected for the local-first platform foundation because cloud deployment is intentionally deferred and the initial worker model benefits from transactional coordination with PostgreSQL.

SQS may be appropriate in a future cloud architecture where managed queue isolation and elastic consumption justify different delivery semantics.

### In-process background tasks

Rejected for durable asynchronous work because they are coupled to the API process lifecycle, can be lost during restart, and do not provide reliable ownership or recovery.

Framework background tasks may still be appropriate for non-critical best-effort work that does not require durable execution.

### Database triggers for asynchronous execution

Rejected because triggers should preserve database invariants rather than own external workflow orchestration.

Triggers may write transactional records where appropriate, but application-owned worker behavior remains explicit and testable.

### Dual-write to PostgreSQL and an external queue

Rejected as the initial approach because writing authoritative state and publishing work independently introduces consistency gaps.

If an external broker is introduced later, a transactional outbox or equivalent pattern must preserve reliable publication.
