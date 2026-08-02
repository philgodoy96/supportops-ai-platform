# ADR 0010: Separate AgentRun and LangGraph Durability

## Status

Accepted

## Context

SupportOps AI Platform already uses PostgreSQL-backed `AgentRun` records as the durable execution boundary for asynchronous ticket processing.

The existing worker model owns:

- transactional ticket-to-run scheduling;
- PostgreSQL claiming with `FOR UPDATE SKIP LOCKED`;
- attempt history;
- lease ownership;
- lease-token fencing;
- bounded retries;
- execution timeout;
- expired lease recovery;
- final success or failure persistence.

The controlled support workflow introduces internal multi-step orchestration:

- classification;
- repeated model decisions;
- bounded read-only tool execution;
- evidence reconstruction;
- terminal analysis;
- recommendation generation;
- recommendation persistence.

LangGraph provides node routing and PostgreSQL-backed checkpoints that can resume internal progress after process interruption.

The architecture must decide whether LangGraph should replace `AgentRun` as the complete workflow lifecycle owner or operate inside the established worker boundary.

Replacing `AgentRun` ownership would duplicate or displace:

- scheduling semantics;
- lease and attempt records;
- retry policy;
- worker claim coordination;
- safe final outcome persistence;
- existing inspection and operational contracts.

Keeping only `AgentRun` without internal checkpoints would require repeating already completed workflow steps after many process failures and would leave post-commit/pre-resume windows harder to recover safely.

## Decision

SupportOps AI Platform will use two explicit durability boundaries with separate responsibilities.

### AgentRun owns outer execution durability

`AgentRun` remains responsible for:

- workflow identity;
- initial scheduling;
- claim eligibility;
- worker ownership;
- attempt identity;
- lease token and lease expiry;
- execution timeout;
- retry scheduling;
- expired lease recovery;
- attempt outcome;
- final run success or failure.

`ProcessClaimedAgentRun` remains the owner of the outer execution lifecycle.

### LangGraph owns bounded internal orchestration

LangGraph is responsible for:

- deterministic node routing;
- bounded graph progress;
- internal decision and tool loops;
- durable node-boundary checkpoints;
- resuming internal graph progress;
- returning only after the persisted recommendation identity is present.

LangGraph does not update the outer `AgentRun` status.

### Runtime fencing remains outside checkpoint state

Attempt identity and lease credentials are provided through runtime context.

Checkpoint state does not contain:

- lease owner;
- lease token;
- lease expiry;
- execution request identity;
- provider clients;
- database sessions.

Every business write performed by graph nodes remains fenced against the current `AgentRunAttempt`.

### Graph completion is not AgentRun completion

Reaching LangGraph `END` is considered successful only when validated graph state contains a persisted recommendation identity.

After the graph executor returns, the outer processor performs the normal lease-fenced `AgentRun` success transition.

Graph errors are translated into retryable or terminal execution errors for the outer processor.

### Checkpoint identity follows AgentRun identity

The LangGraph thread ID is the `AgentRun` UUID.

The checkpoint namespace is versioned by the controlled workflow and graph contract.

A checkpoint resume must validate workspace, ticket, `AgentRun`, workflow, graph, and state-schema compatibility.

## Consequences

### Positive consequences

- Existing transactional scheduling and worker semantics remain stable.
- LangGraph does not become the source of business lifecycle truth.
- Attempt history and retry policy remain consistent across all workflow versions.
- Lease fencing protects graph-triggered persistence from stale workers.
- Internal workflow progress can resume without repeating every completed node.
- The architecture can distinguish process retry from graph resume.
- Checkpoint state remains free from transient ownership credentials.
- Existing worker operational inspection remains valid.
- The graph can evolve internally without redesigning ticket intake.
- Business records remain queryable independently from checkpoint blobs.

### Trade-offs

- Two durability mechanisms must be understood and tested.
- A run may have both `AgentRunAttempt` history and LangGraph checkpoint history.
- Error translation is required between graph and processor boundaries.
- Graph state compatibility requires explicit versioning.
- Checkpoint retention and business-record retention are separate concerns.
- An outer attempt retry may resume an inner graph checkpoint.
- The worker must own both SQLAlchemy and checkpoint connection pools.
- Operators must not interpret graph `END` alone as the public workflow outcome.

### Required engineering practices

This decision requires:

- exact versioned executor dispatch;
- deterministic checkpoint thread identity;
- checkpoint state ownership validation;
- versioned graph and state schemas;
- runtime-context separation;
- lease-fenced business writes;
- provider and tool calls outside database transactions;
- exact recovery after durable commits;
- typed retryable and terminal graph failures;
- graph completion validation;
- outer processor timeout and final outcome ownership;
- integration tests for checkpoint resume;
- tests proving completed nodes are not repeated after interruption;
- no direct `AgentRun` status mutation from LangGraph nodes.

## Alternatives considered

### Let LangGraph own the complete workflow lifecycle

Rejected because the platform already has a durable worker lifecycle with transactional scheduling, claim coordination, attempts, leases, retries, and public operational state.

Replacing that boundary would either duplicate lifecycle information or require redesigning established worker and API contracts.

### Use AgentRun only and restart the complete workflow after failure

Rejected because multi-step LLM and tool workflows create durable progress that should not be repeated unnecessarily.

A full restart would increase provider cost, duplicate retrieval work, complicate tool idempotency, and make committed-progress recovery weaker.

### Persist attempt and lease data inside graph state

Rejected because leases are transient execution ownership credentials.

Persisting them in checkpoints would create a second ownership representation and could cause resumed state to contain stale fencing information.

### Mark AgentRun complete from the final graph node

Rejected because outer lifecycle ownership belongs to the processor.

The processor must retain one consistent path for success, failure, timeout, retry, and stale-lease handling across workflow implementations.

### Use a separate external workflow service

Deferred because the current modular-monolith worker can provide durable bounded orchestration without introducing another deployed control plane.

A separate workflow service may be reconsidered only when operational scale, isolation, or organizational ownership justifies it.

## Related documentation

- [`../architecture/controlled-support-workflow.md`](../architecture/controlled-support-workflow.md)
- [`../architecture/runtime-topology.md`](../architecture/runtime-topology.md)
- [`../architecture/agent-run-scheduling.md`](../architecture/agent-run-scheduling.md)
- [`0004-use-a-postgresql-backed-worker-model.md`](0004-use-a-postgresql-backed-worker-model.md)