# ADR 0014: Use Repository-Owned Evaluation and Evidence-Driven Prompt Promotion

## Status

Accepted

## Context

SupportOps AI Platform contains probabilistic AI behavior across ticket classification, semantic retrieval, controlled workflow execution, recommendations, and approval-sensitive actions.

Tests alone cannot characterize every model behavior. Model-based metrics alone cannot prove safety or correctness. External observability platforms provide operational inspection but do not provide repository-owned reproducibility.

The project requires an evaluation discipline that can:

- preserve versioned datasets and prompts;
- reproduce deterministic regression results;
- isolate development and holdout evidence;
- expose safety failures independently from aggregate quality;
- account for model, prompt, provider, token, latency, and cost provenance;
- compare prompt versions on the same frozen configuration;
- produce an explicit engineering release decision.

The system must avoid turning evaluation into a hosted dashboard dependency, an opaque aggregate score, or an automatic deployment controller.

## Decision

Evaluation datasets, split manifests, prompt definitions, prompt hashes, evaluator configuration, release gates, canonical evidence, and release decisions remain Git-owned project artifacts.

Deterministic evaluation and probabilistic model-based evaluation remain separate.

Deterministic evaluation is suitable for normal CI and covers objective behavior such as:

- schema validity;
- exact labels;
- urgency recall;
- human-review recall;
- tool and approval invariants;
- citation membership;
- workspace isolation;
- idempotency;
- cost arithmetic;
- artifact integrity.

Probabilistic evaluation is manually triggered, provider-dependent, and interpreted alongside deterministic evidence. RAGAS is one evaluation component for grounded recommendation quality. It is not the complete evaluation framework, runtime application boundary, prompt authority, or business-state authority.

Paid execution requires explicit acknowledgement. Normal CI remains deterministic and requires no provider credentials.

Release gates are defined and frozen before candidate prompt execution. Safety gates take precedence over quality and efficiency improvements.

Development cases may guide prompt changes. Holdout cases remain reserved for final comparison and must not guide prompt drafting. Safety-gate cases remain fixed blocking invariants.

Prompt version 1 remains immutable. Prompt version 2 may be created only after:

1. the baseline configuration is frozen;
2. the version 1 baseline is complete;
3. development failures are analyzed;
4. a specific change hypothesis is approved;
5. release gates are frozen.

Prompt comparison is paired by case using the same dataset, split, provider, model, schema, pricing catalog, and evaluator configuration.

The final decision uses exactly:

```text
promoted
rejected
inconclusive
```

Promotion is not automatic. Rejection and inconclusive outcomes are valid engineering results.

Langfuse remains an optional observability projection. It is not the evaluation authority and is not required to reproduce evaluation decisions.

Automatic prompt generation, automatic optimization, automatic promotion, production feedback ingestion, scheduled evaluation, and production A/B testing are intentionally deferred.

## Implementation outcome

Immutable `ticket-classification` prompt version 2 was created from documented development failure analysis and registered as an explicit offline evaluation candidate. Prompt versions 1 and 2 were compared through committed static paired prediction fixtures. The comparison exercises deterministic comparison, safety-gate, provenance, and decision semantics.

The committed decision outcome for prompt version 2 is:

```text
outcome: inconclusive
run_status: incomplete
approved_for_runtime_adoption: false
separate_runtime_adoption_required: true
runtime_prompt_version: 1
candidate_prompt_version: 2
```

Prompt version 1 remains the runtime default. No runtime configuration was changed. No provider-backed model superiority is claimed. Static evidence cannot authorize runtime adoption. Canonical provider-backed evidence and human review remain required before any future adoption.

The intended adoption sequence remains:

```text
evaluation evidence
→ explicit decision artifact
→ separate runtime adoption decision
→ production rollout
```

The repository completed the first two stages for prompt version 2 and intentionally did not execute the final two stages. Rejection and inconclusive outcomes are valid release-governance results, not incomplete engineering implementation.

Committed static evidence for this outcome includes:

- prompt definition: [`src/supportops/ai/prompts/ticket_classification_v2.py`](../../src/supportops/ai/prompts/ticket_classification_v2.py)
- paired predictions: [`evals/ticket-classification/predictions/ticket-classification-eval-v1.prompt-v1.static.jsonl`](../../evals/ticket-classification/predictions/ticket-classification-eval-v1.prompt-v1.static.jsonl) and [`evals/ticket-classification/predictions/ticket-classification-eval-v1.prompt-v2.static.jsonl`](../../evals/ticket-classification/predictions/ticket-classification-eval-v1.prompt-v2.static.jsonl)
- comparison artifact: [`evals/ticket-classification/comparisons/ticket-classification-prompt-v1-v2.static.json`](../../evals/ticket-classification/comparisons/ticket-classification-prompt-v1-v2.static.json)
- decision artifact: [`evals/ticket-classification/decisions/ticket-classification-prompt-v2-decision.static.json`](../../evals/ticket-classification/decisions/ticket-classification-prompt-v2-decision.static.json)
- classification evaluation documentation: [`docs/architecture/classification-evaluation.md`](../architecture/classification-evaluation.md)

## Consequences

### Positive

- Evaluation decisions remain reviewable in Git.
- Dataset and prompt provenance is explicit.
- Normal CI remains deterministic and cost-controlled.
- Safety regressions cannot be averaged away.
- Prompt changes require documented evidence.
- Holdout discipline reduces evaluation leakage.
- Model-based evaluation can evolve without contaminating runtime architecture.
- Langfuse outages cannot invalidate repository-owned evaluation evidence.
- Rejected prompt candidates remain reproducible and defensible.

### Negative

- Canonical evidence requires deliberate artifact selection and review.
- Small synthetic datasets provide limited statistical confidence.
- External evaluation execution remains a manual operational step.
- Holdout discipline requires procedural enforcement.
- Adding RAGAS may increase evaluation-only dependency and lockfile complexity.
- Multiple artifact types increase documentation and review responsibilities.

### Risks

- Engineers may unintentionally inspect holdout failures during prompt drafting.
- Dataset expectations may encode project-specific assumptions.
- Evaluator-model drift may change probabilistic scores.
- Small samples may make quality differences inconclusive.
- Canonical artifacts may become stale if provenance checks are bypassed.

These risks are mitigated through versioned split manifests, deterministic hashes, explicit evaluator provenance, paired comparison, frozen gates, and transparent inconclusive decisions.

## Alternatives considered

### Use Langfuse datasets and experiments as the evaluation authority

Rejected because it would move authoritative datasets, evidence, and decisions outside the repository. Langfuse remains useful for observability but is not required for reproducibility.

### Use RAGAS as the complete evaluation framework

Rejected because RAGAS does not replace deterministic schema, safety, isolation, idempotency, tool, approval, or cost checks.

### Use one weighted aggregate score

Rejected because aggregate scores can conceal blocking safety regressions and make trade-offs difficult to defend.

### Run live evaluations in normal CI

Rejected because live evaluation is paid, probabilistic, credential-dependent, and vulnerable to provider variability.

### Tune prompts against every available case

Rejected because it creates evaluation leakage and removes independent evidence for final promotion.

### Promote prompts automatically when thresholds pass

Rejected because release decisions require combined deterministic, probabilistic, human-review, cost, latency, and case-level evidence.

### Store evaluation history in PostgreSQL

Deferred because the current project boundary prioritizes Git-owned reproducibility. A history database becomes appropriate only when scheduled or production evaluation requires queryable operational history.

### Implement automatic prompt optimization

Deferred because the project focuses on controlled, evidence-driven engineering judgment rather than autonomous prompt search.