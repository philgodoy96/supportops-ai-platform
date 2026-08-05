# Ticket Classification Evaluation Artifacts

## Purpose

This directory owns repository-authoritative artifacts for structured ticket
classification evaluation and prompt iteration.

The corpus measures bounded label prediction for:

- category;
- intent;
- urgency;
- sentiment;
- human-review recommendation.

Artifacts contain no customer data, production tickets, secrets, provider
responses, or chain-of-thought.

## Directory layout

```text
evals/ticket-classification/
├── datasets/
├── splits/
├── analyses/
├── predictions/
├── comparisons/
└── decisions/
```

| Directory | Role |
| --- | --- |
| `datasets/` | Immutable synthetic evaluation cases and expected labels |
| `splits/` | Versioned development, holdout, and safety-gate allocation |
| `analyses/` | Development-only failure analysis used for prompt drafting |
| `predictions/` | Committed static prediction fixtures for contract and CI evidence |
| `comparisons/` | Committed paired v1-versus-v2 comparison artifact |
| `decisions/` | Committed decision template / static inconclusive decision |

Generated provider evidence, regenerated comparisons, manifests, and rebuilt
decisions belong under gitignored `artifacts/`. They are not repository
authority until explicitly selected and committed as canonical fixtures.

## Dataset and split ownership

```text
dataset_id = ticket-classification-eval
version = 1
file = datasets/ticket-classification-eval-v1.jsonl
schema_version = ticket-classification-v1
```

Version 1 contains 24 synthetic cases covering every supported category,
intent, urgency, and sentiment value; both human-review outcomes; ambiguous
requests; credential exposure and privacy-sensitive requests; prompt-injection
text inside untrusted ticket content; and emotional-language versus operational-
impact distinctions.

Split allocation is owned separately:

```text
file = splits/ticket-classification-eval-v1-splits-v1.json
split_manifest_id = ticket-classification-eval-splits
split_manifest_version = 1

development: 12
holdout: 8
safety_gate: 4
```

A behavioral change requires a new dataset version. A procedural split change
requires a new split-manifest version.

## Development-only analysis

```text
analyses/classification-prompt-v1-failure-analysis.json
```

The committed analysis is scoped to the development split. Holdout and
safety-gate cases are not used as prompt-drafting inputs.

Evidence kinds:

```text
provider_observation
static_fixture
dataset_design_hypothesis
```

The committed analysis contains dataset-design hypotheses only. Provider
observation count is zero. Static-fixture observation count is zero. It does
not claim that prompt version 1 was empirically proven to fail on a provider.

Prompt-revision constraints preserve the `ticket-classification-v1` structured
output schema, forbid chain-of-thought, forbid encoding case IDs or exact
dataset wording into prompt version 2, and keep runtime version 1 pinned until
separate adoption approval.

## Static prediction fixtures

```text
predictions/ticket-classification-eval-v1.prompt-v1.static.jsonl
predictions/ticket-classification-eval-v1.prompt-v2.static.jsonl
```

Evidence kind: `static_fixture`.

Purpose:

- contract testing;
- deterministic comparison testing;
- release-gate testing;
- governance and decision-path validation;
- no-network CI.

These fixtures are not OpenAI outputs, provider-backed quality evidence,
production baselines, or statistical claims about real model performance.
Deliberate baseline errors exist only to prove comparison and decision
semantics. They are not observed model failures.

## Comparison artifact

```text
comparisons/ticket-classification-prompt-v1-v2.static.json
```

The committed static comparison demonstrates paired comparison behavior. For
this artifact:

```text
gate status: incomplete
blocking failures: 0
not-applicable gates: mean token increase, mean cost increase
```

The static fixture demonstrates comparison behavior. It is not a
production-quality pass.

## Inconclusive decision artifact

```text
decisions/ticket-classification-prompt-v2-decision.static.json
```

Committed decision fields:

```text
outcome: inconclusive
run status: incomplete
approved_for_runtime_adoption: false
separate_runtime_adoption_required: true
```

Static evidence cannot approve runtime adoption. Prompt version 2 is neither
approved nor deployed by this artifact.

## Record contract

Each dataset JSONL line contains one independent case:

```json
{
  "case_id": "billing-duplicate-charge-005",
  "tags": ["billing", "individual-impact", "negative"],
  "ticket": {
    "subject": "Duplicated invoice charge",
    "description": "The latest invoice contains the same subscription charge twice."
  },
  "expected": {
    "category": "billing",
    "intent": "ask_question",
    "urgency": "normal",
    "sentiment": "negative",
    "requires_human_review": false,
    "schema_version": "ticket-classification-v1"
  }
}
```

Case IDs and tags use lowercase kebab-case. Case IDs are unique within one
dataset version. Expected objects intentionally exclude summary text.

## Versioning policy

A dataset version is immutable after it is used for a published evaluation
result.

Create a new version when:

- expected labels change;
- cases are added, removed, or materially rewritten;
- taxonomy interpretation changes;
- the benchmark scope changes.

Formatting-only JSON changes do not alter the loader's canonical content hash.
Case order remains part of the canonical dataset content.

## Holdout discipline

Development cases may guide failure analysis and prompt-change hypotheses.
Holdout cases remain reserved for final paired evaluation after prompt freeze.
Holdout outcomes must not guide prompt drafting. Safety-gate failures cannot be
compensated for by aggregate quality improvements.

## Runtime adoption separation

Evaluation layers remain distinct from runtime selection:

1. evaluation evidence;
2. prompt decision artifact;
3. runtime prompt selection;
4. production rollout.

Runtime classification remains pinned to prompt version 1. Changing that pin is
intentionally a separate, reviewable repository change.

## Promoting provider evidence

Provider-backed prediction runs write generated evidence under `artifacts/`.
Promoting provider evidence to committed canonical artifacts requires explicit
review, hash binding, and a separate repository change. Generated artifacts are
not automatically authoritative.

## Intentional future boundaries

- provider-backed canonical v1/v2 comparison;
- holdout evaluation after prompt freeze;
- human review of provider evidence;
- separate runtime prompt adoption pull request;
- production rollout monitoring;
- scheduled or online evaluation;
- production feedback ingestion;
- evaluation database storage;
- Langfuse datasets or experiments.

Architecture detail lives in
[`docs/architecture/classification-evaluation.md`](../../docs/architecture/classification-evaluation.md)
and
[`docs/architecture/evaluation-and-regression.md`](../../docs/architecture/evaluation-and-regression.md).
