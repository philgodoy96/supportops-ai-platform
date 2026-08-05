# Grounded Recommendation Evaluation

This directory contains repository-owned evaluation artifacts for grounded support recommendations.

Evaluation remains separate from runtime recommendation generation. Committed fixtures support offline validation and deterministic scoring without PostgreSQL, Qdrant, LangGraph, embeddings, or paid providers. External RAGAS execution is an explicit opt-in operation that evaluates existing predictions and writes generated evidence under `artifacts/`.

## Dataset

The canonical dataset is:

```text
datasets/grounded-recommendations-eval-v1.jsonl
```

Identity:

```text
dataset_id: grounded-recommendations-eval
dataset_version: 1
schema_version: grounded-recommendations-eval-v1
source: synthetic
case_count: 14
sha256: 99a7f6cbdb68feb92d3fde32adc43e20893c9e855f299d35e970d166411662cd
```

Coverage includes:

```text
fully grounded response
partially grounded response
unsupported claim
contradictory claim
missing citation
invalid citation
correct abstention
hallucinated answer under insufficient evidence
cross-workspace evidence
ticket prompt injection
retrieved-document prompt injection
human-review recommendation
correct escalation
incorrect escalation
```

Retrieved context text is embedded directly in the dataset. Offline validation and deterministic scoring do not require PostgreSQL, Qdrant, LangGraph, embeddings, or providers.

The corpus is synthetic and intentionally compact. Results apply within the committed synthetic corpus and are not statistically representative of production traffic.

## Static prediction fixture

Canonical predictions:

```text
predictions/grounded-recommendations-eval-v1.static.jsonl
```

Predictions use the shared `EvaluationPredictionEnvelope`, contain structured recommendation output, omit hidden reasoning and raw provider responses, preserve explicit failure states, and include usage and cost fields when known. They are suitable for deterministic offline scoring.

## Static normalized RAGAS score fixture

Canonical artifact:

```text
ragas-scores/grounded-recommendations-eval-v1.static.jsonl
```

This is a synthetic normalized score fixture for contract, loader, aggregation, and CLI testing. It is not a real OpenAI or RAGAS run, not a canonical external baseline, and does not contain provider or model provenance.

Metrics represented:

```text
faithfulness
answer_relevancy
context_precision
context_recall
```

No-context cases may mark context metrics `not_applicable`. Score artifacts preserve succeeded, failed, and not-applicable outcomes. Offline aggregation requires no network access.

## Human review rubric

Canonical rubric:

```text
rubrics/human-review-rubric-v1.json
```

The rubric defines seven dimensions on a 1–5 scale:

```text
correctness
grounding
actionability
safety
citation quality
abstention quality
human-review appropriateness
```

Policy for version 1:

- all 14 cases are reviewed;
- notes are required for scores at or below 2;
- evidence references are required;
- safety concerns are blocking;
- blocking issues require a second review;
- unresolved blocking disagreement is inconclusive.

This is a lightweight review protocol, not an annotation platform.

## Commands

Offline validation and scoring:

```powershell
uv run supportops-evaluate-grounded-recommendations validate

uv run supportops-evaluate-grounded-recommendations score

uv run supportops-evaluate-grounded-recommendations score `
  --ragas-scores "evals/grounded-recommendations/ragas-scores/grounded-recommendations-eval-v1.static.jsonl"
```

These commands perform no network access, require no secrets, and do not instantiate evaluator models or generate embeddings.

External RAGAS evaluation of existing predictions requires `--allow-external-provider`, credentials from `SUPPORTOPS_EVALUATION_OPENAI_API_KEY` (not an implicit fallback to `SUPPORTOPS_OPENAI_API_KEY`), and an output directory under `artifacts/`. External runs may incur provider cost. Do not place real API keys or real external run results in this directory.

## Hash and immutability policy

After canonical use, dataset, prediction, RAGAS score, and rubric files must not be edited in place. Any semantic change requires a new version and a new pinned content hash.

Generated external artifacts are gitignored and written atomically under paths such as:

```text
artifacts/evaluation/grounded-recommendations/<run-id>/
  manifest.json
  ragas-scores.jsonl
  deterministic-report.json
  ragas-report.json
  failures.jsonl
```

Canonical committed fixtures are never overwritten by failed external runs. The manifest is the provenance authority for external evaluation.

## Scope boundaries

`ragas==0.4.3` is isolated in the `evaluation` dependency group. It is not a runtime dependency. Normal CI may install the group for fake-backed unit tests and offline validation, and never performs paid external evaluation.

A real canonical external baseline is intentionally not committed. Evaluator output never modifies runtime prompts. RAGAS scores are probabilistic evaluation evidence, not absolute truth, and do not replace deterministic safety gates.
