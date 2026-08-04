# Evaluation and Regression Architecture

## Purpose

SupportOps AI Platform treats evaluation as an engineering decision system rather than a single score, hosted dashboard, or automatic deployment mechanism.

The repository owns the artifacts required to reproduce evaluation decisions:

- versioned synthetic datasets;
- versioned split manifests;
- prompt definitions and content hashes;
- prediction artifacts;
- deterministic reports;
- release-gate profiles;
- comparison evidence;
- release decisions.

PostgreSQL remains authoritative for runtime business records. Langfuse remains an optional observability projection. Evaluation artifacts remain repository-owned and reproducible without external observability services.

## Current Scope

The current evaluation foundation provides:

- deterministic classification evaluation;
- immutable ticket-classification dataset version 1;
- versioned development, holdout, and safety-gate allocation;
- explicit prompt-version selection for evaluation execution;
- repository-owned evaluation manifests;
- typed prediction envelopes;
- deterministic canonical serialization and hashing;
- atomic artifact writes;
- structured-output validity metrics;
- urgency and human-review safety recall;
- latency and token aggregates;
- standalone classification release-gate evaluation.

The following capabilities remain outside the current foundation and are introduced in later evaluation milestones:

- deterministic retrieval regression;
- controlled-agent regression;
- approval-workflow regression;
- grounded recommendation evaluation;
- RAGAS integration;
- paired prompt comparison;
- prompt version 2;
- prompt promotion decisions.

## Evaluation Ownership

Git-versioned project artifacts are authoritative for:

- datasets and dataset versions;
- split allocation;
- prompt IDs, versions, and hashes;
- schemas;
- evaluation manifests;
- deterministic gate profiles;
- canonical prediction and report evidence selected for the repository;
- promotion, rejection, or inconclusive decisions.

Runtime business records remain owned by PostgreSQL. Qdrant remains a rebuildable retrieval projection. Langfuse remains optional and non-authoritative.

## Dataset Immutability

The committed ticket-classification dataset remains immutable:

```text
evals/ticket-classification/datasets/ticket-classification-eval-v1.jsonl
```

Its deterministic content hash is:

```text
a42445dff9ded6c5d7f73c3f2704cc065a445c06ebb1a1a4ad36fa46dcce984b
```

The dataset contains synthetic cases only.

Split allocation is stored in a separate versioned sidecar:

```text
evals/ticket-classification/splits/ticket-classification-eval-v1-splits-v1.json
```

This keeps behavioral content separate from experimental procedure:

```text
dataset version
→ cases and expected behavior

split manifest version
→ development, holdout, and safety-gate allocation
```

A behavioral change requires a new dataset version. A procedural split change requires a new split-manifest version.

## Development, Holdout, and Safety Gates

The first classification split contains:

```text
development: 12
holdout: 8
safety_gate: 4
```

Development cases may guide failure analysis and prompt-change hypotheses.

Holdout cases remain reserved for final paired evaluation. Their case-level outcomes must not guide prompt drafting.

Safety-gate cases cover critical urgency, credential exposure, privacy-sensitive behavior, prompt injection, and mandatory human review. Safety failures cannot be compensated for by aggregate quality improvements.

## Evaluation Manifest

The evaluation manifest binds one execution to explicit provenance:

- evaluation identity and version;
- dataset ID, version, and hash;
- split-manifest identity, version, and hash;
- selected split;
- system provider and model;
- workflow name and version when applicable;
- prompt ID, version, and content hash;
- schema version;
- embedding and retrieval configuration when applicable;
- evaluator provider and model when applicable;
- RAGAS version when applicable;
- pricing catalog version;
- capture timestamp;
- Git commit;
- prediction hash;
- execution status.

Applicable provenance groups are validated as all-or-none. A split cannot be declared without complete split-manifest provenance.

Execution status is explicit:

```text
complete
incomplete
failed
```

Only complete executions may become canonical evidence.

## Prediction Contracts

Evaluation predictions use typed domain payloads inside a shared envelope.

The envelope records:

- case ID;
- execution status;
- typed payload or error code;
- latency;
- input, output, and embedding tokens;
- estimated cost;
- optional trace identity.

Successful predictions require a payload and cannot contain an error code. Failed predictions require an error code and cannot contain a successful payload.

Unknown usage or cost remains unknown. Missing values are not converted to zero.

## Canonical Serialization and Hashing

Evaluation artifacts use deterministic JSON serialization:

- object keys are sorted;
- UTF-8 encoding is used;
- null values remain explicit;
- Decimal values retain exact decimal representation;
- non-finite numeric values are rejected;
- unsupported values fail serialization.

SHA-256 content hashes bind manifests, predictions, reports, and gate results to their exact canonical content.

Hashes are evidence identifiers, not security signatures.

## Atomic Artifact Writes

Canonical artifact writes follow a staged replacement sequence:

```text
write temporary file
→ flush and synchronize
→ calculate hash
→ validate expected hash
→ atomically replace destination
```

A hash mismatch or incomplete write cannot overwrite an existing canonical artifact.

Temporary and local execution artifacts remain under ignored artifact directories until explicitly selected as repository evidence.

## Classification Metrics

The classification evaluator preserves exact-match and confusion-matrix metrics while adding safety, validity, latency, and usage evidence.

Current metrics include:

- structured label exact match;
- category accuracy;
- intent accuracy;
- urgency accuracy;
- sentiment accuracy;
- human-review accuracy;
- human-review precision, recall, and F1;
- structured-output validity;
- invalid-output rate;
- high urgency recall;
- critical urgency recall;
- high-risk human-review recall;
- average latency;
- average input tokens;
- average output tokens;
- average total tokens;
- known and unknown usage counts;
- known estimated cost;
- unknown pricing count.

A structurally valid prediction may still be behaviorally incorrect. Structural validity and label correctness remain separate.

Critical urgency recall accepts only an exact critical prediction for an expected critical case.

High-risk human-review recall uses explicit dataset expectations and an application-owned set of high-risk tags. It does not infer review requirements from sentiment or category alone.

## Explicit Prompt Selection

The classification evaluation command selects prompt version explicitly:

```powershell
supportops-evaluate-classification run `
  --prompt-version 1 `
  --provider mock
```

The command remains domain-specific, so the stable prompt ID remains implicit:

```text
ticket-classification
```

There is no implicit latest version.

An unsupported version fails without provider execution, fallback, or artifact replacement.

Runtime classification remains independently pinned to its approved prompt version. Evaluation selection does not change the production default.

## Classification Release Gates

The initial classification gate profile is:

```text
profile_id: ticket-classification-release-gates
profile_version: 1
```

Gate categories are:

```text
safety
quality
reliability
efficiency
```

Individual outcomes are:

```text
passed
failed
not_applicable
```

Aggregate standalone status is:

```text
passed
failed
incomplete
```

Semantics:

```text
failed
→ at least one blocking gate failed

incomplete
→ no blocking gate failed, but at least one blocking gate is not applicable

passed
→ every blocking gate passed
```

Standalone safety and reliability gates evaluate:

- structured-output validity;
- critical urgency recall;
- high-risk human-review recall;
- prediction artifact coverage;
- successful deterministic report generation.

Quality and efficiency non-regression gates require paired baseline evidence. They remain `not_applicable` for a standalone report.

A perfect standalone report is therefore intentionally `incomplete`. Standalone evidence cannot authorize prompt promotion.

## Evaluation Versus Observability

Observability and evaluation have different failure semantics.

```text
observability failure
→ fail open

evaluation failure
→ remain visible
```

Langfuse telemetry must not change evaluation outcomes. Evaluation scoring, report generation, and release gates remain reproducible with the no-op observability adapter.

The existence of an evaluator observation type does not make Langfuse the source of truth for evaluation.

## CI Boundary

Normal CI may execute:

- dataset validation;
- split-manifest validation;
- deterministic evaluator tests;
- static prediction scoring;
- release-gate logic;
- prompt registry and hash tests;
- report and artifact hashing tests;
- atomic-write tests.

Normal CI must not execute:

- paid provider calls;
- live OpenAI evaluation;
- real RAGAS judges;
- paid embeddings;
- Langfuse external calls;
- human review;
- prompt promotion.

External provider execution remains explicit and requires acknowledgement.

## Limitations

The current classification corpus is small, synthetic, and curated. It supports regression detection within the project boundary, but it is not statistically representative of production traffic.

Evaluation reports must use language such as:

```text
Within the versioned synthetic evaluation corpus...
```

They must not claim global or statistical superiority.

## Intentionally Deferred Capabilities

The architecture intentionally defers:

- production feedback ingestion;
- scheduled evaluation;
- online evaluation;
- evaluation history databases;
- automatic prompt optimization;
- automatic prompt promotion;
- automatic deployment;
- Langfuse datasets and experiments;
- production A/B testing;
- large-scale benchmark construction.

These capabilities require additional product, privacy, operational, and statistical design beyond the repository-owned regression foundation.