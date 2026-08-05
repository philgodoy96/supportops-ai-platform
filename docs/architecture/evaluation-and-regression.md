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
- standalone classification release-gate evaluation;
- deterministic semantic-retrieval regression;
- deterministic controlled-support regression;
- deterministic human-approval regression;
- repository-level deterministic regression scoring through `supportops-evaluate-regression score`;
- grounded recommendation evaluation with a committed synthetic dataset and static predictions;
- deterministic complementary grounded recommendation metrics;
- normalized static RAGAS score artifacts and offline RAGAS aggregation;
- an evaluation-only RAGAS dependency boundary and explicit external RAGAS runner;
- a committed human qualitative review rubric for grounded recommendations.

Within the committed synthetic regression corpus, the multi-domain regression command scores committed static fixtures. It does not execute live embeddings, Qdrant, LangGraph, providers, PostgreSQL mutations, approval services, or Langfuse.

Grounded recommendation offline validation and scoring likewise consume committed fixtures without network access. External RAGAS execution is opt-in, evaluates existing predictions only, and writes generated evidence under `artifacts/`.

The following capabilities remain outside the current foundation and are introduced in later evaluation milestones:

- a real canonical external RAGAS baseline;
- paired prompt comparison;
- classification prompt version 2;
- prompt promotion, rejection, or inconclusive decisions.

## Evaluation Ownership

Git-versioned project artifacts are authoritative for:

- datasets and dataset versions;
- split allocation;
- prompt IDs, versions, and hashes;
- schemas;
- evaluation manifests;
- deterministic gate profiles;
- canonical prediction and report evidence selected for the repository;
- normalized static RAGAS score fixtures used for offline aggregation tests;
- human review rubrics;
- promotion, rejection, or inconclusive decisions when those decisions exist.

Runtime business records remain owned by PostgreSQL. Qdrant remains a rebuildable retrieval projection. Langfuse remains optional and non-authoritative. Generated external evaluation artifacts under `artifacts/` are run-specific evidence; they are not repository authority until explicitly selected and committed as canonical fixtures.

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

Committed multi-domain regression datasets are also immutable synthetic corpora:

```text
evals/semantic-retrieval/datasets/semantic-retrieval-eval-v1.jsonl
evals/controlled-support/datasets/controlled-support-eval-v1.jsonl
evals/human-approval/datasets/human-approval-eval-v1.jsonl
```

Case counts within the committed synthetic regression corpus:

```text
semantic retrieval: 10
controlled support: 14
human approval: 14
```

Committed static prediction fixtures used by repository regression scoring:

```text
evals/semantic-retrieval/predictions/semantic-retrieval-eval-v1.static.jsonl
evals/controlled-support/predictions/controlled-support-eval-v1.static.jsonl
evals/human-approval/predictions/human-approval-eval-v1.static.jsonl
```

These fixtures are typed prediction envelopes. Scoring consumes the fixtures as evidence and does not regenerate them through runtime services.

Grounded recommendation evaluation owns a separate committed corpus under:

```text
evals/grounded-recommendations/
```

That domain is documented in the grounded recommendation section below and is not part of `supportops-evaluate-regression score`.

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

Optional repository-regression output follows the same atomic write boundary. Scoring failures do not replace an existing output artifact.

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

## Semantic Retrieval Regression

Deterministic semantic-retrieval evaluation scores committed static retrieval predictions against immutable dataset version 1.

Within the committed synthetic regression corpus:

- dataset path: `evals/semantic-retrieval/datasets/semantic-retrieval-eval-v1.jsonl`;
- static prediction fixture: `evals/semantic-retrieval/predictions/semantic-retrieval-eval-v1.static.jsonl`;
- case count: 10.

Metric family:

- document hit rate at k;
- chunk hit rate at k;
- mean reciprocal rank;
- recall at k;
- no-result accuracy;
- workspace isolation rate;
- citation resolution rate;
- average latency;
- average query embedding tokens;
- estimated query cost.

Scoring performs no live embeddings and no Qdrant execution. Duplicate retrieved chunks count once. Cosine similarity score is ranking evidence only and is not treated as calibrated confidence.

Release-gate profile identity:

```text
semantic-retrieval-release-gates / 1
```

## Controlled-Support Regression

Deterministic controlled-support evaluation scores committed static execution-trace fixtures against immutable dataset version 1.

Within the committed synthetic regression corpus:

- dataset path: `evals/controlled-support/datasets/controlled-support-eval-v1.jsonl`;
- static execution-trace fixture: `evals/controlled-support/predictions/controlled-support-eval-v1.static.jsonl`;
- case count: 14.

Metric family:

- expected outcome accuracy;
- required tool call rate;
- forbidden tool call rate;
- exact tool-sequence acceptance;
- repeated tool acceptance rate;
- step-limit behavior;
- recommended-action accuracy;
- human-review recommendation accuracy;
- citation validity;
- grounded abstention;
- workspace isolation;
- successful completion;
- tool-call, LLM invocation, latency, token, and cost aggregates.

Scoring performs no live LangGraph, tools, providers, PostgreSQL, Qdrant, or Langfuse execution. Expected failures remain explicit and require exact error codes.

Release-gate profile identity:

```text
controlled-support-release-gates / 1
```

## Human-Approval Regression

Deterministic human-approval evaluation scores committed static approval-outcome fixtures against immutable dataset version 1.

Within the committed synthetic regression corpus:

- dataset path: `evals/human-approval/datasets/human-approval-eval-v1.jsonl`;
- static approval outcome fixture: `evals/human-approval/predictions/human-approval-eval-v1.static.jsonl`;
- case count: 14.

Metric family:

- approval-required accuracy;
- unauthorized sensitive execution rate;
- approved execution success;
- rejected non-execution;
- expired non-execution;
- approval decision idempotency;
- resume success;
- sensitive-action idempotency;
- checkpoint match;
- grant match;
- retry-budget preservation;
- duplicate escalation prevention;
- finalization;
- latency, token, and cost aggregates.

Scoring performs no live approval services, checkpoint mutation, API execution, or sensitive tool execution.

Release-gate profile identity:

```text
human-approval-release-gates / 1
```

## Domain Release Gates

Safety and reliability gates are deterministic and blocking for each committed domain profile.

Quality and efficiency non-regression gates require paired baseline evidence. Standalone committed fixtures therefore produce domain status `incomplete` when safety and reliability gates pass and paired quality or efficiency evidence is absent.

`incomplete` is valid deterministic evidence. Blocking gate failures produce `failed`.

## Repository Regression Command

Repository-level deterministic regression scoring uses:

```text
supportops-evaluate-regression score
```

Default committed domains, in deterministic order:

```text
semantic-retrieval
controlled-support
human-approval
```

Classification remains optional when static classification evidence is not supplied. Omitted optional classification evidence is recorded as not provided and does not fail the repository result.

Repository aggregate statuses:

```text
passed
failed
incomplete
```

Standalone committed fixtures without paired quality and efficiency baselines therefore produce repository aggregate status `incomplete`. That status is valid deterministic evidence and exits zero.

Exit semantics:

```text
0 → valid passed or incomplete evidence
1 → blocking gate failure
2 → usage error
3 → artifact validation or scoring failure
```

The command performs no network calls, requires no secrets or runtime services, and writes optional output atomically. Normal CI explicitly runs `supportops-evaluate-regression score`.

## Grounded Recommendation Evaluation

Grounded recommendation evaluation measures drafted recommendation quality against a committed synthetic corpus without changing runtime workflow execution.

The evaluation plane separates four concerns:

```text
runtime recommendation generation
→ worker-owned controlled support drafting and persistence

deterministic offline evaluation
→ committed dataset and static predictions scored without network access

external model-based evaluation
→ explicit RAGAS runs over existing predictions

human qualitative review
→ committed rubric and lightweight review protocol
```

### Evaluation-only dependency

`ragas==0.4.3` is isolated in the `evaluation` dependency group. It is not a runtime dependency of the API or worker. Normal CI may install the group for fake-backed unit tests and offline validation. Normal CI never performs paid external evaluation. Dependency resolution beyond the committed lockfile is not claimed as independently stable.

### Dataset

Canonical dataset:

```text
evals/grounded-recommendations/datasets/grounded-recommendations-eval-v1.jsonl
```

Identity:

```text
dataset_id: grounded-recommendations-eval
dataset_version: 1
schema_version: grounded-recommendations-eval-v1
source: synthetic
case_count: 14
```

Coverage includes fully grounded and partially grounded responses, unsupported and contradictory claims, missing and invalid citations, correct abstention, hallucinated answers under insufficient evidence, cross-workspace evidence, ticket and retrieved-document prompt injection, human-review recommendations, and correct and incorrect escalation.

Retrieved context content is embedded in the dataset. Offline validation does not require PostgreSQL, Qdrant, LangGraph, embeddings, or providers. Results apply within the committed synthetic corpus and are not statistically representative of production traffic.

### Static prediction fixture

Canonical predictions:

```text
evals/grounded-recommendations/predictions/grounded-recommendations-eval-v1.static.jsonl
```

Predictions use the shared `EvaluationPredictionEnvelope`, contain structured recommendation output, contain no hidden reasoning and no raw provider response, preserve explicit failure states, include usage and cost fields when known, and are suitable for deterministic offline scoring.

### Deterministic complementary metrics

Deterministic scoring reports:

```text
recommended_action_accuracy
human_review_accuracy
evidence_sufficiency_accuracy
citation_identity_accuracy
workspace_isolation_rate
grounded_abstention_accuracy
prediction_coverage
average latency
average total tokens
estimated cost
```

Workspace isolation, citation identity, and prediction coverage are safety and reliability evidence. Unknown usage values remain unknown. Failed and missing predictions are not silently dropped. Deterministic scoring is independent of RAGAS.

### Static normalized RAGAS score fixture

Canonical artifact:

```text
evals/grounded-recommendations/ragas-scores/grounded-recommendations-eval-v1.static.jsonl
```

This synthetic normalized score fixture exists for contract, loader, aggregation, and CLI testing. It is not a real OpenAI or RAGAS run, not a canonical external baseline, and does not contain provider or model provenance.

Metrics represented:

```text
faithfulness
answer_relevancy
context_precision
context_recall
```

No-context cases may mark context metrics `not_applicable`. Score artifacts preserve succeeded, failed, and not-applicable outcomes. Offline aggregation requires no network access.

### Adapter isolation

RAGAS execution is isolated behind an application-owned adapter boundary. Fake-backed unit tests exercise normalization and failure visibility without provider calls. The external OpenAI adapter is constructed only for acknowledged `run` executions and is not part of normal CI.

### Command surface

Command:

```text
supportops-evaluate-grounded-recommendations
```

Subcommands:

```text
validate
score
run
```

#### validate

```powershell
uv run supportops-evaluate-grounded-recommendations validate
```

Behavior:

- no network;
- validates the dataset and optional prediction or RAGAS artifacts;
- validates hashes and schema;
- no secrets or services required.

#### score

```powershell
uv run supportops-evaluate-grounded-recommendations score
```

Optional offline RAGAS aggregation:

```powershell
uv run supportops-evaluate-grounded-recommendations score `
  --ragas-scores "evals/grounded-recommendations/ragas-scores/grounded-recommendations-eval-v1.static.jsonl"
```

Behavior:

- no network;
- builds the deterministic complementary report;
- optionally aggregates an existing normalized RAGAS score artifact;
- does not instantiate evaluator models;
- does not generate embeddings.

#### run

External behavior:

- evaluates existing recommendation predictions;
- does not generate recommendations;
- does not execute LangGraph or tools;
- requires `--allow-external-provider`;
- reads evaluator credentials from `SUPPORTOPS_EVALUATION_OPENAI_API_KEY`;
- does not implicitly fall back to `SUPPORTOPS_OPENAI_API_KEY`;
- records system model and evaluator model separately;
- allows same-model evaluation with a warning;
- writes generated artifacts under an explicit `artifacts/` directory;
- rejects committed `evals/grounded-recommendations` output paths;
- may incur provider cost;
- keeps provider and evaluator failures visible.

External runs must not place real API keys or real external results into committed fixtures.

### Exit codes

```text
0: successful validate/score or complete/incomplete external run with valid artifacts
2: CLI usage error
3: artifact validation, composition, or scoring failure
4: missing external-provider acknowledgement
```

### Generated artifact layout

External runs write:

```text
artifacts/evaluation/grounded-recommendations/<run-id>/
  manifest.json
  ragas-scores.jsonl
  deterministic-report.json
  ragas-report.json
  failures.jsonl
```

Generated artifacts are gitignored. Writes are atomic. Canonical committed fixtures are never overwritten by failed external runs. The manifest is the provenance authority for external evaluation.

### Manifest provenance

The manifest binds:

```text
dataset identity and hash
prediction hash
system provider/model
evaluator provider/model
evaluator embedding model
workflow name/version
prompt id/version/hash
RAGAS version
Git commit
run status
```

Evaluator output never modifies runtime prompts.

### Human review rubric

Canonical rubric:

```text
evals/grounded-recommendations/rubrics/human-review-rubric-v1.json
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

For version 1, all 14 cases are reviewed, notes are required for scores at or below 2, evidence references are required, safety concerns are blocking, blocking issues require a second review, and unresolved blocking disagreement is inconclusive. This is a lightweight review protocol, not an annotation platform.

### RAGAS limitations

RAGAS scores are probabilistic evaluation evidence. They are not absolute truth, not statistically representative of production traffic, and do not replace deterministic safety gates. Model choice can affect scores. Same-model evaluation can introduce bias. External provider behavior can drift. Failures and not-applicable metrics must remain visible. No automatic release or prompt promotion decision is made.

### Domain artifact documentation

Committed grounded recommendation artifacts are summarized in [`../../evals/grounded-recommendations/README.md`](../../evals/grounded-recommendations/README.md).

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
- atomic-write tests;
- `supportops-evaluate-regression score` against committed multi-domain fixtures;
- grounded recommendation unit tests with fake-backed adapters;
- `supportops-evaluate-grounded-recommendations validate`;
- `supportops-evaluate-grounded-recommendations score`;
- `supportops-evaluate-grounded-recommendations score` with the committed static RAGAS score fixture.

Normal CI must not execute:

- paid provider calls;
- live OpenAI evaluation;
- real RAGAS judges;
- paid embeddings;
- Langfuse external calls;
- human review sessions;
- prompt promotion;
- grounded recommendation `run` with `--allow-external-provider`.

Normal CI grounded evaluation uses no API keys and no acknowledgement flag. Fake-backed adapter tests only are permitted. No external smoke test runs in GitHub Actions. External provider execution remains explicit and requires acknowledgement.

## Limitations

The current classification corpus is small, synthetic, and curated. It supports regression detection within the project boundary, but it is not statistically representative of production traffic.

The multi-domain retrieval, controlled-support, and human-approval corpora are likewise synthetic and curated. Within the committed synthetic regression corpus they support deterministic regression detection. They are not representative of production traffic.

The grounded recommendation corpus is synthetic and curated. Deterministic complementary metrics and offline RAGAS aggregation apply within that committed corpus. Static RAGAS score fixtures are synthetic contract evidence, not production-quality baselines. External RAGAS scores remain probabilistic and model-dependent.

Evaluation reports must use language such as:

```text
Within the versioned synthetic evaluation corpus...
```

They must not claim global or statistical superiority.

## Intentionally Deferred Capabilities

The architecture intentionally defers:

- a real canonical external RAGAS baseline;
- classification prompt version 2;
- paired v1-versus-v2 comparison;
- prompt promotion, rejection, or inconclusive decisions;
- automatic prompt optimization;
- production feedback ingestion;
- scheduled evaluation;
- online evaluation;
- evaluation database;
- Langfuse datasets or experiments;
- production A/B testing;
- automatic deployment;
- a full annotation platform;
- large-scale benchmark construction.

These capabilities require additional product, privacy, operational, and statistical design beyond the repository-owned evaluation foundation.
