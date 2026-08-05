# Grounded Recommendation Evaluation

This directory contains repository-owned evaluation artifacts for grounded support recommendations.

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

The corpus exercises grounding, unsupported and contradictory claims, citation integrity, abstention, workspace isolation, prompt injection, human review, and escalation decisions.

Retrieved context text is embedded directly in the dataset so validation and deterministic scoring do not require PostgreSQL, Qdrant, or embedding execution.

## Scope

The dataset is synthetic and intentionally compact. Results apply within the committed regression corpus and are not presented as statistically representative of production traffic.

The dataset does not execute providers or RAGAS. Prediction fixtures, deterministic complementary metrics, and external RAGAS evaluation are introduced separately.

## Immutability

After canonical use, this file must not be edited in place. Any semantic change requires a new dataset version and a new pinned content hash.