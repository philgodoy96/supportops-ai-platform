# Ticket Classification Evaluation Dataset

## Purpose

This directory contains versioned synthetic cases for evaluating the structured
ticket-classification contract.

The dataset measures bounded label prediction for:

- category;
- intent;
- urgency;
- sentiment;
- human-review recommendation.

It does not contain customer data, production tickets, secrets, provider
responses, or chain-of-thought.

## Current dataset

```text
dataset_id = ticket-classification-eval
version = 1
file = datasets/ticket-classification-eval-v1.jsonl
schema_version = ticket-classification-v1
```

Version 1 contains 24 synthetic cases covering:

- every supported category;
- every supported intent;
- every urgency level;
- every sentiment value;
- both human-review outcomes;
- ambiguous support requests;
- credential exposure and privacy-sensitive requests;
- prompt-injection text inside untrusted ticket content;
- emotional-language versus operational-impact distinctions.

## Record contract

Each JSONL line contains one independent case:

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
dataset version.

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

## Evaluation boundary

The expected object intentionally excludes summary text. Multiple summaries can
be factually equivalent, so exact string comparison would measure wording rather
than classification quality.

The initial deterministic evaluator will score:

- full structured-label exact match;
- per-field accuracy;
- human-review precision, recall, and F1;
- failed-case counts;
- token and estimated-cost aggregates when predictions provide them.

Semantic summary evaluation and LLM-as-judge scoring remain separate future
decisions.