# Start Here

This repository is a guided path from a minimal RAG pipeline to a production
system. Learn in sequence the first time; later, open any canonical notebook as
a standalone revision note.

## First learning path

```text
RAG mental model
  -> baseline pipeline
  -> ingestion and chunking
  -> embeddings and indexes
  -> retrieval and reranking
  -> query transformation
  -> grounded generation
  -> evaluation
  -> agentic patterns
  -> production operations
```

The foundation notebooks are the next work package. Until they are added, use
the [learning plan](../docs/RAG_LEARNING_101_PLAN.md) as the curriculum map and
the [tracker](../docs/RAG_LEARNING_TRACKER.md) as the source of truth.

## Before opening a notebook

1. Follow [SETUP.md](../docs/SETUP.md).
2. Read the notebook's 30-second summary and scope.
3. Predict where the baseline will fail before running the technique.
4. Run from top to bottom and inspect the evaluation, not only the final answer.
5. Complete the recall questions without looking at the answers.

## Notebook contract

Every canonical lesson will contain:

- a realistic problem and a clear mental model;
- minimal mechanics before framework convenience;
- a baseline and a controlled comparison;
- evaluation, trade-offs, failure modes, and production notes;
- one practice task, recall questions, sources, and a review log.

The reusable source is
[`templates/rag-technique-template.ipynb`](../templates/rag-technique-template.ipynb).

## Local and online lessons

Prefer the local path when learning the mechanism. A lesson may also include an
optional online/provider section when a hosted embedding or chat model adds
meaningful behavior. Provider-dependent cells must state the required key and
must not run unexpectedly.

Callout - Key idea:
A notebook is complete only when you can explain why the technique improved—or
failed to improve—the measured baseline.
