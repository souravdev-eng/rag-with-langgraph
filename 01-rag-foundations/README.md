# RAG Foundations

Complete these lessons in order:

1. [`01-rag-mental-model.ipynb`](01-rag-mental-model.ipynb) — understand the
   offline indexing path, online query path, grounding boundary, and failure map.
2. [`02-minimal-rag-from-scratch.ipynb`](02-minimal-rag-from-scratch.ipynb) —
   implement retrieval, generation, abstention, and evaluation without a RAG
   framework.
3. [`03-end-to-end-baseline.ipynb`](03-end-to-end-baseline.ipynb) — express the
   same baseline with LangChain interfaces while keeping the experiment
   controlled.

The foundation uses a small fictional Northstar product corpus and golden
question set under `data/`. No API key is required.

Callout - Key idea:
Later techniques must beat or meaningfully extend this baseline on the same
questions before their extra complexity is accepted.
