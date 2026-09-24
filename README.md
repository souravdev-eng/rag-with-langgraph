# Production RAG Learning 101

This repository is being developed into a go-to revision guide for building,
evaluating, and operating production RAG systems. Each notebook should stand on
its own as a clear lesson while also fitting into one end-to-end curriculum.

## Start here

- [Learning plan](docs/RAG_LEARNING_101_PLAN.md)
- [Progress tracker](docs/RAG_LEARNING_TRACKER.md)
- [Target folder structure](docs/TARGET_REPOSITORY_STRUCTURE.md)
- [Setup guide](docs/SETUP.md)
- [Canonical notebook template](templates/rag-technique-template.ipynb)

## Learning path

1. [Start here](00-start-here/README.md)
2. [RAG foundations](01-rag-foundations/README.md)

The foundations section contains the RAG mental model, a framework-free
implementation, a LangChain baseline, repository-owned learning data, golden
questions, and a reproducible baseline scorecard.

The current repository already contains practical notebooks for ingestion,
embeddings, chunking, retrieval, query enhancement, multimodal RAG, LangGraph,
agentic RAG, corrective/adaptive RAG, memory, and caching. The plan above
describes how those notebooks will be standardized, verified, and supplemented
with the missing production foundations: a baseline pipeline, evaluation,
citations, observability, security, data lifecycle, testing, and cost/latency
trade-offs.

## Current status

Planning, the course backbone, and the complete RAG foundations package are
ready. Reconstruction of the existing technique notebooks has not started yet;
use the tracker as the source of truth for progress.
