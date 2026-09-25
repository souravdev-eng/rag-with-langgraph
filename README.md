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
3. [Data ingestion and parsing](05-DataIngestParsing/README.md)
4. [Embeddings and vector geometry](06-Vector-embedding-and-vector-databases/README.md)
5. [Semantic chunking](08-advanced-chunking-and-preprocessing/semanti_chunking.ipynb)
6. [Retrieval and reranking](09-hybrid-search-strategies/README.md)
7. [Query transformation](10-query-enhancement/README.md)
8. [Multimodal PDF RAG](11-multiModal-multi-modal-rag/multimodalopenai.ipynb)
9. [LangGraph streaming](section-14-agents-architecture/streaming.ipynb)
10. [Bounded ReAct agents](section-14-agents-architecture/ReActAgents.ipynb)
11. [Agentic RAG correction](Section-15-agentic-rag/1-agenticrag.ipynb)
12. [Agentic RAG routing](Section-15-agentic-rag/1-agenticrag_1.ipynb)
13. [Retrieval-tool ReAct](Section-15-agentic-rag/2-ReAct.ipynb)
14. [Observable RAG planning](section-16-autonomous-rag/3-COTRag.ipynb)
15. [Evidence-bound self-reflection](section-16-autonomous-rag/4-Selfreflection.ipynb)
16. [Dependency-aware query planning](section-16-autonomous-rag/5-QueryPlanningdecomposition.ipynb)
17. [Bounded iterative retrieval](section-16-autonomous-rag/6-Iterativeretrieval.ipynb)
18. [Conflict-aware answer synthesis](section-16-autonomous-rag/7-answersynthesis.ipynb)
19. Multi-agent RAG: [topology guide](section-17-multi-agents-rags/8-multiagent.ipynb), [network](section-17-multi-agents-rags/1-network.ipynb), [supervisor](section-17-multi-agents-rags/2-supervisor.ipynb), and [hierarchical teams](section-17-multi-agents-rags/3-hierarchical-teams.ipynb)
20. [Corrective RAG](section-18-corrective-rag/2-CorrectiveRAG.ipynb)
21. [Adaptive RAG](section-19-adaptive-rag/adaptive-rag.ipynb)
22. [Conversation, checkpoint, and long-term memory](section-20-rag-with-persistant-memory/ragmemory.ipynb)
23. [Cache-augmented generation](section-21-cache-rag/cache_augment_generation.ipynb)

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

The course backbone, foundations, and every existing curriculum notebook are
complete, executable offline, and saved with verified outputs. The multi-agent
section is split into focused topology lessons while the original path remains
as a compatibility overview. Planned production-expansion modules remain a
separate future backlog; use the tracker as the source of truth.
